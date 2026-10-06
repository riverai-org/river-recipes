"""Execution-match grading, gold execution, per-question grading settings, and the asynchronous grading runtime.

Every test runs offline against a tiny SQLite database built in a temporary directory.
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text_to_sql_recipe import grading, sqlite_exec
from text_to_sql_recipe.sqlite_exec import (
    DIAMOND_NUMERIC_POLICY,
    SQLiteInfrastructureError,
    SQLiteQueryTimeout,
    SQLiteResult,
    SQLiteResultLimitExceeded,
    SqlWork,
)

NUMERIC_TIMEOUT = "numeric row matching exceeded its time budget"


def _make_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE animals (name TEXT, legs INTEGER)")
        connection.executemany(
            "INSERT INTO animals VALUES (?, ?)",
            [("ant", 6), ("cat", 4), ("dog", 4), ("emu", 2)],
        )
        connection.execute("CREATE TABLE t (v)")
        connection.commit()
    finally:
        connection.close()


class _DatabaseCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        database = Path(self._tmp.name) / "zoo.sqlite"
        _make_database(database)
        patcher = mock.patch.dict(
            os.environ, {sqlite_exec.SNAPSHOT_ROOT_ENV_VAR: str(Path(self._tmp.name) / "snapshots")}
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.task = {
            "id": "test:1",
            "domain": "zoo",
            "db_id": "zoo",
            "database_path": str(database),
            "gold_sql": "SELECT name FROM animals WHERE legs = 4 ORDER BY name",
            "grading_method": "set",
            "numeric_policy": DIAMOND_NUMERIC_POLICY,
            "numeric_tolerance": False,
        }

    def _grade(self, report: str) -> dict:
        return grading.grade(report, self.task)


class GradingTest(_DatabaseCase):
    def test_missing_database_after_gold_is_not_a_wrong_answer(self) -> None:
        expected = grading.execute_gold_result(self.task)
        Path(self.task["database_path"]).unlink()
        with self.assertRaises(SQLiteInfrastructureError):
            grading.grade_with_expected("<solution>SELECT 1</solution>", self.task, expected)

    def test_snapshot_mismatch_is_not_a_wrong_answer(self) -> None:
        expected = grading.execute_gold_result(self.task)
        self.task["database_sha256"] = "0" * 64
        with self.assertRaises(SQLiteInfrastructureError):
            grading.grade_with_expected("<solution>SELECT 1</solution>", self.task, expected)

    def test_a_pinned_database_grades_normally(self) -> None:
        import hashlib

        self.task["database_sha256"] = hashlib.sha256(Path(self.task["database_path"]).read_bytes()).hexdigest()
        graded = self._grade("<solution>SELECT name FROM animals WHERE legs = 4</solution>")
        self.assertEqual(graded["grade"], "CORRECT")
        self.assertEqual(graded["database_sha256"], self.task["database_sha256"])

    def test_receipt_reports_the_numeric_policy(self) -> None:
        self.task["gold_sql"] = "SELECT 1"
        outside = self._grade("<solution>SELECT 1.005</solution>")
        self.assertEqual(outside["grade"], "INCORRECT")
        self.assertEqual(outside["numeric_policy"], DIAMOND_NUMERIC_POLICY)
        self.assertFalse(outside["numeric_tolerance"])
        self.assertEqual(self._grade("<solution>SELECT 1.0000005</solution>")["grade"], "CORRECT")

    def test_authors_tolerance_credits_single_values_within_one_percent(self) -> None:
        self.task["gold_sql"] = "SELECT 100"
        for tolerance, answer, grade, applied in (
            (True, "SELECT 100.5", "CORRECT", True),
            (True, "SELECT 99.2", "CORRECT", True),
            (True, "SELECT 101", "INCORRECT", False),    # exactly 1% is outside
            (True, "SELECT 100", "CORRECT", False),      # exact: the tolerance is not needed
            (True, "SELECT 100.5, 1", "INCORRECT", False),  # not a single value
            (True, "SELECT 'about 100'", "INCORRECT", False),
            (False, "SELECT 100.5", "INCORRECT", False),  # strict grading
        ):
            with self.subTest(tolerance=tolerance, answer=answer):
                self.task["numeric_tolerance"] = tolerance
                graded = self._grade(f"<solution>{answer}</solution>")
                self.assertEqual(graded["grade"], grade)
                self.assertIs(graded["numeric_tolerance"], tolerance)
                self.assertIs(graded["numeric_tolerance_applied"], applied)

    def test_authors_tolerance_reads_results_as_the_authors_do(self) -> None:
        self.task["numeric_tolerance"] = True
        for gold, answer, grade in (
            ("SELECT 0", "SELECT 0.001", "INCORRECT"),            # no relative tolerance around zero
            ("SELECT 100 UNION ALL SELECT NULL", "SELECT 100.5", "CORRECT"),  # rows with no value are dropped
            ("SELECT 100 UNION ALL SELECT 200", "SELECT 100.5", "INCORRECT"),  # two rows
            ("SELECT '100'", "SELECT 100.5", "CORRECT"),          # a numeric string reads as its number
        ):
            with self.subTest(gold=gold, answer=answer):
                self.task["gold_sql"] = gold
                self.assertEqual(self._grade(f"<solution>{answer}</solution>")["grade"], grade)

    def test_single_value_needs_one_row_and_one_column(self) -> None:
        def result(*rows):
            return SQLiteResult(columns=("v",), rows=tuple(rows), elapsed_seconds=0.0)
        self.assertEqual(grading.single_value(result((2.5,))), 2.5)
        self.assertEqual(grading.single_value(result((None,), ("",), ("3",))), 3.0)
        self.assertIsNone(grading.single_value(result((1,), (2,))))
        self.assertIsNone(grading.single_value(result((1, 2))))
        self.assertIsNone(grading.single_value(result(("n/a",))))
        self.assertIsNone(grading.single_value(result()))

    def test_matching_result_is_correct(self) -> None:
        graded = self._grade("<solution>SELECT name FROM animals WHERE legs = 4</solution>")
        self.assertEqual(graded["grade"], "CORRECT")
        self.assertTrue(graded["format_compliant"])
        self.assertEqual(graded["error_category"], "")
        self.assertEqual((graded["normalized"], graded["pass_rate"]), (1.0, 1.0))
        self.assertEqual((graded["expected_rows"], graded["actual_rows"]), (2, 2))
        self.assertEqual(graded["criteria"][0]["section"], "zoo")
        self.assertEqual(graded["criteria"][0]["reason"], "results match with absolute numeric tolerance 1e-6")

    def test_the_graders_prediction_is_the_extracted_solution(self) -> None:
        for report in (
            "<solution>SELECT name FROM animals WHERE legs = 4</solution>",
            "<solution>SELECT nope FROM missing</solution>",
            "no solution block at all",
            "<solution>SELECT 1</solution><solution>SELECT 2</solution>",
        ):
            with self.subTest(report=report):
                extracted = grading.extract_solution(report)
                graded = self._grade(report)
                self.assertEqual(graded.get("prediction") or "(none)", extracted or "(none)")

    def test_wrong_result_is_incorrect(self) -> None:
        graded = self._grade("<solution>SELECT name FROM animals WHERE legs = 2</solution>")
        self.assertEqual(graded["grade"], "INCORRECT")
        self.assertEqual(graded["error_category"], "wrong_result")

    def test_an_execution_error_is_incorrect(self) -> None:
        graded = self._grade("<solution>SELECT nope FROM missing</solution>")
        self.assertEqual(graded["grade"], "INCORRECT")
        self.assertEqual(graded["error_category"], "execution")
        self.assertIn("no such table", graded["criteria"][0]["reason"])

    def test_missing_solution_is_format_failure(self) -> None:
        graded = self._grade("I think the answer involves cats and dogs.")
        self.assertEqual(graded["grade"], "NO_ANSWER")
        self.assertFalse(graded["format_compliant"])
        self.assertEqual(graded["error_category"], "format")
        self.assertIn("solution", graded["criteria"][0]["reason"])

    def test_write_statement_is_unsafe(self) -> None:
        graded = self._grade("<solution>DELETE FROM animals</solution>")
        self.assertEqual(graded["grade"], "INCORRECT")
        self.assertEqual(graded["error_category"], "unsafe_sql")

    def test_a_result_over_the_grading_limits_is_incorrect(self) -> None:
        self.task["grading_max_rows"] = 2
        self.task["gold_sql"] = "SELECT 1"
        graded = self._grade("<solution>SELECT name FROM animals</solution>")
        self.assertEqual(graded["error_category"], "result_limit")


class NumericPolicyGradingTest(_DatabaseCase):
    def test_task_policy_reaches_the_grader(self):
        self.task["gold_sql"] = "select 'a',24.0"
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.task["streaming_set_grading"] = streaming
                graded = self._grade("<solution>select 'a',24</solution>")
                self.assertEqual(graded["grade"], "CORRECT")
                self.assertEqual(graded["numeric_policy"], DIAMOND_NUMERIC_POLICY)

    def test_the_policy_defaults_when_a_task_omits_it(self):
        del self.task["numeric_policy"]
        self.task["gold_sql"] = "select 'a',24.0"
        self.assertEqual(self._grade("<solution>select 'a',24</solution>")["numeric_policy"], DIAMOND_NUMERIC_POLICY)

    def test_an_unsupported_policy_is_refused_rather_than_guessed(self):
        self.task["numeric_policy"] = "legacy"
        with self.assertRaisesRegex(ValueError, "unsupported numeric comparison policy"):
            self._grade("<solution>SELECT 1</solution>")

    def test_null_gold_rejects_an_empty_prediction(self):
        self.task["gold_sql"] = "SELECT SUM(v) FROM t"
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.task["streaming_set_grading"] = streaming
                self.assertEqual(self._grade("<solution>SELECT v FROM t</solution>")["grade"], "INCORRECT")
                self.assertEqual(self._grade("<solution>SELECT NULL</solution>")["grade"], "CORRECT")

    def test_streaming_uses_the_typed_index_budget(self):
        sql = "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<1000) SELECT x FROM n"
        task = {**self.task, "streaming_set_grading": True, "grading_max_rows": 2000, "grading_max_bytes": 1000000,
                "grading_max_index_bytes": 24576, "grading_temp_root": str(Path(self._tmp.name) / "index")}
        with self.assertRaises(SQLiteResultLimitExceeded):
            grading._execute_grading_result(task, sql)
        self.assertEqual(grading._execute_grading_result({**task, "grading_max_index_bytes": 10**8}, sql).row_count,
                         1000)


class ValidationGradingOverrideTest(unittest.TestCase):
    def test_a_gold_only_budget_never_relaxes_the_answer_budget(self) -> None:
        """A slow gold query (validation 122 has 180 seconds) must not relax the model-answer budget."""
        task: dict[str, object] = {"database_path": "/nowhere/test.sqlite", "timeout_seconds": 90,
                                   "gold_timeout_seconds": 180}
        self.assertEqual(grading._grading_executor(task).timeout_seconds, 90)

    def test_execute_gold_result_uses_only_the_gold_timeout_override(self) -> None:
        task = {"id": "validation:122", "gold_sql": "SELECT 1", "timeout_seconds": 90, "gold_timeout_seconds": 180}
        expected = SQLiteResult(columns=("value",), rows=((1,),), elapsed_seconds=0.0)
        with mock.patch.object(grading, "_execute_grading_result", return_value=expected) as run:
            self.assertIs(grading.execute_gold_result(task), expected)
        run.assert_called_once_with(task, "SELECT 1", timeout_seconds=180)

    def test_null_gold_timeout_falls_back_to_the_prediction_timeout(self) -> None:
        task = {"id": "validation:122", "gold_sql": "SELECT 1", "database_path": "/nowhere/test.sqlite",
                "timeout_seconds": 90, "gold_timeout_seconds": None}
        expected = SQLiteResult(columns=("value",), rows=((1,),), elapsed_seconds=0.0)
        with mock.patch.object(grading, "_execute_grading_result", return_value=expected) as run:
            self.assertIs(grading.execute_gold_result(task), expected)
        run.assert_called_once_with(task, "SELECT 1", timeout_seconds=None)
        self.assertEqual(grading._grading_executor(task).timeout_seconds, 90)

    def test_prediction_grading_keeps_the_prediction_timeout(self) -> None:
        task = {"id": "validation:122", "gold_sql": "SELECT 1", "timeout_seconds": 90, "gold_timeout_seconds": 180,
                "grading_method": "set"}
        expected = SQLiteResult(columns=("value",), rows=((1,),), elapsed_seconds=0.0)
        with (
            mock.patch.object(grading, "_execute_grading_result", return_value=expected) as run,
            mock.patch.object(grading, "_grade_payload", return_value={}),
        ):
            grading.grade_with_expected("<solution>SELECT 1</solution>", task, expected)
        run.assert_called_once_with(task, "SELECT 1")

    def test_prediction_timeout_is_reported_as_an_execution_timeout(self) -> None:
        task = {"id": "validation:122", "domain": "test", "gold_sql": "SELECT 1", "timeout_seconds": 90,
                "gold_timeout_seconds": 180, "grading_method": "set"}
        expected = SQLiteResult(columns=("value",), rows=((1,),), elapsed_seconds=0.0)
        with mock.patch.object(grading, "_execute_grading_result", side_effect=SQLiteQueryTimeout("query timed out")):
            graded = grading.grade_with_expected("<solution>SELECT 1</solution>", task, expected)
        self.assertEqual(graded["grade"], "INCORRECT")
        self.assertEqual(graded["error_category"], "timeout")

    def test_candidate_failures_during_comparison_are_graded_like_execution(self) -> None:
        # The numeric set replay executes the candidate again; its failures must not escape grading.
        task = {"id": "validation:7", "domain": "test", "gold_sql": "SELECT 1", "grading_method": "set"}
        expected = SQLiteResult(columns=("value",), rows=((1,),), elapsed_seconds=0.0)
        actual = SQLiteResult(columns=("value",), rows=((2,),), elapsed_seconds=0.0)
        for failure, category in ((SQLiteQueryTimeout("replay timed out"), "timeout"),
                                  (SQLiteResultLimitExceeded("replay too large"), "result_limit"),
                                  (sqlite_exec.SQLiteQueryError("replay failed"), "execution")):
            with self.subTest(category=category), \
                    mock.patch.object(grading, "_execute_grading_result", return_value=actual), \
                    mock.patch.object(grading, "compare_results", side_effect=failure):
                graded = grading.grade_with_expected("<solution>SELECT 2</solution>", task, expected)
            self.assertEqual((graded["grade"], graded["error_category"]), ("INCORRECT", category))
        # A gold replay failure is infrastructure, not a wrong answer.
        with (
            mock.patch.object(grading, "_execute_grading_result", return_value=actual),
            mock.patch.object(grading, "compare_results", side_effect=SQLiteInfrastructureError("gold replay")),
            self.assertRaises(SQLiteInfrastructureError),
        ):
            grading.grade_with_expected("<solution>SELECT 2</solution>", task, expected)

    def test_a_failing_gold_query_is_a_harness_failure(self) -> None:
        task = {"id": "validation:9", "gold_sql": "SELECT 1"}
        with (
            mock.patch.object(grading, "_execute_grading_result", side_effect=SQLiteQueryTimeout("slow")),
            self.assertRaisesRegex(RuntimeError, "gold SQL failed for ReViSQL task validation:9: slow"),
        ):
            grading.execute_gold_result(task)


class StreamingStorageTest(unittest.TestCase):
    def test_streaming_storage_is_private_and_sized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "grading"
            tasks = [{"id": "a", "streaming_set_grading": True}, {"id": "b"}]
            with mock.patch.object(grading, "STREAMING_GRADING_TEMP_ROOT", root):
                grading.prepare_streaming_grading_storage(tasks)
                self.assertEqual(tasks[0]["grading_temp_root"], str(root.resolve()))
                self.assertEqual(tasks[0]["grading_max_index_bytes"], 2_000_000_000)
                self.assertEqual(tasks[1], {"id": "b"})
                full = mock.Mock(return_value=mock.Mock(free=10))
                with (
                    mock.patch.object(grading.shutil, "disk_usage", full),
                    self.assertRaisesRegex(RuntimeError, "free bytes"),
                ):
                    grading.prepare_streaming_grading_storage([{"streaming_set_grading": True}])


class ExtractSolutionTest(unittest.TestCase):
    """The rules that decide what counts as an answer."""

    def test_the_final_solution_block_is_graded(self) -> None:
        """A model that reconsiders answers with its last block."""
        report = (
            "<solution>SELECT count(*) FROM animals</solution>\n"
            "Wait, the question asks for names.\n"
            "<solution>SELECT name FROM animals</solution>"
        )
        self.assertEqual(grading.extract_solution(report), "SELECT name FROM animals")

    def test_more_than_one_closing_tag_after_it_is_refused(self) -> None:
        self.assertIsNone(grading.extract_solution("<solution>SELECT 1</solution></solution>"))

    def test_an_unclosed_solution_block_is_refused(self) -> None:
        self.assertIsNone(grading.extract_solution("<solution>SELECT 1"))

    def test_a_leaked_tag_inside_the_block_is_refused(self) -> None:
        """Reasoning or tool markup inside the block is a formatting failure, not a wrong answer."""
        for leaked in (
            "<solution>SELECT 1 <think>or maybe 2</think></solution>",
            "<solution><sql>SELECT 1</sql></solution>",
            "<solution>SELECT 1 <observation>4 rows</observation></solution>",
            "<solution>SELECT 1 </think></solution>",
        ):
            with self.subTest(report=leaked):
                self.assertIsNone(grading.extract_solution(leaked))

    def test_an_ordinary_solution_survives_all_three_rules(self) -> None:
        self.assertEqual(
            grading.extract_solution("<solution>\n  SELECT thinking FROM notes\n</solution>"),
            "SELECT thinking FROM notes",
        )

    def test_an_empty_or_absent_block_is_refused(self) -> None:
        for report in ("<solution>   </solution>", "the answer is SELECT 1", "", None):
            with self.subTest(report=report):
                self.assertIsNone(grading.extract_solution(report))


class GradingMethodDefaultTest(_DatabaseCase):
    """Rows without a grading annotation are graded as sets, everywhere."""

    def test_grader_uses_set_semantics_when_the_task_omits_the_method(self) -> None:
        connection = sqlite3.connect(self.task["database_path"])
        connection.executemany("INSERT INTO t VALUES (?)", [(1,), (1,), (2,)])
        connection.commit()
        connection.close()
        task = {k: v for k, v in self.task.items() if k != "grading_method"}
        task["gold_sql"] = "SELECT v FROM t"
        expected = grading.execute_gold_result(task)
        graded = grading.grade_with_expected("<solution>SELECT DISTINCT v FROM t</solution>", task, expected)
        self.assertTrue(graded["normalized"])
        self.assertEqual(graded["grading_method"], "set")

    def test_streaming_set_grading_uses_the_missing_field_default(self) -> None:
        executor = mock.Mock()
        expected = object()
        executor.execute_set_digest.return_value = expected
        with mock.patch.object(grading, "_grading_executor", return_value=executor):
            actual = grading._execute_grading_result({"streaming_set_grading": True}, "SELECT 1")
        self.assertIs(actual, expected)
        executor.execute_set_digest.assert_called_once_with("SELECT 1", numeric_policy=DIAMOND_NUMERIC_POLICY)

    def test_streaming_set_grading_rejects_an_explicit_non_set_method(self) -> None:
        with (
            mock.patch.object(grading, "_grading_executor"),
            self.assertRaisesRegex(RuntimeError, "streaming set grading requires"),
        ):
            grading._execute_grading_result({"streaming_set_grading": True, "grading_method": "list"}, "SELECT 1")


class GoldResultsTest(unittest.TestCase):
    def test_gold_only_override_does_not_lower_the_gold_floor(self):
        task = {"id": "122", "timeout_seconds": 30, "gold_timeout_seconds": 60}
        marker = object()
        received: list[dict] = []

        def execute_gold_result(graded_task):
            received.append(graded_task)
            return marker

        with mock.patch.object(grading, "execute_gold_result", side_effect=execute_gold_result):
            results = grading.gold_results([task], concurrency=1, attempts=1)
        self.assertEqual(results, {"122": marker})
        self.assertEqual(received[0]["timeout_seconds"], 120)
        self.assertEqual(received[0]["gold_timeout_seconds"], 120)

    def test_a_longer_gold_budget_is_kept(self):
        task = {"id": "122", "timeout_seconds": 90, "gold_timeout_seconds": 180}
        with mock.patch.object(grading, "execute_gold_result", side_effect=lambda t: t) as run:
            results = grading.gold_results([task], concurrency=1)
        self.assertEqual((results["122"]["timeout_seconds"], results["122"]["gold_timeout_seconds"]), (120, 180))
        self.assertEqual(run.call_count, 1)

    def test_timeouts_are_retried_and_failures_left_out_with_their_cause(self):
        tasks = [{"id": "slow"}, {"id": "always-slow"}, {"id": "corrupt"}]
        calls = dict.fromkeys(("slow", "always-slow", "corrupt"), 0)

        def execute_gold_result(task):
            calls[task["id"]] += 1
            if task["id"] == "corrupt":
                cause = SQLiteInfrastructureError("database snapshot SHA-256 differs from the frozen task contract")
            elif task["id"] == "always-slow" or calls["slow"] < 3:
                cause = SQLiteQueryTimeout("query timed out")
            else:
                return "ok"
            raise RuntimeError(f"gold SQL failed for ReViSQL task {task['id']}: {cause}") from cause

        failures = {}
        with (
            mock.patch.object(grading, "execute_gold_result", side_effect=execute_gold_result),
            mock.patch.object(grading.time, "sleep"),
        ):
            results = grading.gold_results(tasks, concurrency=2, failures=failures)
        self.assertEqual(results, {"slow": "ok"})
        # Only a timeout is tried again; a corrupt database fails at once.
        self.assertEqual(calls, {"slow": 3, "always-slow": 3, "corrupt": 1})
        self.assertEqual(failures, {
            "always-slow": "SQLiteQueryTimeout: query timed out",
            "corrupt": "SQLiteInfrastructureError: database snapshot SHA-256 differs from the frozen task contract",
        })

    def test_real_gold_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zoo.sqlite"
            _make_database(path)
            base = {"domain": "zoo", "database_path": str(path)}
            tasks = [{**base, "id": "a", "gold_sql": "SELECT COUNT(*) FROM animals"},
                     {**base, "id": "b", "gold_sql": "SELECT nope FROM nowhere"}]
            with mock.patch.object(grading.time, "sleep"):
                results = grading.gold_results(tasks, concurrency=2)
        self.assertEqual(list(results), ["a"])
        self.assertEqual(results["a"].rows, ((4,),))


@dataclass
class _Rollout:
    """The fields grading reads and writes on a rollout record."""

    task: dict
    final_text: str = ""
    unsafe: bool = False
    grade: dict = field(default_factory=dict)
    reward: float = 0.0
    turns: list = field(default_factory=list)


class GradeRolloutTest(_DatabaseCase):
    def test_an_unsafe_rollout_is_not_executed(self):
        rollout = _Rollout(self.task, "<solution>SELECT name FROM animals WHERE legs = 4</solution>", unsafe=True)
        with mock.patch.object(grading, "grade_with_expected") as executed:
            grading.grade_rollout(rollout, object(), {})
        executed.assert_not_called()
        self.assertEqual(rollout.grade, {"grade": "INCORRECT", "error_category": "unsafe_sql",
                                         "format_compliant": False})

    def test_a_committed_answer_is_graded_and_penalties_applied_after(self):
        expected = grading.execute_gold_result(self.task)
        rollout = _Rollout(self.task, "<solution>SELECT name FROM animals WHERE legs = 4</solution>")
        seen = []
        grading.grade_rollout(rollout, expected, {"weight": 1}, lambda r, cfg: seen.append((r.grade["grade"], cfg)))
        self.assertEqual(rollout.grade["grade"], "CORRECT")
        self.assertEqual(seen, [("CORRECT", {"weight": 1})])

    def test_the_dedup_key(self):
        rollout = _Rollout({"id": 7}, "answer", unsafe=True)
        self.assertEqual(grading.grading_dedup_key(rollout), ("7", "answer", True))

    def test_receipts_drop_the_result_sets_and_are_copies(self):
        grade = {"grade": "CORRECT", "expected_output": {"rows": 1}, "actual_output": {}, "criteria": [{"a": 1}]}
        receipt = grading.grading_receipt(grade)
        self.assertEqual(receipt, {"grade": "CORRECT", "criteria": [{"a": 1}]})
        receipt["criteria"][0]["a"] = 2
        self.assertEqual(grade["criteria"][0]["a"], 1)


class NumericTimeoutClassificationTest(unittest.TestCase):
    def test_only_the_numeric_matching_timeout_is_retryable(self):
        self.assertTrue(grading.numeric_comparison_timed_out(SQLiteInfrastructureError(NUMERIC_TIMEOUT)))
        for error in (
            SQLiteInfrastructureError("database snapshot not found: x"),
            SQLiteInfrastructureError("numeric row matching exceeded its disk budget"),
            RuntimeError(NUMERIC_TIMEOUT),
            SQLiteQueryTimeout(NUMERIC_TIMEOUT),
        ):
            with self.subTest(error=repr(error)):
                self.assertFalse(grading.numeric_comparison_timed_out(error))

    def test_a_real_matching_timeout_is_classified(self):
        rows = [sqlite_exec._normalize_row((float(i),)) for i in range(20)]
        near = [sqlite_exec._normalize_row((float(i) + 5e-7,)) for i in range(20)]
        with self.assertRaises(SQLiteInfrastructureError) as caught:
            sqlite_exec._compare_numeric_rows(rows, near, method="set", timeout_seconds=0)
        self.assertTrue(grading.numeric_comparison_timed_out(caught.exception))


class CompletedTaskCacheTest(unittest.IsolatedAsyncioTestCase):
    async def test_only_completed_entries_beyond_the_limit_are_dropped(self):
        cache = grading.CompletedTaskCache(limit=1)
        gate = asyncio.Event()
        pending = asyncio.create_task(gate.wait())
        cache["pending"] = pending
        for name in ("a", "b", "c"):
            done = asyncio.create_task(asyncio.sleep(0))
            await done
            cache[name] = done
        await asyncio.sleep(0)
        self.assertEqual(list(cache), ["pending", "c"])
        gate.set()
        await pending
        await asyncio.sleep(0)
        self.assertEqual(list(cache), ["c"])


class GradingRuntimeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        path = Path(self._tmp.name) / "zoo.sqlite"
        _make_database(path)
        self.task = {"id": "test:1", "domain": "zoo", "database_path": str(path),
                     "gold_sql": "SELECT name FROM animals WHERE legs = 4", "grading_method": "set",
                     "numeric_policy": DIAMOND_NUMERIC_POLICY}
        self.expected = grading.gold_results([self.task], concurrency=1)
        self.answer = "<solution>SELECT name FROM animals WHERE legs = 4</solution>"

    def _runtime(self, **kwargs) -> grading.GradingRuntime:
        options = {"grading_concurrency": 4, "sql_concurrency": 4, **kwargs}
        return grading.GradingRuntime({"lambda": 0.1}, self.expected, **options)

    async def _grade_quietly(self, runtime, rollout):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            await runtime.grade(rollout)
        return output.getvalue()

    async def test_identical_answers_share_one_execution_and_get_their_own_grade(self):
        runtime = self._runtime()
        first, second = _Rollout(self.task, self.answer), _Rollout(self.task, self.answer)
        other = _Rollout(self.task, "<solution>SELECT name FROM animals</solution>")
        with mock.patch.object(grading, "grade_with_expected", wraps=grading.grade_with_expected) as executed:
            await asyncio.gather(runtime.grade(first), runtime.grade(second), runtime.grade(other))
        self.assertEqual(executed.call_count, 2)
        self.assertEqual((first.grade["grade"], second.grade["grade"], other.grade["grade"]),
                         ("CORRECT", "CORRECT", "INCORRECT"))
        self.assertEqual(len(first.grade["grading_attempts"]), 1)
        self.assertNotIn("expected_output", first.grade["grading_attempts"][0])
        self.assertEqual(first.grade["grading_attempts"][0]["timeout_seconds"], 30)
        first.grade["grade"] = "mutated"
        self.assertEqual(second.grade["grade"], "CORRECT")
        await runtime.close()

    async def test_penalties_run_on_each_attempt_copy_not_on_the_rollout(self):
        def apply_penalties(rollout, cfg):
            rollout.grade["process_penalty"] = cfg["lambda"]
            rollout.reward = 123.0

        runtime = self._runtime(apply_penalties=apply_penalties)
        rollout = _Rollout(self.task, self.answer)
        await runtime.grade(rollout)
        self.assertEqual(rollout.grade["process_penalty"], 0.1)
        self.assertEqual(rollout.grade["grading_attempts"][0]["process_penalty"], 0.1)
        self.assertEqual(rollout.reward, 0.0)
        await runtime.close()

    async def test_a_query_timeout_is_retried_with_every_slot_held(self):
        runtime = self._runtime()
        timeout = {"grade": "INCORRECT", "error_category": "timeout", "format_compliant": True}
        correct = {"grade": "CORRECT", "error_category": "", "format_compliant": True}
        rollout = _Rollout(self.task, self.answer)
        with (
            mock.patch.object(grading, "grade_with_expected", side_effect=[timeout, correct]),
            mock.patch.object(runtime.sql, "execute", wraps=runtime.sql.execute) as scheduled,
        ):
            await runtime.grade(rollout)
        self.assertEqual([call.kwargs["exclusive"] for call in scheduled.call_args_list], [False, True])
        self.assertEqual(rollout.grade["grade"], "CORRECT")
        self.assertEqual([a["error_category"] for a in rollout.grade["grading_attempts"]], ["timeout", ""])
        await runtime.close()

    async def test_a_persistent_query_timeout_is_graded_and_not_reused(self):
        runtime = self._runtime(grading_timeout_retries=2)
        timeout = {"grade": "INCORRECT", "error_category": "timeout", "format_compliant": True}
        with mock.patch.object(grading, "grade_with_expected", return_value=timeout) as executed:
            first = _Rollout(self.task, self.answer)
            await runtime.grade(first)
            await asyncio.sleep(0)
            second = _Rollout(self.task, self.answer)
            await runtime.grade(second)
        self.assertEqual(first.grade["error_category"], "timeout")
        self.assertEqual(len(first.grade["grading_attempts"]), 3)
        self.assertEqual(executed.call_count, 6)
        await runtime.close()

    async def test_an_execution_error_grade_is_not_reused(self):
        runtime = self._runtime()
        failed = {"grade": "INCORRECT", "error_category": "execution", "format_compliant": True}
        with mock.patch.object(grading, "grade_with_expected", return_value=failed) as executed:
            await runtime.grade(_Rollout(self.task, self.answer))
            await asyncio.sleep(0)
            await runtime.grade(_Rollout(self.task, self.answer))
        self.assertEqual(executed.call_count, 2)
        await runtime.close()

    async def test_a_numeric_matching_timeout_is_retried_then_graded(self):
        runtime = self._runtime(grading_timeout_retries=2)
        correct = {"grade": "CORRECT", "error_category": "", "format_compliant": True}
        rollout = _Rollout(self.task, self.answer)
        with mock.patch.object(grading, "grade_with_expected",
                               side_effect=[SQLiteInfrastructureError(NUMERIC_TIMEOUT), correct]):
            log = await self._grade_quietly(runtime, rollout)
        self.assertEqual(rollout.grade["grade"], "CORRECT")
        attempts = rollout.grade["grading_attempts"]
        self.assertEqual([a["error_category"] for a in attempts], ["infrastructure_timeout", ""])
        self.assertEqual(attempts[0]["error"], NUMERIC_TIMEOUT)
        self.assertEqual(json.loads(log), {"grading_infrastructure_retry": {"task_id": "test:1", "attempt": 1,
                                                                            "exhausted": False}})
        await runtime.close()

    async def test_a_persistent_numeric_matching_timeout_is_raised(self):
        runtime = self._runtime(grading_timeout_retries=1)
        rollout = _Rollout(self.task, self.answer)
        with (
            mock.patch.object(grading, "grade_with_expected",
                              side_effect=SQLiteInfrastructureError(NUMERIC_TIMEOUT)) as executed,
            self.assertRaisesRegex(SQLiteInfrastructureError, NUMERIC_TIMEOUT),
        ):
            await self._grade_quietly(runtime, rollout)
        self.assertEqual(executed.call_count, 2)
        self.assertEqual(rollout.grade, {})
        await runtime.close()

    async def test_other_infrastructure_failures_are_not_retried(self):
        runtime = self._runtime()
        with (
            mock.patch.object(grading, "grade_with_expected",
                              side_effect=SQLiteInfrastructureError("database snapshot not found: x")) as executed,
            self.assertRaisesRegex(SQLiteInfrastructureError, "not found"),
        ):
            await runtime.grade(_Rollout(self.task, self.answer))
        self.assertEqual(executed.call_count, 1)
        self.assertEqual(len(runtime.grades), 0)
        await runtime.close()

    async def test_a_real_unsafe_answer_and_a_shared_scheduler(self):
        work = SqlWork(4)
        runtime = self._runtime(sql_work=work)
        rollout = _Rollout(self.task, "<solution>DELETE FROM animals</solution>")
        await runtime.grade(rollout)
        self.assertEqual(rollout.grade["error_category"], "unsafe_sql")
        with self.assertRaisesRegex(ValueError, "disagrees with configured concurrency"):
            self._runtime(sql_work=SqlWork(2))
        await runtime.close()

    async def test_close_cancels_pending_grades(self):
        runtime = self._runtime()
        release = asyncio.Event()

        async def never():
            await release.wait()

        runtime.grades["stuck"] = asyncio.create_task(never())
        await runtime.close()
        self.assertTrue(all(task.done() for task in runtime.grades.values()))

    async def test_rollouts_are_graded_on_a_copy(self):
        runtime = self._runtime()
        rollout = _Rollout(self.task, self.answer, turns=[{"role": "assistant"}])
        snapshot = copy.deepcopy(rollout.turns)
        await runtime.grade(rollout)
        self.assertEqual(rollout.turns, snapshot)
        await runtime.close()


if __name__ == "__main__":
    unittest.main()
