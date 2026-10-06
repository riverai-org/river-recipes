"""The SQLite sandbox, result comparison, and the SQL tool's execution.

Comparison tests name one pair of results and the verdict the grading method requires for it, rather than checking
that two code paths agree: agreement survives a pair of identical mistakes. The distinguishing pairs are chosen so
that the method under test grades them differently from ``set``.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import os
import sqlite3
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text_to_sql_recipe import sqlite_exec
from text_to_sql_recipe.grading import numeric_comparison_timed_out
from text_to_sql_recipe.sqlite_exec import (
    DIAMOND_NUMERIC_POLICY,
    ReadOnlySQLite,
    SQLiteInfrastructureError,
    SQLiteQueryError,
    SQLiteQueryTimeout,
    SQLiteResult,
    SQLiteResultLimitExceeded,
    SQLiteSetDigest,
    SQLiteToolResult,
    SqlWork,
    UnsafeSQLiteQuery,
    compare_results,
    prepare_private_temp_root,
    sqlite_result_payload,
)

MATCH = "results match with absolute numeric tolerance 1e-6"
MISMATCH = "result values or row cardinalities differ"


def _result(rows, columns=("value",)) -> SQLiteResult:
    return SQLiteResult(columns=columns, rows=tuple(rows), elapsed_seconds=0.0)


def _digest(*, columns: int, unique: int, sha: str, preview=()) -> SQLiteSetDigest:
    return SQLiteSetDigest(
        column_count=columns,
        row_count=unique,
        unique_row_count=unique,
        result_sha256=sha,
        rows_preview=tuple(preview),
        elapsed_seconds=0.0,
    )


def _database(directory: str, name: str = "zoo.sqlite") -> Path:
    path = Path(directory) / name
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
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# One gold row, one prediction that returns it twice. Graded as a set the
# duplicate is invisible; graded as a multiset it is the whole answer.
_GOLD_ONCE = _result([("cat", "4")], ("name", "legs"))
_PREDICTED_TWICE = _result([("cat", "4"), ("cat", "4")], ("name", "legs"))

# The same two rows in opposite order: invisible to ``set``, decisive for ORDER BY.
_GOLD_ORDERED = _result([("a",), ("b",)])
_PREDICTED_REVERSED = _result([("b",), ("a",)])


class MultiplicityGradingTest(unittest.TestCase):
    """``multiset`` must still count rows."""

    def test_a_duplicated_row_is_correct_as_a_set(self) -> None:
        verdict = compare_results(_GOLD_ONCE, _PREDICTED_TWICE, method="set")
        self.assertTrue(verdict.correct)
        self.assertEqual(verdict.message, MATCH)

    def test_a_duplicated_row_is_incorrect_as_a_multiset(self) -> None:
        verdict = compare_results(_GOLD_ONCE, _PREDICTED_TWICE, method="multiset")
        self.assertFalse(verdict.correct)
        self.assertEqual(verdict.method, "multiset")
        self.assertEqual(verdict.message, MISMATCH)

    def test_the_method_is_normalized_before_use(self) -> None:
        """A released row spells one method `` multiset``."""
        self.assertEqual(compare_results(_GOLD_ONCE, _PREDICTED_TWICE, method=" Multiset").method, "multiset")
        self.assertFalse(compare_results(_GOLD_ONCE, _PREDICTED_TWICE, method=" multiset").correct)
        self.assertTrue(compare_results(_GOLD_ONCE, _PREDICTED_TWICE, method="").correct)


class RowOrderGradingTest(unittest.TestCase):
    """``list``/``ordered`` must still be ordered."""

    def test_reordered_rows_are_correct_as_a_set(self) -> None:
        self.assertTrue(compare_results(_GOLD_ORDERED, _PREDICTED_REVERSED, method="set").correct)

    def test_reordered_rows_are_incorrect_as_an_ordered_result(self) -> None:
        for method in ("list", "ordered"):
            with self.subTest(method=method):
                verdict = compare_results(_GOLD_ORDERED, _PREDICTED_REVERSED, method=method)
                self.assertFalse(verdict.correct)
                self.assertEqual(verdict.message, MISMATCH)

    def test_reordered_rows_are_correct_as_a_multiset(self) -> None:
        """Pins where the ``multiset``/``list`` boundary actually falls."""
        self.assertTrue(compare_results(_GOLD_ORDERED, _PREDICTED_REVERSED, method="multiset").correct)


class SubsetGradingTest(unittest.TestCase):
    """Direction matters: the *prediction* must be contained in the gold set, never the other way round."""

    GOLD = _result([("a",), ("b",), ("c",)])

    def _verdict(self, rows, method):
        return compare_results(self.GOLD, _result(rows), method=method)

    def test_a_proper_subset_of_the_gold_rows_passes(self) -> None:
        verdict = self._verdict([("a",), ("b",)], "subset,>=,2")
        self.assertTrue(verdict.correct)
        self.assertEqual(verdict.method, "subset,>=,2")

    def test_a_superset_of_the_gold_rows_fails(self) -> None:
        verdict = compare_results(_result([("a",), ("b",)]), _result([("a",), ("b",), ("c",)]), method="subset,>=,2")
        self.assertFalse(verdict.correct)

    def test_a_row_outside_the_gold_set_fails(self) -> None:
        self.assertFalse(self._verdict([("a",), ("z",)], "subset,>=,2").correct)

    def test_equal_sets_pass(self) -> None:
        self.assertTrue(self._verdict([("a",), ("b",), ("c",)], "subset,>=,3").correct)

    def test_the_exact_operator_rejects_more_or_fewer_unique_rows(self) -> None:
        """``=`` relaxed to ``>=`` would accept 'top 2' answered with 3."""
        self.assertFalse(self._verdict([("a",), ("b",), ("c",)], "subset,=,2").correct)
        self.assertFalse(self._verdict([("a",)], "subset,=,2").correct)

    def test_the_at_least_operator(self) -> None:
        self.assertFalse(self._verdict([("a",)], "subset,>=,2").correct)
        self.assertTrue(self._verdict([("a",), ("b",), ("c",)], "subset,>=,2").correct)

    def test_the_count_is_over_unique_predicted_rows(self) -> None:
        self.assertTrue(self._verdict([("a",), ("a",), ("b",)], "subset,=,2").correct)

    def test_malformed_subset_methods_are_rejected(self) -> None:
        for method in ("subset,>,1", "subset,>=", "subset,>=,many"):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "invalid subset grading method"):
                self._verdict([("a",)], method)


class UnknownMethodTest(unittest.TestCase):
    def test_an_unknown_method_is_rejected(self) -> None:
        for method in ("jaccard", "bird_set"):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "unsupported SQL grading method"):
                compare_results(_GOLD_ONCE, _GOLD_ONCE, method=method)


class ShapeMismatchTest(unittest.TestCase):
    def test_differing_column_counts_are_incorrect(self) -> None:
        verdict = compare_results(_result([("a", "b")], ("x", "y")), _result([("a",)]), method="set")
        self.assertFalse(verdict.correct)
        self.assertEqual(verdict.message, "column counts differ")

    def test_one_empty_result_is_incorrect_in_either_direction(self) -> None:
        rows = _result([("a",)])
        empty = _result([])
        for expected, actual in ((rows, empty), (empty, rows)):
            with self.subTest(expected=expected.rows, actual=actual.rows):
                verdict = compare_results(expected, actual, method="set")
                self.assertFalse(verdict.correct)
                self.assertEqual(verdict.message, "one result is empty")

    def test_two_empty_results_are_correct(self) -> None:
        verdict = compare_results(_result([]), _result([]), method="set")
        self.assertTrue(verdict.correct)
        self.assertEqual(verdict.message, "both results are empty")


class SetDigestGradingTest(unittest.TestCase):
    """The streamed-digest branch repeats the shape rules on a different code path."""

    def test_differing_column_counts_are_incorrect(self) -> None:
        verdict = compare_results(_digest(columns=2, unique=5, sha="a"), _digest(columns=1, unique=5, sha="a"))
        self.assertFalse(verdict.correct)
        self.assertEqual(verdict.message, "column counts differ")

    def test_equal_digests_are_correct(self) -> None:
        left = _digest(columns=2, unique=5, sha="abc")
        verdict = compare_results(left, left, method="set")
        self.assertTrue(verdict.correct)
        self.assertEqual(verdict.message, "results match as streamed sets")

    def test_digests_of_different_cardinality_differ_without_a_replay(self) -> None:
        verdict = compare_results(_digest(columns=1, unique=5, sha="a"), _digest(columns=1, unique=4, sha="b"))
        self.assertFalse(verdict.correct)
        self.assertEqual(verdict.message, "streamed result sets differ")

    def test_equal_cardinality_with_different_hashes_needs_a_replay(self) -> None:
        with self.assertRaisesRegex(SQLiteInfrastructureError, "requires replayable set results"):
            compare_results(_digest(columns=1, unique=5, sha="a"), _digest(columns=1, unique=5, sha="b"))

    def test_two_empty_digests_are_correct(self) -> None:
        empty = _digest(columns=1, unique=0, sha="e3b0")
        verdict = compare_results(empty, empty, method="set")
        self.assertTrue(verdict.correct)
        self.assertEqual(verdict.message, "both results are empty")

    def test_a_digest_may_only_be_graded_as_a_set(self) -> None:
        digest = _digest(columns=1, unique=1, sha="a")
        with self.assertRaisesRegex(ValueError, "set digests may only use set grading"):
            compare_results(digest, digest, method="multiset")

    def test_a_digest_may_not_be_compared_against_a_row_result(self) -> None:
        with self.assertRaisesRegex(TypeError, "require two set digests"):
            compare_results(_digest(columns=1, unique=1, sha="a"), _result([("a",)]))

    def test_a_digest_must_carry_the_comparison_policy(self) -> None:
        digest = _digest(columns=1, unique=1, sha="a")
        other = dataclasses.replace(digest, numeric_policy="another-policy")
        with self.assertRaisesRegex(ValueError, "numeric policy does not match"):
            compare_results(digest, other)


POLICY = DIAMOND_NUMERIC_POLICY


def _numeric(rows):
    return SQLiteResult(tuple("c" + str(i) for i in range(len(rows[0]) if rows else 1)), tuple(map(tuple, rows)), 0)


def _equal(a, b, method="set", **kwargs):
    return compare_results(_numeric(a), _numeric(b), method=method, numeric_policy=POLICY, **kwargs).correct


class NumericPolicyTest(unittest.TestCase):
    """``diamond-numeric-v1``: absolute 1e-6 in every numeric cell, typed cells, exact cardinality."""

    def test_representation_in_every_cell(self):
        self.assertTrue(_equal([["a", 24.0], ["b", 31.0]], [["b", 31], ["a", 24]]))
        self.assertTrue(compare_results(_numeric([["a", 24.0]]), _numeric([["a", 24]])).correct)

    def test_exact_counts_and_large_identifiers(self):
        for a, b in [(1000, 1001), (2**53 + 1, float(2**53)), (0, 0.0000011)]:
            with self.subTest(a=a, b=b):
                self.assertFalse(_equal([[a]], [[b]]))
        self.assertTrue(_equal([[2**60]], [[float(2**60)]]))

    def test_the_epsilon_bound_is_inclusive(self):
        """The float nearest 1e-6 lies just below 1e-6, so it is within the bound."""
        self.assertTrue(_equal([[0]], [[0.000001]]))
        self.assertTrue(_equal([[0.3]], [[0.1 + 0.2]]))

    def test_no_numeric_text_coercion(self):
        for a, b in [("24", 24), ("001", 1), (None, 0), ("None", None), (b"24", "24")]:
            with self.subTest(a=a, b=b):
                self.assertFalse(_equal([["a", a]], [["a", b]]))

    def test_signed_zero_and_decimal_identity(self):
        self.assertTrue(_equal([[0]], [[-0.0]]))
        self.assertTrue(_equal([[24.125]], [[24.125]]))
        self.assertFalse(_equal([[24.125]], [[24.12]]))

    def test_no_one_percent_single_value_tolerance(self):
        self.assertFalse(compare_results(_numeric([[1000]]), _numeric([[1001]])).correct)
        self.assertFalse(compare_results(_numeric([[100.0]]), _numeric([[100.5]])).correct)

    def test_set_multiset_list_contracts(self):
        self.assertTrue(_equal([[24]], [[24.0], [24.0]]))
        self.assertFalse(_equal([[24]], [[24.0], [24.0]], "multiset"))
        self.assertFalse(_equal([[24], [31]], [[31.0], [24.0]], "list"))
        self.assertTrue(_equal([[24], [31]], [[24.0], [31.0]], "ordered"))

    def test_subset_cardinality_is_numeric_identity(self):
        self.assertTrue(_equal([[24], [31]], [[24.0]], "subset,=,1"))
        self.assertFalse(_equal([[24], [31]], [[24], [24.0]], "subset,=,2"))
        self.assertFalse(_equal([[24], [31]], [[32]], "subset,>=,1"))

    def test_column_order_within_a_row_is_not_compared(self):
        self.assertTrue(_equal([["a", 24]], [[24.0, "a"]]))

    def test_unknown_policy_rejected(self):
        for policy in ("typo", "legacy"):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "unsupported numeric comparison"):
                compare_results(_numeric([[1]]), _numeric([[1]]), numeric_policy=policy)

    def test_nan_is_not_a_supported_result(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            compare_results(_numeric([[float("nan")]]), _numeric([[1.0]]))


class NullResultTest(unittest.TestCase):
    """Rows containing missing values are not absent rows."""

    def test_real_sql_rows_survive_every_comparison_path(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = ReadOnlySQLite(_database(directory))
            empty = executor.execute("SELECT v FROM t")
            for sql in ("SELECT NULL", "SELECT ''", "SELECT '  '"):
                with self.subTest(sql=sql):
                    row = executor.execute(sql)
                    for method in ("set", "multiset", "list", "ordered", "subset,=,1"):
                        self.assertFalse(compare_results(row, empty, method=method).correct)
                        self.assertFalse(compare_results(empty, row, method=method).correct)
                        self.assertTrue(compare_results(row, row, method=method).correct)
                    a = executor.execute_set_digest(sql)
                    b = executor.execute_set_digest("SELECT v FROM t")
                    self.assertEqual(a.unique_row_count, 1)
                    self.assertFalse(compare_results(a, b).correct)
                    self.assertTrue(compare_results(a, a).correct)

    def test_null_multiplicity_and_value_types_remain_distinct(self):
        null = _result([(None,)])
        for rows in ([("",)], [("None",)], [(0,)]):
            self.assertFalse(compare_results(null, _result(rows)).correct)
        twice = _result([(None,), (None,)])
        self.assertTrue(compare_results(null, twice).correct)
        self.assertFalse(compare_results(null, twice, method="multiset").correct)


class StreamedDigestTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = _database(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_streaming_matches_materialized_and_pins_policy(self):
        executor = ReadOnlySQLite(self.path)
        a = executor.execute_set_digest("select 'a', 24 union all select 'a',24.0", numeric_policy=POLICY)
        b = executor.execute_set_digest("select 24.0, 'a'", numeric_policy=POLICY)
        self.assertEqual((a.row_count, a.unique_row_count), (2, 1))
        self.assertEqual(a.numeric_policy, POLICY)
        self.assertTrue(compare_results(a, b, numeric_policy=POLICY).correct)
        with self.assertRaisesRegex(ValueError, "unsupported numeric comparison"):
            executor.execute_set_digest("select 1", numeric_policy="legacy")

    def test_a_digest_mismatch_is_replayed_under_the_epsilon_rule(self):
        executor = ReadOnlySQLite(self.path)
        gold = executor.execute_set_digest("SELECT 0.3 UNION ALL SELECT 2")
        close = executor.execute_set_digest("SELECT 0.1 + 0.2 UNION ALL SELECT 2")
        far = executor.execute_set_digest("SELECT 0.3002 UNION ALL SELECT 2")
        self.assertNotEqual(gold.result_sha256, close.result_sha256)
        self.assertTrue(compare_results(gold, close).correct)
        self.assertFalse(compare_results(gold, far).correct)

    def test_a_result_that_changes_before_its_replay_is_an_infrastructure_failure(self):
        executor = ReadOnlySQLite(self.path)
        gold = executor.execute_set_digest("SELECT legs * 1.0 FROM animals")
        predicted = executor.execute_set_digest("SELECT legs + 0.0000001 FROM animals")
        connection = sqlite3.connect(self.path)
        connection.execute("UPDATE animals SET legs = legs + 100")
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(SQLiteInfrastructureError, "gold replay failed|changed during epsilon"):
            compare_results(gold, predicted)

    def test_an_answer_whose_rows_change_when_run_again_is_a_wrong_answer(self):
        # random() moves every row by an offset fixed within a run but different in the next run: the rows keep the
        # gold's cardinality and fall within the epsilon, so they are replayed, and the replay cannot be trusted.
        executor = ReadOnlySQLite(self.path)
        gold = executor.execute_set_digest("SELECT legs * 1.0 FROM animals")
        predicted = executor.execute_set_digest(
            "SELECT legs * 1.0 + 1e-7 + (SELECT (abs(random()) % 1000000) * 1e-13) FROM animals")
        with self.assertRaisesRegex(SQLiteQueryError, "different rows when it was run again") as raised:
            compare_results(gold, predicted)
        self.assertNotIsInstance(raised.exception, SQLiteInfrastructureError)

    def test_a_gold_replay_that_runs_out_of_time_is_retried_like_a_slow_comparison(self):
        executor = ReadOnlySQLite(self.path)
        gold = executor.execute_set_digest("SELECT legs * 1.0 FROM animals")
        predicted = executor.execute_set_digest("SELECT legs + 0.0000001 FROM animals")
        slow = mock.Mock()
        slow.execute_set_digest.side_effect = SQLiteQueryTimeout("query timed out")
        object.__setattr__(gold, "_replay", (slow, gold._replay[1]))   # the digest is frozen
        with self.assertRaises(SQLiteInfrastructureError) as raised:
            compare_results(gold, predicted)
        self.assertTrue(numeric_comparison_timed_out(raised.exception))

    def test_row_limit_is_enforced_while_streaming(self):
        executor = ReadOnlySQLite(self.path, max_rows=3, truncate=False)
        with self.assertRaisesRegex(SQLiteResultLimitExceeded, "more than 3 rows"):
            executor.execute_set_digest("SELECT name FROM animals")

    def test_the_index_has_a_disk_budget(self):
        sql = "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<5000) SELECT x FROM n"
        with tempfile.TemporaryDirectory() as root:
            executor = ReadOnlySQLite(
                self.path, max_rows=10_000, truncate=False,
                set_index_root=Path(root) / "index", set_index_max_bytes=8192,
            )
            with self.assertRaisesRegex(SQLiteResultLimitExceeded, "streamed set index exceeded"):
                executor.execute_set_digest(sql)
            self.assertEqual(stat.S_IMODE((Path(root) / "index").stat().st_mode), 0o700)


class ReadOnlySandboxTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = _database(self._tmp.name)
        self.executor = ReadOnlySQLite(self.path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_select_and_recursive_queries_run(self):
        result = self.executor.execute("SELECT name FROM animals WHERE legs = 4 ORDER BY name")
        self.assertEqual(result.columns, ("name",))
        self.assertEqual(result.rows, (("cat",), ("dog",)))
        self.assertFalse(result.truncated)
        counted = self.executor.execute("WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<3) "
                                        "SELECT COUNT(*) FROM n")
        self.assertEqual(counted.rows, ((3,),))

    def test_anything_but_a_read_is_unsafe_and_leaves_the_file_untouched(self):
        before = _sha256(self.path)
        for sql in (
            "DELETE FROM animals",
            "INSERT INTO animals VALUES ('yak', 4)",
            "UPDATE animals SET legs = 0",
            "DROP TABLE animals",
            "CREATE TABLE x (y)",
            "ATTACH DATABASE ':memory:' AS other",
            "PRAGMA user_version = 7",
            "SELECT load_extension('anything')",
        ):
            with self.subTest(sql=sql), self.assertRaises(UnsafeSQLiteQuery):
                self.executor.execute(sql)
        self.assertEqual(_sha256(self.path), before)

    def test_query_errors_are_not_infrastructure_errors(self):
        for sql, message in (
            ("", "must not be empty"),
            ("SELEC 1", "SQLite execution failed"),
            ("SELECT nope FROM animals", "no such column"),
            ("SELECT 1 -- " + "x" * 100_000, "100,000-byte SQL limit"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(SQLiteQueryError, message) as caught:
                    self.executor.execute(sql)
                self.assertNotIsInstance(caught.exception, SQLiteInfrastructureError)

    def test_a_runaway_query_times_out(self):
        executor = ReadOnlySQLite(self.path, timeout_seconds=0.05)
        started = time.monotonic()
        with self.assertRaisesRegex(SQLiteQueryTimeout, "timed out after 0.05s"):
            executor.execute("WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n) SELECT COUNT(*) FROM n")
        self.assertLess(time.monotonic() - started, 5)

    def test_exploration_truncates_and_grading_refuses_extra_rows(self):
        truncated = ReadOnlySQLite(self.path, max_rows=2).execute("SELECT name FROM animals ORDER BY name")
        self.assertEqual(truncated.rows, (("ant",), ("cat",)))
        self.assertTrue(truncated.truncated)
        exact = ReadOnlySQLite(self.path, max_rows=4, truncate=False).execute("SELECT name FROM animals")
        self.assertFalse(exact.truncated)
        with self.assertRaisesRegex(SQLiteResultLimitExceeded, "more than 2 rows"):
            ReadOnlySQLite(self.path, max_rows=2, truncate=False).execute("SELECT name FROM animals")

    def test_results_are_charged_their_presented_bytes(self):
        # The same bound is SQLite's length limit for any string, BLOB or row record (schema rows included), so it
        # also stops one oversized value before it is materialized.
        executor = ReadOnlySQLite(self.path, max_bytes=60)
        self.assertEqual(executor.execute("SELECT 12345, 'abc'").rows, ((12345, "abc"),))
        self.assertEqual(len(executor.execute("SELECT name FROM animals").rows), 4)
        with self.assertRaisesRegex(SQLiteResultLimitExceeded, "60-byte limit"):
            executor.execute("WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<40) "
                             "SELECT x FROM n")
        with self.assertRaisesRegex(SQLiteResultLimitExceeded, "60-byte limit"):
            executor.execute("SELECT '" + "y" * 250 + "'")

    def test_optional_shape_limits(self):
        with self.assertRaisesRegex(SQLiteResultLimitExceeded, "more than 1 columns"):
            ReadOnlySQLite(self.path, max_columns=1).execute("SELECT name, legs FROM animals")
        with self.assertRaisesRegex(SQLiteResultLimitExceeded, "3-cell limit"):
            ReadOnlySQLite(self.path, max_cells=3).execute("SELECT name, legs FROM animals")
        with self.assertRaisesRegex(SQLiteResultLimitExceeded, "2-byte cell limit"):
            ReadOnlySQLite(self.path, max_cell_bytes=2).execute("SELECT name FROM animals")

    def test_a_missing_database_is_an_infrastructure_failure(self):
        executor = ReadOnlySQLite(Path(self._tmp.name) / "missing.sqlite")
        with self.assertRaisesRegex(SQLiteInfrastructureError, "not found"):
            executor.execute("SELECT 1")


class PinnedSnapshotTest(unittest.TestCase):
    """A task that pins its database's SHA-256 executes against a verified private copy."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = _database(self._tmp.name)
        self.sha = _sha256(self.path)
        self.root = Path(self._tmp.name) / "snapshots"
        patcher = mock.patch.dict(os.environ, {sqlite_exec.SNAPSHOT_ROOT_ENV_VAR: str(self.root)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_queries_run_on_a_private_content_addressed_copy(self):
        executor = ReadOnlySQLite(self.path, expected_sha256=self.sha)
        self.assertEqual(executor.execute("SELECT COUNT(*) FROM animals").rows, ((4,),))
        snapshot = self.root / f"{self.sha}.sqlite"
        self.assertEqual(stat.S_IMODE(self.root.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(snapshot.stat().st_mode), 0o400)
        self.assertEqual(_sha256(snapshot), self.sha)
        # A second executor reuses the published copy.
        self.assertEqual(ReadOnlySQLite(self.path, expected_sha256=self.sha).execute("SELECT 1").rows, ((1,),))

    def test_a_database_that_does_not_match_its_pin_is_refused(self):
        with self.assertRaisesRegex(SQLiteInfrastructureError, "differs from the frozen task contract"):
            ReadOnlySQLite(self.path, expected_sha256="0" * 64)

    def test_a_database_that_changes_after_verification_is_refused(self):
        executor = ReadOnlySQLite(self.path, expected_sha256=self.sha)
        connection = sqlite3.connect(self.path)
        connection.execute("INSERT INTO animals VALUES ('yak', 4)")
        connection.commit()
        connection.close()
        with self.assertRaises(SQLiteInfrastructureError):
            executor.execute("SELECT COUNT(*) FROM animals")

    def test_a_malformed_pin_is_rejected(self):
        for pin in ("abc", "A" * 64, 123):
            with self.subTest(pin=pin), self.assertRaisesRegex(ValueError, "64 lowercase hex digits"):
                ReadOnlySQLite(self.path, expected_sha256=pin)


class PrivateTempRootTest(unittest.TestCase):
    def test_created_private_and_existing_ones_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = prepare_private_temp_root(Path(directory) / "a" / "b")
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            self.assertEqual(prepare_private_temp_root(root), root)
            shared = Path(directory) / "shared"
            shared.mkdir(mode=0o755)
            shared.chmod(0o755)
            with self.assertRaisesRegex(RuntimeError, "not private"):
                prepare_private_temp_root(shared)
            link = Path(directory) / "link"
            link.symlink_to(root)
            with self.assertRaisesRegex(RuntimeError, "real directory"):
                prepare_private_temp_root(link)


class PresentationTest(unittest.TestCase):
    """What the model reads after a tool call; these strings are part of the prompt."""

    def test_a_result_renders_as_title_header_and_rows(self):
        result = SQLiteResult(columns=("COUNT(*)",), rows=((12,),), elapsed_seconds=0.0)
        # The exact observation the tool-use demonstration shows the model.
        self.assertEqual(SQLiteToolResult(result).format(), "SQL execution: 1 row(s)\nCOUNT(*)\n12")

    def test_a_truncated_result_says_so_twice(self):
        result = SQLiteResult(columns=("a", "b"), rows=((1, None), (b"\x00\xff", "x")), elapsed_seconds=0.0,
                              truncated=True)
        self.assertEqual(
            SQLiteToolResult(result).format(),
            "SQL execution: 2 row(s) (truncated)\na | b\n1 | NULL\nx'00ff' | x\n"
            "Result truncated to the configured exploration row limit.",
        )

    def test_an_empty_result_is_explicit(self):
        result = SQLiteResult(columns=("v",), rows=(), elapsed_seconds=0.0)
        self.assertEqual(SQLiteToolResult(result).format(), "SQL execution: 0 row(s)\nv\n(no rows)")

    def test_wide_cells_are_clipped_for_display_but_charged_in_full(self):
        wide = "é" * 250
        self.assertEqual(sqlite_exec.display_result_cell(wide), "é" * 200 + "...(truncated)")
        self.assertEqual(sqlite_exec.charged_cell_bytes(wide), 500)
        self.assertEqual(sqlite_exec.charged_cell_bytes(None), 4)
        self.assertEqual(sqlite_exec.charged_cell_bytes(b"\x01\x02"), 7)


class ResultPayloadTest(unittest.TestCase):
    def test_row_payload_hashes_every_row_and_previews_a_bounded_prefix(self):
        result = SQLiteResult(columns=("v",), rows=tuple((i,) for i in range(60)), elapsed_seconds=0.5)
        payload = sqlite_result_payload(result)
        self.assertEqual(payload["row_count"], 60)
        self.assertEqual(len(payload["rows_preview"]), 50)
        self.assertTrue(payload["preview_truncated"])
        self.assertEqual(payload["rows_preview"][0], [0])
        other = sqlite_result_payload(SQLiteResult(columns=("v",), rows=tuple((i,) for i in range(61)),
                                                   elapsed_seconds=0.5))
        self.assertNotEqual(payload["result_sha256"], other["result_sha256"])
        # The budget counts the persisted list: "[[0], [1]]" is 10 bytes, and a third row would make it 15.
        self.assertEqual(sqlite_result_payload(result, max_preview_bytes=14)["rows_preview"], [[0], [1]])
        self.assertEqual(sqlite_result_payload(result, max_preview_bytes=15)["rows_preview"], [[0], [1], [2]])

    def test_cells_are_json_safe(self):
        result = SQLiteResult(columns=("v",), rows=((b"\x01",), (float("inf"),)), elapsed_seconds=0.0)
        self.assertEqual(
            sqlite_result_payload(result)["rows_preview"],
            [[{"type": "bytes", "hex": "01"}], [{"type": "float", "value": "inf"}]],
        )

    def test_digest_payload(self):
        payload = sqlite_result_payload(_digest(columns=1, unique=3, sha="abc", preview=(("x",),)))
        self.assertEqual(payload["storage"], "streamed_disk_index")
        self.assertEqual(payload["unique_row_count"], 3)
        self.assertTrue(payload["preview_truncated"])


class SqlToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = _database(self._tmp.name)
        self.task = {"id": "zoo:1", "database_path": str(self.path)}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_the_tool_executor_truncates_at_the_described_row_cap(self):
        executor = sqlite_exec.tool_executor(self.task)
        self.assertEqual(
            (executor.max_rows, executor.max_bytes, executor.timeout_seconds, executor.truncate),
            (50, 1_000_000, 30.0, True),
        )
        self.assertIsNone(executor.expected_sha256)
        rows = sqlite_exec.execute_sql_tool(
            self.task, executor,
            {"query": "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<80) SELECT x FROM n"},
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0].result.rows), 50)
        self.assertTrue(rows[0].format().startswith("SQL execution: 50 row(s) (truncated)\nx\n1\n"))

    def test_the_tool_verifies_a_pinned_database(self):
        with mock.patch.dict(os.environ, {sqlite_exec.SNAPSHOT_ROOT_ENV_VAR: str(Path(self._tmp.name) / "snap")}):
            executor = sqlite_exec.tool_executor({**self.task, "database_sha256": _sha256(self.path)})
            self.assertEqual(executor.expected_sha256, _sha256(self.path))
            result = sqlite_exec.execute_sql_tool(self.task, executor, {"query": "SELECT COUNT(*) FROM animals"})
        self.assertEqual(result[0].format(), "SQL execution: 1 row(s)\nCOUNT(*)\n4")

    def test_a_missing_query_and_an_unsafe_query(self):
        executor = sqlite_exec.tool_executor(self.task)
        with self.assertRaisesRegex(SQLiteQueryError, "must not be empty"):
            sqlite_exec.execute_sql_tool(self.task, executor, {})
        with self.assertRaises(sqlite_exec.SQL_TOOL_TERMINAL_ERRORS):
            sqlite_exec.execute_sql_tool(self.task, executor, {"query": "DELETE FROM animals"})


class DatabaseGateTest(unittest.TestCase):
    def test_ungated_without_a_limit(self):
        gate = sqlite_exec.database_execution_gate({"database_path": "/nowhere/x.sqlite"})
        with gate:
            pass
        self.assertNotIsInstance(gate, threading.BoundedSemaphore)

    def test_one_gate_per_database_and_conflicting_limits_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            key = str(Path(directory) / "db.sqlite")
            first = sqlite_exec.database_execution_gate({"database_path": key, "grading_concurrency": 2})
            second = sqlite_exec.database_execution_gate({"database_path": key, "grading_concurrency": 2})
            self.assertIs(first, second)
            with self.assertRaisesRegex(RuntimeError, "conflicting SQL concurrency limits"):
                sqlite_exec.database_execution_gate({"database_path": key, "grading_concurrency": 3})


class SqlWorkTest(unittest.IsolatedAsyncioTestCase):
    async def test_concurrency_is_bounded(self):
        work = SqlWork(2)
        active, peak = 0, 0
        lock = threading.Lock()

        def job():
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return True

        self.assertEqual(await asyncio.gather(*(work.execute(job) for _ in range(6))), [True] * 6)
        self.assertEqual(peak, 2)
        await work.close()

    async def test_exclusive_work_runs_alone(self):
        work = SqlWork(3)
        running = []
        lock = threading.Lock()

        def job(name, seconds):
            with lock:
                running.append(("start", name))
            time.sleep(seconds)
            with lock:
                running.append(("end", name))

        await asyncio.gather(
            work.execute(job, "a", 0.05),
            work.execute(job, "b", 0.05),
            work.execute(job, "exclusive", 0.01, exclusive=True),
            work.execute(job, "c", 0.01),
        )
        start = running.index(("start", "exclusive"))
        end = running.index(("end", "exclusive"))
        self.assertEqual(running[start + 1], ("end", "exclusive"))
        self.assertEqual(end, start + 1)
        self.assertLess(running.index(("end", "a")), start)
        self.assertLess(running.index(("end", "b")), start)
        await work.close()

    async def test_a_cancelled_waiter_keeps_its_slot_until_the_thread_finishes(self):
        work = SqlWork(1)
        release = threading.Event()
        waiter = asyncio.create_task(work.execute(release.wait, 5))
        await asyncio.sleep(0.02)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertEqual(len(work.calls), 1)
        self.assertTrue(work.slots.locked())
        release.set()
        await work.close()
        self.assertEqual(len(work.calls), 0)
        self.assertFalse(work.slots.locked())


if __name__ == "__main__":
    unittest.main()
