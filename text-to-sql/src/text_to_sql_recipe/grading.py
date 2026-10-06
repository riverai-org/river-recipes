"""Execution-match grading: run the committed answer and compare its result with the gold query's.

`grade_with_expected` takes the final ``<solution>`` block of a response, executes it read-only under the task's
grading budgets, and compares the result with the precomputed gold result under the task's grading method
(`sqlite_exec.compare_results`). The verdict is CORRECT, INCORRECT (a wrong result, an execution error, a timeout,
a result over the limits, or an unsafe query) or NO_ANSWER (no single well-formed solution block). A failure of the
database or of storage raises `SQLiteInfrastructureError` and is never graded as a wrong answer.

A task with `numeric_tolerance` set is graded the way the ReViSQL authors grade (their `grade_single_number`): an
answer the comparison rejects still counts when the gold and the answer are each a single value (one row and one
column, after dropping rows with no non-empty cell) and differ by less than 1% of the gold. Only the Arcwise
evaluation sets it (`evaluation.arcwise_grading = "authors"`); training and validation never do.

`gold_results` executes every gold query once, before any rollout is graded. `GradingRuntime` grades rollouts
concurrently, shares one execution between rollouts that committed the same answer, and retries infrastructure
timeouts with the SQL work drained.

Task keys read here:
  id, domain, gold_sql, database_path       required
  database_sha256                           pins the database; verified before every execution
  grading_method                            default "set"
  numeric_policy                            default "diamond-numeric-v1", the only supported policy
  numeric_tolerance                         default false; true adds the authors' 1% single-value tolerance
  timeout_seconds                           prediction (and gold) budget, default 30 seconds
  gold_timeout_seconds                      gold-only budget, when it must exceed `timeout_seconds`
  grading_max_rows, grading_max_bytes       defaults 100,000 rows and 64,000,000 bytes
  streaming_set_grading                     grade through a streamed set digest (set method only)
  grading_temp_root, grading_max_index_bytes  where and how large that digest's disk index may grow
  grading_concurrency                       optional per-database execution gate
  gold_sql_overridden, source_gold_sql_sha256  copied into the grade receipt
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import shutil
import tempfile
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import numeric_compare
from .sqlite_exec import (
    DEFAULT_GRADING_METHOD,
    DIAMOND_NUMERIC_POLICY,
    ReadOnlySQLite,
    SQLiteInfrastructureError,
    SQLiteQueryError,
    SQLiteQueryTimeout,
    SQLiteResult,
    SQLiteResultLimitExceeded,
    SQLiteSetDigest,
    SqlWork,
    UnsafeSQLiteQuery,
    compare_results,
    database_execution_gate,
    prepare_private_temp_root,
    sqlite_result_payload,
)

# --- Grading budgets ------------------------------------------------------------------------------

# Streamed set grading keeps its disk index in a private directory.
STREAMING_GRADING_TEMP_ROOT = Path(tempfile.gettempdir()) / "text-to-sql-grading"
STREAMING_GRADING_MAX_INDEX_BYTES = 2_000_000_000

# Every gold query gets at least this budget, and up to this many attempts.
GOLD_SQL_TIMEOUT_SECONDS = 120
GOLD_SQL_ATTEMPTS = 3
# Retries of a grade that timed out, each with the SQL work drained.
GRADING_TIMEOUT_RETRIES = 2


def prepare_streaming_grading_storage(tasks: list[dict[str, Any]]) -> None:
    """Give streamed-set tasks a private index directory, after checking it has room for all of them."""
    streaming_tasks = [
        task for task in tasks if bool(task.get("streaming_set_grading"))
    ]
    if not streaming_tasks:
        return
    root = prepare_private_temp_root(STREAMING_GRADING_TEMP_ROOT)
    required_bytes = len(streaming_tasks) * STREAMING_GRADING_MAX_INDEX_BYTES
    available_bytes = shutil.disk_usage(root).free
    if available_bytes < required_bytes:
        raise RuntimeError(
            f"grading the {len(streaming_tasks)} questions with very large results "
            f"needs {required_bytes} free bytes under {root}, but only "
            f"{available_bytes} are available; set TMPDIR to a directory with room"
        )
    for task in streaming_tasks:
        task["grading_temp_root"] = str(root)
        task["grading_max_index_bytes"] = STREAMING_GRADING_MAX_INDEX_BYTES


# --- Grading one answer --------------------------------------------------------


def grading_method(task: Mapping[str, Any]) -> str:
    """The grading method for one task -- the single definition of the default.

    Every caller must use this helper rather than its own fallback: two
    fallbacks that differ (``or "multiset"`` against ``or "set"``) would grade
    the same answer two ways. Rows that carry no annotation are graded as sets,
    the BIRD standard that the annotated methods are exceptions to.
    """
    return str(task.get("grading_method") or DEFAULT_GRADING_METHOD)


def grade(report: str, task: dict) -> dict:
    """Grade one report by executing its SQL against the task's gold result."""
    expected = execute_gold_result(task)
    return grade_with_expected(report, task, expected)


def _grading_executor(
    task: dict,
    *,
    timeout_seconds: float | None = None,
) -> ReadOnlySQLite:
    return ReadOnlySQLite(
        task["database_path"],
        expected_sha256=task.get("database_sha256") or None,
        timeout_seconds=float(
            task.get("timeout_seconds", 30)
            if timeout_seconds is None
            else timeout_seconds
        ),
        max_rows=int(task.get("grading_max_rows", 100_000)),
        max_bytes=int(task.get("grading_max_bytes", 64_000_000)),
        truncate=False,
        set_index_root=task.get("grading_temp_root"),
        set_index_max_bytes=int(task.get("grading_max_index_bytes", 2_000_000_000)),
    )


def _execute_grading_result(
    task: dict,
    sql: str,
    *,
    timeout_seconds: float | None = None,
) -> SQLiteResult | SQLiteSetDigest:
    with database_execution_gate(task):
        executor = _grading_executor(task, timeout_seconds=timeout_seconds)
        if task.get("streaming_set_grading"):
            if grading_method(task).strip().casefold() != "set":
                raise RuntimeError(
                    "streaming set grading requires the ReViSQL set comparison method"
                )
            return executor.execute_set_digest(
                sql, numeric_policy=task.get("numeric_policy", DIAMOND_NUMERIC_POLICY)
            )
        return executor.execute(sql)


def execute_gold_result(task: dict) -> SQLiteResult | SQLiteSetDigest:
    try:
        return _execute_grading_result(
            task,
            task["gold_sql"],
            timeout_seconds=task.get("gold_timeout_seconds"),
        )
    except SQLiteQueryError as exc:
        raise RuntimeError(
            f"gold SQL failed for ReViSQL task {task['id']}: {exc}"
        ) from exc


def grade_with_expected(
    report: str,
    task: dict,
    expected: SQLiteResult | SQLiteSetDigest,
) -> dict:
    prediction = extract_solution(report)
    if prediction is None:
        return _grade_payload(
            task,
            prediction="",
            correct=False,
            grade_label="NO_ANSWER",
            reason="response did not contain exactly one <solution> SQL block",
            error_category="format",
            format_compliant=False,
        )

    try:
        actual = _execute_grading_result(task, prediction)
        # Comparison can execute the candidate again (the numeric set replay). Its failures are graded like the first
        # execution's; a gold replay failure arrives as SQLiteInfrastructureError and propagates.
        comparison = compare_results(
            expected,
            actual,
            method=grading_method(task),
            numeric_policy=task.get("numeric_policy", DIAMOND_NUMERIC_POLICY),
        )
    except UnsafeSQLiteQuery as exc:
        return _grade_payload(
            task,
            prediction=prediction,
            correct=False,
            grade_label="INCORRECT",
            reason=str(exc),
            error_category="unsafe_sql",
            format_compliant=True,
            expected_rows=_result_row_count(expected),
            expected_output=sqlite_result_payload(expected),
        )
    except SQLiteQueryTimeout as exc:
        return _grade_payload(
            task,
            prediction=prediction,
            correct=False,
            grade_label="INCORRECT",
            reason=str(exc),
            error_category="timeout",
            format_compliant=True,
            expected_rows=_result_row_count(expected),
            expected_output=sqlite_result_payload(expected),
        )
    except SQLiteResultLimitExceeded as exc:
        return _grade_payload(
            task,
            prediction=prediction,
            correct=False,
            grade_label="INCORRECT",
            reason=str(exc),
            error_category="result_limit",
            format_compliant=True,
            expected_rows=_result_row_count(expected),
            expected_output=sqlite_result_payload(expected),
        )
    except SQLiteInfrastructureError:
        # A missing/corrupted snapshot or failed storage is not evidence that
        # the model's SQL is wrong. Preserve the failure for run recovery.
        raise
    except SQLiteQueryError as exc:
        return _grade_payload(
            task,
            prediction=prediction,
            correct=False,
            grade_label="INCORRECT",
            reason=str(exc),
            error_category="execution",
            format_compliant=True,
            expected_rows=_result_row_count(expected),
            expected_output=sqlite_result_payload(expected),
        )

    correct, reason, tolerance_applied = comparison.correct, comparison.message, False
    if not correct and task.get("numeric_tolerance") and within_authors_tolerance(expected, actual):
        correct, tolerance_applied = True, True
        reason = "within 1% of the single-value gold (the ReViSQL authors' tolerance)"
    return _grade_payload(
        task,
        prediction=prediction,
        correct=correct,
        grade_label="CORRECT" if correct else "INCORRECT",
        reason=reason,
        error_category="" if correct else "wrong_result",
        format_compliant=True,
        tolerance_applied=tolerance_applied,
        expected_rows=_result_row_count(expected),
        actual_rows=_result_row_count(actual),
        execution_seconds=actual.elapsed_seconds,
        expected_output=sqlite_result_payload(expected),
        actual_output=sqlite_result_payload(actual),
    )


def _result_row_count(result: SQLiteResult | SQLiteSetDigest) -> int:
    return result.row_count if isinstance(result, SQLiteSetDigest) else len(result.rows)


def single_value(result: SQLiteResult | SQLiteSetDigest) -> float | None:
    """A result's one value as a number: one row and one column once rows with no non-empty cell are dropped.

    None for any other shape, a value that is not a number, or a set digest (which keeps no rows). This is the
    authors' `grade_single_number` reading of a result.
    """
    rows = [row for row in getattr(result, "rows", ()) if any(v is not None and str(v).strip() != "" for v in row)]
    if len(rows) != 1 or len(rows[0]) != 1:
        return None
    try:
        return float(rows[0][0])
    except (TypeError, ValueError):
        return None


def within_authors_tolerance(expected: SQLiteResult | SQLiteSetDigest, actual: SQLiteResult | SQLiteSetDigest) -> bool:
    """Whether two single-value results differ by less than 1% of the gold (never for a gold of zero)."""
    gold, answer = single_value(expected), single_value(actual)
    if gold is None or answer is None or gold == 0:
        return False
    return abs((gold - answer) / gold) < 0.01


def _grade_payload(
    task: dict,
    *,
    prediction: str,
    correct: bool,
    grade_label: str,
    reason: str,
    error_category: str,
    format_compliant: bool,
    expected_rows: int | None = None,
    actual_rows: int | None = None,
    execution_seconds: float | None = None,
    expected_output: dict | None = None,
    actual_output: dict | None = None,
    tolerance_applied: bool = False,
) -> dict:
    score = 1.0 if correct else 0.0
    return {
        "normalized": score,
        "pass_rate": score,
        "grade": grade_label,
        "prediction": prediction,
        "gold_sql": task["gold_sql"],
        "grading_method": grading_method(task),
        "numeric_policy": task.get("numeric_policy", DIAMOND_NUMERIC_POLICY),
        # Whether the task is graded with the authors' 1% single-value
        # tolerance, and whether this verdict needed it.
        "numeric_tolerance": bool(task.get("numeric_tolerance")),
        "numeric_tolerance_applied": tolerance_applied,
        "database_sha256": task.get("database_sha256", ""),
        "gold_sql_overridden": bool(task.get("gold_sql_overridden")),
        # The upstream query the override replaced, as a digest: it says *which*
        # gold this run was graded against without carrying a second copy of the
        # SQL into every task record.
        "source_gold_sql_sha256": task.get("source_gold_sql_sha256", ""),
        "format_compliant": format_compliant,
        "error_category": error_category,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "execution_seconds": execution_seconds,
        "expected_output": expected_output,
        "actual_output": actual_output,
        "criteria": [
            {
                "id": "execution",
                "section": task["domain"],
                "weight": 1.0,
                "requirement": (
                    "Predicted SQL must execute to the same result as the verified "
                    f"query using {grading_method(task)} comparison."
                ),
                "verdict": grade_label,
                "score": score,
                "errored": False,
                "reason": reason,
            }
        ],
        "errored": 0,
    }


def extract_solution(report: str) -> str | None:
    # Grade the final solution block, but require exactly one closing tag after
    # that final opening tag, and refuse a block with reasoning or tool markup
    # captured inside it.
    tail = (report or "").split("<solution>")[-1]
    if tail.count("</solution>") != 1:
        return None
    sql, _ = tail.split("</solution>", 1)
    sql = sql.strip()
    if not sql or re.search(r"</?(?:think|sql|observation)\b", sql, re.IGNORECASE):
        return None
    return sql


# --- Grading rollouts ----------------------------------------------------------


def gold_results(
    tasks: list[dict],
    *,
    concurrency: int,
    timeout_floor_seconds: int = GOLD_SQL_TIMEOUT_SECONDS,
    attempts: int = GOLD_SQL_ATTEMPTS,
    failures: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute every gold query once, up front.

    Rollouts are graded against these, so the gold SQL runs once per question
    instead of once per rollout -- at group size 16 that is a 16x saving on the
    most expensive part of grading. A gold query that times out is tried up to
    `attempts` times; any other failure (a database that fails its checksum or
    is missing, unusable storage) is final. A task whose gold query fails is
    absent from the result, so callers check coverage before training; pass
    `failures` to collect each such task's cause.
    """
    floor = int(timeout_floor_seconds)
    attempts = int(attempts)

    def one(task: dict) -> tuple[str, Any, str | None]:
        # The per-question overrides raise the budget where a question needs
        # it; this floor covers the rest, because several gold queries need far
        # longer than the 30-second prediction budget on a loaded machine.
        task_timeout_seconds = int(task.get("timeout_seconds", 0) or 0)
        configured_gold_timeout = task.get("gold_timeout_seconds")
        gold_timeout_seconds = max(
            floor,
            int(
                task_timeout_seconds
                if configured_gold_timeout is None
                else configured_gold_timeout
            ),
        )
        graded_task = {
            **task,
            "timeout_seconds": max(floor, task_timeout_seconds),
            # A gold-only override may be higher than the floor, but must never
            # lower it.
            "gold_timeout_seconds": gold_timeout_seconds,
        }
        for attempt in range(1, attempts + 1):
            try:
                return str(task["id"]), execute_gold_result(graded_task), None
            except RuntimeError as error:
                cause = error.__cause__ or error
                if attempt >= attempts or not isinstance(cause, SQLiteQueryTimeout):
                    return str(task["id"]), None, f"{type(cause).__name__}: {cause}"
                time.sleep(0.25 * attempt)
        return str(task["id"]), None, "no attempts"

    with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
        outcomes = list(pool.map(one, tasks))
    if failures is not None:
        failures.update({task_id: cause for task_id, result, cause in outcomes if result is None})
    return {task_id: result for task_id, result, _ in outcomes if result is not None}


def grade_rollout(
    rollout: Any,
    expected: Any,
    cfg: Mapping[str, Any],
    apply_penalties: Callable[[Any, Mapping[str, Any]], None] | None = None,
) -> None:
    """Grade one rollout's committed answer, then apply its per-rollout penalties.

    Reads ``rollout.task``, ``rollout.final_text`` and ``rollout.unsafe`` and
    sets ``rollout.grade``. An unsafe rollout -- one whose SQL tool call
    attempted an operation outside read-only SELECT execution -- is graded
    INCORRECT without executing anything. ``apply_penalties(rollout, cfg)``,
    when given, is the reward's per-trajectory step, which reads the rollout's
    own turns.
    """
    if rollout.unsafe:
        rollout.grade = {
            "grade": "INCORRECT",
            "error_category": "unsafe_sql",
            "format_compliant": False,
        }
    else:
        rollout.grade = grade_with_expected(
            rollout.final_text, rollout.task, expected
        )
    if apply_penalties is not None:
        apply_penalties(rollout, cfg)


def grading_dedup_key(rollout: Any) -> tuple[str, str, bool]:
    """Key answers whose result-grade and verifier outcome can be reused."""
    return (str(rollout.task["id"]), rollout.final_text, rollout.unsafe)


def grading_receipt(grade):
    """Keep the verdict and its evidence, without duplicating large result sets."""
    return copy.deepcopy(
        {
            k: v
            for k, v in grade.items()
            if k not in {"expected_output", "actual_output"}
        }
    )


def numeric_comparison_timed_out(error):
    # Other infrastructure failures (missing/corrupt databases, disk exhaustion)
    # must fail closed rather than be repeatedly retried.
    return (
        isinstance(error, SQLiteInfrastructureError)
        and str(error) == numeric_compare.TIME_BUDGET_MESSAGE
    )


class CompletedTaskCache(OrderedDict):
    """Bound retained results without evicting work that still has waiters."""

    def __init__(self, limit=1024):
        super().__init__()
        self.limit = limit

    def __setitem__(self, key, task):
        super().__setitem__(key, task)
        task.add_done_callback(lambda completed: self.prune())
        self.prune()

    def prune(self):
        completed = [key for key, task in self.items() if task.done()]
        for key in completed[: max(0, len(completed) - self.limit)]:
            del self[key]


class GradingRuntime:
    """Grade rollouts concurrently against precomputed gold results.

    Rollouts that committed the same answer to the same question
    (`grading_dedup_key`) share one grading task, and each receives its own copy
    of the grade. A grade whose query timed out or failed to execute is never
    reused, so a later duplicate grades it afresh.

    Timeouts are retried up to ``grading_timeout_retries`` times, each retry
    holding every SQL slot so it runs without contention: a prediction that
    still times out is graded INCORRECT (``error_category`` "timeout"); a
    numeric-matching timeout (`numeric_comparison_timed_out`) that persists is
    raised. Any other infrastructure failure propagates at once. Each attempt's
    receipt is kept in ``grade["grading_attempts"]``.

    ``expected`` maps task id to the gold result from `gold_results`.
    ``apply_penalties(rollout, cfg)``, when given, runs on each attempt's copy
    of the rollout, so attempt receipts carry the same fields as a finished
    grade. It is not applied to ``rollout`` itself: the caller does that after
    any verifier verdict, because penalties read the rollout's own trajectory
    while the cached grade is shared.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        expected: dict[str, Any],
        *,
        grading_concurrency: int,
        sql_concurrency: int,
        grading_timeout_retries: int = GRADING_TIMEOUT_RETRIES,
        sql_work: SqlWork | None = None,
        apply_penalties: Callable[[Any, Mapping[str, Any]], None] | None = None,
    ):
        self.cfg, self.expected = cfg, expected
        self.graders = asyncio.Semaphore(grading_concurrency)
        self.sql = sql_work if sql_work is not None else SqlWork(sql_concurrency)
        if self.sql.concurrency != sql_concurrency:
            raise ValueError(
                "Shared SQL scheduler disagrees with configured concurrency"
            )
        self.grading_timeout_retries = grading_timeout_retries
        self.apply_penalties = apply_penalties
        self.grades = CompletedTaskCache()

    async def grade(self, rollout):
        key = grading_dedup_key(rollout)
        if key not in self.grades:

            async def compute():
                async with self.graders:
                    attempts = []
                    for attempt in range(self.grading_timeout_retries + 1):
                        trial = copy.deepcopy(rollout)
                        started = time.monotonic()
                        try:
                            await self.sql.execute(
                                grade_rollout,
                                trial,
                                self.expected[str(trial.task["id"])],
                                self.cfg,
                                self.apply_penalties,
                                exclusive=attempt > 0,
                            )
                        except SQLiteInfrastructureError as error:
                            if not numeric_comparison_timed_out(error):
                                raise
                            attempts.append(
                                {
                                    "error_category": "infrastructure_timeout",
                                    "error": str(error),
                                    "wall_seconds": time.monotonic() - started,
                                }
                            )
                            print(
                                json.dumps(
                                    {
                                        "grading_infrastructure_retry": {
                                            "task_id": str(trial.task["id"]),
                                            "attempt": attempt + 1,
                                            "exhausted": attempt
                                            == self.grading_timeout_retries,
                                        }
                                    }
                                ),
                                flush=True,
                            )
                            if attempt == self.grading_timeout_retries:
                                raise
                            continue
                        attempts.append(
                            {
                                **grading_receipt(trial.grade),
                                "wall_seconds": time.monotonic() - started,
                                "timeout_seconds": trial.task.get(
                                    "timeout_seconds", 30
                                ),
                            }
                        )
                        if trial.grade.get("error_category") != "timeout":
                            break
                    trial.grade["grading_attempts"] = attempts
                    return trial.grade

            pending = asyncio.create_task(compute())
            self.grades[key] = pending

            def completed(task):
                # This also runs when every waiter has been cancelled. A failed
                # execution must never become a cached semantic judgment.
                failed = (
                    task.cancelled()
                    or task.exception() is not None
                    or task.result().get("error_category") in {"timeout", "execution"}
                )
                if failed and self.grades.get(key) is task:
                    del self.grades[key]

            pending.add_done_callback(completed)
        pending = self.grades[key]
        rollout.grade = copy.deepcopy(await asyncio.shield(pending))

    async def close(self):
        tasks = list(self.grades.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.sql.close()
