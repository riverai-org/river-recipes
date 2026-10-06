"""Typed absolute-epsilon matching without rounding or merging distinct rows.

Inputs use sqlite_exec's exact canonical cells. Disk indexes keep large set
results bounded; the same matcher serves streamed and materialized results.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
from collections import deque
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path

EPSILON = Fraction(1, 1_000_000)
_INSERT_BATCH = 2000
# Python rational comparisons contend on the GIL. Queue expensive index work
# before starting its deadline instead of timing out many competing matchers.
_INDEX_SLOT = threading.BoundedSemaphore(1)
_MATCH = "results match with absolute numeric tolerance 1e-6"
_MISMATCH = "result values or row cardinalities differ"


# The message of every comparison that runs out of time; grading retries a comparison that fails with it.
TIME_BUDGET_MESSAGE = "numeric row matching exceeded its time budget"


class NumericComparisonError(RuntimeError):
    """Matching infrastructure failed; this is not a wrong model answer."""


@lru_cache(maxsize=4096)
def _fraction(value: str) -> Fraction:
    return Fraction(value)


def _parts(row):
    numbers, other = [], []
    for cell in row:
        if cell.startswith("number:") and "/" in cell[7:]:
            numbers.append(_fraction(cell[7:]))
        else:
            other.append(cell)
    return tuple(sorted(other)), tuple(sorted(numbers))


def _close(left, right):
    return (
        left[0] == right[0]
        and len(left[1]) == len(right[1])
        and all(abs(a - b) <= EPSILON for a, b in zip(left[1], right[1], strict=True))
    )


def _subset_count(method):
    if not method.startswith("subset,"):
        if method not in {"set", "multiset", "list", "ordered"}:
            raise ValueError(f"unsupported SQL grading method: {method!r}")
        return None
    try:
        prefix, operator, count = method.split(",")
        count = int(count)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid subset grading method: {method!r}") from exc
    if prefix != "subset" or operator not in {"=", ">="}:
        raise ValueError(f"invalid subset grading method: {method!r}")
    return operator, count


class _Matcher:
    def __init__(self, index, max_index_bytes, timeout_seconds):
        self.index = index
        self.deadline = time.monotonic() + timeout_seconds
        self.max_index_bytes = max_index_bytes
        index.execute("PRAGMA journal_mode=OFF")
        index.execute("PRAGMA synchronous=OFF")
        index.execute("PRAGMA cache_size=-2048")
        index.execute("PRAGMA temp_store=FILE")
        index.create_collation(
            "rational",
            lambda a, b: (_fraction(a) > _fraction(b)) - (_fraction(a) < _fraction(b)),
        )
        index.set_progress_handler(lambda: int(time.monotonic() > self.deadline), 10000)

    def check_budget(self):
        if time.monotonic() > self.deadline:
            raise NumericComparisonError(
                TIME_BUDGET_MESSAGE
            )
        size = (
            self.index.execute("PRAGMA page_count").fetchone()[0]
            * self.index.execute("PRAGMA page_size").fetchone()[0]
        )
        if size > self.max_index_bytes:
            raise NumericComparisonError(
                "numeric row matching exceeded its disk budget"
            )

    def load(self, table, rows, *, unique):
        # Only these internal identifiers are permitted; values are all bound.
        if table not in {"expected", "actual"}:
            raise ValueError("unknown matching table")
        constraint = "UNIQUE" if unique else ""
        self.index.execute(
            f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, value TEXT {constraint}, "
            "bucket TEXT, anchor TEXT COLLATE rational)"
        )
        statement = f"INSERT OR IGNORE INTO {table}(value,bucket,anchor) VALUES (?,?,?)"
        pending = []
        for row in rows:
            other, numbers = _parts(row)
            pending.append(
                (
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(
                        (other, len(numbers)), ensure_ascii=False, separators=(",", ":")
                    ),
                    str(numbers[0]) if numbers else "0",
                )
            )
            if len(pending) >= _INSERT_BATCH:
                self.index.executemany(statement, pending)
                pending.clear()
                self.check_budget()
        if pending:
            self.index.executemany(statement, pending)
        self.index.execute(f"CREATE INDEX {table}_bucket ON {table}(bucket,anchor)")
        self.check_budget()

    def match(self, subset):
        actual_count = self.index.execute("SELECT COUNT(*) FROM actual").fetchone()[0]
        expected_count = self.index.execute("SELECT COUNT(*) FROM expected").fetchone()[
            0
        ]
        if subset is not None:
            operator, count = subset
            if (operator == "=" and actual_count != count) or (
                operator == ">=" and actual_count < count
            ):
                return False
        elif actual_count != expected_count:
            return False
        self.index.execute(
            "CREATE TABLE matches (expected_id INTEGER PRIMARY KEY, actual_id INTEGER UNIQUE)"
        )
        for bucket, count in self.index.execute(
            "SELECT bucket,COUNT(*) FROM actual GROUP BY bucket"
        ):
            self.check_budget()
            expected_count = self.index.execute(
                "SELECT COUNT(*) FROM expected WHERE bucket=?", (bucket,)
            ).fetchone()[0]
            if count > expected_count or (subset is None and count != expected_count):
                return False
            number_count = json.loads(bucket)[1]
            if number_count and not (
                self.match_scalar(bucket)
                if number_count == 1
                else self.match_vectors(bucket)
            ):
                return False
        self.check_budget()
        return True

    def match_scalar(self, bucket):
        # Sorted scalar intervals have a monotone injective match, avoiding a
        # dense graph for large, nearly equal numeric populations.
        gold = iter(
            self.index.execute(
                "SELECT anchor FROM expected WHERE bucket=? ORDER BY anchor", (bucket,)
            )
        )
        value = next(gold, None)
        for (anchor,) in self.index.execute(
            "SELECT anchor FROM actual WHERE bucket=? ORDER BY anchor", (bucket,)
        ):
            self.check_budget()
            target = _fraction(anchor)
            while value is not None and _fraction(value[0]) < target - EPSILON:
                value = next(gold, None)
            if value is None or abs(_fraction(value[0]) - target) > EPSILON:
                return False
            value = next(gold, None)
        return True

    def candidates(self, left, bucket):
        encoded = self.index.execute(
            "SELECT value FROM actual WHERE id=?", (left,)
        ).fetchone()[0]
        row = _parts(json.loads(encoded))
        low, high = str(row[1][0] - EPSILON), str(row[1][0] + EPSILON)
        candidates = self.index.execute(
            "SELECT e.id,e.value,m.actual_id FROM expected e LEFT JOIN matches m ON m.expected_id=e.id "
            "WHERE e.bucket=? AND e.anchor>=? COLLATE rational AND e.anchor<=? COLLATE rational",
            (bucket, low, high),
        )
        for right, encoded, matched_left in candidates:
            # A dense Python-side scan must honor the deadline too.
            if time.monotonic() > self.deadline:
                raise NumericComparisonError(
                    TIME_BUDGET_MESSAGE
                )
            if _close(row, _parts(json.loads(encoded))):
                yield right, matched_left

    def find_path(self, start, bucket):
        queue = deque([start])
        parents = {start: None}
        seen_right = set()
        while queue:
            self.check_budget()
            left = queue.popleft()
            for right, matched_left in self.candidates(left, bucket):
                if right in seen_right:
                    continue
                seen_right.add(right)
                if matched_left is None:
                    return self.path(parents, left, right)
                if matched_left not in parents:
                    parents[matched_left] = (left, right)
                    queue.append(matched_left)
        return None

    @staticmethod
    def path(parents, left, right):
        path = []
        while True:
            path.append((right, left))
            parent = parents[left]
            if parent is None:
                return path
            left, right = parent

    def match_vectors(self, bucket):
        # Augmenting paths can reassign an earlier exact match. Fixing exact
        # rows greedily can otherwise reject a valid epsilon matching.
        self.index.execute("DELETE FROM matches")
        for (start,) in self.index.execute(
            "SELECT id FROM actual WHERE bucket=? ORDER BY anchor", (bucket,)
        ):
            path = self.find_path(start, bucket)
            if path is None:
                return False
            self.index.executemany(
                "DELETE FROM matches WHERE actual_id=?", ((pair[1],) for pair in path)
            )
            self.index.executemany(
                "INSERT OR REPLACE INTO matches(expected_id,actual_id) VALUES (?,?)",
                path,
            )
        return True


def _materialized_match(expected, actual, method, subset):
    """Resolve exact identities and impossible cardinalities without an index."""
    if method != "multiset":
        expected, actual = set(expected), set(actual)
    if subset is None and len(expected) != len(actual):
        return False
    if subset is not None:
        operator, count = subset
        if (
            (operator == "=" and len(actual) != count)
            or (operator == ">=" and len(actual) < count)
            or len(actual) > len(expected)
        ):
            return False
    if len(expected) == len(actual) == 1:
        return _close(_parts(next(iter(expected))), _parts(next(iter(actual))))
    if (
        (method == "set" and expected == actual)
        or (method == "multiset" and sorted(expected) == sorted(actual))
        or (subset is not None and actual <= expected)
    ):
        return True
    return None


def compare_numeric_rows(
    expected,
    actual,
    *,
    method,
    temp_root=None,
    max_index_bytes=2_000_000_000,
    timeout_seconds=120.0,
):
    """Return (correct, reason), preserving exact duplicate/cardinality rules.

    Set/multiset equality requires an injective match, not a nearest-neighbor
    existence test. Subsets inject actual rows into expected rows. Within-row
    column permutation is inherited from the existing grader.
    """
    subset = _subset_count(method)
    if method in {"list", "ordered"}:
        missing = object()
        correct = all(
            a is not missing and b is not missing and _close(_parts(a), _parts(b))
            for a, b in zip_longest(expected, actual, fillvalue=missing)
        )
        return correct, _MATCH if correct else _MISMATCH
    # Cardinality remains exact under epsilon; do not index impossible sets.
    if isinstance(expected, Sequence) and isinstance(actual, Sequence):
        correct = _materialized_match(expected, actual, method, subset)
        if correct is not None:
            return correct, _MATCH if correct else _MISMATCH
    with (
        _INDEX_SLOT,
        tempfile.TemporaryDirectory(
            prefix="sqlite-epsilon-", dir=temp_root
        ) as directory,
    ):
        index = sqlite3.connect(Path(directory) / "matching.sqlite")
        matcher = None
        try:
            matcher = _Matcher(index, max_index_bytes, timeout_seconds)
            matcher.load("expected", expected, unique=method != "multiset")
            matcher.load("actual", actual, unique=method != "multiset")
            correct = matcher.match(subset)
        except sqlite3.Error as exc:
            if matcher is not None and time.monotonic() > matcher.deadline:
                raise NumericComparisonError(
                    TIME_BUDGET_MESSAGE
                ) from exc
            raise NumericComparisonError("numeric row matching failed") from exc
        finally:
            index.close()
    return correct, _MATCH if correct else _MISMATCH
