"""Read-only SQLite execution, result comparison, and the SQL tool's execution.

- `ReadOnlySQLite` runs one bounded query against one database: SELECT-only (an authorizer denies everything else),
  a wall-clock timeout, row, byte and cell limits, and, when a task pins its database's SHA-256, execution against a
  verified private copy so a changed or corrupted file is an infrastructure failure rather than a wrong answer.
- `execute_set_digest` streams a very large result through an on-disk index and keeps only a digest, for gold
  results graded as sets.
- `compare_results` decides whether an executed answer matches the gold result under a task's grading method, with
  numeric cells compared under the `diamond-numeric-v1` policy (absolute difference at most 1e-6).
- `tool_executor` and `execute_sql_tool` run the model's `execute_sql_query` calls; `SQLiteToolResult.format()` is
  the text the model reads.
- `SqlWork` bounds the threads that run SQL for tools and grading together.

Errors are typed so callers can tell a wrong query from a broken environment: `SQLiteQueryError` and its subclasses
are failures of the query, except `SQLiteInfrastructureError`, which means the database or storage failed
independently of the query.
"""

from __future__ import annotations

import asyncio
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import numeric_compare

# Results are compared as sets unless a task is annotated with another grading method.
DEFAULT_GRADING_METHOD = "set"
# The only numeric comparison policy: numeric cells match within an absolute 1e-6, see `compare_results`. It is named
# for BIRD-Diamond, the dataset release that introduced it; every task row records it.
DIAMOND_NUMERIC_POLICY = "diamond-numeric-v1"
# Overrides the directory that holds private, content-addressed copies of verified databases.
SNAPSHOT_ROOT_ENV_VAR = "TEXT_TO_SQL_SQLITE_SNAPSHOT_ROOT"


class SQLiteQueryError(RuntimeError):
    """Base class for bounded, model-visible SQLite failures."""


class SQLiteInfrastructureError(SQLiteQueryError):
    """The database or execution environment failed independently of the query."""


class UnsafeSQLiteQuery(SQLiteQueryError):
    """The query attempted an operation outside the read-only contract."""


class SQLiteQueryTimeout(SQLiteQueryError):
    """The query exceeded its wall-clock execution budget."""


class SQLiteResultLimitExceeded(SQLiteQueryError):
    """The query produced more data than the configured grading limit."""


def prepare_private_temp_root(path: Path) -> Path:
    """Create or validate a caller-owned grading directory."""
    root = path.expanduser().absolute()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        pass
    metadata = root.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"grading temp root must be a real directory: {root}")
    if metadata.st_uid != os.getuid():
        raise RuntimeError(f"grading temp root is not owned by this process: {root}")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise RuntimeError(f"grading temp root is not private (mode 0700): {root}")
    descriptor = os.open(
        root,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    os.close(descriptor)
    return root.resolve()


@dataclass(frozen=True)
class SQLiteResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    elapsed_seconds: float
    truncated: bool = False


@dataclass(frozen=True)
class SQLiteSetDigest:
    """Disk-bounded summary for very large results graded as mathematical sets."""

    column_count: int
    row_count: int
    unique_row_count: int
    result_sha256: str
    rows_preview: tuple[tuple[str, ...], ...]
    elapsed_seconds: float
    numeric_policy: str = DIAMOND_NUMERIC_POLICY
    # Digests cannot encode non-transitive epsilon equality. Retain a replay
    # recipe, not rows or live file handles, for the uncommon digest mismatch.
    # Excluded from public receipts, equality and repr; remains pickleable.
    _replay: tuple[ReadOnlySQLite, str] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ResultComparison:
    correct: bool
    method: str
    message: str


@dataclass(frozen=True)
class SQLiteToolResult:
    """Agent-compatible rendering of one SQL execution result."""

    result: SQLiteResult

    @property
    def title(self) -> str:
        suffix = " (truncated)" if self.result.truncated else ""
        return f"SQL execution: {len(self.result.rows)} row(s){suffix}"

    @property
    def highlights(self) -> list[str]:
        return _table_lines(self.result)

    @property
    def notice(self) -> str:
        if self.result.truncated:
            return "Result truncated to the configured exploration row limit."
        return ""

    def format(self) -> str:
        lines = [self.title]
        lines.extend(self.highlights)
        if self.notice:
            lines.append(self.notice)
        return "\n".join(lines)


def _json_cell(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if isinstance(value, float) and not math.isfinite(value):
        return {"type": "float", "value": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__, "value": str(value)}


# No single cell may spend an unbounded amount of the reader's context, so a cell
# wider than this many characters is shown as its first
# `RESULT_CELL_DISPLAY_LIMIT` characters followed by this marker.
RESULT_CELL_DISPLAY_LIMIT = 200
RESULT_CELL_TRUNCATION_MARKER = "...(truncated)"


def format_result_cell(value: Any) -> str:
    """Return the untruncated presentation of one result cell.

    A BLOB is presented as its SQLite hex literal rather than as a Python
    ``bytes`` repr. The literal is deterministic at exactly two characters per
    byte whatever those bytes are, it is valid SQL the reader can paste back,
    and it is what :func:`charged_cell_bytes` charges. A repr costs between one
    and four characters per byte depending on content, so under a repr the same
    byte budget would buy a presentation of wildly varying size -- which is what
    makes a result-size limit unsound rather than merely imprecise.
    """
    if value is None:
        return "NULL"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"x'{bytes(value).hex()}'"
    return str(value)


def display_result_cell(value: Any) -> str:
    """Present one cell, clipped to the per-cell display limit."""
    rendered = format_result_cell(value)
    if len(rendered) <= RESULT_CELL_DISPLAY_LIMIT:
        return rendered
    return rendered[:RESULT_CELL_DISPLAY_LIMIT] + RESULT_CELL_TRUNCATION_MARKER


def charged_cell_bytes(value: Any) -> int:
    """Return the UTF-8 bytes one cell charges against the result byte limits.

    Every cell -- text, number, NULL, or BLOB -- is charged the size of the
    presentation a reader receives, so `max_bytes` and `max_cell_bytes` bound
    model-visible output instead of an internal representation that can render
    several times larger than it charges.
    """
    return len(format_result_cell(value).encode("utf-8", errors="replace"))


def sqlite_result_payload(
    result: SQLiteResult | SQLiteSetDigest,
    *,
    max_preview_rows: int = 50,
    max_preview_bytes: int | None = None,
) -> dict:
    """Return an auditable, JSON-safe summary of a complete SQLite result."""
    if isinstance(result, SQLiteSetDigest):
        preview_limit = max(0, int(max_preview_rows))
        preview = result.rows_preview[:preview_limit]
        return {
            "columns": [],
            "column_count": result.column_count,
            "row_count": result.row_count,
            "unique_row_count": result.unique_row_count,
            "rows_preview": [list(row) for row in preview],
            "preview_truncated": result.unique_row_count > len(preview),
            "result_sha256": result.result_sha256,
            "hash_semantics": (
                "SHA-256 over sorted unique grader-normalized rows and column count"
            ),
            "elapsed_seconds": result.elapsed_seconds,
            "execution_truncated": False,
            "storage": "streamed_disk_index",
        }
    columns = list(result.columns)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "columns": columns,
                "execution_truncated": result.truncated,
                "row_count": len(result.rows),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    preview = []
    # The run store uses json.dumps() defaults, including ensure_ascii=True and
    # ", " between list items. Match that representation exactly.
    preview_bytes = 2
    preview_open = True
    preview_byte_limit = (
        max(0, int(max_preview_bytes)) if max_preview_bytes is not None else None
    )
    for index, row in enumerate(result.rows):
        normalized = [_json_cell(value) for value in row]
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(b"\n")
        digest.update(encoded)
        if preview_open and index < max(0, int(max_preview_rows)):
            persisted_row_bytes = len(json.dumps(normalized).encode("utf-8")) + (
                2 if preview else 0
            )
            if (
                preview_byte_limit is not None
                and preview_bytes + persisted_row_bytes > preview_byte_limit
            ):
                preview_open = False
                continue
            preview.append(normalized)
            preview_bytes += persisted_row_bytes
    return {
        "columns": columns,
        "row_count": len(result.rows),
        "rows_preview": preview,
        "preview_truncated": len(preview) < len(result.rows),
        "result_sha256": digest.hexdigest(),
        "hash_semantics": "ordered columns and returned rows, including truncation",
        "elapsed_seconds": result.elapsed_seconds,
        "execution_truncated": result.truncated,
    }


_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_SELECT,
}
if hasattr(sqlite3, "SQLITE_RECURSIVE"):
    _ALLOWED_ACTIONS.add(sqlite3.SQLITE_RECURSIVE)

_BLOCKED_FUNCTIONS = {
    "fts3_tokenizer",
    "load_extension",
    "readfile",
    "writefile",
}
_SQLITE_SNAPSHOT_LOCK = threading.Lock()
_VERIFIED_SQLITE_SNAPSHOTS: dict[Path, tuple[int, ...]] = {}

# Source databases whose SHA-256 already verified, shared across executor
# instances so grading does not re-hash multi-hundred-MB files per call.
_VERIFIED_IDENTITY_LOCK = threading.Lock()
_VERIFIED_IDENTITY_CACHE: dict[tuple[str, str], tuple[int, ...]] = {}


@contextmanager
def _sqlite_snapshot_publication_lock(root: Path) -> Iterator[None]:
    """Serialize snapshot publication across threads and evaluator processes."""
    lock_path = root / ".publication.lock"
    with _SQLITE_SNAPSHOT_LOCK:
        descriptor = os.open(
            lock_path,
            os.O_CREAT
            | os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise SQLiteInfrastructureError(
                    "private database snapshot lock has unsafe metadata"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class ReadOnlySQLite:
    """Execute bounded SELECT queries against one immutable database snapshot."""

    def __init__(
        self,
        database: str | Path,
        *,
        timeout_seconds: float = 30.0,
        max_rows: int = 50,
        max_bytes: int = 1_000_000,
        max_columns: int | None = None,
        max_column_bytes: int | None = None,
        max_cells: int | None = None,
        max_cell_bytes: int | None = None,
        truncate: bool = True,
        set_index_root: str | Path | None = None,
        set_index_max_bytes: int = 2_000_000_000,
        expected_sha256: str | None = None,
        text_factory: Callable[[bytes], str] | None = None,
    ):
        self.database = Path(database).expanduser().resolve()
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        ):
            raise ValueError(
                "expected database SHA-256 must be 64 lowercase hex digits"
            )
        self.expected_sha256 = expected_sha256
        self._verified_database_identity: tuple[int, ...] | None = None
        self._execution_database = self.database
        self._execution_database_identity: tuple[int, ...] | None = None
        self.timeout_seconds = max(0.01, float(timeout_seconds))
        self.max_rows = max(1, int(max_rows))
        self.max_bytes = max(1, int(max_bytes))
        self.max_columns = max(1, int(max_columns)) if max_columns is not None else None
        self.max_column_bytes = (
            max(1, int(max_column_bytes)) if max_column_bytes is not None else None
        )
        self.max_cells = max(1, int(max_cells)) if max_cells is not None else None
        self.max_cell_bytes = (
            max(1, int(max_cell_bytes)) if max_cell_bytes is not None else None
        )
        self.truncate = bool(truncate)
        self.set_index_root = (
            Path(set_index_root).expanduser().absolute()
            if set_index_root is not None
            else None
        )
        self.set_index_max_bytes = max(1, int(set_index_max_bytes))
        self.text_factory = text_factory
        if self.expected_sha256 is not None:
            self._verify_database_snapshot()
            self._execution_database = self._prepare_execution_snapshot()

    @staticmethod
    def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    def _verify_database_snapshot(self) -> tuple[int, ...] | None:
        if self.expected_sha256 is None:
            return None
        try:
            before = self.database.stat()
        except OSError as exc:
            raise SQLiteInfrastructureError(
                f"database snapshot not found: {self.database}"
            ) from exc
        before_identity = self._file_identity(before)
        if not stat.S_ISREG(before.st_mode):
            raise SQLiteInfrastructureError(
                f"database snapshot is not a regular file: {self.database}"
            )
        if before_identity == self._verified_database_identity:
            return before_identity
        # Grading builds a fresh executor per call, so without a shared cache
        # each task would re-hash its whole database two to three times per
        # grade. The identity tuple (device, inode, size, mtime, ctime)
        # invalidates the cache if the file changes in any way.
        cache_key = (str(self.database), self.expected_sha256)
        with _VERIFIED_IDENTITY_LOCK:
            cached_identity = _VERIFIED_IDENTITY_CACHE.get(cache_key)
        if before_identity == cached_identity:
            self._verified_database_identity = before_identity
            return before_identity

        digest = hashlib.sha256()
        try:
            with self.database.open("rb") as database_file:
                for chunk in iter(lambda: database_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            after = self.database.stat()
        except OSError as exc:
            raise SQLiteInfrastructureError(
                f"could not verify database snapshot: {self.database}"
            ) from exc
        after_identity = self._file_identity(after)
        if before_identity != after_identity:
            raise SQLiteInfrastructureError(
                "database snapshot changed while its digest was verified"
            )
        if digest.hexdigest() != self.expected_sha256:
            raise SQLiteInfrastructureError(
                "database snapshot SHA-256 differs from the frozen task contract"
            )
        self._verified_database_identity = after_identity
        with _VERIFIED_IDENTITY_LOCK:
            _VERIFIED_IDENTITY_CACHE[cache_key] = after_identity
        return after_identity

    def _prepare_execution_snapshot(self) -> Path:
        """Copy verified bytes to a private content-addressed query snapshot."""
        assert self.expected_sha256 is not None
        configured_root = os.environ.get(SNAPSHOT_ROOT_ENV_VAR, "").strip()
        root = prepare_private_temp_root(
            Path(configured_root)
            if configured_root
            else Path(tempfile.gettempdir()) / "text-to-sql-sqlite-snapshots"
        )
        snapshot = root / f"{self.expected_sha256}.sqlite"
        with _sqlite_snapshot_publication_lock(root):
            if not snapshot.exists():
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{self.expected_sha256}.",
                    dir=root,
                )
                temporary = Path(temporary_name)
                digest = hashlib.sha256()
                try:
                    with (
                        self.database.open("rb") as source,
                        os.fdopen(descriptor, "wb") as destination,
                    ):
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(chunk)
                            destination.write(chunk)
                        destination.flush()
                        os.fsync(destination.fileno())
                    if digest.hexdigest() != self.expected_sha256:
                        raise SQLiteInfrastructureError(
                            "database snapshot changed while its private query "
                            "snapshot was created"
                        )
                    temporary.chmod(0o400)
                    os.replace(temporary, snapshot)
                finally:
                    temporary.unlink(missing_ok=True)
            metadata = snapshot.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o400
            ):
                raise SQLiteInfrastructureError(
                    "private database query snapshot has unsafe metadata"
                )
            identity = self._file_identity(metadata)
            if _VERIFIED_SQLITE_SNAPSHOTS.get(snapshot) != identity:
                digest = hashlib.sha256()
                with snapshot.open("rb") as snapshot_file:
                    for chunk in iter(
                        lambda: snapshot_file.read(1024 * 1024),
                        b"",
                    ):
                        digest.update(chunk)
                if digest.hexdigest() != self.expected_sha256:
                    raise SQLiteInfrastructureError(
                        "private database query snapshot SHA-256 is invalid"
                    )
                _VERIFIED_SQLITE_SNAPSHOTS[snapshot] = identity
            self._execution_database_identity = identity
        return snapshot

    def _verify_execution_snapshot(self) -> None:
        if self.expected_sha256 is None:
            return
        try:
            metadata = self._execution_database.stat()
        except OSError as exc:
            raise SQLiteInfrastructureError(
                "private database query snapshot is unavailable"
            ) from exc
        if self._file_identity(metadata) != self._execution_database_identity:
            raise SQLiteInfrastructureError(
                "private database query snapshot changed after verification"
            )

    @contextmanager
    def _cursor(self, query: str) -> Iterator[tuple[sqlite3.Cursor, float]]:
        sql = str(query or "").strip()
        if not sql:
            raise SQLiteQueryError("query must not be empty")
        if len(sql.encode("utf-8")) > 100_000:
            raise SQLiteQueryError("query exceeds the 100,000-byte SQL limit")
        if not self.database.is_file():
            raise SQLiteInfrastructureError(f"database snapshot not found: {self.database}")
        verified_identity = self._verify_database_snapshot()

        started = time.monotonic()
        timed_out = False
        denied: list[tuple[int, str, str]] = []

        def progress() -> int:
            nonlocal timed_out
            timed_out = time.monotonic() - started > self.timeout_seconds
            return 1 if timed_out else 0

        def authorize(
            action: int,
            arg1: str | None,
            arg2: str | None,
            database_name: str | None,
            trigger_name: str | None,
        ) -> int:
            del database_name, trigger_name
            function_name = (arg2 or arg1 or "").casefold()
            if (
                action == sqlite3.SQLITE_FUNCTION
                and function_name in _BLOCKED_FUNCTIONS
            ):
                denied.append((action, arg1 or "", arg2 or ""))
                return sqlite3.SQLITE_DENY
            if action not in _ALLOWED_ACTIONS:
                denied.append((action, arg1 or "", arg2 or ""))
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        self._verify_execution_snapshot()
        uri = f"{self._execution_database.as_uri()}?mode=ro&immutable=1"
        connection = sqlite3.connect(
            uri,
            uri=True,
            timeout=min(5.0, self.timeout_seconds),
        )
        try:
            if self.text_factory is not None:
                connection.text_factory = self.text_factory
            if (
                verified_identity is not None
                and self._file_identity(self.database.stat()) != verified_identity
            ):
                raise SQLiteInfrastructureError(
                    "database snapshot changed while opening the frozen task database"
                )
            if hasattr(connection, "enable_load_extension"):
                connection.enable_load_extension(False)
            if hasattr(connection, "setlimit"):
                connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, self.max_bytes)
                connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 100_000)
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.set_authorizer(authorize)
            connection.set_progress_handler(progress, 10_000)
            try:
                cursor = connection.execute(sql)
                yield cursor, started
            except SQLiteResultLimitExceeded:
                raise
            except sqlite3.DatabaseError as exc:
                if timed_out or "interrupted" in str(exc).casefold():
                    raise SQLiteQueryTimeout(
                        f"SQL query timed out after {self.timeout_seconds:g}s"
                    ) from exc
                if isinstance(exc, sqlite3.DataError) and (
                    "string or blob too big" in str(exc).casefold()
                ):
                    raise SQLiteResultLimitExceeded(
                        f"query result exceeded the {self.max_bytes}-byte limit"
                    ) from exc
                if denied:
                    raise UnsafeSQLiteQuery(
                        "query attempted an operation outside read-only SELECT execution"
                    ) from exc
                if (getattr(exc, "sqlite_errorcode", 0) & 0xFF) in {
                    sqlite3.SQLITE_IOERR, sqlite3.SQLITE_CANTOPEN,
                    sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB,
                    sqlite3.SQLITE_FULL, sqlite3.SQLITE_NOMEM,
                    sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED,
                    sqlite3.SQLITE_PERM, sqlite3.SQLITE_READONLY,
                }:
                    raise SQLiteInfrastructureError(f"SQLite execution failed: {exc}") from exc
                raise SQLiteQueryError(f"SQLite execution failed: {exc}") from exc
        finally:
            connection.set_progress_handler(None, 0)
            connection.set_authorizer(None)
            connection.close()
            if self.expected_sha256 is not None:
                self._verify_execution_snapshot()
                self._verify_database_snapshot()

    def execute(self, query: str) -> SQLiteResult:
        with self._cursor(query) as (cursor, started):
            columns = tuple(
                str(description[0]) for description in (cursor.description or ())
            )
            column_count = len(columns)
            if self.max_columns is not None and column_count > self.max_columns:
                raise SQLiteResultLimitExceeded(
                    f"query returned more than {self.max_columns} columns"
                )
            if (
                self.max_column_bytes is not None
                and max(
                    (charged_cell_bytes(column) for column in columns),
                    default=0,
                )
                > self.max_column_bytes
            ):
                raise SQLiteResultLimitExceeded(
                    "query returned a column name exceeding the "
                    f"{self.max_column_bytes}-byte limit"
                )
            bounded_rows: list[tuple[Any, ...]] = []
            byte_count = 0
            truncated = False
            for _ in range(self.max_rows + 1):
                row = cursor.fetchone()
                if row is None:
                    break
                if len(bounded_rows) == self.max_rows:
                    truncated = True
                    break
                if (
                    self.max_cells is not None
                    and (len(bounded_rows) + 1) * column_count > self.max_cells
                ):
                    raise SQLiteResultLimitExceeded(
                        f"query result exceeded the {self.max_cells}-cell limit"
                    )
                normalized_row = tuple(row)
                cell_lengths = tuple(
                    charged_cell_bytes(value) for value in normalized_row
                )
                if (
                    self.max_cell_bytes is not None
                    and max(cell_lengths, default=0) > self.max_cell_bytes
                ):
                    raise SQLiteResultLimitExceeded(
                        "query result contained a value exceeding the "
                        f"{self.max_cell_bytes}-byte cell limit"
                    )
                byte_count += sum(cell_lengths)
                if byte_count > self.max_bytes:
                    raise SQLiteResultLimitExceeded(
                        f"query result exceeded the {self.max_bytes}-byte limit"
                    )
                bounded_rows.append(normalized_row)

            if truncated and not self.truncate:
                raise SQLiteResultLimitExceeded(
                    f"query returned more than {self.max_rows} rows"
                )
            return SQLiteResult(
                columns=columns,
                rows=tuple(bounded_rows),
                elapsed_seconds=time.monotonic() - started,
                truncated=truncated,
            )

    def execute_set_digest(
        self, query: str, *, numeric_policy: str = DIAMOND_NUMERIC_POLICY, _consume_rows=None
    ) -> SQLiteSetDigest:
        """Stream a set-graded result through a disk index instead of RAM."""
        _validate_numeric_policy(numeric_policy)
        index_root = (
            prepare_private_temp_root(self.set_index_root)
            if self.set_index_root is not None
            else None
        )
        with tempfile.TemporaryDirectory(
            prefix="sqlite-result-set-",
            dir=index_root,
        ) as temp_dir:
            index = sqlite3.connect(Path(temp_dir) / "rows.sqlite")
            try:
                index.execute("PRAGMA journal_mode=OFF")
                index.execute("PRAGMA synchronous=OFF")
                index.execute(
                    "CREATE TABLE rows (value TEXT PRIMARY KEY) WITHOUT ROWID"
                )
                with self._cursor(query) as (cursor, started):
                    columns = tuple(
                        str(description[0])
                        for description in (cursor.description or ())
                    )
                    column_count = len(columns)
                    if self.max_columns is not None and column_count > self.max_columns:
                        raise SQLiteResultLimitExceeded(
                            f"query returned more than {self.max_columns} columns"
                        )
                    if (
                        self.max_column_bytes is not None
                        and max(
                            (charged_cell_bytes(column) for column in columns),
                            default=0,
                        )
                        > self.max_column_bytes
                    ):
                        raise SQLiteResultLimitExceeded(
                            "query returned a column name exceeding the "
                            f"{self.max_column_bytes}-byte limit"
                        )
                    row_count = 0
                    byte_count = 0
                    pending: list[tuple[str]] = []
                    while True:
                        row = cursor.fetchone()
                        if row is None:
                            break
                        normalized_row = tuple(row)
                        row_count += 1
                        if row_count > self.max_rows:
                            raise SQLiteResultLimitExceeded(
                                f"query returned more than {self.max_rows} rows"
                            )
                        if (
                            self.max_cells is not None
                            and row_count * column_count > self.max_cells
                        ):
                            raise SQLiteResultLimitExceeded(
                                f"query result exceeded the {self.max_cells}-cell limit"
                            )
                        cell_lengths = tuple(
                            charged_cell_bytes(value) for value in normalized_row
                        )
                        if (
                            self.max_cell_bytes is not None
                            and max(cell_lengths, default=0) > self.max_cell_bytes
                        ):
                            raise SQLiteResultLimitExceeded(
                                "query result contained a value exceeding the "
                                f"{self.max_cell_bytes}-byte cell limit"
                            )
                        byte_count += sum(cell_lengths)
                        if byte_count > self.max_bytes:
                            raise SQLiteResultLimitExceeded(
                                f"query result exceeded the {self.max_bytes}-byte limit"
                            )
                        value = json.dumps(
                            _normalize_row(normalized_row, numeric_policy),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        pending.append((value,))
                        if len(pending) >= 2_000:
                            index.executemany(
                                "INSERT OR IGNORE INTO rows(value) VALUES (?)",
                                pending,
                            )
                            pending.clear()
                            self._check_set_index_size(index)
                    if pending:
                        index.executemany(
                            "INSERT OR IGNORE INTO rows(value) VALUES (?)",
                            pending,
                        )
                        self._check_set_index_size(index)

                    unique_row_count = int(
                        index.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
                    )
                    digest = hashlib.sha256()
                    digest.update(
                        json.dumps(
                            {
                                "column_count": len(columns),
                                "unique_row_count": unique_row_count,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode()
                    )
                    preview: list[tuple[str, ...]] = []
                    for (value,) in index.execute(
                        "SELECT value FROM rows ORDER BY value"
                    ):
                        digest.update(b"\n")
                        digest.update(value.encode("utf-8"))
                        if len(preview) < 50:
                            preview.append(tuple(json.loads(value)))
                    result = SQLiteSetDigest(
                        column_count=len(columns),
                        row_count=row_count,
                        unique_row_count=unique_row_count,
                        result_sha256=digest.hexdigest(),
                        rows_preview=tuple(preview),
                        numeric_policy=numeric_policy,
                        elapsed_seconds=time.monotonic() - started,
                        _replay=(copy.copy(self), query)
                        if numeric_policy == DIAMOND_NUMERIC_POLICY and _consume_rows is None
                        else None,
                    )
                    if _consume_rows is not None:
                        _consume_rows(result, (
                            tuple(json.loads(value))
                            for (value,) in index.execute("SELECT value FROM rows ORDER BY value")
                        ))
                    return result
            finally:
                index.close()

    def _check_set_index_size(self, index: sqlite3.Connection) -> None:
        page_count = int(index.execute("PRAGMA page_count").fetchone()[0])
        page_size = int(index.execute("PRAGMA page_size").fetchone()[0])
        size = page_count * page_size
        if size > self.set_index_max_bytes:
            raise SQLiteResultLimitExceeded(
                f"streamed set index exceeded the {self.set_index_max_bytes}-byte limit"
            )


def compare_results(
    expected: SQLiteResult | SQLiteSetDigest,
    actual: SQLiteResult | SQLiteSetDigest,
    *,
    method: str = DEFAULT_GRADING_METHOD,
    numeric_policy: str = DIAMOND_NUMERIC_POLICY,
) -> ResultComparison:
    """Compare an executed result with the gold result under a grading method.

    Methods, matched case-insensitively after stripping whitespace:

    - ``set`` (the default): the distinct rows must match; row multiplicity is
      not part of the answer.
    - ``multiset``: as ``set``, with row multiplicity counted.
    - ``list`` (alias ``ordered``): rows must match in order.
    - ``subset,=,N`` / ``subset,>=,N``: exactly / at least N distinct rows, every
      one of them among the gold rows.

    Within a row, values are compared without regard to column order. Numeric
    cells match when their absolute difference is at most 1e-6
    (``diamond-numeric-v1``); text, NULL and BLOB stay distinct from numbers and
    from each other, so ``'24'`` never matches ``24``. Exact numeric identity
    still determines duplicate rows and subset cardinality, and matching is
    one-to-one, because epsilon equality is not transitive and must not merge
    nearby distinct rows.

    A row containing NULL or blank text is an answer and differs from a result
    with no rows. Two streamed set digests are compared by digest first; only a
    pair with equal cardinality and different hashes is replayed through the
    epsilon matcher.
    """
    _validate_numeric_policy(numeric_policy)
    normalized_method = (method or DEFAULT_GRADING_METHOD).strip().casefold()
    if isinstance(expected, SQLiteSetDigest) or isinstance(actual, SQLiteSetDigest):
        if not isinstance(expected, SQLiteSetDigest) or not isinstance(
            actual, SQLiteSetDigest
        ):
            raise TypeError("set-digest comparisons require two set digests")
        if expected.numeric_policy != numeric_policy or actual.numeric_policy != numeric_policy:
            raise ValueError("set digest numeric policy does not match comparison policy")
        if normalized_method != "set":
            raise ValueError("set digests may only use set grading")
        if expected.unique_row_count == actual.unique_row_count == 0:
            return ResultComparison(
                True,
                normalized_method,
                "both results are empty",
            )
        if expected.column_count != actual.column_count:
            return ResultComparison(False, normalized_method, "column counts differ")
        correct = (
            expected.unique_row_count == actual.unique_row_count
            and expected.result_sha256 == actual.result_sha256
        )
        if not correct and expected.unique_row_count == actual.unique_row_count:
            correct = _compare_numeric_digests(expected, actual)
        return ResultComparison(
            correct,
            normalized_method,
            "results match as streamed sets"
            if correct
            else "streamed result sets differ",
        )

    expected_rows = list(expected.rows)
    actual_rows = list(actual.rows)
    if expected_rows and actual_rows and len(expected_rows[0]) != len(actual_rows[0]):
        return ResultComparison(False, normalized_method, "column counts differ")
    if bool(expected_rows) != bool(actual_rows):
        return ResultComparison(False, normalized_method, "one result is empty")
    if not expected_rows:
        return ResultComparison(True, normalized_method, "both results are empty")

    expected_normalized = [_normalize_row(row, numeric_policy) for row in expected_rows]
    actual_normalized = [_normalize_row(row, numeric_policy) for row in actual_rows]
    correct, message = _compare_numeric_rows(
        expected_normalized, actual_normalized, method=normalized_method
    )
    return ResultComparison(correct, normalized_method, message)


def _compare_numeric_rows(expected, actual, **kwargs):
    if kwargs.get("temp_root") is not None:
        kwargs["temp_root"] = prepare_private_temp_root(kwargs["temp_root"])
    try:
        return numeric_compare.compare_numeric_rows(expected, actual, **kwargs)
    except (numeric_compare.NumericComparisonError, OSError) as exc:
        raise SQLiteInfrastructureError(str(exc)) from exc


def _compare_numeric_digests(expected: SQLiteSetDigest, actual: SQLiteSetDigest) -> bool:
    """Re-execute only mismatched streamed sets, keeping rows on bounded disk.

    Exact equality and cardinality failures need no replay. Validate each replay
    against its original exact digest so changed databases/nondeterministic SQL
    cannot silently replace the results being graded. Gold replay failures are
    infrastructure failures; candidate replay failures retain their category.
    """

    if expected._replay is None or actual._replay is None:
        raise SQLiteInfrastructureError("epsilon comparison requires replayable set results")
    gold_executor, gold_query = expected._replay
    actual_executor, actual_query = actual._replay
    comparison = None

    def consume_gold(gold_digest, gold_rows):
        if expected.result_sha256 != gold_digest.result_sha256:
            raise SQLiteInfrastructureError("streamed result changed during epsilon comparison")

        def consume_actual(actual_digest, actual_rows):
            nonlocal comparison
            if actual.result_sha256 != actual_digest.result_sha256:
                # The answer's own rows changed between runs (random(), the current time): it cannot be graded.
                raise SQLiteQueryError("the answer returned different rows when it was run again")
            comparison = _compare_numeric_rows(
                gold_rows, actual_rows, method="set",
                temp_root=actual_executor.set_index_root,
                max_index_bytes=actual_executor.set_index_max_bytes,
                timeout_seconds=actual_executor.timeout_seconds,
            )[0]

        actual_executor.execute_set_digest(
            actual_query, numeric_policy=DIAMOND_NUMERIC_POLICY,
            _consume_rows=consume_actual,
        )

    # Distinguish query failure in the gold replay from candidate failures
    # raised inside the consumer, which the harness must grade normally.
    candidate_failure = None
    def consume(gold_digest, gold_rows):
        nonlocal candidate_failure
        try:
            consume_gold(gold_digest, gold_rows)
        except SQLiteQueryError as exc:
            candidate_failure = exc
            raise

    try:
        gold_executor.execute_set_digest(
            gold_query, numeric_policy=DIAMOND_NUMERIC_POLICY, _consume_rows=consume,
        )
    except SQLiteQueryError as exc:
        if candidate_failure is exc:
            raise
        if isinstance(exc, SQLiteQueryTimeout):
            # A slow gold replay under load is retried, like any comparison that ran out of time.
            raise SQLiteInfrastructureError(numeric_compare.TIME_BUDGET_MESSAGE) from exc
        raise SQLiteInfrastructureError("gold replay failed during epsilon comparison") from exc
    if comparison is None:
        raise SQLiteInfrastructureError("epsilon comparison did not complete")
    return comparison


def _validate_numeric_policy(policy: str) -> None:
    if policy not in {DIAMOND_NUMERIC_POLICY}:
        raise ValueError(f"unsupported numeric comparison policy: {policy!r}")


def _numeric_cell_key(value: Any) -> str:
    # Exact rational identity avoids rounding large INTEGERs through float.
    # Prefixes keep numeric-looking text, NULL and BLOB distinct from numbers.
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            if math.isnan(value):
                raise ValueError("NaN is not a supported SQLite numeric result")
            return "number:" + str(value)
        numerator, denominator = value.as_integer_ratio()
        return f"number:{numerator}/{denominator}"
    if value is None:
        return "null:"
    if isinstance(value, bytes):
        return "blob:" + value.hex()
    return "text:" + str(value)


def _normalize_row(row: tuple[Any, ...], numeric_policy: str = DIAMOND_NUMERIC_POLICY) -> tuple[str, ...]:
    # Cell keys are sorted within a row, so two rows holding the same values in a
    # different projection order compare equal under every grading method.
    _validate_numeric_policy(numeric_policy)
    return tuple(sorted(_numeric_cell_key(value) for value in row))


def _table_lines(result: SQLiteResult) -> list[str]:
    lines: list[str] = []
    if result.columns:
        lines.append(" | ".join(result.columns))
    lines.extend(
        " | ".join(display_result_cell(value) for value in row) for row in result.rows
    )
    if not result.rows:
        lines.append("(no rows)")
    return lines


# --- Execution gating and scheduling ---------------------------------------

_DATABASE_GATES_LOCK = threading.Lock()
_DATABASE_GATES: dict[str, tuple[int, threading.BoundedSemaphore]] = {}


def database_execution_gate(task: dict):
    """Bound concurrent executions against one database when the task asks for it.

    A task carrying a positive ``grading_concurrency`` shares one bounded
    semaphore with every task on the same database file; a task without it runs
    ungated.
    """
    limit = int(task.get("grading_concurrency") or 0)
    if limit < 1:
        return nullcontext()
    key = str(task["database_path"])
    with _DATABASE_GATES_LOCK:
        configured = _DATABASE_GATES.get(key)
        if configured is None:
            gate = threading.BoundedSemaphore(limit)
            _DATABASE_GATES[key] = (limit, gate)
        else:
            configured_limit, gate = configured
            if configured_limit != limit:
                raise RuntimeError(
                    "conflicting SQL concurrency limits for database "
                    f"{key}: {configured_limit} and {limit}"
                )
        return gate


class SqlWork:
    """Bound tool/grader CPU work together; timeout retries drain local SQL work."""

    def __init__(self, concurrency):
        self.concurrency = concurrency
        self.slots = asyncio.Semaphore(concurrency)
        self.admission = asyncio.Lock()
        self.calls = set()

    async def execute(self, function, *args, exclusive=False, **kwargs):
        async def work():
            acquired = 0
            try:
                async with self.admission:
                    for index in range(self.concurrency if exclusive else 1):
                        await self.slots.acquire()
                        acquired = index + 1
                return await asyncio.to_thread(function, *args, **kwargs)
            finally:
                while acquired:
                    self.slots.release()
                    acquired -= 1

        pending = asyncio.create_task(work())
        self.calls.add(pending)

        def completed(task):
            self.calls.discard(task)
            if not task.cancelled():
                task.exception()  # Observe failures even if the waiter was cancelled.

        pending.add_done_callback(completed)
        # Cancelling an asyncio waiter cannot stop sqlite running in a thread.
        # The work task retains its slots until the thread actually finishes.
        return await asyncio.shield(pending)

    async def close(self):
        await asyncio.gather(*list(self.calls), return_exceptions=True)


# --- The model's SQL tool ----------------------------------------------------

# The SQL tool's execution limits. The tool description the model reads states
# the row cap, so it must be rendered from `SQL_TOOL_MAX_ROWS`. The byte cap is
# the executor default and is not described.
SQL_TOOL_TIMEOUT_SECONDS = 30
SQL_TOOL_MAX_ROWS = 50
SQL_TOOL_MAX_BYTES = 1_000_000
# Raising one of these ends the rollout as unsafe instead of returning an error
# the model can correct; every other `SQLiteQueryError` except
# `SQLiteInfrastructureError` is reported back to the model.
SQL_TOOL_TERMINAL_ERRORS = (UnsafeSQLiteQuery,)


def tool_executor(task: dict) -> ReadOnlySQLite:
    """The executor behind the model's SQL tool for one task.

    Exploration results are truncated to `SQL_TOOL_MAX_ROWS` rows rather than
    failing; the task's pinned database SHA-256, when present, is verified.
    """
    return ReadOnlySQLite(
        task["database_path"],
        expected_sha256=task.get("database_sha256") or None,
        timeout_seconds=float(SQL_TOOL_TIMEOUT_SECONDS),
        max_rows=SQL_TOOL_MAX_ROWS,
        max_bytes=SQL_TOOL_MAX_BYTES,
        truncate=True,
    )


def execute_sql_tool(
    task: dict,
    executor: ReadOnlySQLite,
    arguments: dict,
) -> list[SQLiteToolResult]:
    """Run one tool call's ``query`` argument; each result's ``format()`` is what the model reads."""
    with database_execution_gate(task):
        return [SQLiteToolResult(executor.execute(arguments.get("query") or ""))]
