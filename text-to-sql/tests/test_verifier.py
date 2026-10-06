"""The VeriEQL verifier, tested with no VeriEQL, solver, network or cluster.

VeriEQL's process is replaced by a stub: `NativeVerifier._attempt` returns the helper's exit code and output, and
the helper script itself runs against a fake `environment`/`errors` source tree. The executed-counterexample gate runs for real on small in-memory schemas. Tests that call `confirm` past its early
returns parse SQL with sqlglot, which the gate imports only when it is on; they are skipped when sqlglot is not
installed, and the same gate scenarios also run through `refutation_gate` directly.

The reward penalizes exactly the verdicts whose `verdict` is "refuted", so the tests assert on that key.
"""
from __future__ import annotations

import asyncio
import collections
import functools
import hashlib
import importlib.util
import io
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from text_to_sql_recipe import sqlite_exec, verifier
from text_to_sql_recipe.config import ExecutionConfig, RewardConfig

HAS_SQLGLOT = importlib.util.find_spec("sqlglot") is not None
needs_sqlglot = unittest.skipUnless(HAS_SQLGLOT, "the gate lists referenced tables with sqlglot, an optional package")
SCHEMA_QUERY = "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
UNWITNESSED = (
    '{"result": false, "refutation_route": "solver_model", '
    '"counterexample": false, "counterexample_shape": "none"}'
)


def manifest_stub():
    return {'kind': 'verieql-native-linux-aarch64-v1', 'files': {}, 'python': 'python', 'source_root': '/source',
            'z3_root': '/site/z3', 'stdlib_root': '/stdlib',
            'site_packages_root': '/site'}


def source_schema(ddl):
    with closing(sqlite3.connect(':memory:')) as c:
        c.executescript(ddl)
        return c.execute(SCHEMA_QUERY).fetchall()


class EmbeddedScriptTest(unittest.TestCase):
    """The scripts run inside VeriEQL's interpreter; their text is pinned so that any change is deliberate."""

    def test_scripts_are_pinned(self):
        for script, digest in (
            (verifier._HELPER_SCRIPT, "43f19891d0d9a54307580b12cd2326ae902460dc5339d3f87dfda6e54eabae1a"),
            (verifier._EXTRACTION_SCRIPT, "d4a786c087682f06944696394c9f07eaa477fa5963e199fd2ec2337f42bfa8de"),
            (verifier._RUNTIME_PROBE, "012bc1897467f3e6bb472a10d532031f5f1fe4daccfa64d116047bc5449160e9"),
        ):
            self.assertEqual(hashlib.sha256(script.encode()).hexdigest(), digest)

    def test_helper_reserves_its_exit_code_for_its_own_failures(self):
        # The helper carries the literal and the module carries the constant, so
        # pin them together: they are one contract read from two sides, and a
        # drift between them would silently reclassify every adapter failure as
        # an undecided check.
        self.assertIn(
            f"sys.exit({verifier.HELPER_ADAPTER_FAILURE_EXIT_CODE})",
            verifier._HELPER_SCRIPT,
        )


class HelperScriptTest(unittest.TestCase):
    """The equivalence helper against a fake stock source tree: how each way a check can end is reported."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def _run_helper(
        self,
        analyze_body,
        *,
        accepts_environment_kwargs=True,
        create_database_body="pass",
        exit_body="return False",
        analyze_signature="self, *args",
        extra_files=None,
        stock_files=None,
        environment_is_package=False,
    ):
        root = Path(self.directory.name)
        environment_root = root / "VeriEQL"
        environment_root.mkdir(exist_ok=True)
        init = "def __init__(self, **kwargs): pass"
        if not accepts_environment_kwargs:
            init = "def __init__(self): pass"
        (environment_root / "errors.py").write_text(
            "class NotEquivalenceError(Exception):\n    pass\n"
            "class NotSupportedError(Exception):\n    pass\n"
            "class SyntaxError(Exception):\n    pass\n"
            "class UnknownError(Exception):\n    pass\n"
        )
        environment_module = environment_root / "environment.py"
        if environment_is_package:
            environment_module = environment_root / "environment" / "__init__.py"
            environment_module.parent.mkdir(exist_ok=True)
        environment_module.write_text(
            "from errors import (\n"
            "    NotEquivalenceError,\n"
            "    NotSupportedError,\n"
            "    SyntaxError,\n"
            "    UnknownError,\n"
            ")\n"
            "class Environment:\n"
            f"    {init}\n"
            "    def __enter__(self): return self\n"
            "    def __exit__(self, *args):\n"
            f"        {exit_body}\n"
            "    def create_database(self, **kwargs):\n"
            f"        {create_database_body}\n"
            "    def save_checkpoints(self): pass\n"
            f"    def analyze({analyze_signature}):\n"
            f"        {analyze_body}\n"
        )
        for name, source in (stock_files or {}).items():
            path = environment_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        helper_root = root / "work"
        helper_root.mkdir(exist_ok=True)
        for name, source in (extra_files or {}).items():
            # The helper script directory is on sys.path but outside the mocked
            # stock module tree, which lets tests distinguish an analyzer-origin
            # exception from an arbitrary callback failure.
            path = helper_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        helper = helper_root / "helper.py"
        helper.write_text(verifier._HELPER_SCRIPT)
        # Python caches bytecode on (mtime, size). Two stubs of the same length
        # written in the same mtime tick -- `self.counterexample = ''` and
        # `... = []` are both 2 characters -- resolve to the same cache entry,
        # so the second subprocess silently imports the first one's code. Purge
        # the cache rather than relying on stub bodies never colliding.
        for cached in root.rglob("__pycache__"):
            shutil.rmtree(cached, ignore_errors=True)
        return subprocess.run(
            [sys.executable, str(helper)],
            cwd=environment_root,
            input=json.dumps(
                {
                    "schema": {"orders": {"id": "INT"}},
                    "bound_size": 2,
                    "source_root": str(environment_root),
                    "z3_root": str(helper_root / "site-packages" / "z3"),
                    "stdlib_root": str(Path(os.__file__).resolve().parent),
                    "pred_sql": "SELECT id FROM orders",
                    "gold_sql": "SELECT id FROM orders",
                }
            ),
            check=False,
            capture_output=True,
            text=True,
        )

    def test_helper_reports_witness_free_nonequivalence_as_inconclusive(self):
        completed = self._run_helper(
            "raise NotEquivalenceError('Symbolic reasoning: NOT EQUIVALENT.')"
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "result": None,
                "error": "NotEquivalenceError: Symbolic reasoning: NOT EQUIVALENT.",
            },
        )

    def test_helper_reports_the_solver_counterexample_attestation(self):
        """`counterexample` is set the moment `solver.check()` returns `sat`."""
        completed = self._run_helper(
            "self.counterexample = '-- an counterexample found by VeriEQL'; "
            "return False"
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "result": False,
                "refutation_route": "solver_model",
                "counterexample": True,
                "counterexample_rows": 0,
                "counterexample_shape": "str",
            },
        )

    def test_the_attestation_survives_environment_teardown(self):
        """The read happens inside the `with`, so teardown cannot erase it.

        A stock build that cleared the attribute during teardown would otherwise
        turn every refutation into an unknown, the weight would never be
        subtracted, and every reported metric would still look plausible.
        """
        completed = self._run_helper(
            "self.counterexample = '-- found'; return False",
            # Teardown wipes exactly what the gate depends on.
            exit_body="self.counterexample = None; "
            "self.counterexample_dict = {}; return False",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(
            [payload["counterexample"], payload["counterexample_shape"]],
            [True, "str"],
        )

    def test_the_attestation_shape_is_reported_for_every_route(self):
        """`is not None` is a guess about a third-party sentinel; record the fact.

        A build that pre-initialised the attribute falsy-but-not-None would make
        the gate a no-op. It would still be a no-op -- but the record would say
        so, instead of the change being invisible.
        """
        for body, shape, attested in (
            ("self.counterexample = '-- found'; return False", "str", True),
            ("self.counterexample = ''; return False", "str:empty", True),
            ("self.counterexample = []; return False", "empty:list", True),
            ("self.counterexample = None; return False", "none", False),
            ("return False", "absent", False),
        ):
            with self.subTest(shape=shape):
                completed = self._run_helper(body)
                payload = json.loads(completed.stdout)
                self.assertEqual(payload["counterexample_shape"], shape)
                self.assertIs(payload["counterexample"], attested)

    def test_helper_reports_a_refutation_that_never_reached_the_solver(self):
        """No `counterexample` attribute means no `solver.check()` call."""
        completed = self._run_helper("return False")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "result": False,
                "refutation_route": "solver_model",
                "counterexample": False,
                "counterexample_rows": 0,
                "counterexample_shape": "absent",
            },
        )

    def test_helper_reports_the_projection_arity_route(self):
        completed = self._run_helper("return -1")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "result": False,
                "refutation_route": "column_arity",
                "counterexample": False,
                "counterexample_rows": 0,
                "witness_gate": "column_mismatch",
                "error": "column mismatch (result=-1)",
            },
        )

    def test_helper_counts_materialized_witness_rows_when_they_exist(self):
        """`generate_code=True` would populate these; today they are always 0."""
        completed = self._run_helper(
            "self.counterexample = '-- header'; "
            "self.counterexample_dict = {'ORDERS': [{'ID': '1'}, {'ID': 'NULL'}]}; "
            "return False"
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "result": False,
                "refutation_route": "solver_model",
                "counterexample": True,
                "counterexample_rows": 2,
                "counterexample_shape": "str",
            },
        )

    def test_helper_preserves_known_inconclusive_outcomes(self):
        for exception in (
            "NotSupportedError",
            "SyntaxError",
            "UnknownError",
            "NotImplementedError",
        ):
            with self.subTest(exception=exception):
                completed = self._run_helper(
                    f"raise {exception}('known solver outcome')"
                )

                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(
                    json.loads(completed.stdout),
                    {"result": None, "error": f"{exception}: known solver outcome"},
                )

    def test_helper_maps_stock_analyzer_errors_to_released_unknown_semantics(self):
        for exception, message in (
            ("TypeError", "expected string or bytes-like object, got 'list'"),
            ("AttributeError", "'int' object has no attribute 'sort'"),
            ("AttributeError", "'FExpressionTuple' object has no attribute 'sort'"),
            (
                "AttributeError",
                "'FProjectionTable' object has no attribute 'uninterpreted_func'",
            ),
            ("TypeError", "unsupported operand type(s) for &: 'int' and 'str'"),
            ("ValueError", "unexpected stock expression shape"),
            ("RuntimeError", "unexpected stock analyzer failure"),
        ):
            with self.subTest(exception=exception):
                completed = self._run_helper(f"raise {exception}({message!r})")

                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(
                    json.loads(completed.stdout),
                    {
                        "result": None,
                        "stock_analyzer_error": True,
                        "error": f"{exception}: {message}",
                    },
                )

    def test_helper_reports_stock_teardown_attribute_errors_as_inconclusive(self):
        completed = self._run_helper(
            "return True",
            exit_body="raise AttributeError(\"'int' object has no attribute 'sort'\")",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "result": None,
                "stock_analyzer_error": True,
                "error": "AttributeError: 'int' object has no attribute 'sort'",
            },
        )

    def test_helper_preserves_known_teardown_outcomes_as_inconclusive(self):
        completed = self._run_helper(
            "return True",
            exit_body="raise NotSupportedError('teardown solver outcome')",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "result": None,
                "error": "NotSupportedError: teardown solver outcome",
            },
        )

    def test_helper_keeps_stock_z3_dependency_errors_inconclusive(self):
        completed = self._run_helper(
            "from pathlib import Path; import sys; "
            "sys.path.append(str(Path(__file__).resolve().parents[1] / 'work' / 'site-packages')); "
            "from z3 import fail; return fail()",
            extra_files={
                "site-packages/z3/__init__.py": (
                    "def fail():\n"
                    "    raise AttributeError(\"'int' object has no attribute 'sort'\")\n"
                )
            },
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "result": None,
                "dependency_error": True,
                "error": "AttributeError: 'int' object has no attribute 'sort'",
            },
        )

    def test_helper_keeps_non_pinned_z3_errors_fatal(self):
        completed = self._run_helper(
            "from z3 import fail; return fail()",
            extra_files={
                "z3/__init__.py": (
                    "def fail():\n"
                    "    raise AttributeError(\"'int' object has no attribute 'sort'\")\n"
                )
            },
        )

        self.assertEqual(completed.returncode, verifier.HELPER_ADAPTER_FAILURE_EXIT_CODE)
        self.assertIn("AttributeError", completed.stderr)

    def test_helper_keeps_setup_phase_z3_errors_fatal(self):
        completed = self._run_helper(
            "return True",
            create_database_body=(
                "from pathlib import Path; import sys; "
                "sys.path.append(str(Path(__file__).resolve().parents[1] / 'work' / 'site-packages')); "
                "from z3 import fail; fail()"
            ),
            extra_files={
                "site-packages/z3/__init__.py": (
                    "def fail():\n"
                    "    raise AttributeError(\"'int' object has no attribute 'sort'\")\n"
                )
            },
        )

        self.assertEqual(completed.returncode, verifier.HELPER_ADAPTER_FAILURE_EXIT_CODE)
        self.assertIn("AttributeError", completed.stderr)

    def test_helper_keeps_setup_phase_known_outcomes_fatal(self):
        completed = self._run_helper(
            "return True",
            create_database_body="raise NotSupportedError('setup solver outcome')",
        )

        self.assertEqual(completed.returncode, verifier.HELPER_ADAPTER_FAILURE_EXIT_CODE)
        self.assertIn("NotSupportedError: setup solver outcome", completed.stderr)

    def test_helper_passes_two_string_sql_arguments_to_analyze(self):
        completed = self._run_helper(
            "assert args == ('SELECT id FROM orders', 'SELECT id FROM orders'); return True"
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), {"result": True})

    def test_helper_keeps_analyzer_signature_mismatch_fatal(self):
        completed = self._run_helper("return True", analyze_signature="self, pred_sql")

        self.assertEqual(completed.returncode, verifier.HELPER_ADAPTER_FAILURE_EXIT_CODE)
        self.assertIn("TypeError", completed.stderr)

    def test_helper_keeps_external_callback_errors_fatal(self):
        for exception, message in (
            ("TypeError", "expected string or bytes-like object, got 'list'"),
            ("RuntimeError", "unexpected callback failure"),
        ):
            with self.subTest(exception=exception):
                completed = self._run_helper(
                    "from external import fail; return fail()",
                    extra_files={
                        "external.py": (
                            f"def fail():\n    raise {exception}({message!r})\n"
                        )
                    },
                )

                self.assertEqual(
                    completed.returncode, verifier.HELPER_ADAPTER_FAILURE_EXIT_CODE
                )
                self.assertIn(exception, completed.stderr)

    def test_helper_keeps_external_known_verieql_errors_fatal(self):
        completed = self._run_helper(
            "from external import fail; return fail()",
            extra_files={
                "external.py": (
                    "from errors import NotSupportedError\n"
                    "def fail():\n"
                    "    raise NotSupportedError('external callback failure')\n"
                )
            },
        )

        self.assertEqual(completed.returncode, verifier.HELPER_ADAPTER_FAILURE_EXIT_CODE)
        self.assertIn("NotSupportedError: external callback failure", completed.stderr)

    def test_an_unrecognised_exception_origin_stays_fatal_end_to_end(self):
        """The decision, closed across both processes rather than either alone.

        The helper's two re-raise paths -- an exception whose origin it cannot
        place, and a known solver error type raised from outside the pinned
        solver -- are our adapter breaking, not the third-party verifier being
        stuck. They stay fatal. Run the real helper down each path, take the code
        it actually exits with, and put that same code through the native
        verifier. Asserting the two halves separately would pass with the
        contract between them broken.
        """
        for description, analyze_body, extra_files in (
            (
                "unrecognised origin",
                "from external import fail; return fail()",
                {"external.py": "def fail():\n    raise RuntimeError('outside')\n"},
            ),
            (
                "known solver error from outside the pinned solver",
                "from external import fail; return fail()",
                {
                    "external.py": (
                        "from errors import NotSupportedError\n"
                        "def fail():\n"
                        "    raise NotSupportedError('outside')\n"
                    )
                },
            ),
        ):
            with self.subTest(path=description):
                completed = self._run_helper(analyze_body, extra_files=extra_files)
                observed = completed.returncode

                self.assertEqual(observed, verifier.HELPER_ADAPTER_FAILURE_EXIT_CODE)

                native = verifier.NativeVerifier(manifest_stub(), "identity", Path(self.directory.name) / "receipts")
                failure = subprocess.CompletedProcess(["python"], observed, "", "RuntimeError: outside")
                with patch.object(native, "_attempt", AsyncMock(return_value=failure)), patch.object(
                    verifier, "_input_row", return_value={}
                ), self.assertRaisesRegex(verifier.VeriEQLError, "adapter failure"):
                    asyncio.run(native.check({"id": "task-1", "gold_sql": "SELECT 1"}, "SELECT 2"))

    def test_helper_preserves_stock_stdlib_analyzer_errors_as_unknown(self):
        completed = self._run_helper("import re; return re.sub('x', 'y', [])")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertIsNone(result["result"])
        self.assertTrue(result["stock_unknown"])
        self.assertTrue(result["error"].startswith("TypeError: "))

    def test_helper_preserves_single_file_stock_stdlib_errors_as_unknown(self):
        completed = self._run_helper("import glob; return glob.glob(None)")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertIsNone(result["result"])
        self.assertTrue(result["stock_unknown"])
        self.assertTrue(result["error"].startswith("TypeError: "))

    def test_helper_keeps_mid_stack_third_party_stdlib_errors_fatal(self):
        completed = self._run_helper(
            "from thirdparty import fail; return fail()",
            extra_files={
                "thirdparty.py": (
                    "import re\ndef fail():\n    return re.sub('x', 'y', [])\n"
                )
            },
        )

        self.assertEqual(completed.returncode, verifier.HELPER_ADAPTER_FAILURE_EXIT_CODE)
        self.assertIn("TypeError", completed.stderr)

    def test_helper_keeps_synthetic_analyzer_frames_fatal(self):
        completed = self._run_helper(
            "exec(\"raise TypeError('expected string or bytes-like object, got list')\")"
        )

        self.assertEqual(completed.returncode, verifier.HELPER_ADAPTER_FAILURE_EXIT_CODE)
        self.assertIn("TypeError", completed.stderr)

    def test_helper_tolerates_sibling_stock_module_errors_for_package_layout(self):
        completed = self._run_helper(
            "from formula import fail; return fail()",
            environment_is_package=True,
            stock_files={
                "formula.py": (
                    "def fail():\n"
                    "    raise AttributeError(\"'int' object has no attribute 'sort'\")\n"
                )
            },
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "result": None,
                "stock_analyzer_error": True,
                "error": "AttributeError: 'int' object has no attribute 'sort'",
            },
        )

    def test_helper_keeps_environment_setup_type_error_fatal(self):
        completed = self._run_helper("return True", accepts_environment_kwargs=False)

        self.assertEqual(completed.returncode, verifier.HELPER_ADAPTER_FAILURE_EXIT_CODE)
        self.assertIn("TypeError", completed.stderr)

    def test_helper_keeps_setup_body_errors_fatal(self):
        completed = self._run_helper(
            "return True",
            create_database_body="raise AttributeError(\"'int' object has no attribute 'sort'\")",
        )

        self.assertEqual(completed.returncode, verifier.HELPER_ADAPTER_FAILURE_EXIT_CODE)
        self.assertIn(
            "AttributeError: 'int' object has no attribute 'sort'", completed.stderr
        )

    def test_helper_maps_stock_runtime_errors_to_released_unknown_semantics(self):
        completed = self._run_helper("raise RuntimeError('unexpected helper failure')")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "result": None,
                "stock_analyzer_error": True,
                "error": "RuntimeError: unexpected helper failure",
            },
        )


class NativeVerdictTest(unittest.IsolatedAsyncioTestCase):
    """How the helper's exit code and output become a verdict in the native verifier (the process is stubbed)."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        database = root / "orders.sqlite"
        with closing(sqlite3.connect(database)) as connection:
            connection.executescript(
                """
                CREATE TABLE orders (
                    id INTEGER PRIMARY KEY,
                    placed_at DATE,
                    amount NUMERIC
                );
                CREATE TABLE auxiliary (label TEXT);
                """
            )
        self.task = {"id": "task-1", "database_path": str(database), "gold_sql": "SELECT id FROM orders"}
        self.receipts = root / "receipts"
        self.verifier = verifier.NativeVerifier(manifest_stub(), "identity", self.receipts)
        verifier.extract_schema_from_sqlite.cache_clear()
        self.addCleanup(verifier.extract_schema_from_sqlite.cache_clear)

    async def check(self, *outcomes):
        """One check whose helper attempts end as `outcomes`: (returncode, stdout, stderr), or None for a timeout."""
        completed = [
            None if outcome is None else subprocess.CompletedProcess(["python"], *outcome) for outcome in outcomes
        ]
        with patch.object(self.verifier, "_attempt", AsyncMock(side_effect=completed)) as attempt:
            result = await self.verifier.check(self.task, " SELECT id FROM orders ")
        self.attempt = attempt
        return result

    async def one_verdict(self, stdout):
        return await self.check((0, stdout, ""))

    async def test_the_solver_reads_every_table_with_coarse_types_and_the_manifest_roots(self):
        verdict = await self.one_verdict('{"result": true}')

        row = self.attempt.await_args.args[0]
        self.assertEqual(
            row["schema"],
            {
                "orders": {"id": "INT", "placed_at": "VARCHAR", "amount": "INT"},
                "auxiliary": {"label": "VARCHAR"},
            },
        )
        self.assertNotIn("constraint", row)
        self.assertEqual(row["bound_size"], 2)
        self.assertEqual([row["pred_sql"], row["gold_sql"]], ["SELECT id FROM orders"] * 2)
        self.assertEqual([row["source_root"], row["z3_root"], row["stdlib_root"]], ["/source", "/site/z3", "/stdlib"])
        self.assertLessEqual(self.attempt.await_args.args[1], 30)
        self.assertEqual(verdict["verdict"], "confirmed")
        self.assertEqual(verdict["timeout_seconds"], 30)

    async def test_a_verdict_carries_the_runtime_fields_and_leaves_a_receipt(self):
        verdict = await self.one_verdict('{"result": true}')

        fields = ("verdict", "state", "backend", "runtime_sha256", "bound", "startup_retries")
        self.assertEqual(
            {key: verdict[key] for key in fields},
            {"verdict": "confirmed", "state": "EQU", "backend": "native-linux-aarch64", "runtime_sha256": "identity",
             "bound": 2, "startup_retries": 0},
        )
        self.assertGreaterEqual(verdict["queue_seconds"], 0)
        self.assertGreaterEqual(verdict["service_seconds"], 0)
        receipts = [json.loads(p.read_text()) for p in self.receipts.glob("*.json")
                    if not p.name.startswith("inventory-")]
        self.assertEqual(len(receipts), 1)
        self.assertEqual(
            [receipts[0]["task_id"], receipts[0]["outcome"], receipts[0]["failed_attempts"]], ["task-1", verdict, []]
        )
        self.assertEqual([self.verifier.counts["checked"], self.verifier.counts["EQU"]], [1, 1])

    async def test_refuted_and_unknown_results_are_distinguished(self):
        result = [
            await self.one_verdict(stdout)
            for stdout in (
                '{"result": false, "counterexample": true}',
                '{"result": null, "error": "unsupported"}',
                ('{"result": null, "dependency_error": true, '
                "\"error\": \"AttributeError: 'int' object has no attribute 'sort'\"}"),
                ('{"result": null, "stock_unknown": true, '
                '"error": "TypeError: expected string or bytes-like object, got \'list\'"}'),
            )
        ]
        self.assertEqual(
            [row["verdict"] for row in result],
            ["refuted", "inconclusive", "inconclusive", "inconclusive"],
        )
        self.assertEqual(
            [row["state"] for row in result],
            ["NEQ", "UNKNOWN", "UNKNOWN", "UNKNOWN"],
        )
        self.assertEqual(
            [row.get("dependency_error") for row in result],
            [None, False, True, False],
        )
        self.assertEqual(
            [row.get("stock_unknown") for row in result],
            [None, False, False, True],
        )

    async def test_a_refutation_without_a_witness_is_not_a_refutation(self):
        """A `False` the solver never backed with a counterexample must not spend the weight.

        It becomes an unknown that keeps the execution reward.
        """
        verdict = await self.one_verdict(UNWITNESSED)

        self.assertEqual(
            [
                verdict["verdict"],
                verdict["state"],
                verdict["unwitnessed_refutation"],
                verdict["witness_gate"],
                verdict["counterexample_shape"],
            ],
            ["inconclusive", "UNWITNESSED", True, "no_counterexample", "none"],
        )
        # Not a penalty, and not a crash: none of the analyzer flags are set.
        self.assertEqual(
            [verdict.get("stock_analyzer_error"), verdict.get("stock_unknown"), verdict.get("dependency_error")],
            [None, None, None],
        )

    async def test_a_witnessed_refutation_still_spends_the_penalty(self):
        verdict = await self.one_verdict('{"result": false, "counterexample": true, "counterexample_rows": 0}')

        self.assertEqual(verdict["verdict"], "refuted")
        self.assertEqual(verdict["state"], "NEQ")
        self.assertEqual(verdict["counterexample_rows"], 0)
        self.assertEqual(verdict["refutation_route"], "solver_model")

    async def test_the_projection_arity_route_is_a_refutation_without_a_model(self):
        """Different output arity is a *proof* of non-equivalence.

        Two relations of different arity cannot be equal under any database, so
        no model is needed and none can exist. Gating it would score a provably
        non-equivalent prediction 1.0 instead of 1 - weight -- and would do so in
        exactly the case the check exists for: both sides empty on the graded
        database, result-matching there, still not equivalent.
        """
        verdict = await self.one_verdict(
            '{"result": false, "refutation_route": "column_arity", '
            '"counterexample": false, "witness_gate": "column_mismatch", '
            '"error": "column mismatch (result=-1)"}'
        )

        self.assertEqual(verdict["verdict"], "refuted")
        self.assertEqual(verdict["state"], "NEQ")
        # The route stays recorded, so its share of refutations is measurable.
        self.assertEqual(verdict["witness_gate"], "column_mismatch")
        self.assertEqual(verdict["refutation_route"], "column_arity")

    async def test_malformed_output_fails_closed(self):
        for output in ('{"result": "NEQ"}', '{"error": "missing result"}', "not json"):
            with self.subTest(output=output), self.assertRaises(verifier.VeriEQLError):
                await self.one_verdict(output)

    async def test_adapter_failure_exit_code_fails_closed(self):
        # The helper reached its own outer handler, so the adapter's contract is
        # what broke. That stays fatal.
        with self.assertRaisesRegex(verifier.VeriEQLError, "adapter failure"):
            await self.check(
                (verifier.HELPER_ADAPTER_FAILURE_EXIT_CODE, "", "RuntimeError: unrecognised exception origin")
            )

    async def test_the_native_verifier_refuses_an_aborted_check(self):
        """The native verifier never turns a dead solver into a verdict: an abort ends the run."""
        for returncode in (1, 134, -6):
            with (
                self.subTest(returncode=returncode),
                self.assertRaisesRegex(verifier.VeriEQLError, f"Native verifier aborted: stock VeriEQL exited "
                                       f"{returncode}: Assertion violation in z3"),
            ):
                await self.check((returncode, "", "Assertion violation in z3"))
        self.assertEqual(self.verifier.counts["checked"], 0)
        failures = [json.loads(p.read_text()) for p in self.receipts.glob("failure-*.json")]
        self.assertEqual(len(failures), 3)
        self.assertTrue(all(f["failed_attempts"][0]["stderr"] == "Assertion violation in z3" for f in failures))

    async def test_timeout_is_not_reported_as_a_verifier_abort(self):
        result = await self.check(None)

        self.assertEqual([result["verdict"], result["state"]], ["timeout", "TMO"])
        self.assertNotIn("verifier_abort", result)
        self.assertNotIn("verifier_exit_code", result)

    async def test_a_verifier_that_never_attests_fails_closed(self):
        """Many unwitnessed solver refutations and no attested one is evidence about the mechanism, not the queries.

        The attestation is all-or-nothing: the attribute is either read correctly or it is not. A verifier whose
        solver-route refutations never attest has stopped working -- which would disable the weight silently while
        every metric still looked healthy. A few early ones are not evidence.
        """
        for _ in range(verifier.MIN_SOLVER_REFUTATIONS_FOR_ATTESTATION_AUDIT - 1):
            self.assertEqual((await self.one_verdict(UNWITNESSED))["state"], "UNWITNESSED")
        with self.assertRaisesRegex(verifier.VeriEQLError, "attested a counterexample"):
            await self.one_verdict(UNWITNESSED)
        self.assertEqual(len(list(self.receipts.glob("failure-*.json"))), 1)

    async def test_arity_refutations_never_trip_the_attestation_audit(self):
        """They do not claim the solver route, so they are not evidence about it."""
        for _ in range(verifier.MIN_SOLVER_REFUTATIONS_FOR_ATTESTATION_AUDIT * 2):
            verdict = await self.one_verdict(
                '{"result": false, "refutation_route": "column_arity", '
                '"counterexample": false, "witness_gate": "column_mismatch"}'
            )
            self.assertEqual(verdict["state"], "NEQ")

    async def test_one_attested_refutation_clears_the_audit(self):
        """The mechanism works, so the rest are real unknowns."""
        attested = await self.one_verdict(
            '{"result": false, "refutation_route": "solver_model", '
            '"counterexample": true, "counterexample_shape": "str"}'
        )
        self.assertEqual(attested["state"], "NEQ")
        for _ in range(verifier.MIN_SOLVER_REFUTATIONS_FOR_ATTESTATION_AUDIT):
            self.assertEqual((await self.one_verdict(UNWITNESSED))["state"], "UNWITNESSED")


class NonzeroExitTest(unittest.TestCase):
    def failure(self, returncode, stderr="Assertion violation in z3"):
        return subprocess.CompletedProcess(["python"], returncode, "", stderr)

    def test_an_abort_is_undecided_and_keeps_its_exit_code(self):
        # Negative codes included: a native SIGABRT reaches the parent as one.
        for returncode in (1, 134, -6):
            with self.subTest(returncode=returncode):
                result = verifier._nonzero_exit_outcome(self.failure(returncode))

                self.assertEqual(result["verdict"], "inconclusive")
                self.assertEqual(result["state"], "ABORTED")
                self.assertTrue(result["verifier_abort"])
                self.assertEqual(result["verifier_exit_code"], returncode)
                self.assertIn(f"exited {returncode}", result["detail"])
                self.assertIn("Assertion violation in z3", result["detail"])

    def test_the_reserved_code_and_never_ran_codes_raise(self):
        with self.assertRaisesRegex(verifier.VeriEQLError, "adapter failure: RuntimeError: x"):
            verifier._nonzero_exit_outcome(self.failure(verifier.HELPER_ADAPTER_FAILURE_EXIT_CODE, "RuntimeError: x"))
        for returncode in verifier.INVOCATION_FAILURE_EXIT_CODES:
            with (
                self.subTest(returncode=returncode),
                self.assertRaisesRegex(verifier.VeriEQLError, f"did not start \\(exit {returncode}\\)"),
            ):
                verifier._nonzero_exit_outcome(self.failure(returncode, "not found"))


class InputRowTest(unittest.TestCase):
    def setUp(self):
        verifier.extract_schema_from_sqlite.cache_clear()
        self.addCleanup(verifier.extract_schema_from_sqlite.cache_clear)

    def test_only_sqlite_databases_are_supported(self):
        with self.assertRaisesRegex(verifier.VeriEQLError, "only supports SQLite tasks: x"):
            verifier._input_row({"id": "x", "database_path": "db.duckdb", "gold_sql": "SELECT 1"}, "SELECT 1", bound=2)

    def test_a_missing_or_empty_database_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing.sqlite")
            with self.assertRaisesRegex(verifier.VeriEQLError, "missing SQLite database"):
                verifier._input_row({"id": "x", "database_path": missing, "gold_sql": "SELECT 1"}, "", bound=2)
            empty = Path(directory) / "empty.sqlite"
            empty.touch()
            with self.assertRaisesRegex(verifier.VeriEQLError, "no tables"):
                verifier._input_row({"id": "x", "database_path": str(empty), "gold_sql": "SELECT 1"}, "", bound=2)

    def test_the_row_carries_the_stock_roots_until_the_verifier_replaces_them(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "db.sqlite"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE t (v REAL, flag BOOLEAN, name VARCHAR(20), raw BLOB, other)")
            row = verifier._input_row(
                {"id": "x", "database_path": str(database), "gold_sql": " SELECT v FROM t\n"}, "\tSELECT 1 ", bound=2
            )
        self.assertEqual(
            row,
            {
                "schema": {"t": {"v": "INT", "flag": "INT", "name": "VARCHAR", "raw": "VARCHAR", "other": "VARCHAR"}},
                "pred_sql": "SELECT 1",
                "gold_sql": "SELECT v FROM t",
                "bound_size": 2,
                "source_root": verifier.STOCK_SOURCE_ROOT,
                "z3_root": verifier.STOCK_Z3_ROOT,
                "stdlib_root": verifier.STOCK_STDLIB_ROOT,
            },
        )


class StartupRetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.verifier = verifier.NativeVerifier(manifest_stub(), 'identity', self.root)
        self.task = {'id': 'task', 'gold_sql': 'SELECT 1'}
        self.input_patch = patch.object(verifier, '_input_row', return_value={})
        self.input_patch.start()
        self.addCleanup(self.input_patch.stop)

    def failure(self, code=97, detail=None):
        stderr = detail or ('missing expected call export("mo_parsing.core", SkipTo)\n'
                            'Traceback (most recent call last):\n'
                            '    from environment import Environment\n'
                            '  File "/site/mo_imports/__init__.py", line 168, in export\n'
                            'Exception: mo_parsing.core is not expecting an export to SkipTo')
        return subprocess.CompletedProcess(['python'], code, '', stderr)

    async def test_retry_uses_fresh_attempt_and_preserves_actual_verdict(self):
        for result, expected in [(True, 'confirmed'), (False, 'refuted')]:
            with self.subTest(result=result):
                success = subprocess.CompletedProcess(
                    ['python'], 0, json.dumps({'result': result, 'refutation_route': 'column_arity'}), '')
                attempt = AsyncMock(side_effect=[self.failure(), success])
                with patch.object(self.verifier, '_attempt', attempt):
                    outcome = await self.verifier.check(self.task, 'SELECT 2')
                self.assertEqual(outcome['verdict'], expected)
                self.assertEqual(outcome['startup_retries'], 1)
                self.assertEqual(attempt.await_count, 2)
                self.assertEqual(attempt.call_args_list[0].args[0], attempt.call_args_list[1].args[0])
                self.assertLessEqual(attempt.call_args_list[1].args[1], attempt.call_args_list[0].args[1])
        receipts = [json.loads(p.read_text()) for p in self.root.glob('*.json') if not p.name.startswith('inventory-')]
        self.assertTrue(all('SkipTo' in r['failed_attempts'][0]['stderr'] for r in receipts))

    async def test_repeated_failure_and_retry_timeout_remain_fatal(self):
        for second in [self.failure(), None]:
            with self.subTest(second=second):
                attempt = AsyncMock(side_effect=[self.failure(), second])
                with patch.object(self.verifier, '_attempt', attempt), self.assertRaises(verifier.VeriEQLError):
                    await self.verifier.check(self.task, 'SELECT 2')
                self.assertEqual(attempt.await_count, 2)
                self.assertEqual(self.verifier.counts['checked'], 0)
                self.assertEqual(self.verifier.slots._value, 8)
        self.assertEqual(len(list(self.root.glob('failure-*.json'))), 2)

    async def test_other_adapter_and_attestation_errors_never_retry(self):
        failures = [self.failure(detail='RuntimeError: setup failed'),
                    self.failure(code=1, detail='RuntimeError: Unpinned or changed loaded runtime file'),
                    self.failure(detail=self.failure().stderr.replace('/site/', '/untrusted/'))]
        for failure in failures:
            with self.subTest(failure=failure):
                attempt = AsyncMock(return_value=failure)
                with patch.object(self.verifier, '_attempt', attempt), self.assertRaises(verifier.VeriEQLError):
                    await self.verifier.check(self.task, 'SELECT 2')
                self.assertEqual(attempt.await_count, 1)

    async def test_normal_timeout_keeps_existing_semantics(self):
        with patch.object(self.verifier, '_attempt', AsyncMock(return_value=None)) as attempt:
            outcome = await self.verifier.check(self.task, 'SELECT 2')
        self.assertEqual(outcome['verdict'], 'timeout')
        self.assertEqual(outcome['startup_retries'], 0)
        self.assertEqual(attempt.await_count, 1)

    async def test_expired_retry_budget_is_fatal_without_another_process(self):
        self.verifier.timeout_seconds = 0.001

        async def slow_failure(row, timeout_seconds):
            await asyncio.sleep(timeout_seconds + 0.01)
            return self.failure()
        with (
            patch.object(self.verifier, '_attempt', AsyncMock(side_effect=slow_failure)) as attempt,
            self.assertRaisesRegex(verifier.VeriEQLError, 'exhausted its deadline'),
        ):
            await self.verifier.check(self.task, 'SELECT 2')
        self.assertEqual(attempt.await_count, 1)
        self.assertEqual(self.verifier.counts['checked'], 0)

    async def test_timed_out_subprocess_is_reaped(self):
        self.verifier.manifest['python'] = sys.executable
        self.verifier.manifest['source_root'] = str(self.root)
        with patch.object(verifier, 'checked_helper', return_value='import time; time.sleep(30)'):
            result = await self.verifier._attempt({}, 0.05)
        self.assertIsNone(result)
        self.assertEqual(self.verifier.processes, set())

    async def test_close_kills_running_checks(self):
        self.verifier.manifest['python'] = sys.executable
        self.verifier.manifest['source_root'] = str(self.root)
        with patch.object(verifier, 'checked_helper', return_value='import time; time.sleep(30)'):
            running = asyncio.create_task(self.verifier._attempt({}, 30))
            for _ in range(500):
                if self.verifier.processes:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(len(self.verifier.processes), 1)
            await self.verifier.close()
            completed = await asyncio.wait_for(running, 5)
        self.assertEqual(completed.returncode, -signal.SIGKILL)
        self.assertEqual(self.verifier.processes, set())

    async def test_an_expired_queue_deadline_is_an_error_not_a_verdict(self):
        self.verifier.queue_timeout_seconds = 0.01
        for _ in range(8):
            await self.verifier.slots.acquire()
        with patch.object(self.verifier, '_attempt', AsyncMock()) as attempt, self.assertRaises(asyncio.TimeoutError):
            await self.verifier.check(self.task, 'SELECT 2')
        attempt.assert_not_awaited()


class InventoryTests(unittest.TestCase):
    def fixture(self, root):
        """A Linux arm64 installation as `setup verieql` describes it, and what its probe reports."""
        manifest = {"kind": "verieql-native-linux-aarch64-v1", "platform": "linux-aarch64",
                    "source": verifier.VERIEQL_SOURCE, "z3_version": verifier.Z3_VERSION,
                    "verieql_revision": verifier.VERIEQL_REVISION, "python_version": "3.11.14"}
        for key in ("source_root", "stdlib_root", "site_packages_root", "z3_root"):
            folder = root / key
            folder.mkdir()
            (folder / "module.py").write_text("pass")
            manifest[key] = str(folder.resolve())
        executable = root / "python"
        executable.write_text("interpreter")
        manifest["python"] = str(executable.resolve())
        manifest["files"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                             for name in verifier.runtime_inventory(manifest)}
        observed = {"arch": "aarch64", "version": verifier.Z3_VERSION, "source": manifest["source_root"],
                    "stdlib": manifest["stdlib_root"], "site": manifest["site_packages_root"],
                    "z3": manifest["z3_root"], "files": list(manifest["files"]), "system": "Linux",
                    "python_version": "3.11.14"}
        return manifest, observed

    def validate(self, root, manifest, observed, expected_sha256=None):
        """`validate_manifest` on a Linux machine, with the fixture's source tree standing in for the pinned one."""
        path = root / "manifest.json"
        raw = json.dumps(manifest).encode()
        path.write_bytes(raw)
        tree = verifier.source_tree_sha256(verifier.source_tree_entries(manifest))
        with patch.object(verifier.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(observed))), \
                patch.object(verifier.platform, "system", return_value="Linux"), \
                patch.object(verifier, "VERIEQL_SOURCE_SHA256", tree):
            return verifier.validate_manifest(str(path), expected_sha256 or hashlib.sha256(raw).hexdigest())

    def test_empty_or_missing_executable_inventory_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, observed = self.fixture(root)
            del manifest["files"][manifest["python"]]
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                self.validate(root, manifest, observed)
            manifest["files"] = {}
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                self.validate(root, manifest, observed)

    def test_new_source_file_and_changed_dependency_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, observed = self.fixture(root)
            self.validate(root, manifest, observed)
            target = Path(manifest["site_packages_root"]) / "module.py"
            target.write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                self.validate(root, manifest, observed)
            (Path(manifest["source_root"]) / "extra.py").write_text("pass")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                self.validate(root, manifest, observed)

    def test_actual_loaded_library_and_root_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, observed = self.fixture(root)
            observed["files"].append("/unexpected/libz3.so")
            with self.assertRaisesRegex(ValueError, "libraries missing"):
                self.validate(root, manifest, observed)
            observed["source"] = "/wrong/source"
            with self.assertRaisesRegex(ValueError, "root differs"):
                self.validate(root, manifest, observed)

    def test_the_manifest_contract_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, observed = self.fixture(root)
            self.assertEqual(self.validate(root, manifest, observed), manifest)
            with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                self.validate(root, manifest, observed, expected_sha256="0" * 64)
            for key, value, message in (
                ("kind", "verieql-native-amd64-v1", "Unknown native verifier contract"),
                ("source", "https://github.com/VeriEQL/VeriEQL/tree/main", "pinned VeriEQL source"),
            ):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                    self.validate(root, {**manifest, key: value}, observed)
            for key, value in (("arch", "x86_64"), ("version", "4.12.2")):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "architecture or solver version"):
                    self.validate(root, manifest, {**observed, key: value})

    @unittest.skipUnless(Path('/proc/self/maps').exists(), 'Linux runtime attestation')
    def test_subprocess_withholds_verdict_for_unpinned_lazy_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe = verifier._RUNTIME_PROBE + '''
import contextlib, io, hashlib
print(json.dumps({name: hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
                  for name in loaded_files()}))
'''
            inventory = subprocess.check_output([sys.executable, '-I', '-B', '-c', probe])
            path = root / 'inventory.json'
            path.write_bytes(inventory)
            digest = hashlib.sha256(inventory).hexdigest()
            with patch.object(verifier, '_HELPER_SCRIPT', 'print("verified")'):
                valid = subprocess.run([sys.executable, '-I', '-B', '-c',
                                        verifier.checked_helper(digest), str(path)],
                                       capture_output=True, text=True, check=False)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertEqual(valid.stdout.strip(), 'verified')
            (root / 'late_dependency.py').write_text('value = 1')
            helper = f'import sys; sys.path.insert(0, {str(root)!r}); import late_dependency; print("verified")'
            with patch.object(verifier, '_HELPER_SCRIPT', helper):
                invalid = subprocess.run([sys.executable, '-I', '-B', '-c',
                                          verifier.checked_helper(digest), str(path)],
                                         capture_output=True, text=True, check=False)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertEqual(invalid.stdout, '')
            self.assertIn('Unpinned or changed', invalid.stderr)


def gate(gold, prediction, *, ddl='CREATE TABLE t(v INTEGER)', rows=None, referenced=('t',), policies=(),
         method='set', tables=None):
    """`refutation_gate` as `confirm` calls it, with the referenced tables given instead of parsed."""
    case = {'referenced_tables': list(referenced), 'grading_method': method,
            'variants': {'original': {'gold_sql': gold, 'pred_sql': prediction}}}
    sql = SimpleNamespace(
        ReadOnlySQLite=sqlite_exec.ReadOnlySQLite,
        compare_results=functools.partial(sqlite_exec.compare_results,
                                          numeric_policy=sqlite_exec.DIAMOND_NUMERIC_POLICY),
    )
    if tables is None:
        tables = [{'name': 't', 'columns': ['v'], 'rows': [[1], [2]] if rows is None else rows}]
    return verifier.refutation_gate(case, tables, source_schema(ddl),
                                    {(p['table'], p['id']): p for p in policies}, sql, dedupe_hypothesis=True)


FK_DDL = 'CREATE TABLE parent(id INTEGER PRIMARY KEY); CREATE TABLE t(v INTEGER REFERENCES parent(id));'
FK_QUERY = 'SELECT COUNT(*) FROM t WHERE v IS NOT NULL AND NOT EXISTS (SELECT 1 FROM parent WHERE parent.id=t.v)'


class RefutationGateTest(unittest.TestCase):
    """The executed-counterexample gate on a materialized witness (no SQL parsing needed)."""

    def test_real_difference_penalized(self):
        result = gate('SELECT SUM(v) FROM t', 'SELECT COUNT(*) FROM t')
        self.assertEqual([result['decision'], result['reason']], ['penalize', 'validated_counterexample'])
        self.assertEqual(result['results']['typed']['original']['gold'], ((3,),))
        self.assertEqual(result['results']['typed']['original']['pred'], ((2,),))
        self.assertEqual(result['fidelity'], [{'table': 't', 'ok': True, 'kind': 'exact', 'conversions': 0}])
        self.assertEqual(result['failed_checks'], [])

    def test_multiplicity_is_not_penalized(self):
        self.assertEqual(gate('SELECT DISTINCT v FROM t', 'SELECT v FROM t', rows=[[1], [1]])['reason'],
                         'grader_matches')

    def test_numeric_contract_used(self):
        self.assertEqual(gate('SELECT v FROM t', 'SELECT v*1.0 FROM t')['reason'], 'grader_matches')
        self.assertEqual(gate('SELECT v FROM t', "SELECT CAST(v AS TEXT) FROM t")['decision'], 'penalize')

    def test_list_grading_compares_rows_in_order(self):
        self.assertEqual(gate('SELECT v FROM t ORDER BY v', 'SELECT v FROM t ORDER BY v DESC',
                              method='list')['decision'], 'penalize')
        self.assertEqual(gate('SELECT v FROM t ORDER BY v', 'SELECT v FROM t ORDER BY v DESC')['reason'],
                         'grader_matches')

    def test_generated_rowid_rejected(self):
        self.assertEqual(gate('SELECT v FROM t', 'SELECT 5 FROM t', ddl='CREATE TABLE t(v INTEGER PRIMARY KEY)',
                              rows=[[None]])['reason'], 'unfaithful')

    def test_not_null_rejected(self):
        self.assertEqual(gate('SELECT v FROM t', 'SELECT 5 FROM t', ddl='CREATE TABLE t(v INTEGER NOT NULL)',
                              rows=[[None]])['reason'], 'invalid_constraint')

    def test_affinity_changes_query_rejected(self):
        result = gate('SELECT v/2 FROM t', 'SELECT 0 FROM t', ddl='CREATE TABLE t(v REAL)', rows=[[3]])
        self.assertEqual(result['reason'], 'twin_changes_results_or_verdict')
        self.assertEqual(result['fidelity'][0]['kind'], 'lossless_int_to_real')

    def test_unsupported_numeric_values_abstain_before_binding(self):
        for value in [2**63, -(2**63) - 1, float('inf'), float('-inf'), float('nan')]:
            with self.subTest(value=value):
                result = gate('SELECT v FROM t', 'SELECT COUNT(*) FROM t', rows=[[value]])
                self.assertEqual([result['decision'], result['reason']], ['abstain', 'unsupported_numeric_value'])

    def test_integer_sum_overflow_abstains(self):
        result = gate('SELECT SUM(v) FROM t', 'SELECT COUNT(*) FROM t', rows=[[2**63 - 1], [1]])
        self.assertEqual([result['decision'], result['reason']], ['abstain', 'witness_query_error'])
        self.assertEqual(result['materialization'], 'typed')

    def test_integer_boundaries_remain_supported(self):
        for value in [-(2**63), 2**63 - 1]:
            with self.subTest(value=value):
                self.assertEqual(gate('SELECT v FROM t', 'SELECT 0 FROM t', rows=[[value]])['decision'], 'penalize')

    def test_foreign_key_policies(self):
        def result(status):
            policies = [] if status is None else [{'table': 't', 'id': 0, 'status': status, 'query': FK_QUERY}]
            return gate('SELECT SUM(v) FROM t', 'SELECT COUNT(*) FROM t', ddl=FK_DDL, rows=[[2]], policies=policies)

        self.assertEqual(result('enforced')['reason'], 'invalid_foreign_key')
        self.assertEqual(result('enforced')['constraint_checks'],
                         [{'table': 't', 'id': 0, 'status': 'enforced', 'violations': 1}])
        for status in ['advisory_source_violated', 'advisory_malformed']:
            self.assertEqual(result(status)['decision'], 'penalize')
        self.assertEqual(result(None)['reason'], 'unresolved_policy')

    def test_referenced_tables_pull_in_their_foreign_key_parents(self):
        result = gate('SELECT SUM(v) FROM t', 'SELECT COUNT(*) FROM t', ddl=FK_DDL, rows=[[2]],
                      policies=[{'table': 't', 'id': 0, 'status': 'advisory_source_violated', 'query': FK_QUERY}])
        self.assertEqual([f['table'] for f in result['fidelity']], ['parent', 't'])

    def test_unique_deduplication_revalidates_new_witness(self):
        result = gate('SELECT SUM(v) FROM t', 'SELECT COUNT(*) FROM t', ddl='CREATE TABLE t(v INTEGER UNIQUE)',
                      rows=[[2], [2]])
        self.assertEqual(result['dedupe_removed'], 1)
        self.assertEqual(result['decision'], 'penalize')

    def test_unreferenced_witness_tables_are_ignored_and_duplicates_refused(self):
        other = {'name': 'other', 'columns': ['v'], 'rows': [['x']]}
        witness = {'name': 'T', 'columns': ['v'], 'rows': [[1], [2]]}
        result = gate('SELECT SUM(v) FROM t', 'SELECT COUNT(*) FROM t', tables=[other, witness],
                      ddl='CREATE TABLE t(v INTEGER); CREATE TABLE other(v TEXT)')
        self.assertEqual(result['decision'], 'penalize')
        with self.assertRaisesRegex(ValueError, 'duplicate witness table'):
            gate('SELECT v FROM t', 'SELECT 1 FROM t', tables=[witness, witness])

    def test_storage_failure_propagates(self):
        with (
            patch('text_to_sql_recipe.verifier.tempfile.TemporaryDirectory', side_effect=OSError('disk full')),
            self.assertRaises(OSError),
        ):
            gate('SELECT SUM(v) FROM t', 'SELECT COUNT(*) FROM t')

    def test_execution_limits_abstain(self):
        with patch.object(sqlite_exec.ReadOnlySQLite, 'execute', side_effect=sqlite_exec.SQLiteQueryTimeout('slow')):
            result = gate('SELECT SUM(v) FROM t', 'SELECT COUNT(*) FROM t')
        self.assertEqual([result['decision'], result['reason']], ['abstain', 'witness_execution_limit'])
        self.assertEqual(result['error'], 'SQLiteQueryTimeout: slow')


class ConfirmEarlyReturnTest(unittest.TestCase):
    """`confirm` refuses witnesses it cannot use before it parses any SQL."""

    def test_decode_failure_abstains(self):
        r = verifier.confirm({}, '', {'result': False, 'attested': True, 'decode_issues': ['unknown encoding']}, {})
        self.assertEqual(r['decision'], 'abstain')

    def test_extraction_error_payload_propagates(self):
        with self.assertRaises(verifier.VeriEQLError):
            verifier.confirm({}, '', {'error': 'adapter failed'}, {})

    def test_unattested_or_tableless_witnesses_abstain(self):
        for witness, reason in (
            ({'result': True, 'attested': True, 'tables': []}, 'no_attested_witness'),
            ({'result': False, 'attested': False, 'tables': []}, 'no_attested_witness'),
            ({'result': False, 'attested': True}, 'missing_witness_tables'),
        ):
            with self.subTest(reason=reason):
                self.assertEqual(verifier.confirm({}, '', witness, {}), {'decision': 'abstain', 'reason': reason})


@needs_sqlglot
class ConfirmationTests(unittest.TestCase):
    def check(self, gold, prediction, *, ddl='CREATE TABLE t(v INTEGER)', rows=None,
              numeric_policy='diamond-numeric-v1'):
        schema = source_schema(ddl)
        task = {'gold_sql': gold, 'grading_method': 'set', 'numeric_policy': numeric_policy,
                'numeric_tolerance': False}
        witness = {'result': False, 'attested': True,
                   'tables': [{'name': 't', 'columns': ['v'], 'rows': [[1], [2]] if rows is None else rows}]}
        return verifier.confirm(task, prediction, witness, {'schema': schema, 'policies': []})

    def test_real_difference_penalized(self):
        self.assertEqual(self.check('SELECT SUM(v) FROM t', 'SELECT COUNT(*) FROM t')['decision'], 'penalize')

    def test_multiplicity_is_not_penalized(self):
        self.assertEqual(self.check('SELECT DISTINCT v FROM t', 'SELECT v FROM t', rows=[[1], [1]])['reason'],
                         'grader_matches')

    def test_sql_too_deeply_nested_to_parse_abstains(self):
        nested = 'SELECT ' + '(' * 200 + 'v' + ')' * 200 + ' FROM t'
        result = self.check('SELECT v FROM t', nested)
        self.assertEqual((result['decision'], result['reason']), ('abstain', 'unresolved_query'))

    def test_numeric_contract_used(self):
        self.assertEqual(self.check('SELECT v FROM t', 'SELECT v*1.0 FROM t')['reason'], 'grader_matches')
        self.assertEqual(self.check('SELECT v FROM t', "SELECT CAST(v AS TEXT) FROM t")['decision'], 'penalize')

    def test_generated_rowid_rejected(self):
        self.assertEqual(self.check('SELECT v FROM t', 'SELECT 5 FROM t', ddl='CREATE TABLE t(v INTEGER PRIMARY KEY)',
                                    rows=[[None]])['reason'], 'unfaithful')

    def test_not_null_rejected(self):
        self.assertEqual(self.check('SELECT v FROM t', 'SELECT 5 FROM t', ddl='CREATE TABLE t(v INTEGER NOT NULL)',
                                    rows=[[None]])['reason'], 'invalid_constraint')

    def test_affinity_changes_query_rejected(self):
        self.assertEqual(self.check('SELECT v/2 FROM t', 'SELECT 0 FROM t', ddl='CREATE TABLE t(v REAL)',
                                    rows=[[3]])['reason'], 'twin_changes_results_or_verdict')

    def test_unparseable_or_unknown_queries_abstain(self):
        self.assertEqual(self.check('SELECT v FROM t', 'SELECT FROM WHERE (')['reason'], 'unresolved_query')
        self.assertEqual(self.check('SELECT 1', 'SELECT 2')['reason'], 'no_resolved_source_tables')

    def test_a_task_without_a_numeric_policy_uses_the_recipe_policy(self):
        """Every recipe task names diamond-numeric-v1; a task without one is compared under it too.

        Within 1e-6 the values match; 1.001 against 1 differs (the removed one-percent single-value rule would have
        matched it).
        """
        schema = source_schema('CREATE TABLE t(v INTEGER)')
        witness = {'result': False, 'attested': True, 'tables': [{'name': 't', 'columns': ['v'], 'rows': [[1]]}]}
        for prediction, reason in (('SELECT v + 0.0000001 FROM t', 'grader_matches'),
                                   ('SELECT v * 1.001 FROM t', 'validated_counterexample')):
            with self.subTest(prediction=prediction):
                result = verifier.confirm({'gold_sql': 'SELECT v FROM t'}, prediction, witness,
                                          {'schema': schema, 'policies': []})
                self.assertEqual(result['reason'], reason)


@needs_sqlglot
class FailureAndConstraintTests(unittest.TestCase):
    check = ConfirmationTests.check

    def test_unsupported_numeric_values_abstain_before_binding(self):
        for value in [2**63, -(2**63) - 1, float('inf'), float('-inf'), float('nan')]:
            with self.subTest(value=value):
                result = self.check('SELECT v FROM t', 'SELECT COUNT(*) FROM t', rows=[[value]])
                self.assertEqual(result['decision'], 'abstain')
                self.assertEqual(result['reason'], 'unsupported_numeric_value')

    def test_integer_sum_overflow_abstains(self):
        result = self.check('SELECT SUM(v) FROM t', 'SELECT COUNT(*) FROM t', rows=[[2**63 - 1], [1]])
        self.assertEqual(result['decision'], 'abstain')
        self.assertEqual(result['reason'], 'witness_query_error')

    def test_integer_boundaries_remain_supported(self):
        for value in [-(2**63), 2**63 - 1]:
            with self.subTest(value=value):
                self.assertEqual(self.check('SELECT v FROM t', 'SELECT 0 FROM t', rows=[[value]])['decision'],
                                 'penalize')

    def test_comparator_adapter_failure_propagates(self):
        with (
            patch('text_to_sql_recipe.sqlite_exec.compare_results',
                  side_effect=sqlite_exec.SQLiteQueryError('adapter failure')),
            self.assertRaises(sqlite_exec.SQLiteQueryError),
        ):
            self.check('SELECT SUM(v) FROM t', 'SELECT COUNT(*) FROM t')

    def test_storage_failure_propagates(self):
        with (
            patch('text_to_sql_recipe.verifier.tempfile.TemporaryDirectory', side_effect=OSError('disk full')),
            self.assertRaises(OSError),
        ):
            self.check('SELECT SUM(v) FROM t', 'SELECT COUNT(*) FROM t')

    def test_sqlite_infrastructure_failure_propagates(self):
        with (
            patch('text_to_sql_recipe.sqlite_exec.ReadOnlySQLite.execute',
                  side_effect=sqlite_exec.SQLiteInfrastructureError('storage unavailable')),
            self.assertRaises(sqlite_exec.SQLiteInfrastructureError),
        ):
            self.check('SELECT SUM(v) FROM t', 'SELECT COUNT(*) FROM t')

    def foreign_key_result(self, status):
        schema = source_schema(FK_DDL)
        policies = [] if status is None else [{'table': 't', 'id': 0, 'status': status, 'query': FK_QUERY}]
        return verifier.confirm(
            {'gold_sql': 'SELECT SUM(v) FROM t', 'grading_method': 'set'}, 'SELECT COUNT(*) FROM t',
            {'result': False, 'attested': True, 'tables': [{'name': 't', 'columns': ['v'], 'rows': [[2]]}]},
            {'schema': schema, 'policies': policies})

    def test_enforced_fk_abstains_on_orphan(self):
        self.assertEqual(self.foreign_key_result('enforced')['reason'], 'invalid_foreign_key')

    def test_advisory_fk_does_not_reject_source_permitted_orphan(self):
        for status in ['advisory_source_violated', 'advisory_malformed']:
            self.assertEqual(self.foreign_key_result(status)['decision'], 'penalize')

    def test_missing_fk_policy_abstains(self):
        self.assertEqual(self.foreign_key_result(None)['reason'], 'unresolved_policy')

    def test_unique_deduplication_revalidates_new_witness(self):
        result = self.check('SELECT SUM(v) FROM t', 'SELECT COUNT(*) FROM t', ddl='CREATE TABLE t(v INTEGER UNIQUE)',
                            rows=[[2], [2]])
        self.assertEqual(result['dedupe_removed'], 1)
        self.assertEqual(result['decision'], 'penalize')


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    """The gated verifier around a stubbed equivalence check, extraction process and confirmation."""

    async def invoke(self, gate=None, completed=None, original=None, inspect=None):
        obj = object.__new__(verifier.GatedNativeVerifier)
        obj.slots = asyncio.Semaphore(1)
        obj.manifest = {'source_root': '/fixture'}
        obj.timeout_seconds = 1
        obj.extraction_script = 'fixture'
        obj.gate_policy = {'/db': {'policy': 'fixture'}}
        obj.counts = collections.Counter()
        obj._attempt = AsyncMock(return_value=completed)
        original = original or {'verdict': 'refuted', 'state': 'REFUTED'}
        with tempfile.TemporaryDirectory() as directory:
            obj.receipt_dir = Path(directory)
            with patch.object(verifier.NativeVerifier, 'check', AsyncMock(return_value=original)), \
                    patch.object(verifier, '_input_row', return_value={}), \
                    patch.object(verifier, 'confirm', return_value=gate) as confirm:
                result = await obj.check({'id': 'test', 'database_path': '/db', 'gold_sql': 'SELECT 1'}, 'SELECT 2')
            if inspect is not None:
                inspect(obj, confirm, [json.loads(p.read_text()) for p in Path(directory).glob('gate-*.json')])
            return result

    async def test_timeout_abstains(self):
        r = await self.invoke()
        self.assertEqual(r['verdict'], 'unknown')
        self.assertEqual(r['counterexample_gate']['reason'], 'witness_timeout')

    async def test_matching_witness_drops_refutation(self):
        r = await self.invoke({'decision': 'abstain', 'reason': 'grader_matches'},
                              SimpleNamespace(returncode=0, stdout='{}', stderr=''))
        self.assertEqual(r['verdict'], 'unknown')
        self.assertEqual(r['ungated_verdict'], 'refuted')

    async def test_validated_witness_retains_refutation(self):
        r = await self.invoke({'decision': 'penalize', 'reason': 'validated_counterexample'},
                              SimpleNamespace(returncode=0, stdout='{}', stderr=''))
        self.assertEqual(r['verdict'], 'refuted')

    async def test_runtime_failure_is_not_reward(self):
        with self.assertRaises(verifier.VeriEQLError):
            await self.invoke(completed=SimpleNamespace(returncode=1, stdout='', stderr='fixture runtime failure'))

    async def test_only_refutations_are_examined(self):
        for verdict in ('confirmed', 'inconclusive', 'timeout'):
            with self.subTest(verdict=verdict):
                original = {'verdict': verdict, 'state': 'X'}

                def inspect(obj, confirm, receipts):
                    obj._attempt.assert_not_awaited()
                    confirm.assert_not_called()
                    self.assertEqual(receipts, [])

                self.assertEqual(await self.invoke(original=original, inspect=inspect), original)

    async def test_the_gate_records_its_decision(self):
        witness = {'result': False, 'attested': True, 'tables': []}
        gate = {'decision': 'abstain', 'reason': 'grader_matches'}

        def inspect(obj, confirm, receipts):
            obj._attempt.assert_awaited_once_with({'source_root': '/fixture'}, 1, 'fixture')
            confirm.assert_called_once_with({'id': 'test', 'database_path': '/db', 'gold_sql': 'SELECT 1'},
                                            'SELECT 2', witness, {'policy': 'fixture'})
            self.assertEqual(obj.counts, collections.Counter({'gate_abstain': 1}))
            self.assertEqual(len(receipts), 1)
            self.assertEqual([receipts[0]['witness'], receipts[0]['prediction']], [witness, 'SELECT 2'])

        result = await self.invoke(gate, SimpleNamespace(returncode=0, stdout='noise\n' + json.dumps(witness) + '\n',
                                                         stderr=''), inspect=inspect)
        self.assertEqual(
            {key: result[key] for key in ('verdict', 'state', 'refutation_route', 'ungated_verdict',
                                          'counterexample_gate')},
            {'verdict': 'unknown', 'state': 'UNKNOWN', 'refutation_route': 'unvalidated_counterexample',
             'ungated_verdict': 'refuted', 'counterexample_gate': gate},
        )
        self.assertGreaterEqual(result['gate_total_seconds'], 0)


class FakeInt:
    def __init__(self, value):
        self.value = value

    def as_long(self):
        return self.value

    def __str__(self):
        return str(self.value)


class ExtractionScriptTest(unittest.TestCase):
    """The witness extraction script, run in this interpreter against fake `environment` and `z3` modules."""

    def run_script(self, payload, environment, z3):
        stdout = io.StringIO()
        with patch.dict(sys.modules, {'environment': environment, 'z3': z3}), \
                patch('sys.stdin', io.StringIO(json.dumps(payload))), patch('sys.stdout', stdout), \
                patch.object(sys, 'path', list(sys.path)):
            exec(compile(verifier._EXTRACTION_SCRIPT, 'extraction', 'exec'), {'__name__': '__main__'})  # noqa: S102
        return json.loads(stdout.getvalue())

    def test_extractor_setup_and_model_failures_propagate(self):
        payload = {'source_root': '/fixture', 'schema': {}, 'gold_sql': 'SELECT 1', 'pred_sql': 'SELECT 2',
                   'bound_size': 2}
        for stage in ['setup', 'model']:
            environment = types.ModuleType('environment')
            factory = MagicMock()
            environment.Environment = factory
            if stage == 'setup':
                factory.side_effect = OSError('fixture storage failure')
            else:
                env = factory.return_value.__enter__.return_value
                env.analyze.return_value = False
                env.counterexample = object()
                env.solver.model.side_effect = RuntimeError('fixture adapter failure')
            z3 = types.ModuleType('z3')
            z3.is_true = z3.is_int_value = z3.is_rational_value = lambda value: False
            with self.subTest(stage=stage), self.assertRaises((OSError, RuntimeError)):
                self.run_script(payload, environment, z3)

    def test_the_witness_is_read_from_the_solver_model(self):
        """Deleted tuples are dropped; NULLs stay; strings decode to the queries' own literal casing."""
        table = SimpleNamespace(
            name='T',
            attributes=[SimpleNamespace(name=name, NULL=functools.partial(lambda n, s: ('null', n, s), name),
                                        VALUE=functools.partial(lambda n, s: ('value', n, s), name))
                        for name in ('id', 'name')],
            tuples=[SimpleNamespace(SORT=sort) for sort in range(4)],
        )
        values = {('deleted', 0): False, ('deleted', 1): True, ('deleted', 2): False, ('deleted', 3): False}
        for sort, (ident, name) in enumerate([(1, 7), (9, 9), (2, None), (3, 8)]):
            values[('null', 'id', sort)] = False
            values[('value', 'id', sort)] = FakeInt(ident)
            values[('null', 'name', sort)] = name is None
            values[('value', 'name', sort)] = FakeInt(name or 0)
        model = MagicMock()
        model.eval.side_effect = lambda key, model_completion: values[key]
        model.__str__.return_value = 'model'

        class Environment:
            def __init__(self, **kwargs):
                self.counterexample = '-- found'
                self.solver = SimpleNamespace(model=lambda: model)
                self.base_databases = {'T': table}
                self.variables = {7: 'String_ABC'}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def create_database(self, **kwargs):
                pass

            def save_checkpoints(self):
                pass

            def analyze(self, pred, gold):
                return False

            def DELETED_FUNCTION(self, sort):  # the stock attribute's name
                return ('deleted', sort)

        environment = types.ModuleType('environment')
        environment.Environment = Environment
        z3 = types.ModuleType('z3')
        z3.is_true = lambda value: value is True
        z3.is_int_value = lambda value: isinstance(value, FakeInt)
        z3.is_rational_value = lambda value: False
        out = self.run_script({'source_root': '/fixture', 'schema': {'t': {'id': 'INT', 'name': 'VARCHAR'}},
                               'gold_sql': "SELECT id FROM t WHERE name = 'abc'", 'pred_sql': 'SELECT id FROM t',
                               'bound_size': 2}, environment, z3)

        self.assertEqual([out['result'], out['attested'], out['decode_issues']], [False, True, []])
        self.assertEqual(out['tables'], [{'name': 'T', 'columns': ['id', 'name'],
                                          'rows': [[1, 'abc'], [2, None], [3, '__witness_8']],
                                          'raw_rows': [['1', '7'], ['2', 'NULL'], ['3', '8']]}])
        self.assertEqual([out['model'], out['variables']], ['model', {'7': 'String_ABC'}])


class LoadPolicyTest(unittest.TestCase):
    def test_the_policy_must_cover_each_training_database_at_its_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'a.sqlite'
            with closing(sqlite3.connect(database)) as connection:
                connection.execute('CREATE TABLE t (v INTEGER)')
            info = {'sha256': hashlib.sha256(database.read_bytes()).hexdigest(), 'schema': [], 'policies': []}
            raw = json.dumps({str(database): info}).encode()
            path = Path(directory) / 'gate-policy.json'
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            tasks = [{'database_path': str(database)}, {'database_path': database}]

            self.assertEqual(verifier.load_policy(path, digest, tasks), {str(database): info})
            with self.assertRaisesRegex(ValueError, 'policy hash mismatch'):
                verifier.load_policy(path, '0' * 64, tasks)
            with self.assertRaisesRegex(ValueError, 'no entry for the database'):
                verifier.load_policy(path, digest, [{'database_path': str(Path(directory) / 'b.sqlite')}])
            database.write_bytes(database.read_bytes() + b'\0')
            with self.assertRaisesRegex(ValueError, 'policy database mismatch'):
                verifier.load_policy(path, digest, tasks)

    def test_tasks_find_their_entry_by_database_digest_wherever_it_is_stored(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'moved' / 'a.sqlite'
            database.parent.mkdir()
            with closing(sqlite3.connect(database)) as connection:
                connection.execute('CREATE TABLE t (v INTEGER)')
            sha = hashlib.sha256(database.read_bytes()).hexdigest()
            info = {'sha256': sha, 'schema': [], 'policies': []}
            # The policy was written where the databases used to live.
            raw = json.dumps({'/elsewhere/a/a.sqlite': info}).encode()
            path = Path(directory) / 'gate-policy.json'
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            tasks = [{'database_path': str(database), 'database_sha256': sha}]
            self.assertEqual(verifier.load_policy(path, digest, tasks), {str(database): info})
            with self.assertRaisesRegex(ValueError, 'no entry for the database'):   # a digest it does not cover
                verifier.load_policy(path, digest, [{'database_path': str(database), 'database_sha256': '0' * 64}])


class FactoryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = {**manifest_stub(), 'files': {'/runtime/python': '0' * 64}}
        self.environ = {verifier.MANIFEST_ENV_VAR: str(self.root / 'manifest.json'),
                        verifier.MANIFEST_SHA256_ENV_VAR: 'a' * 64}
        (self.root / 'manifest.json').write_text('{}')   # validate_manifest is patched; the file need only exist
        self.tasks = [{'id': 't1', 'database_path': '/data/db.sqlite'}]
        self.gate = {'tasks': self.tasks, 'gate_policy_path': 'gate-policy.json', 'gate_policy_sha256': 'b' * 64}
        self.sqlglot = patch.dict(sys.modules, {} if HAS_SQLGLOT else {'sqlglot': types.ModuleType('sqlglot')})

    def test_none_needs_nothing(self):
        reward = RewardConfig(verifier='none')
        with patch.object(verifier, 'validate_manifest') as validate, patch.object(verifier, 'load_policy') as load:
            self.assertIsNone(verifier.setup_verifier(reward, ExecutionConfig(), environ={}))
            self.assertIsNone(verifier.make_verifier(reward, ExecutionConfig(), self.root / 'receipts', environ={}))
        validate.assert_not_called()
        load.assert_not_called()
        self.assertFalse((self.root / 'receipts').exists())

    def test_verieql_without_a_runtime_refuses_to_train_unshaped(self):
        with self.assertRaisesRegex(verifier.VeriEQLError, verifier.MANIFEST_ENV_VAR):
            verifier.setup_verifier(RewardConfig(verifier_gate=False), ExecutionConfig(), environ={})
        with (
            patch.dict(os.environ, {verifier.MANIFEST_ENV_VAR: '', verifier.MANIFEST_SHA256_ENV_VAR: ''}),
            self.assertRaisesRegex(verifier.VeriEQLError, 'CC BY-NC-SA'),
        ):
            verifier.setup_verifier(RewardConfig(verifier_gate=False), ExecutionConfig())

    def test_settings_are_checked_before_anything_runs(self):
        for reward, message in (
            (RewardConfig(verifier='other'), 'unknown reward.verifier'),
            (RewardConfig(verifier_weight=0.0), 'verifier_weight'),
            (RewardConfig(verifier_weight=1.5), 'verifier_weight'),
        ):
            with self.subTest(reward=reward), self.assertRaisesRegex(ValueError, message):
                verifier.setup_verifier(reward, ExecutionConfig(), environ=self.environ, **self.gate)
        with self.assertRaisesRegex(ValueError, '64 lowercase hex'):
            verifier.setup_verifier(RewardConfig(verifier_gate=False), ExecutionConfig(),
                                    environ={**self.environ, verifier.MANIFEST_SHA256_ENV_VAR: 'A' * 64})

    def test_ungated_verifier(self):
        with patch.object(verifier, 'validate_manifest', return_value=self.manifest) as validate:
            built = verifier.make_verifier(RewardConfig(verifier_gate=False), ExecutionConfig(verifier_workers=4),
                                           self.root / 'epoch-001', environ=self.environ)
        validate.assert_called_once_with(str(self.root / 'manifest.json'), 'a' * 64)
        self.assertIs(type(built), verifier.NativeVerifier)
        self.assertIsInstance(built, verifier.Verifier)
        self.assertEqual([built.slots._value, built.identity, built.timeout_seconds], [4, 'a' * 64, 30])
        self.assertTrue(built.inventory_path.is_file())

    def test_gated_verifier_checks_the_policy_then_the_runtime_once_per_run(self):
        calls = []
        with self.sqlglot, \
                patch.object(verifier, 'load_policy', side_effect=lambda *args: calls.append(args) or {'/p': {}}), \
                patch.object(verifier, 'validate_manifest', side_effect=lambda *args: calls.append(args) or
                             self.manifest):
            setup = verifier.setup_verifier(RewardConfig(verifier_gate=True), ExecutionConfig(), environ=self.environ, **self.gate)
        self.assertEqual(calls, [('gate-policy.json', 'b' * 64, self.tasks),
                                 (str(self.root / 'manifest.json'), 'a' * 64)])
        first, second = setup.verifier(self.root / 'epoch-001'), setup.verifier(self.root / 'epoch-002')
        for built in (first, second):
            self.assertIs(type(built), verifier.GatedNativeVerifier)
            self.assertIsInstance(built, verifier.Verifier)
            self.assertEqual([built.gate_policy, built.slots._value], [{'/p': {}}, 16])
            self.assertEqual(built.extraction_script, verifier._EXTRACTION_SCRIPT)
        self.assertNotEqual(first.receipt_dir, second.receipt_dir)

    def test_the_gate_needs_its_policy_and_the_training_tasks(self):
        for missing, message in (
            ({'gate_policy_path': None}, 'gate policy file'),
            ({'gate_policy_sha256': None}, 'gate policy file'),
            ({'gate_policy_sha256': 'x'}, '64 lowercase hex'),
            ({'tasks': []}, 'training tasks'),
        ):
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, message):
                verifier.setup_verifier(RewardConfig(verifier_gate=True), ExecutionConfig(), environ=self.environ,
                                        **{**self.gate, **missing})

    def test_the_gate_without_sqlglot_fails_at_setup(self):
        with (
            patch.dict(sys.modules, {'sqlglot': None}),
            patch.object(verifier, 'load_policy') as load,
            patch.object(verifier, 'validate_manifest') as validate,
            self.assertRaisesRegex(verifier.VeriEQLError, 'sqlglot'),
        ):
            verifier.setup_verifier(RewardConfig(verifier_gate=True), ExecutionConfig(), environ=self.environ, **self.gate)
        load.assert_not_called()
        validate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
