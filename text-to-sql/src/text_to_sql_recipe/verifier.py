"""VeriEQL counterexample shaping: an equivalence check for answers that already match the gold result.

An answer whose executed result matches the gold result earns the full outcome reward. With the verifier on, the
environment also asks VeriEQL, a bounded SMT-based SQL equivalence checker, whether the predicted query is equivalent
to the gold query (one check per answer, database bound 2, 30-second deadline). Only a refutation changes the reward:
a refuted answer earns `1 - reward.verifier_weight` instead of 1. Every other outcome (equivalent, timeout,
unsupported, inconclusive) keeps the execution reward. Evaluation never uses the verifier.

VeriEQL is a separately licensed tool, not part of this package
---------------------------------------------------------------
VeriEQL (https://github.com/VeriEQL/VeriEQL) is distributed under CC BY-NC-SA 4.0, a non-commercial licence. This
package does not include VeriEQL and never imports it. To train with `reward.verifier = "verieql"` (the default),
VeriEQL and its solver (z3-solver) must be installed in a Python interpreter of their own, and the licence must fit
your use. `python -m text_to_sql_recipe setup verieql` (`verieql_setup`) does the installation once you accept the
licence: it fetches the pinned VeriEQL revision from GitHub, builds that interpreter, and writes the runtime manifest
below. With `reward.verifier = "none"` nothing in this module runs and VeriEQL is not needed.

How the recipe finds it
-----------------------
VeriEQL runs as a subprocess of the interpreter it is installed in. A runtime manifest describes that installation:
a JSON file with `kind` (its runtime contract, below), `source` (`VERIEQL_SOURCE`, the VeriEQL commit the
installation is built from), `python` (the interpreter), `source_root` (the VeriEQL source directory), `stdlib_root` and
`site_packages_root` (the interpreter's standard library and site-packages), `z3_root` and `z3_version` (the z3
package), and `files`, the SHA-256 of every file under those directories and of every file the interpreter loads.
Two environment variables point the recipe at it:

  TEXT_TO_SQL_VERIEQL_MANIFEST          path of the runtime manifest
  TEXT_TO_SQL_VERIEQL_MANIFEST_SHA256   SHA-256 of the manifest file

At startup `validate_manifest` checks the manifest against that pin, re-hashes every listed file, and runs a probe in
the interpreter, which must report the platform of the manifest's contract, the declared z3 version and directories,
and no loaded file missing from the inventory. Every check then attests, inside the subprocess, that each file it
loaded is listed with unchanged contents before its verdict is released.

Runtime contracts (`RUNTIME_CONTRACTS`, keyed by the manifest's `kind`):

  verieql-native-linux-aarch64-v1   Linux on arm64 (aarch64)
  verieql-native-linux-x86_64-v1    Linux on x86_64
  verieql-native-macos-arm64-v1     macOS on Apple silicon (arm64)

`setup verieql` writes the contract of the machine it runs on. A manifest also records `platform`, `verieql_revision`
and `python_version`, and must carry the pinned VeriEQL revision (`VERIEQL_REVISION`) with exactly its source tree
(`VERIEQL_SOURCE_SHA256`, see `source_tree_sha256`), z3 `Z3_VERSION` with VeriEQL's own z3py_libs copied over the z3
package's Python files (as the stock release has them), and Python `PYTHON_SERIES`.

On Linux the probe reads /proc/self/maps, which lists every file mapped into the process. macOS has no /proc: there
the probe asks dyld for every loaded image (`_dyld_image_count`, `_dyld_get_image_name`, `_dyld_get_image_header`)
and walks the process's memory regions (`proc_pidinfo`, `proc_regionfilename`) for every other mapped file, the
dynamic linker included. An image in the dyld shared cache is not a file on disk: it belongs to the operating
system's sealed system-library cache, which the manifest pins by its UUID (`dyld_shared_cache_uuid`) and which every
check verifies again. A macOS update replaces that cache, so run `setup verieql` again after one.

The executed-counterexample gate
--------------------------------
With `reward.verifier_gate` on (it is off by default), a refutation is honoured only when its counterexample survives
execution. The solver's model is read back as table rows, materialized as a small SQLite database under the source
schema's own constraints, and both queries are run there with the recipe's grader (`sqlite_exec.compare_results`).
The penalty applies only when the grader tells the two results apart on that database; every other outcome abstains,
which turns the refutation into an `unknown` verdict that keeps the execution reward. The gate reads the dataset's
gate policy file, passed to `setup_verifier` (see `load_policy`): for each training database, its SHA-256, its schema
and the audited status of each foreign key. It also needs sqlglot, which lists the tables each query reads; sqlglot
is imported only when the gate is on.

Interface
---------
`setup_verifier` validates the runtime (and the gate policy) once per run and returns None for
`reward.verifier = "none"`; `VeriEQLSetup.verifier(receipt_dir)` builds the verifier for one epoch; `make_verifier`
does both. A verifier offers `await check(task, prediction)`, which returns the verdict dictionary, and
`await close()`. Each check writes a JSON receipt (input, outcome, failed attempts) under its receipt directory.
A runtime that does not complete a valid check raises `VeriEQLError` rather than returning a verdict that would keep
the execution reward.

Verdict keys. Every verdict has `verdict` and `state`:
  "confirmed" / "EQU"; "refuted" / "NEQ" (the only verdict the reward penalizes); "timeout" / "TMO";
  "inconclusive" / "UNKNOWN" or "UNWITNESSED" (a solver refutation with no counterexample); and, from the gate,
  "unknown" / "UNKNOWN" for a refutation whose counterexample did not survive execution.
The native verifier adds `backend`, `runtime_sha256`, `bound`, `timeout_seconds`, `queue_seconds`, `service_seconds`
and `startup_retries`; refutations carry `refutation_route` and the counterexample attestation. The gated verifier
adds, for each refutation it examined, `ungated_verdict`, `counterexample_gate` (with its `decision`, "penalize" or
"abstain", and the `reason`) and `gate_total_seconds`.
"""
from __future__ import annotations

import asyncio
import collections
import functools
import hashlib
import importlib
import json
import math
import os
import platform
import re
import signal
import sqlite3
import subprocess
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from . import sqlite_exec
from .sqlite_exec import SQLiteInfrastructureError, SQLiteQueryError, SQLiteQueryTimeout, SQLiteResultLimitExceeded

if TYPE_CHECKING:
    from .config import ExecutionConfig, RewardConfig

# Environment variables that locate the VeriEQL runtime manifest on this machine (see the module docstring).
MANIFEST_ENV_VAR = "TEXT_TO_SQL_VERIEQL_MANIFEST"
MANIFEST_SHA256_ENV_VAR = "TEXT_TO_SQL_VERIEQL_MANIFEST_SHA256"

# The solver directories of VeriEQL's own container build (the Dockerfile at `VERIEQL_REVISION`, on the public
# yanghece96/verieql image), which this module calls the stock release. `_input_row` writes them into every input
# row; the native verifier replaces them with the manifest's directories before an equivalence check, and the witness
# extraction reads only `source_root`, which it replaces too.
STOCK_SOURCE_ROOT = "/VeriEQL"
STOCK_Z3_ROOT = "/opt/conda/envs/VeriEQL/lib/python3.11/site-packages/z3"
STOCK_STDLIB_ROOT = "/opt/conda/envs/VeriEQL/lib/python3.11"
# What the stock release runs, pinned for the platform contracts: the public VeriEQL commit its source directory was
# built from (https://github.com/VeriEQL/VeriEQL), the `source_tree_sha256` of that commit's 248 files, the solver
# version `z3.get_version_string()` reports (the z3-solver 4.13 distribution), and the Python release series.
VERIEQL_REVISION = "493cbb81000205e33b0623cfd1c39106fa035fae"
# The source a runtime is built from; a runtime manifest names it as its `source`.
VERIEQL_SOURCE = "https://github.com/VeriEQL/VeriEQL/tree/" + VERIEQL_REVISION
VERIEQL_SOURCE_SHA256 = "98510b2f7a8a420820245eaea87a8286ae9bca6ab05f343b5e254eb7b7ae9c52"
Z3_VERSION = "4.13.0"
PYTHON_SERIES = "3.11"
# The paper's single equivalence check per pair: symbolic tables of at most two rows.
PAPER_BOUND = 2
# How many solver-route refutations a verifier must have seen before "none of them attested a counterexample" is
# read as a broken attestation rather than as a solver that genuinely found no models. Eight because the attestation
# is all-or-nothing -- the attribute is either read correctly or it is not -- so a handful is already decisive, while
# a step with only two or three refutations early in training must not be able to end a run on its own.
MIN_SOLVER_REFUTATIONS_FOR_ATTESTATION_AUDIT = 8

# The exit code the helper reserves for its own orderly failures, and the whole reason a hard abort of the solver can
# be told apart from a broken adapter. An unrecognised exception origin, a setup-phase failure, or an import that did
# not resolve are this adapter's contract breaking, and they must stay fatal. A native abort is different: the pinned
# solver's C++ layer calls `exit` on an assertion violation (Z3Prover/z3#7189), and no Python handler in the child can
# intercept it. So the helper labels its own failures instead of leaving the caller to guess from an exit code every
# layer below it also uses. Chosen outside every range that means something else here: 1 (an uncaught Python
# exception), 2 (argparse), 125-127 (a command that could not be run, invoked or found), and 128+N (killed by
# signal N).
HELPER_ADAPTER_FAILURE_EXIT_CODE = 97

# Exit codes meaning the helper never started (the command could not be run, invoked or found), so the solver never
# ran and could not have aborted. Always fatal.
INVOCATION_FAILURE_EXIT_CODES = (125, 126, 127)

# The VeriEQL column type for each SQLite declared type, by SQLite's affinity keywords: integer, boolean and numeric
# types become INT, date, time and text types VARCHAR. Only types are declared; no primary-key, foreign-key or NOT
# NULL constraints from the schema reach VeriEQL's symbolic database, the setup the ReViSQL authors' verdicts come from.
_TYPE_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("INT", "INTEGER", "TINYINT", "SMALLINT", "MEDIUMINT", "BIGINT"), "INT"),
    (("BOOL",), "INT"),
    (("REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL", "NUMBER"), "INT"),
    (("DATE", "TIME", "YEAR"), "VARCHAR"),
    (
        ("CHAR", "TEXT", "CLOB", "STRING", "VARCHAR", "NVARCHAR", "BLOB", "BINARY"),
        "VARCHAR",
    ),
)

# The equivalence check, run in VeriEQL's interpreter (wrapped by `checked_helper`). It calls VeriEQL's Environment
# directly at one configured bound, as the ReViSQL authors' setup does, and maps each way the check can end to one
# JSON line on stdout, or to exit code 97 for the adapter's own failures.
_HELPER_SCRIPT = r"""
import json
import os
import sys
import traceback
from pathlib import Path

try:
    # Inside the guard, not above it. Everything this process can fail at in an
    # orderly way has to leave by the one exit below, or the caller's rule --
    # "the reserved code is our bug, anything else is the solver dying" -- has a
    # hole in it that silently reclassifies an adapter failure as an undecided
    # check. Reading a malformed input row is exactly such a failure.
    data = json.loads(sys.stdin.read())
    sys.path.insert(0, os.getcwd())
    from errors import (
        NotEquivalenceError,
        NotSupportedError,
        SyntaxError as VeriEQLSyntaxError,
        UnknownError,
    )
    from environment import Environment

    def _inside_stock_environment(error):
        # The adapter passes its pinned solver source root in the input row.
        # Do not infer this boundary from the shell working directory or module
        # layout: either could silently widen or narrow the tolerated set.
        root = Path(data["source_root"]).resolve()
        frames = traceback.extract_tb(error.__traceback__)
        if not frames:
            return False
        origin = Path(frames[-1].filename)
        # ``<string>`` and other synthetic traceback names resolve relative to
        # the working directory. They are not provenance for the stock solver.
        if not origin.is_absolute() or not origin.is_file():
            return False
        return root in origin.resolve().parents

    def _inside_stock_stdlib_dependency(error):
        # A stock parser can raise from Python's standard library (for example,
        # ``re``) after calling it with a malformed intermediate token.  That is
        # an analyzer outcome, not an adapter/setup failure.  Keep this narrow:
        # the leaf must be in the active interpreter's stdlib and an earlier
        # traceback frame must be inside the pinned stock source.  Arbitrary
        # callbacks and third-party site packages remain fatal.
        root = Path(data["source_root"]).resolve()
        frames = traceback.extract_tb(error.__traceback__)
        if len(frames) < 2:
            return False
        stdlib = Path(data["stdlib_root"]).resolve()

        def _is_stdlib_frame(frame):
            origin = Path(frame.filename)
            if not origin.is_absolute() or not origin.is_file():
                return False
            origin = origin.resolve()
            if stdlib not in origin.parents:
                return False
            relative = origin.relative_to(stdlib)
            if not relative.parts:
                return False
            # The image digest pins its interpreter and thus this positive
            # standard-library inventory. Do not accept an arbitrary file just
            # because it happens to live under the interpreter prefix.
            top_level = relative.parts[0]
            if len(relative.parts) == 1:
                top_level = Path(top_level).stem
            return top_level in sys.stdlib_module_names

        stock_indices = []
        for index, frame in enumerate(frames[:-1]):
            caller = Path(frame.filename)
            if caller.is_absolute() and caller.is_file() and root in caller.resolve().parents:
                stock_indices.append(index)
        if not stock_indices:
            return False
        # The chain from the deepest stock frame to the exception leaf must be
        # all standard library. This rejects a third-party parser or callback
        # interposed between stock source and a stdlib helper such as ``re``.
        return all(_is_stdlib_frame(frame) for frame in frames[max(stock_indices) + 1 :])

    def _inside_pinned_z3_dependency(error):
        # Some stock solver failures surface from its pinned z3 dependency
        # rather than the final /VeriEQL frame. Treat only that dependency path
        # as a solver-inconclusive outcome, not an arbitrary callback error.
        root = Path(data["source_root"]).resolve()
        frames = traceback.extract_tb(error.__traceback__)
        if len(frames) < 2:
            return False
        origin = Path(frames[-1].filename)
        if not origin.is_absolute() or not origin.is_file():
            return False
        origin = origin.resolve()
        dependency_root = Path(data["z3_root"]).resolve()
        if dependency_root not in (origin, *origin.parents):
            return False
        for frame in frames[:-1]:
            caller = Path(frame.filename)
            if caller.is_absolute() and caller.is_file() and root in caller.resolve().parents:
                return True
        return False

    def _inside_pinned_ordered_set_dependency(error):
        # A stock projection object can be unhashable when passed to OrderedSet.
        # This is an unsupported analyzer shape, not a setup/infrastructure error.
        if not isinstance(error, TypeError):
            return False
        frames = traceback.extract_tb(error.__traceback__)
        if len(frames) < 2:
            return False
        leaf = Path(frames[-1].filename)
        dependency = Path(data["z3_root"]).resolve().parent / "ordered_set"
        stock = Path(data["source_root"]).resolve()
        if not leaf.is_absolute() or not leaf.is_file() or dependency not in leaf.resolve().parents:
            return False
        return any(
            Path(frame.filename).is_absolute()
            and Path(frame.filename).is_file()
            and stock in Path(frame.filename).resolve().parents
            for frame in frames[:-1]
        )

    _MISSING = object()

    def _counterexample_rows(rows):
        # ``counterexample_dict`` is the structured form of the witness: one
        # entry per base-table row of the model. It is populated only when the
        # stock Environment was built with ``generate_code=True``; this adapter
        # runs ``generate_code=False``, so this is 0 today and is reported as
        # evidence of exactly that, not as the gate.
        if not isinstance(rows, dict):
            return 0
        total = 0
        for table_rows in rows.values():
            try:
                total += len(table_rows)
            except TypeError:
                return 0
        return total

    def _attestation_shape(value):
        # What was actually observed on the attribute, not just whether it
        # passed the test. The gate turns on ``is not None``; if a future stock
        # build pre-initialises the attribute to ``""`` or ``[]``, or drops it,
        # the gate silently changes meaning in one direction or the other. The
        # record has to be able to say which of those happened, so the shape
        # travels with every refutation.
        if value is _MISSING:
            return "absent"
        if value is None:
            return "none"
        if isinstance(value, str):
            return "str:empty" if not value else "str"
        return ("empty:" if not value else "") + type(value).__name__

    def _inside_pinned_solver(error):
        # Exception classes alone are not provenance: an external callback can
        # import and raise VeriEQL's public error types. A released-helper
        # unknown must still originate in the pinned solver or one of its two
        # explicitly admitted dependency paths.
        return (
            _inside_stock_environment(error)
            or _inside_pinned_z3_dependency(error)
            or _inside_stock_stdlib_dependency(error)
        )

    setup_complete = False
    counterexample = _MISSING
    counterexample_dict = _MISSING
    analyzer_error = None
    dependency_error = None
    stock_unknown_error = None
    known_outcome = None
    # The check never asks VeriEQL for the witness rows: they exist only under
    # ``generate_code=True``, and ``Environment.analyze`` upper-cases both query
    # strings before parsing whenever that is set, which would change string
    # literals and quoted identifiers in the queries being checked. The
    # counterexample gate (`GatedNativeVerifier`) instead reads the witness from
    # the solver's model in a run of its own (`_EXTRACTION_SCRIPT`) and replays
    # it in SQLite.
    try:
        with Environment(generate_code=False, timer=False, show_counterexample=False) as env:
            for table_name, columns in data["schema"].items():
                env.create_database(
                    attributes=columns,
                    bound_size=data["bound_size"],
                    name=table_name,
                )
            env.save_checkpoints()
            setup_complete = True
            result = env.analyze(data["pred_sql"], data["gold_sql"])
            # Read the attestation here, inside the ``with``, while the
            # environment is still alive. Reading it after the block would rest
            # the whole gate on an unverified assumption about third-party
            # teardown -- and ``Environment.__exit__`` does clear state and run
            # ``gc.collect()``. If a future build cleared this attribute too,
            # every refutation would silently become unwitnessed, the verifier
            # deduction would never apply, and every reported metric would look
            # plausible. That failure must not be reachable by accident.
            counterexample = getattr(env, "counterexample", _MISSING)
            counterexample_dict = getattr(env, "counterexample_dict", _MISSING)
    except (
        NotEquivalenceError,
        NotSupportedError,
        VeriEQLSyntaxError,
        UnknownError,
        NotImplementedError,
    ) as error:
        # A solver outcome only has its released-helper unknown semantics after
        # the stock environment reached the analysis phase. The same exception
        # during setup means the verifier never built the database to check.
        if not setup_complete:
            raise RuntimeError("stock VeriEQL setup raised a solver outcome") from error
        if not _inside_pinned_solver(error):
            raise
        known_outcome = error
    except Exception as error:
        # Setup failures and errors outside the pinned stock implementation are
        # adapter failures. Once setup completes, ReViSQL treats any exception
        # originating in the stock analyzer or teardown as inconclusive. Keep
        # the ``with`` teardown in this branch intentionally: the released
        # helper likewise reports a post-analysis teardown exception as unknown.
        if not setup_complete:
            raise
        if _inside_pinned_z3_dependency(error):
            dependency_error = error
        elif _inside_pinned_ordered_set_dependency(error):
            dependency_error = error
        elif _inside_stock_stdlib_dependency(error):
            stock_unknown_error = error
        else:
            if not _inside_stock_environment(error):
                raise
            analyzer_error = error
    if analyzer_error is not None:
        # A stock verifier exception is an inconclusive check, as in the ReViSQL
        # authors' setup: only a formal False applies the verifier deduction.
        # Keep the diagnostic string, but do not turn an unsupported stock SQL
        # construct into a harness-level parser failure.
        print(
            json.dumps(
                {
                    "result": None,
                    "stock_analyzer_error": True,
                    "error": type(analyzer_error).__name__ + ": " + str(analyzer_error),
                }
            )
        )
    elif dependency_error is not None:
        print(
            json.dumps(
                {
                    "result": None,
                    "dependency_error": True,
                    "error": type(dependency_error).__name__ + ": " + str(dependency_error),
                }
            )
        )
    elif stock_unknown_error is not None:
        print(
            json.dumps(
                {
                    "result": None,
                    "stock_unknown": True,
                    "error": type(stock_unknown_error).__name__ + ": " + str(stock_unknown_error),
                }
            )
        )
    elif known_outcome is not None:
        # These are VeriEQL's known solver outcomes: it cannot establish the
        # relation, so Appendix E.5(3) retains the result-grade reward.
        print(
            json.dumps(
                {
                    "result": None,
                    "error": type(known_outcome).__name__ + ": " + str(known_outcome),
                }
            )
        )
    elif isinstance(result, bool):
        # ``counterexample`` is assigned its header the moment ``solver.check()``
        # returns ``sat`` -- before, and independently of, the row rendering that
        # ``generate_code``/``show_counterexample`` control. It therefore stays
        # ``None`` on every route to ``False`` that never reached the solver.
        # That makes it the solver's own attestation that a model exists, and it
        # is readable without changing what the solver is asked to do.
        payload = {"result": result}
        if result is False:
            # Only a solver refutation is gated, so only it carries the
            # attestation; a confirmation's output stays exactly as it was.
            payload["refutation_route"] = "solver_model"
            payload["counterexample"] = (
                counterexample is not _MISSING and counterexample is not None
            )
            payload["counterexample_rows"] = _counterexample_rows(counterexample_dict)
            payload["counterexample_shape"] = _attestation_shape(counterexample)
        print(json.dumps(payload))
    else:
        # Environment's projection-arity sentinel, and a *sound* refutation.
        #
        # ``compare`` returns -1 when the two queries' outermost tuples have
        # different numbers of attributes. Two relations of different arity
        # cannot be equal under any database, so this is a proof of
        # non-equivalence: no model is needed, and none can exist. That is why
        # it is not gated on a witness, and why "the solver was never consulted"
        # is not an argument against it -- there is nothing for the solver to do.
        #
        # It remains sound in exactly the case that makes result-matching
        # misleading: both queries empty on the graded database, matching there,
        # and still not equivalent. Unlike NotEquivalenceError, which fires on an
        # uninterpreted-function marker mismatch and can be raised for
        # semantically equal wrappers, arity is a syntactic property of the
        # output schema and cannot be wrong in that direction.
        #
        # The route is still recorded so its share of refutations stays visible
        # in evidence and its cost stays measurable.
        print(
            json.dumps(
                {
                    "result": False,
                    "refutation_route": "column_arity",
                    "counterexample": False,
                    "counterexample_rows": 0,
                    "witness_gate": "column_mismatch",
                    "error": f"column mismatch (result={result})",
                }
            )
        )
except Exception:
    # A broken stock environment is not a solver-inconclusive result. Fail the
    # process so the caller fails closed instead of disabling the deduction.
    #
    # Unchanged in what it decides; changed only in how it says so. Exiting with
    # the reserved adapter-failure code rather than re-raising keeps this class
    # fatal at the caller now that a plain non-zero exit no longer is: a bare
    # ``raise`` leaves exit code 1, which is also what a wrapper process and a
    # solver that died below Python leave. Everything above
    # this line -- an unrecognised exception origin, a setup-phase solver
    # outcome, an import that did not resolve -- keeps failing closed because
    # this hands the caller a code only this handler can produce.
    #
    # The traceback still goes to stderr. It is the entire diagnostic for these,
    # and the caller records the last line of it.
    traceback.print_exc(file=sys.stderr)
    sys.exit(97)
"""

# The witness extraction for the executed-counterexample gate, run in VeriEQL's interpreter (wrapped by
# `checked_helper`): it reads the same stock solver model without enabling query-changing code generation, and
# prints the counterexample's tables as one JSON line.
_EXTRACTION_SCRIPT = r'''"""Read the same stock solver model without enabling query-changing code generation."""
import json,sys,re
from fractions import Fraction
x=json.load(sys.stdin)
sys.path.insert(0,x['source_root'])
from environment import Environment
from z3 import is_true, is_int_value, is_rational_value
literals={}
for sql in [x['pred_sql'],x['gold_sql']]:
 for lit in re.findall(r"'((?:''|[^'])*)'",sql):
  lit=lit.replace("''","'");literals.setdefault(lit.upper(),set()).add(lit)
out={'mode':'generate_code=False; direct model extraction','query_text_unchanged':True}
with Environment(generate_code=False,timer=False,show_counterexample=False) as env:
 for table,cols in x['schema'].items():env.create_database(attributes=cols,bound_size=x['bound_size'],name=table)
 env.save_checkpoints();result=env.analyze(x['pred_sql'],x['gold_sql']);out['result']=result
 out['attested']=env.counterexample is not None
 if result is False and out['attested']:
  m=env.solver.model();tables=[];issues=[]
  for table in env.base_databases.values():
   rows=[];raw=[]
   for tup in table.tuples:
    if is_true(m.eval(env.DELETED_FUNCTION(tup.SORT),model_completion=True)):continue
    vals=[];rawvals=[]
    for attr in table.attributes:
     if is_true(m.eval(attr.NULL(tup.SORT),model_completion=True)): vals.append(None);rawvals.append('NULL');continue
     v=m.eval(attr.VALUE(tup.SORT),model_completion=True);rawvals.append(str(v))
     if is_int_value(v):value=v.as_long()
     elif is_rational_value(v):value=float(Fraction(v.numerator_as_long(),v.denominator_as_long()))
     else:issues.append('non-numeric symbolic value: '+str(v));value=str(v)
     col=next((typ for tn,cols in x['schema'].items() for cn,typ in cols.items() if tn.lower()==str(table.name).lower() and cn.lower()==str(attr.name).lower()),None)
     if col and str(col).upper().startswith('VARCHAR'):
      if value in env.variables:
       decoded=str(env.variables[value]);value=decoded[7:] if decoded.startswith('String_') else decoded
       if value.endswith('__Int'):value=value[:-5]
       if value in literals and len(literals[value])==1:value=next(iter(literals[value]))
       elif value in literals:issues.append('ambiguous literal casing: '+value)
      else:
       value='__witness_'+str(value)
     elif col and str(col).upper()=='DATE':issues.append('date conversion not certified')
     vals.append(value)
    rows.append(vals);raw.append(rawvals)
   tables.append({'name':str(table.name),'columns':[str(a.name) for a in table.attributes],'rows':rows,'raw_rows':raw})
  out.update(tables=tables,decode_issues=issues,model=str(m),variables={str(k):str(v) for k,v in env.variables.items()})
print(json.dumps(out))
'''


class VeriEQLError(RuntimeError):
    """The configured stock verifier did not complete a valid check."""


def _sqlite_type_to_verieql(sqlite_type: str) -> str:
    normalized = (sqlite_type or "").upper().strip()
    for keywords, target in _TYPE_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return target
    return "VARCHAR"


@functools.cache
def extract_schema_from_sqlite(db_file: str) -> dict[str, dict[str, str]]:
    """Read the released adapter's all-table schema from a pinned SQLite file."""
    path = Path(db_file)
    if not path.is_file():
        raise VeriEQLError(f"missing SQLite database for VeriEQL: {path}")
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        schema: dict[str, dict[str, str]] = {}
        for (table,) in cursor.fetchall():
            escaped = str(table).replace("'", "''")
            cursor.execute(f"PRAGMA table_info('{escaped}')")
            columns = {
                str(row[1]): _sqlite_type_to_verieql(str(row[2]))
                for row in cursor.fetchall()
            }
            if columns:
                schema[str(table)] = columns
    finally:
        connection.close()
    if not schema:
        raise VeriEQLError(f"SQLite database has no tables for VeriEQL: {path}")
    return schema


def _input_row(
    task: Mapping[str, Any], prediction: str, *, bound: int
) -> dict[str, Any]:
    if not str(task.get("database_path") or "").endswith(".sqlite"):
        raise VeriEQLError(f"VeriEQL only supports SQLite tasks: {task.get('id')}")
    return {
        "schema": extract_schema_from_sqlite(str(task["database_path"])),
        "pred_sql": prediction.strip(),
        "gold_sql": str(task["gold_sql"]).strip(),
        "bound_size": bound,
        "source_root": STOCK_SOURCE_ROOT,
        "z3_root": STOCK_Z3_ROOT,
        "stdlib_root": STOCK_STDLIB_ROOT,
    }


def _nonzero_exit_outcome(
    completed: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    """Read a dead verifier process: our bug, or a check that did not decide.

    One fact arrives -- a non-zero exit -- and it has two meanings that must not be collapsed. The helper's reserved
    code (and a command that never ran) raises here. Any other code is the verifier dying rather than deciding, and
    comes back as an `ABORTED` outcome with its exit code; `NativeVerifier.check` refuses that outcome too, so in this
    recipe every non-zero exit that is not the retried import race ends the run with a `VeriEQLError`.
    """
    detail = (completed.stderr or completed.stdout or "").strip().splitlines()
    suffix = f": {detail[-1]}" if detail else ""
    if completed.returncode == HELPER_ADAPTER_FAILURE_EXIT_CODE:
        # The helper reached its own outer handler and named itself. That is this adapter's contract breaking -- an
        # exception whose origin it does not recognise, a setup phase that never completed, an import that did not
        # resolve -- and it stays fatal: a harness that quietly absorbs its own bugs is no longer evidence about
        # anything. See HELPER_ADAPTER_FAILURE_EXIT_CODE.
        raise VeriEQLError(f"stock VeriEQL adapter failure{suffix}")
    if completed.returncode in INVOCATION_FAILURE_EXIT_CODES:
        # The helper never ran, so nothing was checked and nothing could have aborted.
        raise VeriEQLError(
            f"the VeriEQL interpreter did not start (exit {completed.returncode}){suffix}"
        )
    # Any other non-zero exit is the verifier dying, not deciding. The exit code is retained rather than only
    # rendered into the detail string: a negative code is the process killed by a signal, and telling that apart from
    # an orderly failure below Python matters afterwards.
    return {
        "verdict": "inconclusive",
        "state": "ABORTED",
        "detail": f"stock VeriEQL exited {completed.returncode}{suffix}",
        "verifier_abort": True,
        "verifier_exit_code": completed.returncode,
    }


def decode_helper_output(stdout: str) -> dict[str, Any]:
    """Map the helper's JSON output line to a verdict."""
    try:
        output = json.loads(stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise VeriEQLError("stock VeriEQL emitted invalid JSON") from exc
    missing = object()
    result = output.get("result", missing) if isinstance(output, dict) else missing
    if result is missing or (
        result is not True and result is not False and result is not None
    ):
        raise VeriEQLError("stock VeriEQL emitted an invalid result")
    if result is True:
        return {"verdict": "confirmed", "state": "EQU"}
    if result is False:
        # The witness gate, and it applies to exactly one of the two routes to
        # a refutation.
        #
        # ``column_arity`` is a proof: different output arity means the two
        # relations cannot be equal under any database, so no model is needed
        # and none can exist. Gating it on a witness would score a provably
        # non-equivalent prediction 1.0 instead of 1 - verifier_weight, and so
        # precisely in the case the arity check exists for -- both sides empty
        # on the graded database, result-matching there, not equivalent.
        #
        # ``solver_model`` is a claim that a model was found, and that is what
        # must carry evidence. Failing that gate becomes an unknown that keeps
        # the execution reward -- never a penalty, never an error -- with its
        # own state so it is counted apart from the solver's own inconclusive
        # outcomes rather than folded in with them.
        route = str(output.get("refutation_route") or "solver_model")
        if route == "column_arity":
            return {
                "verdict": "refuted",
                "state": "NEQ",
                "detail": str(output.get("error") or "column mismatch"),
                "refutation_route": route,
                "counterexample": False,
                "counterexample_rows": 0,
                # Recorded on a refutation that passed, not on one that was
                # gated: it names the route, so the arity share of refutations
                # stays measurable in evidence.
                "witness_gate": "column_mismatch",
            }
        if output.get("counterexample") is True:
            return {
                "verdict": "refuted",
                "state": "NEQ",
                "detail": "refuted with a solver counterexample",
                "refutation_route": route,
                "counterexample": True,
                "counterexample_rows": int(output.get("counterexample_rows") or 0),
                "counterexample_shape": str(output.get("counterexample_shape") or ""),
            }
        return {
            "verdict": "inconclusive",
            "state": "UNWITNESSED",
            "detail": str(output.get("error") or "refuted without a counterexample"),
            "unwitnessed_refutation": True,
            "refutation_route": route,
            "witness_gate": str(output.get("witness_gate") or "no_counterexample"),
            "counterexample_shape": str(output.get("counterexample_shape") or ""),
        }
    return {
        "verdict": "inconclusive",
        "state": "UNKNOWN",
        "detail": str(output.get("error") or "verifier returned unknown"),
        "dependency_error": output.get("dependency_error") is True,
        "stock_unknown": output.get("stock_unknown") is True,
        "stock_analyzer_error": output.get("stock_analyzer_error") is True,
    }


def _audit_attestation(observed: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a batch whose witness attestation looks broken rather than absent.

    The gate reads one third-party attribute. If a future stock build cleared it
    during teardown, or renamed it, every solver refutation would become an
    unknown, the verifier deduction would never apply, and the run would train on a plain
    execution reward with every reported metric still looking healthy. That is
    unrecoverable and invisible, so it fails closed here.

    The signal is all-or-nothing by construction: the attribute either carries
    the solver's ``sat`` marker or it does not, so a batch in which *no* solver
    refutation attested while many claimed that route is evidence about the
    mechanism, not about the queries. A minimum count keeps a small or unlucky
    batch from tripping it.
    """
    solver_route = [
        result
        for result in observed
        if str(result.get("refutation_route") or "") == "solver_model"
    ]
    if len(solver_route) < MIN_SOLVER_REFUTATIONS_FOR_ATTESTATION_AUDIT:
        return
    if any(result.get("counterexample") is True for result in solver_route):
        return
    shapes = sorted(
        {str(result.get("counterexample_shape") or "") for result in solver_route}
    )
    raise VeriEQLError(
        f"stock VeriEQL returned {len(solver_route)} solver refutations and "
        "attested a counterexample for none of them; the witness attestation is "
        f"not being read correctly (observed shapes: {', '.join(shapes)}). "
        "Refusing the batch rather than silently disabling the verifier deduction."
    )


# Executed inside the isolated interpreter, both at admission and after each
# verifier invocation. /proc maps includes dlopen libraries that sys.modules
# cannot see (notably libz3 and the interpreter's shared libraries).
_RUNTIME_PROBE = r"""
import sys, sysconfig, platform, json, pathlib

def loaded_files():
    paths = {str(pathlib.Path(sys.executable).resolve())}
    for module in tuple(sys.modules.values()):
        for attribute in ('__file__', '__cached__'):
            name = getattr(module, attribute, None)
            if name and pathlib.Path(name).is_file():
                paths.add(str(pathlib.Path(name).resolve()))
    for line in pathlib.Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith('/'):
            path = pathlib.Path(parts[5])
            if not path.is_file():
                raise RuntimeError('Loaded library disappeared: ' + str(path))
            paths.add(str(path.resolve()))
    return sorted(paths)
"""

# The macOS counterpart of `_RUNTIME_PROBE`, executed the same way. dyld lists every Mach-O image the process has
# loaded; walking the process's memory regions finds every other mapped file (the dynamic linker itself, and any data
# file mapped into memory), as /proc/self/maps does on Linux. An image whose header carries MH_DYLIB_IN_CACHE was
# loaded from the dyld shared cache: it is not a file, so `loaded_files` leaves it to `shared_cache_uuid`, which
# identifies the whole sealed cache. Region walking skips submaps, which is how the kernel maps the shared cache.
_RUNTIME_PROBE_MACOS = r"""
import sys, sysconfig, platform, json, pathlib, os, ctypes

_LIBC = ctypes.CDLL(None, use_errno=True)
_MH_DYLIB_IN_CACHE = 0x80000000
_PROC_PIDREGIONINFO = 7
_PROC_REGION_SUBMAP = 1
_EINVAL = 22


class _RegionInfo(ctypes.Structure):
    # struct proc_regioninfo, <sys/proc_info.h>
    _fields_ = ([(name, ctypes.c_uint32) for name in (
                    'pri_protection', 'pri_max_protection', 'pri_inheritance', 'pri_flags')]
                + [('pri_offset', ctypes.c_uint64)]
                + [(name, ctypes.c_uint32) for name in (
                    'pri_behavior', 'pri_user_wired_count', 'pri_user_tag', 'pri_pages_resident',
                    'pri_pages_shared_now_private', 'pri_pages_swapped_out', 'pri_pages_dirtied', 'pri_ref_count',
                    'pri_shadow_depth', 'pri_share_mode', 'pri_private_pages_resident',
                    'pri_shared_pages_resident', 'pri_obj_id', 'pri_depth')]
                + [('pri_address', ctypes.c_uint64), ('pri_size', ctypes.c_uint64)])


def _function(name, restype, *argtypes):
    function = getattr(_LIBC, name)
    function.restype, function.argtypes = restype, list(argtypes)
    return function


def _images():
    count = _function('_dyld_image_count', ctypes.c_uint32)
    image_name = _function('_dyld_get_image_name', ctypes.c_char_p, ctypes.c_uint32)
    image_header = _function('_dyld_get_image_header', ctypes.c_void_p, ctypes.c_uint32)
    in_shared_cache = _function('_dyld_shared_cache_contains_path', ctypes.c_bool, ctypes.c_char_p)
    images = []
    for index in range(count()):
        name, header = image_name(index), image_header(index)
        if not name or not header:
            raise RuntimeError('dyld image without a name or header: ' + str(index))
        # mach_header(_64).flags sits at offset 24 in both header layouts.
        cached = bool(ctypes.c_uint32.from_address(header + 24).value & _MH_DYLIB_IN_CACHE)
        if cached != bool(in_shared_cache(name)):
            raise RuntimeError('Inconsistent shared cache membership: ' + os.fsdecode(name))
        images.append((os.fsdecode(name), cached))
    return images


def _mapped_files():
    region_info = _function('proc_pidinfo', ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                            ctypes.c_void_p, ctypes.c_int)
    region_file = _function('proc_regionfilename', ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                            ctypes.c_void_p, ctypes.c_uint32)
    pid, address, names, regions = os.getpid(), 0, set(), 0
    buffer = ctypes.create_string_buffer(4096)
    while True:
        info = _RegionInfo()
        if region_info(pid, _PROC_PIDREGIONINFO, address, ctypes.byref(info),
                       ctypes.sizeof(info)) != ctypes.sizeof(info):
            # EINVAL: no region at or above the address, the end of the walk.
            if ctypes.get_errno() != _EINVAL or not regions:
                raise RuntimeError('Cannot read the process memory regions')
            return names
        regions += 1
        if not info.pri_flags & _PROC_REGION_SUBMAP:
            length = region_file(pid, info.pri_address, buffer, len(buffer))
            if length > 0:
                names.add(os.fsdecode(buffer.raw[:length]))
        if info.pri_address + info.pri_size <= address:
            raise RuntimeError('The memory region walk did not advance')
        address = info.pri_address + info.pri_size


def shared_cache_uuid():
    uuid = (ctypes.c_ubyte * 16)()
    if not _function('_dyld_get_shared_cache_uuid', ctypes.c_bool, ctypes.c_void_p)(uuid):
        raise RuntimeError('dyld reports no shared cache')
    return bytes(uuid).hex()


def loaded_files():
    paths = {str(pathlib.Path(sys.executable).resolve())}
    for module in tuple(sys.modules.values()):
        for attribute in ('__file__', '__cached__'):
            name = getattr(module, attribute, None)
            if name and pathlib.Path(name).is_file():
                paths.add(str(pathlib.Path(name).resolve()))
    for name in _mapped_files() | {name for name, cached in _images() if not cached}:
        path = pathlib.Path(name)
        if not path.is_file():
            raise RuntimeError('Loaded library disappeared: ' + str(path))
        paths.add(str(path.resolve()))
    return sorted(paths)
"""


@dataclass(frozen=True)
class RuntimeContract:
    """What a runtime manifest of one `kind` promises: the platform its probe must report and how it finds files."""

    kind: str
    platform: str              # "<os>-<machine>", recorded in the manifest as `platform`
    system: str                # platform.system() inside the runtime
    machine: str               # platform.machine() inside the runtime
    backend: str               # the `backend` of every verdict
    probe: str                 # the loaded-file probe, run in the runtime's interpreter


RUNTIME_CONTRACTS = {contract.kind: contract for contract in (
    RuntimeContract("verieql-native-linux-aarch64-v1", "linux-aarch64", "Linux", "aarch64", "native-linux-aarch64",
                    _RUNTIME_PROBE),
    RuntimeContract("verieql-native-linux-x86_64-v1", "linux-x86_64", "Linux", "x86_64", "native-linux-x86_64",
                    _RUNTIME_PROBE),
    RuntimeContract("verieql-native-macos-arm64-v1", "macos-arm64", "Darwin", "arm64", "native-macos-arm64",
                    _RUNTIME_PROBE_MACOS),
)}


_SHARED_CACHE_UUID = re.compile(r"[0-9a-f]{32}")


def contract_for(manifest: Mapping[str, Any]) -> RuntimeContract:
    """The contract a manifest names in its `kind`."""
    contract = RUNTIME_CONTRACTS.get(manifest.get("kind"))
    if contract is None:
        raise ValueError('Unknown native verifier contract')
    return contract


def platform_contract(system: str, machine: str) -> RuntimeContract:
    """The platform contract `setup verieql` writes for a machine, from platform.system() and platform.machine()."""
    for contract in RUNTIME_CONTRACTS.values():
        if (contract.system, contract.machine) == (system, machine):
            return contract
    supported = ", ".join(c.platform for c in RUNTIME_CONTRACTS.values())
    raise VeriEQLError(f"no VeriEQL runtime contract for {system} on {machine}; supported: {supported}")


def source_tree_sha256(entries: Mapping[str, str]) -> str:
    """The SHA-256 of a file tree given as {relative path: file SHA-256}.

    It hashes one `<file SHA-256>  <relative path>` line per file, each ending in a newline, in sorted path order: the
    lines `sha256sum` prints, sorted by path. `VERIEQL_SOURCE_SHA256` is this digest of the pinned revision.
    """
    digest = hashlib.sha256()
    for name in sorted(entries):
        digest.update(f"{entries[name]}  {name}\n".encode())
    return digest.hexdigest()


def source_tree_entries(manifest: Mapping[str, Any]) -> dict[str, str]:
    """{path relative to `source_root`: SHA-256} for the manifest's source files, leaving out bytecode caches."""
    root = str(Path(manifest['source_root']).resolve()) + os.sep
    entries = {}
    for name, digest in manifest['files'].items():
        relative = name[len(root):] if name.startswith(root) else None
        if relative and '__pycache__' not in relative.split(os.sep):
            entries[relative] = digest
    return entries


# The report the probe prints, run in the runtime's interpreter after the probe's own definitions. `exercise` runs
# between importing VeriEQL and listing the loaded files (`setup verieql` runs real checks there).
_PROBE_IMPORTS = """
sys.path.insert(0, sys.argv[1])
import environment, z3
"""
_PROBE_REPORT = """
report = dict(arch=platform.machine(), version=z3.get_version_string(),
    source=str(pathlib.Path(environment.__file__).resolve().parent),
    z3=str(pathlib.Path(z3.__file__).resolve().parent),
    stdlib=str(pathlib.Path(sysconfig.get_path('stdlib')).resolve()),
    site=str(pathlib.Path(sysconfig.get_path('purelib')).resolve()), files=loaded_files(),
    system=platform.system(), python_version=platform.python_version())
"""
_PROBE_REPORT_SHARED_CACHE = """report['dyld_shared_cache_uuid'] = shared_cache_uuid()
"""


def runtime_probe_code(contract: RuntimeContract, exercise: str = "") -> str:
    """The probe program `validate_manifest` runs: argv[1] is the source root; it prints one JSON report line."""
    code = contract.probe + _PROBE_IMPORTS + exercise + _PROBE_REPORT
    if contract.system == "Darwin":
        code += _PROBE_REPORT_SHARED_CACHE
    return code + "print(json.dumps(report))\n"


def _check_declared_platform(manifest: Mapping[str, Any], contract: RuntimeContract) -> None:
    """A platform contract's own fields: the platform, and the pinned revision, solver and Python series."""
    if manifest.get('platform') != contract.platform:
        raise ValueError('Native verifier manifest platform differs from its contract')
    if platform.system() != contract.system:
        raise ValueError(f'The VeriEQL runtime manifest is for {contract.platform}; this machine runs '
                         f'{platform.system()}')
    if manifest.get('verieql_revision') != VERIEQL_REVISION:
        raise ValueError('Native verifier must run the pinned VeriEQL revision ' + VERIEQL_REVISION)
    if manifest.get('z3_version') != Z3_VERSION:
        raise ValueError('Native verifier must run the pinned solver version ' + Z3_VERSION)
    if not str(manifest.get('python_version', '')).startswith(PYTHON_SERIES + '.'):
        raise ValueError('Native verifier must run Python ' + PYTHON_SERIES)
    if contract.system == 'Darwin' and not _SHARED_CACHE_UUID.fullmatch(str(manifest.get('dyld_shared_cache_uuid'))):
        raise ValueError('A macOS runtime manifest must pin the dyld shared cache UUID')


def _check_pinned_source(manifest: Mapping[str, Any]) -> None:
    """The source directory holds exactly the pinned revision, and z3 carries VeriEQL's z3py_libs over its own."""
    entries = source_tree_entries(manifest)
    if source_tree_sha256(entries) != VERIEQL_SOURCE_SHA256:
        raise ValueError('The VeriEQL source tree differs from the pinned revision ' + VERIEQL_REVISION)
    z3_root = Path(manifest['z3_root']).resolve()
    for relative, digest in entries.items():
        parts = relative.split(os.sep)
        if (len(parts) == 2 and parts[0] == 'z3py_libs' and parts[1].endswith('.py')
                and manifest['files'].get(str(z3_root / parts[1])) != digest):
            raise ValueError("The z3 package lacks VeriEQL's z3py_libs copy of " + parts[1])


def runtime_inventory(manifest):
    roots = [Path(manifest[key]).resolve(strict=True)
             for key in ('source_root', 'stdlib_root', 'site_packages_root', 'z3_root')]
    required = {str(Path(manifest['python']).resolve(strict=True))}
    for root in roots:
        if not root.is_dir():
            raise ValueError('Runtime root is not a directory')
        required.update(str(p.resolve()) for p in root.rglob('*') if p.is_file())
    return required


def validate_manifest(path: str, expected_sha256: str) -> dict:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Native VeriEQL manifest hash mismatch')
    manifest = json.loads(raw)
    contract = RUNTIME_CONTRACTS.get(manifest.get('kind'))
    if contract is None:
        raise ValueError('Unknown native verifier contract')
    if manifest.get('source') != VERIEQL_SOURCE:
        raise ValueError('The runtime manifest must name the pinned VeriEQL source ' + VERIEQL_SOURCE
                         + '; run `python -m text_to_sql_recipe setup verieql` again to rebuild the runtime')
    _check_declared_platform(manifest, contract)
    files = manifest['files']
    if not files or not runtime_inventory(manifest) <= files.keys():
        raise ValueError('Incomplete native runtime inventory')
    for name, digest in files.items():
        if str(Path(name).resolve(strict=True)) != name:
            raise ValueError('Runtime inventory paths must be canonical')
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError('Native verifier file changed: ' + name)
    _check_pinned_source(manifest)
    probe = subprocess.run([manifest['python'], '-I', '-B', '-c', runtime_probe_code(contract),
                            manifest['source_root']], check=True, capture_output=True,
                           text=True, timeout=30)
    observed = json.loads(probe.stdout.splitlines()[-1])
    if observed['arch'] != contract.machine or observed['version'] != manifest['z3_version']:
        raise ValueError('Native verifier architecture or solver version mismatch')
    if (observed.get('system') != contract.system
            or observed.get('python_version') != manifest['python_version']):
        raise ValueError('Native verifier operating system or Python version mismatch')
    for observed_key, declared_key in (('source', 'source_root'), ('z3', 'z3_root'),
                                       ('stdlib', 'stdlib_root'), ('site', 'site_packages_root')):
        if observed[observed_key] != str(Path(manifest[declared_key]).resolve()):
            raise ValueError('Loaded runtime root differs from manifest: ' + declared_key)
    if not set(observed['files']) <= files.keys():
        raise ValueError('Loaded modules or native libraries missing from inventory')
    if contract.system == 'Darwin' and observed.get('dyld_shared_cache_uuid') != manifest['dyld_shared_cache_uuid']:
        raise ValueError('The macOS system library cache differs from the manifest (an OS update replaces it); '
                         'run `python -m text_to_sql_recipe setup verieql` again')
    return manifest


def checked_helper(inventory_sha256, helper_script=None, *, probe=None, shared_cache_uuid=None):
    # Buffer the solver output: no verdict is released if a lazy import or
    # native dependency falls outside the admitted inventory or changed bytes.
    # On macOS (`_RUNTIME_PROBE_MACOS`) the system library cache must also
    # still be the pinned one.
    code = (_RUNTIME_PROBE if probe is None else probe) + """
import contextlib, io, hashlib
raw_inventory = pathlib.Path(sys.argv[1]).read_bytes()
if hashlib.sha256(raw_inventory).hexdigest() != """ + repr(inventory_sha256) + """:
    raise RuntimeError("Runtime inventory changed")
inventory = json.loads(raw_inventory)
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    try:
        exec(""" + repr(_HELPER_SCRIPT if helper_script is None else helper_script) + """, {'__name__': '__main__'})
    finally:
        for name in loaded_files():
            if name not in inventory or hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() != inventory[name]:
                raise RuntimeError('Unpinned or changed loaded runtime file: ' + name)
"""
    if shared_cache_uuid is not None:
        code += """        if shared_cache_uuid() != """ + repr(shared_cache_uuid) + """:
            raise RuntimeError('Unpinned or changed system library cache')
"""
    return code + """print(buffer.getvalue(), end='')
"""


def _retryable_import_failure(completed, site_packages_root):
    """Retry only the pinned parser's expired import-export watchdog.

    A fresh process can complete the same imports after temporary scheduling or
    filesystem pressure. Other adapter failures and attestation failures must
    remain fatal, never be converted to an inconclusive verifier verdict.
    """
    lines = (completed.stderr or '').strip().splitlines()
    return (
        completed.returncode == HELPER_ADAPTER_FAILURE_EXIT_CODE
        and bool(lines)
        and lines[-1].startswith('Exception: mo_parsing.core is not expecting an export to ')
        and lines[-1].removeprefix('Exception: mo_parsing.core is not expecting an export to ').isidentifier()
        and ('missing expected call export("mo_parsing.core", '
             + lines[-1].rsplit(' ', 1)[-1] + ')') in completed.stderr
        and f'File "{site_packages_root}/mo_imports/__init__.py"' in completed.stderr
    )


class NativeVerifier:
    """VeriEQL equivalence checks in the pinned interpreter, `workers` subprocesses at a time."""

    def __init__(self, manifest, manifest_sha256, receipt_dir, *, workers=8,
                 timeout_seconds=30, queue_timeout_seconds=600):
        self.manifest, self.identity = manifest, manifest_sha256
        self.contract = contract_for(manifest)
        self.receipt_dir = Path(receipt_dir)
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        inventory = json.dumps(manifest["files"], sort_keys=True).encode()
        self.inventory_sha256 = hashlib.sha256(inventory).hexdigest()
        self.inventory_path = self.receipt_dir / ("inventory-" + self.inventory_sha256 + ".json")
        self.inventory_path.write_bytes(inventory)
        self.slots = asyncio.Semaphore(workers)
        self.timeout_seconds = timeout_seconds
        self.queue_timeout_seconds = queue_timeout_seconds
        self.processes = set()
        self.solver_routes = []
        self.counts = Counter()

    async def check(self, task, prediction):
        queued = time.monotonic()
        # Expired queue deadlines are infrastructure failures, never reward1.
        await asyncio.wait_for(self.slots.acquire(), self.queue_timeout_seconds)
        started = time.monotonic()
        attempts = []
        startup_retries = 0
        try:
            row = await asyncio.to_thread(_input_row, task, prediction,
                                          bound=PAPER_BOUND)
            row.update(source_root=self.manifest['source_root'],
                       z3_root=self.manifest['z3_root'], stdlib_root=self.manifest['stdlib_root'])
            deadline = time.monotonic() + self.timeout_seconds
            for attempt in range(2):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise VeriEQLError('Native verifier startup retry exhausted its deadline')
                completed = await self._attempt(row, remaining)
                if completed is None:
                    if attempts:
                        raise VeriEQLError('Native verifier startup retry timed out')
                    result = {'verdict': 'timeout', 'state': 'TMO'}
                    break
                if completed.returncode:
                    attempts.append({'attempt': attempt + 1, 'returncode': completed.returncode,
                                     'stdout': completed.stdout, 'stderr': completed.stderr})
                    if attempt == 0 and _retryable_import_failure(
                        completed, self.manifest['site_packages_root']
                    ):
                        self.counts['startup_retries'] += 1
                        startup_retries += 1
                        continue
                    result = _nonzero_exit_outcome(completed)
                    if result.get('verifier_abort'):
                        raise VeriEQLError('Native verifier aborted: ' + result['detail'])
                else:
                    result = decode_helper_output(completed.stdout)
                break
            result.update(backend=self.contract.backend, runtime_sha256=self.identity,
                          bound=PAPER_BOUND, timeout_seconds=self.timeout_seconds,
                          queue_seconds=started-queued, service_seconds=time.monotonic()-started,
                          startup_retries=startup_retries)
            self.counts["checked"] += 1
            self.counts[result["state"]] += 1
            self.counts["analyzer_errors"] += int(result.get("stock_analyzer_error", False))
            self.receipt_dir.mkdir(parents=True, exist_ok=True)
            receipt = {'task_id': str(task['id']), 'input': row, 'outcome': result,
                       'failed_attempts': attempts}
            (self.receipt_dir / (uuid.uuid4().hex + '.json')).write_text(json.dumps(receipt) + '\n')
            if result.get('refutation_route') == 'solver_model':
                self.solver_routes.append(result)
                _audit_attestation(self.solver_routes)
            return result
        except Exception as error:
            self.receipt_dir.mkdir(parents=True, exist_ok=True)
            (self.receipt_dir / ('failure-' + uuid.uuid4().hex + '.json')).write_text(
                json.dumps({'task_id': str(task['id']), 'prediction': prediction,
                            'gold_sql': task['gold_sql'], 'runtime_sha256': self.identity,
                            'failed_attempts': attempts,
                            'error': type(error).__name__ + ': ' + str(error)}) + '\n')
            raise
        finally:
            self.slots.release()

    def _checked_helper(self, helper_script):
        if self.contract.system == 'Darwin':
            return checked_helper(self.inventory_sha256, helper_script, probe=self.contract.probe,
                                  shared_cache_uuid=self.manifest['dyld_shared_cache_uuid'])
        return checked_helper(self.inventory_sha256, helper_script)

    async def _attempt(self, row, timeout_seconds, helper_script=None):
        process = await asyncio.create_subprocess_exec(
                self.manifest['python'], '-I', '-B', '-c', self._checked_helper(helper_script),
                str(self.inventory_path),
                cwd=self.manifest['source_root'], stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=True)
        self.processes.add(process)
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(json.dumps(row).encode()), timeout_seconds)
            return subprocess.CompletedProcess(
                [self.manifest['python']], process.returncode,
                stdout.decode(errors='replace'), stderr.decode(errors='replace'))
        except TimeoutError:
            return None
        finally:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.communicate()
            self.processes.discard(process)

    async def close(self):
        for process in list(self.processes):
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()


def _quote(s):
    return '"' + s.replace('"', '""') + '"'


def _typed_bag(rows):
    return collections.Counter(tuple((type(v).__name__, repr(v)) for v in row) for row in rows)


def _real_affinity(declared):
    t = declared.upper()
    return not ('INT' in t or any(x in t for x in ['CHAR', 'CLOB', 'TEXT']) or 'BLOB' in t or not t) and any(
        x in t for x in ['REAL', 'FLOA', 'DOUB'])


def _fidelity(expected, actual, types):
    if _typed_bag(expected) == _typed_bag(actual):
        return {'ok': True, 'kind': 'exact', 'conversions': 0}
    converted = []
    count = 0
    for row in expected:
        dst = []
        for v, t in zip(row, types):
            if type(v) is int and _real_affinity(t):
                f = float(v)
                if f != v:
                    return {'ok': False, 'kind': 'lossy_real_affinity'}
                dst.append(f)
                count += 1
            else:
                dst.append(v)
        converted.append(dst)
    if _typed_bag(converted) == _typed_bag(actual):
        return {'ok': True, 'kind': 'lossless_int_to_real', 'conversions': count}
    return {'ok': False, 'kind': 'changed_values_or_unsupported_affinity'}


def refutation_gate(case, tables, schema, policies, sql, dedupe_hypothesis=False):
    """Return penalize/abstain with reasons, fidelity, constraints and comparisons.

    Schema and per-FK policy must be frozen against the same source bundle.
    Dedupe produces a new candidate world; its removals are explicitly returned.
    Executed counterexample confirmation; never modifies the source database.
    """
    out = {'decision': 'abstain', 'dedupe_hypothesis': dedupe_hypothesis, 'dedupe_removed': 0}
    try:
        with tempfile.TemporaryDirectory() as d:
            typed = str(Path(d) / 'typed.sqlite')
            twin = str(Path(d) / 'twin.sqlite')
            c = sqlite3.connect(typed)
            try:
                for kind in ['table', 'index', 'view', 'trigger']:
                    for typ, name, table, ddl in schema:
                        if typ == kind:
                            c.execute(ddl)
                names = {n.lower(): n for t, n, tn, ddl in schema if t == 'table'}
                relevant = {x.lower() for x in case['referenced_tables']}
                todo = list(relevant)
                while todo:
                    n = todo.pop()
                    for fk in c.execute('PRAGMA foreign_key_list(' + _quote(names[n]) + ')'):
                        p = fk[2].lower()
                        if p in names and p not in relevant:
                            relevant.add(p)
                            todo.append(p)
                infos = {n: list(c.execute('PRAGMA table_info(' + _quote(names[n]) + ')')) for n in relevant}
                expected = {}
                for table in tables:
                    n = table['name'].lower()
                    if n not in relevant:
                        continue
                    if n in expected:
                        raise ValueError('duplicate witness table')
                    cols = table['columns']
                    rows = table['rows']
                    # Validate solver values before binding them to SQLite. Do not
                    # coerce unbounded integers or non-finite reals into a new world.
                    for row in rows:
                        for value in row:
                            if ((type(value) is int and not -(2**63) <= value < 2**63)
                                    or (type(value) is float and not math.isfinite(value))):
                                out.update(reason='unsupported_numeric_value', table=n)
                                return out
                    if dedupe_hypothesis:
                        pk = tuple(r[1].lower() for r in sorted(infos[n], key=lambda r: r[5]) if r[5])
                        keys = [pk] if pk else []
                        for ix in c.execute('PRAGMA index_list(' + _quote(names[n]) + ')'):
                            if ix[2] and not ix[4]:
                                parts = [r[2] for r in c.execute('PRAGMA index_info(' + _quote(ix[1]) + ')')]
                                if all(x is not None for x in parts):
                                    keys.append(tuple(x.lower() for x in parts))
                        lc = [x.lower() for x in cols]
                        seen = [set() for k in keys]
                        kept = []
                        for row in rows:
                            sigs = [tuple(row[lc.index(k)] for k in key) for key in keys]
                            if any(all(x is not None for x in sig) and sig in seen[i] for i, sig in enumerate(sigs)):
                                out['dedupe_removed'] += 1
                                continue
                            kept.append(row)
                            for i, sig in enumerate(sigs):
                                if all(x is not None for x in sig):
                                    seen[i].add(sig)
                        rows = kept
                    expected[n] = (cols, rows)
                # Native non-FK constraints always apply. FK policy is checked separately.
                c.execute('PRAGMA foreign_keys=OFF')
                try:
                    for n, (cols, rows) in expected.items():
                        c.executemany('INSERT INTO ' + _quote(names[n]) + ' (' + ','.join(map(_quote, cols))
                                      + ') VALUES (' + ','.join('?' for col in cols) + ')', rows)
                    c.commit()
                except sqlite3.IntegrityError as e:
                    out.update(reason='invalid_constraint', error=str(e))
                    return out
                checks = []
                # Do not short circuit on fidelity or on the first failed FK.
                for n in sorted(relevant):
                    for fid in sorted({r[0] for r in c.execute('PRAGMA foreign_key_list(' + _quote(names[n]) + ')')}):
                        pol = policies.get((n, fid))
                        x = {'table': n, 'id': fid, 'status': 'unresolved_policy'}
                        if pol:
                            x['status'] = pol['status']
                            if pol['status'] == 'enforced':
                                x['violations'] = c.execute(pol['query']).fetchone()[0]
                        checks.append(x)
                out['constraint_checks'] = checks
                fids = []
                for n in sorted(relevant):
                    allcols = [r[1] for r in infos[n]]
                    cols, rows = expected.get(n, (allcols, []))
                    types = {r[1].lower(): r[2] for r in infos[n]}
                    actual = c.execute('SELECT ' + ','.join(map(_quote, cols)) + ' FROM ' + _quote(names[n])).fetchall()
                    fids.append({'table': n, **_fidelity(rows, actual, [types[x.lower()] for x in cols])})
                out['fidelity'] = fids
                invalid_fk = any(x.get('violations', 0) > 0 for x in checks)
                bad_policy = any(x['status'] not in ['enforced', 'advisory_source_violated', 'advisory_malformed']
                                 for x in checks)
                bad_fidelity = any(not x['ok'] for x in fids)
                out['failed_checks'] = ((['invalid_foreign_key'] if invalid_fk else [])
                                        + (['unresolved_policy'] if bad_policy else [])
                                        + (['unfaithful'] if bad_fidelity else []))
                if out['failed_checks']:
                    out['reason'] = out['failed_checks'][0]
                    return out
                u = sqlite3.connect(twin)
                try:
                    for typ, name, table, ddl in schema:
                        if typ == 'table':
                            cols = [r[1] for r in c.execute('PRAGMA table_info(' + _quote(name) + ')')]
                            u.execute('CREATE TABLE ' + _quote(name) + ' (' + ','.join(map(_quote, cols)) + ')')
                    for typ, name, table, ddl in schema:
                        if typ == 'view':
                            u.execute(ddl)
                    for n, (cols, rows) in expected.items():
                        u.executemany('INSERT INTO ' + _quote(names[n]) + ' (' + ','.join(map(_quote, cols))
                                      + ') VALUES (' + ','.join('?' for col in cols) + ')', rows)
                    u.commit()
                finally:
                    u.close()
            finally:
                c.close()
            results = {}
            for label, path in [('typed', typed), ('twin', twin)]:
                results[label] = {}
                for variant, pair in case['variants'].items():
                    opts = {"timeout_seconds": 5, "max_rows": 100000, "max_bytes": 64000000, "truncate": False}
                    try:
                        g = sql.ReadOnlySQLite(path, **opts).execute(pair['gold_sql'])
                        p = sql.ReadOnlySQLite(path, **opts).execute(pair['pred_sql'])
                    except SQLiteInfrastructureError:
                        raise
                    except (SQLiteQueryTimeout, SQLiteResultLimitExceeded):
                        raise  # Classified by the execution-limit handler below.
                    except SQLiteQueryError as error:
                        # These queries ran on the source DB; this synthetic world
                        # may nevertheless cause arithmetic overflow, etc. Only
                        # catch execution errors here, not adapter/comparator bugs.
                        out.update(reason='witness_query_error', error=str(error),
                                   materialization=label, variant=variant)
                        return out
                    v = sql.compare_results(g, p, method=case['grading_method'])
                    results[label][variant] = {'gold': g.rows, 'pred': p.rows, 'distinguishes': not v.correct,
                                               'reason': v.message}
            out['results'] = results
            for variant in case['variants']:
                a = results['typed'][variant]
                b = results['twin'][variant]

                def output_key(rows):
                    if case['grading_method'] in ['list', 'ordered']:
                        return tuple(tuple((type(v).__name__, repr(v)) for v in row) for row in rows)
                    return _typed_bag(rows)
                if (output_key(a['gold']) != output_key(b['gold']) or output_key(a['pred']) != output_key(b['pred'])
                        or a['distinguishes'] != b['distinguishes']):
                    out['reason'] = 'twin_changes_results_or_verdict'
                    return out
            verdicts = {x['distinguishes'] for x in results['typed'].values()}
            if len(verdicts) != 1:
                out['reason'] = 'wrapper_changes_verdict'
                return out
            if results['typed']['original']['distinguishes']:
                out.update(decision='penalize', reason='validated_counterexample')
            else:
                out['reason'] = 'grader_matches'
    except (SQLiteQueryTimeout, SQLiteResultLimitExceeded) as e:
        out.update(reason='witness_execution_limit', error=type(e).__name__ + ': ' + str(e))
    return out


def load_policy(path, expected_sha256, tasks):
    """Read the gate policy and check it covers every training database at its pinned SHA-256.

    The policy has one entry per database: its `sha256`, its `schema` (the `type, name, tbl_name, sql` rows of
    `sqlite_master`) and `policies`, the audited status of each foreign key. A task finds its entry by the database's
    pinned digest (`database_sha256`), so the policy holds wherever the databases are stored; a task without a digest
    finds it by the entry's key, its `database_path`. Returns the entries keyed by each task's `database_path`.
    Rejected witnesses are abstentions, never equivalence.
    """
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Counterexample policy hash mismatch')
    policy = json.loads(raw)
    by_digest = {info['sha256']: info for info in policy.values()}
    covered = {}
    for task in tasks:
        database = str(task['database_path'])
        if database in covered:
            continue
        pinned = task.get('database_sha256')
        info = by_digest.get(pinned) if pinned else policy.get(database)
        if info is None:
            raise ValueError(f'the counterexample policy has no entry for the database {database}')
        digest = hashlib.sha256()
        with open(database, 'rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != info['sha256']:
            raise ValueError('Counterexample policy database mismatch: ' + database)
        covered[database] = info
    return covered


def confirm(task, prediction, witness, info):
    """Decide whether an extracted witness confirms a refutation: "penalize" or "abstain", with the reason."""
    if "error" in witness:
        raise VeriEQLError("Counterexample extraction reported failure: " + str(witness["error"]))
    if witness.get('result') is not False or not witness.get('attested'):
        return {'decision': 'abstain', 'reason': 'no_attested_witness'}
    if witness.get('decode_issues'):
        return {'decision': 'abstain', 'reason': 'decode_issues'}
    if 'tables' not in witness:
        return {'decision': 'abstain', 'reason': 'missing_witness_tables'}
    # sqlglot lists the tables each query reads. It is needed only from here on, and only with the gate on.
    import sqlglot
    from sqlglot import exp
    try:
        names = {n.lower() for typ, n, table, ddl in info['schema'] if typ == 'table'}
        refs = set()
        for query in [task['gold_sql'], prediction]:
            refs.update(t.name.lower() for t in sqlglot.parse_one(query, read='sqlite').find_all(exp.Table))
        if not refs & names:
            return {'decision': 'abstain', 'reason': 'no_resolved_source_tables'}
        case = {
            'referenced_tables': sorted(refs & names),
            'grading_method': task.get('grading_method', 'set'),
            'variants': {'original': {'gold_sql': task['gold_sql'], 'pred_sql': prediction}},
        }
        sql = SimpleNamespace(
            ReadOnlySQLite=sqlite_exec.ReadOnlySQLite,
            compare_results=functools.partial(
                sqlite_exec.compare_results,
                numeric_policy=task.get('numeric_policy', sqlite_exec.DIAMOND_NUMERIC_POLICY),
            ),
        )
        policies = {(x['table'].lower(), x['id']): x for x in info['policies']}
        return refutation_gate(case, witness['tables'], info['schema'], policies, sql,
                               dedupe_hypothesis=True)
    except (sqlglot.errors.ParseError, RecursionError) as error:   # sqlglot recurses on deeply nested SQL
        return {'decision': 'abstain', 'reason': 'unresolved_query', 'error': type(error).__name__ + ': ' + str(error)}


class GatedNativeVerifier(NativeVerifier):
    """`NativeVerifier` whose refutations count only when an executed counterexample confirms them."""

    def __init__(self, *args, gate_policy, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_policy = gate_policy
        self.extraction_script = _EXTRACTION_SCRIPT

    async def check(self, task, prediction):
        started = time.monotonic()
        original = await super().check(task, prediction)
        if original['verdict'] != 'refuted':
            return original
        async with self.slots:
            row = await asyncio.to_thread(_input_row, task, prediction,
                                          bound=PAPER_BOUND)
            row['source_root'] = self.manifest['source_root']
            completed = await self._attempt(row, self.timeout_seconds, self.extraction_script)
            witness = None
            if completed is None:
                gate = {'decision': 'abstain', 'reason': 'witness_timeout'}
            elif completed.returncode:
                # Runtime or attestation failure must not silently award reward.
                raise VeriEQLError('Counterexample extraction failed: ' + completed.stderr[-2000:])
            else:
                witness = json.loads(completed.stdout.strip().splitlines()[-1])
                gate = await asyncio.to_thread(confirm, task, prediction, witness,
                                               self.gate_policy[str(task['database_path'])])
            result = {**original, 'ungated_verdict': original['verdict'],
                      'counterexample_gate': gate,
                      'gate_total_seconds': time.monotonic() - started}
            if gate['decision'] != 'penalize':
                result.update(verdict='unknown', state='UNKNOWN',
                              refutation_route='unvalidated_counterexample')
            self.counts['gate_' + gate['decision']] += 1
            (self.receipt_dir / ('gate-' + uuid.uuid4().hex + '.json')).write_text(
                json.dumps({'task_id': task['id'], 'prediction': prediction,
                            'gold_sql': task['gold_sql'], 'witness': witness,
                            'outcome': result}) + '\n')
            return result


@runtime_checkable
class Verifier(Protocol):
    """What the environment needs from an answer verifier."""

    async def check(self, task: Mapping[str, Any], prediction: str) -> dict[str, Any]:
        """The verdict for one result-matched answer; raises when no valid check was completed."""
        ...

    async def close(self) -> None:
        """Stop any check still running."""
        ...


_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class VeriEQLSetup:
    """A validated VeriEQL runtime, checked once per run; each epoch's verifier is built from it."""

    manifest: dict[str, Any]
    manifest_sha256: str
    workers: int
    gate_policy: dict[str, Any] | None = None  # None: every refutation counts (no executed-counterexample gate)

    def verifier(self, receipt_dir: str | os.PathLike[str]) -> NativeVerifier:
        """A new verifier that writes its receipts under `receipt_dir` (one directory per epoch)."""
        if self.gate_policy is None:
            return NativeVerifier(self.manifest, self.manifest_sha256, receipt_dir, workers=self.workers)
        return GatedNativeVerifier(self.manifest, self.manifest_sha256, receipt_dir, workers=self.workers,
                                   gate_policy=self.gate_policy)


def require_runtime(reward: RewardConfig, environ: Mapping[str, str] | None = None) -> tuple[str, str] | None:
    """The VeriEQL runtime manifest's path and pinned SHA-256; None when `reward.verifier` is "none".

    Checks only that the variables are set and well formed and that the manifest exists, so a run can stop at once
    on a missing runtime; `setup_verifier` validates the runtime itself.
    """
    if reward.verifier == "none":
        return None
    if reward.verifier != "verieql":
        raise ValueError(f"unknown reward.verifier: {reward.verifier!r}")
    if not 0 < reward.verifier_weight <= 1:
        raise ValueError("reward.verifier = 'verieql' needs reward.verifier_weight in (0, 1]; "
                         "set reward.verifier = 'none' to train without the verifier")
    environ = os.environ if environ is None else environ
    manifest_path = environ.get(MANIFEST_ENV_VAR, "")
    manifest_sha256 = environ.get(MANIFEST_SHA256_ENV_VAR, "")
    if not manifest_path or not manifest_sha256:
        raise VeriEQLError(
            f"reward.verifier = 'verieql' needs a VeriEQL runtime: set {MANIFEST_ENV_VAR} to its runtime manifest and "
            f"{MANIFEST_SHA256_ENV_VAR} to the manifest's SHA-256 (`python -m text_to_sql_recipe setup verieql` "
            "prints both), or set reward.verifier = 'none'. VeriEQL is licensed separately (CC BY-NC-SA 4.0) and is "
            "not installed with this package."
        )
    if not _SHA256_HEX.fullmatch(manifest_sha256):
        raise ValueError(f"{MANIFEST_SHA256_ENV_VAR} must be 64 lowercase hex digits")
    if not Path(manifest_path).is_file():
        raise VeriEQLError(f"{MANIFEST_ENV_VAR} names {manifest_path}, which does not exist")
    return manifest_path, manifest_sha256


def setup_verifier(
    reward: RewardConfig,
    execution: ExecutionConfig,
    *,
    tasks: Sequence[Mapping[str, Any]] = (),
    gate_policy_path: str | os.PathLike[str] | None = None,
    gate_policy_sha256: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> VeriEQLSetup | None:
    """Validate the configured verifier once per run; None when `reward.verifier` is "none".

    With `reward.verifier = "verieql"` this never returns None: a missing or invalid runtime raises, so a run cannot
    silently train without the shaping it was configured with. `tasks` are the training tasks, whose databases the
    gate policy must cover; `gate_policy_path` and `gate_policy_sha256` locate the dataset's gate policy file (both
    needed only with `reward.verifier_gate`). The runtime manifest is located through `environ` (default
    `os.environ`), see the module docstring.
    """
    located = require_runtime(reward, environ)
    if located is None:
        return None
    manifest_path, manifest_sha256 = located
    gate_policy = None
    if reward.verifier_gate:
        if not gate_policy_path or not gate_policy_sha256:
            raise ValueError("reward.verifier_gate needs the dataset's gate policy file and its SHA-256")
        if not _SHA256_HEX.fullmatch(gate_policy_sha256):
            raise ValueError("the gate policy SHA-256 must be 64 lowercase hex digits")
        if not tasks:
            raise ValueError("reward.verifier_gate needs the training tasks to check the gate policy against")
        try:
            importlib.import_module("sqlglot")
        except ImportError as error:
            raise VeriEQLError(
                "reward.verifier_gate needs the sqlglot package (it lists the tables each query reads); install it "
                "alongside VeriEQL, or set reward.verifier_gate = false"
            ) from error
        gate_policy = load_policy(gate_policy_path, gate_policy_sha256, tasks)
    manifest = validate_manifest(manifest_path, manifest_sha256)
    return VeriEQLSetup(manifest, manifest_sha256, execution.verifier_workers, gate_policy)


def make_verifier(
    reward: RewardConfig,
    execution: ExecutionConfig,
    receipt_dir: str | os.PathLike[str],
    **locations: Any,
) -> Verifier | None:
    """`setup_verifier(reward, execution, **locations)`, then the verifier for `receipt_dir`; None for "none"."""
    setup = setup_verifier(reward, execution, **locations)
    return None if setup is None else setup.verifier(receipt_dir)
