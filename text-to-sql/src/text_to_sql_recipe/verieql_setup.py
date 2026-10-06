"""`python -m text_to_sql_recipe setup verieql`: install VeriEQL for the recipe's verifier, outside this package.

VeriEQL is licensed separately from this recipe, under CC BY-NC-SA 4.0 (non-commercial), and this package never
bundles or imports it. The command shows that licence and requires it to be accepted (`--accept-license`, or "yes" at
the prompt) before it downloads anything. Then it

1. fetches VeriEQL at the pinned revision (`verifier.VERIEQL_REVISION`) from GitHub into `<prefix>/source`, and keeps
   it only if its file tree hashes to `verifier.VERIEQL_SOURCE_SHA256`;
2. builds an isolated Python 3.11 environment in `<prefix>/venv` with the packages VeriEQL's checks load, at the
   versions VeriEQL's own container build (the stock release) has
   (`WHEELS` and `SOURCES`, each pinned by version and SHA-256), and copies VeriEQL's z3py_libs over the z3 package's
   Python files, as the stock release has them. With uv on PATH, uv also supplies the interpreter (CPython
   `PYTHON_VERSION`, kept in `<prefix>/python`); `--python` names a CPython 3.11 interpreter to use instead, and is
   required without uv, in which case pip installs;
3. writes the runtime manifest of this installation for this machine's platform contract (`verifier.RUNTIME_CONTRACTS`)
   to `<prefix>/runtime-manifest.json`, and validates it as a training run does;
4. runs a self-check through the recipe's own verifier: a handful of equivalence checks with fixed expected verdicts
   (`SELF_CHECKS`);
5. prints the two environment variables that point the recipe at the manifest.

Progress goes to stderr and only the `export` lines to stdout, so `eval "$(python -m text_to_sql_recipe setup verieql
--accept-license)"` sets them in the current shell. Running the command again reuses what is already in place once it
verifies, and writes the manifest again (needed after an operating-system update, for example); a directory it cannot
verify is reported, never replaced. It removes nothing but its own temporary files.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from . import sqlite_exec, verifier

VERIEQL_REPOSITORY = "https://github.com/VeriEQL/VeriEQL"
SOURCE_ARCHIVE_URL = f"{VERIEQL_REPOSITORY}/archive/{verifier.VERIEQL_REVISION}.tar.gz"
SOURCE_FILE_COUNT = 248                    # files in the pinned revision
SOURCE_ARCHIVE_MAX_BYTES = 64 * 1024 * 1024
LICENSE_NAME = "CC BY-NC-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
PYTHON_VERSION = "3.11.14"                 # the CPython release uv supplies (the stock release runs Python 3.11)

# The packages VeriEQL's checks load, at the stock release's versions: (distribution, version, SHA-256 of every
# accepted file). WHEELS install from pre-built wheels only; the accepted files are the CPython 3.11 wheels for macOS
# arm64 and manylinux aarch64 and x86_64, and pure-Python wheels.
WHEELS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("lark", "1.3.1", ("c629b661023a014c37da873b4ff58a817398d12635d3bbb2c5a03be7fe5d1e12",)),
    ("numpy", "2.3.4", (
        "7fd09cc5d65bda1e79432859c40978010622112e9194e581e3415a3eccc7f43f",
        "1b219560ae2c1de48ead517d085bc2d05b9433f8e49d0955c82e8cd37bd7bf36",
        "36dc13af226aeab72b7abad501d370d606326a0029b9f435eacb3b8c94b8a8b7",
        "a7b2f9a18b5ff9824a6af80de4f37f4ec3c2aab05ef08f51c77a093f5b89adda",
    )),
    ("ordered-set", "4.1.0", ("046e1132c71fcf3330438a539928932caf51ddbc582496833e23de611de14562",)),
    ("pandas", "2.3.3", (
        "8fe25fc7b623b0ef6b5009149627e34d2a4657e880948ec3c840e9402e5c1b45",
        "b468d3dad6ff947df92dcb32ede5b7bd41a9b3cceef0a30ed925f6d01fb8fa66",
        "b98560e98cb334799c0b07ca7967ac361a47326e9b4e5a7dfb5ab2b1c9d35a1b",
    )),
    ("prettytable", "3.16.0", ("b5eccfabb82222f5aa46b798ff02a8452cf530a352c31bddfa29be41242863aa",)),
    ("python-dateutil", "2.9.0.post0", ("a8b2bc7bffae282281c8140a97d3aa9c14da0b136dfe83f850eea9a5f7470427",)),
    ("pytz", "2025.2", ("5ddf76296dd8c44c26eb8f4b6f35488f3ccbf6fbbd7adee0b7262d43f0ec2f00",)),
    ("pyyaml", "6.0.3", (
        "652cb6edd41e718550aad172851962662ff2681490a8a711af6a4d288dd96824",
        "10892704fc220243f5305762e276552a0395f7beb4dbf9b14ec8fd43b57f126c",
        "b8bb0864c5a28024fac8a632c443c87c5aa6f215c0b126c449ae1a150412f31d",
    )),
    # z3core.py imports pkg_resources, which setuptools 81 removes.
    ("setuptools", "80.9.0", ("062d34222ad13e0cc312a4c02d73f059e86a4acbfbdea8f8f76b28c99f306922",)),
    ("six", "1.17.0", ("4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274",)),
    ("tqdm", "4.67.1", ("26445eca388f82e72884e0d580d5464cd801a3ea01e63e5601bdff9ba6a48de2",)),
    ("tzdata", "2025.2", ("1a403fada01ff9221ca8044d701868fa132215d84beb92242d9acd2147f667a8",)),
    ("ujson", "5.11.0", (
        "d8951bb7a505ab2a700e26f691bdfacf395bc7e3111e3416d325b513eea03a58",
        "952c0be400229940248c0f5356514123d428cba1946af6fa2bbd7503395fef26",
        "7e0ec1646db172beb8d3df4c32a9d78015e671d2000af548252769e33079d9a6",
    )),
    ("wcwidth", "0.2.14", ("a7bb560c8aee30f9957e5f9895805edd20602f2d7f720186dfd906e82b4982e1",)),
    ("z3-solver", "4.13.0.0", (
        "bca7d59a699a440247537c2180c519d682c9df3520a16ce288fced61a70d253d",
        "9d622022a3511c059915c56b2c231c84b5c1be1b82f457d7560dda3d916474fe",
        "8c42de82b6e3ff7ee61287d03c7af8a99f9f6554cdd1204c6b9bca96ff1cb7fb",
    )),
)
# VeriEQL's SQL parser and its helpers, published only as source archives of pure Python. They build with the
# environment's own setuptools (pinned above), so nothing unpinned enters the build.
SOURCES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("mo-dots", "9.189.22201", ("cb332611b9662499d564400612f9b3a5c90a2b473128e2068224a34f2d2bf3c9",)),
    ("mo-future", "6.2.21303", ("1594c1c126b4a7fabc01867ea7be181bcda2b32ac21065dae75b02c19c54b816",)),
    ("mo-imports", "7.187.22201", ("a9779b9e63017b68fb079df0158b772de300770722e51497f5bf64476802a893",)),
    ("mo-parsing", "8.204.22252", ("df1ef04b4c6ac036295992add7d6e384b10ae594d6a81c3b6e526efe4829286b",)),
    ("mo-sql-parsing", "8.205.22260", ("d7e244cd05a9cfdca214c4a326ff4ab4e12145167ceeb629c1a10c7e612c50bb",)),
)


class SetupError(RuntimeError):
    """Setup could not produce a verified VeriEQL runtime; the message says what to do."""


class LicenseNotAccepted(SetupError):
    """VeriEQL's licence was not accepted, so nothing was downloaded."""


# --------------------------------------------------------------------------------------------------------------------
# Self-check: equivalence checks with fixed expected verdicts, run through the recipe's own verifier.


@dataclass(frozen=True)
class SelfCheck:
    name: str
    gold_sql: str
    prediction: str
    expected: Mapping[str, Any]   # verdict fields that must match exactly; "gate" is [decision, reason]
    gated: bool = False           # through the executed-counterexample gate, which needs sqlglot


SELF_CHECK_SCHEMA = (
    "CREATE TABLE orders (id INTEGER, customer TEXT, amount INTEGER, status TEXT);"
    "CREATE TABLE customers (name TEXT, city TEXT);"
)
# Each expected verdict follows from the queries alone, whichever model the solver finds: the gate's two pairs are
# chosen so that every counterexample to the first changes the executed result (the counts differ by the rows with
# amount 10) and no counterexample to the second does (the two differ only in duplicate rows, which set grading
# ignores).
SELF_CHECKS: tuple[SelfCheck, ...] = (
    SelfCheck("equivalent rewrite", "SELECT id FROM orders WHERE amount > 10",
              "SELECT id FROM orders WHERE 10 < amount", {"verdict": "confirmed", "state": "EQU"}),
    SelfCheck("equivalent join", "SELECT o.id FROM orders AS o JOIN customers AS c ON o.customer = c.name "
                                 "WHERE c.city = 'Paris'",
              "SELECT orders.id FROM customers JOIN orders ON customers.name = orders.customer "
              "WHERE customers.city = 'Paris'", {"verdict": "confirmed", "state": "EQU"}),
    SelfCheck("boundary refutation", "SELECT id FROM orders WHERE amount > 10",
              "SELECT id FROM orders WHERE amount >= 10",
              {"verdict": "refuted", "state": "NEQ", "refutation_route": "solver_model", "counterexample": True}),
    SelfCheck("projection arity", "SELECT id FROM orders", "SELECT id, amount FROM orders",
              {"verdict": "refuted", "state": "NEQ", "refutation_route": "column_arity"}),
    SelfCheck("unsupported query", "SELECT COUNT(*) FROM orders", "SELECT 1",
              {"verdict": "inconclusive", "state": "UNKNOWN"}),
    SelfCheck("gate: executed counterexample", "SELECT COUNT(*) FROM orders WHERE amount > 10",
              "SELECT COUNT(*) FROM orders WHERE amount >= 10",
              {"verdict": "refuted", "state": "NEQ", "gate": ["penalize", "validated_counterexample"]}, gated=True),
    SelfCheck("gate: grader agrees", "SELECT DISTINCT customer FROM orders", "SELECT customer FROM orders",
              {"verdict": "unknown", "state": "UNKNOWN", "ungated_verdict": "refuted",
               "gate": ["abstain", "grader_matches"]}, gated=True),
)


def _observed(verdict: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    gate = verdict.get("counterexample_gate")
    return {key: ([gate.get("decision"), gate.get("reason")] if gate else None) if key == "gate" else verdict.get(key)
            for key in keys}


def run_self_check(manifest: Mapping[str, Any], manifest_sha256: str, work_dir: Path,
                   checks: Sequence[SelfCheck] = SELF_CHECKS) -> list[dict[str, Any]]:
    """Run `checks` through `verifier.VeriEQLSetup`; one result per check, with "ok", or "skipped" and why."""
    work_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=work_dir))
    database = run_dir / "self-check.sqlite"
    with closing(sqlite3.connect(database)) as connection:
        connection.executescript(SELF_CHECK_SCHEMA)
        schema = [list(row) for row in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'")]
    gate_policy = {str(database): {"sha256": _sha256_file(database), "schema": schema, "policies": []}}
    gate_available = importlib.util.find_spec("sqlglot") is not None
    plain = verifier.VeriEQLSetup(dict(manifest), manifest_sha256, workers=4).verifier(run_dir / "receipts")
    gated = (verifier.VeriEQLSetup(dict(manifest), manifest_sha256, workers=4, gate_policy=gate_policy)
             .verifier(run_dir / "gate-receipts") if gate_available else None)

    async def one(check: SelfCheck) -> dict[str, Any]:
        result: dict[str, Any] = {"name": check.name, "expected": dict(check.expected)}
        if check.gated and gated is None:
            return {**result, "skipped": "the executed-counterexample gate needs sqlglot (pip install .[verieql])"}
        task = {"id": "self-check/" + check.name, "database_path": str(database), "gold_sql": check.gold_sql,
                "grading_method": "set", "numeric_policy": sqlite_exec.DIAMOND_NUMERIC_POLICY}
        try:
            verdict = await (gated if check.gated else plain).check(task, check.prediction)
        except Exception as error:  # noqa: BLE001 -- a broken runtime is a failed check, reported with the others
            return {**result, "ok": False, "error": f"{type(error).__name__}: {error}"}
        observed = _observed(verdict, list(check.expected))
        return {**result, "ok": observed == dict(check.expected), "observed": observed}

    async def everything() -> list[dict[str, Any]]:
        try:
            return list(await asyncio.gather(*(one(check) for check in checks)))
        finally:
            await plain.close()
            if gated is not None:
                await gated.close()

    results = asyncio.run(everything())
    (run_dir / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    return results


# --------------------------------------------------------------------------------------------------------------------
# The licence.


def license_notice(prefix: Path) -> str:
    revision = verifier.VERIEQL_REVISION
    return (
        "VeriEQL is not part of this recipe. It is licensed separately, under the Creative Commons\n"
        f"Attribution-NonCommercial-ShareAlike 4.0 International licence ({LICENSE_NAME}):\n"
        f"  {LICENSE_URL}\n"
        "In short, it may be used and adapted for non-commercial purposes only, with credit to its authors,\n"
        "and adaptations you share must carry the same licence. The licence text, not this summary, is what\n"
        "applies: make sure your use fits it before you accept.\n"
        "\n"
        f"Accepting downloads VeriEQL revision {revision} from\n"
        f"{VERIEQL_REPOSITORY} (its licence: {VERIEQL_REPOSITORY}/blob/{revision}/license.md)\n"
        "and installs it, with z3-solver and VeriEQL's other dependencies from PyPI (each under its own\n"
        f"licence), into {prefix}\n"
    )


def accept_license(accepted: bool, prefix: Path, *, stdin: TextIO, stderr: TextIO) -> str:
    """How the licence was accepted ("flag" or "prompt"); raises `LicenseNotAccepted` otherwise."""
    stderr.write(license_notice(prefix) + "\n")
    if accepted:
        stderr.write("Accepted with --accept-license.\n\n")
        return "flag"
    if not stdin.isatty():
        raise LicenseNotAccepted("VeriEQL's licence was not accepted, so nothing was downloaded. Pass "
                                 "--accept-license to accept it without a prompt.")
    stderr.write(f"Accept the VeriEQL licence ({LICENSE_NAME}) and install it? Type 'yes' to accept: ")
    stderr.flush()
    if stdin.readline().strip().lower() != "yes":
        raise LicenseNotAccepted("VeriEQL's licence was not accepted, so nothing was downloaded.")
    stderr.write("\n")
    return "prompt"


# --------------------------------------------------------------------------------------------------------------------
# Fetching and verifying the source.


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_entries(root: Path) -> dict[str, str]:
    """{relative path: SHA-256} for every file under `root`, leaving out bytecode caches; refuses links."""
    entries = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise SetupError(f"unexpected symbolic link in the VeriEQL source: {path}")
        if path.is_file():
            entries[relative.as_posix()] = _sha256_file(path)
    return entries


def _download(url: str, destination: Path, *, max_bytes: int = SOURCE_ARCHIVE_MAX_BYTES,
              opener: Callable[..., Any] = urllib.request.urlopen) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "text-to-sql-recipe-setup"})
    partial = destination.with_name(destination.name + ".partial")
    total = 0
    with opener(request, timeout=120) as response, open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            total += len(chunk)
            if total > max_bytes:
                raise SetupError(f"the VeriEQL archive at {url} is larger than {max_bytes} bytes")
            out.write(chunk)
    os.replace(partial, destination)


def extract_verified(archive: Path, destination: Path, *, expected_sha256: str | None = None,
                     file_count: int | None = None) -> None:
    """Unpack the archive's single top-level directory into `destination` if its files match the pinned tree."""
    expected_sha256 = verifier.VERIEQL_SOURCE_SHA256 if expected_sha256 is None else expected_sha256
    file_count = SOURCE_FILE_COUNT if file_count is None else file_count
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        tops = {member.name.split("/", 1)[0] for member in members}
        if len(tops) != 1:
            raise SetupError(f"the VeriEQL archive should hold one top-level directory, not {sorted(tops)}")
        top = tops.pop()
        files = {}
        for member in members:
            relative = member.name[len(top) + 1:]
            if member.isdir():
                continue
            if not member.isfile() or not relative or relative.startswith("/") or ".." in relative.split("/"):
                raise SetupError(f"unexpected entry in the VeriEQL archive: {member.name}")
            files[relative] = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
        if len(files) != file_count or verifier.source_tree_sha256(files) != expected_sha256:
            raise SetupError(f"the downloaded VeriEQL source does not match revision {verifier.VERIEQL_REVISION} "
                             f"({len(files)} files, tree SHA-256 {verifier.source_tree_sha256(files)}); "
                             "nothing was installed")
        with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
            tar.extractall(staging, filter="data")
            if tree_entries(Path(staging) / top) != files:
                raise SetupError("the extracted VeriEQL source differs from the verified archive")
            os.replace(Path(staging) / top, destination)


def fetch_source(prefix: Path, *, url: str = SOURCE_ARCHIVE_URL, log: Callable[[str], None] = print,
                 opener: Callable[..., Any] = urllib.request.urlopen) -> Path:
    """`<prefix>/source`, holding exactly the pinned revision: verified in place, or downloaded and verified."""
    source = prefix / "source"
    if source.exists():
        entries = tree_entries(source)
        if verifier.source_tree_sha256(entries) != verifier.VERIEQL_SOURCE_SHA256:
            raise SetupError(f"{source} does not hold VeriEQL revision {verifier.VERIEQL_REVISION}; move it away or "
                             "choose another --prefix")
        log(f"Verified the VeriEQL source already in {source}")
        return source
    archive = prefix / "downloads" / f"VeriEQL-{verifier.VERIEQL_REVISION}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        log(f"Downloading {url}")
        _download(url, archive, opener=opener)
    extract_verified(archive, source)
    log(f"Verified VeriEQL revision {verifier.VERIEQL_REVISION} (tree SHA-256 {verifier.VERIEQL_SOURCE_SHA256}) "
        f"in {source}")
    return source


# --------------------------------------------------------------------------------------------------------------------
# The environment.


def requirements_text(entries: Sequence[tuple[str, str, Sequence[str]]]) -> str:
    """Hash-pinned requirement lines, as pip and uv read them with --require-hashes."""
    return "".join(f"{name}=={version}" + "".join(f" --hash=sha256:{digest}" for digest in digests) + "\n"
                   for name, version, digests in entries)


def _run(command: Sequence[str], *, env: Mapping[str, str] | None = None, cwd: Path | None = None,
         what: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run([str(part) for part in command], env=None if env is None else dict(env), cwd=cwd,
                               capture_output=True, text=True, check=False)
    if completed.returncode:
        detail = "\n".join((completed.stderr or completed.stdout).strip().splitlines()[-15:])
        raise SetupError(f"{what} failed (exit {completed.returncode}): {shlex.join(map(str, command))}\n{detail}")
    return completed


def _interpreter(python: str | os.PathLike[str]) -> dict[str, str]:
    completed = _run([python, "-I", "-c", ("import json, platform; print(json.dumps({'implementation': "
                      "platform.python_implementation(), 'version': platform.python_version()}))")],
                     what=f"Inspecting the Python interpreter {python}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _require_python_series(python: str | os.PathLike[str], what: str) -> dict[str, str]:
    found = _interpreter(python)
    if found["implementation"] != "CPython" or not found["version"].startswith(verifier.PYTHON_SERIES + "."):
        raise SetupError(f"{what} {python} is {found['implementation']} {found['version']}; VeriEQL's environment "
                         f"needs CPython {verifier.PYTHON_SERIES}")
    return found


def check_prerequisites(python: str | None, uv: str | None) -> None:
    """Before the licence prompt and any download: something must supply the environment's Python 3.11."""
    if python is None and uv is None:
        raise SetupError("setting up VeriEQL needs uv (https://docs.astral.sh/uv/), which also supplies Python "
                         f"{PYTHON_VERSION}, or --python naming a CPython {verifier.PYTHON_SERIES} interpreter")
    if python is not None:
        _require_python_series(python, "--python")


def build_environment(prefix: Path, *, python: str | None, uv: str | None,
                      log: Callable[[str], None] = print) -> Path:
    """Create or reuse `<prefix>/venv` with the pinned dependencies; returns its interpreter."""
    check_prerequisites(python, uv)
    venv = prefix / "venv"
    venv_python = venv / "bin" / "python"
    if venv.exists():
        try:
            _require_python_series(venv_python, "The existing environment's interpreter")
        except SetupError as error:
            raise SetupError(f"{error}\nThe environment in {venv} is not usable; move it away or choose another "
                             "--prefix") from error
        log(f"Reusing the environment in {venv}")
    elif uv is not None:
        if python is None:
            install_dir = prefix / "python"
            log(f"Installing CPython {PYTHON_VERSION} with uv into {install_dir}")
            _run([uv, "python", "install", PYTHON_VERSION, "--install-dir", install_dir, "--no-bin"],
                 what="Installing Python with uv")
            found = _run([uv, "python", "find", "--managed-python", "--no-config", PYTHON_VERSION],
                         env={**os.environ, "UV_PYTHON_INSTALL_DIR": str(install_dir)},
                         what="Locating the uv-installed Python")
            python = found.stdout.strip().splitlines()[-1]
            if not Path(python).resolve().is_relative_to(install_dir.resolve()):
                raise SetupError(f"uv found {python} instead of the interpreter it installed in {install_dir}")
        log(f"Creating the environment in {venv}")
        _run([uv, "venv", "--python", python, "--no-project", "--no-config", venv], what="Creating the environment")
    else:
        log(f"Creating the environment in {venv}")
        _run([python, "-m", "venv", venv], what="Creating the environment")
    requirements = prefix / "requirements"
    requirements.mkdir(parents=True, exist_ok=True)
    for name, entries, sources in (("wheels.txt", WHEELS, False), ("sources.txt", SOURCES, True)):
        path = requirements / name
        path.write_text(requirements_text(entries))
        if uv is not None:
            command = [uv, "pip", "install", "--python", venv_python, "--no-deps", "--require-hashes",
                       "--link-mode", "copy", "-r", path]
            command += ["--no-build-isolation", "--no-binary", ":all:"] if sources else ["--no-build"]
        else:
            command = [venv_python, "-m", "pip", "install", "--no-deps", "--require-hashes", "--no-input",
                       "--disable-pip-version-check", "-r", path]
            command += ["--no-build-isolation", "--no-binary", ":all:"] if sources else ["--only-binary", ":all:"]
        log(f"Installing the pinned {'source-only' if sources else 'wheel'} dependencies ({len(entries)})")
        _run(command, what="Installing VeriEQL's dependencies")
    return venv_python


def apply_z3_overlay(source: Path, venv_python: Path) -> Path:
    """Copy VeriEQL's z3py_libs over the z3 package's Python files, as the stock release has them; returns z3's dir."""
    completed = _run([venv_python, "-I", "-c", ("import importlib.util; "
                      "print(importlib.util.find_spec('z3').submodule_search_locations[0])")],
                     what="Locating the z3 package")
    z3_root = Path(completed.stdout.strip().splitlines()[-1])
    for library in sorted((source / "z3py_libs").glob("*.py")):
        target = z3_root / library.name
        if target.is_file() and target.read_bytes() == library.read_bytes():
            continue
        # Write beside the target and rename over it: an installer may have linked the target to a shared cache.
        with tempfile.NamedTemporaryFile(dir=z3_root, delete=False) as staged:
            staged.write(library.read_bytes())
        os.replace(staged.name, target)
        os.chmod(target, 0o644)
    return z3_root


def compile_bytecode(venv_python: Path, roots: Sequence[Path], log: Callable[[str], None] = print) -> None:
    """Precompile with checked-hash caches: each check starts a fresh interpreter, which never writes bytecode."""
    completed = subprocess.run([str(venv_python), "-I", "-m", "compileall", "-q", "-j", "0", "--invalidation-mode",
                                "checked-hash", *map(str, roots)], capture_output=True, text=True, check=False)
    if completed.returncode:
        log("Some files did not precompile (they are compiled when imported instead): "
            + " ".join((completed.stdout + completed.stderr).split()[-20:]))


# --------------------------------------------------------------------------------------------------------------------
# The runtime manifest.


def _exercise() -> str:
    """Probe code that runs the recipe's own equivalence check and witness extraction once, in process.

    The probe lists the files the interpreter has loaded when it reports, so running the real checks first makes the
    inventory include everything they load lazily (solver libraries, codecs, locale data).
    """
    return """
import contextlib as _contextlib, io as _io
_row = {"schema": {"t": {"v": "INT", "w": "VARCHAR"}}, "bound_size": 2,
        "pred_sql": "SELECT v FROM t WHERE v >= 1", "gold_sql": "SELECT v FROM t WHERE v > 1",
        "source_root": str(pathlib.Path(sys.argv[1]).resolve()),
        "z3_root": str(pathlib.Path(z3.__file__).resolve().parent),
        "stdlib_root": str(pathlib.Path(sysconfig.get_path('stdlib')).resolve())}
_outputs = []
for _script in (""" + repr(verifier._HELPER_SCRIPT) + ", " + repr(verifier._EXTRACTION_SCRIPT) + """):
    _stdin, sys.stdin, _out = sys.stdin, _io.StringIO(json.dumps(_row)), _io.StringIO()
    try:
        with _contextlib.redirect_stdout(_out):
            exec(_script, {'__name__': '__main__'})
    except SystemExit as _exit:
        raise RuntimeError('the VeriEQL helper failed with exit code ' + str(_exit.code))
    finally:
        sys.stdin = _stdin
    _outputs.append(json.loads(_out.getvalue().strip().splitlines()[-1]))
if _outputs[0].get('result') is not False or not _outputs[1].get('attested'):
    raise RuntimeError('VeriEQL did not refute a known non-equivalent pair: ' + json.dumps(_outputs))
"""


def probe_runtime(contract: verifier.RuntimeContract, venv_python: Path, source: Path) -> dict[str, Any]:
    """What the runtime's own interpreter reports after running a real check (see `verifier.runtime_probe_code`)."""
    completed = _run([venv_python, "-I", "-B", "-c", verifier.runtime_probe_code(contract, _exercise()), source],
                     cwd=source, what="Probing the VeriEQL runtime")
    observed = json.loads(completed.stdout.strip().splitlines()[-1])
    if (observed["system"], observed["arch"]) != (contract.system, contract.machine):
        raise SetupError(f"the VeriEQL interpreter runs on {observed['system']} {observed['arch']}, not "
                         f"{contract.platform}")
    if observed["version"] != verifier.Z3_VERSION:
        raise SetupError(f"the environment's z3 reports {observed['version']}, not {verifier.Z3_VERSION}")
    return observed


def build_manifest(contract: verifier.RuntimeContract, venv_python: Path, observed: Mapping[str, Any]) -> dict:
    """The runtime manifest of this installation: its contract, roots and the SHA-256 of every file it can load."""
    manifest: dict[str, Any] = {
        "kind": contract.kind,
        "source": verifier.VERIEQL_SOURCE,
        "platform": contract.platform,
        "verieql_revision": verifier.VERIEQL_REVISION,
        "python": str(venv_python),
        "python_version": observed["python_version"],
        "source_root": observed["source"],
        "stdlib_root": observed["stdlib"],
        "site_packages_root": observed["site"],
        "z3_root": observed["z3"],
        "z3_version": observed["version"],
    }
    if contract.system == "Darwin":
        manifest["dyld_shared_cache_uuid"] = observed["dyld_shared_cache_uuid"]
    names = verifier.runtime_inventory(manifest) | set(observed["files"])
    manifest["files"] = {name: _sha256_file(name) for name in sorted(names)}
    return manifest


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> str:
    """Write the manifest (deterministically: the same installation gives the same bytes); returns its SHA-256."""
    raw = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    staged = path.with_name(path.name + ".partial")
    staged.write_bytes(raw)
    os.replace(staged, path)
    return hashlib.sha256(raw).hexdigest()


def environment_exports(manifest_path: Path, manifest_sha256: str) -> str:
    return (f"export {verifier.MANIFEST_ENV_VAR}={shlex.quote(str(manifest_path))}\n"
            f"export {verifier.MANIFEST_SHA256_ENV_VAR}={manifest_sha256}\n")


# --------------------------------------------------------------------------------------------------------------------
# The command.


def default_prefix(contract: verifier.RuntimeContract) -> Path:
    """Under the user's cache directory, one installation per pinned revision and platform."""
    if sys.platform == "darwin":
        cache = Path.home() / "Library" / "Caches"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME", "")
        cache = Path(xdg) if os.path.isabs(xdg) else Path.home() / ".cache"
    return cache / "text-to-sql-recipe" / f"verieql-{verifier.VERIEQL_REVISION[:12]}-{contract.platform}"


def run(*, prefix: str | os.PathLike[str] | None = None, accept: bool = False, python: str | None = None,
        stdin: TextIO | None = None, stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    """`setup verieql`; returns the exit code (0 ready, 1 failed, 2 licence not accepted)."""
    stdin, stdout, stderr = stdin or sys.stdin, stdout or sys.stdout, stderr or sys.stderr

    def log(message: str) -> None:
        stderr.write(f"setup verieql: {message}\n")
        stderr.flush()

    try:
        contract = verifier.platform_contract(platform.system(), platform.machine())
        root = Path(prefix).expanduser() if prefix is not None else default_prefix(contract)
        root = root.resolve()
        uv = shutil.which("uv")
        check_prerequisites(python, uv)
        how = accept_license(accept, root, stdin=stdin, stderr=stderr)
        root.mkdir(parents=True, exist_ok=True)
        source = fetch_source(root, log=log)
        venv_python = build_environment(root, python=python, uv=uv, log=log)
        z3_root = apply_z3_overlay(source, venv_python)
        log("Precompiling bytecode")
        owned = [source, z3_root.parent]
        if (root / "python").is_dir():
            owned.append(root / "python")
        compile_bytecode(venv_python, owned, log=log)
        log("Probing the runtime and writing its manifest")
        observed = probe_runtime(contract, venv_python, source)
        manifest = build_manifest(contract, venv_python, observed)
        manifest_path = root / "runtime-manifest.json"
        manifest_sha256 = write_manifest(manifest_path, manifest)
        try:
            validated = verifier.validate_manifest(str(manifest_path), manifest_sha256)
        except (ValueError, subprocess.SubprocessError) as error:
            raise SetupError(f"the new runtime manifest does not validate: {error}") from error
        log(f"Wrote {manifest_path} ({contract.kind}, {len(manifest['files'])} files); running the self-check")
        results = run_self_check(validated, manifest_sha256, root / "self-check")
        report = {"verieql_revision": verifier.VERIEQL_REVISION, "source_sha256": verifier.VERIEQL_SOURCE_SHA256,
                  "license": {"name": LICENSE_NAME, "accepted": how}, "kind": contract.kind,
                  "python_version": manifest["python_version"], "z3_version": manifest["z3_version"],
                  "manifest": str(manifest_path), "manifest_sha256": manifest_sha256,
                  "packages": {name: version for name, version, _ in (*WHEELS, *SOURCES)}, "self_check": results}
        (root / "setup-report.json").write_text(json.dumps(report, indent=1) + "\n")
        for result in results:
            if "skipped" in result:
                log(f"  skipped  {result['name']}: {result['skipped']}")
            else:
                shown = result.get("observed", result.get("error"))
                log(f"  {'ok     ' if result['ok'] else 'FAILED '}  {result['name']}: {shown}")
        failed = [result["name"] for result in results if not result.get("ok", True)]
        if failed:
            raise SetupError(f"the self-check failed ({', '.join(failed)}); see {root / 'setup-report.json'}. "
                             "The manifest is not ready for training.")
    except LicenseNotAccepted as error:
        stderr.write(f"setup verieql: {error}\n")
        return 2
    except (SetupError, verifier.VeriEQLError, OSError) as error:
        stderr.write(f"setup verieql: {error}\n")
        return 1
    log("VeriEQL is ready. Set these before training (or eval this command's output):")
    stdout.write(environment_exports(manifest_path, manifest_sha256))
    return 0
