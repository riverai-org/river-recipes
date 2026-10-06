"""`setup verieql`, tested offline: no network, no VeriEQL, no uv or pip runs.

Downloads come from a fake opener serving an archive built here, the runtime's interpreter is replaced by a faked
probe report, and the verifier the self-check runs is stubbed. What is real: the licence gate, archive verification
and extraction, the requirements, manifest generation and its validation by the recipe's own `validate_manifest`.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from text_to_sql_recipe import __main__ as cli
from text_to_sql_recipe import verieql_setup as setup
from text_to_sql_recipe import verifier

UUID = "0123456789abcdef0123456789abcdef"


class Terminal(io.StringIO):
    """Standard input attached to a terminal, for the interactive prompt."""

    def isatty(self):
        return True


def archive_bytes(files: dict[str, bytes], *, top="VeriEQL-" + verifier.VERIEQL_REVISION, extra=()) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for info in extra:
            tar.addfile(info)
    return buffer.getvalue()


def tree_digest(files: dict[str, bytes]) -> str:
    return verifier.source_tree_sha256({name: hashlib.sha256(data).hexdigest() for name, data in files.items()})


SOURCE = {"environment.py": b"class Environment: pass\n", "errors.py": b"",
          "z3py_libs/z3.py": b"# VeriEQL's z3.py\n", "license.md": b"Attribution-NonCommercial-ShareAlike 4.0\n"}


class Opener:
    """`urllib.request.urlopen` serving fixed bytes, recording what was requested."""

    def __init__(self, data: bytes):
        self.data, self.requests = data, []

    def __call__(self, request, timeout):
        self.requests.append(request.full_url)
        return io.BytesIO(self.data)


class LicenceTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.prefix = Path(self.directory.name) / "prefix"
        for machine in (patch.object(setup.platform, "system", return_value="Linux"),
                        patch.object(setup.platform, "machine", return_value="x86_64")):
            machine.start()
            self.addCleanup(machine.stop)

    def test_the_notice_names_the_licence_the_revision_and_the_destination(self):
        notice = setup.license_notice(self.prefix)
        for text in ("CC BY-NC-SA 4.0", "NonCommercial", "non-commercial", setup.LICENSE_URL,
                     verifier.VERIEQL_REVISION, f"{setup.VERIEQL_REPOSITORY}/blob/{verifier.VERIEQL_REVISION}/license.md",
                     str(self.prefix)):
            self.assertIn(text, notice)

    def test_the_flag_accepts_without_reading_input(self):
        stdin, stderr = MagicMock(), io.StringIO()
        self.assertEqual(setup.accept_license(True, self.prefix, stdin=stdin, stderr=stderr), "flag")
        stdin.readline.assert_not_called()
        self.assertIn("CC BY-NC-SA 4.0", stderr.getvalue())

    def test_only_yes_at_the_prompt_accepts(self):
        for answer, accepted in (("yes\n", True), ("YES \n", True), ("y\n", False), ("no\n", False), ("", False)):
            with self.subTest(answer=answer):
                stderr = io.StringIO()
                if accepted:
                    self.assertEqual(setup.accept_license(False, self.prefix, stdin=Terminal(answer), stderr=stderr),
                                     "prompt")
                else:
                    with self.assertRaises(setup.LicenseNotAccepted):
                        setup.accept_license(False, self.prefix, stdin=Terminal(answer), stderr=stderr)
                self.assertIn("Type 'yes' to accept", stderr.getvalue())

    def test_without_a_terminal_only_the_flag_accepts(self):
        with self.assertRaisesRegex(setup.LicenseNotAccepted, "--accept-license"):
            setup.accept_license(False, self.prefix, stdin=io.StringIO("yes\n"), stderr=io.StringIO())

    def test_nothing_is_fetched_or_created_until_the_licence_is_accepted(self):
        with patch.object(setup, "fetch_source") as fetch, patch.object(setup, "build_environment") as build, \
                patch.object(setup, "check_prerequisites"), patch.object(setup.shutil, "which", return_value="uv"):
            stdout, stderr = io.StringIO(), io.StringIO()
            code = setup.run(prefix=self.prefix, stdin=io.StringIO(""), stdout=stdout, stderr=stderr)
        self.assertEqual(code, 2)
        fetch.assert_not_called()
        build.assert_not_called()
        self.assertFalse(self.prefix.exists())
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("nothing was downloaded", stderr.getvalue())

    def test_missing_prerequisites_fail_before_the_licence_prompt(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(setup.shutil, "which", return_value=None):
            code = setup.run(prefix=self.prefix, accept=True, stdin=io.StringIO(), stdout=stdout, stderr=stderr)
        self.assertEqual(code, 1)
        self.assertIn("needs uv", stderr.getvalue())
        self.assertNotIn("CC BY-NC-SA", stderr.getvalue())
        self.assertFalse(self.prefix.exists())

    def test_python_must_be_cpython_3_11(self):
        for found, ok in (({"implementation": "CPython", "version": "3.11.9"}, True),
                          ({"implementation": "CPython", "version": "3.12.4"}, False),
                          ({"implementation": "PyPy", "version": "3.11.13"}, False)):
            with self.subTest(found=found), patch.object(setup, "_interpreter", return_value=found):
                if ok:
                    setup.check_prerequisites("python3.11", None)
                else:
                    with self.assertRaisesRegex(setup.SetupError, "needs CPython 3.11"):
                        setup.check_prerequisites("python3.11", None)


class CommandLineTest(unittest.TestCase):
    def test_setup_verieql_passes_its_options_and_needs_no_recipe(self):
        with patch.object(setup, "run", return_value=0) as run, \
                patch.object(cli.config, "load", side_effect=AssertionError("setup loads no recipe")):
            self.assertEqual(cli.main(["setup", "verieql", "--prefix", "somewhere", "--accept-license",
                                       "--python", "python3.11"]), 0)
            run.assert_called_once_with(prefix="somewhere", accept=True, python="python3.11")
            run.reset_mock(return_value=True)
            run.return_value = 2
            self.assertEqual(cli.main(["setup", "verieql"]), 2)
            run.assert_called_once_with(prefix=None, accept=False, python=None)

    def test_setup_needs_a_target(self):
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as raised:
            cli.main(["setup"])
        self.assertEqual(raised.exception.code, 2)

    def test_the_command_is_documented(self):
        self.assertIn("setup verieql", cli.__doc__)


class SourceTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.prefix = Path(self.directory.name)
        for pin in (patch.object(verifier, "VERIEQL_SOURCE_SHA256", tree_digest(SOURCE)),
                    patch.object(setup, "SOURCE_FILE_COUNT", len(SOURCE))):
            pin.start()
            self.addCleanup(pin.stop)

    def extract(self, data: bytes, **kwargs):
        archive = self.prefix / "archive.tar.gz"
        archive.write_bytes(data)
        options = {"expected_sha256": tree_digest(SOURCE), "file_count": len(SOURCE), **kwargs}
        setup.extract_verified(archive, self.prefix / "source", **options)

    def test_a_matching_archive_is_unpacked_without_its_top_directory(self):
        self.extract(archive_bytes(SOURCE))
        self.assertEqual(setup.tree_entries(self.prefix / "source"),
                         {name: hashlib.sha256(data).hexdigest() for name, data in SOURCE.items()})

    def test_anything_else_is_refused_and_nothing_is_unpacked(self):
        link = tarfile.TarInfo(f"VeriEQL-{verifier.VERIEQL_REVISION}/environment_link.py")
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
        escape = tarfile.TarInfo(f"VeriEQL-{verifier.VERIEQL_REVISION}/../escape.py")
        cases = {
            "changed file": (archive_bytes({**SOURCE, "errors.py": b"tampered"}), "does not match revision"),
            "extra file": (archive_bytes({**SOURCE, "extra.py": b""}), "does not match revision"),
            "missing file": (archive_bytes({k: v for k, v in SOURCE.items() if k != "errors.py"}),
                             "does not match revision"),
            "symbolic link": (archive_bytes(SOURCE, extra=[link]), "unexpected entry"),
            "path escape": (archive_bytes(SOURCE, extra=[escape]), "unexpected entry"),
        }
        for label, (data, message) in cases.items():
            with self.subTest(label=label), self.assertRaisesRegex(setup.SetupError, message):
                self.extract(data)
            self.assertFalse((self.prefix / "source").exists())
        two_tops = io.BytesIO()
        with tarfile.open(fileobj=two_tops, mode="w:gz") as tar:
            for name in ("a/x.py", "b/y.py"):
                tar.addfile(tarfile.TarInfo(name), io.BytesIO(b""))
        with self.assertRaisesRegex(setup.SetupError, "one top-level directory"):
            self.extract(two_tops.getvalue())

    def test_fetch_downloads_the_pinned_revision_once(self):
        opener = Opener(archive_bytes(SOURCE))
        source = setup.fetch_source(self.prefix, log=lambda message: None, opener=opener)
        self.assertEqual(opener.requests, [setup.SOURCE_ARCHIVE_URL])
        self.assertIn(f"/archive/{verifier.VERIEQL_REVISION}.tar.gz", setup.SOURCE_ARCHIVE_URL)
        self.assertEqual(source, self.prefix / "source")
        # Precompiled bytecode in the tree does not stop a later run from reusing it, and nothing is downloaded again.
        (source / "__pycache__").mkdir()
        (source / "__pycache__" / "errors.cpython-311.pyc").write_bytes(b"bytecode")
        self.assertEqual(setup.fetch_source(self.prefix, log=lambda message: None, opener=opener), source)
        self.assertEqual(len(opener.requests), 1)

    def test_fetch_refuses_a_source_directory_it_cannot_verify(self):
        (self.prefix / "source").mkdir()
        (self.prefix / "source" / "environment.py").write_text("something else")
        opener = Opener(archive_bytes(SOURCE))
        with self.assertRaisesRegex(setup.SetupError, "does not hold VeriEQL revision"):
            setup.fetch_source(self.prefix, log=lambda message: None, opener=opener)
        self.assertEqual(opener.requests, [])
        self.assertEqual((self.prefix / "source" / "environment.py").read_text(), "something else")

    def test_a_download_has_a_size_limit(self):
        with self.assertRaisesRegex(setup.SetupError, "larger than 10 bytes"):
            setup._download("https://example.com/a.tar.gz", self.prefix / "a.tar.gz", max_bytes=10,
                            opener=Opener(b"x" * 11))
        self.assertFalse((self.prefix / "a.tar.gz").exists())

class RequirementsTest(unittest.TestCase):
    def test_the_pinned_source(self):
        self.assertEqual(setup.SOURCE_FILE_COUNT, 248)
        self.assertEqual(setup.SOURCE_ARCHIVE_URL,
                         f"https://github.com/VeriEQL/VeriEQL/archive/{verifier.VERIEQL_REVISION}.tar.gz")

    def test_every_dependency_is_pinned_by_version_and_hash(self):
        names = [name for name, _, _ in (*setup.WHEELS, *setup.SOURCES)]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("mo-sql-parsing", names)
        for name, version, digests in (*setup.WHEELS, *setup.SOURCES):
            with self.subTest(name=name):
                self.assertRegex(version, r"^\d+(\.\d+)*(\.post\d+)?$")
                self.assertTrue(digests)
                self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in digests))
        solver = {name: version for name, version, _ in setup.WHEELS}["z3-solver"]
        self.assertTrue(solver.startswith(verifier.Z3_VERSION + "."))
        self.assertTrue(setup.PYTHON_VERSION.startswith(verifier.PYTHON_SERIES + "."))

    def test_requirement_lines(self):
        text = setup.requirements_text([("pkg", "1.0", ("a" * 64, "b" * 64))])
        self.assertEqual(text, f"pkg==1.0 --hash=sha256:{'a' * 64} --hash=sha256:{'b' * 64}\n")


class EnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.prefix = Path(self.directory.name)

    def run_environment(self, *, python, uv, venv_exists=False):
        commands = []

        def fake_run(command, *, env=None, cwd=None, what):
            commands.append([str(part) for part in command])
            if command[1:3] == ["python", "find"]:
                return subprocess.CompletedProcess(command, 0, str(self.prefix / "python" / "bin" / "python3.11"), "")
            return subprocess.CompletedProcess(command, 0, "", "")

        if venv_exists:
            (self.prefix / "venv").mkdir()
        found = {"implementation": "CPython", "version": "3.11.14"}
        with patch.object(setup, "_run", side_effect=fake_run), patch.object(setup, "_interpreter",
                                                                              return_value=found):
            venv_python = setup.build_environment(self.prefix, python=python, uv=uv, log=lambda message: None)
        self.assertEqual(venv_python, self.prefix / "venv" / "bin" / "python")
        return commands

    def test_uv_supplies_the_interpreter_without_touching_the_users_path(self):
        commands = self.run_environment(python=None, uv="uv")
        install = next(c for c in commands if c[1:3] == ["python", "install"])
        self.assertEqual(install[3], setup.PYTHON_VERSION)
        self.assertIn("--no-bin", install)
        self.assertEqual(install[install.index("--install-dir") + 1], str(self.prefix / "python"))
        venv = next(c for c in commands if c[1] == "venv")
        self.assertEqual(venv[venv.index("--python") + 1], str(self.prefix / "python" / "bin" / "python3.11"))
        wheels, sources = [c for c in commands if c[1:3] == ["pip", "install"]]
        for command in (wheels, sources):
            self.assertIn("--require-hashes", command)
            self.assertIn("--no-deps", command)
            self.assertEqual(command[command.index("--link-mode") + 1], "copy")
        self.assertIn("--no-build", wheels)
        self.assertIn("--no-build-isolation", sources)
        self.assertEqual((self.prefix / "requirements" / "wheels.txt").read_text(),
                         setup.requirements_text(setup.WHEELS))
        self.assertEqual((self.prefix / "requirements" / "sources.txt").read_text(),
                         setup.requirements_text(setup.SOURCES))

    def test_without_uv_pip_builds_from_the_given_python(self):
        commands = self.run_environment(python="python3.11", uv=None)
        self.assertEqual(commands[0], ["python3.11", "-m", "venv", str(self.prefix / "venv")])
        wheels, sources = commands[1:]
        self.assertEqual(wheels[:4], [str(self.prefix / "venv" / "bin" / "python"), "-m", "pip", "install"])
        self.assertEqual(wheels[wheels.index("--only-binary") + 1], ":all:")
        self.assertIn("--no-build-isolation", sources)

    def test_an_existing_environment_is_reused(self):
        commands = self.run_environment(python=None, uv="uv", venv_exists=True)
        self.assertFalse([c for c in commands if c[1] in ("venv",) or c[1:3] == ["python", "install"]])
        self.assertEqual(len([c for c in commands if c[1:3] == ["pip", "install"]]), 2)

    def test_the_z3_overlay_replaces_files_without_writing_through_links(self):
        source = self.prefix / "source"
        (source / "z3py_libs").mkdir(parents=True)
        (source / "z3py_libs" / "z3.py").write_text("VeriEQL's z3.py")
        (source / "z3py_libs" / "z3_copy.py").write_text("VeriEQL's addition")
        z3 = self.prefix / "site-packages" / "z3"
        z3.mkdir(parents=True)
        cache = self.prefix / "installer-cache-z3.py"
        cache.write_text("the solver's z3.py")
        os.link(cache, z3 / "z3.py")
        completed = subprocess.CompletedProcess([], 0, str(z3) + "\n", "")
        with patch.object(setup, "_run", return_value=completed):
            self.assertEqual(setup.apply_z3_overlay(source, self.prefix / "python"), z3)
        self.assertEqual((z3 / "z3.py").read_text(), "VeriEQL's z3.py")
        self.assertEqual((z3 / "z3_copy.py").read_text(), "VeriEQL's addition")
        self.assertEqual(cache.read_text(), "the solver's z3.py")


def fake_installation(root: Path, contract: verifier.RuntimeContract):
    """Directories shaped like a setup installation, and the report its probe would print."""
    source = root / "source"
    (source / "z3py_libs").mkdir(parents=True)
    for name, data in SOURCE.items():
        (source / name).write_bytes(data)
    stdlib = root / "python" / "lib" / "python3.11"
    stdlib.mkdir(parents=True)
    (stdlib / "os.py").write_text("pass")
    site = root / "venv" / "lib" / "python3.11" / "site-packages"
    z3 = site / "z3"
    z3.mkdir(parents=True)
    (z3 / "z3.py").write_bytes(SOURCE["z3py_libs/z3.py"])
    (root / "venv" / "bin").mkdir()
    interpreter = root / "python" / "bin" / "python3.11"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("interpreter")
    (root / "venv" / "bin" / "python").symlink_to(interpreter)
    loader = root / "system" / ("dyld" if contract.system == "Darwin" else "ld-linux.so.1")
    loader.parent.mkdir()
    loader.write_text("loader")
    observed = {"arch": contract.machine, "system": contract.system, "version": verifier.Z3_VERSION,
                "python_version": "3.11.14", "source": str(source.resolve()), "stdlib": str(stdlib.resolve()),
                "site": str(site.resolve()), "z3": str(z3.resolve()),
                "files": [str(interpreter.resolve()), str(loader.resolve()), str((source / "environment.py").resolve())]}
    if contract.system == "Darwin":
        observed["dyld_shared_cache_uuid"] = UUID
    return root / "venv" / "bin" / "python", observed


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()

    def test_each_platform_gets_a_manifest_that_the_recipe_validates(self):
        for system, machine in (("Linux", "aarch64"), ("Linux", "x86_64"), ("Darwin", "arm64")):
            contract = verifier.platform_contract(system, machine)
            with self.subTest(platform=contract.platform):
                root = Path(tempfile.mkdtemp(dir=self.root))
                venv_python, observed = fake_installation(root, contract)
                manifest = setup.build_manifest(contract, venv_python, observed)
                self.assertEqual(
                    {key: manifest[key] for key in ("kind", "platform", "verieql_revision", "source",
                                                     "z3_version", "python_version", "python")},
                    {"kind": contract.kind, "platform": contract.platform,
                     "verieql_revision": verifier.VERIEQL_REVISION, "source": verifier.VERIEQL_SOURCE,
                     "z3_version": verifier.Z3_VERSION, "python_version": "3.11.14", "python": str(venv_python)})
                self.assertEqual(manifest.get("dyld_shared_cache_uuid"), UUID if system == "Darwin" else None)
                # Every file under the roots, the interpreter, and every loaded file outside them (the loader).
                self.assertEqual(set(manifest["files"]), verifier.runtime_inventory(manifest) | set(observed["files"]))
                self.assertIn(observed["files"][1], manifest["files"])
                for name, digest in manifest["files"].items():
                    self.assertEqual(digest, hashlib.sha256(Path(name).read_bytes()).hexdigest())
                path = root / "runtime-manifest.json"
                digest = setup.write_manifest(path, manifest)
                self.assertEqual(setup.write_manifest(path, manifest), digest)   # deterministic bytes
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
                probe = SimpleNamespace(stdout=json.dumps(observed))
                with patch.object(verifier.subprocess, "run", return_value=probe), \
                        patch.object(verifier.platform, "system", return_value=system), \
                        patch.object(verifier, "VERIEQL_SOURCE_SHA256", tree_digest(SOURCE)):
                    self.assertEqual(verifier.validate_manifest(str(path), digest), manifest)

    def test_the_probe_must_report_this_platform_and_the_pinned_solver(self):
        contract = verifier.platform_contract("Linux", "x86_64")
        _, observed = fake_installation(self.root, contract)
        for change, message in (({"arch": "aarch64"}, "not linux-x86_64"), ({"system": "Darwin"}, "not linux-x86_64"),
                                ({"version": "4.12.2"}, "not 4.13.0")):
            completed = subprocess.CompletedProcess([], 0, "warning\n" + json.dumps({**observed, **change}), "")
            with self.subTest(change=change), patch.object(setup, "_run", return_value=completed), \
                    self.assertRaisesRegex(setup.SetupError, message):
                setup.probe_runtime(contract, self.root / "venv" / "bin" / "python", self.root / "source")

    def test_the_probe_runs_real_checks_before_listing_loaded_files(self):
        contract = verifier.platform_contract("Darwin", "arm64")
        completed = subprocess.CompletedProcess([], 0, json.dumps(fake_installation(self.root, contract)[1]), "")
        with patch.object(setup, "_run", return_value=completed) as run:
            setup.probe_runtime(contract, self.root / "venv" / "bin" / "python", self.root / "source")
        command = run.call_args.args[0]
        code = command[4]
        self.assertEqual(command[1:4], ["-I", "-B", "-c"])
        self.assertEqual(run.call_args.kwargs["cwd"], self.root / "source")
        self.assertLess(code.index(repr(verifier._HELPER_SCRIPT)), code.index("files=loaded_files()"))
        self.assertIn(repr(verifier._EXTRACTION_SCRIPT), code)
        self.assertIn("dyld_shared_cache_uuid", code)
        compile(code, "<probe>", "exec")

    def test_exports(self):
        text = setup.environment_exports(Path("/opt/run time/runtime-manifest.json"), "a" * 64)
        self.assertEqual(text, "export TEXT_TO_SQL_VERIEQL_MANIFEST='/opt/run time/runtime-manifest.json'\n"
                               f"export TEXT_TO_SQL_VERIEQL_MANIFEST_SHA256={'a' * 64}\n")

    def test_default_prefix(self):
        contract = verifier.platform_contract("Linux", "aarch64")
        name = f"verieql-{verifier.VERIEQL_REVISION[:12]}-linux-aarch64"
        with patch.object(setup.sys, "platform", "linux"), patch.dict(os.environ, {"XDG_CACHE_HOME": "/cache"}):
            self.assertEqual(setup.default_prefix(contract), Path("/cache/text-to-sql-recipe") / name)
        with patch.object(setup.sys, "platform", "linux"), patch.dict(os.environ, {"XDG_CACHE_HOME": "relative"}):
            self.assertEqual(setup.default_prefix(contract), Path.home() / ".cache" / "text-to-sql-recipe" / name)
        with patch.object(setup.sys, "platform", "darwin"):
            self.assertEqual(setup.default_prefix(contract).parent,
                             Path.home() / "Library" / "Caches" / "text-to-sql-recipe")


class SelfCheckTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def run_checks(self, answer, *, sqlglot=True):
        """`run_self_check` with a verifier that answers each check with `answer(check)`."""
        built = []

        def fake_verifier(self_, receipt_dir):
            fake = MagicMock()
            fake.check = AsyncMock(side_effect=lambda task, prediction: answer(
                next(c for c in setup.SELF_CHECKS if c.prediction == prediction and c.gold_sql == task["gold_sql"]
                     and c.gated == (self_.gate_policy is not None))))
            fake.close = AsyncMock()
            built.append((self_, fake))
            return fake

        real_find_spec = setup.importlib.util.find_spec
        with patch.object(verifier.VeriEQLSetup, "verifier", fake_verifier), \
                patch.object(setup.importlib.util, "find_spec",
                             side_effect=lambda name: (object() if sqlglot else None) if name == "sqlglot"
                             else real_find_spec(name)):
            results = setup.run_self_check({"kind": "verieql-native-linux-aarch64-v1"}, "identity", self.root)
        self.built = built
        return results

    @staticmethod
    def expected(check):
        verdict = dict(check.expected)
        if "gate" in verdict:
            decision, reason = verdict.pop("gate")
            verdict["counterexample_gate"] = {"decision": decision, "reason": reason}
        return verdict

    def test_matching_verdicts_pass(self):
        results = self.run_checks(self.expected)
        self.assertEqual([r["name"] for r in results], [c.name for c in setup.SELF_CHECKS])
        self.assertTrue(all(r["ok"] for r in results), results)
        gated = [s for s, _ in self.built if s.gate_policy is not None]
        self.assertEqual(len(gated), 1)
        [(database, info)] = gated[0].gate_policy.items()
        self.assertTrue(database.endswith("self-check.sqlite"))
        self.assertEqual(info["policies"], [])
        self.assertTrue(all(fake.close.await_count == 1 for _, fake in self.built))
        self.assertEqual(json.loads(next(self.root.glob("*/results.json")).read_text()), results)

    def test_a_different_verdict_or_an_error_fails(self):
        def answer(check):
            if check.name == "boundary refutation":
                return {"verdict": "confirmed", "state": "EQU"}
            if check.name == "equivalent rewrite":
                raise verifier.VeriEQLError("Native verifier aborted")
            return self.expected(check)

        results = {r["name"]: r for r in self.run_checks(answer)}
        self.assertFalse(results["boundary refutation"]["ok"])
        self.assertEqual(results["boundary refutation"]["observed"]["verdict"], "confirmed")
        self.assertFalse(results["equivalent rewrite"]["ok"])
        self.assertIn("Native verifier aborted", results["equivalent rewrite"]["error"])
        self.assertTrue(results["projection arity"]["ok"])

    def test_without_sqlglot_the_gate_checks_are_skipped(self):
        results = self.run_checks(self.expected, sqlglot=False)
        skipped = [r["name"] for r in results if "skipped" in r]
        self.assertEqual(skipped, [c.name for c in setup.SELF_CHECKS if c.gated])
        self.assertTrue(all(r.get("ok", True) for r in results))

    def test_the_checks_cover_every_verdict_the_reward_reads(self):
        verdicts = {c.expected["verdict"] for c in setup.SELF_CHECKS}
        self.assertEqual(verdicts, {"confirmed", "refuted", "inconclusive", "unknown"})
        routes = {c.expected.get("refutation_route") for c in setup.SELF_CHECKS}
        self.assertTrue({"solver_model", "column_arity"} <= routes)
        self.assertEqual({tuple(c.expected["gate"]) for c in setup.SELF_CHECKS if c.gated},
                         {("penalize", "validated_counterexample"), ("abstain", "grader_matches")})


class RunTest(unittest.TestCase):
    """The whole command with every step that would touch the network or a real runtime faked."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.prefix = Path(self.directory.name).resolve() / "prefix"

    def run_setup(self, self_check):
        contract = verifier.platform_contract("Linux", "x86_64")
        stdout, stderr = io.StringIO(), io.StringIO()
        manifest = {"kind": contract.kind, "files": {"/x": "0" * 64}, "python_version": "3.11.14",
                    "z3_version": verifier.Z3_VERSION}
        with patch.object(setup.platform, "system", return_value="Linux"), \
                patch.object(setup.platform, "machine", return_value="x86_64"), \
                patch.object(setup.shutil, "which", return_value="uv"), \
                patch.object(setup, "fetch_source", return_value=self.prefix / "source") as fetch, \
                patch.object(setup, "build_environment", return_value=self.prefix / "venv" / "bin" / "python"), \
                patch.object(setup, "apply_z3_overlay", return_value=self.prefix / "site" / "z3"), \
                patch.object(setup, "compile_bytecode"), \
                patch.object(setup, "probe_runtime", return_value={}), \
                patch.object(setup, "build_manifest", return_value=manifest), \
                patch.object(verifier, "validate_manifest", return_value=manifest) as validate, \
                patch.object(setup, "run_self_check", return_value=self_check):
            code = setup.run(prefix=self.prefix, accept=True, stdin=io.StringIO(), stdout=stdout, stderr=stderr)
        self.fetch, self.validate = fetch, validate
        return code, stdout.getvalue(), stderr.getvalue()

    def test_success_prints_only_the_exports_on_stdout(self):
        code, stdout, stderr = self.run_setup([{"name": "equivalent rewrite", "ok": True, "observed": {}}])
        self.assertEqual(code, 0, stderr)
        path = self.prefix / "runtime-manifest.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(stdout, setup.environment_exports(path, digest))
        self.validate.assert_called_once_with(str(path), digest)
        report = json.loads((self.prefix / "setup-report.json").read_text())
        self.assertEqual((report["manifest_sha256"], report["license"]["accepted"]), (digest, "flag"))
        self.assertEqual(report["verieql_revision"], verifier.VERIEQL_REVISION)
        self.assertIn("VeriEQL is ready", stderr)

    def test_a_failed_self_check_prints_no_exports(self):
        code, stdout, stderr = self.run_setup([{"name": "boundary refutation", "ok": False, "observed": {}}])
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("self-check failed (boundary refutation)", stderr)


if __name__ == "__main__":
    unittest.main()
