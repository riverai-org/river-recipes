"""The platform-aware VeriEQL runtime contract: Linux aarch64 and x86_64 and macOS arm64 next to the reference one.

Offline, like test_verifier: the runtime probe's report is faked (`subprocess.run` in the verifier is patched) and the
runtime's directories are small fake trees. Two tests run the real loaded-file probes on the machine running the
suite: the macOS one (dyld and memory regions) on macOS; the Linux one lives in test_verifier.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from text_to_sql_recipe import verifier

PLATFORM_KINDS = ("verieql-native-linux-aarch64-v1", "verieql-native-linux-x86_64-v1", "verieql-native-macos-arm64-v1")
UUID = "0123456789abcdef0123456789abcdef"


def runtime(root: Path, kind: str):
    """A fake installation for `kind`: the manifest setup would write for it, and what its probe would report."""
    contract = verifier.RUNTIME_CONTRACTS[kind]
    manifest = {"kind": kind, "source": verifier.VERIEQL_SOURCE, "z3_version": verifier.Z3_VERSION}
    for key in ("source_root", "stdlib_root", "site_packages_root"):
        folder = root / key
        folder.mkdir()
        (folder / "module.py").write_text("pass")
        manifest[key] = str(folder.resolve())
    z3 = Path(manifest["site_packages_root"]) / "z3"
    z3.mkdir()
    manifest["z3_root"] = str(z3.resolve())
    libraries = Path(manifest["source_root"]) / "z3py_libs"
    libraries.mkdir()
    for name in ("z3.py", "z3core.py"):
        (libraries / name).write_text(f"# VeriEQL's copy of {name}\n")
        (z3 / name).write_text(f"# VeriEQL's copy of {name}\n")
    (Path(manifest["source_root"]) / "__pycache__").mkdir()
    (Path(manifest["source_root"]) / "__pycache__" / "module.cpython-311.pyc").write_bytes(b"bytecode")
    executable = root / "python"
    executable.write_text("interpreter")
    manifest["python"] = str(executable.resolve())
    manifest.update(platform=contract.platform, verieql_revision=verifier.VERIEQL_REVISION, python_version="3.11.14")
    if contract.system == "Darwin":
        manifest["dyld_shared_cache_uuid"] = UUID
    manifest["files"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                         for name in verifier.runtime_inventory(manifest)}
    observed = {"arch": contract.machine, "version": verifier.Z3_VERSION, "source": manifest["source_root"],
                "stdlib": manifest["stdlib_root"], "site": manifest["site_packages_root"], "z3": manifest["z3_root"],
                "files": list(manifest["files"]), "system": contract.system, "python_version": "3.11.14"}
    if contract.system == "Darwin":
        observed["dyld_shared_cache_uuid"] = UUID
    return manifest, observed


def pinned_tree(manifest):
    """The digest a fake source tree gets, to stand in for `VERIEQL_SOURCE_SHA256`."""
    return verifier.source_tree_sha256(verifier.source_tree_entries(manifest))


class PlatformContractTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def validate(self, manifest, observed, *, host=None, tree=None):
        """`validate_manifest` on this manifest, with the probe reporting `observed` on a `host` machine."""
        contract = verifier.RUNTIME_CONTRACTS[manifest["kind"]]
        path = self.root / "manifest.json"
        raw = json.dumps(manifest).encode()
        path.write_bytes(raw)
        probe = SimpleNamespace(stdout="noise\n" + json.dumps(observed))
        with patch.object(verifier.subprocess, "run", return_value=probe) as run, \
                patch.object(verifier.platform, "system", return_value=host or contract.system), \
                patch.object(verifier, "VERIEQL_SOURCE_SHA256", tree or pinned_tree(manifest)):
            result = verifier.validate_manifest(str(path), hashlib.sha256(raw).hexdigest())
        self.probe_code = run.call_args.args[0][4]
        return result

    def fresh(self, kind):
        folder = Path(tempfile.mkdtemp(dir=self.root))
        return runtime(folder, kind)

    def test_every_platform_contract_validates_with_its_own_probe(self):
        for kind in PLATFORM_KINDS:
            with self.subTest(kind=kind):
                manifest, observed = self.fresh(kind)
                self.assertEqual(self.validate(manifest, observed), manifest)
                contract = verifier.RUNTIME_CONTRACTS[kind]
                self.assertTrue(self.probe_code.startswith(contract.probe))
                self.assertEqual("dyld_shared_cache_uuid" in self.probe_code, contract.system == "Darwin")
                self.assertEqual("/proc/self/maps" in self.probe_code, contract.system == "Linux")

    def test_kind_names(self):
        for kind, contract in verifier.RUNTIME_CONTRACTS.items():
            self.assertEqual(kind, contract.kind)
            self.assertEqual(kind, f"verieql-native-{contract.platform}-v1")
        self.assertEqual({c.platform for c in verifier.RUNTIME_CONTRACTS.values()},
                         {"linux-aarch64", "linux-x86_64", "macos-arm64"})

    def test_a_manifest_must_name_its_contract(self):
        for manifest in ({}, {"kind": "verieql-native-arm64-v1"}):
            with self.subTest(manifest=manifest), self.assertRaisesRegex(ValueError, "Unknown native verifier"):
                verifier.contract_for(manifest)
        self.assertNotIn("verieql-native-amd64-v1", verifier.RUNTIME_CONTRACTS)

    def test_the_declared_fields_are_checked(self):
        for kind in PLATFORM_KINDS:
            manifest, observed = self.fresh(kind)
            other = "linux-x86_64" if manifest["platform"] != "linux-x86_64" else "linux-aarch64"
            cases = [
                ({"platform": other}, "platform differs from its contract"),
                ({"verieql_revision": "0" * 40}, "pinned VeriEQL revision"),
                ({"z3_version": "4.12.2"}, "pinned solver version"),
                ({"python_version": "3.12.4"}, "must run Python 3.11"),
            ]
            if manifest["platform"] == "macos-arm64":
                cases.append(({"dyld_shared_cache_uuid": "not-a-uuid"}, "shared cache UUID"))
            for change, message in cases:
                with self.subTest(kind=kind, change=change), self.assertRaisesRegex(ValueError, message):
                    self.validate({**manifest, **change}, observed)

    def test_a_manifest_for_another_operating_system_is_refused_before_the_probe(self):
        manifest, observed = self.fresh("verieql-native-macos-arm64-v1")
        with self.assertRaisesRegex(ValueError, "is for macos-arm64; this machine runs Linux"):
            self.validate(manifest, observed, host="Linux")
        manifest, observed = self.fresh("verieql-native-linux-x86_64-v1")
        with self.assertRaisesRegex(ValueError, "is for linux-x86_64; this machine runs Darwin"):
            self.validate(manifest, observed, host="Darwin")

    def test_the_probe_must_report_the_contract_platform(self):
        manifest, observed = self.fresh("verieql-native-linux-x86_64-v1")
        for change, message in (({"arch": "aarch64"}, "architecture or solver version"),
                                ({"version": "4.12.2"}, "architecture or solver version"),
                                ({"system": "Darwin"}, "operating system or Python version"),
                                ({"python_version": "3.11.9"}, "operating system or Python version")):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                self.validate(manifest, {**observed, **change})

    def test_the_source_tree_must_be_the_pinned_revision(self):
        manifest, observed = self.fresh("verieql-native-linux-aarch64-v1")
        tree = pinned_tree(manifest)
        with self.assertRaisesRegex(ValueError, "source tree differs from the pinned revision"):
            self.validate(manifest, observed, tree="0" * 64)
        # A bytecode cache is not part of the tree, so precompiling the source keeps it valid.
        extra = Path(manifest["source_root"]) / "__pycache__" / "extra.cpython-311.pyc"
        extra.write_bytes(b"more bytecode")
        manifest["files"][str(extra.resolve())] = hashlib.sha256(extra.read_bytes()).hexdigest()
        self.assertEqual(self.validate(manifest, {**observed, "files": list(manifest["files"])}, tree=tree), manifest)
        # Any other file is.
        added = Path(manifest["source_root"]) / "extra.py"
        added.write_text("pass")
        manifest["files"][str(added.resolve())] = hashlib.sha256(added.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "source tree differs"):
            self.validate(manifest, observed, tree=tree)

    def test_z3_must_carry_the_z3py_libs_copies(self):
        manifest, observed = self.fresh("verieql-native-macos-arm64-v1")
        (Path(manifest["z3_root"]) / "z3core.py").write_text("the solver's own z3core.py")
        manifest["files"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in manifest["files"]}
        with self.assertRaisesRegex(ValueError, "z3py_libs copy of z3core.py"):
            self.validate(manifest, observed)

    def test_the_macos_shared_cache_must_be_the_pinned_one(self):
        manifest, observed = self.fresh("verieql-native-macos-arm64-v1")
        with self.assertRaisesRegex(ValueError, "system library cache differs.*setup verieql"):
            self.validate(manifest, {**observed, "dyld_shared_cache_uuid": "f" * 32})

    def test_loaded_files_outside_the_inventory_are_refused_on_every_platform(self):
        for kind in PLATFORM_KINDS:
            manifest, observed = self.fresh(kind)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "missing from inventory"):
                self.validate(manifest, {**observed, "files": [*observed["files"], "/elsewhere/libz3.dylib"]})

    def test_platform_contract_for_this_machine(self):
        self.assertEqual(verifier.platform_contract("Linux", "aarch64").kind, "verieql-native-linux-aarch64-v1")
        self.assertEqual(verifier.platform_contract("Linux", "x86_64").kind, "verieql-native-linux-x86_64-v1")
        self.assertEqual(verifier.platform_contract("Darwin", "arm64").kind, "verieql-native-macos-arm64-v1")
        for system, machine in (("Darwin", "x86_64"), ("Windows", "AMD64"), ("Linux", "riscv64")):
            with self.subTest(system=system), self.assertRaisesRegex(verifier.VeriEQLError, "no VeriEQL runtime"):
                verifier.platform_contract(system, machine)

    def test_contract_for_a_manifest(self):
        self.assertEqual(verifier.contract_for({"kind": "verieql-native-macos-arm64-v1"}).backend, "native-macos-arm64")
        with self.assertRaisesRegex(ValueError, "Unknown native verifier contract"):
            verifier.contract_for({"kind": "verieql-native-amd64-v1"})


class SourceTreeDigestTest(unittest.TestCase):
    def test_the_digest_is_sorted_sha256sum_lines(self):
        entries = {"b.py": "2" * 64, "a/c.py": "1" * 64}
        expected = hashlib.sha256(f"{'1' * 64}  a/c.py\n{'2' * 64}  b.py\n".encode()).hexdigest()
        self.assertEqual(verifier.source_tree_sha256(entries), expected)
        self.assertEqual(verifier.source_tree_sha256(dict(reversed(entries.items()))), expected)

    def test_the_pins(self):
        self.assertRegex(verifier.VERIEQL_REVISION, r"^[0-9a-f]{40}$")
        self.assertRegex(verifier.VERIEQL_SOURCE_SHA256, r"^[0-9a-f]{64}$")
        self.assertEqual((verifier.Z3_VERSION, verifier.PYTHON_SERIES), ("4.13.0", "3.11"))


class CheckedHelperTest(unittest.TestCase):
    def test_the_linux_helper_is_unchanged(self):
        """The helper of the reference contract is byte for byte what it was before the contract became per-platform."""
        for args, digest in (
            (("0" * 64,), "b8202ffbc493213b35d64f6949f06180c8dadf8a53ab14ace8e236ab967fd3ec"),
            (("1" * 64, verifier._EXTRACTION_SCRIPT), "62884694cc3b517ee1d4f7785370fc2c0603e36c67fbdbd88bde8624b306ebc8"),
        ):
            self.assertEqual(hashlib.sha256(verifier.checked_helper(*args).encode()).hexdigest(), digest)

    def test_the_macos_helper_also_checks_the_shared_cache(self):
        code = verifier.checked_helper("0" * 64, probe=verifier._RUNTIME_PROBE_MACOS, shared_cache_uuid=UUID)
        self.assertTrue(code.startswith(verifier._RUNTIME_PROBE_MACOS))
        self.assertNotIn("/proc/self/maps", code)
        self.assertIn(f"        if shared_cache_uuid() != {UUID!r}:\n"
                      "            raise RuntimeError('Unpinned or changed system library cache')\n", code)
        compile(code, "<helper>", "exec")

    def test_the_macos_probe_is_pinned(self):
        """It runs inside VeriEQL's interpreter, like the scripts test_verifier pins: any change must be deliberate."""
        self.assertEqual(hashlib.sha256(verifier._RUNTIME_PROBE_MACOS.encode()).hexdigest(),
                         "05c8c69ada97050edb0646e1277e496f954bd0f28c3f01d58674d5820628afb2")


class NativeVerifierContractTest(unittest.IsolatedAsyncioTestCase):
    def verifier_for(self, kind):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        manifest = {"kind": kind, "files": {}, "python": sys.executable, "source_root": directory.name,
                    "z3_root": "/site/z3", "stdlib_root": "/stdlib", "site_packages_root": "/site",
                    "dyld_shared_cache_uuid": UUID}
        return verifier.NativeVerifier(manifest, "identity", Path(directory.name) / "receipts")

    def test_backends(self):
        self.assertEqual(
            {kind: self.verifier_for(kind).contract.backend for kind in PLATFORM_KINDS},
            {"verieql-native-linux-aarch64-v1": "native-linux-aarch64",
             "verieql-native-linux-x86_64-v1": "native-linux-x86_64",
             "verieql-native-macos-arm64-v1": "native-macos-arm64"})

    async def test_each_check_runs_the_helper_of_its_contract(self):
        for kind, expected_kwargs in (("verieql-native-linux-aarch64-v1", {}), ("verieql-native-linux-x86_64-v1", {}),
                                      ("verieql-native-macos-arm64-v1",
                                       {"probe": verifier._RUNTIME_PROBE_MACOS, "shared_cache_uuid": UUID})):
            native = self.verifier_for(kind)
            with self.subTest(kind=kind), patch.object(verifier, "checked_helper",
                                                       return_value="print('{\"result\": true}')") as helper:
                completed = await native._attempt({}, 30)
                self.assertEqual(completed.returncode, 0)
                self.assertEqual(helper.call_args.args, (native.inventory_sha256, None))
                self.assertEqual(helper.call_args.kwargs, expected_kwargs)


@unittest.skipUnless(sys.platform == "darwin", "the macOS probe reads dyld and the memory regions of a macOS process")
class MacOSProbeTest(unittest.TestCase):
    """The real macOS probe, run in the interpreter running these tests."""

    def run_code(self, code, *args):
        return subprocess.run([sys.executable, "-I", "-B", "-c", code, *map(str, args)], capture_output=True,
                              text=True, check=False)

    def test_the_probe_lists_loaded_files_and_leaves_the_shared_cache_to_its_uuid(self):
        completed = self.run_code(verifier._RUNTIME_PROBE_MACOS + "import sqlite3\n"
                                  "print(json.dumps({'files': loaded_files(), 'uuid': shared_cache_uuid(),\n"
                                  "                  'cached': [n for n, c in _images() if c]}))\n")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        files = set(report["files"])
        self.assertIn(str(Path(sys.executable).resolve()), files)
        self.assertTrue(any(Path(name).name == "dyld" for name in files))   # found by walking the memory regions
        self.assertTrue(all(Path(name).is_file() and str(Path(name).resolve()) == name for name in files))
        self.assertRegex(report["uuid"], r"^[0-9a-f]{32}$")
        self.assertIn("/usr/lib/libSystem.B.dylib", report["cached"])
        self.assertFalse(files & set(report["cached"]))

    def test_the_helper_withholds_a_verdict_for_an_unpinned_load_or_another_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self.run_code(verifier._RUNTIME_PROBE_MACOS + "import contextlib, io, hashlib\n"
                                     "print(json.dumps({'uuid': shared_cache_uuid(), 'files': {name: hashlib.sha256("
                                     "pathlib.Path(name).read_bytes()).hexdigest() for name in loaded_files()}}))\n")
            self.assertEqual(snapshot.returncode, 0, snapshot.stderr)
            report = json.loads(snapshot.stdout)
            inventory = json.dumps(report["files"]).encode()
            path = root / "inventory.json"
            path.write_bytes(inventory)
            digest = hashlib.sha256(inventory).hexdigest()

            def helper(script, uuid):
                with patch.object(verifier, "_HELPER_SCRIPT", script):
                    code = verifier.checked_helper(digest, probe=verifier._RUNTIME_PROBE_MACOS,
                                                   shared_cache_uuid=uuid)
                return self.run_code(code, path)

            valid = helper('print("verified")', report["uuid"])
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertEqual(valid.stdout.strip(), "verified")
            (root / "late_dependency.py").write_text("value = 1")
            late = helper(f"import sys; sys.path.insert(0, {str(root)!r}); import late_dependency; print('verified')",
                          report["uuid"])
            self.assertNotEqual(late.returncode, 0)
            self.assertEqual(late.stdout, "")
            self.assertIn("Unpinned or changed loaded runtime file", late.stderr)
            other_cache = helper('print("verified")', "0" * 32)
            self.assertNotEqual(other_cache.returncode, 0)
            self.assertEqual(other_cache.stdout, "")
            self.assertIn("Unpinned or changed system library cache", other_cache.stderr)


class DocumentationTest(unittest.TestCase):
    def test_the_module_docstring_names_every_contract_and_the_setup_command(self):
        for kind in verifier.RUNTIME_CONTRACTS:
            self.assertIn(kind, verifier.__doc__)
        self.assertRegex(verifier.__doc__, r"python -m text_to_sql_recipe setup\s+verieql")


if __name__ == "__main__":
    unittest.main()
