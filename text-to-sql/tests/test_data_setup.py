"""`setup data` against a small release registered for the test: copy or download, then check every file."""
import contextlib
import hashlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from test_data import file_hash, make_database, task_rows, write_json, write_release

from text_to_sql_recipe import data, data_setup


def build_release(root: Path) -> tuple[dict, dict]:
    """A release, its hints, gate policy and subset, and an evaluation file; returns registry entries for them."""
    database = make_database(root / "toy" / "databases" / "fixture.sqlite")
    train, validation = task_rows(database, stored_path="../databases/fixture.sqlite", release_id="toy-1")
    _, manifest_pin = write_release(root / "toy", train, validation, release_id="toy-1")
    references = {"references": {str(row["id"]): {"raw": "join on n"} for row in train}}
    bindings = {str(row["id"]): {"current_role": "paper_train",
                                 "prompt_sha256": hashlib.sha256(row["problem"].encode()).hexdigest(),
                                 "gold_sql_sha256": hashlib.sha256(row["gold_sql"].encode()).hexdigest(),
                                 "database_sha256": row["database_sha256"]} for row in train}
    write_json(root / "toy" / "hints" / "training-references.json", references)
    write_json(root / "toy" / "hints" / "current-training-bindings.json", bindings)
    write_json(root / "toy" / "verifier" / "gate-policy.json",
               {"fixture": {"sha256": file_hash(database), "schema": [], "policies": []}})
    write_json(root / "toy" / "subsets" / "one.json", [str(train[0]["id"])])
    evaluation = [{**row, "database_path": "databases/fixture.sqlite"} for row in validation]
    make_database(root / "eval" / "databases" / "fixture.sqlite")
    write_json(root / "eval" / "tasks.json", evaluation)
    releases = {"toy": data.Release("toy", manifest_pin,
                                    file_hash(root / "toy" / "hints" / "training-references.json"),
                                    file_hash(root / "toy" / "hints" / "current-training-bindings.json"),
                                    file_hash(root / "toy" / "verifier" / "gate-policy.json"))}
    datasets = {
        "toy": data.Split("toy", "train"),
        "toy-validation": data.Split("toy", "validation"),
        "toy-one": data.Split("toy", "train", subset="one.json",
                              subset_sha256=file_hash(root / "toy" / "subsets" / "one.json"), subset_count=1),
        "toy-eval": data.TaskFile("eval/tasks.json", file_hash(root / "eval" / "tasks.json"), 2),
    }
    return releases, datasets


class SetupDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "release"
        self.target = Path(self.tmp.name) / "data"
        releases, datasets = build_release(self.source)
        for registry, values in ((data.RELEASES, releases), (data.DATASETS, datasets)):
            patcher = mock.patch.dict(registry, values, clear=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_setup(self, **kwargs):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = data_setup.run(data_dir=self.target, **kwargs)
        return code, out.getvalue(), err.getvalue()

    def test_a_local_release_is_copied_checked_and_exported(self):
        code, out, err = self.run_setup(source=self.source)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.strip(), f"export TEXT_TO_SQL_DATA_DIR={self.target.resolve()}")
        self.assertIn("toy-one: 1 tasks", err)
        self.assertIn("toy-eval: 2 tasks", err)
        # A second run over a complete directory changes nothing and passes again.
        self.assertEqual(self.run_setup(source=self.source)[0], 0)

    def test_a_file_that_does_not_match_its_pin_fails_the_check(self):
        self.assertEqual(self.run_setup(source=self.source)[0], 0)
        for relative in ("toy/hints/training-references.json", "toy/verifier/gate-policy.json",
                         "eval/databases/fixture.sqlite", "toy/databases/fixture.sqlite", "toy/subsets/one.json"):
            with self.subTest(relative=relative):
                path = self.target / relative
                original = path.read_bytes()
                path.unlink()   # a hard link into the source: replace the file, never edit through it
                path.write_bytes(original + b" ")
                with self.assertRaises((data_setup.SetupError, ValueError)):
                    data_setup.verify(self.target)
                path.unlink()
                path.write_bytes(original)
        data_setup.verify(self.target)

    def test_copying_another_release_never_writes_through_an_earlier_copy(self):
        # The first copy hard-links into the source; a second source must replace those links, not write through them.
        self.assertEqual(self.run_setup(source=self.source)[0], 0)
        manifest = self.source / "toy" / "dataset" / "manifest.json"
        original = manifest.read_bytes()
        other = Path(self.tmp.name) / "other"
        data_setup.copy_release(self.source, other)
        replacement = other / "toy" / "dataset" / "manifest.json"
        replacement.unlink()
        replacement.write_bytes(original + b"\n")
        data_setup.copy_release(other, self.target)
        self.assertEqual(manifest.read_bytes(), original)
        self.assertEqual((self.target / "toy" / "dataset" / "manifest.json").read_bytes(), original + b"\n")

    def test_a_changed_copy_of_the_same_size_is_replaced_by_the_next_run(self):
        copied = Path(self.tmp.name) / "copied"
        with mock.patch.object(data_setup.os, "link", side_effect=OSError("cross-device")):
            data_setup.copy_release(self.source, copied)   # real copies, as across filesystems
        database = copied / "toy" / "databases" / "fixture.sqlite"
        original = database.read_bytes()
        database.write_bytes(bytes(len(original)))   # corrupted in place: same size, newer modification time
        with self.assertRaisesRegex(ValueError, "fixture.sqlite"):   # the failure names the file
            data_setup.verify(copied)
        with mock.patch.object(data_setup.os, "link", side_effect=OSError("cross-device")):
            data_setup.copy_release(self.source, copied)
        self.assertEqual(database.read_bytes(), original)
        data_setup.verify(copied)

    def test_each_release_is_loaded_once(self):
        self.assertEqual(self.run_setup(source=self.source)[0], 0)
        with mock.patch.object(data, "load_release", wraps=data.load_release) as load:
            counts = data_setup.verify(self.target)
        self.assertEqual(load.call_count, 1)   # toy, toy-validation and toy-one share one release
        self.assertEqual(counts, {"toy": 1, "toy-eval": 2, "toy-one": 1, "toy-validation": 2})

    def test_the_gate_policy_must_cover_every_training_database(self):
        self.assertEqual(self.run_setup(source=self.source)[0], 0)
        policy = self.target / "toy" / "verifier" / "gate-policy.json"
        policy.unlink()
        write_json(policy, {"fixture": {"sha256": "0" * 64, "schema": [], "policies": []}})
        pinned = data.Release(*{**data.RELEASES["toy"].__dict__, "gate_policy_sha256": file_hash(policy)}.values())
        with (
            mock.patch.dict(data.RELEASES, {"toy": pinned}),
            self.assertRaisesRegex(data_setup.SetupError, "has no entry for 1 training database"),
        ):
            data_setup.verify(self.target)

    def test_a_failed_download_is_a_message_not_a_traceback(self):
        class ReadTimeout(Exception):   # huggingface_hub re-raises its HTTP library's errors, which are not OSErrors
            pass

        def snapshot_download(**kwargs):
            raise ReadTimeout("the read operation timed out")

        fake = types.SimpleNamespace(snapshot_download=snapshot_download)
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake}):
            code, out, err = self.run_setup()
        self.assertEqual((code, out), (1, ""))
        self.assertIn("ReadTimeout: the read operation timed out", err)
        self.assertIn("run the command again", err)

    def test_an_incomplete_release_exits_1_without_the_export_line(self):
        broken = Path(self.tmp.name) / "broken"
        data_setup.copy_release(self.source, broken)
        (broken / "toy" / "subsets" / "one.json").unlink()
        code, out, err = self.run_setup(source=broken)
        self.assertEqual((code, out), (1, ""))
        self.assertIn("setup data failed", err)

    def test_download_goes_through_huggingface_hub_at_the_pinned_revision(self):
        calls = []

        def snapshot_download(**kwargs):
            calls.append(kwargs)
            data_setup.copy_release(self.source, Path(kwargs["local_dir"]))

        fake = types.SimpleNamespace(snapshot_download=snapshot_download)
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake}):
            code, _, err = self.run_setup(repo="owner/data", revision="abc123")
        self.assertEqual(code, 0, err)
        self.assertEqual(calls, [{"repo_id": "owner/data", "repo_type": "dataset", "revision": "abc123",
                                  "local_dir": str(self.target.resolve())}])

    def test_downloading_without_huggingface_hub_says_what_to_install(self):
        with mock.patch.dict(sys.modules, {"huggingface_hub": None}):
            code, out, err = self.run_setup()
        self.assertEqual((code, out), (1, ""))
        self.assertIn("pip install '.[data]'", err)


class ReleaseDefaultsTest(unittest.TestCase):
    def test_the_default_release_is_a_hugging_face_dataset(self):
        self.assertRegex(data_setup.DATA_REPO, r"^[\w.-]+/[\w.-]+$")
        self.assertRegex(data_setup.DATA_REVISION, r"^[0-9a-f]{40}$")   # a commit, never a branch that can move


if __name__ == "__main__":
    unittest.main()
