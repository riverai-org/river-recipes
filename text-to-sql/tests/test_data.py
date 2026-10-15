"""Dataset loading: every check fails closed, rows come back as stored, and the registry covers every preset."""
import copy
import hashlib
import json
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from text_to_sql_recipe import config, data

ROOT = Path(__file__).resolve().parents[1]
PRESETS = sorted((ROOT / "src/text_to_sql_recipe/recipes").glob("*.toml"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_database(path, value=7):
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE t (n INTEGER)")
        connection.execute("INSERT INTO t VALUES (?)", (value,))
        connection.commit()
    return path


def task_rows(database, *, stored_path=None, release_id="fixture"):
    """One training row and two validation rows on one database."""
    rows = [
        {
            "id": str(i),
            "db_id": "fixture",
            "domain": "sql",
            "question": f"Item {i}?",
            "evidence": "",
            "problem": f"Item {i}: return n",
            "database_path": stored_path or str(database),
            "database_sha256": file_hash(database),
            "gold_sql": "SELECT 7" if i == 1 else "SELECT 8",
            "grading_method": "set",
            "numeric_policy": "diamond-numeric-v1",
            "numeric_tolerance": False,
            "timeout_seconds": 5,
            "dataset_release_id": release_id,
            "current_role": "paper_validation",
        }
        for i in (1, 2)
    ]
    train = [{**rows[0], "id": "train", "current_role": "paper_train", "grading_max_rows": 200_000}]
    return train, rows


def write_release(directory, train, validation, *, release_id="fixture", counts=None):
    """Write dataset/manifest.json and both runtime files; return (dataset root, manifest sha256)."""
    root = Path(directory) / "dataset"
    write_json(root / "runtime-train.json", train)
    write_json(root / "runtime-validation.json", validation)
    write_json(
        root / "manifest.json",
        {
            "release_id": release_id,
            "summary": {
                "active_by_role": counts or {"paper_train": len(train), "paper_validation": len(validation)}
            },
            "artifact_sha256": {
                name: file_hash(root / name) for name in ("runtime-train.json", "runtime-validation.json")
            },
        },
    )
    return root, file_hash(root / "manifest.json")


class LoadReleaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.database = make_database(self.dir / "fixture.sqlite")
        self.train, self.validation = task_rows(self.database)

    def load(self, train=None, validation=None, *, release_id="fixture", counts=None, pin=None):
        root, manifest_sha256 = write_release(
            self.dir / "release",
            self.train if train is None else train,
            self.validation if validation is None else validation,
            release_id=release_id,
            counts=counts,
        )
        return data.load_release(root, manifest_sha256=pin or manifest_sha256)

    def test_rows_come_back_as_stored(self):
        train, validation = self.load()
        self.assertEqual(train, self.train)
        self.assertEqual(validation, self.validation)
        self.assertEqual([row["id"] for row in validation], ["1", "2"])

    def test_manifest_pin_is_checked(self):
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            self.load(pin="0" * 64)
        # The pin covers the release id: a manifest naming another release is a different manifest.
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            root, _ = write_release(self.dir / "renamed", self.train, self.validation, release_id="another-release")
            data.load_release(root, manifest_sha256=write_release(self.dir / "pinned", self.train, self.validation)[1])

    def test_runtime_files_are_checked_against_the_manifest(self):
        root, manifest_sha256 = write_release(self.dir / "release", self.train, self.validation)
        write_json(root / "runtime-validation.json", self.validation[:1] + self.validation[:1])
        with self.assertRaisesRegex(ValueError, "runtime hash mismatch"):
            data.load_release(root, manifest_sha256=manifest_sha256)

    def test_split_counts_are_checked_against_the_manifest(self):
        with self.assertRaisesRegex(ValueError, "split count mismatch"):
            self.load(counts={"paper_train": 1, "paper_validation": 3})

    def test_splits_are_disjoint_and_roles_match(self):
        with self.assertRaisesRegex(ValueError, "overlap or role"):
            self.load(train=[{**self.train[0], "id": "1"}])
        with self.assertRaisesRegex(ValueError, "overlap or role"):
            self.load(validation=[self.validation[0], {**self.validation[1], "current_role": "paper_train"}])
        with self.assertRaisesRegex(ValueError, "overlap or role"):
            self.load(validation=[self.validation[0], {**self.validation[1], "id": "1"}])

    def test_numeric_policy_and_row_release_are_checked(self):
        for change in ({"numeric_policy": "legacy"}, {"numeric_tolerance": True}, {"numeric_tolerance": None}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "numeric policy mismatch"):
                self.load(validation=[self.validation[0], {**self.validation[1], **change}])
        missing = copy.deepcopy(self.validation)
        del missing[1]["numeric_tolerance"]
        with self.assertRaisesRegex(ValueError, "numeric policy mismatch"):
            self.load(validation=missing)
        with self.assertRaisesRegex(ValueError, "task release mismatch"):
            self.load(validation=[self.validation[0], {**self.validation[1], "dataset_release_id": "other"}])

    def test_databases_are_checked_and_read_once(self):
        with mock.patch.object(data, "open", wraps=open, create=True) as opened:
            self.load()
        self.assertEqual([c.args[0] for c in opened.call_args_list], [str(self.database)])
        make_database(self.database.with_name("other.sqlite"), value=8)
        changed = [{**self.validation[0], "database_path": str(self.database.with_name("other.sqlite"))}]
        with self.assertRaisesRegex(ValueError, r"database mismatch: .*other\.sqlite \(fixture\)"):
            self.load(validation=changed, counts={"paper_train": 1, "paper_validation": 1})

    def test_relative_database_paths_resolve_against_the_dataset_directory(self):
        database = make_database(self.dir / "release" / "dataset" / "databases" / "fixture.sqlite")
        train, validation = task_rows(database, stored_path="databases/fixture.sqlite")
        loaded_train, loaded_validation = self.load(train, validation)
        resolved = str((self.dir / "release" / "dataset" / "databases" / "fixture.sqlite").resolve())
        self.assertEqual({row["database_path"] for row in loaded_train + loaded_validation}, {resolved})

    def test_absolute_paths_are_kept_exactly_as_stored(self):
        stored = str(self.dir) + "/./fixture.sqlite"
        train, validation = task_rows(self.database, stored_path=stored)
        loaded_train, loaded_validation = self.load(train, validation)
        self.assertEqual({row["database_path"] for row in loaded_train + loaded_validation}, {stored})


class EvaluationTaskTest(unittest.TestCase):
    def test_arcwise_evaluation_uses_diamond_without_mutating_artifact(self):
        row = {"id": "arcwise:1", "gold_sql": "SELECT 24.0", "numeric_tolerance": True}
        patched = data.evaluation_task(row)
        self.assertEqual(patched["numeric_policy"], "diamond-numeric-v1")
        self.assertFalse(patched["numeric_tolerance"])
        self.assertTrue(row["numeric_tolerance"])
        self.assertNotIn("numeric_policy", row)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        database = make_database(self.dir / "arcwise" / "databases" / "fixture.sqlite")
        self.rows = [
            {"id": f"arcwise-sql:{i}", "db_id": "fixture", "problem": "p", "question": "q", "evidence": "",
             "database_path": "databases/fixture.sqlite" if i == 1 else str(database),
             "database_sha256": file_hash(database), "gold_sql": "SELECT 7", "grading_method": "set",
             "source_gold_sql": "SELECT 7", "source_gold_sql_sha256": "x", "gold_sql_overridden": False,
             "source": "fixture", "domain": "fixture"}
            for i in (1, 2, 3)
        ]
        self.path = self.dir / "arcwise" / "tasks.json"
        write_json(self.path, self.rows)
        self.database = database

    def test_task_file_is_pinned_counted_and_graded_by_the_evaluation_contract(self):
        tasks = data.load_task_file(self.path, sha256=file_hash(self.path), count=3)
        self.assertEqual([t["id"] for t in tasks], [r["id"] for r in self.rows])
        for task, row in zip(tasks, self.rows, strict=True):
            self.assertEqual(task["numeric_policy"], "diamond-numeric-v1")
            self.assertIs(task["numeric_tolerance"], False)
            self.assertEqual({k: v for k, v in task.items() if k not in ("numeric_policy", "numeric_tolerance",
                                                                          "database_path")},
                             {k: v for k, v in row.items() if k != "database_path"})
        self.assertEqual(tasks[0]["database_path"], str(self.database.resolve()))
        self.assertEqual(tasks[1]["database_path"], self.rows[1]["database_path"])

    def test_task_file_checks_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            data.load_task_file(self.path, sha256="0" * 64, count=3)
        with self.assertRaisesRegex(ValueError, "exactly 4 distinct"):
            data.load_task_file(self.path, sha256=file_hash(self.path), count=4)
        write_json(self.path, [self.rows[0], self.rows[0], self.rows[2]])
        with self.assertRaisesRegex(ValueError, "exactly 3 distinct"):
            data.load_task_file(self.path, sha256=file_hash(self.path), count=3)


class RegistryTest(unittest.TestCase):
    def test_every_preset_names_a_registered_train_and_validation_split_of_one_release(self):
        recipes = [config.RecipeConfig()] + [config.load(path) for path in PRESETS]
        self.assertEqual(len(recipes), 8)
        for recipe in recipes:
            with self.subTest(train=recipe.data.train, validation=recipe.data.validation):
                train, validation = data.DATASETS[recipe.data.train], data.DATASETS[recipe.data.validation]
                self.assertEqual((train.split, validation.split), ("train", "validation"))
                self.assertEqual(train.release, validation.release)
                self.assertIn(train.release, data.RELEASES)

    def test_registry_entries_are_pinned(self):
        for release in data.RELEASES.values():
            for pin in (release.manifest_sha256, release.hints_sha256, release.hint_bindings_sha256):
                self.assertRegex(pin, r"^[0-9a-f]{64}$")
        arcwise = data.DATASETS[data.ARCWISE]
        self.assertIsInstance(arcwise, data.TaskFile)
        self.assertRegex(arcwise.sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(arcwise.count, 498)

    def test_unknown_names_are_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown dataset 'birds'"):
            data.load_dataset("birds", "data")


class NamedDatasetTest(unittest.TestCase):
    """load_dataset, load_splits and hint_files against small releases registered for the test."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_dir = Path(self.tmp.name)
        database = make_database(self.data_dir / "toy" / "dataset" / "databases" / "fixture.sqlite")
        copied = self.data_dir / "other" / "dataset" / "databases" / "fixture.sqlite"
        copied.parent.mkdir(parents=True)
        shutil.copyfile(database, copied)
        self.train, self.validation = task_rows(database, stored_path="databases/fixture.sqlite", release_id="toy-1")
        _, toy_pin = write_release(self.data_dir / "toy", self.train, self.validation, release_id="toy-1")
        other_train = [{**self.validation[0], "current_role": "paper_train", "dataset_release_id": "other-1"}]
        other_validation = [{**self.train[0], "current_role": "paper_validation", "dataset_release_id": "other-1"}]
        _, other_pin = write_release(self.data_dir / "other", other_train, other_validation, release_id="other-1")
        releases = {
            "toy": data.Release("toy", toy_pin, "a" * 64, "b" * 64, "e" * 64),
            "other": data.Release("other", other_pin, "c" * 64, "d" * 64, "f" * 64),
        }
        datasets = {
            "toy": data.Split("toy", "train"),
            "toy-validation": data.Split("toy", "validation"),
            "other": data.Split("other", "train"),
            "other-validation": data.Split("other", "validation"),
            "toy-eval": data.TaskFile("toy/dataset/runtime-validation.json",
                                      file_hash(self.data_dir / "toy" / "dataset" / "runtime-validation.json"), 2),
        }
        for target, values in ((data.RELEASES, releases), (data.DATASETS, datasets)):
            patcher = mock.patch.dict(target, values, clear=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_load_dataset_returns_one_split(self):
        self.assertEqual([r["id"] for r in data.load_dataset("toy", self.data_dir)], ["train"])
        self.assertEqual([r["id"] for r in data.load_dataset("toy-validation", self.data_dir)], ["1", "2"])
        evaluation = data.load_dataset("toy-eval", self.data_dir)
        self.assertEqual([r["id"] for r in evaluation], ["1", "2"])

    def test_dataset_names_are_checked_without_reading_data(self):
        self.assertEqual(data.check_names("toy", "toy-validation"), [])
        problems = data.check_names("toy-validation", "toy-eval")
        self.assertEqual(len(problems), 2)
        self.assertIn("data.train must name a train split (other, toy)", problems[0])

    def test_a_missing_dataset_says_how_to_get_it(self):
        data.require_present(["toy", "toy-validation", "toy-eval"], self.data_dir)
        with self.assertRaisesRegex(ValueError, "setup data") as raised:
            data.require_present(["toy", "toy-validation", "toy-eval"], self.data_dir / "elsewhere")
        self.assertEqual(str(raised.exception).count("manifest.json"), 1)   # named once for its two splits

    def test_load_splits_loads_a_shared_release_once(self):
        with mock.patch.object(data, "load_release", wraps=data.load_release) as load:
            train, validation = data.load_splits("toy", "toy-validation", self.data_dir)
        self.assertEqual(load.call_count, 1)
        self.assertEqual([r["id"] for r in train], ["train"])
        self.assertEqual([r["id"] for r in validation], ["1", "2"])
        resolved = str((self.data_dir / "toy" / "dataset" / "databases" / "fixture.sqlite").resolve())
        self.assertEqual({r["database_path"] for r in train + validation}, {resolved})

    def test_load_splits_refuses_wrong_kinds_and_overlap(self):
        for train, validation in (("toy-validation", "toy"), ("toy", "toy"), ("toy-eval", "toy-validation")):
            with self.subTest(train=train, validation=validation), self.assertRaisesRegex(ValueError, "split"):
                data.load_splits(train, validation, self.data_dir)
        with self.assertRaisesRegex(ValueError, "overlap the training pool"):
            data.load_splits("other", "toy-validation", self.data_dir)

    def test_hint_files_belong_to_a_train_split(self):
        hints = data.hint_files("toy", self.data_dir)
        self.assertEqual(hints, (self.data_dir / "toy" / "hints" / "training-references.json", "a" * 64,
                                 self.data_dir / "toy" / "hints" / "current-training-bindings.json", "b" * 64))
        for name in ("toy-validation", "toy-eval"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "train split"):
                data.hint_files(name, self.data_dir)

    def test_a_subset_narrows_its_split_to_the_pinned_task_list(self):
        validation_ids = [r["id"] for r in self.validation]
        subset = self.data_dir / "toy" / "subsets" / "one.json"
        subset.parent.mkdir()
        subset.write_text(json.dumps([validation_ids[1]]))
        pin = file_hash(subset)
        with mock.patch.dict(data.DATASETS, {
            "toy-one": data.Split("toy", "validation", subset="one.json", subset_sha256=pin, subset_count=1),
            "toy-one-bad-pin": data.Split("toy", "validation", subset="one.json", subset_sha256="0" * 64, subset_count=1),
            "toy-one-bad-count": data.Split("toy", "validation", subset="one.json", subset_sha256=pin, subset_count=2),
            "toy-one-wrong-split": data.Split("toy", "train", subset="one.json", subset_sha256=pin, subset_count=1),
        }):
            self.assertEqual([r["id"] for r in data.load_dataset("toy-one", self.data_dir)], [validation_ids[1]])
            _, validation = data.load_splits("toy", "toy-one", self.data_dir)
            self.assertEqual([r["id"] for r in validation], [validation_ids[1]])
            for name, message in (("toy-one-bad-pin", "subset hash mismatch"), ("toy-one-bad-count", "exactly 2"),
                                  ("toy-one-wrong-split", "outside its train split")):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                    data.load_dataset(name, self.data_dir)

    def test_gate_policy_belongs_to_a_train_split(self):
        self.assertEqual(data.gate_policy_file("toy", self.data_dir),
                         (self.data_dir / "toy" / "verifier" / "gate-policy.json", "e" * 64))
        for name in ("toy-validation", "toy-eval"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "train split"):
                data.gate_policy_file(name, self.data_dir)


class TaskManifestTest(unittest.TestCase):
    def test_digest_is_ordered_lines(self):
        self.assertEqual(data.task_manifest_sha256(["a", "b"]), hashlib.sha256(b"a\nb\n").hexdigest())
        self.assertNotEqual(data.task_manifest_sha256(["a", "b"]), data.task_manifest_sha256(["b", "a"]))

    def test_malformed_manifests_are_refused(self):
        for ids, message in (([], "at least one"), (["a", "a"], "repeat"), (["a\nb"], "line separator")):
            with self.subTest(ids=ids), self.assertRaisesRegex(RuntimeError, message):
                data.task_manifest_sha256(ids)



class HintMaskTest(unittest.TestCase):
    def test_the_mask_names_training_tasks(self):
        self.assertEqual(len(data.HINT_MASK), 31)
        for task_id in data.HINT_MASK:
            self.assertRegex(task_id, r"^revisql-train:\d+$")

    def test_masked_rows_are_left_out_and_the_rest_keep_their_order(self):
        rows = [{"id": "revisql-train:7"}, {"id": "revisql-train:33"}, {"id": "revisql-train:3"},
                {"id": "revisql-train:1538"}, {"id": "revisql-validation:33"}]
        self.assertEqual(data.apply_hint_mask(rows),
                         [{"id": "revisql-train:7"}, {"id": "revisql-train:3"}, {"id": "revisql-validation:33"}])

if __name__ == "__main__":
    unittest.main()
