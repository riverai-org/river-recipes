"""Datasets: named task sets, each loaded only after every file passes its pinned SHA-256 check.

A recipe names its datasets in `DataConfig.train` and `DataConfig.validation`; evaluation also runs the Arcwise set
(`ARCWISE`). Names resolve under the data directory (`TEXT_TO_SQL_DATA_DIR`, default `./data`; see
`config.connection`), which `python -m text_to_sql_recipe setup data` fills (`data_setup`), laid out as:

    revisql-verified/                 a release: a train and a validation split published together
      dataset/
        manifest.json                 release_id; artifact_sha256 of each runtime file; summary.active_by_role counts
        runtime-train.json            JSON list of task rows, current_role "paper_train"
        runtime-validation.json       JSON list of task rows, current_role "paper_validation"
      hints/
        training-references.json        a training hint per training task (read by the curriculum)
        current-training-bindings.json  the prompt, gold query and database each hint was written for
      verifier/
        gate-policy.json                per training database, how the VeriEQL counterexample gate confirms a
                                        refutation (read only with reward.verifier = "verieql" and the gate on); a
                                        task finds its entry by its database's SHA-256
      subsets/
        bird-platinum-hard.json         a JSON list of task ids: a named subset of the training split
      databases/                      the SQLite databases the rows name
    arcwise/
      arcwise498.json                 JSON list of the 498 evaluation task rows
      databases/                      the SQLite databases they name

Each row names its SQLite database in `database_path`: an absolute path, or a path relative to the directory of the
file that lists the row (`dataset/` for a release, `arcwise/` for the evaluation set). A relative path is rewritten
to the resolved path; an absolute one is kept exactly as stored.

`load_release` checks, before returning any row: manifest.json against the registry's pin (which fixes the release id); each
runtime file against the manifest; each split's row count against the manifest; that no task id repeats across the
splits and every row has its split's role; every row's numeric policy and release id; and every database file
against the row's `database_sha256` (each file is read once). `load_task_file` checks the evaluation file against
its pin and its exact number of distinct tasks, then applies the evaluation grading contract
(`evaluation_task`); `train` then sets the Arcwise rows' `numeric_tolerance` from
`evaluation.arcwise_grading`.

Rows are otherwise returned as stored, so every field reaches the environment and the grader unchanged. Every row
of a release has `id`, `db_id`, `domain`, `question`, `evidence`, `problem`, `gold_sql`, `grading_method`,
`database_path`, `database_sha256`, `current_role`, `dataset_release_id`, `numeric_policy` ("diamond-numeric-v1")
and `numeric_tolerance` (false). Some rows add per-task grading limits: `timeout_seconds`, `gold_timeout_seconds`,
`grading_max_rows`, `grading_max_bytes`, `streaming_set_grading`, `grading_max_index_bytes` (training sets where
the streamed index lives, `grading.prepare_streaming_grading_storage`).
The remaining fields record provenance: `source`, `source_gold_sql`, `source_gold_sql_sha256` and
`gold_sql_overridden` (the grader copies the last two into its receipt), and on Arcwise rows `gold_sql_source`, the
edition each gold query comes from. Arcwise rows carry the same task fields except `current_role` and
`dataset_release_id`.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

# Training questions the hint mask leaves out (`DataConfig.hint_mask`, on by default). In each, the gold query does
# not answer the question as written: a join or filter the question does not ask for, an output form it does not ask
# for (a code instead of a name, one wide row instead of one row per item, an extra column, scaling or rounding it does
# not mention), or, for 1538, an answer that moves with today's date while gold results are computed once per run.
# Training on them rewards a wrong answer, and their training hints teach it.
HINT_MASK = frozenset(f"revisql-train:{n}" for n in (
    33, 340, 535, 536, 596, 1000, 1234, 1501, 1538, 1558, 2155, 2174, 2415, 2441, 2807, 2934, 3207, 3266, 4071, 4416,
    4718, 4733, 4792, 5196, 5198, 6727, 8742, 9023, 9047, 9081, 9087,
))


def apply_hint_mask(rows: list[dict]) -> list[dict]:
    """The rows whose task ids are not in `HINT_MASK`, in their original order."""
    return [row for row in rows if str(row["id"]) not in HINT_MASK]

@dataclass(frozen=True)
class Release:
    """Train and validation splits published together under one manifest, with their training hints."""

    directory: str              # under the data directory
    manifest_sha256: str        # dataset/manifest.json, which pins the release id and both runtime files
    hints_sha256: str           # hints/training-references.json
    hint_bindings_sha256: str   # hints/current-training-bindings.json
    gate_policy_sha256: str     # verifier/gate-policy.json, read by the VeriEQL counterexample gate


@dataclass(frozen=True)
class Split:
    release: str                # key in RELEASES
    split: str                  # "train" or "validation"
    subset: str | None = None   # only the tasks listed in <release>/subsets/<subset>, a JSON list of task ids
    subset_sha256: str | None = None
    subset_count: int | None = None


@dataclass(frozen=True)
class TaskFile:
    path: str                   # under the data directory
    sha256: str
    count: int                  # exact number of distinct tasks


class HintFiles(NamedTuple):
    """The training-hint files of a release with their pins, in the order the hint loader takes them."""

    references: Path
    references_sha256: str
    bindings: Path
    bindings_sha256: str


RELEASES = {
    # ReViSQL's verified BIRD tasks (BIRD-Platinum) as the authors released them: 2,064 training and 398 validation
    # tasks over BIRD's training databases. `setup data` downloads the release.
    "revisql-verified": Release(
        directory="revisql-verified",
        manifest_sha256="c297521a167f417fe75d022dfbbf8874de21e0e0cfe7738024a8fd5483bcb768",
        hints_sha256="ea8d1c5d84a7cc60e465db26dc357c4c7a878e58e6625d83432e21a5e505097d",
        hint_bindings_sha256="897000ae65b2d7b56df712270da4586d063316929ce6b845e163c34b792db46c",
        gate_policy_sha256="9daa051f1f2dd3ca8723e15dd272f57e5c14acad3f64dd2ae6d023d7d66fca39",
    ),
}

# The Arcwise-Plat-SQL set: 498 BIRD Mini-Dev tasks, used for evaluation only, with the gold queries the ReViSQL paper
# evaluated (uiuc-kang-lab/text_to_sql_benchmarks@bef1f2e42b).
ARCWISE = "arcwise"

DATASETS: dict[str, Split | TaskFile] = {
    "revisql-verified": Split("revisql-verified", "train"),
    "revisql-verified-validation": Split("revisql-verified", "validation"),
    # BIRD-Platinum-Hard: the 689 training tasks strong models most often failed on every attempt early in training.
    "bird-platinum-hard": Split("revisql-verified", "train", subset="bird-platinum-hard.json",
                                subset_sha256="b8e193cc5f9e0a86865c06bc533b02e6256ddaff60ce0dabdbe3000b63796990",
                                subset_count=689),
    ARCWISE: TaskFile(
        path="arcwise/arcwise498.json",
        sha256="09906a1dd9c0442d4bee18c2080ebdf2f0f485df1bcff22487b2ca2763094523",
        count=498,
    ),
}


def evaluation_task(row):
    """Apply the evaluation contract without changing the pinned task artifact."""
    return {**row, "numeric_policy": "diamond-numeric-v1", "numeric_tolerance": False}


def load_release(root, *, manifest_sha256):
    """The (train, validation) rows of the release whose dataset/ directory is `root`, after every check.

    The manifest pin fixes the release id; every row must name that release.
    """
    root = Path(root)
    raw = (root / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError(f"dataset manifest hash mismatch: {root / 'manifest.json'}")
    manifest = json.loads(raw)
    release_id = manifest["release_id"]
    result = []
    seen = set()
    verified = set()
    for split, role in (("train", "paper_train"), ("validation", "paper_validation")):
        name = f"runtime-{split}.json"
        raw = (root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["artifact_sha256"][name]:
            raise ValueError(f"dataset runtime hash mismatch: {root / name}")
        rows = json.loads(raw)
        if len(rows) != manifest["summary"]["active_by_role"][role]:
            raise ValueError("dataset split count mismatch")
        for row in rows:
            if row["id"] in seen or row["current_role"] != role:
                raise ValueError("dataset split overlap or role mismatch")
            seen.add(row["id"])
            if (
                row.get("numeric_policy") != "diamond-numeric-v1"
                or row.get("numeric_tolerance") is not False
            ):
                raise ValueError("dataset numeric policy mismatch")
            if row["dataset_release_id"] != release_id:
                raise ValueError("dataset task release mismatch")
            if not Path(row["database_path"]).is_absolute():
                row["database_path"] = str((root / row["database_path"]).resolve())
            identity = (row["database_path"], row["database_sha256"])
            if identity not in verified:
                digest = hashlib.sha256()
                with open(identity[0], "rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != identity[1]:
                    raise ValueError(f"dataset database mismatch: {identity[0]} ({row['db_id']})")
                verified.add(identity)
        result.append(rows)
    return tuple(result)


def load_task_file(path, *, sha256, count):
    """Evaluation tasks from one pinned JSON file, with the evaluation grading contract applied."""
    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"evaluation task file hash mismatch: {path}")
    rows = json.loads(raw)
    if len(rows) != count or len({str(r["id"]) for r in rows}) != count:
        raise ValueError(f"evaluation task file must hold exactly {count} distinct tasks")
    tasks = [evaluation_task(row) for row in rows]
    for task in tasks:
        if not Path(task["database_path"]).is_absolute():
            task["database_path"] = str((path.parent / task["database_path"]).resolve())
    return tasks


def _spec(name: str) -> Split | TaskFile:
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; known datasets: {', '.join(sorted(DATASETS))}") from None


def _load(release: str, data_dir: str | Path) -> tuple[list[dict], list[dict]]:
    spec = RELEASES[release]
    return load_release(
        Path(data_dir) / spec.directory / "dataset",
        manifest_sha256=spec.manifest_sha256,
    )


def _select(rows: list[dict], spec: Split, data_dir: str | Path) -> list[dict]:
    """A split's rows, narrowed to its subset's pinned task-id list when it names one (release order kept)."""
    if spec.subset is None:
        return rows
    path = Path(data_dir) / RELEASES[spec.release].directory / "subsets" / spec.subset
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec.subset_sha256:
        raise ValueError(f"dataset subset hash mismatch: {path}")
    ids = [str(task_id) for task_id in json.loads(raw)]
    if len(ids) != spec.subset_count or len(set(ids)) != len(ids):
        raise ValueError(f"dataset subset {spec.subset} must list exactly {spec.subset_count} distinct tasks")
    wanted = set(ids)
    selected = [row for row in rows if str(row["id"]) in wanted]
    if len(selected) != len(wanted):
        raise ValueError(f"dataset subset {spec.subset} names tasks outside its {spec.split} split")
    return selected


def load_dataset(name: str, data_dir: str | Path) -> list[dict]:
    """The tasks of one named dataset. A split name verifies its whole release, as both splits share one manifest."""
    return load_datasets([name], data_dir)[name]


def load_datasets(names: Iterable[str], data_dir: str | Path) -> dict[str, list[dict]]:
    """The tasks of several named datasets, checking each release once however many of its splits are named."""
    releases: dict[str, tuple[list[dict], list[dict]]] = {}
    loaded = {}
    for name in names:
        spec = _spec(name)
        if isinstance(spec, TaskFile):
            loaded[name] = load_task_file(Path(data_dir) / spec.path, sha256=spec.sha256, count=spec.count)
            continue
        if spec.release not in releases:
            releases[spec.release] = _load(spec.release, data_dir)
        train, validation = releases[spec.release]
        loaded[name] = _select(train if spec.split == "train" else validation, spec, data_dir)
    return loaded


def check_names(train: str, validation: str) -> list[str]:
    """What is wrong with a recipe's dataset names, without reading any data (empty when nothing is)."""
    problems = []
    for name, split in ((train, "train"), (validation, "validation")):
        spec = DATASETS.get(name)
        if not isinstance(spec, Split) or spec.split != split:
            known = ", ".join(sorted(n for n, s in DATASETS.items() if isinstance(s, Split) and s.split == split))
            problems.append(f"data.{split} must name a {split} split ({known}), not {name!r}")
    return problems


def require_present(names: Iterable[str], data_dir: str | Path) -> None:
    """Fail at once, saying what to run, when the data directory lacks a named dataset; checks no contents."""
    missing = []
    for name in names:
        spec = _spec(name)
        path = (Path(data_dir) / spec.path if isinstance(spec, TaskFile)
                else Path(data_dir) / RELEASES[spec.release].directory / "dataset" / "manifest.json")
        if not path.is_file() and str(path) not in missing:
            missing.append(str(path))
    if missing:
        raise ValueError(f"the data directory {Path(data_dir).resolve()} lacks {', '.join(missing)}: run "
                         "`python -m text_to_sql_recipe setup data`, or set TEXT_TO_SQL_DATA_DIR to the directory it "
                         "filled")


def load_splits(train: str, validation: str, data_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """Training and validation tasks for a recipe: each release is loaded once, and no task may be in both."""
    specs = {}
    for name, split in ((train, "train"), (validation, "validation")):
        spec = _spec(name)
        if not isinstance(spec, Split) or spec.split != split:
            raise ValueError(f"dataset {name!r} is not a {split} split")
        specs[split] = spec
    loaded = {}
    for spec in specs.values():
        if spec.release not in loaded:
            loaded[spec.release] = _load(spec.release, data_dir)
    train_rows = _select(loaded[specs["train"].release][0], specs["train"], data_dir)
    validation_rows = _select(loaded[specs["validation"].release][1], specs["validation"], data_dir)
    if {str(row["id"]) for row in train_rows} & {str(row["id"]) for row in validation_rows}:
        raise ValueError("validation tasks overlap the training pool")
    return train_rows, validation_rows


def hint_files(name: str, data_dir: str | Path) -> HintFiles:
    """Where the training hints of a train split's release live, with their pins."""
    spec = _spec(name)
    if not isinstance(spec, Split) or spec.split != "train":
        raise ValueError(f"training hints belong to a train split, not {name!r}")
    release = RELEASES[spec.release]
    root = Path(data_dir) / release.directory / "hints"
    return HintFiles(
        root / "training-references.json",
        release.hints_sha256,
        root / "current-training-bindings.json",
        release.hint_bindings_sha256,
    )


def gate_policy_file(name: str, data_dir: str | Path) -> tuple[Path, str]:
    """Where the VeriEQL gate policy of a train split's release lives, with its pin.

    The policy lists, per training database, how the counterexample gate may confirm a refutation; the verifier
    checks that it covers every training task's database.
    """
    spec = _spec(name)
    if not isinstance(spec, Split) or spec.split != "train":
        raise ValueError(f"the gate policy belongs to a train split, not {name!r}")
    release = RELEASES[spec.release]
    return Path(data_dir) / release.directory / "verifier" / "gate-policy.json", release.gate_policy_sha256


def task_manifest_sha256(task_ids: list[str]) -> str:
    """Digest an ordered task list: one id per line, newline-terminated, UTF-8.

    Order is part of the claim, so this never sorts. Recording it for the tasks
    a run used, and re-deriving it later, is what catches a dataset that has
    silently drifted underneath the recipe.
    """
    if not task_ids:
        raise RuntimeError("a task manifest states at least one task id")
    if len(set(task_ids)) != len(task_ids):
        raise RuntimeError("a task manifest cannot repeat a task id")
    if any("\n" in task_id for task_id in task_ids):
        raise RuntimeError("a task id cannot contain the line separator")
    body = "".join(f"{task_id}\n" for task_id in task_ids)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()
