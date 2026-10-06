"""`python -m text_to_sql_recipe setup data`: download the recipe's data release and check every file in it.

The release is a Hugging Face dataset repository (`DATA_REPO`, at `DATA_REVISION`) laid out the way `data` reads it:

- `revisql-verified/`: the BIRD-Platinum tasks the presets train on (the ReViSQL authors' verified release), with
  their training hints, the VeriEQL gate policy, the BIRD-Platinum-Hard subset and BIRD's 69 training databases;
- `arcwise/`: the Arcwise-Plat-SQL evaluation set, with the gold queries the ReViSQL paper evaluated, and the 11
  BIRD Mini-Dev databases.

About 34 GB in all. The task data is licensed CC BY-SA 4.0, like BIRD it derives from; the repository's card has the
attributions.

Nothing is trusted because it was downloaded: the command then loads every dataset the recipe names with the same
checks a training run makes, so every file must match the package's pins and every database the SHA-256 of the rows
that name it. `--source DIR` copies a local copy of the release instead of downloading it (a mirror, or a release
built elsewhere); downloading needs huggingface_hub (`pip install '.[data]'`).

Progress goes to stderr and the `export TEXT_TO_SQL_DATA_DIR=...` line to stdout, so
`eval "$(python -m text_to_sql_recipe setup data)"` sets it in the current shell.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

from . import curriculum, data

DATA_REPO = "scottfitz/text-to-sql-recipe-data"
# The commit the package's file pins describe; a later push to the repository cannot change what this downloads.
DATA_REVISION = "124ebc1c80f7e21d388d880e7686d381fef383aa"
SIZE = "about 34 GB"


class SetupError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def download(data_dir: Path, *, repo: str, revision: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SetupError("downloading the data needs huggingface_hub: pip install '.[data]'") from None
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", revision=revision, local_dir=str(data_dir))
    except Exception as error:   # network and HTTP errors, which huggingface_hub raises in several families
        raise SetupError(f"downloading {repo} at {revision} failed ({type(error).__name__}: {error}); run the "
                         "command again to resume") from error


def _unchanged(path: Path, target: Path) -> bool:
    """Whether `target` already holds `path`: the same file (a hard link), or a copy of it left untouched since."""
    if not target.is_file() or target.is_symlink():
        return False
    if os.path.samefile(path, target):
        return True
    source, existing = path.stat(), target.stat()
    return (source.st_size, source.st_mtime_ns) == (existing.st_size, existing.st_mtime_ns)


def copy_release(source: Path, data_dir: Path) -> None:
    """Copy (hard-link where the filesystem allows) every file of a local release into the data directory.

    A file already there is replaced unless it is the same file or an untouched copy (same size and modification
    time). A replacement is written under a temporary name and renamed over the old entry, so nothing is ever written
    through an existing file, which may be a hard link into another copy of the release or a symlink into a cache.
    """
    if not source.is_dir():
        raise SetupError(f"{source} is not a directory")
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        target = data_dir / path.relative_to(source)
        if _unchanged(path, target):
            continue   # still checked below, like every other file
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.partial")
        temporary.unlink(missing_ok=True)
        try:
            os.link(path, temporary)
        except OSError:
            shutil.copy2(path, temporary)
        os.replace(temporary, target)


def _check_file(path: Path, sha256: str) -> None:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise SetupError(f"{path} does not match its pinned SHA-256")


def verify(data_dir: Path) -> dict[str, int]:
    """Load every named dataset as a training run would; returns each dataset's task count.

    A release's loader (run once per release) checks its manifest, runtime files and every database its rows name;
    a train split also checks its training hints against the rows they were written for, and its gate policy against
    its pin and for an entry for every database; the evaluation set checks its file, and here also every database it
    names. A file that fails is named, so it can be deleted and fetched again.
    """
    datasets = data.load_datasets(sorted(data.DATASETS), data_dir)
    for name, rows in datasets.items():
        spec = data.DATASETS[name]
        if isinstance(spec, data.Split) and spec.split == "train" and spec.subset is None:
            hints = data.hint_files(name, data_dir)
            curriculum.load_hints(hints.references, hints.references_sha256, hints.bindings, hints.bindings_sha256,
                                  rows)
            path, sha256 = data.gate_policy_file(name, data_dir)
            _check_file(path, sha256)
            covered = {entry["sha256"] for entry in json.loads(path.read_bytes()).values()}
            uncovered = sorted({row["database_path"] for row in rows if row["database_sha256"] not in covered})
            if uncovered:
                raise SetupError(f"{path} has no entry for {len(uncovered)} training database(s), such as "
                                 f"{uncovered[0]}")
        if isinstance(spec, data.TaskFile):
            for database, sha256 in sorted({(row["database_path"], row["database_sha256"]) for row in rows}):
                _check_file(Path(database), sha256)
    return {name: len(rows) for name, rows in datasets.items()}


def run(*, data_dir: str | os.PathLike[str] | None = None, source: str | os.PathLike[str] | None = None,
        repo: str = DATA_REPO, revision: str = DATA_REVISION) -> int:
    target = Path(data_dir or os.environ.get("TEXT_TO_SQL_DATA_DIR") or "data").resolve()
    try:
        target.mkdir(parents=True, exist_ok=True)
        if source is not None:
            log(f"Copying the data release from {source} into {target}")
            copy_release(Path(source).resolve(), target)
        else:
            log(f"Downloading {repo} at {revision} ({SIZE}) into {target}")
            download(target, repo=repo, revision=revision)
    except (SetupError, OSError) as error:
        log(f"setup data failed: {error}")
        return 1
    log("Checking every file against the recipe's pins")
    try:
        counts = verify(target)
    except (SetupError, ValueError, KeyError, OSError) as error:
        log(f"setup data failed: {error}")
        log("If the message names a file, delete it and run the command again to fetch it again.")
        return 1
    for name, count in counts.items():
        log(f"  {name}: {count} tasks")
    print(f"export TEXT_TO_SQL_DATA_DIR={shlex.quote(str(target))}")
    return 0
