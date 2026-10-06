"""Restart training from durable state after transient failures.

A multi-day run outlives incidents on the serving side: inference workers are replaced during rollouts, the API
restarts, a pool runs out of capacity for a while. The API retries a failed request itself and reports a failure
only once its own retries are spent; without recovery that one failure would end the run. Training state is saved
after every optimizer step, so the trainer can resume from it: restoring weights, committed steps, rollout state and
the curriculum, and publishing the sampling snapshot again. An infrastructure failure never becomes a zero-reward
training example.

Recovery is bounded. Restarts from one saved step wait longer each time (`RECOVERY_DELAYS_SECONDS`) and stop when
that schedule is spent, so an error that keeps recurring without progress still ends the run. A restart that commits
new optimizer steps resets the count. `RECOVERY_LIMIT` caps restarts over the whole run. The history is kept in
`recovery.json` in the run directory, so the limits hold across driver restarts too.

Every start, in-process or a fresh process after a hard kill, first reconciles the run directory with the trainer's
saved progress (`reconcile`). The trainer saves its state some time after it reports a step, so a process can die
with steps reported but not saved; those steps will be trained again. Their step records, their evaluations and a
collapse-guard state that ran ahead are set aside under `rolled-back/`, so nothing reported describes weights the run
no longer has. Each optimizer step's gradient norm and clip threshold are written to `optimizer-steps.jsonl` the
moment River acknowledges the step (`OptimizerLog`). The trainer can only save a step after that, so every saved step
has its record, and the adaptive clip's history is rebuilt from it rather than from the diagnostic step log.

Only errors classified as transient are recovered: the serving failures below, plus whatever the trainer's own
classifier adds (for example infrastructure timeouts during grading). Everything else, including a request a worker
rejected as invalid, fails closed.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .files import atomic_json

# Terminal error categories the API reports for a request after spending its own retries.
SERVING_FAILURE_CATEGORIES = frozenset(
    {
        "inference_worker_retry_exhausted",
        "inference_processor_lost",
        "scan_window_expired",
        "checkpoint_save_retry_exhausted",
        "checkpoint_recovery_exhausted",
        "inference_export_memory_save_timeout",
        "inference_export_sharded_persist_timeout",
    }
)
# Connection failures that mean the service is briefly unreachable, not that the request is wrong.
TRANSIENT_CONNECTION_STATUSES = frozenset({"UNAVAILABLE", "DEADLINE_EXCEEDED", "502", "503", "504"})
RECOVERY_DELAYS_SECONDS = (30, 60, 120, 240, 480, 900)
RECOVERY_LIMIT = 20
RECEIPT = "recovery.json"
STEP_LOG = "steps.jsonl"
OPTIMIZER_LOG = "optimizer-steps.jsonl"
ROLLED_BACK = "rolled-back"


def recorded_steps(path: Path) -> list[dict]:
    """The step records in `steps.jsonl`, ignoring a final line cut short by a crash mid-write."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
    return records


def repair_step_log(path: Path) -> Path | None:
    """Make `steps.jsonl` safe to append to after a crash mid-write; returns the quarantine file, if any.

    A record cut short by a crash is the log's final, unterminated line, and appending after it would fuse the next
    record into it. The fragment moves to a quarantine file beside the log and the log is cut back to its last complete
    record. A final record that is complete but lacks its newline gets one.
    """
    if not path.exists():
        return None
    raw = path.read_bytes()
    if not raw or raw.endswith(b"\n"):
        return None
    cut = raw.rfind(b"\n") + 1
    tail = raw[cut:]
    try:
        complete = isinstance(json.loads(tail), dict)
    except ValueError:
        complete = False
    if complete:
        with path.open("ab") as stream:
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        return None
    quarantine = path.with_name(f"{path.name}.torn-{time.time_ns()}")
    with quarantine.open("wb") as stream:
        stream.write(tail)
        stream.flush()
        os.fsync(stream.fileno())
    with path.open("r+b") as stream:
        stream.truncate(cut)
        stream.flush()
        os.fsync(stream.fileno())
    return quarantine


def saved_progress(run_dir: Path, steps_per_epoch: int) -> tuple[int, int]:
    """The trainer's saved progress: (training batches, optimizer step) of its newest saved state, or (0, 0)."""
    best = (0, 0)
    for state_path in (Path(run_dir) / "epochs").glob("*/trainer.json"):
        saved = json.loads(state_path.read_text())
        epoch = int(state_path.parent.name)
        # Every epoch before the newest one completed all of its batches.
        best = max(best, ((epoch - 1) * steps_per_epoch + int(saved.get("n", 0)), int(saved.get("model_step", 0))))
    return best


class OptimizerLog:
    """Each optimizer step's gradient norm and clip threshold, written durably the moment River acknowledges it."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: dict[int, dict[str, Any]] = {}

    def load(self, through_step: int) -> list[dict[str, Any]]:
        """The records of steps up to `through_step`, oldest first; a later record of a step replaces an earlier one."""
        self.entries = {}
        for record in recorded_steps(self.path):
            if record["model_step"] <= through_step:
                self.entries[record["model_step"]] = record
        return [self.entries[step] for step in sorted(self.entries)]

    def record(self, model_step: int, metrics: dict[str, Any], threshold: float | None) -> dict[str, Any]:
        norm = metrics.get("grad_norm")
        entry = {
            "model_step": model_step,
            "grad_norm": norm if isinstance(norm, (int, float)) and math.isfinite(norm) else None,
            "grad_norm_finite": metrics.get("grad_norm_finite"),
            "grad_clip_norm": threshold,
            "recorded_at": time.time(),
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.entries[model_step] = entry
        return entry


def reconcile(run_dir: Path, steps_per_epoch: int) -> tuple[int, int]:
    """Bring the run directory back to the trainer's saved progress; returns it as (training batches, optimizer step).

    Step records and evaluations past the saved progress, a collapse-guard state that ran ahead of it, and the torn
    final line of a log are moved under `rolled-back/<time>/`, keeping their relative paths.
    """
    run_dir = Path(run_dir)
    batches, model_step = saved_progress(run_dir, steps_per_epoch)
    aside = run_dir / ROLLED_BACK / str(time.time_ns())
    moved: list[str] = []

    def set_aside(path: Path, contents: str | None = None) -> None:
        target = aside / path.relative_to(run_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        if contents is None:
            path.rename(target)
        else:
            target.write_text(contents)
        moved.append(str(path.relative_to(run_dir)))

    for name, key, limit in ((STEP_LOG, "batch", batches), (OPTIMIZER_LOG, "model_step", model_step)):
        path = run_dir / name
        torn = repair_step_log(path)
        if torn is not None:
            set_aside(torn)
        if not path.exists():
            continue
        lines = path.read_text().splitlines()
        kept = [line for line in lines if json.loads(line)[key] <= limit]
        if len(kept) != len(lines):
            set_aside(path, "\n".join(line for line in lines if line not in kept) + "\n")
            temporary = path.with_name(path.name + ".tmp")
            with temporary.open("w") as stream:
                stream.write("".join(line + "\n" for line in kept))
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
    numbered = re.compile(r"(?:checkpoint-)?(\d+)\.json|.+-step-(\d+)")
    for directory in (run_dir / "pending-evaluations", run_dir, run_dir / "eval"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            match = numbered.fullmatch(path.name)
            if not match or (directory == run_dir and not path.name.startswith("checkpoint-")):
                continue
            if int(match.group(1) or match.group(2)) > batches:
                set_aside(path)
    collapse = run_dir / "collapse-state.json"
    # The guard records the batch it is about to train, so it may lead the saved progress by one.
    if collapse.exists() and json.loads(collapse.read_text()).get("batch", 0) > batches + 1:
        set_aside(collapse)
    if moved:
        print(json.dumps({"rolled_back": {"to_batch": batches, "to_model_step": model_step, "moved": moved,
                                          "into": str(aside)}}), flush=True)
    return batches, model_step


def is_serving_failure(error: BaseException) -> bool:
    """True for a River client error that a restart from durable state can recover."""
    import river_client as river

    if isinstance(error, (river.CapacityError, river.SessionHeartbeatError)):
        return True
    if isinstance(error, river.RiverConnectionError):
        return error.status_code in TRANSIENT_CONNECTION_STATUSES
    return type(error) is river.RiverError and str(error).partition(":")[0] in SERVING_FAILURE_CATEGORIES


def run_with_recovery[T](
    run: Callable[[], T],
    run_dir: str | os.PathLike,
    run_id: str,
    classify: Callable[[BaseException], str | None],
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `run()` until it returns, restarting it after transient failures.

    `run` must resume from the newest saved trainer state (`epochs/*/trainer.json`), reconciling the run directory
    with it (`reconcile`), and close its API session before raising. `classify(error)` names the kind of a recoverable
    error ("serving", "timeout", ...) or returns None for an error that must propagate.
    """
    root = Path(run_dir)
    receipt = root / RECEIPT
    history = json.loads(receipt.read_text()) if receipt.exists() else {"run_id": run_id, "recoveries": []}
    if history["run_id"] != run_id:
        raise ValueError("recovery receipt belongs to another run")
    while True:
        try:
            return run()
        except Exception as error:
            kind = classify(error)
            if kind is None:
                raise
            if len(history["recoveries"]) >= RECOVERY_LIMIT:
                raise
            states = sorted((root / "epochs").glob("*/trainer.json"))
            if states:
                # Validate the saved state before reserving another attempt. The trainer performs full
                # fingerprint and provenance checks on restore.
                saved = json.loads(states[-1].read_text())
                if not saved.get("base") or "model_step" not in saved:
                    raise
                model_step = saved["model_step"]
            else:
                # Nothing trained yet (the failure came during the base evaluation, say): restart from the start.
                model_step = 0
            # Saved steps never move backwards, so earlier recoveries at this step are the attempts since progress.
            at_step = 1 + sum(event["model_step"] == model_step for event in history["recoveries"])
            if at_step > len(RECOVERY_DELAYS_SECONDS):
                raise
            event = {
                "attempt": len(history["recoveries"]) + 1,
                "attempt_at_step": at_step,
                "recorded_at": time.time(),
                "model_step": model_step,
                "trainer_state": str(states[-1]) if states else None,
                "kind": kind,
                "reason": type(error).__name__,
                "error": str(error),
                "delay_seconds": RECOVERY_DELAYS_SECONDS[at_step - 1],
            }
            history["recoveries"].append(event)
            atomic_json(receipt, history)
            print(json.dumps({"recovery": event}), flush=True)
            sleep(event["delay_seconds"])
