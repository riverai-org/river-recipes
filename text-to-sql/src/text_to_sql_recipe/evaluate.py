"""Evaluate checkpoints during training: one greedy sample per question, graded like training with no penalties.

Every `evaluation.eval_steps` training batches, and at the end, the trainer saves its weights and a sampling
snapshot and records a request in `pending-evaluations/`. A background queue (`EvaluationQueue`) evaluates the
requests one at a time, in order, on every suite (the validation split, and the Arcwise set unless
`evaluation.arcwise` is false) while training continues, and marks each completed. A restarted run replays any request still pending, and an
evaluation resumes from the per-question records it already wrote, after checking they are for the same checkpoint
and the same questions.

Evaluations are labelled by training batch. A batch whose groups all carry no learning signal makes no optimizer
step, so each summary records both counts: `batch` and the evaluated checkpoint's `model_step`.

Layout under the run directory:

    pending-evaluations/<batch>.json   the request: batch, sampling checkpoint, suites, completed
    checkpoint-<batch>.json            the training checkpoint saved with it
    eval/<suite>-step-<batch>/
      identity.json                    the sampling checkpoint evaluated and the digest of its question ids
      checkpoint.json                  the sampling checkpoint evaluated
      tasks/<sha256 of task id>.json   one record per question: answer, grade, metrics, conversation
      summary.json                     accuracy and the other suite metrics
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import river_client as river
from river_client import rl
from river_client.rl.engine import call

from .files import atomic_json


def tasks_sha256(rows: list[dict]) -> str:
    """A digest of a suite's question ids."""
    return hashlib.sha256(json.dumps(sorted(str(row["id"]) for row in rows)).encode()).hexdigest()


def summarize(records: list[dict], checkpoint: river.Checkpoint, step: int, suite: str) -> dict[str, Any]:
    """The suite summary from its per-question records."""
    count = len(records)
    metrics = {
        "eval/accuracy": sum(r["metrics"]["correct"] for r in records) / count,
        # Without the authors' 1% tolerance (equal to eval/accuracy for a strictly graded suite).
        "eval/accuracy_strict": sum(
            r["metrics"]["correct"] and not (r.get("grade") or {}).get("numeric_tolerance_applied") for r in records
        ) / count,
        "eval/format_compliance": sum(r["metrics"]["format_compliant"] for r in records) / count,
        "eval/reward_mean": sum(r["reward"] for r in records) / count,
        "eval/generated_tokens_mean": sum(r["generated_tokens"] for r in records) / count,
        "eval/tool_calls_mean": sum(r["metrics"].get("tool_calls", 0) for r in records) / count,
        "eval/cap_hit_rate": sum(bool(r.get("cap_hit")) for r in records) / count,
        "eval/loop_rate": sum(bool(r.get("loop")) for r in records) / count,
        "eval/drafts_mean": sum(float(r.get("drafts") or 0.0) for r in records) / count,
        "eval/redrafts_mean": sum(float(r.get("redrafts") or 0.0) for r in records) / count,
        "eval/questions": count,
        "eval/model_step": checkpoint.step,
        "eval/grading_error_count": sum(
            r["metrics"].get("grading/" + category, 0)
            for r in records
            for category in ("timeout", "execution_error", "result_limit")
        ),
        "eval/grading_recovered_timeouts": sum(r["metrics"].get("grading/recovered_timeout", 0) for r in records),
    }
    # A greedy score is only comparable across checkpoints when the checkpoint is not looping.
    metrics["eval/loop_affected"] = float(metrics["eval/loop_rate"] > 0.005)
    return {
        "step": step,  # the training batch the evaluation was requested at
        "batch": step,
        "model_step": checkpoint.step,
        "suite": suite,
        "checkpoint": asdict(checkpoint),
        "tasks_sha256": tasks_sha256([{"id": r["task_id"]} for r in records]),
        "metrics": metrics,
        # The grading policies actually observed in the verdicts.
        "observed_grading_policies": [
            {"numeric_policy": policy, "numeric_tolerance": tolerance}
            for policy, tolerance in sorted(
                {
                    (r.get("grade", {}).get("numeric_policy"), r.get("grade", {}).get("numeric_tolerance"))
                    for r in records
                },
                key=repr,
            )
        ],
    }


def record_of(group) -> dict[str, Any]:
    """The per-question evaluation record for a finished one-member group."""
    trajectory = group.trajectories[0]
    return {
        "task_id": str(group.row["id"]),
        "reward": trajectory.reward,
        "metrics": trajectory.metrics,
        "turns": trajectory.turns,
        "generated_tokens": trajectory.generated_tokens,
        "truncated": trajectory.truncated,
        "last_stop_reason": trajectory.last_stop_reason,
        # From the guards' turn scan: any turn that hit its token cap (or a budget truncation), not only the last.
        **{key: trajectory.metrics.get(key, 0) for key in ("cap_hit", "loop", "drafts", "redrafts", "repetition")},
        "repetition_kind": int(trajectory.metrics.get("repetition_kind", 0.0)),
        "messages": trajectory.messages,
        "grade": trajectory.sql_grade,
    }


async def evaluate_suite(
    session_factory: Callable[[], Any],
    runtime_factory: Callable[[], Any],
    engine_factory: Callable[[Any, Any], rl.RolloutEngine],
    *,
    base_model: str,
    tokenizer,
    rows: list[dict],
    run_dir: Path,
    step: int,
    suite: str,
    checkpoint: river.Checkpoint,
) -> dict[str, Any]:
    """Evaluate `checkpoint` on `rows`, resuming from records already written; returns the suite summary.

    `session_factory()` opens a River session context; `runtime_factory()` builds the grading runtime (no
    verifier, no penalties), closed when the suite is done; `engine_factory(sampler, runtime)` builds the evaluation
    engine (greedy decoding, no loop guards) around a sampler for the checkpoint.
    """
    directory = run_dir / "eval" / f"{suite}-step-{step:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    # Anything already here, complete or partial, must be for this checkpoint and these questions.
    identity = {"checkpoint": asdict(checkpoint), "tasks_sha256": tasks_sha256(rows)}
    identity_path = directory / "identity.json"
    checkpoint_path = directory / "checkpoint.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise RuntimeError(f"{directory} holds an evaluation of a different checkpoint or question set")
    elif checkpoint_path.exists() or (directory / "summary.json").exists():
        raise RuntimeError(f"{directory} holds an evaluation with no recorded identity; move it aside to redo it")
    else:
        atomic_json(identity_path, identity)
        atomic_json(checkpoint_path, asdict(checkpoint))
    summary_path = directory / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    task_dir = directory / "tasks"
    task_dir.mkdir(exist_ok=True)
    records = [json.loads(path.read_text()) for path in sorted(task_dir.glob("*.json"))]
    expected_ids = {str(row["id"]) for row in rows}
    if any(record["task_id"] not in expected_ids for record in records):
        raise RuntimeError(f"{directory} holds records for questions outside this suite")
    completed = {record["task_id"] for record in records}
    pending = [row for row in rows if str(row["id"]) not in completed]
    print(json.dumps({"evaluation_start": {"suite": suite, "step": step, "remaining": len(pending),
                                           "total": len(rows)}}), flush=True)
    if pending:
        runtime = runtime_factory()
        try:
            with session_factory() as session:
                sampler = rl.CheckpointSampler(
                    session, base_model=base_model, checkpoint=checkpoint, tokenizer=tokenizer
                )
                engine = engine_factory(sampler, runtime)
                async with engine:
                    for row in pending:
                        engine.submit(row, group_size=1, completion=rl.GroupCompletion("wait", min_members=1))
                    for _ in range(len(pending)):
                        group = await engine.next_group()
                        engine.accept(group)
                        record = record_of(group)
                        name = hashlib.sha256(record["task_id"].encode()).hexdigest()
                        atomic_json(task_dir / f"{name}.json", record)
                        records.append(record)
                        if len(records) % 25 == 0 or len(records) == len(rows):
                            progress = {"suite": suite, "step": step, "done": len(records), "total": len(rows)}
                            print(json.dumps({"evaluation_progress": progress}), flush=True)
                        engine.acknowledge(group.id)
        finally:
            await runtime.close()
    if len(records) != len(rows) or {r["task_id"] for r in records} != {str(row["id"]) for row in rows}:
        raise RuntimeError("evaluation coverage mismatch")
    summary = summarize(records, checkpoint, step, suite)
    atomic_json(summary_path, summary)
    print(json.dumps({"evaluation_complete": summary}), flush=True)
    return summary


async def drain_requests(run_dir: Path, execute: Callable[[dict], Awaitable[None]]) -> None:
    """Evaluate every pending request in step order, marking each completed."""
    for path in sorted((run_dir / "pending-evaluations").glob("*.json")):
        request = json.loads(path.read_text())
        if request.get("completed"):
            continue
        await execute(request)
        atomic_json(path, {**request, "completed": True})


async def record_request(model, run_dir: Path, step: int, suites: list[str]) -> None:
    """Save the weights at training batch `step` and record an evaluation request for them.

    An already-recorded request stays authoritative on recovery.
    """
    pending = run_dir / "pending-evaluations"
    pending.mkdir(exist_ok=True)
    request_path = pending / f"{step:04d}.json"
    if request_path.exists():
        return
    checkpoint = await call(model.save_weights, f"text-to-sql-step-{step:04d}", mode="training")
    if checkpoint.step != model.step:
        raise RuntimeError("saved training checkpoint disagrees with the model's policy step")
    atomic_json(run_dir / f"checkpoint-{step:04d}.json", asdict(checkpoint))
    inference = await call(model.save_weights, f"text-to-sql-eval-step-{step:04d}", mode="inference")
    if inference.step != checkpoint.step:
        raise RuntimeError("evaluation snapshot disagrees with the training checkpoint")
    atomic_json(request_path, {"step": step, "batch": step, "model_step": inference.step,
                               "checkpoint": asdict(inference), "suites": suites, "completed": False})


class EvaluationQueue:
    """Evaluate recorded requests in the background, one at a time and in batch order, while training continues.

    A failure surfaces on the training path at the next `check` or `submit`, and from `finish`.
    """

    def __init__(self, run_dir: Path, execute: Callable[[dict], Awaitable[None]]):
        self.run_dir, self.execute = run_dir, execute
        self._wake = asyncio.Event()
        self._finishing = False
        self._task: asyncio.Task | None = None

    def start(self) -> EvaluationQueue:
        self._task = asyncio.create_task(self._work())
        self._wake.set()  # requests recorded before this start come first
        return self

    def check(self) -> None:
        """Raise a background evaluation's failure."""
        if self._task is not None and self._task.done() and not self._task.cancelled():
            error = self._task.exception()
            if error is not None:
                raise error

    def submit(self) -> None:
        """A new request was recorded."""
        self.check()
        self._wake.set()

    async def finish(self) -> None:
        """Wait until every recorded request is evaluated."""
        self._finishing = True
        self._wake.set()
        if self._task is not None:
            await self._task

    async def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _work(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            await drain_requests(self.run_dir, self.execute)
            # A request recorded during that pass set the event again: drain once more before finishing.
            if self._finishing and not self._wake.is_set():
                return
