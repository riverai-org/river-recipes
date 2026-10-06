"""Train a model on River with the recipe, one epoch at a time, evaluating as it goes.

Before the model session opens, the run loads and checks its datasets, validates the verifier setup, and executes
every gold query once (`grading.gold_results`); a question whose gold query fails stops the run. Each epoch then
builds the curriculum schedule (`curriculum.EvidenceMixtureSchedule`), the grading runtime with the epoch's
verifier, and an asynchronous trainer: groups of `training.group_size` rollouts per question, `training.batch_size`
questions per optimizer step, group-centred advantages, the CISPO loss, Adam with linear warmup and cosine decay,
and the adaptive gradient clip (`clipping.MedianGradClip`). Before each step the batch passes the collapse pause and
the format-compliance floor. Every `evaluation.eval_steps` steps, and at the end, the weights are evaluated
(`evaluate`).

All state lives in the run directory, so a run restarted with the same recipe resumes where it stopped: the trainer
restores weights, committed steps, in-flight rollouts and the curriculum; pending evaluations are replayed; the
gradient clip's history is rebuilt from `optimizer-steps.jsonl`. Every start first reconciles the directory with the
trainer's saved progress (`recovery.reconcile`), so a step reported but never saved is trained again rather than
counted twice. `main` also restarts the run in-process after transient serving failures and timeouts (`recovery`).

The run plan pins what decides a run's results: the recipe, the datasets, this package's source, the River client's
source, the tokenizer and the VeriEQL runtime. A run directory refuses to resume under any other.

Run directory:

    plan.json                  the resolved recipe, the datasets' task digests and the implementation identities
    steps.jsonl                one record per training batch: progress and metrics, with its optimizer step
    optimizer-steps.jsonl      one record per optimizer step: gradient norm and clip threshold, written durably
    rolled-back/               records and evaluations set aside by `recovery.reconcile`
    progress.json              the trainer's progress, refreshed every 30 seconds
    epochs/<epoch>/trainer.json  the trainer's saved state (and its rollout and curriculum state)
    pending-evaluations/, checkpoint-<step>.json, eval/   see `evaluate`
    collapse-state.json, collapse-pause.json             see `guards.CollapsePause`
    recovery.json              see `recovery`
    last-failure.json          the error and traceback that last stopped the run, if one did
    training-complete.json     written when the last epoch finishes
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import river_client as river
from river_client import rl
from river_client.renderers import get_renderer
from river_client.rl.env import InfrastructureError

from . import curriculum, data, evaluate, grading, guards, prompts, recovery, sqlite_exec
from . import verifier as verifier_module
from .clipping import MedianGradClip
from .config import Connection, RecipeConfig
from .files import atomic_json
from .sql_env import EpisodeSettings, RecipeRenderer, SqlEnv, SqlRuntime

SAMPLE_TIMEOUT_SECONDS = 7200.0
ENVIRONMENT_TIMEOUT_SECONDS = 1800.0  # a reward call may wait on the verifier
ADMIT_AHEAD = 2  # batches of groups admitted ahead of training (at most the rollout staleness)
RPC_WORKERS = 16
WEIGHTS_EVERY = 5  # optimizer steps between full training checkpoints of the trainer's state
MIN_FORMAT_COMPLIANCE = 0.5
WATCHDOG_MESSAGE = "environment watchdog expired; rollout remains unscored"


def implementation_sha256() -> str:
    """A digest of this package's source, recorded with a run so its results name the code that produced them."""
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for source in sorted(package.glob("*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def client_identity() -> dict[str, str]:
    """The River client's version and a digest of its source, which decides sampling, rendering and training."""
    root = Path(river.__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted(root.rglob("*.py")):
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
    return {"version": river.__version__, "sha256": digest.hexdigest()}


def tokenizer_identity(name: str, revision: str | None, tokenizer) -> dict[str, Any]:
    """The tokenizer as loaded: its vocabulary, special tokens and chat template, which decide every rendered token."""
    digest = hashlib.sha256()
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocab = get_vocab() if callable(get_vocab) else {}
    digest.update(json.dumps(sorted(vocab.items()), ensure_ascii=False).encode())
    digest.update(json.dumps(sorted(getattr(tokenizer, "all_special_tokens", None) or [])).encode())
    digest.update(str(getattr(tokenizer, "chat_template", "") or "").encode())
    return {"name": name, "revision": revision, "sha256": digest.hexdigest()}


def verifier_identity(verifier_setup, gate: dict[str, Any]) -> dict[str, Any] | None:
    """The validated VeriEQL runtime and gate policy, which decide the verifier's verdicts, or None without one."""
    if verifier_setup is None:
        return None
    return {"manifest_sha256": verifier_setup.manifest_sha256, "gate_policy_sha256": gate.get("gate_policy_sha256")}


class TrainingAbort(RuntimeError):
    """A stop rule fired: snapshot and reassess rather than spend more."""


@dataclass(frozen=True)
class RunCosine:
    """The cosine schedule over the whole run, for a trainer that starts `offset` steps in (one trainer per epoch)."""

    offset: int
    total: int
    warmup: int
    floor: float

    def __call__(self, step, steps):
        return rl.cosine(warmup=self.warmup, floor=self.floor)(self.offset + step, self.total)


@dataclass(frozen=True)
class MedianClipAdam(rl.Adam):
    """Adam whose gradient-clip threshold follows the run's recent gradient norms.

    The fields are the clip's fixed settings, so they are part of the trainer's checkpoint fingerprint; the clip's
    history (`clipping.MedianGradClip`) is attached with `attach` and kept out of it.
    """

    clip_multiple: float | None = 2.0
    clip_window: int = 20
    clip_warmup_steps: int = 10

    def attach(self, clip: MedianGradClip) -> MedianClipAdam:
        object.__setattr__(self, "_clip", clip)
        return self

    def kwargs(self, step, steps):
        values = super().kwargs(step, steps)
        clip = getattr(self, "_clip", None)
        if clip is not None:
            values["grad_clip_norm"] = clip.threshold()
        return values


def check_plan(path: Path, plan: dict[str, Any]) -> None:
    """Refuse to continue a run directory that was started with a different recipe, dataset or implementation."""
    if not path.exists():
        return
    saved = json.loads(path.read_text())
    if saved != plan:
        changed = sorted(key for key in set(saved) | set(plan) if saved.get(key) != plan.get(key))
        raise RuntimeError(f"{path.parent} was started with a different {', '.join(changed)}; use a new run name")


def gradient_clip(recipe: RecipeConfig, entries: list[dict[str, Any]]) -> MedianGradClip:
    """The adaptive clip, its history rebuilt from the optimizer-step records of the saved progress."""
    training = recipe.training
    return MedianGradClip.replay(
        ((entry["model_step"], entry["grad_norm"], entry["grad_clip_norm"])
         for entry in entries if entry.get("grad_clip_norm") is not None),
        ceiling=training.max_grad_norm,
        multiple=training.grad_clip_median_multiple,
        window=training.grad_clip_window,
        warmup_steps=training.warmup_steps,
    )


def record_optimizer_steps(model, log: recovery.OptimizerLog, clip: MedianGradClip) -> None:
    """Record each optimizer step durably, and give its norm to the clip, before the trainer learns it completed.

    The trainer can save a step only after `optim_step` returns, so every saved step has its record; a step replayed
    from the trainer's journal on resume records again and replaces its entry.
    """
    original = model.optim_step

    def note(result, kwargs):
        metrics = dict(result.metrics or {})
        step = int(metrics.get("step", model.step))
        entry = log.record(step, metrics, kwargs.get("grad_clip_norm"))
        if entry["grad_clip_norm"] is not None:
            clip.observe(step, entry["grad_norm"], entry["grad_clip_norm"])

    if inspect.iscoroutinefunction(original):
        async def optim_step(*args, **kwargs):
            result = await original(*args, **kwargs)
            note(result, kwargs)
            return result
    else:
        def optim_step(*args, **kwargs):
            result = original(*args, **kwargs)
            note(result, kwargs)
            return result
    model.optim_step = optim_step


def make_engine(model, renderer, runtime: SqlRuntime, recipe: RecipeConfig, *, evaluation=False, hints=None):
    """The rollout engine: the curriculum's engine for training, a plain greedy one for evaluation."""
    rollout = recipe.rollout
    settings = EpisodeSettings.from_recipe(recipe, evaluation=evaluation)
    options = {
        "env": lambda: SqlEnv(runtime, renderer, settings),
        "renderer": renderer,
        "budget": rl.Budget(
            max_turns=rollout.max_turns + 1,
            max_turn_tokens=rollout.max_tokens_per_turn,
            segment_tokens=rollout.max_tokens_per_turn,
            max_generated_tokens=(rollout.max_turns + 1) * rollout.max_tokens_per_turn,
            max_context_tokens=rollout.max_context_length,
            tool_output_tokens=rollout.max_context_length,
        ),
        "schedule": rl.Schedule(
            concurrency=recipe.evaluation.concurrency if evaluation else recipe.execution.max_concurrent_rollouts,
            max_batch=recipe.execution.sample_batch_size,
            max_rpc_workers=RPC_WORKERS,
            admit_ahead=0 if evaluation else min(ADMIT_AHEAD, rollout.max_staleness),
        ),
        "sampling_policy": "trajectory",
        "kv_cache": rl.KVCache(),
        "seed": recipe.training.seed,
        "sample_timeout": SAMPLE_TIMEOUT_SECONDS,
        "environment_timeout": (
            ENVIRONMENT_TIMEOUT_SECONDS if recipe.reward.verifier != "none" and not evaluation else None
        ),
        "top_k": 1 if evaluation else -1,
        "return_expert_routing": recipe.model.routing_replay and not evaluation,
    }
    if evaluation:
        return rl.RolloutEngine(model, **options)
    return curriculum.AssistanceEngine(model, hints=hints, **options)


def policy_metrics(trajectories) -> dict[str, int]:
    mixed = 0
    retained = 0
    cross_policy = 0
    max_kv_age = 0
    for traj in trajectories:
        versions = set()
        for span in traj.spans:
            if span.kind != "generated" or span.policy_version is None:
                continue
            versions.add(span.policy_version.id)
            retained += int(span.retained_kv)
            origin = span.kv_cache_policy_version
            if origin is not None:
                age = span.policy_version.step - origin.step
                max_kv_age = max(max_kv_age, age)
                cross_policy += int(span.retained_kv and age > 0)
        mixed += int(len(versions) > 1)
    return {
        "sampling/mixed_policy_trajectories": mixed,
        "sampling/retained_kv_spans": retained,
        "sampling/cross_policy_kv_spans": cross_policy,
        "sampling/max_kv_age": max_kv_age,
    }


def make_trainer(
    model,
    renderer,
    runtime: SqlRuntime,
    recipe: RecipeConfig,
    *,
    schedule,
    hints,
    clip: MedianGradClip,
    step_offset: int,
    total_steps: int,
    epoch_dir: Path,
    run_dir: Path,
    run_config: dict[str, Any],
):
    training = recipe.training
    engine = make_engine(model, renderer, runtime, recipe, hints=hints)
    pause = guards.CollapsePause(run_dir, recipe.guards) if recipe.guards.collapse_pause else None
    optimizer = MedianClipAdam(
        lr=training.learning_rate,
        beta2=training.adam_beta2,
        grad_clip_norm=training.max_grad_norm,
        schedule=RunCosine(step_offset, total_steps, training.warmup_steps, training.min_lr_ratio),
        clip_multiple=training.grad_clip_median_multiple,
        clip_window=training.grad_clip_window,
        clip_warmup_steps=training.warmup_steps,
    ).attach(clip)

    def validate(trajectories, metrics):
        batch = trainer.progress()["batches"]
        if pause:
            pause.check(step_offset + batch + 1, trajectories, metrics)
        metrics["optimizer/learning_rate"] = trainer.optimizer.kwargs(batch, total_steps)["lr"]
        metrics.update(guards.guard_metrics(trajectories, training.group_size))
        compliance = sum(t.metrics.get("format_compliant", 0) for t in trajectories) / len(trajectories)
        if compliance < MIN_FORMAT_COMPLIANCE:
            raise TrainingAbort(f"format compliance {compliance:.3f} fell below {MIN_FORMAT_COMPLIANCE:.3f}")

    trainer = rl.AsyncTrainer(
        engine=engine,
        optimizer=optimizer,
        completion=rl.GroupCompletion("wait"),
        normalize="sum",
        groups_per_step=training.batch_size,
        group_size=training.group_size,
        advantage=rl.GroupCentered(),
        loss="cispo",
        eps_max=training.epsilon_high,
        max_staleness=recipe.rollout.max_staleness,
        stale_policy="wait",
        routing_replay=recipe.model.routing_replay,
        repeat_dataset=False,
        batch_order="completion",
        truncation=rl.Truncation(train="reward"),
        validate_batch=validate,
        dataset_scheduler=schedule,
        # The collapse pause must see the whole batch before any of its gradients reach the trainer.
        forward_backward_batch=None,
        checkpoint=rl.Checkpointing(epoch_dir, weights_every=WEIGHTS_EVERY, on_signal=("SIGINT", "SIGTERM")),
        run_config=run_config,
    )
    return trainer


async def consume(trainer, rows, steps, run_dir: Path, *, epoch: int, offset: int, optimizer_log: recovery.OptimizerLog,
                  after_step=None):
    """Run one epoch's trainer, recording every training batch in `steps.jsonl`.

    A batch whose groups all carry no learning signal makes no optimizer step; its record says so (`updated`), and
    its `model_step` is the previous batch's.
    """

    async def report():
        while True:
            await asyncio.sleep(30)
            progress = {"epoch": epoch, "progress": trainer.progress(), "updated_at": time.time()}
            atomic_json(run_dir / "progress.json", progress)
            print(json.dumps(progress), flush=True)

    async def recovered(batch):
        if after_step and batch:
            await after_step(offset + batch)

    reporter = asyncio.create_task(report())
    previous = time.monotonic()
    warned = False
    batches = trainer.run(rows, steps=steps, after_recovery=recovered)
    try:
        async for step in batches:
            now = time.monotonic()
            step.metrics.update(policy_metrics(step.trajectories))
            updated = bool(step.metrics.get("train/updated", 1.0))
            # The step's own durable record holds the threshold it was clipped at and its pre-clip norm.
            entry = optimizer_log.entries.get(step.model_step) if updated else None
            if entry is not None and entry.get("grad_clip_norm") is not None:
                threshold, norm = entry["grad_clip_norm"], entry["grad_norm"]
                step.metrics["optimizer/grad_clip_norm"] = threshold
                if norm is not None:
                    step.metrics["optimizer/clipped"] = float(norm > threshold)
                    step.metrics["optimizer/clip_scale"] = min(1.0, threshold / max(1e-9, norm))
                elif not warned:
                    warned = True
                    print(json.dumps({"warning": "the optimizer reported no grad_norm; the gradient clip stays at "
                                                 "its last threshold"}), flush=True)
            record = {
                "batch_seconds": now - previous,
                "progress": trainer.progress(),
                "batch": offset + step.n,
                "epoch": epoch,
                "model_step": step.model_step,
                "updated": updated,
                "metrics": step.metrics,
            }
            previous = now
            with (run_dir / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(json.dumps(record), flush=True)
            if after_step:
                await after_step(offset + step.n)
    finally:
        try:
            await batches.aclose()
        finally:
            reporter.cancel()
            await asyncio.gather(reporter, return_exceptions=True)


def arcwise_suite(rows: list[dict], grading: str) -> list[dict]:
    """The Arcwise evaluation rows, with the authors' 1% single-value tolerance unless `grading` is "strict"."""
    return [{**row, "numeric_tolerance": grading == "authors"} for row in rows]


def missing_client_features() -> list[str]:
    """River client features this recipe uses that the installed river-client lacks."""
    from river_client.rl import trainer as client_trainer

    missing = []
    if "loss_mask_end" not in vars(rl.Trajectory([{"type": "text", "tokens": [0]}])):
        missing.append("Trajectory.loss_mask_end (the loop guards)")
    if not callable(getattr(rl.RolloutEngine, "resubmit", None)):
        missing.append("RolloutEngine.resubmit (the curriculum's retests)")
    if "dataset_scheduler" not in inspect.signature(rl.AsyncTrainer).parameters:
        missing.append("AsyncTrainer(dataset_scheduler=...) (the curriculum schedule)")
    if not callable(getattr(client_trainer, "_require_finite_update", None)):
        missing.append("the trainer's stop on a non-finite optimizer step")
    return missing


def preflight(recipe: RecipeConfig, connection: Connection) -> None:
    """Check in seconds, before any database is read, what a run cannot start without.

    The River client's features, the named datasets' files in the data directory (their contents are checked when
    they load), and with VeriEQL the variables that locate its runtime. Raises RuntimeError or ValueError.
    """
    missing = missing_client_features()
    if missing:
        raise RuntimeError(f"river-client {river.__version__} lacks what this recipe uses: " + "; ".join(missing)
                           + ". Install the river-client release that provides them.")
    names = [recipe.data.train, recipe.data.validation, *([data.ARCWISE] if recipe.evaluation.arcwise else [])]
    data.require_present(names, connection.data_dir)
    verifier_module.require_runtime(recipe.reward)


def run(recipe: RecipeConfig, run_dir: Path, connection: Connection, *, max_steps: int | None = None) -> None:
    """One attempt at the whole run; resumes from the run directory's saved state.

    `max_steps` stops the run after that many training batches (a short trial); the schedule then spans them. A batch
    whose groups all carry no learning signal makes no optimizer step, so a run can make fewer steps than batches.
    """
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    preflight(recipe, connection)
    run_id = run_dir.name
    training, execution = recipe.training, recipe.execution
    data_dir = connection.data_dir
    train_rows, validation_rows = data.load_splits(recipe.data.train, recipe.data.validation, data_dir)
    suites = {"validation": validation_rows}
    if recipe.evaluation.arcwise:
        # Arcwise is graded as the ReViSQL authors grade it unless the recipe asks for strict grading; training and
        # validation are always strict.
        suites["arcwise"] = arcwise_suite(data.load_dataset(data.ARCWISE, data_dir), recipe.evaluation.arcwise_grading)
    graded = train_rows + [row for rows in suites.values() for row in rows]
    ids = [str(row["id"]) for row in graded]
    if len(set(ids)) != len(ids):
        raise ValueError("task ids repeat across the training and evaluation sets")
    steps_per_epoch = math.ceil(len(train_rows) / training.batch_size)
    total_steps = steps_per_epoch * training.num_train_epochs
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    elif total_steps <= training.warmup_steps:
        raise ValueError("the run has no more optimizer steps than its warmup")
    datasets = {
        "train": recipe.data.train,
        "validation": recipe.data.validation,
        **{f"{name}_tasks_sha256": data.task_manifest_sha256([str(row["id"]) for row in rows])
           for name, rows in (("train", train_rows), *suites.items())},
    }
    hints = {}
    if recipe.curriculum.hints:
        hints = curriculum.load_hints(*data.hint_files(recipe.data.train, data_dir), train_rows)
    gate = {}
    if recipe.reward.verifier != "none" and recipe.reward.verifier_gate:
        path, sha256 = data.gate_policy_file(recipe.data.train, data_dir)
        gate = {"gate_policy_path": path, "gate_policy_sha256": sha256}
    verifier_setup = verifier_module.setup_verifier(recipe.reward, execution, tasks=train_rows, **gate)
    tokenizer = river.load_tokenizer(recipe.model.name, revision=recipe.model.tokenizer_revision)
    run_config = {"recipe": recipe.as_dict(), "datasets": datasets}
    plan = {
        **run_config,
        "implementation_sha256": implementation_sha256(),
        "identities": {
            "river_client": client_identity(),
            "tokenizer": tokenizer_identity(recipe.model.name, recipe.model.tokenizer_revision, tokenizer),
            "verifier": verifier_identity(verifier_setup, gate),
        },
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "suites": {name: len(rows) for name, rows in suites.items()},
    }
    # A run directory belongs to one recipe, dataset, implementation, client, tokenizer and verifier: check before
    # touching anything in it, and before the expensive gold preflight.
    plan_path = run_dir / "plan.json"
    check_plan(plan_path, plan)
    for task in train_rows:
        curriculum.canonical_parts(task)  # every row must split into the parts a curriculum stage recomposes
    # Questions whose results are too large to grade in memory stream them through an index on local disk.
    grading.prepare_streaming_grading_storage(graded)
    print(json.dumps({"preflight": "gold queries", "questions": len(graded)}), flush=True)
    failures: dict[str, str] = {}
    expected = grading.gold_results(graded, concurrency=execution.grading_concurrency, failures=failures)
    missing = sorted(set(ids) - set(expected))
    if missing:
        examples = "; ".join(f"{task_id}: {failures.get(task_id, 'no result')}" for task_id in missing[:3])
        raise RuntimeError(f"gold queries failed for {len(missing)} question(s), for example {examples}")

    renderer = RecipeRenderer(
        get_renderer(
            recipe.model.name,
            tokenizer=tokenizer,
            thinking=True,
            strip_thinking_from_history=not recipe.model.keep_thinking_in_history,
            reasoning_effort=recipe.model.reasoning_effort,
        ),
        [prompts.sql_tool_schema({})["function"]],
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if not plan_path.exists():
        atomic_json(plan_path, plan)

    endpoint = urlparse(connection.base_url)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise ValueError("RIVER_BASE_URL must be an http(s) URL")
    client = river.Client(
        api_key=connection.api_key,
        endpoint=endpoint.hostname,
        port=endpoint.port or (443 if endpoint.scheme == "https" else 80),
        use_ssl=endpoint.scheme == "https",
    )
    try:
        client.get_server_capabilities().require(
            "sampling_batch_v1", "policy_versions_v1", "sampling_policy_pinning_v1"
        )
        with client.session(experiment="text-to-sql-rl", run=run_id) as session:
            model = session.create_model(
                base_model=recipe.model.name,
                tokenizer=tokenizer,
                lora=river.LoraConfig(
                    rank=recipe.model.lora_rank, train_attn=True, train_mlp=True, train_unembed=False, seed=0
                ),
            )
            asyncio.run(_train(model, client, recipe, run_dir, tokenizer, renderer, train_rows, suites, hints,
                               expected, verifier_setup, steps_per_epoch, run_config, total_steps=total_steps))
    finally:
        client.close()


async def _train(model, client, recipe, run_dir, tokenizer, renderer, train_rows, suites, hints, expected,
                 verifier_setup, steps_per_epoch, run_config, *, total_steps=None):
    training, execution = recipe.training, recipe.execution
    if total_steps is None:
        total_steps = steps_per_epoch * training.num_train_epochs
    if total_steps < 1:
        raise ValueError("a run needs at least one training batch")
    run_dir.mkdir(parents=True, exist_ok=True)
    # Anything reported past the trainer's saved progress describes weights the run no longer has.
    _, saved_step = recovery.reconcile(run_dir, steps_per_epoch)
    optimizer_log = recovery.OptimizerLog(run_dir / recovery.OPTIMIZER_LOG)
    clip = gradient_clip(recipe, optimizer_log.load(saved_step))
    record_optimizer_steps(model, optimizer_log, clip)
    sql_work = sqlite_exec.SqlWork(execution.sql_concurrency)
    epoch_root = run_dir / "epochs"
    evaluation_settings = EpisodeSettings.from_recipe(recipe, evaluation=True)

    def runtime(verifier=None, *, evaluation=False):
        return SqlRuntime(
            evaluation_settings.reward if evaluation else recipe.reward,
            expected,
            tool_concurrency=execution.tool_concurrency,
            grading_concurrency=execution.grading_concurrency,
            sql_concurrency=execution.sql_concurrency,
            verifier=verifier,
            sql_work=sql_work,
        )

    async def execute(request):
        checkpoint = river.Checkpoint(**request["checkpoint"])
        for suite in request["suites"]:
            await evaluate.evaluate_suite(
                lambda suite=suite: client.session(experiment="text-to-sql-eval", run=run_dir.name, suite=suite),
                lambda: runtime(evaluation=True),
                lambda sampler, eval_runtime: make_engine(sampler, renderer, eval_runtime, recipe, evaluation=True),
                base_model=recipe.model.name,
                tokenizer=tokenizer,
                rows=suites[suite],
                run_dir=run_dir,
                step=request["step"],
                suite=suite,
                checkpoint=checkpoint,
            )

    # Evaluations run in the background, one at a time, while training continues.
    evaluations = evaluate.EvaluationQueue(run_dir, execute)

    async def checkpoint_evaluation(batch):
        evaluations.check()
        if batch % recipe.evaluation.eval_steps and batch != total_steps:
            return
        await evaluate.record_request(model, run_dir, batch, list(suites))
        evaluations.submit()

    evaluations.start()  # requests recorded before a restart are evaluated first
    try:
        if not any(epoch_root.glob("*/trainer.json")):
            await evaluate.record_request(model, run_dir, 0, list(suites))  # the untrained model
            evaluations.submit()
        offset = 0
        for epoch in range(1, training.num_train_epochs + 1):
            steps = min(steps_per_epoch, total_steps - offset)
            if steps <= 0:
                break
            epoch_dir = epoch_root / f"{epoch:03d}"
            previous = None
            if epoch > 1:
                previous = json.loads((epoch_root / f"{epoch - 1:03d}" / "trainer.json").read_text())["dataset_scheduler"]
            schedule = curriculum.EvidenceMixtureSchedule(
                train_rows,
                epoch=epoch,
                batch_size=training.batch_size,
                seed=training.seed,
                hints=hints,
                keep_probability=recipe.curriculum.evidence_keep_probability,
                weighting=recipe.curriculum.evidence_weighting,
                failed_weight=recipe.curriculum.evidence_failed_weight,
                solved_weight=recipe.curriculum.evidence_solved_weight,
                fade_probability=recipe.curriculum.evidence_fade_probability,
                include_empty_evidence=recipe.curriculum.evidence_draws_include_empty,
                previous=previous,
                reservations_dir=epoch_dir / "evidence-reservations",
            )
            epoch_verifier = None
            if verifier_setup is not None:
                epoch_verifier = verifier_setup.verifier(run_dir / "verieql" / f"epoch-{epoch:03d}")
            epoch_runtime = runtime(epoch_verifier)
            trainer = make_trainer(
                model,
                renderer,
                epoch_runtime,
                recipe,
                schedule=schedule,
                hints=hints,
                clip=clip,
                step_offset=offset,
                total_steps=total_steps,
                epoch_dir=epoch_dir,
                run_dir=run_dir,
                run_config=run_config,
            )
            try:
                await consume(trainer, train_rows, steps, run_dir, epoch=epoch, offset=offset,
                              optimizer_log=optimizer_log, after_step=checkpoint_evaluation)
                if trainer.progress()["status"] != "completed":
                    raise RuntimeError("training stopped before the epoch completed")
            finally:
                await epoch_runtime.close()
            offset += steps
        await evaluations.finish()
    except BaseException:
        await evaluations.cancel()  # an unfinished evaluation resumes from its records on the next start
        raise
    atomic_json(run_dir / "training-complete.json",
                {"completed_at": time.time(), "batches": total_steps, "total_steps": total_steps,
                 "model_step": model.step})


def recoverable(error: BaseException) -> str | None:
    """The kind of failure a restart from the saved state can recover, or None."""
    if (
        isinstance(error, river.RiverTimeoutError)
        or grading.numeric_comparison_timed_out(error)
        or (type(error) is InfrastructureError and str(error) == WATCHDOG_MESSAGE)
    ):
        return "timeout"
    if recovery.is_serving_failure(error):
        return "serving"
    return None


def default_run_name(recipe: RecipeConfig) -> str:
    model = recipe.model.name.rsplit("/", 1)[-1].lower()
    return f"{model}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"


def main(
    recipe: RecipeConfig,
    run_name: str | None = None,
    connection: Connection | None = None,
    *,
    max_steps: int | None = None,
) -> Path:
    """Train, restarting from the saved state after transient failures; returns the run directory."""
    from .config import connection as load_connection

    connection = connection or load_connection()
    run_dir = connection.output_dir / (run_name or default_run_name(recipe))
    try:
        recovery.run_with_recovery(
            lambda: run(recipe, run_dir, connection, max_steps=max_steps), run_dir, run_dir.name, recoverable
        )
    except Exception as error:
        record_failure(run_dir, error)
        raise
    return run_dir


def record_failure(run_dir: Path, error: BaseException) -> None:
    """Keep the error that stopped a run in its run directory (`last-failure.json`), which outlives the process."""
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(run_dir / "last-failure.json", {
            "failed_at": time.time(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": "".join(traceback.format_exception(error)),
        })
    except OSError:
        pass   # the error itself still propagates
