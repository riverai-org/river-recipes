"""End to end, offline: the environment, curriculum, grading, reward, trainer and evaluation on a scripted model.

The model is scripted (no service): on its first turn it runs one SQL query through the tool, on its second it
answers, correctly on even sampling seeds and incorrectly on odd ones, so every group has signal. Everything else is
the recipe's own code: a real SQLite database, the gold-query preflight, the curriculum engine and schedule, the
River trainer with the adaptive clip and the cosine schedule, the collapse pause, and checkpoint evaluation.
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import river_client as river
from river_client.renderers import ParsedResponse, SamplePrompt

from text_to_sql_recipe import __main__, config, evaluate, grading, prompts, recovery, sql_env, train
from text_to_sql_recipe.clipping import MedianGradClip

SCHEMA = "CREATE TABLE t(n INTEGER);"


class Tokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return list(text.encode())

    def decode(self, tokens):
        return bytes(tokens).decode(errors="replace")


class Renderer:
    tokenizer = Tokenizer()

    def get_stop_strings(self):
        return ["#"]

    def build_prompt_str(self, messages, *, tools=None):
        return "".join(f"{m['role']}:{m['content']}\n" for m in messages) + "assistant:"

    def build_sample_prompt(self, messages, *, tools=None):
        return SamplePrompt(self.build_prompt_str(messages, tools=tools), [], [])

    def build_continuation_prompt(self, messages, *, last_stop):
        prompt = self.build_sample_prompt(messages)
        return SamplePrompt(("" if last_stop else "#") + "\n" + prompt.prompt, prompt.images, prompt.image_formats)

    def parse_response(self, text, *, tools=None):
        return ParsedResponse({"role": "assistant", "content": text.rstrip("#")}, text.endswith("#"))


def prompt_text(kwargs, i):
    return bytes(t for chunk in kwargs["model_input"][i] for t in chunk["tokens"]).decode()


def answer_for(prompt: str, seed: int, *, greedy: bool = False) -> str:
    """The scripted policy: query once, then answer (correct on even seeds, and always when decoding greedily)."""
    if "Results for execute_sql_query" not in prompt:
        return '<tool_call>{"name": "execute_sql_query", "arguments": {"query": "SELECT n FROM t"}}</tool_call>'
    threshold = prompt.rsplit("greater than ", 1)[1].split("?", 1)[0]
    sql = f"SELECT n FROM t WHERE n > {threshold}" if greedy or seed % 2 == 0 else "SELECT n FROM t"
    return f"<solution>{sql}</solution>"


class Model:
    """A scripted River training model; records every call."""

    def __init__(self, grad_norms=None):
        self.step = 0
        self.calls = []
        self.grad_norms = list(grad_norms or [])

    async def sample(self, **kwargs):
        self.calls.append(("sample", self.step))
        # As on River, a trajectory's later turns name the policy its first turn was sampled from
        # (`pinned_policy_id`) and are sampled from it, even after an optimizer step has moved the model on.
        pinned = kwargs.get("pinned_policy_id")
        step = int(pinned.rsplit("-", 1)[1]) if pinned else self.step
        version = river.PolicyVersion(f"fake-policy-{step}", "fake-lineage", step)
        out = []
        for i, seed in enumerate(kwargs["seeds"]):
            text = answer_for(prompt_text(kwargs, i), seed, greedy=kwargs.get("top_k") == 1)
            out.append([river.Sample(text=text, tokens=list(text.encode()), logprobs=[-1.0] * len(text.encode()),
                                     stop_reason="stop", model_step=step, policy_version=version)])
        return out

    async def forward(self, data, **kwargs):
        return river.ForwardResult({}, [[-0.5] * (len(d["old_logprobs"]) - 1) for d in data])

    async def forward_backward(self, data, **kwargs):
        self.calls.append(("backward", len(data)))
        await asyncio.sleep(0)
        return river.ForwardResult({"loss": 0.5})

    async def optim_step(self, **kwargs):
        self.calls.append(("optim", copy.deepcopy(kwargs)))
        self.step += 1
        metrics = {"step": self.step}
        if self.grad_norms:
            metrics["grad_norm"] = self.grad_norms[(self.step - 1) % len(self.grad_norms)]
        return river.OptimStepResult(metrics)

    async def save_weights(self, name, *, mode, **kwargs):
        self.calls.append(("save", name, self.step, mode))
        return river.Checkpoint(f"river://fake/{name}", self.step, mode)

    async def load_weights(self, checkpoint, *, load_optimizer=True, **kwargs):
        self.calls.append(("load", checkpoint, load_optimizer))
        self.step = checkpoint.step


class Client:
    def session(self, **tags):
        return contextlib.nullcontext(object())


def frozen_sampler(model):
    """The evaluation sampler: the scripted policy pinned at the evaluated checkpoint's step, as River's is.

    Evaluation runs in the background while training moves on, so it must not sample the live training model.
    """
    def sampler(session, **kwargs):
        frozen = copy.copy(model)
        frozen.step = kwargs["checkpoint"].step
        return frozen
    return sampler


def make_task(qid, threshold, database):
    question = f"Which numbers are greater than {threshold}?"
    evidence = f"greater than {threshold} refers to n > {threshold}"
    return {
        "id": qid,
        "db_id": "toy",
        "domain": "toy",
        "question": question,
        "evidence": evidence,
        "problem": prompts.compose_problem(SCHEMA, evidence=evidence, question=question),
        "gold_sql": f"SELECT n FROM t WHERE n > {threshold}",
        "grading_method": "set",
        "database_path": str(database),
        "numeric_policy": "diamond-numeric-v1",
        "numeric_tolerance": False,
    }


def recipe(**overrides):
    raw = {
        "model": {"name": "test/model"},
        "training": {"num_train_epochs": 2, "batch_size": 2, "group_size": 2, "warmup_steps": 1},
        "rollout": {"max_turns": 2, "max_tokens_per_turn": 512, "max_context_length": 8192},
        "reward": {"verifier": "none"},
        "curriculum": {"hints": False, "evidence_keep_probability": 1.0},
        "evaluation": {"eval_steps": 2, "concurrency": 4},
        "execution": {"max_concurrent_rollouts": 8, "sql_concurrency": 2, "grading_concurrency": 2,
                      "tool_concurrency": 2},
    }
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(values)
    return config.from_dict(raw)


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        database = root / "toy.sqlite"
        with contextlib.closing(sqlite3.connect(database)) as connection:
            connection.execute(SCHEMA)
            connection.executemany("INSERT INTO t VALUES (?)", [(n,) for n in range(10)])
            connection.commit()
        self.train_rows = [make_task(f"q{i}", i, database) for i in range(4)]
        self.validation_rows = [make_task(f"v{i}", 5 + i, database) for i in range(2)]
        self.run_dir = root / "run"
        self.run_dir.mkdir()

    def run_training(self, settings, model):
        suites = {"validation": self.validation_rows}
        graded = self.train_rows + self.validation_rows
        expected = grading.gold_results(graded, concurrency=2)
        self.assertEqual(set(expected), {str(row["id"]) for row in graded})
        renderer = sql_env.RecipeRenderer(Renderer(), [prompts.sql_tool_schema({})["function"]])
        steps_per_epoch = 2
        run_config = {"recipe": settings.as_dict(), "datasets": {}}

        with patch.object(evaluate.rl, "CheckpointSampler", frozen_sampler(model)):
            asyncio.run(train._train(model, Client(), settings, self.run_dir, Tokenizer(), renderer, self.train_rows,
                                     suites, {}, expected, None, steps_per_epoch, run_config))
        return [json.loads(line) for line in (self.run_dir / "steps.jsonl").read_text().splitlines()]

    def test_two_epochs_train_grade_and_evaluate(self):
        model = Model()
        records = self.run_training(recipe(), model)
        self.assertEqual([r["batch"] for r in records], [1, 2, 3, 4])
        self.assertEqual([r["epoch"] for r in records], [1, 1, 2, 2])
        self.assertTrue((self.run_dir / "training-complete.json").exists())
        for epoch in ("001", "002"):
            state = json.loads((self.run_dir / "epochs" / epoch / "trainer.json").read_text())
            self.assertIn("dataset_scheduler", state)
        # Scripted answers are right on even seeds only, so the batch is half correct and every group has signal.
        for record in records:
            metrics = record["metrics"]
            self.assertAlmostEqual(metrics["sampled/accuracy"], 0.5)
            self.assertEqual(metrics["guard/loop_rate"], 0.0)
            self.assertIn("optimizer/learning_rate", metrics)
            self.assertEqual(metrics["optimizer/grad_clip_norm"], 20000.0)  # no norms reported, so no history
        # Evaluation at steps 0, 2 and 4 (the end), each request marked completed.
        for step in (0, 2, 4):
            summary = json.loads((self.run_dir / "eval" / f"validation-step-{step:04d}" / "summary.json").read_text())
            self.assertEqual(summary["metrics"]["eval/questions"], 2)
            # Evaluation decodes greedily, which the scripted model always answers correctly.
            self.assertEqual(summary["metrics"]["eval/accuracy"], 1.0)
            request = json.loads((self.run_dir / "pending-evaluations" / f"{step:04d}.json").read_text())
            self.assertTrue(request["completed"])
        # The learning rate warms up for one step, then decays on a cosine over the run's four steps.
        rates = [r["metrics"]["optimizer/learning_rate"] for r in records]
        self.assertEqual(rates, sorted(rates, reverse=True))

    def test_reward_and_grade_records_follow_the_episode(self):
        model = Model()
        self.run_training(recipe(training={"num_train_epochs": 1}), model)
        task_dir = self.run_dir / "eval" / "validation-step-0002" / "tasks"
        records = [json.loads(path.read_text()) for path in task_dir.glob("*.json")]
        self.assertEqual(len(records), 2)
        for record in records:
            # One tool turn, then the answer; graded correct.
            self.assertEqual(record["turns"], 2)
            self.assertEqual(record["metrics"]["tool_calls"], 1.0)
            self.assertEqual(record["grade"]["grade"], "CORRECT")
            # Evaluation has no penalties, so the reward is the outcome.
            self.assertEqual(record["reward"], 1.0)
            roles = [m["role"] for m in record["messages"]]
            self.assertEqual(roles[0], "system")

    def test_adaptive_clip_threshold_reaches_the_optimizer(self):
        norms = [1000.0, 1200.0, 900.0, 1100.0]
        model = Model(grad_norms=norms)
        records = self.run_training(recipe(), model)
        sent = [call[1]["grad_clip_norm"] for call in model.calls if call[0] == "optim"]
        logged = [r["metrics"]["optimizer/grad_clip_norm"] for r in records]
        self.assertEqual(sent, logged)
        # One warm-up step at the ceiling, then twice the median of the clipped norms seen so far.
        expected = MedianGradClip(ceiling=20000.0, multiple=2.0, window=20, warmup_steps=1)
        thresholds = []
        for step, norm in enumerate(norms, 1):
            threshold = expected.threshold()
            thresholds.append(threshold)
            expected.observe(step, norm, threshold)
        self.assertEqual(logged, thresholds)
        self.assertEqual(logged[0], 20000.0)
        self.assertEqual(logged[1], 2000.0)

    def test_restart_rebuilds_the_clip_history_from_the_optimizer_records(self):
        model = Model(grad_norms=[1000.0, 3000.0])
        settings = recipe(training={"num_train_epochs": 1})
        records = self.run_training(settings, model)
        log = recovery.OptimizerLog(self.run_dir / recovery.OPTIMIZER_LOG)
        rebuilt = train.gradient_clip(settings, log.load(through_step=2))
        self.assertEqual(rebuilt.history, {r["model_step"]: min(r["metrics"]["optimizer/grad_norm"],
                                                                r["metrics"]["optimizer/grad_clip_norm"])
                                           for r in records})

    def test_restart_after_a_failure_resumes_without_repeating_steps(self):
        class FlakyModel(Model):
            failed = False

            async def save_weights(self, name, *, mode, **kwargs):
                if name == "text-to-sql-step-0001" and not FlakyModel.failed:
                    FlakyModel.failed = True
                    raise river.RiverError("inference_worker_retry_exhausted: worker failed after 10 attempts")
                return await super().save_weights(name, mode=mode, **kwargs)

        settings = recipe(evaluation={"eval_steps": 1})
        with self.assertRaises(river.RiverError):
            self.run_training(settings, FlakyModel())
        first = [json.loads(line) for line in (self.run_dir / "steps.jsonl").read_text().splitlines()]
        self.assertEqual([r["batch"] for r in first], [1])
        # A new process: a fresh model handle, the same run directory.
        resumed = Model()
        records = self.run_training(settings, resumed)
        self.assertEqual([r["batch"] for r in records], [1, 2, 3, 4])
        self.assertEqual([r["model_step"] for r in records], [1, 2, 3, 4])
        self.assertTrue(any(call[0] == "load" for call in resumed.calls))
        for step in range(5):
            request = json.loads((self.run_dir / "pending-evaluations" / f"{step:04d}.json").read_text())
            self.assertTrue(request["completed"], step)
        self.assertTrue((self.run_dir / "training-complete.json").exists())

    def test_training_continues_cleanly_after_a_torn_step_record(self):
        class FlakyModel(Model):
            failed = False

            async def save_weights(self, name, *, mode, **kwargs):
                if name == "text-to-sql-step-0001" and not FlakyModel.failed:
                    FlakyModel.failed = True
                    raise river.RiverError("inference_worker_retry_exhausted: worker failed after 10 attempts")
                return await super().save_weights(name, mode=mode, **kwargs)

        settings = recipe(evaluation={"eval_steps": 1})
        with self.assertRaises(river.RiverError):
            self.run_training(settings, FlakyModel())
        log = self.run_dir / "steps.jsonl"
        with log.open("a") as stream:  # the process died while writing a record
            stream.write('{"batch": 2, "model_st')
        records = self.run_training(settings, Model())
        self.assertEqual([r["batch"] for r in records], [1, 2, 3, 4])
        # The fragment is set aside with the rest of what the restart reconciled.
        [quarantine] = (self.run_dir / recovery.ROLLED_BACK).glob("*/steps.jsonl.torn-*")
        self.assertEqual(quarantine.read_text(), '{"batch": 2, "model_st')

    def test_a_non_finite_step_ends_the_run_before_its_policy_is_used(self):
        class NonFiniteModel(Model):
            async def optim_step(self, **kwargs):
                result = await super().optim_step(**kwargs)
                if self.step == 2:  # the service applied the update and reports its norm as non-finite
                    result.metrics.update(grad_norm=float("nan"), grad_norm_finite=0.0)
                return result

        model = NonFiniteModel()
        with self.assertRaisesRegex(FloatingPointError, "optimizer step 2"):
            self.run_training(recipe(evaluation={"eval_steps": 1}), model)
        saved_steps = {call[2] for call in model.calls if call[0] == "save"}
        self.assertIn(1, saved_steps)
        self.assertNotIn(2, saved_steps)  # never published for sampling or evaluation, never a resume point
        records = [json.loads(line) for line in (self.run_dir / "steps.jsonl").read_text().splitlines()]
        self.assertEqual([r["model_step"] for r in records], [1])
        self.assertIsNone(train.recoverable(FloatingPointError("optimizer step 2 reported a non-finite norm")))

    def test_a_hard_kill_after_a_reported_step_does_not_repeat_it(self):
        # The process dies after logging batch 1 but before the trainer saved it: no finally blocks run.
        inputs = self.run_dir.parent / "inputs.json"
        inputs.write_text(json.dumps({"run_dir": str(self.run_dir), "train": self.train_rows,
                                      "validation": self.validation_rows}))
        child = (
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "import test_train as T\n"
            "details = json.loads(Path(sys.argv[1]).read_text())\n"
            "fixture = T.EndToEndTest()\n"
            "fixture.run_dir = Path(details['run_dir'])\n"
            "fixture.train_rows, fixture.validation_rows = details['train'], details['validation']\n"
            "class ExitAfterLog(T.Model):\n"
            "    async def save_weights(self, name, *, mode, **kwargs):\n"
            "        if name == 'text-to-sql-step-0001':\n"
            "            os._exit(77)\n"
            "        return await super().save_weights(name, mode=mode, **kwargs)\n"
            "fixture.run_training(T.recipe(training={'num_train_epochs': 1}, evaluation={'eval_steps': 1}),"
            " ExitAfterLog(grad_norms=[1000.0]))\n"
        )
        tests_dir = Path(__file__).resolve().parent
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tests_dir), str(tests_dir.parent / "src"),
                                                                   os.environ.get("PYTHONPATH", "")]))
        result = subprocess.run([sys.executable, "-c", child, str(inputs)], env=environment, capture_output=True,
                                text=True, timeout=300, check=False)
        self.assertEqual(result.returncode, 77, result.stderr[-2000:])
        reported = [json.loads(line)["batch"] for line in (self.run_dir / "steps.jsonl").read_text().splitlines()]
        self.assertEqual(reported, [1])
        saved = json.loads((self.run_dir / "epochs" / "001" / "trainer.json").read_text())
        self.assertEqual(saved["model_step"], 0)
        resumed = Model(grad_norms=[1000.0])
        settings = recipe(training={"num_train_epochs": 1}, evaluation={"eval_steps": 1})
        with contextlib.redirect_stdout(io.StringIO()):
            records = self.run_training(settings, resumed)
        self.assertEqual([(r["batch"], r["model_step"]) for r in records], [(1, 1), (2, 2)])
        # The redone step 1 is clipped like a first step; nothing from the lost attempt reaches the clip.
        sent = [call[1]["grad_clip_norm"] for call in resumed.calls if call[0] == "optim"]
        self.assertEqual(sent, [20000.0, 2000.0])
        [aside] = (self.run_dir / recovery.ROLLED_BACK).iterdir()
        self.assertIn('"batch": 1', (aside / "steps.jsonl").read_text())

    def test_a_saved_step_whose_log_record_was_lost_keeps_its_clip_history(self):
        class FailAfterFirstStep(Model):
            async def save_weights(self, name, *, mode, **kwargs):
                if name == "text-to-sql-step-0001":
                    raise river.RiverError("inference_worker_retry_exhausted: injected interruption")
                return await super().save_weights(name, mode=mode, **kwargs)

        settings = recipe(training={"num_train_epochs": 1}, evaluation={"eval_steps": 1})
        with self.assertRaises(river.RiverError), contextlib.redirect_stdout(io.StringIO()):
            self.run_training(settings, FailAfterFirstStep(grad_norms=[1000.0]))
        self.assertEqual(json.loads((self.run_dir / "epochs" / "001" / "trainer.json").read_text())["model_step"], 1)
        # The step is saved, but its diagnostic log record did not survive.
        (self.run_dir / "steps.jsonl").write_text('{"batch": 1, "model_st')
        resumed = Model(grad_norms=[1000.0])
        with contextlib.redirect_stdout(io.StringIO()):
            self.run_training(settings, resumed)
        sent = [call[1]["grad_clip_norm"] for call in resumed.calls if call[0] == "optim"]
        # Step 1 is replayed from the trainer's journal at the threshold it recorded; step 2 then clips at twice the
        # median of step 1's norm, rebuilt from the step's durable record although its log record was lost.
        self.assertEqual(sent, [20000.0, 2000.0])

    def test_a_cached_evaluation_must_match_the_requested_checkpoint_and_questions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.run_training(recipe(training={"num_train_epochs": 1}), Model())
        cached = json.loads((self.run_dir / "eval" / "validation-step-0002" / "summary.json").read_text())
        same = river.Checkpoint(**cached["checkpoint"])
        summary = asyncio.run(evaluate.evaluate_suite(None, None, None, base_model="test/model", tokenizer=None,
                                                      rows=self.validation_rows, run_dir=self.run_dir, step=2,
                                                      suite="validation", checkpoint=same))
        self.assertEqual(summary, cached)
        self.assertEqual((summary["batch"], summary["model_step"]), (2, 2))
        other = river.Checkpoint("river://different-policy", 99, "inference")
        for checkpoint, rows in ((other, self.validation_rows), (same, self.validation_rows[:1])):
            with self.subTest(checkpoint=checkpoint.path, rows=len(rows)), \
                    self.assertRaisesRegex(RuntimeError, "different checkpoint or question set"):
                asyncio.run(evaluate.evaluate_suite(None, None, None, base_model="test/model", tokenizer=None,
                                                    rows=rows, run_dir=self.run_dir, step=2, suite="validation",
                                                    checkpoint=checkpoint))

    def test_the_plan_pins_the_verifier_runtime(self):
        settings = recipe(training={"warmup_steps": 0}, reward={"verifier": "verieql", "verifier_gate": False},
                          evaluation={"arcwise": False})
        client = SimpleNamespace(get_server_capabilities=lambda: SimpleNamespace(require=lambda *args: None),
                                 session=lambda **kwargs: contextlib.nullcontext(
                                     SimpleNamespace(create_model=lambda **kwargs: Model())),
                                 close=lambda: None)

        async def skip_training(*args, **kwargs):
            return None

        connection = config.Connection(api_key="offline-test", base_url="https://example.invalid",
                                       data_dir=self.run_dir, output_dir=self.run_dir)
        runtimes = [SimpleNamespace(manifest_sha256="a" * 64), SimpleNamespace(manifest_sha256="b" * 64)]
        with patch.object(train, "preflight"), \
                patch.object(train.data, "load_splits", return_value=(self.train_rows, self.validation_rows)), \
                patch.object(train.verifier_module, "setup_verifier", side_effect=runtimes), \
                patch.object(train.river, "load_tokenizer", return_value=Tokenizer()), \
                patch.object(train, "get_renderer", return_value=Renderer()), \
                patch.object(train.river, "Client", return_value=client), \
                patch.object(train, "_train", skip_training), contextlib.redirect_stdout(io.StringIO()):
            train.run(settings, self.run_dir, connection)
            with self.assertRaisesRegex(RuntimeError, "different identities"):
                train.run(settings, self.run_dir, connection)
        plan = json.loads((self.run_dir / "plan.json").read_text())
        self.assertEqual(plan["identities"]["verifier"]["manifest_sha256"], "a" * 64)
        self.assertEqual(set(plan["identities"]), {"river_client", "tokenizer", "verifier"})

    def test_training_continues_while_a_checkpoint_is_evaluated(self):
        original = evaluate.evaluate_suite
        log = self.run_dir / "steps.jsonl"

        async def slow_evaluation(*args, **kwargs):
            if kwargs["step"] == 2:
                # Finish only once training has moved past batch 2; a blocking evaluation would never see that.
                async def trained_past():
                    while not any(json.loads(line)["batch"] >= 3 for line in log.read_text().splitlines()):
                        await asyncio.sleep(0.01)
                await asyncio.wait_for(trained_past(), timeout=30)
            return await original(*args, **kwargs)

        with patch.object(train.evaluate, "evaluate_suite", slow_evaluation):
            records = self.run_training(recipe(), Model())
        self.assertEqual([r["batch"] for r in records], [1, 2, 3, 4])
        for step in (0, 2, 4):
            self.assertTrue((self.run_dir / "eval" / f"validation-step-{step:04d}" / "summary.json").exists())

    def test_batches_without_a_learning_signal_are_labelled(self):
        class AllCorrect(Model):
            async def sample(self, **kwargs):
                out = []
                for i, seed in enumerate(kwargs["seeds"]):
                    text = answer_for(prompt_text(kwargs, i), seed, greedy=True)
                    out.append([river.Sample(text=text, tokens=list(text.encode()),
                                             logprobs=[-1.0] * len(text.encode()), stop_reason="stop",
                                             model_step=self.step)])
                return out

        settings = recipe(training={"num_train_epochs": 1}, reward={"process_penalty_weight": 0.0},
                          evaluation={"eval_steps": 1})
        model = AllCorrect()
        with contextlib.redirect_stdout(io.StringIO()):
            records = self.run_training(settings, model)
        self.assertEqual(model.step, 0)
        self.assertEqual([(r["batch"], r["model_step"], r["updated"]) for r in records], [(1, 0, False), (2, 0, False)])
        summary = json.loads((self.run_dir / "eval" / "validation-step-0002" / "summary.json").read_text())
        self.assertEqual((summary["batch"], summary["model_step"]), (2, 0))
        complete = json.loads((self.run_dir / "training-complete.json").read_text())
        self.assertEqual((complete["batches"], complete["model_step"]), (2, 0))

    def test_the_installed_client_has_every_feature_the_recipe_uses(self):
        self.assertEqual(train.missing_client_features(), [])
        from river_client.rl import trainer as client_trainer
        with patch.object(train.rl.RolloutEngine, "resubmit", None), \
                patch.object(client_trainer, "_require_finite_update", None):
            missing = train.missing_client_features()
        self.assertEqual(len(missing), 2)
        self.assertTrue(missing[0].startswith("RolloutEngine.resubmit"))
        self.assertIn("non-finite", missing[1])

    def test_max_steps_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "at least one training batch"):
            asyncio.run(train._train(Model(), Client(), recipe(), self.run_dir, Tokenizer(), None, self.train_rows,
                                     {}, {}, {}, None, 2, {}, total_steps=0))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            __main__.main(["train", "--recipe", "unused.toml", "--max-steps", "0"])

    def test_max_steps_stops_a_trial_run_early(self):
        model = Model()
        settings = recipe()
        suites = {"validation": self.validation_rows}
        expected = grading.gold_results(self.train_rows + self.validation_rows, concurrency=2)
        renderer = sql_env.RecipeRenderer(Renderer(), [prompts.sql_tool_schema({})["function"]])
        with patch.object(evaluate.rl, "CheckpointSampler", frozen_sampler(model)):
            asyncio.run(train._train(model, Client(), settings, self.run_dir, Tokenizer(), renderer, self.train_rows,
                                     suites, {}, expected, None, 2,
                                     {"recipe": settings.as_dict(), "datasets": {}}, total_steps=3))
        records = [json.loads(line) for line in (self.run_dir / "steps.jsonl").read_text().splitlines()]
        self.assertEqual([r["batch"] for r in records], [1, 2, 3])
        # The last step evaluates even when it is not a multiple of eval_steps.
        self.assertTrue((self.run_dir / "eval" / "validation-step-0003" / "summary.json").exists())

    def test_training_with_the_verifier_configured(self):
        checked = []

        class Verifier:
            async def check(self, task, prediction):
                checked.append((task["id"], prediction))
                refuted = task["id"] == "q0"
                return {"verdict": "refuted" if refuted else "confirmed", "queue_seconds": 0.0,
                        "service_seconds": 0.0}

            async def close(self):
                pass

        class Setup:
            def verifier(self, receipt_dir):
                return Verifier()

        settings = recipe(reward={"verifier": "verieql"}, training={"num_train_epochs": 1})
        model = Model()
        suites = {"validation": self.validation_rows}
        expected = grading.gold_results(self.train_rows + self.validation_rows, concurrency=2)
        renderer = sql_env.RecipeRenderer(Renderer(), [prompts.sql_tool_schema({})["function"]])
        with patch.object(evaluate.rl, "CheckpointSampler", frozen_sampler(model)):
            asyncio.run(train._train(model, Client(), settings, self.run_dir, Tokenizer(), renderer, self.train_rows,
                                     suites, {}, expected, Setup(), 2,
                                     {"recipe": settings.as_dict(), "datasets": {}}))
        records = [json.loads(line) for line in (self.run_dir / "steps.jsonl").read_text().splitlines()]
        self.assertEqual([r["batch"] for r in records], [1, 2])
        # Training consults the verifier for correct answers only; evaluation never does.
        self.assertTrue(checked)
        self.assertTrue(all(task_id.startswith("q") for task_id, _ in checked))
        refuted = sum(r["metrics"].get("rollout/verieql/refuted", 0.0) for r in records)
        self.assertGreater(refuted, 0)

    def test_a_run_directory_refuses_a_changed_plan(self):
        path = self.run_dir / "plan.json"
        plan = {"recipe": recipe().as_dict(), "implementation_sha256": train.implementation_sha256()}
        train.check_plan(path, plan)  # a new run directory
        path.write_text(json.dumps(plan))
        train.check_plan(path, plan)  # the same plan resumes
        changed = {**plan, "recipe": recipe(rollout={"max_tokens_per_turn": 256}).as_dict()}
        with self.assertRaisesRegex(RuntimeError, "different recipe"):
            train.check_plan(path, changed)

    def test_eval_steps_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "eval_steps"):
            recipe(evaluation={"eval_steps": 0})

    def test_recoverable_classification(self):
        self.assertEqual(train.recoverable(river.RiverTimeoutError("Sampling timed out")), "timeout")
        self.assertEqual(train.recoverable(river.RiverError("inference_worker_retry_exhausted: failed")), "serving")
        self.assertIsNone(train.recoverable(river.RiverError("invalid_input: bad")))
        self.assertIsNone(train.recoverable(ValueError("bad")))

    def test_evaluation_settings_drop_penalties_and_guards(self):
        settings = recipe(reward={"tool_format_penalty": 0.05, "verifier": "verieql"})
        episode = sql_env.EpisodeSettings.from_recipe(settings, evaluation=True)
        self.assertEqual((episode.reward.process_penalty_weight, episode.reward.redraft_penalty,
                          episode.reward.tool_format_penalty), (0.0, 0.0, 0.0))
        # Evaluation never consults the verifier, so its runtime is built without one.
        self.assertEqual(episode.reward.verifier, "none")
        sql_env.SqlRuntime(episode.reward, {}, tool_concurrency=1, grading_concurrency=1, sql_concurrency=1)
        with self.assertRaises(ValueError):
            sql_env.SqlRuntime(settings.reward, {}, tool_concurrency=1, grading_concurrency=1, sql_concurrency=1)
        self.assertFalse(episode.guards.loop_stop or episode.guards.repetition_guard)
        training = sql_env.EpisodeSettings.from_recipe(settings)
        self.assertEqual(training.reward, settings.reward)
        self.assertEqual(dataclasses.replace(training.guards), settings.guards)



class EvaluationGradingTest(unittest.TestCase):
    """Arcwise alone gets the authors' tolerance, and the summary keeps the strict score beside it."""

    def test_only_the_arcwise_suite_gets_the_tolerance(self):
        rows = [{"id": "arcwise:1", "numeric_tolerance": False}]
        self.assertIs(train.arcwise_suite(rows, "authors")[0]["numeric_tolerance"], True)
        self.assertIs(train.arcwise_suite(rows, "strict")[0]["numeric_tolerance"], False)
        self.assertIs(rows[0]["numeric_tolerance"], False)   # the loaded rows are not changed

    def test_summary_reports_accuracy_with_and_without_the_tolerance(self):
        def record(correct, applied):
            return {"task_id": f"arcwise:{len(records)}", "metrics": {"correct": float(correct), "format_compliant": 1.0},
                    "reward": float(correct),
                    "generated_tokens": 1, "grade": {"numeric_policy": "diamond-numeric-v1",
                                                     "numeric_tolerance": True, "numeric_tolerance_applied": applied}}
        records = []
        for correct, applied in ((1, False), (1, True), (0, False), (1, False)):
            records.append(record(correct, applied))
        summary = evaluate.summarize(records, river.Checkpoint(path="p", step=3, checkpoint_type="inference"), 3, "arcwise")
        self.assertEqual(summary["metrics"]["eval/accuracy"], 0.75)
        self.assertEqual(summary["metrics"]["eval/accuracy_strict"], 0.5)
        self.assertEqual(summary["observed_grading_policies"],
                         [{"numeric_policy": "diamond-numeric-v1", "numeric_tolerance": True}])


class PreflightTest(unittest.TestCase):
    """What a run cannot start without fails in seconds, as a message, and a stopped run leaves its error behind."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_train_without_data_says_to_run_setup_data(self):
        env = {"RIVER_API_KEY": "offline-test", "TEXT_TO_SQL_DATA_DIR": str(self.root / "empty"),
               "TEXT_TO_SQL_OUTPUT_DIR": str(self.root / "runs")}
        with patch.dict(os.environ, env), self.assertRaises(SystemExit) as stop:
            __main__.main(["train", "--recipe", "glm-5.3-flash", "--set", 'reward.verifier="none"'])
        self.assertIn("python -m text_to_sql_recipe setup data", str(stop.exception.code))
        self.assertFalse((self.root / "runs").exists())   # nothing was started

    def test_a_verieql_recipe_without_its_runtime_stops_before_reading_data(self):
        settings = recipe(reward={"verifier": "verieql"})
        connection = config.Connection(api_key="k", base_url="https://example.invalid", data_dir=self.root,
                                       output_dir=self.root)
        with patch.object(train.data, "require_present"), patch.dict(os.environ, {
                "TEXT_TO_SQL_VERIEQL_MANIFEST": "", "TEXT_TO_SQL_VERIEQL_MANIFEST_SHA256": ""}), \
                self.assertRaisesRegex(RuntimeError, "setup verieql"):
            train.preflight(settings, connection)

    def test_a_stopped_run_records_its_error_in_the_run_directory(self):
        connection = config.Connection(api_key="k", base_url="https://example.invalid", data_dir=self.root,
                                       output_dir=self.root / "runs")
        with patch.object(train, "run", side_effect=RuntimeError("Training paused by the collapse guard")), \
                self.assertRaises(RuntimeError):
            train.main(recipe(), "stopped", connection)
        failure = json.loads((self.root / "runs" / "stopped" / "last-failure.json").read_text())
        self.assertEqual(failure["error"], "RuntimeError: Training paused by the collapse guard")
        self.assertIn("Traceback", failure["traceback"])


if __name__ == "__main__":
    unittest.main()
