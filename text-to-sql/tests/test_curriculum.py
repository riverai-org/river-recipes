"""Curriculum: hint loading, staged rows, rescue in the rollout engine, and the evidence-mixture scheduler.

Most tests check one behaviour each: scheduler state and commits directly, and rescue, restart and batching through
the River trainer with a scripted model (no service needed). `PracticeOptionTests` covers the scheduler's practice
options (lane weighting, the weights of a full fail and a full pass, the evidence fade, and evidence draws that may
pick a question with empty evidence), which are off by default.

`ScheduleTraceTest` replays scripted epochs through the scheduler (fixed seed, scripted group outcomes, admission
ahead of commits, out-of-order completion, resumes from saved state and from reservation files) and checks every
reservation, rescue stage, commit and saved state against the golden trace in `fixtures/curriculum/`, with the
practice options at their defaults whether passed or not; it also runs the scenarios with the options on.
"""
from __future__ import annotations

import asyncio
import copy
import functools
import hashlib
import json
import random
import re
import tempfile
import unittest
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import river_client as river
from river_client import rl
from river_client.renderers import ParsedResponse, SamplePrompt
from river_client.rl.engine import CompletedGroup

from text_to_sql_recipe import curriculum
from text_to_sql_recipe.prompts import TRAINING_HINT_PREFIX, compose_problem

# --- Scripted River model, renderer and trainer -----------------------------------------------------------------


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


def sample(text="A", *, step=0, reason="stop"):
    return river.Sample(
        text=text, tokens=list(text.encode()), logprobs=[-1.0] * len(text), stop_reason=reason, model_step=step
    )


def trajectory(reward=1.0, *, step=0, text="A"):
    traj = rl.Trajectory([{"type": "text", "tokens": [10, 11]}])
    traj.append_span(sample(text, step=step))
    traj.done, traj.reward, traj.truncated = True, reward, None
    return traj


class Model:
    def __init__(self, sampler=None):
        self.step = 0
        self.calls = []
        self.sampler = sampler

    async def sample(self, **kwargs):
        self.calls.append(("sample", copy.deepcopy(kwargs), self.step))
        if self.sampler:
            return await self.sampler(self, kwargs)
        return [[sample("A" if seed % 2 else "B", step=self.step)] for seed in kwargs["seeds"]]

    async def forward(self, data, **kwargs):
        self.calls.append(("forward", copy.deepcopy(data), kwargs))
        return river.ForwardResult({}, [[-0.5] * (len(d["old_logprobs"]) - 1) for d in data])

    async def forward_backward(self, data, **kwargs):
        self.calls.append(("backward", copy.deepcopy(data), kwargs))
        await asyncio.sleep(0)
        return river.ForwardResult({"loss": 0.5})

    async def optim_step(self, **kwargs):
        self.calls.append(("optim", kwargs))
        self.step += 1
        return river.OptimStepResult({"step": self.step})

    async def save_weights(self, name, *, mode):
        self.calls.append(("save", name, self.step, mode))
        return river.Checkpoint(f"river://fake/{name}", self.step, mode)

    async def load_weights(self, checkpoint, *, load_optimizer):
        self.calls.append(("load", checkpoint, load_optimizer))
        self.step = checkpoint.step


def trainer(model, **kwargs):
    defaults = {
        "optimizer": rl.Adam(lr=1e-4),
        "completion": rl.GroupCompletion("wait", min_members=2),
        "normalize": "token",
        "groups_per_step": 1,
        "group_size": 2,
        "max_staleness": 0,
    }
    defaults.update(kwargs)
    return rl.AsyncTrainer(**defaults)


class Trial(rl.Env):
    recovery = "stateless"

    async def reset(self, row):
        return [{"role": "user", "content": row["problem"]}]

    async def reward(self, traj, row):
        traj.metrics["correct"] = float(traj.final_text == "A")
        # Raw correctness, not shaped reward sign, must gate escalation.
        return -0.1 if traj.final_text == "A" else -1.0


def assistance_engine(model, hints=None, *, env=Trial, schedule=None, engine_class=curriculum.AssistanceEngine):
    return engine_class(
        model,
        env=env,
        renderer=Renderer(),
        hints=hints,
        budget=rl.Budget(segment_tokens=16, max_context_tokens=2048),
        schedule=schedule or rl.Schedule(window=0.001, concurrency=16, max_batch=16),
    )


def prompt_text(kwargs, i):
    return bytes(t for chunk in kwargs["model_input"][i] for t in chunk["tokens"]).decode()


# --- Rows, groups and outcomes ----------------------------------------------------------------------------------


def task(qid="q", evidence="stored representation"):
    return {
        "id": qid,
        "question": "Which item?",
        "evidence": evidence,
        "problem": compose_problem("CREATE TABLE t(n INTEGER);", evidence=evidence, question="Which item?"),
        "gold_sql": "SELECT n FROM t",
        "database_sha256": "db",
    }


def group(row, correct, total=16, index=0):
    trajectories = [trajectory() for i in range(total)]
    for i, traj in enumerate(trajectories):
        traj.metrics["correct"] = int(i < correct)
    return CompletedGroup(f"group-{index}", row, trajectories)


def outcome(schedule, rows, counts, *, index=0):
    """Reserve row `index`, run the stages `counts` ((level, correct of 16), ...), and commit the last one."""
    row = schedule.row(index, rows)
    history = [
        {"level": level, "correct": correct, "total": 16, "tokens": 42, "policy_steps": [0]}
        for level, correct in counts[:-1]
    ]
    row = curriculum.stage_row(row, counts[-1][0], hint=schedule.hints.get(row["id"]), history=history)
    return schedule.commit(index + 1, [group(row, counts[-1][1], index=index)])


def schedule(rows, **kwargs):
    return curriculum.EvidenceMixtureSchedule(rows, epoch=1, batch_size=2, **kwargs)


def unaided_reservation(qid, index, epoch=1):
    """A reservation in the "none" lane, as the scheduler records one."""
    return {
        "qid": qid,
        "level": 0,
        "lane": "none",
        "requested_lane": "none",
        "evidence": "",
        "evidence_sha256": curriculum.digest(""),
        "visit": f"{epoch}:{index}",
        "policy": curriculum.VERSION,
    }


def record(last_p=None, p=None):
    """One level's record as the scheduler keeps it: untested, or last scored `last_p` with pass rate `p`."""
    if last_p is None:
        return {"p": None, "last_p": None, "visits": 0, "failures": 0}
    return {"p": last_p if p is None else p, "last_p": last_p, "visits": 1, "failures": int(last_p == 0)}


def practice_schedule(rows, records, **kwargs):
    """A schedule past its coverage pass: each question visited once, with records[id] = (unaided, evidence)."""
    s = curriculum.EvidenceMixtureSchedule(rows, epoch=2, batch_size=4, **kwargs)
    for qid, (unaided, evidence) in records.items():
        s.questions[qid]["visits"] = 1
        s.questions[qid]["levels"][:2] = [copy.deepcopy(unaided), copy.deepcopy(evidence)]
    return s


def independent_draws(s, rows, n):
    """Reserve groups 0..n-1, releasing each reservation so that every draw sees the same state."""
    choices = []
    for index in range(n):
        s.row(index, rows)
        choices.append(s.reservations.pop(str(index)))
    return choices


# --- Hints and staged rows --------------------------------------------------------------------------------------


class HintTests(unittest.TestCase):
    def write(self, root, references, bindings):
        refs, binds = Path(root) / "refs.json", Path(root) / "bindings.json"
        refs.write_text(json.dumps(references))
        binds.write_text(json.dumps(bindings))
        return (
            refs,
            hashlib.sha256(refs.read_bytes()).hexdigest(),
            binds,
            hashlib.sha256(binds.read_bytes()).hexdigest(),
        )

    def binding(self, row):
        return {
            "current_role": "paper_train",
            "prompt_sha256": hashlib.sha256(row["problem"].encode()).hexdigest(),
            "gold_sql_sha256": hashlib.sha256(row["gold_sql"].encode()).hexdigest(),
            "database_sha256": row["database_sha256"],
        }

    def test_fail_closed_on_unknown_prompt_codec_and_hint_identity(self):
        row = task()
        with self.assertRaisesRegex(ValueError, "codec"):
            curriculum.canonical_parts({**row, "problem": "unknown format"})
        with tempfile.TemporaryDirectory() as root:
            args = self.write(root, {"references": {"q": {"raw": "hint"}}}, {"q": self.binding(row)})
            self.assertEqual(curriculum.load_hints(*args, [row]), {"q": "hint"})
            with self.assertRaisesRegex(ValueError, "stale"):
                curriculum.load_hints(*args, [{**row, "database_sha256": "changed"}])

    def test_a_row_without_a_hint_or_binding_names_the_file_that_lacks_it(self):
        row = task()
        with tempfile.TemporaryDirectory() as root:
            refs, refs_sha, binds, binds_sha = self.write(root, {"references": {}}, {"q": self.binding(row)})
            with self.assertRaisesRegex(ValueError, f"q: no training hint in {re.escape(str(refs))}"):
                curriculum.load_hints(refs, refs_sha, binds, binds_sha, [row])
            refs, refs_sha, binds, binds_sha = self.write(root, {"references": {"q": {"raw": "hint"}}}, {})
            with self.assertRaisesRegex(ValueError, f"q: no training hint binding in {re.escape(str(binds))}"):
                curriculum.load_hints(refs, refs_sha, binds, binds_sha, [row])
            with self.assertRaisesRegex(ValueError, f"hash mismatch: {re.escape(str(refs))}"):
                curriculum.load_hints(refs, "0" * 64, binds, binds_sha, [row])

    def test_hint_files_are_pinned_and_hints_are_training_only_and_nonempty(self):
        row = task()
        with tempfile.TemporaryDirectory() as root:
            refs, refs_sha, binds, binds_sha = self.write(
                root, {"references": {"q": {"raw": "hint"}}}, {"q": self.binding(row)}
            )
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                curriculum.load_hints(refs, "0" * 64, binds, binds_sha, [row])
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                curriculum.load_hints(refs, refs_sha, binds, "0" * 64, [row])
            with self.assertRaisesRegex(ValueError, "training-only"):
                curriculum.load_hints(refs, refs_sha, binds, binds_sha, [{**row, "current_role": "paper_validation"}])
            args = self.write(root, {"references": {"q": {"raw": "hint"}}}, {"q": {**self.binding(row),
                                                                                 "current_role": "validation"}})
            with self.assertRaisesRegex(ValueError, "training-only"):
                curriculum.load_hints(*args, [row])
            args = self.write(root, {"references": {"q": {"raw": "  "}}}, {"q": self.binding(row)})
            with self.assertRaisesRegex(ValueError, "empty"):
                curriculum.load_hints(*args, [row])


class StageRowTests(unittest.TestCase):
    def test_levels_withhold_show_or_show_and_hint(self):
        row = task()
        schema = curriculum.canonical_parts(row)
        self.assertEqual(schema, "CREATE TABLE t(n INTEGER);")
        none = curriculum.stage_row(row, 0)
        self.assertEqual(none["evidence"], "")
        self.assertEqual(none["problem"], compose_problem(schema, evidence="", question=row["question"]))
        self.assertIn("External Knowledge:\n(none)\n", none["problem"])
        shown = curriculum.stage_row(row, 1)
        self.assertEqual(shown["evidence"], row["evidence"])
        self.assertEqual(shown["problem"], row["problem"])
        hinted = curriculum.stage_row(row, 2, hint="use table t")
        self.assertEqual(hinted["problem"], row["problem"] + TRAINING_HINT_PREFIX + "use table t")
        self.assertEqual(TRAINING_HINT_PREFIX, "\n\nTraining hint:\n")
        self.assertEqual(
            hinted[curriculum.KEY],
            {"level": 2, "original": {"schema": schema, "evidence": row["evidence"]}, "history": []},
        )
        self.assertEqual(row, task())  # the input row is not modified
        for level, hint in ((2, None), (2, ""), (3, "hint"), (-1, None)):
            with self.subTest(level=level, hint=hint), self.assertRaisesRegex(ValueError, "stage"):
                curriculum.stage_row(row, level, hint=hint)

    def test_restaging_starts_from_the_original_and_keeps_the_reservation(self):
        row = curriculum.stage_row(task(), 0)
        row[curriculum.KEY]["mixture"] = {"qid": "q", "lane": "none"}
        history = [{"level": 0, "correct": 0, "total": 2, "tokens": 4, "policy_steps": [0]}]
        rescued = curriculum.stage_row(row, 1, history=history)
        self.assertEqual(rescued["evidence"], "stored representation")
        self.assertEqual(rescued["problem"], task()["problem"])
        self.assertEqual(rescued[curriculum.KEY]["history"], history)
        self.assertEqual(rescued[curriculum.KEY]["mixture"], {"qid": "q", "lane": "none"})
        history.append("changed")
        self.assertEqual(len(rescued[curriculum.KEY]["history"]), 1)

    def test_task_text_must_be_the_composition_of_its_fields(self):
        row = task()
        padded = {**row, "problem": row["problem"].replace("CREATE TABLE", " CREATE TABLE", 1)}
        with self.assertRaisesRegex(ValueError, "disagree"):
            curriculum.canonical_parts(padded)
        with self.assertRaisesRegex(ValueError, "codec"):
            curriculum.canonical_parts({**row, "evidence": "other evidence"})


# --- Scheduler --------------------------------------------------------------------------------------------------


class ScheduleTests(unittest.TestCase):
    def test_fifty_percent_schedule_preserves_coverage_and_restart_choices(self):
        rows = [task(str(i)) for i in range(1000)]
        original = schedule(rows)
        chosen = [original.row(i, rows) for i in range(500)]
        restored = schedule(rows)
        restored.load_state_dict(json.loads(json.dumps(original.state_dict())))
        self.assertEqual(restored.row(42, rows), chosen[42])
        chosen += [restored.row(i, rows) for i in range(500, 1000)]
        self.assertEqual(len({r["id"] for r in chosen}), 1000)
        kept = sum(bool(r["evidence"]) for r in chosen)
        self.assertTrue(450 < kept < 550, kept)
        self.assertTrue(all(r["gold_sql"] == rows[0]["gold_sql"] for r in chosen))

    def test_trusted_quota_survives_first_pass_coverage(self):
        rows = [task(str(i), evidence="" if i >= 1961 else "trusted") for i in range(2079)]
        for keep, trusted, shortfall in ((0.5, 1040, 0), (1.0, 1961, 118)):
            with self.subTest(keep_probability=keep):
                s = curriculum.EvidenceMixtureSchedule(rows, epoch=1, batch_size=64, keep_probability=keep)
                chosen = [s.row(i, rows) for i in range(2079)]
                choices = [r[curriculum.KEY]["mixture"] for r in chosen]
                lanes = [c["lane"] for c in choices]
                self.assertEqual(lanes.count("trusted"), trusted)
                self.assertEqual(lanes.count("none"), 2079 - trusted)
                self.assertEqual(sum(c["lane"] != c["requested_lane"] for c in choices), shortfall)
                self.assertEqual(len({r["id"] for r in chosen}), 2079)

    def test_first_pass_reservations_are_unique_and_durable(self):
        rows = [task(str(i)) for i in range(12)]
        first = curriculum.EvidenceMixtureSchedule(rows, epoch=1, batch_size=4)
        chosen = [first.row(i, rows) for i in range(8)]
        restored = curriculum.EvidenceMixtureSchedule(rows, epoch=1, batch_size=4)
        restored.load_state_dict(json.loads(json.dumps(first.state_dict())))
        self.assertEqual(restored.row(0, rows), chosen[0])
        chosen += [restored.row(i, rows) for i in range(8, 12)]
        self.assertEqual(len({r["id"] for r in chosen}), 12)
        for i, row in enumerate(chosen):
            restored.commit(i + 1, [group(row, 8, index=i)])
        with self.assertRaisesRegex(ValueError, "reservation"):
            restored.commit(13, [group(chosen[0], 8)])
        self.assertTrue(all(q["visits"] == 1 for q in restored.questions.values()))

    def test_practice_does_not_touch_unaided_history_and_commits_once(self):
        rows = [task()]
        s = schedule(rows, keep_probability=1)
        unaided = s.questions["q"]["levels"][0]
        unaided.update(p=0.5, visits=2)
        before = copy.deepcopy(unaided)
        r = s.row(0, rows)
        self.assertEqual(r["evidence"], rows[0]["evidence"])
        s.commit(5, [group(r, 16)])
        self.assertEqual(s.questions["q"]["levels"][0], before)
        self.assertEqual(s.questions["q"]["conditions"]["evidence"]["correct"], 16)
        saved = s.state_dict()
        with self.assertRaisesRegex(ValueError, "reservation"):
            s.commit(5, [group(r, 16)])
        self.assertEqual(saved, s.state_dict())

    def test_rescue_records_each_level_and_weights_by_unaided_results(self):
        rows = [task()]
        first = schedule(rows, keep_probability=0)
        metrics = outcome(first, rows, [(0, 0), (1, 16)])
        q = first.questions["q"]
        self.assertEqual([i["p"] for i in q["levels"]], [0, 1, None])
        self.assertEqual(first.weight(q), 0.02)
        self.assertEqual(metrics["assistance/extra_groups"], 1)
        self.assertEqual(metrics["assistance/extra_tokens"], 42)
        self.assertEqual(metrics["mixture/terminal/evidence"], 1)
        second = curriculum.EvidenceMixtureSchedule(rows, epoch=2, batch_size=2, keep_probability=0,
                                                    previous=first.state_dict())
        practice = second.row(0, rows)
        self.assertEqual(practice[curriculum.KEY]["level"], 0)
        outcome(second, rows, [(0, 8)])
        self.assertEqual(second.questions["q"]["levels"][0]["p"], 0.25)
        self.assertEqual(second.weight(second.questions["q"]), 0.25 * 0.75)
        self.assertEqual(second.questions["q"]["levels"][1]["visits"], 1)
        # A mixed rescue does not lift the weight: only unaided results count.
        mixed = schedule(rows, keep_probability=0)
        outcome(mixed, rows, [(0, 0), (1, 8)])
        self.assertEqual(mixed.weight(mixed.questions["q"]), 0.02)

    def test_hint_success_does_not_update_evidence_mastery(self):
        rows, hints = [task()], {"q": "training reference"}
        s = schedule(rows, keep_probability=0, hints=hints)
        metrics = outcome(s, rows, [(0, 0), (1, 0), (2, 16)])
        q = s.questions["q"]
        self.assertEqual([i["p"] for i in q["levels"]], [0, 0, 1])
        self.assertEqual([i["failures"] for i in q["levels"]], [1, 1, 0])
        self.assertEqual({k: v["visits"] for k, v in q["conditions"].items()}, {"none": 1, "evidence": 1, "hint": 1})
        self.assertEqual(metrics["mixture/terminal/hint"], 1)
        self.assertEqual(metrics["assistance/extra_groups"], 2)

    def test_weight(self):
        def question(p, last_p):
            return {"levels": [{"p": p, "last_p": last_p, "visits": 1, "failures": 0}] + [{}] * 2, "visits": 1}

        cases = [((None, None), 0.25), ((0.0, 0.0), 0.02), ((1.0, 1.0), 0.02), ((0.5, 0.5), 0.25),
                 ((0.25, 0.5), 0.1875), ((0.99, 0.75), 0.02)]
        for (p, last_p), expected in cases:
            with self.subTest(p=p, last_p=last_p):
                self.assertAlmostEqual(curriculum.AssistanceSchedule.weight(question(p, last_p)), expected)

    def test_epoch_visits_are_bounded_and_running_questions_are_not_drawn(self):
        rows = [task("a"), task("b")]
        first = schedule(rows)
        for i in range(2):
            outcome(first, rows, [(first.row(i, rows)[curriculum.KEY]["level"], 8)], index=i)
        second = curriculum.EvidenceMixtureSchedule(rows, epoch=2, batch_size=2, previous=first.state_dict())
        running = [second.row(i, rows)["id"] for i in range(2)]
        self.assertEqual(sorted(running), ["a", "b"])
        with self.assertRaisesRegex(ValueError, "capacity exhausted"):
            second.row(2, rows)
        single = [task()]
        one = schedule(single)
        outcome(one, single, [(one.row(0, single)[curriculum.KEY]["level"], 8)])
        practice = curriculum.EvidenceMixtureSchedule(single, epoch=2, batch_size=2, previous=one.state_dict())
        self.assertEqual(curriculum.MAX_EPOCH_VISITS, 4)
        for i in range(4):
            row = practice.row(i, single)
            practice.commit(i + 1, [group(row, 8, index=i)])
        with self.assertRaisesRegex(ValueError, "capacity exhausted"):
            practice.row(4, single)

    def test_previous_epoch_must_match_and_be_fully_committed(self):
        rows = [task("a"), task("b")]
        first = schedule(rows)
        first.row(0, rows)
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            curriculum.EvidenceMixtureSchedule(rows, epoch=2, batch_size=2, previous=first.state_dict())
        for settings in ({"keep_probability": 0.8}, {"seed": 7}, {"batch_size": 4}, {"hints": {"a": "hint"}},
                         {"weighting": "lane"}, {"failed_weight": 0.05}, {"solved_weight": 0.0625},
                         {"fade_probability": 0.5}, {"include_empty_evidence": True}):
            with (
                self.subTest(**{k: str(v) for k, v in settings.items()}),
                self.assertRaisesRegex(ValueError, "mismatch"),
            ):
                curriculum.EvidenceMixtureSchedule(
                    rows, epoch=2, previous=first.state_dict(), **{"batch_size": 2, **settings}
                )
        with self.assertRaisesRegex(ValueError, "mismatch"):
            curriculum.EvidenceMixtureSchedule(rows, epoch=3, batch_size=2, previous=first.state_dict())

    def test_commit_rejects_a_rescue_chain_the_engine_cannot_produce(self):
        rows, hints = [task()], {"q": "training reference"}
        none = schedule(rows, keep_probability=0, hints=hints)
        row = none.row(0, rows)
        observed = {"level": 0, "correct": 0, "total": 16, "tokens": 4, "policy_steps": [0]}
        for history, level in (
            ([], 1),  # starts above its lane
            ([{**observed, "correct": 3}], 1),  # rescued although a rollout was correct
            ([observed, {**observed, "level": 2}], 1),  # moves down a level
        ):
            with self.subTest(history=history, level=level):
                staged = curriculum.stage_row(row, level, hint=hints["q"], history=history)
                with self.assertRaisesRegex(ValueError, "rescue history"):
                    none.commit(1, [group(staged, 8)])
        trusted = schedule(rows, keep_probability=1, hints=hints)
        row = trusted.row(0, rows)
        with self.assertRaisesRegex(ValueError, "rescue history"):
            trusted.commit(1, [group(curriculum.stage_row(row, 2, hint="h", history=[observed]), 8)])
        with self.assertRaisesRegex(ValueError, "reservation"):
            trusted.commit(1, [group({**row, curriculum.KEY: {**row[curriculum.KEY], "mixture": {}}}, 8)])
        self.assertEqual(trusted.completed, set())

    def test_contract_rejects_silent_policy_changes(self):
        rows = [task()]
        s = schedule(rows)
        s.row(0, rows)
        changed = schedule(rows, keep_probability=0.8)
        with self.assertRaisesRegex(ValueError, "contract"):
            changed.load_state_dict(s.state_dict())
        for settings in OPTIONS:
            with self.subTest(**settings):
                option = schedule(rows, **settings)
                with self.assertRaisesRegex(ValueError, "contract"):
                    option.load_state_dict(s.state_dict())
                option.row(0, rows)
                with self.assertRaisesRegex(ValueError, "contract"):
                    schedule(rows).load_state_dict(option.state_dict())
        other_schema =compose_problem("CREATE TABLE u(n INTEGER);", evidence="stored representation",
                                       question="Which item?")
        retitled = [{**task(), "problem": other_schema}]
        with self.assertRaisesRegex(ValueError, "contract"):
            schedule(retitled).load_state_dict(s.state_dict())
        base = curriculum.AssistanceSchedule(rows, epoch=1, batch_size=2)
        with self.assertRaisesRegex(ValueError, "contract"):
            base.load_state_dict(s.state_dict())

    def test_reservation_survives_crash_before_trainer_snapshot(self):
        rows = [task(str(i)) for i in range(8)]
        with tempfile.TemporaryDirectory() as root:
            original = schedule(rows, reservations_dir=root)
            chosen = original.row(0, rows)
            self.assertTrue((Path(root) / "0.json").exists())
            # No state_dict: the process died after admission but before snapshot.
            fresh = schedule(rows, reservations_dir=root)
            self.assertEqual(fresh.row(0, rows), chosen)
            changed = schedule(rows, reservations_dir=root, keep_probability=1)
            with self.assertRaisesRegex(ValueError, "contract"):
                changed.row(0, rows)

    def test_empty_evidence_emits_shortfall_not_false_trusted_exposure(self):
        rows = [task(evidence="")]
        s = schedule(rows, keep_probability=1)
        r = s.row(0, rows)
        metrics = s.commit(1, [group(r, 8)])
        self.assertEqual(metrics["mixture/lane_shortfall"], 1)
        self.assertEqual(metrics["mixture/scheduled/none"], 1)
        self.assertEqual(metrics["mixture/scheduled/trusted"], 0)

    def test_invalid_settings_fail_closed(self):
        rows = [task()]
        for kwargs in (
            {"keep_probability": 1.5}, {"keep_probability": -0.1}, {"epoch": 0}, {"batch_size": 0},
            {"weighting": "Lane"}, {"weighting": None}, {"failed_weight": 0}, {"failed_weight": -0.05},
            {"failed_weight": 1.5}, {"failed_weight": float("nan")}, {"solved_weight": 0}, {"solved_weight": -0.05},
            {"solved_weight": 1.5}, {"solved_weight": float("nan")}, {"fade_probability": -0.1},
            {"fade_probability": 1.5}, {"fade_probability": float("nan")}, {"include_empty_evidence": 1},
            {"include_empty_evidence": "yes"}, {"include_empty_evidence": None},
        ):
            error = TypeError if "include_empty_evidence" in kwargs else ValueError   # a flag that is not a bool
            with self.subTest(**kwargs), self.assertRaises(error):
                curriculum.EvidenceMixtureSchedule(rows, **{"epoch": 1, "batch_size": 2, **kwargs})
        for kwargs in ({"failed_weight": 1}, {"solved_weight": 1}, {"fade_probability": 0},
                       {"fade_probability": 1}):  # bounds are allowed
            with self.subTest(**kwargs):
                curriculum.EvidenceMixtureSchedule(rows, **{"epoch": 1, "batch_size": 2, **kwargs})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            schedule([task(), task()])


# --- Practice options: lane weighting, the weights of a full fail and a full pass, the evidence fade, and empty
# evidence in evidence draws -----------------------------------------------------------------------------------

# Each option at a value other than its default.
OPTIONS = ({"weighting": "lane"}, {"failed_weight": 0.05}, {"solved_weight": 0.0625}, {"fade_probability": 0.5},
           {"include_empty_evidence": True})


class PracticeOptionTests(unittest.TestCase):
    """Each option is off by default; `ScheduleTraceTest` checks the default draws against the golden trace.

    Draw counts come from fixed seeds, so they are exact; the bounds sit well clear of the expected share.
    """

    def test_defaults_stay_out_of_the_contract_and_the_reservations(self):
        rows = [task(str(i)) for i in range(4)]
        implicit = schedule(rows)
        explicit = schedule(rows, weighting="unaided", failed_weight=0.02, solved_weight=0.02, fade_probability=0.0,
                            include_empty_evidence=False)
        self.assertEqual(explicit.contract(), implicit.contract())
        self.assertFalse({name for option in OPTIONS for name in option} & set(implicit.contract()))
        self.assertEqual([explicit.row(i, rows) for i in range(4)], [implicit.row(i, rows) for i in range(4)])
        self.assertTrue(all("faded" not in choice for choice in implicit.reservations.values()))
        for settings in OPTIONS:
            with self.subTest(**settings):
                self.assertEqual(schedule(rows, **settings).contract(), {**implicit.contract(), **settings})

    def test_lane_weighting_reads_the_record_of_the_lane_drawn(self):
        question = {"levels": [record(0.5, p=0.25), record(0.25, p=0.125), record()], "visits": 2}
        unaided, lane = schedule([task()]), schedule([task()], weighting="lane")
        for drawn in curriculum.LANES:
            self.assertEqual(unaided.draw_weight(question, drawn), 0.25 * 0.75)  # the unaided record, always
        self.assertEqual(lane.draw_weight(question, "none"), 0.25 * 0.75)
        self.assertEqual(lane.draw_weight(question, "trusted"), 0.125 * 0.875)
        # In a draw: "a" is partly solved with its evidence and solved unaided, "b" the other way round, so in each
        # case one weighs 0.25 and the other the 0.02 floor (about 93% of the draws against 7%).
        rows = [task("a"), task("b")]
        records = {"a": (record(1.0), record(0.5)), "b": (record(0.5), record(1.0))}
        for keep, weighting, favoured in ((1, "lane", "a"), (1, "unaided", "b"), (0, "lane", "b"), (0, "unaided", "b")):
            with self.subTest(keep_probability=keep, weighting=weighting):
                s = practice_schedule(rows, records, keep_probability=keep, weighting=weighting)
                drawn = Counter(choice["qid"] for choice in independent_draws(s, rows, 400))
                self.assertGreater(drawn[favoured], 340, drawn)

    def test_a_question_never_tried_in_the_drawn_lane_gets_the_top_weight(self):
        top = 0.25  # p(1 - p) at p = 0.5, the weight before any result
        lane = schedule([task()], weighting="lane", failed_weight=0.05)
        for other in (record(0.0), record(1.0), record(0.5, p=0.25)):
            with self.subTest(other_lane=other):
                # No fallback to the other lane's record, whatever it holds.
                self.assertEqual(lane.draw_weight({"levels": [other, record(), record()]}, "trusted"), top)
                self.assertEqual(lane.draw_weight({"levels": [record(), other, record()]}, "none"), top)
        # In a draw: "new" was solved unaided on its first visit, so it has never been shown its evidence; the others
        # were solved with theirs. Its first evidence draw weighs 0.25 against two floors: about 86% of the draws.
        rows = [task("new"), task("x"), task("y")]
        records = {"new": (record(1.0), record()), "x": (record(), record(1.0)), "y": (record(), record(1.0))}
        s = practice_schedule(rows, records, keep_probability=1, weighting="lane")
        drawn = Counter(choice["qid"] for choice in independent_draws(s, rows, 300))
        self.assertGreater(drawn["new"], 230, drawn)

    def test_failed_weight_is_the_weight_of_a_last_full_fail_in_the_record_read(self):
        failed, solved, partly = record(0.0), record(1.0), record(0.5)
        for weighting in curriculum.WEIGHTINGS:
            s = schedule([task()], weighting=weighting, failed_weight=0.05)
            with self.subTest(weighting=weighting):
                self.assertEqual(s.draw_weight({"levels": [failed, partly, record()]}, "none"), 0.05)
                self.assertEqual(s.draw_weight({"levels": [solved, partly, record()]}, "none"), curriculum.WEIGHT_FLOOR)
        lane = schedule([task()], weighting="lane", failed_weight=0.05)
        self.assertEqual(lane.draw_weight({"levels": [partly, failed, record()]}, "trusted"), 0.05)
        self.assertEqual(lane.draw_weight({"levels": [failed, partly, record()]}, "trusted"), 0.25)  # not this lane's
        self.assertEqual(lane.draw_weight({"levels": [partly, solved, record()]}, "trusted"), curriculum.WEIGHT_FLOOR)
        # By default a full fail weighs the floor, like a full pass.
        self.assertEqual(schedule([task()]).draw_weight({"levels": [failed, partly, record()]}, "trusted"), 0.02)
        # In a draw (unaided weighting): a question last failed unaided, then rescued with its evidence, against one
        # solved unaided. It takes half the draws by default and about 5/7 with a weight of 0.05.
        rows = [task("failed"), task("solved")]
        records = {"failed": (record(0.0), record(1.0)), "solved": (record(1.0), record())}
        for failed_weight, low, high in ((0.02, 160, 240), (0.05, 255, 315)):
            with self.subTest(failed_weight=failed_weight):
                s = practice_schedule(rows, records, keep_probability=1, failed_weight=failed_weight)
                drawn = Counter(choice["qid"] for choice in independent_draws(s, rows, 400))
                self.assertTrue(low < drawn["failed"] < high, drawn)

    def test_solved_weight_is_the_weight_of_a_last_full_pass_in_the_record_read(self):
        failed, solved, partly = record(0.0), record(1.0), record(0.5)
        for weighting in curriculum.WEIGHTINGS:
            s = schedule([task()], weighting=weighting, solved_weight=0.0625)
            with self.subTest(weighting=weighting):
                self.assertEqual(s.draw_weight({"levels": [solved, partly, record()]}, "none"), 0.0625)
                self.assertEqual(s.draw_weight({"levels": [failed, partly, record()]}, "none"), curriculum.WEIGHT_FLOOR)
        lane = schedule([task()], weighting="lane", solved_weight=0.0625)
        self.assertEqual(lane.draw_weight({"levels": [partly, solved, record()]}, "trusted"), 0.0625)
        self.assertEqual(lane.draw_weight({"levels": [solved, partly, record()]}, "trusted"), 0.25)  # not this lane's
        # By default a full pass weighs the floor.
        default = schedule([task()], weighting="lane")
        self.assertEqual(default.draw_weight({"levels": [partly, solved, record()]}, "trusted"), curriculum.WEIGHT_FLOOR)
        # In a draw (lane weighting, evidence always): a question solved with its evidence against one half solved
        # with it (0.25). The solved one takes about 7% of the draws at the floor and 20% with a weight of 0.0625.
        rows = [task("solved"), task("half")]
        records = {"solved": (record(), record(1.0)), "half": (record(), record(0.5))}
        for solved_weight, low, high in ((0.02, 10, 50), (0.0625, 55, 110)):
            with self.subTest(solved_weight=solved_weight):
                s = practice_schedule(rows, records, keep_probability=1, weighting="lane", solved_weight=solved_weight)
                drawn = Counter(choice["qid"] for choice in independent_draws(s, rows, 400))
                self.assertTrue(low < drawn["solved"] < high, drawn)

    def test_the_fade_never_changes_which_question_is_drawn(self):
        # Three in four questions were last solved 16/16 with their evidence, some of them also solved or failed
        # unaided; the others are untested, half solved or failed with their evidence. With the fade on, every
        # reservation draws the same question as with it off; only the draws of solved questions may lose the evidence.
        rows = [task(f"q{i:02d}") for i in range(40)]
        unaided = (record(), record(1.0), record(0.0))
        others = (record(), record(0.5), record(0.0))
        records = {row["id"]: (unaided[i % 3], record(1.0) if i % 4 else others[i % 3]) for i, row in enumerate(rows)}
        settings = {"keep_probability": 1, "weighting": "lane", "failed_weight": 0.05}
        baseline = independent_draws(practice_schedule(rows, records, **settings), rows, 600)
        for fade, low, high in ((0.5, 0.35, 0.65), (1.0, 1.0, 1.0)):
            with self.subTest(fade_probability=fade):
                s = practice_schedule(rows, records, fade_probability=fade, **settings)
                choices = independent_draws(s, rows, 600)
                self.assertEqual([c["qid"] for c in choices], [c["qid"] for c in baseline])
                for before, after in zip(baseline, choices, strict=True):
                    if after["faded"]:
                        self.assertEqual(records[after["qid"]][1]["last_p"], 1)
                        self.assertEqual((after["lane"], after["level"], after["evidence"]), ("none", 0, ""))
                        self.assertEqual(after["requested_lane"], "trusted")
                    else:
                        self.assertEqual({k: v for k, v in after.items() if k != "faded"}, before)
                solved = [c for c in choices if records[c["qid"]][1]["last_p"] == 1]
                share = sum(c["faded"] for c in solved) / len(solved)
                self.assertTrue(low <= share <= high, share)

    def test_a_solved_question_keeps_its_weight_through_draws_with_and_without_its_evidence(self):
        # Solved 16/16 with its evidence and never tried unaided. Over three epochs each draw flips its own coin: the
        # draws without the evidence get half right, the ones with it stay 16/16. Its weight is its evidence record's
        # throughout, and its unaided results are kept apart.
        rows = [task()]
        settings = {"keep_probability": 1, "weighting": "lane", "failed_weight": 0.05, "solved_weight": 0.0625,
                    "fade_probability": 0.5}
        s = practice_schedule(rows, {"q": (record(), record(1.0))}, **settings)
        lanes = []
        for epoch in (2, 3, 4):
            if epoch > 2:
                s = curriculum.EvidenceMixtureSchedule(rows, epoch=epoch, batch_size=4, previous=s.state_dict(),
                                                       **settings)
            for index in range(curriculum.MAX_EPOCH_VISITS):
                self.assertEqual(s.draw_weight(s.questions["q"], "trusted"), 0.0625)
                choice = s.row(index, rows)[curriculum.KEY]["mixture"]
                lanes.append(choice["lane"])
                outcome(s, rows, [(0, 8)] if choice["faded"] else [(1, 16)], index=index)
        self.assertGreaterEqual(lanes.count("none"), 2, lanes)
        self.assertGreaterEqual(lanes.count("trusted"), 2, lanes)
        levels = s.questions["q"]["levels"]
        self.assertEqual((levels[0]["visits"], levels[0]["last_p"]), (lanes.count("none"), 0.5))
        self.assertEqual((levels[1]["visits"], levels[1]["last_p"]), (1 + lanes.count("trusted"), 1.0))

    def test_empty_evidence_questions_join_evidence_draws_when_included(self):
        # With the evidence always kept, every practice draw is an evidence draw, so by default a question whose
        # evidence is empty is never drawn again after its first pass. Included, it runs unaided, is weighted by its
        # unaided record (half solved: 0.25 against two solved questions at the floor, about 86% of the draws), is
        # never faded, and counts as a lane shortfall, as in its first pass.
        rows = [task("bare", evidence=""), task("a"), task("b")]
        records = {"bare": (record(0.5), record()), "a": (record(), record(1.0)), "b": (record(), record(1.0))}
        settings = {"keep_probability": 1, "weighting": "lane", "fade_probability": 0.5}
        excluded = independent_draws(practice_schedule(rows, records, **settings), rows, 300)
        self.assertNotIn("bare", {choice["qid"] for choice in excluded})
        choices = independent_draws(practice_schedule(rows, records, include_empty_evidence=True, **settings), rows, 300)
        bare = [choice for choice in choices if choice["qid"] == "bare"]
        self.assertGreater(len(bare), 230)
        for choice in bare:
            shown = (choice["lane"], choice["requested_lane"], choice["level"], choice["evidence"], choice["faded"])
            self.assertEqual(shown, ("none", "trusted", 0, "", False))
        index = next(i for i, choice in enumerate(choices) if choice["qid"] == "bare")
        s = practice_schedule(rows, records, include_empty_evidence=True, **settings)
        metrics = outcome(s, rows, [(0, 8)], index=index)
        self.assertEqual((metrics["mixture/lane_shortfall"], metrics["mixture/faded"]), (1, 0))
        self.assertEqual(s.questions["bare"]["levels"][0]["visits"], 2)

    def test_fade_withholds_the_evidence_from_mastered_questions_only(self):
        rows = [task(f"q{i:02d}") for i in range(80)]
        # Odd questions were last solved 16/16 with their evidence; even ones were not (untested, partly solved or
        # failed with it). None was tried unaided, so under unaided weighting all weigh the same.
        others = (record(), record(0.5), record(0.0))
        records = {row["id"]: (record(), record(1.0) if i % 2 else others[i % 3]) for i, row in enumerate(rows)}
        mastered = {row["id"] for i, row in enumerate(rows) if i % 2}
        for fade, low, high in ((0.5, 0.44, 0.56), (1.0, 1.0, 1.0)):
            with self.subTest(fade_probability=fade):
                s = practice_schedule(rows, records, keep_probability=1, fade_probability=fade)
                choices = independent_draws(s, rows, 1000)
                faded = [choice for choice in choices if choice["faded"]]
                self.assertTrue(faded)
                self.assertLessEqual({choice["qid"] for choice in faded}, mastered)
                share = len(faded) / sum(choice["qid"] in mastered for choice in choices)
                self.assertTrue(low <= share <= high, share)
                # With the evidence always kept, the fade is the only source of unaided draws.
                self.assertEqual([choice for choice in choices if choice["lane"] == "none"], faded)
                for choice in choices:
                    shown = ("trusted", 1, True) if not choice["faded"] else ("none", 0, False)
                    self.assertEqual((choice["lane"], choice["level"], bool(choice["evidence"])), shown)
                    self.assertEqual(choice["requested_lane"], "trusted")
        # The coin is per draw, not per question: one mastered question alone is faded on about half its draws.
        alone = [task()]
        s = practice_schedule(alone, {"q": (record(), record(1.0))}, keep_probability=1, fade_probability=0.5)
        share = sum(choice["faded"] for choice in independent_draws(s, alone, 300)) / 300
        self.assertTrue(0.4 <= share <= 0.6, share)

    def test_fade_choices_are_reproducible_and_survive_a_restart(self):
        rows = [task(f"q{i:02d}") for i in range(12)]
        records = {row["id"]: (record(), record(1.0)) for row in rows}  # all mastered with their evidence
        settings = {"keep_probability": 1, "fade_probability": 0.5}
        with tempfile.TemporaryDirectory() as root:
            original = practice_schedule(rows, records, reservations_dir=root, **settings)
            chosen = [original.row(i, rows) for i in range(8)]
            self.assertEqual({row[curriculum.KEY]["mixture"]["faded"] for row in chosen}, {False, True})
            again = practice_schedule(rows, records, **settings)
            self.assertEqual([again.row(i, rows) for i in range(8)], chosen)
            # From the saved state, after a JSON round trip: reserved rows come back unchanged and later draws agree.
            saved = json.loads(json.dumps(original.state_dict()))
            restored = practice_schedule(rows, records, **settings)
            restored.load_state_dict(saved)
            self.assertEqual([restored.row(i, rows) for i in range(8)], chosen)
            later = [original.row(i, rows) for i in range(8, 11)]
            self.assertEqual([restored.row(i, rows) for i in range(8, 11)], later)
            # A crash before the trainer saved its state: the reservation files alone restore every choice.
            fresh = practice_schedule(rows, records, reservations_dir=root, **settings)
            self.assertEqual([fresh.row(i, rows) for i in range(11)], chosen + later)
            # Another fade probability reuses neither.
            other = practice_schedule(rows, records, reservations_dir=root, keep_probability=1, fade_probability=0.25)
            with self.assertRaisesRegex(ValueError, "contract"):
                other.load_state_dict(saved)
            with self.assertRaisesRegex(ValueError, "contract"):
                other.row(0, rows)

    def test_a_faded_draw_that_fails_is_rescued_with_its_evidence_then_its_hint(self):
        rows, hints = [task()], {"q": "training reference"}
        for stages, terminal in (
            ([(0, 0), (1, 0), (2, 16)], "hint"),
            ([(0, 0), (1, 16)], "evidence"),
            ([(0, 16)], "none"),
        ):
            with self.subTest(stages=stages):
                s = practice_schedule(rows, {"q": (record(), record(1.0))}, keep_probability=1, fade_probability=1,
                                      hints=hints)
                metrics = outcome(s, rows, stages)
                entry = s.history[0]
                self.assertEqual((entry["lane"], entry["requested_lane"], entry["faded"]), ("none", "trusted", True))
                self.assertEqual([obs["level"] for obs in entry["observations"]], [level for level, _ in stages])
                self.assertEqual(metrics["mixture/faded"], 1)
                self.assertEqual(metrics["mixture/faded/correct"], stages[0][1])
                self.assertEqual(metrics["mixture/faded/total"], 16)
                self.assertEqual(metrics["mixture/faded/rescued"], float(len(stages) > 1))
                self.assertEqual(metrics["mixture/scheduled/none"], 1)
                self.assertEqual(metrics["mixture/lane_shortfall"], 0)  # a fade is not a shortfall
                self.assertEqual(metrics[f"mixture/terminal/{terminal}"], 1)
                self.assertEqual(metrics["assistance/extra_groups"], len(stages) - 1)
                # Mastery is the last evidence result: a failed evidence stage ends it and the next draw shows the
                # evidence again; otherwise the question keeps fading.
                following = s.row(1, rows)[curriculum.KEY]["mixture"]
                self.assertEqual(following["faded"], terminal != "hint")

    def test_a_question_mastered_both_ways_sits_at_the_floor(self):
        both = {"levels": [record(1.0), record(1.0), record()]}
        for weighting in curriculum.WEIGHTINGS:
            s = schedule([task()], weighting=weighting, failed_weight=0.05, fade_probability=0.5)
            for drawn in curriculum.LANES:
                with self.subTest(weighting=weighting, lane=drawn):
                    self.assertEqual(s.draw_weight(both, drawn), curriculum.WEIGHT_FLOOR)


# --- Rescue in the rollout engine and the trainer ---------------------------------------------------------------


class AsyncAssistanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_restore_completed_failed_stage_can_resubmit_once(self):
        async def sampler(model, kwargs):
            return [
                [sample("A" if "stored representation" in prompt_text(kwargs, i) else "B")]
                for i in range(len(kwargs["model_input"]))
            ]

        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                original = assistance_engine(Model(sampler))
                async with original:
                    original.submit(
                        curriculum.stage_row(task(), 0),
                        group_size=2,
                        completion=rl.GroupCompletion("wait"),
                        id="q",
                        priority=19,
                    )
                    # Interrupt after completion but before the assistance
                    # receiver consumes the failed stage from the outbox.
                    await asyncio.wait_for(rl.RolloutEngine.next_group(original), 5)
                    state = await original.state_dict()
                if legacy:
                    for group_state in state["groups"]:
                        del group_state["admission_order"]
                restored = assistance_engine(Model(sampler))
                async with restored:
                    restored.load_state_dict(state)
                    result = await asyncio.wait_for(restored.next_group(), 5)
                    self.assertEqual(restored._groups["q"].priority, 19)
                    self.assertEqual(result.row[curriculum.KEY]["level"], 1)
                    self.assertEqual(len(result.row[curriculum.KEY]["history"]), 1)
                    self.assertTrue(all(t.metrics["correct"] == 1 for t in result.trajectories))
                    restored.accept(result)
                    restored.acknowledge(result.id)
                    self.assertTrue(restored._completed.empty())
                    self.assertEqual(restored._outbox, [])

    async def test_queued_rescues_retain_admission_priority_through_hint_and_restart(self):
        for restart in (False, True):
            with self.subTest(restart=restart):
                await self.queued_rescue_case(restart)

    async def queued_rescue_case(self, restart):
        started = []
        blocking = asyncio.Event()
        release = asyncio.Event()

        class QueuedTrial(Trial):
            async def reset(self, row):
                started.append((row["id"], row[curriculum.KEY]["level"]))
                if row["id"] == "occupant":
                    if started.count(("occupant", 0)) == 2:
                        blocking.set()
                    await release.wait()
                return await super().reset(row)

            async def reward(self, traj, row):
                correct = row["id"] != "rescue" or row[curriculum.KEY]["level"] == 2
                traj.metrics["correct"] = float(correct)
                return float(correct)

        def build():
            return assistance_engine(
                Model(),
                hints={"rescue": "training reference"},
                env=QueuedTrial,
                schedule=rl.Schedule(window=0.001, concurrency=2, max_batch=2),
            )

        async def collect(engine):
            results = []
            while len(results) < 3:
                result = await engine.next_group()
                engine.accept(result)
                engine.acknowledge(result.id)
                results.append(result)
            return results

        engine = build()
        async with engine:
            for qid in ("rescue", "occupant", "later"):
                engine.submit(
                    curriculum.stage_row(task(qid), 0),
                    group_size=2,
                    completion=rl.GroupCompletion("wait"),
                    id=qid,
                    priority=24,
                )
            receiving = asyncio.create_task(collect(engine))
            await asyncio.wait_for(blocking.wait(), 5)

            async def wait_rescue():
                while engine._groups["rescue"].row[curriculum.KEY]["level"] == 0:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_rescue(), 5)
            self.assertNotIn(("rescue", 1), started)
            state = await engine.state_dict()
            rescue = next(g for g in state["groups"] if g["id"] == "rescue")
            self.assertEqual(rescue["priority"], 24)
            self.assertEqual(rescue["admission_order"], 0)
            self.assertEqual(len(rescue["row"][curriculum.KEY]["history"]), 1)
            if restart:
                receiving.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await receiving
            else:
                release.set()
                results = await asyncio.wait_for(receiving, 5)
        if restart:
            restored = build()
            async with restored:
                restored.load_state_dict(state)
                release.set()
                results = await asyncio.wait_for(collect(restored), 5)
        self.assertLess(started.index(("rescue", 1)), started.index(("later", 0)))
        terminal = next(g for g in results if g.id == "rescue")
        self.assertEqual(terminal.row[curriculum.KEY]["level"], 2)
        self.assertEqual([o["correct"] for o in terminal.row[curriculum.KEY]["history"]], [0, 0])
        self.assertEqual(len(terminal.trajectories), 2)
        self.assertTrue(all(t.metrics["correct"] == 1 for t in terminal.trajectories))
        self.assertEqual(sorted(g.id for g in results), ["later", "occupant", "rescue"])

    async def test_first_update_does_not_wait_for_later_queued_group(self):
        occupied, release_occupant, release_later = asyncio.Event(), asyncio.Event(), asyncio.Event()
        started = []

        class SaturatedTrial(Trial):
            async def reset(self, row):
                started.append((row["id"], row[curriculum.KEY]["level"]))
                if row["id"] == "occupant":
                    occupied.set()
                    await release_occupant.wait()
                if row["id"] == "later":
                    await release_later.wait()
                return await super().reset(row)

            async def reward(self, traj, row):
                correct = row["id"] != "rescue" or row[curriculum.KEY]["level"] == 1
                traj.metrics["correct"] = float(correct)
                # Keep a within-group shaping contrast so this fixture performs
                # an optimizer update, rather than skipping a uniform group.
                return float(correct) - 0.1 * (traj.final_text == "B")

        model = Model()
        rows = [task(qid) for qid in ("rescue", "occupant", "later")]
        schedule = curriculum.EvidenceMixtureSchedule(rows, epoch=1, batch_size=1, keep_probability=0)
        # Freeze the admitted question order, as restored reservations do; the
        # curriculum's randomized coverage order is not the behavior under test.
        schedule.reservations = {str(i): unaided_reservation(row["id"], i) for i, row in enumerate(rows)}
        worker = trainer(
            model,
            engine=assistance_engine(
                model,
                env=SaturatedTrial,
                schedule=rl.Schedule(window=0.001, concurrency=2, max_batch=2),
            ),
            dataset_scheduler=schedule,
            max_staleness=2,
            repeat_dataset=False,
            batch_order="admission",
        )
        stream = worker.run(rows, steps=3)
        first = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(occupied.wait(), 5)

            # Let the failed group's receiver submit the rescue while the next
            # group owns all slots and a third group is already waiting.
            async def wait_rescue():
                while worker.engine._groups["group-0"].row[curriculum.KEY]["level"] == 0:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_rescue(), 5)
            release_occupant.set()
            step = await asyncio.wait_for(first, 5)
            self.assertEqual(model.step, 1)
            self.assertEqual(step.metrics["assistance/extra_groups"], 1)
            self.assertTrue(all(t.metrics["correct"] == 1 for t in step.trajectories))
            self.assertLess(started.index(("rescue", 1)), started.index(("later", 0)))
            release_later.set()
            remaining = [s async for s in stream]
            self.assertEqual(len(remaining), 2)
            self.assertEqual([h["qid"] for h in schedule.history], [r["id"] for r in rows])
        finally:
            release_occupant.set()
            release_later.set()
            if not first.done():
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
            await stream.aclose()

    async def test_whole_group_rescue_is_fresh_and_only_terminal_group_trains(self):
        prompts = []

        async def sampler(model, kwargs):
            results = []
            for i in range(len(kwargs["model_input"])):
                prompt = prompt_text(kwargs, i)
                prompts.append(prompt)
                informed = "stored representation" in prompt
                results.append([sample("A" if informed and i % 2 else "B", step=model.step)])
            return results

        model = Model(sampler)
        rows = [task()]
        schedule = curriculum.EvidenceMixtureSchedule(rows, epoch=1, batch_size=1, keep_probability=0)
        worker = trainer(
            model,
            engine=assistance_engine(model),
            dataset_scheduler=schedule,
            group_size=16,
            repeat_dataset=False,
        )
        steps = [s async for s in worker.run(rows, steps=1)]
        self.assertEqual(len(prompts), 32)
        self.assertTrue(all(p.count("assistant:") == 1 for p in prompts))
        self.assertEqual(len(steps[0].trajectories), 16)
        self.assertTrue(all("stored representation" in t.messages[0]["content"] for t in steps[0].trajectories))
        self.assertEqual([o["correct"] for o in schedule.history[0]["observations"]], [0, 8])
        self.assertEqual(steps[0].metrics["assistance/extra_groups"], 1)
        self.assertEqual(model.step, 1)

    async def test_uniform_rescue_does_not_manufacture_correctness_contrast(self):
        async def sampler(model, kwargs):
            return [
                [sample("A" if "stored representation" in prompt_text(kwargs, i) else "B", step=model.step)]
                for i in range(len(kwargs["model_input"]))
            ]

        model, rows = Model(sampler), [task()]
        schedule = curriculum.EvidenceMixtureSchedule(rows, epoch=1, batch_size=1, keep_probability=0)
        worker = trainer(
            model,
            engine=assistance_engine(model),
            dataset_scheduler=schedule,
            group_size=16,
            repeat_dataset=False,
        )
        steps = [s async for s in worker.run(rows, steps=1)]
        self.assertEqual(steps[0].metrics["reward/std"], 0)
        self.assertEqual([i["p"] for i in schedule.questions["q"]["levels"]], [0, 1, None])
        for call in model.calls:
            if call[0] == "backward":
                self.assertTrue(all(not any(item["advantages"]) for item in call[1]))

    async def test_restart_during_rescue_keeps_history_and_commits_once(self):
        rows = [task()]
        failed = False
        stages = []
        rescue_started, interrupt = asyncio.Event(), asyncio.Event()

        async def sampler(model, kwargs):
            nonlocal failed
            informed = "stored representation" in prompt_text(kwargs, 0)
            stages.append(informed)
            if informed and not failed:
                failed = True
                rescue_started.set()
                await interrupt.wait()
                raise RuntimeError("transport unavailable")
            return [
                [sample("A" if informed and i % 2 else "B", step=model.step)]
                for i in range(len(kwargs["model_input"]))
            ]

        with tempfile.TemporaryDirectory() as root:

            def build():
                model = Model(sampler)
                schedule = curriculum.EvidenceMixtureSchedule(
                    rows, epoch=1, batch_size=1, keep_probability=0, reservations_dir=Path(root) / "reservations"
                )
                worker = trainer(
                    model,
                    engine=assistance_engine(model),
                    dataset_scheduler=schedule,
                    group_size=2,
                    repeat_dataset=False,
                    checkpoint=rl.Checkpointing(root, weights_every=1),
                )
                return worker, schedule

            worker, schedule = build()
            running = asyncio.create_task(anext(worker.run(rows, steps=1)))
            await asyncio.wait_for(rescue_started.wait(), 5)
            # Inject a checkpoint during the assisted stage, then interrupt it.
            # Unsaved work may replay; committed scheduler outcomes may not.
            async with worker._lock:
                await worker._save_state()
            interrupt.set()
            with self.assertRaisesRegex(RuntimeError, "transport"):
                await running
            self.assertEqual(schedule.questions["q"]["visits"], 0)
            saved = json.loads((Path(root) / "trainer.json").read_text())
            self.assertEqual(saved["engine"]["groups"][0]["row"][curriculum.KEY]["history"][0]["correct"], 0)
            worker, schedule = build()
            self.assertEqual(len([s async for s in worker.run(rows, steps=1)]), 1)
            self.assertEqual(schedule.questions["q"]["visits"], 1)
            worker, schedule = build()
            self.assertEqual([s async for s in worker.run(rows, steps=1)], [])
            self.assertEqual(schedule.questions["q"]["visits"], 1)
            self.assertEqual(stages.count(False), 1)

    async def test_hint_escalation_and_empty_evidence_skip_duplicate_stage(self):
        for evidence, hints, expected_levels, expected_correct in (
            ("stored representation", {"q": "training reference"}, [0, 1, 2], 1),
            ("", {"q": "training reference"}, [0, 2], 1),
            ("", {}, [0], 0),  # nothing left to show: the failed group trains as it is
            ("stored representation", {}, [0, 1], 0),
        ):

            async def sampler(model, kwargs):
                return [
                    [sample("A" if "Training hint:" in prompt_text(kwargs, i) else "B", step=model.step)]
                    for i in range(len(kwargs["model_input"]))
                ]

            with self.subTest(evidence=evidence, hints=hints):
                model = Model(sampler)
                engine = assistance_engine(model, hints=hints)
                async with engine:
                    engine.submit(
                        curriculum.stage_row(task(evidence=evidence), 0),
                        group_size=2,
                        completion=rl.GroupCompletion("wait"),
                    )
                    result = await engine.next_group()
                    levels = [o["level"] for o in result.row[curriculum.KEY]["history"]] + [
                        result.row[curriculum.KEY]["level"]
                    ]
                    self.assertEqual(levels, expected_levels)
                    self.assertTrue(all(t.metrics["correct"] == expected_correct for t in result.trajectories))
                    self.assertEqual(len(result.trajectories), 2)

    async def test_checkpoint_receive_cancellation_does_not_lose_failed_group(self):
        async def sampler(model, kwargs):
            return [
                [sample("A" if "stored representation" in prompt_text(kwargs, i) else "B", step=model.step)]
                for i in range(len(kwargs["model_input"]))
            ]

        engine = assistance_engine(Model(sampler))
        async with engine:
            await engine.stage_lock.acquire()
            engine.submit(curriculum.stage_row(task(), 0), group_size=2, completion=rl.GroupCompletion("wait"))
            receiving = asyncio.create_task(engine.next_group())
            # Wait for the completed stage to leave the queue and block on the
            # checkpoint lock. A cancelled receiver must put it back.
            for attempt in range(1000):
                if engine._outbox and engine._completed.empty():
                    break
                await asyncio.sleep(0.001)
            self.assertTrue(engine._outbox)
            receiving.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await receiving
            engine.stage_lock.release()
            result = await asyncio.wait_for(engine.next_group(), 5)
            self.assertEqual(result.row[curriculum.KEY]["level"], 1)
            self.assertEqual(len(result.row[curriculum.KEY]["history"]), 1)

    async def test_slow_rescue_does_not_block_other_question(self):
        rescue_started, release = asyncio.Event(), asyncio.Event()

        async def sampler(model, kwargs):
            informed = "stored representation" in prompt_text(kwargs, 0)
            slow = "slow-question" in prompt_text(kwargs, 0)
            if slow and informed:
                rescue_started.set()
                await release.wait()
            return [
                [sample("A" if not slow or informed else "B", step=model.step)]
                for i in range(len(kwargs["model_input"]))
            ]

        engine = assistance_engine(Model(sampler))
        slow = task("slow")
        slow["question"] = "slow-question"
        slow["problem"] = compose_problem("CREATE TABLE t(n INTEGER);", evidence=slow["evidence"],
                                          question=slow["question"])
        async with engine:
            engine.submit(curriculum.stage_row(slow, 0), group_size=2, completion=rl.GroupCompletion("wait"),
                          id="slow")
            pending = asyncio.create_task(engine.next_group())
            await asyncio.wait_for(rescue_started.wait(), 5)
            engine.submit(curriculum.stage_row(task("fast"), 0), group_size=2, completion=rl.GroupCompletion("wait"),
                          id="fast")
            result = await asyncio.wait_for(pending, 5)
            self.assertEqual(result.id, "fast")
            release.set()
            self.assertEqual((await engine.next_group()).id, "slow")

    async def test_infrastructure_failure_does_not_escalate(self):
        async def sampler(model, kwargs):
            raise RuntimeError("transport unavailable")

        engine = assistance_engine(Model(sampler))
        async with engine:
            engine.submit(curriculum.stage_row(task(), 0), group_size=2, completion=rl.GroupCompletion("wait"))
            with self.assertRaisesRegex(RuntimeError, "transport"):
                await engine.next_group()
            state = await engine.state_dict()
            self.assertEqual(state["groups"][0]["row"][curriculum.KEY]["history"], [])
            self.assertEqual(state["groups"][0]["row"][curriculum.KEY]["level"], 0)

    async def test_trusted_lane_rescues_with_hint_and_only_the_hint_stage_trains(self):
        prompts = []

        async def sampler(model, kwargs):
            results = []
            for i in range(len(kwargs["model_input"])):
                text = prompt_text(kwargs, i)
                prompts.append(text)
                results.append([sample("A" if "Training hint:" in text else "B", step=model.step)])
            return results

        rows = [task()]
        s = curriculum.EvidenceMixtureSchedule(
            rows, epoch=1, batch_size=1, keep_probability=1, hints={"q": "reference"}
        )
        model = Model(sampler)
        worker = trainer(
            model,
            engine=assistance_engine(model, hints=s.hints),
            dataset_scheduler=s,
            group_size=2,
            repeat_dataset=False,
        )
        steps = [step async for step in worker.run(rows, steps=1)]
        self.assertEqual(len(prompts), 4)
        self.assertEqual(len(steps[0].trajectories), 2)
        self.assertTrue(all("stored representation" in p and "Training hint:" not in p for p in prompts[:2]))
        self.assertTrue(all("stored representation" in p and "Training hint:\nreference" in p for p in prompts[2:]))
        self.assertEqual(steps[0].metrics["mixture/scheduled/trusted"], 1)
        self.assertEqual(steps[0].metrics["mixture/terminal/hint"], 1)
        self.assertEqual(steps[0].metrics["mixture/evidence/total"], 2)
        self.assertEqual(steps[0].metrics["mixture/hint/correct"], 2)
        self.assertEqual(s.questions["q"]["levels"][0]["visits"], 0)
        self.assertEqual(sorted(s.questions["q"]["conditions"]), ["evidence", "hint"])

    async def test_two_epochs_chain_through_saved_trainer_state(self):
        rows = [task()]
        with tempfile.TemporaryDirectory() as root:
            prior = None
            for epoch in (1, 2):

                async def sampler(model, kwargs, epoch=epoch):
                    results = []
                    for i in range(len(kwargs["model_input"])):
                        informed = "stored representation" in prompt_text(kwargs, i)
                        correct = informed if epoch == 1 else i % 2 == 0
                        results.append([sample("A" if correct else "B", step=model.step)])
                    return results

                run_dir = Path(root) / f"{epoch:03d}"
                run_dir.mkdir()
                schedule = curriculum.EvidenceMixtureSchedule(
                    rows,
                    epoch=epoch,
                    batch_size=1,
                    keep_probability=0,
                    previous=prior,
                    reservations_dir=run_dir / "evidence-reservations",
                )
                model = Model(sampler)
                worker = trainer(
                    model,
                    engine=assistance_engine(model),
                    dataset_scheduler=schedule,
                    group_size=2,
                    repeat_dataset=False,
                    checkpoint=rl.Checkpointing(run_dir, weights_every=1),
                )
                steps = [step async for step in worker.run(rows, steps=1)]
                self.assertEqual(len(steps), 1)
                saved = json.loads((run_dir / "trainer.json").read_text())
                prior = saved["dataset_scheduler"]
                self.assertEqual(prior, schedule.state_dict())
                self.assertEqual(steps[0].metrics["assistance/extra_groups"], 2 - epoch)
                receipt = json.loads((run_dir / "evidence-reservations" / "0.json").read_text())
                self.assertEqual(receipt, {"contract": schedule.contract(), "choice": schedule.reservations["0"]})
            levels = prior["questions"]["q"]["levels"]
            self.assertEqual([level["visits"] for level in levels], [2, 1, 0])
            self.assertEqual(levels[0]["last_p"], 0.5)
            self.assertEqual(prior["questions"]["q"]["visits"], 2)
            self.assertEqual([h["visit"] for h in prior["history"]], ["2:0"])

    async def test_faded_draw_that_fails_is_rescued_with_its_evidence_in_the_same_visit(self):
        prompts = []

        async def sampler(model, kwargs):
            results = []
            for i in range(len(kwargs["model_input"])):
                text = prompt_text(kwargs, i)
                prompts.append(text)
                results.append([sample("A" if "stored representation" in text else "B", step=model.step)])
            return results

        rows = [task()]
        settings = {"keep_probability": 1, "weighting": "lane", "failed_weight": 0.05, "fade_probability": 1}
        with tempfile.TemporaryDirectory() as root:
            prior = None
            for epoch in (1, 2):
                run_dir = Path(root) / f"{epoch:03d}"
                run_dir.mkdir()
                schedule = curriculum.EvidenceMixtureSchedule(
                    rows,
                    epoch=epoch,
                    batch_size=1,
                    previous=prior,
                    reservations_dir=run_dir / "evidence-reservations",
                    **settings,
                )
                model = Model(sampler)
                worker = trainer(
                    model,
                    engine=assistance_engine(model),
                    dataset_scheduler=schedule,
                    group_size=2,
                    repeat_dataset=False,
                    checkpoint=rl.Checkpointing(run_dir, weights_every=1),
                )
                steps = [step async for step in worker.run(rows, steps=1)]
                prior = json.loads((run_dir / "trainer.json").read_text())["dataset_scheduler"]
                self.assertEqual(steps[0].metrics["mixture/faded"], epoch - 1)
            receipt = json.loads((Path(root) / "002" / "evidence-reservations" / "0.json").read_text())
        # Epoch 1 showed the evidence and solved the question; epoch 2 withheld it, failed every rollout, and re-ran
        # the group with the evidence. Only that last stage trained.
        self.assertEqual(["stored representation" in p for p in prompts], [True, True, False, False, True, True])
        self.assertTrue(all("stored representation" in t.messages[0]["content"] for t in steps[0].trajectories))
        entry = prior["history"][0]
        self.assertEqual((entry["lane"], entry["requested_lane"], entry["faded"]), ("none", "trusted", True))
        self.assertEqual([(obs["level"], obs["correct"]) for obs in entry["observations"]], [(0, 0), (1, 2)])
        self.assertEqual(receipt["choice"], schedule.reservations["0"])
        metrics = steps[0].metrics
        self.assertEqual(metrics["mixture/faded/rescued"], 1)
        self.assertEqual(metrics["mixture/terminal/evidence"], 1)
        self.assertEqual(metrics["assistance/extra_groups"], 1)
        self.assertEqual(metrics["mixture/lane_shortfall"], 0)


# --- Scheduler trace: golden fixture ----------------------------------------------------------------------

TRACE_FIXTURE = Path(__file__).parent / "fixtures" / "curriculum" / "schedule_trace.json"

# Scenarios: keep probabilities 1.0 and 0.5 (the presets' values) with training hints, and 0.5 without them.
# `window` is how many batches are admitted ahead of commits (3 with two steps of policy staleness). The "skewed"
# outcomes leave few questions with mixed unaided results, so practice concentrates on them until the per-epoch visit
# cap binds.
SCENARIOS = {
    "keep-1.0-hints": {"keep_probability": 1.0, "hints": True, "window": 3, "epochs": 3, "outcomes": "mixed"},
    "keep-0.5-hints": {"keep_probability": 0.5, "hints": True, "window": 3, "epochs": 3, "outcomes": "mixed"},
    "keep-0.5-no-hints": {"keep_probability": 0.5, "hints": False, "window": 3, "epochs": 3, "outcomes": "mixed"},
    "keep-0.5-hints-capped": {"keep_probability": 0.5, "hints": True, "window": 2, "epochs": 5, "outcomes": "skewed"},
}
TRACE_SETTINGS = {
    "pool_size": 23,        # not a multiple of the batch size, so each epoch ends with a short batch
    "batch_size": 4,
    "seed": 42,
    "group_size": 16,
    # Resume points (epoch, batch): after committing that batch, the next window is admitted on the running
    # schedule; a new schedule then loads the saved state from before those admissions and re-admits them, which
    # must restore them from their reservation files unchanged.
    "resumes": [[1, 2], [2, 3]],
}


def trace_row(i):
    evidence = "" if i % 4 == 3 else f"item {i} refers to n = {i}; recent refers to year > {2000 + i}"
    question = f"How many recent items match case {i}?"
    schema = f"CREATE TABLE t{i % 5}(n INTEGER, year INTEGER);"
    return {
        "id": f"t{i:02d}",
        "question": question,
        "evidence": evidence,
        "problem": compose_problem(schema, evidence=evidence, question=question),
        "gold_sql": f"SELECT count(*) FROM t{i % 5} WHERE n = {i} AND year > {2000 + i}",
        "database_sha256": "db",
    }


def trace_hints(rows):
    return {row["id"]: f"Filter t{i % 5} on n = {i} first." for i, row in enumerate(rows) if i % 3 != 2}


def scripted_correct(qid, epoch, level, index, outcomes="mixed"):
    """Correct rollouts of a stage.

    "mixed": none for 40% of stages, all for 20%, otherwise between. "skewed": the same for assisted stages; unaided
    stages of every eighth question get half right, and other unaided stages all or none (alternating by epoch).
    """
    size = TRACE_SETTINGS["group_size"]
    if outcomes == "skewed" and level == 0:
        i = int(qid[1:])
        return size // 2 if i % 8 == 0 else 0 if (i + epoch) % 2 else size
    h = int(hashlib.sha256(f"{qid}|{epoch}|{level}|{index}".encode()).hexdigest()[:8], 16)
    if h % 10 < 4:
        return 0
    if h % 10 < 6:
        return size
    return 1 + (h // 10) % (size - 1)


def scripted_group(index, row, correct, policy_step):
    trajectories = []
    for i in range(TRACE_SETTINGS["group_size"]):
        traj = trajectory(text="A" * (1 + (i + index) % 3), step=policy_step)
        traj.metrics["correct"] = int(i < correct)
        trajectories.append(traj)
    return CompletedGroup(f"group-{index}", row, trajectories)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class Implementation:
    """The curriculum entry points a scenario drives, and views that read its output in this module's format."""

    make_schedule: Callable[..., Any]
    stage_row: Callable[..., dict]
    observation: Callable[[CompletedGroup], dict]
    engine_class: type
    key: str = curriculum.KEY
    hint_level: int = curriculum.HINT

    def row_view(self, row):
        return copy.deepcopy(row)

    def state_view(self, state):
        return copy.deepcopy(state)

    def metrics_view(self, metrics):
        return dict(metrics)

    def receipt_view(self, receipt):
        return copy.deepcopy(receipt)


def package_implementation(**settings):
    """This package, passing `settings` (practice options) to every schedule it builds."""
    return Implementation(
        make_schedule=functools.partial(curriculum.EvidenceMixtureSchedule, **settings),
        stage_row=curriculum.stage_row,
        observation=curriculum.observation,
        engine_class=curriculum.AssistanceEngine,
    )


def run_stages(impl, events, epoch, index, row, step, hints, outcomes):
    """Run a reserved row's stages the way the rescue engine does and return the group that trains."""
    while True:
        stage = row[impl.key]
        correct = scripted_correct(str(row["id"]), epoch, stage["level"], index, outcomes)
        group = scripted_group(index, row, correct, step + stage["level"])
        current = impl.observation(group)
        hint = hints.get(str(row["id"]))
        next_level = None
        if current["correct"] == 0:
            if current["level"] == 0 and stage["original"]["evidence"].strip():
                next_level = 1
            elif current["level"] < impl.hint_level and hint:
                next_level = impl.hint_level
        if next_level is None:
            return group
        row = impl.stage_row(row, next_level, hint=hint, history=[*stage["history"], current])
        events.append(["stage", epoch, index, impl.row_view(row)])


def read_receipts(impl, directory):
    return {path.stem: impl.receipt_view(json.loads(path.read_text())) for path in sorted(directory.glob("*.json"))}


def run_scenario(impl, name, directory):
    """Drive `impl` through the scenario's epochs and return every event in this module's format.

    Admission runs up to `window` batches ahead of the oldest uncommitted batch; each batch commits the groups that
    complete first, in an order drawn from a generator independent of the implementation. Each epoch's final saved
    state, after a JSON round trip as in the trainer's checkpoint, seeds the next epoch.
    """
    settings, scenario = TRACE_SETTINGS, SCENARIOS[name]
    rows = [trace_row(i) for i in range(settings["pool_size"])]
    hints = trace_hints(rows) if scenario["hints"] else {}
    size, batch_size, window = len(rows), settings["batch_size"], scenario["window"]
    resumes = {tuple(point) for point in settings["resumes"]}
    events = []
    previous = None
    for epoch in range(1, scenario["epochs"] + 1):
        receipts = Path(directory) / f"{epoch:03d}"

        def build(epoch=epoch, previous=previous, receipts=receipts):
            return impl.make_schedule(
                rows,
                epoch=epoch,
                batch_size=batch_size,
                seed=settings["seed"],
                hints=hints,
                previous=previous,
                keep_probability=scenario["keep_probability"],
                reservations_dir=receipts,
            )

        schedule = build()
        completion = random.Random(f"completion:{name}:{epoch}")
        admitted = {}  # reservation index -> (row, committed batches at admission)
        next_index, batch = 0, 0
        while batch * batch_size < size:
            while next_index < min(size, (batch + window) * batch_size):
                row = schedule.row(next_index, rows)
                events.append(["row", epoch, next_index, impl.row_view(row)])
                admitted[next_index] = (row, batch)
                next_index += 1
            chosen = completion.sample(sorted(admitted), min(batch_size, size - batch * batch_size))
            groups = [
                run_stages(impl, events, epoch, index, *admitted.pop(index), hints, scenario["outcomes"])
                for index in chosen
            ]
            batch += 1
            metrics = schedule.commit(batch, groups)
            events.append(["commit", epoch, batch, chosen, impl.metrics_view(metrics)])
            saved = json.loads(json.dumps(schedule.state_dict()))
            events.append(["state", epoch, batch, impl.state_view(saved)])
            if (epoch, batch) in resumes:
                ahead = {}
                while next_index < min(size, (batch + window) * batch_size):
                    ahead[next_index] = schedule.row(next_index, rows)
                    events.append(["row", epoch, next_index, impl.row_view(ahead[next_index])])
                    next_index += 1
                schedule = build()
                schedule.load_state_dict(saved)
                for index, row in ahead.items():
                    again = schedule.row(index, rows)
                    events.append(["readmit", epoch, index, impl.row_view(again), again == row])
                    admitted[index] = (again, batch)
        events.append(["receipts", epoch, read_receipts(impl, receipts)])
        previous = json.loads(json.dumps(schedule.state_dict()))
    return events


def compact(events):
    """The golden-trace form of a scenario's events: choices in full, rows and states as digests."""
    out = []
    for event in events:
        kind, epoch = event[0], event[1]
        if kind in ("row", "readmit"):
            row = event[3]
            stage = row[curriculum.KEY]
            choice = stage["mixture"]
            item = [kind, epoch, event[2], choice["qid"], choice["lane"], choice["requested_lane"], stage["level"],
                    canonical_digest(row)]
            out.append(item + ([event[4]] if kind == "readmit" else []))
        elif kind == "stage":
            out.append([kind, epoch, event[2], event[3][curriculum.KEY]["level"], canonical_digest(event[3])])
        elif kind == "commit":
            out.append(event)
        elif kind == "state":
            out.append([kind, epoch, event[2], canonical_digest(event[3])])
        elif kind == "receipts":
            out.append([kind, epoch, len(event[2]), canonical_digest(event[2])])
    return out


def scenario_traces(impl):
    traces = {}
    for name in SCENARIOS:
        with tempfile.TemporaryDirectory() as directory:
            traces[name] = run_scenario(impl, name, directory)
    return traces


def format_trace(traces):
    """One event per line, so a change shows up as a readable diff."""
    lines = [
        "{",
        '  "about": "Golden trace of the evidence-mixture scheduler: see ScheduleTraceTest in test_curriculum.py.",',
        f'  "settings": {json.dumps(TRACE_SETTINGS, sort_keys=True)},',
        f'  "scenarios_settings": {json.dumps(SCENARIOS, sort_keys=True)},',
        '  "scenarios": {',
    ]
    names = list(traces)
    for n, name in enumerate(names):
        events = compact(traces[name])
        lines.append(f"    {json.dumps(name)}: [")
        lines += [
            f"      {json.dumps(e, sort_keys=True)}{',' if i + 1 < len(events) else ''}" for i, e in enumerate(events)
        ]
        lines.append("    ]" + ("," if n + 1 < len(names) else ""))
    lines += ["  }", "}"]
    return "\n".join(lines) + "\n"


class ScheduleTraceTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(TRACE_FIXTURE.read_text())
        cls.traces = scenario_traces(package_implementation())

    def test_fixture_matches_the_scenarios(self):
        self.assertEqual(self.fixture["settings"], TRACE_SETTINGS)
        self.assertEqual(self.fixture["scenarios_settings"], SCENARIOS)
        self.assertEqual(sorted(self.fixture["scenarios"]), sorted(SCENARIOS))

    def test_draws_stages_commits_and_states_match_the_golden_trace(self):
        for name, events in self.traces.items():
            expected = self.fixture["scenarios"][name]
            actual = json.loads(json.dumps(compact(events)))
            with self.subTest(scenario=name):
                for i, (want, got) in enumerate(zip(expected, actual)):
                    self.assertEqual(got, want, f"first difference at event {i}")
                self.assertEqual(len(actual), len(expected))

    def test_scenarios_exercise_every_path(self):
        for name, events in self.traces.items():
            with self.subTest(scenario=name):
                kinds = {e[0] for e in events}
                self.assertEqual(kinds, {"row", "stage", "commit", "state", "readmit", "receipts"})
                self.assertTrue(all(e[4] for e in events if e[0] == "readmit"))
                lanes = {(e[3][curriculum.KEY]["mixture"]["lane"], e[3][curriculum.KEY]["mixture"]["requested_lane"])
                         for e in events if e[0] == "row"}
                stages = {e[3][curriculum.KEY]["level"] for e in events if e[0] == "stage"}
                keep, hints = SCENARIOS[name]["keep_probability"], SCENARIOS[name]["hints"]
                self.assertEqual({lane for lane, _ in lanes}, {"none", "trusted"})
                if keep == 1:
                    # Only questions without evidence start unaided; their trusted draws fall short to "none", and
                    # their rescue goes straight to the hint.
                    self.assertEqual({requested for _, requested in lanes}, {"trusted"})
                    self.assertIn(("none", "trusted"), lanes)
                    self.assertEqual(stages, {2})
                else:
                    self.assertIn(("none", "none"), lanes)
                    self.assertEqual(stages, {1, 2} if hints else {1})
                epochs = SCENARIOS[name]["epochs"]
                final = [e for e in events if e[0] == "state" and e[1] == epochs][-1][3]
                self.assertTrue(any(q["visits"] > epochs for q in final["questions"].values()))
                reservations = Counter((e[1], e[3][curriculum.KEY]["mixture"]["qid"]) for e in events if e[0] == "row")
                self.assertLessEqual(max(reservations.values()), curriculum.MAX_EPOCH_VISITS)
                if name.endswith("-capped"):
                    self.assertEqual(max(reservations.values()), curriculum.MAX_EPOCH_VISITS)

    def test_practice_options_passed_at_their_defaults_draw_the_golden_trace(self):
        # The trainer passes every option explicitly; at the defaults nothing may change.
        defaults = package_implementation(weighting="unaided", failed_weight=0.02, solved_weight=0.02,
                                          fade_probability=0.0, include_empty_evidence=False)
        for name, events in scenario_traces(defaults).items():
            with self.subTest(scenario=name):
                self.assertEqual(json.loads(json.dumps(compact(events))), self.fixture["scenarios"][name])

    def test_practice_options_survive_admission_ahead_resumes_and_rescues(self):
        fade = 0.5
        options = package_implementation(weighting="lane", failed_weight=0.05, solved_weight=0.0625,
                                         fade_probability=fade, include_empty_evidence=True)
        totals = Counter()
        for name, events in scenario_traces(options).items():
            with self.subTest(scenario=name):
                self.assertNotEqual(json.loads(json.dumps(compact(events))), self.fixture["scenarios"][name])
                # Re-admissions after each resume restore their reservation, faded or not, from its file.
                self.assertTrue(all(e[4] for e in events if e[0] == "readmit"))
                state, choices = None, []
                for event in events:
                    if event[0] == "state":
                        state = event[3]
                    elif event[0] == "row":
                        choice = event[3][curriculum.KEY]["mixture"]
                        choices.append(choice)
                        # Only a practice draw that picked a question with evidence for the evidence lane, when its
                        # last evidence result as of the draw was all correct, flips the coin, and the coin alone
                        # decides; the coin is the draw's own (seed, epoch, reservation index, question).
                        epoch, index, qid = event[1], event[2], choice["qid"]
                        eligible = bool(
                            epoch > 1
                            and choice["requested_lane"] == "trusted"
                            and trace_row(int(qid[1:]))["evidence"]
                            and state["questions"][qid]["levels"][1]["last_p"] == 1
                        )
                        coin = random.Random(f"fade:{TRACE_SETTINGS['seed']}:{epoch}:{index}:{qid}").random()
                        self.assertEqual(choice["faded"], eligible and coin < fade)
                        if choice["faded"]:
                            self.assertEqual((choice["requested_lane"], choice["lane"]), ("trusted", "none"))
                        totals["eligible"] += eligible
                commits = [e[4] for e in events if e[0] == "commit"]
                self.assertEqual(sum(m["mixture/faded"] for m in commits), sum(c["faded"] for c in choices))
                totals["faded"] += sum(m["mixture/faded"] for m in commits)
                totals["rescued"] += sum(m["mixture/faded/rescued"] for m in commits)
                self.assertEqual(
                    sum(m["mixture/lane_shortfall"] for m in commits),
                    sum(c["lane"] != c["requested_lane"] and not c["faded"] for c in choices),
                )
                if SCENARIOS[name]["keep_probability"] == 1:
                    # Questions with empty evidence are drawn again after their first pass, unaided and never faded.
                    later = [e[3][curriculum.KEY]["mixture"] for e in events if e[0] == "row" and e[1] > 1]
                    bare = [c for c in later if not trace_row(int(c["qid"][1:]))["evidence"]]
                    self.assertTrue(bare)
                    self.assertTrue(all((c["lane"], c["faded"]) == ("none", False) for c in bare))
        # The scenarios exercise the fade both ways, and its rescue.
        self.assertGreater(totals["faded"], 0)
        self.assertGreater(totals["eligible"], totals["faded"])
        self.assertGreater(totals["rescued"], 0)


if __name__ == "__main__":
    unittest.main()
