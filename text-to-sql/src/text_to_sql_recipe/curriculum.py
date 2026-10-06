"""Training curriculum: which question each rollout group trains on, how much help its prompt shows, and the rescue
of questions on which every rollout fails.

Levels (`LEVELS`). `stage_row` stages a training row at one of three levels: "none" withholds the question's
evidence (the external-knowledge section then reads "(none)"), "evidence" shows it, and "hint" shows it and appends a
training hint after `prompts.TRAINING_HINT_PREFIX`. Evaluation never stages rows.

Rescue (`AssistanceEngine`). When every rollout of a group is wrong, the whole group is re-run one level up: with the
evidence if it was withheld and the question has any, otherwise with the training hint if there is one. Only the last
stage trains. The failed stages never enter a batch; their observations (`observation`) travel on the re-staged row
and reach the scheduler when the trainer commits the group.

Scheduling (`EvidenceMixtureSchedule`). Every admitted group reserves a question and a lane before it runs: "trusted"
(start with the evidence shown) with probability `keep_probability`, otherwise "none". The first epoch covers every
question once, assigning lanes so that min(round(n * keep_probability), questions with evidence) of the n questions
start with their evidence. Later epochs draw each question with probability proportional to its weight
(`EvidenceMixtureSchedule.draw_weight`, from `AssistanceSchedule.weight`), at most `MAX_EPOCH_VISITS` times per
question per epoch, and never a question whose previous reservation has not been committed yet. The weight reads the
question's unaided record ("unaided" weighting) or the record of the lane the draw will use ("lane" weighting,
which the recipe uses). A trusted draw that finds no eligible question with evidence falls back to "none" and is counted as a
shortfall; optionally, a trusted draw may also pick a question that has no evidence, which then runs unaided. With a
fade probability, once a trusted draw has picked a question whose last result with its evidence was all correct, a
coin withholds the evidence with that probability: the draw is "faded" into the "none" lane, and a full fail is
rescued as usual. The coin comes after the pick, so the fade never changes which question is drawn. Every draw comes
from generators seeded by the seed, the epoch and the reservation index (and, for a fade, the question), so a run's
choices are reproducible.

Durability. Reservations live in the scheduler state, which the trainer saves with each checkpoint and restores on
resume. Each reservation, including its lane and whether it was faded, is also written to `reservations_dir` before
its group is admitted, so groups admitted after the last checkpoint are re-admitted unchanged. Outcomes change the
state only when the trainer commits a batch. `EvidenceMixtureSchedule` documents the saved state.

Hints enter only through `load_hints`, which checks every hint against the training row it was written for.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import itertools
import json
import math
import os
import random
import tempfile
from collections import Counter
from pathlib import Path

from river_client import rl

from .prompts import TRAINING_HINT_PREFIX, compose_problem

KEY = "assistance_curriculum"           # where a staged row records its stage
LEVELS = ("none", "evidence", "hint")
HINT = 2
MAX_EPOCH_VISITS = 4                    # reservations of one question per epoch
VERSION = "evidence-mixture-v1"         # the scheduling policy, recorded in the contract and in every reservation
LANES = ("none", "trusted")
WEIGHTINGS = ("unaided", "lane")        # which record a practice draw is weighted by (`draw_weight`)
WEIGHT_FLOOR = 0.02                     # the smallest practice weight; also the default weight of a 0/16 record


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def canonical_parts(row):
    """Return the schema part of a row's task text, failing closed on a different prompt layout.

    The curriculum re-composes the task text from the schema, the evidence it shows and the question, so it accepts
    only a row whose text is exactly `compose_problem(schema, evidence=row["evidence"], question=row["question"])`.
    """
    suffix = compose_problem("", evidence=row["evidence"], question=row["question"]).split("Database Schema:\n", 1)[1]
    prefix = "Database Engine:\nSQLite\n\nDatabase Schema:\n"
    problem = row["problem"]
    if not problem.startswith(prefix) or not problem.endswith(suffix):
        raise ValueError(f"{row['id']}: unsupported task prompt codec")
    schema = problem[len(prefix) : -len(suffix)]
    if compose_problem(schema, evidence=row["evidence"], question=row["question"]) != problem:
        raise ValueError(f"{row['id']}: task fields disagree with prompt")
    return schema


def load_hints(path, sha256, binding_path, binding_sha256, rows):
    """Load the training hint of every training row, after checking each against the row it was written for.

    `path` holds {"references": {task id: {"raw": hint text}}} and `binding_path` holds {task id: {"current_role",
    "prompt_sha256", "gold_sql_sha256", "database_sha256"}}; both files must match their pinned sha256. Every row must
    be a training row ("paper_train") with a binding to its current prompt, gold query and database, and a nonempty
    hint. Returns {task id: hint text}.
    """

    def read(filename, expected):
        raw = Path(filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"training hint file hash mismatch: {filename}")
        return json.loads(raw)

    references = read(path, sha256)["references"]
    bindings = read(binding_path, binding_sha256)
    hints = {}
    for row in rows:
        qid = str(row["id"])
        binding = bindings.get(qid)
        if binding is None:
            raise ValueError(f"{qid}: no training hint binding in {binding_path}")
        if binding["current_role"] != "paper_train" or row.get("current_role", "paper_train") != "paper_train":
            raise ValueError("hints require training-only bindings")
        values = {
            "prompt_sha256": hashlib.sha256(row["problem"].encode()).hexdigest(),
            "gold_sql_sha256": hashlib.sha256(row["gold_sql"].encode()).hexdigest(),
            "database_sha256": row["database_sha256"],
        }
        if any(binding[k] != value for k, value in values.items()):
            raise ValueError(f"{qid}: stale training hint binding")
        if qid not in references:
            raise ValueError(f"{qid}: no training hint in {path}")
        text = references[qid]["raw"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{qid}: empty training hint")
        hints[qid] = text
    return hints


def stage_row(row, level, *, hint=None, history=()):
    """Return a copy of a training row staged at `level` (0 none, 1 evidence, 2 hint).

    The stage is recorded under `KEY`: the level, the original schema and evidence, and `history`, the observations
    of the failed stages before this one. Re-staging an already staged row starts again from the recorded original
    and keeps the scheduler's reservation ("mixture").
    """
    result = copy.deepcopy(row)
    if KEY in result:
        original = result[KEY]["original"]
    else:
        original = {"schema": canonical_parts(row), "evidence": row["evidence"]}
    if level not in range(3) or (level == HINT and not hint):
        raise ValueError("invalid assistance stage or missing hint")
    result["evidence"] = original["evidence"] if level else ""
    result["problem"] = compose_problem(original["schema"], evidence=result["evidence"], question=row["question"])
    if level == HINT:
        result["problem"] += TRAINING_HINT_PREFIX + hint
    result[KEY] = {
        "level": level,
        "original": original,
        "history": copy.deepcopy(list(history)),
    }
    if KEY in row and "mixture" in row[KEY]:
        result[KEY]["mixture"] = copy.deepcopy(row[KEY]["mixture"])
    return result


def observation(group):
    """Summarise a complete group: its level, correct rollouts out of total, generated tokens and policy steps.

    Only complete groups scored with binary correctness (`metrics["correct"]` of 0 or 1 on every rollout) are
    accepted; anything else fails closed.
    """
    values = [t.metrics.get("correct") for t in group.trajectories]
    if not group.final or group.closed_by_deadline or not values or any(v not in (0, 1) for v in values):
        raise ValueError("assistance requires complete, scored binary-correctness groups")
    return {
        "level": group.row[KEY]["level"],
        "correct": int(sum(values)),
        "total": len(values),
        "tokens": sum(t.generated_tokens for t in group.trajectories),
        "policy_steps": sorted({s.policy_step for t in group.trajectories for s in t.spans if s.kind == "generated"}),
    }


class AssistanceEngine(rl.RolloutEngine):
    """A rollout engine that rescues one question without blocking unrelated completed groups.

    `hints` maps task ids to training hints (see `load_hints`); without one a question is rescued with its evidence
    only. Failed stages are replaced synchronously between awaits. The replacement row contains their observations,
    so engine snapshots also persist the chain. Each stage starts fresh and retains ordinary trajectory policy
    pinning and trainer staleness reservations; different stages may use newer snapshots.
    """

    def __init__(self, *args, hints=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hints = hints or {}
        self.stage_lock = asyncio.Lock()

    async def next_group(self):
        while True:
            group = await super().next_group()
            current = observation(group)
            state = group.row[KEY]
            hint = self.hints.get(str(group.row["id"]))
            next_level = None
            if current["correct"] == 0:
                if current["level"] == 0 and state["original"]["evidence"].strip():
                    next_level = 1
                elif current["level"] < HINT and hint:
                    next_level = HINT
            if next_level is None:
                return group
            try:
                await self.stage_lock.acquire()
            except asyncio.CancelledError:
                # The trainer can cancel a receive when its checkpoint completes.
                # Preserve this dequeued result for the next receiver.
                self._completed.put_nowait(group)
                raise
            try:
                row = stage_row(group.row, next_level, hint=hint, history=[*state["history"], current])
                # No await: a checkpoint sees either the old complete stage or the
                # new group with its history, never a lost intermediate result.
                self.accept(group)
                self.resubmit(group.id, row)
            finally:
                self.stage_lock.release()

    async def state_dict(self):
        async with self.stage_lock:
            return await super().state_dict()


class AssistanceSchedule:
    """Per-question state across epochs: pass rates per level, visit counts, and the contract a saved state is valid
    for, with the practice weighting and resume.

    `EvidenceMixtureSchedule` draws the rows and commits the outcomes. `previous` is the saved state of the
    preceding epoch: its contract must equal this one's apart from the epoch, and all of its reservations must have
    been committed.
    """

    def __init__(self, rows, *, epoch, batch_size, seed=42, hints=None, previous=None):
        self.ids = [str(r["id"]) for r in rows]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate assistance task IDs")
        if epoch < 1 or batch_size < 1:
            raise ValueError("invalid assistance schedule")
        self.epoch, self.batch_size, self.seed = epoch, batch_size, seed
        self.hints = hints or {}
        for row in rows:
            canonical_parts(row)
        self.questions = {
            qid: {
                "levels": [{"p": None, "last_p": None, "visits": 0, "failures": 0} for level in LEVELS],
                "visits": 0,
            }
            for qid in self.ids
        }
        if previous is not None:
            if previous["contract"] != {**self.contract(), "epoch": epoch - 1}:
                raise ValueError("previous assistance epoch/contract mismatch")
            if len(previous["completed"]) != len(previous["reservations"]):
                raise ValueError("previous epoch has uncommitted assistance reservations")
            self.questions = copy.deepcopy(previous["questions"])
            self._check_questions()
        self.reservations, self.completed, self.history = {}, set(), []

    def contract(self):
        return {
            "version": VERSION,
            "epoch": self.epoch,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "ids": self.ids,
            "hints_sha256": hashlib.sha256(json.dumps(self.hints, sort_keys=True).encode()).hexdigest(),
        }

    def _check_questions(self):
        if set(self.questions) != set(self.ids):
            raise ValueError("assistance pool changed")

    @staticmethod
    def weight(question, level=0, failed_weight=WEIGHT_FLOOR, solved_weight=WEIGHT_FLOOR):
        """Practice weight from the question's record at one level, by default the unaided one (level 0).

        p(1 - p) of that level's pass rate p (each observation is averaged with the previous estimate; 0.25, the top
        weight, before any), floored at `WEIGHT_FLOOR`. A question whose last group at that level was all wrong gets
        `failed_weight`, and one whose last group was all right gets `solved_weight` (both the floor by default).
        """
        item = question["levels"][level]
        if item["last_p"] == 0:
            return failed_weight
        if item["last_p"] == 1:
            return solved_weight
        p = item["p"]
        return max((0.5 if p is None else p) * (0.5 if p is None else 1 - p), WEIGHT_FLOOR)

    def state_dict(self):
        return copy.deepcopy(
            {
                "contract": self.contract(),
                "questions": self.questions,
                "reservations": self.reservations,
                "completed": sorted(self.completed),
                "history": self.history,
            }
        )

    def load_state_dict(self, state):
        if state["contract"] != self.contract():
            raise ValueError("assistance scheduling contract changed")
        self.questions = copy.deepcopy(state["questions"])
        self._check_questions()
        self.reservations = copy.deepcopy(state["reservations"])
        self.completed, self.history = set(state["completed"]), copy.deepcopy(state["history"])


class EvidenceMixtureSchedule(AssistanceSchedule):
    """Choose a lane before the question; preserve whole-group prompts on restart.

    The River trainer drives it as its `dataset_scheduler`: `row(index, rows)` reserves and stages the row of group
    `index`, `commit(batch, groups)` records a trained batch and returns its metrics, and `contract`, `state_dict`
    and `load_state_dict` make it part of the trainer's checkpoint. Coverage and the per-epoch visit bound take
    precedence over the keep probability. A lane with no eligible question falls back to "none" and counts as a
    shortfall; no question is repeated to force the requested ratio.

    Practice settings. The recipe passes its own values from `config.CurriculumConfig`, which differ from the
    constructor's defaults below:

        weighting         "unaided" (default) weights every practice draw by the question's unaided record, so
                          evidence practice cannot erase unaided difficulty. "lane" weights it by the record of the
                          lane the draw uses: the evidence record (level 1) for an evidence draw, the unaided record
                          otherwise, never the other lane's. A question never tried in that lane gets the top weight.
                          See `draw_weight`.
        failed_weight     the weight of a question whose last result in that record was all wrong (default
                          `WEIGHT_FLOOR`).
        solved_weight     the weight of a question whose last result in that record was all right (default
                          `WEIGHT_FLOOR`). Raising it draws solved questions more often; with a fade, part of those
                          draws withhold the evidence.
        fade_probability  0 (default) or the chance that a trusted practice draw, once it has picked a question whose
                          last evidence result (level 1) was all correct, withholds the evidence: the draw is "faded"
                          into the "none" lane, and a full fail there is rescued with the evidence, then the hint, as
                          usual. Questions not yet mastered with their evidence always keep it. The coin is per draw
                          (seeded by the seed, the epoch, the reservation index and the question), so a mastered
                          question can run with its evidence on some draws and without it on others. It is flipped
                          after the pick with its own generator, so the fade changes neither the weights nor which
                          question is drawn. A faded draw's unaided result goes to the unaided record; under "lane"
                          weighting an evidence draw never reads that record, so with the evidence always kept, every
                          weight comes from results with the evidence shown.
        include_empty_evidence
                          False (default) or True: a trusted practice draw may also pick a question whose evidence is
                          empty. It runs unaided (its prompt is the same in either lane), is weighted by its unaided
                          record, and counts as a lane shortfall. Without it, a schedule that always keeps the
                          evidence never draws such a question again after its first pass.

    Saved state (`state_dict`; the trainer stores it as "dataset_scheduler" in each checkpoint, and the next epoch's
    schedule takes the final one as `previous`):

        contract      the settings the state belongs to: "version" (`VERSION`), "epoch", "batch_size", "seed", "ids"
                      (task ids in pool order), "hints_sha256", "keep_probability" and "prompts_sha256" (every task
                      text); and "weighting", "failed_weight", "solved_weight", "fade_probability" and
                      "include_empty_evidence", each only when it differs from its default. A resume or a next epoch
                      under different settings is refused.
        questions     per task id: "visits" (committed groups); "levels", one entry per level in `LEVELS` with "p"
                      (pass rate, each observation averaged with the previous value), "last_p", "visits" and
                      "failures" (consecutive all-wrong observations); and, from its first commit, "conditions" (per
                      level name: summed "correct" and "total", "visits", "last_batch").
        reservations  per group index (a string): "qid", "level" (0 or 1), "lane", "requested_lane", "evidence" (as
                      shown), "evidence_sha256", "visit" ("epoch:index") and "policy" (`VERSION`); with a fade
                      probability, also "faded" (a faded draw has lane "none" and requested lane "trusted").
        completed     the committed group indices.
        history       one entry per committed group in commit order: "batch", "reservation", the reservation's
                      fields, and "observations" (every stage's level, correct, total, tokens and policy_steps).

    Each file in `reservations_dir` holds one reservation: {"contract": ..., "choice": ...}, named "<index>.json".
    """

    def __init__(
        self,
        rows,
        *,
        keep_probability=0.5,
        weighting="unaided",
        failed_weight=WEIGHT_FLOOR,
        solved_weight=WEIGHT_FLOOR,
        fade_probability=0.0,
        include_empty_evidence=False,
        reservations_dir=None,
        **kwargs,
    ):
        self.reservations_dir = Path(reservations_dir) if reservations_dir is not None else None
        self.keep_probability = keep_probability
        if not 0 <= keep_probability <= 1 or not 0 <= fade_probability <= 1:
            raise ValueError("invalid evidence mixture probabilities")
        if weighting not in WEIGHTINGS or not 0 < failed_weight <= 1 or not 0 < solved_weight <= 1:
            raise ValueError("invalid evidence mixture weighting")
        if not isinstance(include_empty_evidence, bool):
            raise TypeError("invalid evidence mixture candidates")
        self.weighting, self.failed_weight, self.solved_weight = weighting, failed_weight, solved_weight
        self.fade_probability, self.include_empty_evidence = fade_probability, include_empty_evidence
        # Pin every task text, so a changed prompt cannot reuse reservations made for the old one.
        self.prompts_sha256 = digest(json.dumps({str(r["id"]): r["problem"] for r in rows}, sort_keys=True))
        super().__init__(rows, **kwargs)
        self.coverage_lanes, self.coverage_order = self._coverage_plan(rows)

    def contract(self):
        contract = {
            **super().contract(),
            "keep_probability": self.keep_probability,
            "prompts_sha256": self.prompts_sha256,
        }
        # A practice setting enters the contract only when it differs from the constructor's default, which keeps a
        # default schedule's contract short; any change of setting still refuses a mismatched resume.
        for name, default in (("weighting", "unaided"), ("failed_weight", WEIGHT_FLOOR), ("solved_weight", WEIGHT_FLOOR),
                              ("fade_probability", 0.0), ("include_empty_evidence", False)):
            if getattr(self, name) != default:
                contract[name] = getattr(self, name)
        return contract

    def row(self, index, rows):
        key = str(index)
        by_id = {str(r["id"]): r for r in rows}
        receipt = self.reservations_dir / f"{index}.json" if self.reservations_dir else None
        self._restore_choice(key, receipt)
        if key not in self.reservations:
            rng = random.Random(f"mixture:{self.seed}:{self.epoch}:{index}")
            requested = "none"
            if rng.random() < self.keep_probability:
                # Unused, but part of the seeded sequence: removing it would shift the question draw below and change
                # every choice a given seed makes.
                rng.random()
                requested = "trusted"
            reserved = {v["qid"] for k, v in self.reservations.items() if k not in self.completed}
            counts = Counter(v["qid"] for v in self.reservations.values())
            unseen = [q for q in self.ids if not self.questions[q]["visits"] and not counts[q]]
            eligible = unseen or [q for q in self.ids if q not in reserved and counts[q] < MAX_EPOCH_VISITS]
            if not eligible:
                raise ValueError("assistance reservation capacity exhausted; reduce admission concurrency")
            if unseen:
                requested = self.coverage_order[index % len(self.coverage_order)]
                candidates = [q for q in unseen if self.coverage_lanes[q] == requested]
                # Handles resumed/partially covered pools without discarding coverage.
                lane = requested if candidates else self.coverage_lanes[unseen[0]]
                candidates = [q for q in unseen if self.coverage_lanes[q] == lane]
                uses = dict.fromkeys(candidates, lane)
            else:
                lane, candidates = self._practice_candidates(eligible, requested, by_id)
                # A trusted draw of a question with empty evidence runs unaided: its prompt is the same either way.
                uses = {q: "none" if lane == "trusted" and not by_id[q]["evidence"].strip() else lane
                        for q in candidates}
            # Each candidate is weighted by the record of the lane its draw uses.
            qid = max(
                candidates,
                key=lambda q: math.log(max(rng.random(), 1e-15))
                / (1 if unseen else self.draw_weight(self.questions[q], uses[q])),
            )
            lane = uses[qid]
            # The fade comes after the pick, so it never changes which question is drawn.
            faded = not unseen and self._fades(qid, lane, index)
            if faded:
                lane = "none"
            evidence = by_id[qid]["evidence"] if lane == "trusted" else ""
            choice = {
                "qid": qid,
                "level": 0 if lane == "none" else 1,
                "lane": lane,
                "requested_lane": requested,
                "evidence": evidence,
                "evidence_sha256": digest(evidence),
                "visit": f"{self.epoch}:{index}",
                "policy": VERSION,
            }
            if self.fade_probability:
                choice["faded"] = faded
            self.reservations[key] = choice
        choice = self.reservations[key]
        if receipt is not None and not receipt.exists():
            self._persist_choice(receipt, choice)
        row = stage_row(by_id[choice["qid"]], choice["level"])
        if digest(row["evidence"]) != choice["evidence_sha256"]:
            raise ValueError("reserved evidence changed")
        row[KEY]["mixture"] = copy.deepcopy(choice)
        return row

    def _coverage_plan(self, rows):
        """Assign each question its first-pass lane, and shuffle the order in which lanes are requested."""
        rng = random.Random(f"coverage:{self.seed}:{self.epoch}")
        evidenced = sorted(str(row["id"]) for row in rows if row["evidence"].strip())
        requested_keep = round(len(rows) * self.keep_probability)
        keep = min(requested_keep, len(evidenced))
        trusted = list(evidenced)
        rng.shuffle(trusted)
        trusted = set(trusted[:keep])
        lanes = {qid: "trusted" if qid in trusted else "none" for qid in self.ids}
        order = ["trusted"] * requested_keep + ["none"] * (len(rows) - requested_keep)
        rng.shuffle(order)
        return lanes, order

    def _practice_candidates(self, eligible, requested, rows):
        lane = requested
        candidates = self._candidates(eligible, lane, rows)
        if not candidates:
            lane, candidates = "none", eligible
        return lane, candidates

    def draw_weight(self, question, lane):
        """Practice weight of drawing `question` into `lane`, the lane the draw uses before any fade.

        "unaided" weighting reads the unaided record (level 0) whatever the lane; "lane" weighting reads the evidence
        record (level 1) for a "trusted" draw and the unaided record for a "none" draw, with no fallback to the other
        lane's record. Either way the value is `AssistanceSchedule.weight` of that record with `failed_weight` and
        `solved_weight`. A fade comes after the draw, so a mastered question weighs the same whether or not this draw
        then withholds its evidence.
        """
        level = 1 if self.weighting == "lane" and lane == "trusted" else 0
        return self.weight(question, level, self.failed_weight, self.solved_weight)

    def _fades(self, qid, lane, index):
        """Whether the practice draw of `qid` into `lane` at reservation `index` withholds the question's evidence.

        Only a trusted draw of a question whose last evidence result (level 1) was all correct can fade. It does when
        the draw's own coin, seeded by the seed, the epoch, the index and the question, falls below `fade_probability`.
        The coin has its own generator, so the reservation's generator (the lane coin and the question keys) draws the
        same numbers with or without the fade.
        """
        if lane != "trusted" or not self.fade_probability or self.questions[qid]["levels"][1]["last_p"] != 1:
            return False
        coin = random.Random(f"fade:{self.seed}:{self.epoch}:{index}:{qid}")
        return coin.random() < self.fade_probability

    def _restore_choice(self, key, receipt):
        if receipt is not None and receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["contract"] != self.contract():
                raise ValueError("durable evidence reservation contract changed")
            if key in self.reservations and self.reservations[key] != saved["choice"]:
                raise ValueError("durable evidence reservation differs from checkpoint")
            self.reservations[key] = saved["choice"]

    def _persist_choice(self, receipt, choice):
        """Publish before engine admission, including work ahead of the trainer's checkpoint."""
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=receipt.parent, delete=False) as stream:
            json.dump({"contract": self.contract(), "choice": choice}, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        Path(temporary).replace(receipt)
        directory = os.open(receipt.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _candidates(self, eligible, lane, rows):
        if lane == "trusted" and not self.include_empty_evidence:
            return [q for q in eligible if rows[q]["evidence"].strip()]
        return eligible

    def commit(self, batch, groups):
        """Record a trained batch: every stage's outcome per question, then the reservation as completed.

        Validates every group against its reservation and rescue chain before changing anything. Returns metrics:
        "mixture/scheduled/<lane>" (groups per lane drawn; a faded draw counts as "none"), "mixture/terminal/<level>"
        (groups per level of the stage that trained), "mixture/<level>/correct|total|tokens" (summed over every stage
        at that level, failed ones included), "mixture/lane_shortfall" (groups not in their requested lane because no
        eligible question had evidence, or because the question drawn has empty evidence; faded draws are not
        shortfalls), and "assistance/extra_groups" and
        "assistance/extra_tokens" (the failed stages that were re-run). With a fade probability, also "mixture/faded"
        (faded groups), "mixture/faded/correct|total" (their unaided stage) and "mixture/faded/rescued" (faded groups
        that failed outright and were re-run with the evidence).
        """
        events = []
        seen = set()
        for group in groups:
            key = group.id.removeprefix("group-")
            choice = self.reservations[key]
            if (
                key in seen
                or key in self.completed
                or group.row[KEY].get("mixture") != choice
                or str(group.row["id"]) != choice["qid"]
            ):
                raise ValueError("mixture commit differs from reservation")
            seen.add(key)
            obs = [*group.row[KEY]["history"], observation(group)]
            conditions = [LEVELS[o["level"]] for o in obs]
            start = {"none": "none", "trusted": "evidence"}[choice["lane"]]
            allowed = {
                "none": {"evidence", "hint"},
                "evidence": {"hint"},
                "hint": set(),
            }
            if conditions[0] != start or any(
                b not in allowed[a] or obs[i]["correct"] != 0
                for i, (a, b) in enumerate(itertools.pairwise(conditions))
            ):
                raise ValueError("invalid mixture rescue history")
            events.append((key, choice, obs, conditions))
        for key, choice, observations, conditions in events:
            question = self.questions[choice["qid"]]
            stats = question.setdefault("conditions", {})
            for obs, condition in zip(observations, conditions, strict=True):
                item = stats.setdefault(
                    condition,
                    {"correct": 0, "total": 0, "visits": 0, "last_batch": None},
                )
                item["correct"] += obs["correct"]
                item["total"] += obs["total"]
                item["visits"] += 1
                item["last_batch"] = batch
                # Only observations at this condition update its own competence.
                level = question["levels"][obs["level"]]
                p = obs["correct"] / obs["total"]
                level["p"] = p if level["p"] is None else (level["p"] + p) / 2
                level["last_p"] = p
                level["visits"] += 1
                level["failures"] = level["failures"] + 1 if p == 0 else 0
            question["visits"] += 1
            self.completed.add(key)
            self.history.append(
                {
                    "batch": batch,
                    "reservation": key,
                    **choice,
                    "observations": observations,
                }
            )
        metrics = {
            **{f"mixture/scheduled/{lane}": float(sum(c["lane"] == lane for k, c, o, cs in events)) for lane in LANES},
            **{
                f"mixture/terminal/{condition}": float(sum(cs[-1] == condition for k, c, o, cs in events))
                for condition in LEVELS
            },
            **{
                f"mixture/{condition}/{field}": float(
                    sum(
                        obs[field]
                        for k, c, oo, cs in events
                        for obs, cc in zip(oo, cs, strict=True)
                        if cc == condition
                    )
                )
                for condition in LEVELS
                for field in ("correct", "total", "tokens")
            },
            "mixture/lane_shortfall": float(
                sum(c["lane"] != c["requested_lane"] and not c.get("faded") for k, c, o, cs in events)
            ),
            "assistance/extra_groups": float(sum(len(o) - 1 for k, c, o, cs in events)),
            "assistance/extra_tokens": float(sum(obs["tokens"] for k, c, o, cs in events for obs in o[:-1])),
        }
        if self.fade_probability:
            faded = [o for k, c, o, cs in events if c.get("faded")]
            metrics["mixture/faded"] = float(len(faded))
            metrics["mixture/faded/correct"] = float(sum(o[0]["correct"] for o in faded))
            metrics["mixture/faded/total"] = float(sum(o[0]["total"] for o in faded))
            metrics["mixture/faded/rescued"] = float(sum(len(o) > 1 for o in faded))
        return metrics
