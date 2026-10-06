"""Loop guards for sampled turns, batch guard metrics, and the collapse pause.

A policy can fall into a loop that runs a turn to its token cap: a SQL draft written out again verbatim (a
``Draft:`` marker followed by the same query, over and over), or plain repetition (a word, a few words or a sentence
repeated back to back). Sampling a loop to the cap teaches nothing, and thousands of looping tokens would dominate a
batch's summed loss. On a turn that hit its cap without calling a tool, the guards find where the loop starts, end
the rollout there and stop crediting the tokens after the first copy:

- a *verbatim re-draft* is a draft whose whitespace-collapsed body equals an earlier draft's body (`drafts`,
  `repeats`); the turn keeps everything up to and including the first repeated draft (`loop_cut`). Guarded by
  ``guards.loop_stop``;
- a *repetition* is a run of at least ``repetition_min_tokens`` generated tokens that repeats with a period of at most
  ``repetition_max_period`` tokens (`periodic_run`), or a 40-160 character span of the decoded text repeated at least
  ``repetition_span_repeats`` times back to back (`span_loop`); the turn keeps everything up to and including the first
  copy. Guarded by ``guards.repetition_guard``.

When both guards find a loop, the earlier cut wins (the re-draft on a tie). The answer is whatever preceded the cut,
normally nothing, so the rollout scores the no-answer reward. Detection runs on every turn that hit its cap and is
recorded in the trajectory's metrics whether or not a guard acts on it; turns that ended normally are only counted.

Per sampled turn the environment calls `scan_turn` once, then `apply_loop_guards`. A `LoopStop` result carries the
generated-token offset at which the loss mask starts (`LoopStop.loss_mask_end`): generated tokens at or past it keep
no advantage. Per training batch, `guard_metrics` summarises the rollouts, and `CollapsePause.check` stops training,
durably, once the cap-hit and repetition rates have both stayed above their limits for consecutive steps after the
pause is armed.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import GuardConfig
from .reward import TRUNCATED_STOP_REASON

# --- Verbatim re-drafts ------------------------------------------------------------------------------------------
#
# A draft is the first non-empty paragraph after a ``Draft:`` marker line (or the fenced code block that starts
# it). Two drafts are identical when their whitespace-collapsed text is equal. Commentary between drafts is not a
# draft, and a draft that *changes* is a real revision, never counted.

# A marker line: optional qualifier ("Revised", "Final", "New"), "Draft" with an
# optional number, then a colon, alone on its line.
_MARKER = re.compile(r"(?m)^[ \t]*(?:[A-Za-z]+[ \t]+)?[Dd]raft(?:[ \t]*\d+)?[ \t]*:[ \t]*$")
_FENCE = re.compile(r"^[ \t]*```")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class Draft:
    marker_start: int  # character offset of the marker line
    body_end: int  # character offset just past the draft's body
    normalized: str  # whitespace-collapsed body; empty when no body followed


def _paragraph_end(text: str, start: int, stop: int) -> int:
    """End of the paragraph starting at ``start`` (first blank line or ``stop``)."""
    lines_end = start
    position = start
    while position < stop:
        newline = text.find("\n", position, stop)
        line_end = stop if newline < 0 else newline
        line = text[position:line_end]
        if not line.strip():
            break
        lines_end = line_end
        if newline < 0:
            break
        position = newline + 1
    return lines_end


def drafts(text: str) -> list[Draft]:
    """Every draft a turn wrote, in order, with its span and normalized body."""
    markers = list(_MARKER.finditer(text))
    found = []
    for index, match in enumerate(markers):
        limit = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        position = match.end()
        # Skip blank lines between the marker and the body.
        while position < limit:
            newline = text.find("\n", position, limit)
            line_end = limit if newline < 0 else newline
            if text[position:line_end].strip():
                break
            position = limit if newline < 0 else newline + 1
        if position >= limit:
            found.append(Draft(match.start(), match.end(), ""))
            continue
        if _FENCE.match(text[position:limit]):
            opening = text.find("\n", position, limit)
            body_start = limit if opening < 0 else opening + 1
            closing = text.find("```", body_start, limit)
            body_end = limit if closing < 0 else closing
            body = text[body_start:body_end]
            end = body_end if closing < 0 else closing + 3
        else:
            end = _paragraph_end(text, position, limit)
            body = text[position:end]
        found.append(Draft(match.start(), end, _SPACES.sub(" ", body).strip()))
    return found


def repeats(found: list[Draft]) -> list[int]:
    """Indexes of drafts whose body is identical to an earlier draft's body."""
    seen: set[str] = set()
    result = []
    for index, draft in enumerate(found):
        if not draft.normalized:
            continue
        if draft.normalized in seen:
            result.append(index)
        seen.add(draft.normalized)
    return result


def loop_cut(found: list[Draft]) -> int | None:
    """Character offset just past the first identical re-draft, or None.

    Everything up to and including that draft stays in the loss: the marker and
    the repeated body are the decision tokens the penalty must land on. What
    follows is the loop, and gets no gradient.
    """
    repeated = repeats(found)
    return found[repeated[0]].body_end if repeated else None


def token_cut(tokens: list[int], decode, char_end: int) -> int:
    """Smallest token prefix whose decoded text covers ``char_end`` characters.

    ``decode`` maps a token id list to text. Decoded length is monotone in the
    prefix length for practical purposes; the final loop repairs the rare
    off-by-one that byte-level merges can introduce.
    """
    if char_end <= 0:
        return 0
    low, high = 0, len(tokens)
    while low < high:
        middle = (low + high) // 2
        if len(decode(tokens[:middle])) >= char_end:
            high = middle
        else:
            low = middle + 1
    while low < len(tokens) and len(decode(tokens[:low])) < char_end:
        low += 1
    return low


# --- Repetition ---------------------------------------------------------------------------------------------------
#
# Two detectors for loops the draft detector cannot see:
#
# * ``periodic_run``: the generated token ids of one turn repeat with a short
#   period (a word or a few words) for at least ``min_tokens`` tokens.
# * ``span_loop``: a 40-160 character span of the decoded text occurs at least
#   ``min_repeats`` times back to back (whitespace between copies allowed).
#
# Both return the position where the loop *starts* so the caller can keep the
# advantage on everything up to and including the first copy and mask the
# rest, exactly as ``loop_cut`` does for re-drafts. Neither fires on ordinary
# SQL or prose: identical 40-character spans four times in a row, or a
# sub-8-token pattern sustained for 48 tokens, do not occur in legitimate
# answers (see tests/test_guards.py for the corpus).


def periodic_run(
    tokens: list[int], *, max_period: int = 8, min_tokens: int = 48
) -> tuple[int, int] | None:
    """Earliest ``(start, period)`` of a run of >= ``min_tokens`` tokens that
    repeats with period <= ``max_period``; ``None`` when there is none."""
    n = len(tokens)
    best = None
    for period in range(1, max_period + 1):
        run = 0
        for i in range(period, n):
            if tokens[i] == tokens[i - period]:
                run += 1
                if run + period >= min_tokens:
                    start = i - run - period + 1
                    if best is None or start < best[0]:
                        best = (start, period)
                    break
            else:
                run = 0
    return best


_SPAN_LOOP = re.compile(r"(?s)(\S.{39,159}?)(?:\s*\1){3,}")


def span_loop(
    text: str, *, min_repeats: int = 4
) -> tuple[int, int] | None:
    """``(start, first_copy_end)`` of the earliest span of 40-160 characters
    repeated at least ``min_repeats`` times back to back, else ``None``."""
    if min_repeats != 4:
        pattern = re.compile(
            r"(?s)(\S.{39,159}?)(?:\s*\1){" + str(max(1, min_repeats - 1)) + r",}"
        )
    else:
        pattern = _SPAN_LOOP
    match = pattern.search(text)
    if match is None:
        return None
    return match.start(), match.start() + len(match.group(1))


def detect_repetition(
    tokens: list[int],
    text: str,
    decode: Callable[[list[int]], str] | None,
    *,
    max_period: int = 8,
    min_tokens: int = 48,
    span_repeats: int = 4,
) -> tuple[int, int, int] | None:
    """(kind, keep_tokens, char_end) for a loop in the turn just sampled.

    kind 1 = periodic token run, 2 = repeated text span. ``keep_tokens`` is
    the number of turn tokens that keep their advantage (up to and including
    the first copy); ``char_end`` the matching decoded-text boundary.

    ``tokens`` are the turn's generated token ids, ``text`` their decoded text
    and ``decode`` the tokenizer's decode. Without ``decode`` only token runs
    are found, with ``char_end`` 0. The earliest loop wins; on the same start,
    the one that keeps fewer tokens.
    """
    hit = periodic_run(
        tokens,
        max_period=int(max_period),
        min_tokens=int(min_tokens),
    )
    candidates = []
    if hit is not None:
        keep = hit[0] + hit[1]
        char_end = len(decode(tokens[:keep])) if decode else 0
        candidates.append((hit[0], (1, keep, char_end)))
    span = span_loop(
        text or "",
        min_repeats=int(span_repeats),
    )
    if span is not None and decode is not None:
        start = token_cut(tokens, decode, span[0])
        keep = token_cut(tokens, decode, span[1])
        candidates.append((start, (2, keep, span[1])))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1][1]))[1]


# --- Per-turn guards -----------------------------------------------------------------------------------------------

# Truncation causes the rollout engine reports to `on_truncated`:
# - BUDGET_TRUNCATION_CAUSES: the trajectory's token budget cut the turn off mid-generation, so the turn counts as a
#   cap hit although its stop reason is left over from the previous turn;
# - LOOP_GUARD_TRUNCATION_CAUSES: the causes on which the loop guards run, as after a budget truncation (no
#   stop-reason check; the raw turn text is parsed again for tool calls).
BUDGET_TRUNCATION_CAUSES = frozenset({"context", "generated_tokens"})
LOOP_GUARD_TRUNCATION_CAUSES = frozenset({"context", "generated_tokens", "turns", "final_answer"})


@dataclass(frozen=True)
class TurnScan:
    """What `scan_turn` found in one turn; `apply_loop_guards` acts on it."""

    drafts: list[Draft]
    repeated: list[int]  # indexes into `drafts` of verbatim re-drafts
    repetition: tuple[int, int, int] | None = None  # `detect_repetition` result; cap-hit turns only


def scan_turn(
    text: str,
    tokens: list[int],
    metrics: dict[str, float],
    settings: GuardConfig,
    *,
    decode: Callable[[list[int]], str] | None,
    budget_truncated: bool = False,
    last_stop_reason: str = "",
) -> TurnScan:
    """Draft and repetition bookkeeping for the turn just sampled.

    ``text`` is the turn's decoded text (the trajectory's ``final_text``),
    ``tokens`` the turn's generated token ids (every generated span since the
    previous turn was recorded), ``metrics`` the trajectory's metrics, updated
    in place:

    - ``drafts``: the most drafts any turn wrote; ``redrafts``: verbatim
      re-drafts summed over turns (read by the re-draft penalty);
    - on a cap-hit turn (``budget_truncated``, or the sampler stopped it with
      reason ``"length"``): ``cap_hit``, ``loop`` when it re-drafted, and
      ``repetition`` / ``repetition_kind`` when `detect_repetition` found a loop.

    Call it once per sampled turn: it adds to ``metrics``, so a second scan of
    the same turn would count it twice. The environment keeps the number of the
    last scanned turn in its saved state; for a turn it already scanned it
    passes ``TurnScan([], [], repetition)`` to `apply_loop_guards` instead, with
    the repetition its earlier scan found (None after a restore).
    """
    repetition = None
    found = drafts(text)
    repeated = repeats(found)
    metrics["drafts"] = max(metrics.get("drafts", 0.0), float(len(found)))
    metrics["redrafts"] = metrics.get("redrafts", 0.0) + float(
        len(repeated)
    )
    if budget_truncated or last_stop_reason == TRUNCATED_STOP_REASON:
        metrics["cap_hit"] = 1.0
        if repeated:
            metrics["loop"] = 1.0
        repetition = detect_repetition(
            tokens,
            text,
            decode,
            max_period=settings.repetition_max_period,
            min_tokens=settings.repetition_min_tokens,
            span_repeats=settings.repetition_span_repeats,
        )
        if repetition is not None:
            metrics["repetition"] = 1.0
            metrics["repetition_kind"] = float(repetition[0])
    return TurnScan(found, repeated, repetition)


@dataclass(frozen=True)
class LoopStop:
    """A guard ended the rollout inside the turn.

    The environment then sets the trajectory's loss mask end to
    ``loss_mask_end``, adds ``metrics`` to the trajectory's metrics, records the
    turn, and commits as the answer the text of the renderer's parse of the turn
    text up to ``char_end`` (stripped), with the turn's stop reason. The rollout
    ends there and is scored.
    """

    kind: str  # "draft" (verbatim re-draft) or "repetition"
    keep_tokens: int  # turn tokens that keep their advantage
    char_end: int  # matching boundary in the turn's decoded text
    loss_mask_end: int  # generated-token offset in the whole trajectory where the loss mask starts
    metrics: dict[str, float]


def draft_loop_stop(
    found: list[Draft],
    *,
    turn_tokens: list[int],
    decode: Callable[[list[int]], str],
    generated_tokens: int,
) -> LoopStop:
    """Mask the loss past the first identical re-draft and end the rollout.

    The turn hit its cap while repeating a draft verbatim. Tokens up to and
    including the first repeated draft keep their advantage (the decision
    to re-draft is what the failure reward must reach); the loop after it
    becomes context only. The answer is whatever preceded the cut, which
    is normally nothing, so the rollout scores the existing no-answer
    reward.
    """
    char_end = loop_cut(found)
    keep = token_cut(turn_tokens, decode, char_end)
    return LoopStop(
        "draft",
        keep,
        char_end,
        generated_tokens - len(turn_tokens) + keep,
        {"loop_stopped": 1.0, "loop_masked_tokens": float(len(turn_tokens) - keep)},
    )


def repetition_loop_stop(
    repetition: tuple[int, int, int],
    *,
    turn_tokens: list[int],
    generated_tokens: int,
) -> LoopStop:
    """Mask the loss past the first copy of a repetition loop and end the
    rollout; the answer is whatever preceded the loop (normally nothing)."""
    _kind, keep, char_end = repetition
    keep = max(0, min(len(turn_tokens), keep))
    return LoopStop(
        "repetition",
        keep,
        char_end,
        generated_tokens - len(turn_tokens) + keep,
        {
            "repetition_stopped": 1.0,
            "repetition_masked_tokens": float(len(turn_tokens) - keep),
        },
    )


def apply_loop_guards(
    scan: TurnScan,
    settings: GuardConfig,
    *,
    turn_tokens: list[int],
    message: Mapping[str, Any],
    decode: Callable[[list[int]], str] | None,
    generated_tokens: int,
    budget_truncated: bool = False,
    last_stop_reason: str = "",
) -> LoopStop | None:
    """The guard that ends this turn's rollout at a loop, or None to continue.

    Acts only on a turn that hit its cap (``budget_truncated``, or stop reason
    ``"length"``) and did not call a tool. ``message`` is the turn's parsed
    message; after a budget truncation the engine stores the raw text instead,
    so pass the renderer's parse of the turn text (with the task's tool specs).
    ``generated_tokens`` is the trajectory's generated-token count, this turn
    included. From `on_truncated`, pass ``budget_truncated=True`` for every
    cause in `LOOP_GUARD_TRUNCATION_CAUSES`.
    """
    if decode is None or not turn_tokens:
        return None
    if not budget_truncated and last_stop_reason != TRUNCATED_STOP_REASON:
        return None
    if message.get("tool_calls"):
        return None
    cuts = []
    if scan.repeated and settings.loop_stop:
        cut = token_cut(
            turn_tokens,
            decode,
            loop_cut(scan.drafts),
        )
        cuts.append((cut, "draft"))
    if scan.repetition is not None and settings.repetition_guard:
        cuts.append((scan.repetition[1], "repetition"))
    if not cuts:
        return None
    if min(cuts)[1] == "draft":
        return draft_loop_stop(
            scan.drafts, turn_tokens=turn_tokens, decode=decode, generated_tokens=generated_tokens
        )
    return repetition_loop_stop(
        scan.repetition, turn_tokens=turn_tokens, generated_tokens=generated_tokens
    )


# --- Batch monitors -----------------------------------------------------------------------------------------------


def guard_metrics(trajectories: Sequence[Any], group_size: int | None = None) -> dict[str, float]:
    """Loop-guard and answer-diversity monitors for one training batch.

    ``guard/*`` averages per-rollout metrics over the batch. ``diversity/*``
    reads each group (contiguous runs of ``group_size`` trajectories): distinct
    answers, majority-vote correctness and its lift over the batch accuracy
    (``sampled/accuracy``), and the pass rate. Trajectories expose ``metrics``
    and, once scored, ``answer_signature`` (the committed SQL, whitespace
    collapsed; empty for no answer).
    """
    flat = list(trajectories)
    if not flat:
        return {}

    def rate(items, key):
        return sum(float(t.metrics.get(key, 0.0)) for t in items) / max(1, len(items))

    out = {
        "guard/loop_rate": rate(flat, "loop"),
        "guard/loop_stopped_rate": rate(flat, "loop_stopped"),
        "guard/cap_hit_rate": rate(flat, "cap_hit"),
        "guard/drafts_mean": rate(flat, "drafts"),
        "guard/redrafts_mean": rate(flat, "redrafts"),
        "guard/redraft_penalty_mean": rate(flat, "redraft_penalty"),
        "guard/process_penalty_mean": rate(flat, "process_penalty"),
        "guard/process_violations_mean": rate(flat, "process_violations"),
        "guard/repetition_rate": rate(flat, "repetition"),
        "guard/repetition_stopped_rate": rate(flat, "repetition_stopped"),
        "guard/repetition_masked_tokens_mean": rate(flat, "repetition_masked_tokens"),
        "sampled/accuracy": rate(flat, "correct"),
    }
    if group_size and len(flat) % group_size == 0:
        distinct, vote, passed = [], [], []
        for start in range(0, len(flat), group_size):
            members = flat[start : start + group_size]
            signatures = [getattr(t, "answer_signature", "") or "" for t in members]
            answered = [s for s in signatures if s]
            distinct.append(float(len(set(answered))))
            passed.append(float(any(t.metrics.get("correct") for t in members)))
            if answered:
                # Majority answer; ties go to the earliest member, deterministically.
                best = max(answered.count(s) for s in answered)
                top = next(s for s in answered if answered.count(s) == best)
                vote.append(
                    float(
                        next(
                            t.metrics.get("correct", 0.0)
                            for t, s in zip(members, signatures, strict=True)
                            if s == top
                        )
                    )
                )
            else:
                vote.append(0.0)
        count = len(distinct)
        out["diversity/distinct_answers_mean"] = sum(distinct) / count
        out["diversity/vote_correct_rate"] = sum(vote) / count
        out["diversity/pass_rate"] = sum(passed) / count
        out["diversity/vote_lift"] = (
            out["diversity/vote_correct_rate"] - out["sampled/accuracy"]
        )
    return out


class CollapsePause:
    """Stop training when cap hits and repetition persist; resuming needs a review.

    `check` runs before each optimizer step with the step's trajectories. A step
    is *bad* when the fraction of rollouts that hit the token cap exceeds
    ``collapse_cap_rate`` and the fraction with a detected repetition exceeds
    ``collapse_repetition_rate``. After ``collapse_consecutive_steps`` bad steps
    in a row, once the pause is armed (steps after
    ``collapse_arm_after_steps``), it writes ``collapse-pause.json`` and raises;
    while that marker exists, every later construction or check raises too, so
    a restarted run stays stopped until the marker is removed.

    The streak is kept in ``collapse-state.json`` (written atomically and
    fsynced) and is tracked and logged before the pause is armed. Checking the
    same step again (a retried step) recomputes it from the streak before that
    step, and a state file ahead of the trainer is refused.
    """

    def __init__(self, root: str | os.PathLike, settings: GuardConfig):
        self.root = Path(root)
        self.settings = settings
        self.state_path = self.root / "collapse-state.json"
        self.marker = self.root / "collapse-pause.json"
        self.check_resume()

    def persist(self, path, state):
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            stream.write(json.dumps(state, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def check_resume(self):
        if self.marker.exists():
            raise RuntimeError(f"Training paused by the collapse guard: review the run's recent rollouts, then delete "
                               f"{self.marker} and run the same command, with the same --run-name, to resume")

    def check(self, batch: int, trajectories: Sequence[Any], metrics: dict[str, float]) -> None:
        """Record step ``batch`` (the global step number, 1-based) and pause if needed.

        Adds ``guard/pause_cap_rate``, ``guard/pause_repetition_rate`` and
        ``guard/pause_armed`` to ``metrics``. Raises RuntimeError when the pause
        triggers or was triggered before.
        """
        self.check_resume()
        n = len(trajectories)
        if not n:
            return
        cap = sum(bool(t.metrics.get("cap_hit", 0)) for t in trajectories) / n
        repetition = sum(bool(t.metrics.get("repetition", 0)) for t in trajectories) / n
        metrics.update(
            {"guard/pause_cap_rate": cap, "guard/pause_repetition_rate": repetition}
        )
        state = (
            json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        )
        if state.get("batch", -1) > batch:
            raise RuntimeError(
                "Collapse state is ahead of trainer; refusing unsafe rewind"
            )
        bad = (
            cap > self.settings.collapse_cap_rate
            and repetition > self.settings.collapse_repetition_rate
        )
        previous = (
            state.get("previous_streak", 0)
            if state.get("batch") == batch
            else (state.get("streak", 0) if state.get("batch") == batch - 1 else 0)
        )
        armed = batch > self.settings.collapse_arm_after_steps
        metrics["guard/pause_armed"] = float(armed)
        streak = previous + 1 if bad else 0
        state = {
            "batch": batch,
            "previous_streak": previous,
            "armed": armed,
            "streak": streak,
            "cap_rate": cap,
            "repetition_rate": repetition,
            "time": time.time(),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        # Write the blocking marker first so a crash cannot lose a triggered pause.
        # Before the pause is armed the streak is still tracked and logged, but no marker is written.
        if armed and streak >= self.settings.collapse_consecutive_steps:
            self.persist(self.marker, state)
        self.persist(self.state_path, state)
        self.check_resume()
