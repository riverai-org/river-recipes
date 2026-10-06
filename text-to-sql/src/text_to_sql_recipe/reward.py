"""The reward a training rollout earns: the outcome, and the shaping terms applied to it.

`rollout_reward` applies the terms in this order:

1. **Outcome** (`reward_for`): +1 when the committed answer's result matches the gold result, 0 when it runs (or
   fails) but does not match, ``NO_ANSWER_REWARD`` (-1) when there is no well-formed ``<solution>`` block, and
   ``UNSAFE_REWARD`` (-1) when the rollout attempted an unsafe query. A matching answer that the equivalence verifier
   refutes scores ``1 - verifier_weight``.
2. **Process rules** (`process_penalty`): ``process_penalty_weight`` for each of the two evidence-supervision rules
   the rollout breaks, subtracted from the outcome. Only questions shown external knowledge are checked.
3. **Re-drafts** (`apply_redraft_penalty`): ``redraft_penalty`` per verbatim re-draft, at most
   ``redraft_penalty_cap``, unless the loop guard already ended the rollout at the re-draft.
4. **Tool format** (`apply_tool_format_penalty`): ``tool_format_penalty`` per malformed tool-call attempt, at most
   ``tool_format_penalty_cap``.

Terms 3 and 4 never take a reward below the no-answer reward and never change a reward already at or below it.

Rewards are centred within each question's group of rollouts, so a term only moves the gradient when it differs
between members of a group: a deduction every member pays cancels out.

The tool-format term reads per-turn records of the tool calls the policy attempted (`observe_tool_calls`). Recording
them does not change the turn protocol: which call runs, and when a malformed call is asked for again, are decided
by the parser as before.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .config import RewardConfig
from .parsing import ToolCall, has_tool_call_intent, message_text, parse_tool_calls
from .prompts import REQUIREMENT_MARKER, SQL_TOOL_NAME, VERIFICATION_MARKER, external_knowledge_pieces

# A solution that does not satisfy the output contract (no `<solution>` block) scores -1, as in the ReViSQL paper
# (arXiv:2603.20004, App. E.5). The value is load-bearing beyond the failing rollout: rewards are group-mean
# centred, so the reward *range* scales every advantage in the group.
NO_ANSWER_REWARD = -1.0
# An unsafe query (anything but read-only SQL) also scores -1. The paper has no unsafe-query concept; a policy
# that is not penalised for one has no reason to stop trying.
UNSAFE_REWARD = -1.0
# The sampler's stop reason for a turn that ended because it hit its token cap; the only evidence a rollout
# carries that its answer was cut off rather than finished.
TRUNCATED_STOP_REASON = "length"


# --- Outcome --------------------------------------------------------------------------------------------------------


def reward_for(
    grade: Mapping[str, Any], *, unsafe: bool, refuted: bool, verifier_weight: float
) -> float:
    """The scalar outcome reward for one graded rollout.

    ``grade`` is the grader's verdict (``grade`` CORRECT, INCORRECT or
    NO_ANSWER, and ``error_category``); ``unsafe`` is true when the rollout's
    SQL tool calls attempted an unsafe query; ``refuted`` is true when the
    equivalence verifier's final verdict on a matching answer is a refutation
    (after its counterexample gate). Every other verifier outcome (confirmed,
    unknown, inconclusive, timeout, unsupported) keeps the full reward.
    """
    if unsafe or grade.get("error_category") == "unsafe_sql":
        return UNSAFE_REWARD
    verdict = grade.get("grade")
    if verdict == "CORRECT":
        if refuted:
            return 1.0 - float(verifier_weight)
        return 1.0
    if verdict == "NO_ANSWER":
        return NO_ANSWER_REWARD
    return 0.0


# --- Process rules --------------------------------------------------------------------------------------------------


def assistant_turns(messages: Sequence[Mapping[str, Any]], prompt_message_count: int = 0) -> list[str]:
    """The assistant's own text, one entry per turn, in order.

    The first ``prompt_message_count`` messages are prompt (system prompt,
    demonstration, question): the demonstration's assistant turn is not the
    policy's behaviour, so the requirement rule ("by the second turn") scores
    the model's first two turns rather than the demonstration's.
    """
    start = int(prompt_message_count or 0)
    return [
        message_text(m)
        for m in messages[start:]
        if m.get("role") == "assistant"
    ]


def process_violations(task: Mapping[str, Any], turns: Sequence[str]) -> dict:
    """The two evidence-supervision rules of the paper (App. E.5).

    Both are string-level checks on the trajectory against the indexed formats
    the system prompt names (``REQUIREMENT_MARKER``, ``VERIFICATION_MARKER``).

    1. **Requirement rule.** The requirement block must appear by the second
       turn. It applies whether or not the rollout ultimately produces a
       solution, because it is checked before the model commits an answer.
    2. **Verification rule.** The verification block must appear in the same
       response as the final ``<solution>``. Evaluated when the model commits
       one.

    Relaxed when the model answers on the **first** turn (easy problems that
    need no exploration): then only the absence of *both* blocks is a
    violation, and either one present satisfies both rules. A rollout therefore
    breaks at most two rules, and at most one when it ends on turn one.

    A question shown no external knowledge (``task["evidence"]`` empty) requires
    neither block.
    """
    pieces = external_knowledge_pieces(task)
    if not pieces or not turns:
        return {"requirement": False, "verification": False, "applicable": False}

    # Requirement: anywhere in the first two turns.
    requirement_present = any(REQUIREMENT_MARKER in text for text in turns[:2])
    # Verification: in the response that carries the solution. That is the
    # final turn when the rollout committed one.
    committed = "<solution>" in (turns[-1] or "")
    verification_present = (
        VERIFICATION_MARKER in (turns[-1] or "") if committed else False
    )

    if len(turns) == 1:
        # First-turn relaxation: either block satisfies both rules.
        if requirement_present or verification_present:
            return {"requirement": False, "verification": False, "applicable": True}
        return {"requirement": True, "verification": False, "applicable": True}

    return {
        "requirement": not requirement_present,
        "verification": committed and not verification_present,
        "applicable": True,
    }


def process_penalty(
    task: Mapping[str, Any], turns: Sequence[str], *, weight: float
) -> tuple[float, dict]:
    """Total process penalty for a rollout, and which rules it violated."""
    violations = process_violations(task, turns)
    lam = float(weight)
    total = lam * sum(
        1 for key in ("requirement", "verification") if violations.get(key)
    )
    return total, violations


# --- Truncated rollouts --------------------------------------------------------------------------------------------


def commits_answer_on_truncation(cause: str, *, turns: int, max_turns: int) -> bool:
    """Whether a rollout the engine ended with ``cause`` is graded on its last turn.

    Only the answer turn (turn ``max_turns + 1``) ending on the turn limit is
    committed as the answer, with that turn's stop reason. Every other
    truncation ends the rollout with no answer (empty answer text), which grades
    NO_ANSWER and scores ``NO_ANSWER_REWARD``.
    """
    return cause == "turns" and turns == max_turns + 1


# --- The outcome and the process rules -----------------------------------------------------------------------------


def apply_process_penalty(
    grade: Mapping[str, Any],
    settings: RewardConfig,
    *,
    task: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    prompt_message_count: int,
    unsafe: bool,
    refuted: bool,
) -> tuple[float, dict[str, Any]]:
    """The reward from the outcome grade and the rollout's *own* trajectory.

    Returns the reward (outcome minus process penalty) and the fields to set on
    the rollout's grade: ``process_violations`` and ``process_penalty``.

    Grades are shared between rollouts that committed the same answer, but the
    process penalty reads the whole trajectory, so it is computed for each
    rollout on its own; copying one rollout's penalty to its duplicates would
    shift the group mean and every advantage in the group. On a grading
    attempt, before any verifier verdict, pass ``refuted=False``.
    """
    outcome = reward_for(grade, unsafe=unsafe, refuted=refuted, verifier_weight=settings.verifier_weight)
    penalty, violations = process_penalty(
        task, assistant_turns(messages, prompt_message_count), weight=settings.process_penalty_weight
    )
    return outcome - penalty, {"process_violations": violations, "process_penalty": penalty}


# --- Re-drafts ------------------------------------------------------------------------------------------------------


def apply_redraft_penalty(
    reward: float, *, redrafts: float, loop_stopped: bool, penalty: float, cap: float
) -> tuple[float, float | None]:
    """Deduct ``penalty`` per verbatim re-draft, at most ``cap``.

    ``redrafts`` is the trajectory's re-draft count summed over its turns (the
    ``redrafts`` metric `guards.scan_turn` keeps); ``loop_stopped`` is true when
    the loop guard ended the rollout at a re-draft, which already scores it as
    no answer. A reward at or below ``NO_ANSWER_REWARD`` is left alone, and none
    is taken below it. Returns the new reward and the deduction, or None when
    the penalty did not apply (then no ``redraft_penalty`` is recorded on the
    grade).
    """
    penalty = float(penalty or 0.0)
    amount = None
    if penalty and redrafts and not loop_stopped:
        floor = NO_ANSWER_REWARD
        if reward > floor:
            amount = min(penalty * float(redrafts), float(cap))
            reward = max(floor, reward - amount)
    return reward, amount


# --- Tool-call format -----------------------------------------------------------------------------------------------
#
# Each sampled turn that goes through the ordinary turn protocol is checked for tool-call attempts, and every invalid
# attempt gets one error category. SQL strings are opaque: an empty result, invalid SQL or a database error is not a
# protocol violation. Only generated messages are checked, never the demonstration.

TOOL_PROTOCOL_POLICY = "tool-protocol-v1"
TOOL_PROTOCOL_ERRORS = (
    "invalid_envelope",
    "invalid_arguments",
    "tool_after_limit",
    "unknown_tool",
    "missing_query",
    "invalid_query_type",
    "empty_query",
)


def inspect_calls(message: dict, *, tools_allowed: bool = True):
    """Return executable parsed calls and one error category per invalid attempt.

    Validate arguments before the permissive parser normalizes them. Reads the
    structured ``tool_calls`` and the message text, so pass the turn as the
    tool-call parser reads it (`parsing.execution_message`: call syntax the
    renderer rejected is back in the text). ``tools_allowed`` is false on the
    answer turn, where every call is an error.
    """
    calls = []
    errors = []
    entries = message.get("tool_calls") or []
    if not isinstance(entries, list):
        return [], ["invalid_envelope"]
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("function"), dict):
            errors.append("invalid_envelope")
            continue
        function = entry["function"]
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (ValueError, TypeError, RecursionError):
                errors.append("invalid_arguments")
                continue
        calls.append(
            ToolCall(
                name=function.get("name"),
                arguments=arguments,
                call_id=str(entry["id"]) if entry.get("id") else None,
            )
        )
    # With structured calls the renderer has already removed those calls from
    # content. Any remaining call syntax is a separate (possibly rejected) call.
    content_message = {"content": message_text(message)}
    content_calls = parse_tool_calls(content_message)
    calls.extend(content_calls)
    if not content_calls and has_tool_call_intent(
        content_message, kimi_markers=True
    ):
        errors.append("invalid_envelope")
    valid = []
    for call in calls:
        error = _call_error(call, tools_allowed)
        if error:
            errors.append(error)
        else:
            valid.append(call)
    return valid, errors


def _call_error(call, tools_allowed):
    if not tools_allowed:
        return "tool_after_limit"
    if call.name != SQL_TOOL_NAME:
        return "unknown_tool"
    if not isinstance(call.arguments, dict):
        return "invalid_arguments"
    if "query" not in call.arguments:
        return "missing_query"
    if not isinstance(call.arguments["query"], str):
        return "invalid_query_type"
    return "empty_query" if not call.arguments["query"].strip() else None


def observe_tool_calls(events: list[dict], message: dict, *, turn: int, tools_allowed: bool):
    """Record turn ``turn``'s attempts in ``events`` (in place); returns `inspect_calls`' result.

    One event per turn, ``{"turn", "valid_calls", "errors"}``. Observing a turn
    again replaces its event, so a turn replayed after recovery is not charged
    twice. Keep ``events`` with the rollout's saved state.
    """
    calls, errors = inspect_calls(message, tools_allowed=tools_allowed)
    # A repeated observation after recovery must not charge a sampled turn twice.
    events[:] = [e for e in events if e["turn"] != turn]
    events.append({"turn": turn, "valid_calls": len(calls), "errors": errors})
    return calls, errors


def tool_protocol_metrics(events: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Trajectory metrics for the recorded attempts: totals and one count per error category."""
    counts = Counter(
        error for event in events for error in event["errors"]
    )
    invalid = sum(counts.values())
    return {
        "tool_protocol/invalid_calls": float(invalid),
        "tool_protocol/invalid_trajectory": float(bool(invalid)),
        "tool_protocol/valid_calls": float(
            sum(e["valid_calls"] for e in events)
        ),
        **{f"tool_protocol/{key}": float(counts[key]) for key in TOOL_PROTOCOL_ERRORS},
    }


def apply_tool_format_penalty(
    reward: float, events: Sequence[Mapping[str, Any]], *, per_call: float, cap: float
) -> tuple[float, dict[str, Any], dict[str, float]]:
    """Deduct ``per_call`` per invalid attempt, at most ``cap``, never below ``NO_ANSWER_REWARD``.

    Applied after every other term. Returns the new reward, the receipt for the
    grade (``tool_protocol``: the actual deduction, after the floor) and the
    trajectory metrics (`tool_protocol_metrics` plus ``tool_protocol/penalty``).
    """
    counts = Counter(
        error for event in events for error in event["errors"]
    )
    invalid = sum(counts.values())
    per_call = float(per_call)
    cap = float(cap)
    floor = NO_ANSWER_REWARD
    before = reward
    amount = min(per_call * invalid, cap, max(0.0, before - floor))
    after = before - amount
    receipt = {
        "policy": TOOL_PROTOCOL_POLICY,
        "invalid_calls": invalid,
        "errors": dict(counts),
        "penalty_per_call": per_call,
        "penalty_cap": cap,
        "reward_before": before,
        "penalty": amount,
        "reward_after": after,
    }
    return after, receipt, {**tool_protocol_metrics(events), "tool_protocol/penalty": amount}


# --- The rollout's reward -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutReward:
    reward: float
    grade: dict[str, Any]  # fields to set on the rollout's grade: the per-term receipts
    metrics: dict[str, float]  # per-term trajectory metrics


def rollout_reward(
    grade: Mapping[str, Any],
    settings: RewardConfig,
    *,
    task: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    prompt_message_count: int,
    unsafe: bool,
    refuted: bool,
    redrafts: float,
    loop_stopped: bool,
    tool_protocol_events: Sequence[Mapping[str, Any]],
) -> RolloutReward:
    """Score one finished, graded rollout: every term, in order.

    Inputs:
      grade                     the grader's verdict for the committed answer
      task                      the question as the rollout saw it (its ``evidence`` decides the process rules)
      messages, prompt_message_count
                                the rollout's conversation (assistant turns as sampled, with their reasoning) and
                                how many leading messages are prompt
      unsafe                    the rollout attempted an unsafe query
      refuted                   the equivalence verifier refuted a matching answer (after its gate)
      redrafts, loop_stopped    the trajectory's ``redrafts`` and ``loop_stopped`` metrics (`guards`)
      tool_protocol_events      the per-turn events `observe_tool_calls` recorded

    Returns the reward, the grade fields (``process_violations``,
    ``process_penalty``, ``redraft_penalty`` when it applied, ``tool_protocol``)
    and the metrics (``process_penalty``, ``process_violations`` as a count,
    ``redraft_penalty``, and ``tool_protocol/*``).
    """
    reward, fields = apply_process_penalty(
        grade,
        settings,
        task=task,
        messages=messages,
        prompt_message_count=prompt_message_count,
        unsafe=unsafe,
        refuted=refuted,
    )
    reward, redraft_amount = apply_redraft_penalty(
        reward,
        redrafts=redrafts,
        loop_stopped=loop_stopped,
        penalty=settings.redraft_penalty,
        cap=settings.redraft_penalty_cap,
    )
    if redraft_amount is not None:
        fields["redraft_penalty"] = redraft_amount
    reward, receipt, tool_metrics = apply_tool_format_penalty(
        reward,
        tool_protocol_events,
        per_call=settings.tool_format_penalty,
        cap=settings.tool_format_penalty_cap,
    )
    fields["tool_protocol"] = receipt
    metrics = {
        **tool_metrics,
        "redraft_penalty": redraft_amount or 0.0,
        "process_penalty": float(fields["process_penalty"]),
        "process_violations": float(
            sum(
                1
                for key in ("requirement", "verification")
                if (fields["process_violations"] or {}).get(key)
            )
        ),
    }
    return RolloutReward(reward, fields, metrics)
