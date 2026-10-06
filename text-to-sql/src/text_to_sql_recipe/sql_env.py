"""The text-to-SQL episode: one question, up to `rollout.max_turns` SQL tool turns, then an answer.

The model sees the opening messages from `prompts.opening_messages` and the `execute_sql_query` tool. Each turn it
either calls the tool once, which runs one read-only query against the question's database and returns the result
(or the error) as the next user message, or answers. After the last tool turn it is told to answer
(`prompts.SQL_LIMIT_PROMPT`), and the turn after that is graded as the answer whatever it contains. A call the
renderer could not parse gets one request for a valid call (`parsing.MALFORMED_ACTION_PROMPT`); a query that tries
to write or escape the sandbox ends the rollout as unsafe.

`Rollout` is the record the grader and the reward read. `SqlRuntime` grades committed answers against the
precomputed gold results, consults the verifier for answers graded correct, and applies the per-rollout penalties.
"""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any

from river_client import rl
from river_client.renderers import SamplePrompt

from . import grading, guards, parsing, prompts, reward, sqlite_exec
from .config import GuardConfig, RecipeConfig, RewardConfig


@dataclass
class Rollout:
    """One sampled trajectory, plus everything grading and the reward read."""

    prompt_index: int
    sample_index: int
    task: dict
    messages: list[dict]
    initial_prompt_ids: list[int] = field(default_factory=list)
    current_prompt_ids: list[int] = field(default_factory=list)
    turn_framing_ids: list[list[int]] = field(default_factory=list)
    turn_tokens: list[list[int]] = field(default_factory=list)
    turn_logprobs: list[list[float]] = field(default_factory=list)
    # None until the first sample answers it; then True for MoE, False for dense.
    routing_required: bool | None = None
    expert_routing_handle: bytes = b""
    expert_routing_num_tokens: int = 0
    prompt_token_total: int = 0
    tool_calls_made: int = 0
    final_text: str = ""
    # Why the turn that produced `final_text` stopped, as the sampler reported it: "length" when the turn was cut off
    # at its token cap, "stop" when the model emitted its stop string. Empty when no answer turn was sampled (the
    # rollout ran out of context, was cut short as unsafe, or exhausted its turns); those already score the
    # no-answer reward. Kept with the rollout for diagnosis; the reward does not read it.
    final_answer_stop_reason: str = ""
    grade: dict = field(default_factory=dict)
    reward: float = 0.0
    unsafe: bool = False
    done: bool = False
    # How many leading messages are prompt (system, demonstrations, question). The process rules read assistant
    # turns after this index only.
    prompt_message_count: int = 0
    # One event per sampled turn: its valid and invalid tool calls (`reward.observe_tool_calls`).
    tool_protocol_events: list[dict] = field(default_factory=list)


def commit_final_answer(rollout: Rollout, stop_reason: str | None, text: str) -> None:
    """Record the answer this rollout will be graded on, and how its turn ended."""
    rollout.final_text = text
    rollout.final_answer_stop_reason = str(stop_reason or "")
    rollout.done = True


def make_sql_tool(task: dict) -> parsing.ToolSpec:
    """The `execute_sql_query` tool for one task: schema, executor and error handling."""
    executor = sqlite_exec.tool_executor(task)
    return parsing.ToolSpec(
        name=prompts.SQL_TOOL_NAME,
        schema=prompts.sql_tool_schema({}),
        run=lambda arguments, executor=executor, task=task: sqlite_exec.execute_sql_tool(task, executor, arguments),
        terminal_errors=sqlite_exec.SQL_TOOL_TERMINAL_ERRORS,
        # The submitted SQL is unbounded, so a failure reports only a bounded excerpt of the executor's message.
        # The record keeps the original error whole; only the model's context is bounded.
        context_error_max_bytes=prompts.SQL_TOOL_CONTEXT_ERROR_MAX_BYTES,
        context_error_guidance=prompts.SQL_TOOL_CONTEXT_ERROR_GUIDANCE,
    )


def tool_result_message(
    spec: parsing.ToolSpec,
    results: list[Any],
    error: str,
    guidance: str,
    *,
    turns_left: int | None = None,
) -> dict:
    """The user message that reports one tool call's results (or its error) to the model."""
    if error:
        bounded = error
        limit = spec.context_error_max_bytes
        if limit is not None and limit > 0:
            encoded = bounded.encode("utf-8", "ignore")
            if len(encoded) > limit:
                bounded = encoded[:limit].decode("utf-8", "ignore")
        if spec.context_error_guidance:
            bounded = f"{bounded} {spec.context_error_guidance}"
        body = f"Results for {spec.name}:\nError: {bounded}"
    else:
        formatted = "\n\n".join(result.format() for result in results) or "(no output)"
        body = f"Results for {spec.name}:\n{formatted}"
    if guidance:
        body = f"{body}\n\n{guidance}"
    if turns_left is not None:
        body = f"{body}\n\nYou have {turns_left} turns left to complete the task."
    return {"role": "user", "content": body}


def execute_tool_call(
    rollout: Rollout,
    call: parsing.ToolCall,
    spec: parsing.ToolSpec,
    guidance: str,
    *,
    turns_left: int,
) -> None:
    """Run one tool call and append its result message; a terminal error ends the rollout as unsafe."""
    try:
        results = spec.run(call.arguments)
        observation = tool_result_message(spec, results, "", guidance, turns_left=turns_left)
    except sqlite_exec.SQLiteInfrastructureError:
        # A storage failure must not become model-visible feedback followed by a no-answer penalty when the
        # environment itself is unavailable.
        raise
    except spec.terminal_errors as exc:
        rollout.messages.append(tool_result_message(spec, [], str(exc), ""))
        rollout.unsafe = True
        rollout.done = True
        return
    except Exception as exc:  # noqa: BLE001 - reported to the model, which may recover
        observation = tool_result_message(spec, [], str(exc), guidance)
    rollout.messages.append(observation)


class RecipeRenderer:
    """The model's chat-template renderer with the recipe's incremental framing; never re-renders sampled ids."""

    def __init__(self, base, tools):
        self.base, self.tools, self.tokenizer = base, tools, base.tokenizer

    def get_stop_strings(self):
        return self.base.get_stop_strings()

    def build_sample_prompt(self, messages, *, tools=None):
        return self.base.build_sample_prompt(messages, tools=tools)

    def build_continuation_prompt(self, messages, *, last_stop):
        stops = self.get_stop_strings()
        if len(stops) != 1:
            # The incremental framing supports one terminator. A template with separate answer, tool and user stops
            # (GLM) uses its native framing.
            return self.base.build_continuation_prompt(messages, last_stop=last_stop)
        return SamplePrompt(("" if last_stop else stops[0]) + "\n" + self.base.build_prompt_str(messages), [], [])

    def parse_response(self, text, *, tools=None):
        # Use the member's schemas when called by the rollout engine.
        return self.base.parse_response(text, tools=self.tools if tools is None else tools)


class SqlRuntime:
    """Shared by every rollout of an epoch: the tool slots, the grader, the verifier and their caches.

    Outcome grades are shared between rollouts that committed the same answer (`grading.GradingRuntime`). The
    verifier runs only for answers graded correct and not unsafe, keyed by (task, answer, gold query, database);
    only decisive verdicts ("confirmed" or "refuted") are reused. The reward, which reads each rollout's own turns,
    is computed by the environment afterwards.
    """

    def __init__(
        self,
        settings: RewardConfig,
        expected: dict[str, Any],
        *,
        tool_concurrency: int,
        grading_concurrency: int,
        sql_concurrency: int,
        verifier=None,
        sql_work: sqlite_exec.SqlWork | None = None,
    ):
        if settings.verifier != "none" and settings.verifier_weight and verifier is None:
            raise ValueError("verifier shaping is configured but no verifier was given; refusing an unshaped run")
        self.settings = settings
        self.verifier = verifier
        self.tools = asyncio.Semaphore(tool_concurrency)
        self.grader = grading.GradingRuntime(
            {},
            expected,
            grading_concurrency=grading_concurrency,
            sql_concurrency=sql_concurrency,
            sql_work=sql_work,
            apply_penalties=self._attempt_receipt,
        )
        self.graders = self.grader.graders
        self.sql = self.grader.sql
        self.verifications = grading.CompletedTaskCache()

    def _attempt_receipt(self, rollout: Rollout, cfg: Mapping[str, Any]) -> None:
        # Each grading attempt's receipt carries the fields a finished grade has, before any verifier verdict.
        rollout.reward, fields = reward.apply_process_penalty(
            rollout.grade,
            self.settings,
            task=rollout.task,
            messages=rollout.messages,
            prompt_message_count=rollout.prompt_message_count,
            unsafe=rollout.unsafe,
            refuted=False,
        )
        rollout.grade.update(fields)

    async def grade(self, rollout: Rollout) -> None:
        await self.grader.grade(rollout)
        if self.verifier is not None and rollout.grade.get("grade") == "CORRECT" and not rollout.unsafe:
            pair_key = (
                str(rollout.task["id"]),
                rollout.grade["prediction"],
                rollout.task["gold_sql"],
                rollout.task.get("database_sha256"),
            )
            if pair_key not in self.verifications:
                self.verifications[pair_key] = asyncio.create_task(
                    self.verifier.check(rollout.task, rollout.grade["prediction"])
                )
            pending = self.verifications[pair_key]
            try:
                verdict = await asyncio.shield(pending)
                rollout.grade["verieql"] = copy.deepcopy(verdict)
            finally:
                # Only decisive results are reusable across later groups.
                if (
                    pending.done()
                    and (
                        pending.cancelled()
                        or pending.exception() is not None
                        or pending.result().get("verdict") not in {"confirmed", "refuted"}
                    )
                    and self.verifications.get(pair_key) is pending
                ):
                    del self.verifications[pair_key]

    async def close(self) -> None:
        tasks = list(self.verifications.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.grader.close()
        if self.verifier is not None:
            await self.verifier.close()


@dataclass(frozen=True)
class EpisodeSettings:
    """The parts of the recipe one episode reads."""

    max_turns: int
    few_shot: bool
    reward: RewardConfig
    guards: GuardConfig

    @classmethod
    def from_recipe(cls, recipe: RecipeConfig, *, evaluation: bool = False) -> EpisodeSettings:
        """Training settings, or for evaluation: no verifier, no penalties and no loop guards, so the score is the
        outcome."""
        rewards, guard_settings = recipe.reward, recipe.guards
        if evaluation:
            rewards = dataclasses.replace(
                rewards, verifier="none", process_penalty_weight=0.0, redraft_penalty=0.0, tool_format_penalty=0.0
            )
            guard_settings = dataclasses.replace(guard_settings, loop_stop=False, repetition_guard=False)
        return cls(recipe.rollout.max_turns, recipe.rollout.few_shot, rewards, guard_settings)


class SqlEnv(rl.Env):
    """One rollout of the episode; the rollout engine makes one per trajectory."""

    recovery = "snapshot"

    def __init__(self, runtime: SqlRuntime, renderer: RecipeRenderer, settings: EpisodeSettings):
        self.runtime, self.renderer, self.settings = runtime, renderer, settings
        self.cursor = 0  # index of the first span not yet recorded as a turn
        self.scanned = 0  # the last turn the guards scanned
        self.repetition = None  # the repetition loop the latest scan found

    async def reset(self, row):
        self.record = Rollout(0, 0, row, prompts.opening_messages(row, few_shot=self.settings.few_shot))
        # The process rules read the policy's own turns; everything up to here is prompt.
        self.record.prompt_message_count = len(self.record.messages)
        self.spec = make_sql_tool(row)
        self.tools = [SimpleNamespace(spec=self.spec.schema["function"])]
        return copy.deepcopy(self.record.messages)

    def _decode(self):
        return self.renderer.tokenizer.decode if self.renderer is not None else None

    def _turn_tokens(self, traj) -> list[int]:
        spans = [s for s in traj.spans[self.cursor :] if s.kind == "generated"]
        return [t for s in spans for c in s.chunks for t in c["tokens"]]

    def _scan(self, traj, *, budget_truncated: bool = False) -> guards.TurnScan:
        if traj.turns <= self.scanned:
            return guards.TurnScan([], [], self.repetition)
        self.scanned = traj.turns
        scan = guards.scan_turn(
            traj.final_text,
            self._turn_tokens(traj),
            traj.metrics,
            self.settings.guards,
            decode=self._decode(),
            budget_truncated=budget_truncated,
            last_stop_reason=traj.last_stop_reason,
        )
        self.repetition = scan.repetition
        return scan

    def _apply_loop_guards(self, traj, scan: guards.TurnScan, *, budget_truncated: bool = False) -> bool:
        if self.renderer is None:
            return False
        message = traj.messages[-1]
        if budget_truncated:
            # The engine stores a budget-truncated turn as raw text; parse it to keep the guard's no-tool-call rule.
            message = self.renderer.parse_response(traj.final_text, tools=[tool.spec for tool in self.tools]).message
        stop = guards.apply_loop_guards(
            scan,
            self.settings.guards,
            turn_tokens=self._turn_tokens(traj),
            message=message,
            decode=self._decode(),
            generated_tokens=traj.generated_tokens,
            budget_truncated=budget_truncated,
            last_stop_reason=traj.last_stop_reason,
        )
        if stop is None:
            return False
        traj.loss_mask_end = stop.loss_mask_end
        traj.metrics.update(stop.metrics)
        self.record_turn(traj)
        kept = self.renderer.parse_response(traj.final_text[: stop.char_end]).message
        commit_final_answer(self.record, traj.last_stop_reason, parsing.message_text(kept).strip())
        return True

    def record_turn(self, traj) -> None:
        spans = [s for s in traj.spans[self.cursor :] if s.kind == "generated"]
        self.cursor = len(traj.spans)
        if spans:
            self.record.turn_tokens.append([t for s in spans for c in s.chunks for t in c["tokens"]])
            self.record.turn_logprobs.append([p for s in spans for p in s.logprobs.tolist()])
            # The raw decoded turn, reasoning included: the process rules read it.
            self.record.messages.append({"role": "assistant", "content": traj.final_text})

    async def on_turn(self, traj):
        scan = self._scan(traj)
        if self._apply_loop_guards(traj, scan):
            return None
        self.record_turn(traj)
        before = len(self.record.messages)
        parsed = parsing.execution_message(traj.messages[-1])
        tool_turns = self.settings.max_turns
        tools_allowed = traj.turns <= tool_turns
        reward.observe_tool_calls(self.record.tool_protocol_events, parsed, turn=traj.turns, tools_allowed=tools_allowed)
        traj.metrics.update(reward.tool_protocol_metrics(self.record.tool_protocol_events))
        if not tools_allowed:
            commit_final_answer(self.record, traj.last_stop_reason, parsing.message_text(parsed).strip())
            return None
        calls = parsing.parse_tool_calls(parsed)
        known = next((c for c in calls if c.name == prompts.SQL_TOOL_NAME), None)
        if known is not None:
            self.record.tool_calls_made += 1
            started = time.monotonic()
            async with self.runtime.tools:
                await self.runtime.sql.execute(
                    execute_tool_call,
                    self.record,
                    known,
                    self.spec,
                    prompts.tool_result_prompt(),
                    turns_left=tool_turns - traj.turns,
                )
            traj.metrics["sql_seconds"] = traj.metrics.get("sql_seconds", 0.0) + time.monotonic() - started
        elif parsing.has_tool_call_intent(parsed):
            self.record.messages.append({"role": "user", "content": parsing.MALFORMED_ACTION_PROMPT})
        else:
            commit_final_answer(self.record, traj.last_stop_reason, parsing.message_text(parsed).strip())
        if self.record.done:
            return None
        if traj.turns == tool_turns:
            self.record.messages.append({"role": "user", "content": prompts.SQL_LIMIT_PROMPT})
        return copy.deepcopy(self.record.messages[before:])

    async def reward(self, traj, row):
        await self.runtime.grade(self.record)
        verdict = self.record.grade.get("verieql")
        scored = reward.rollout_reward(
            self.record.grade,
            self.settings.reward,
            task=self.record.task,
            messages=self.record.messages,
            prompt_message_count=self.record.prompt_message_count,
            unsafe=self.record.unsafe,
            refuted=(verdict or {}).get("verdict") == "refuted",
            redrafts=float(traj.metrics.get("redrafts", 0.0)),
            loop_stopped=bool(traj.metrics.get("loop_stopped")),
            tool_protocol_events=self.record.tool_protocol_events,
        )
        self.record.reward = scored.reward
        self.record.grade.update(scored.grade)
        correct = float(self.record.grade.get("grade") == "CORRECT")
        traj.sql_grade = grading.grading_receipt(self.record.grade)
        attempts = self.record.grade.get("grading_attempts", [])
        category = self.record.grade.get("error_category", "")
        traj.metrics.update(
            {
                "correct": correct,
                "format_compliant": float(bool(self.record.grade.get("format_compliant"))),
                "tool_calls": float(self.record.tool_calls_made),
                "grading/attempts": float(len(attempts)),
                "grading/seconds": sum(a["wall_seconds"] for a in attempts),
                "grading/timeout": float(category == "timeout"),
                "grading/execution_error": float(category == "execution"),
                "grading/result_limit": float(category == "result_limit"),
                "grading/recovered_timeout": float(
                    category != "timeout" and any(a.get("error_category") == "timeout" for a in attempts)
                ),
                **scored.metrics,
            }
        )
        traj.answer_signature = " ".join(str(self.record.grade.get("prediction") or "").split())
        if verdict is not None:
            traj.metrics.update(
                {
                    "verieql/refuted": float(verdict["verdict"] == "refuted"),
                    "verieql/unknown": float(verdict["verdict"] not in {"confirmed", "refuted"}),
                    "verieql/queue_seconds": verdict.get("queue_seconds", 0),
                    "verieql/service_seconds": verdict.get("service_seconds", 0),
                }
            )
        if verdict is not None and "counterexample_gate" in verdict:
            gate = verdict["counterexample_gate"]
            traj.metrics["verieql/gate_penalize"] = float(gate["decision"] == "penalize")
            traj.metrics["verieql/gate_abstain"] = float(gate["decision"] == "abstain")
            traj.metrics["verieql/gate_total_seconds"] = verdict["gate_total_seconds"]
        return self.record.reward

    async def on_truncated(self, traj, row, cause):
        budget_truncated = cause in guards.BUDGET_TRUNCATION_CAUSES
        scan = self._scan(traj, budget_truncated=budget_truncated)
        if cause in guards.LOOP_GUARD_TRUNCATION_CAUSES and self._apply_loop_guards(traj, scan, budget_truncated=True):
            return await self.reward(traj, row)
        self.record_turn(traj)
        if reward.commits_answer_on_truncation(cause, turns=traj.turns, max_turns=self.settings.max_turns):
            commit_final_answer(self.record, traj.last_stop_reason, parsing.message_text(traj.messages[-1]).strip())
        else:
            self.record.final_text = ""
            self.record.done = True
        return await self.reward(traj, row)

    async def snapshot(self, traj):
        return {"record": asdict(self.record), "cursor": self.cursor, "scanned": self.scanned}

    async def restore(self, traj, state):
        await self.reset(state["record"]["task"])
        self.record = Rollout(**state["record"])
        self.cursor = state["cursor"]
        self.scanned = state["scanned"]
