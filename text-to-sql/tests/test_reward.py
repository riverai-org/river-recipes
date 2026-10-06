import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

from text_to_sql_recipe import reward
from text_to_sql_recipe.config import RewardConfig
from text_to_sql_recipe.parsing import execution_message

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "reward"


# --- Outcome --------------------------------------------------------------------------------------------------------


class RewardTest(unittest.TestCase):
    """The reward scale, and why its range is load-bearing."""

    def reward(self, verdict, *, unsafe=False, category=None, refuted=False, weight=0.2):
        grade = {"grade": verdict, "error_category": category}
        return reward.reward_for(grade, unsafe=unsafe, refuted=refuted, verifier_weight=weight)

    def test_correct_and_incorrect_match_the_paper(self):
        self.assertEqual(self.reward("CORRECT"), 1.0)
        self.assertEqual(self.reward("INCORRECT"), 0.0)

    def test_no_answer_scores_minus_one(self):
        """A rollout that never produced a solution has to score below a wrong answer."""
        self.assertEqual(reward.NO_ANSWER_REWARD, -1.0)
        self.assertEqual(reward.UNSAFE_REWARD, -1.0)
        self.assertEqual(self.reward("NO_ANSWER"), -1.0)

    def test_unsafe_is_penalised(self):
        self.assertEqual(self.reward("INCORRECT", unsafe=True), -1.0)
        self.assertEqual(self.reward("INCORRECT", category="unsafe_sql"), -1.0)

    def test_unsafe_wins_over_a_correct_result(self):
        """A query can return the right rows and still be refused."""
        self.assertEqual(self.reward("CORRECT", unsafe=True), -1.0)
        self.assertEqual(self.reward("CORRECT", unsafe=True, refuted=True), -1.0)

    def test_verifier_refutation_receives_the_downweight(self):
        self.assertEqual(self.reward("CORRECT", refuted=True, weight=0.2), 0.8)
        self.assertEqual(self.reward("CORRECT", refuted=True, weight=0.0), 1.0)

    def test_nonrefuting_verifier_outcomes_keep_full_reward(self):
        self.assertEqual(self.reward("CORRECT", refuted=False), 1.0)

    def test_a_refutation_only_affects_a_matching_answer(self):
        self.assertEqual(self.reward("INCORRECT", refuted=True), 0.0)
        self.assertEqual(self.reward("NO_ANSWER", refuted=True), -1.0)

    def test_negative_rewards_inflate_every_advantage_in_the_group(self):
        """Why the no-answer value is not a cosmetic choice.

        Advantages are reward minus the group mean, so widening the reward
        range scales the whole group's gradient, not just the failing
        rollout's. One correct sample among seven no-answers:
        """

        def advantages(no_answer_reward):
            rewards = [1.0] + [no_answer_reward] * 7
            mean = sum(rewards) / len(rewards)
            return [r - mean for r in rewards]

        paper = advantages(0.0)
        shipped = advantages(reward.NO_ANSWER_REWARD)
        self.assertAlmostEqual(paper[0], 0.875)
        self.assertAlmostEqual(shipped[0], 1.75)


# --- Process rules --------------------------------------------------------------------------------------------------


class ProcessPenaltyTest(unittest.TestCase):
    """The paper's two evidence-supervision rules (App. E.5).

    Rules are string-level checks against the indexed formats the
    evidence-structured prompt names, so the markers are asserted literally:
    paraphrasing the prompt would silently break the reward.
    """

    REQ = "Requirement of external knowledge 1 (Austen refers to author = 'Jane Austen'): ..."
    VER = "Verification of external knowledge 1 (Austen refers to author = 'Jane Austen'): ..."

    def penalty(self, turns, evidence="Austen refers to author = 'Jane Austen'", weight=0.1):
        messages = [{"role": "assistant", "content": t} for t in turns]
        task = {"id": "t", "evidence": evidence}
        return reward.process_penalty(task, reward.assistant_turns(messages), weight=weight)

    def test_both_blocks_present_no_penalty(self):
        total, v = self.penalty([self.REQ, f"{self.VER}\n<solution>SELECT 1</solution>"])
        self.assertEqual(total, 0.0)
        self.assertFalse(v["requirement"])
        self.assertFalse(v["verification"])

    def test_missing_requirement_costs_lambda(self):
        total, v = self.penalty(["thinking", f"{self.VER}\n<solution>SELECT 1</solution>"])
        self.assertAlmostEqual(total, 0.1)
        self.assertTrue(v["requirement"])

    def test_missing_verification_costs_lambda(self):
        total, v = self.penalty([self.REQ, "<solution>SELECT 1</solution>"])
        self.assertAlmostEqual(total, 0.1)
        self.assertTrue(v["verification"])

    def test_both_missing_costs_the_maximum(self):
        total, _ = self.penalty(["thinking", "<solution>SELECT 1</solution>"])
        self.assertAlmostEqual(total, 0.2)

    def test_requirement_may_appear_on_the_second_turn(self):
        total, _v = self.penalty(["exploring", self.REQ, f"{self.VER}\n<solution>SELECT 1</solution>"])
        self.assertEqual(total, 0.0)

    def test_requirement_on_the_third_turn_is_too_late(self):
        total, v = self.penalty(
            ["exploring", "more exploring", f"{self.REQ}\n{self.VER}\n<solution>S</solution>"]
        )
        self.assertAlmostEqual(total, 0.1)
        self.assertTrue(v["requirement"])

    def test_first_turn_answer_is_relaxed_either_block_suffices(self):
        """Easy problems needing no exploration: either block satisfies both."""
        total, _ = self.penalty([f"{self.REQ}\n<solution>SELECT 1</solution>"])
        self.assertEqual(total, 0.0)
        total, _ = self.penalty([f"{self.VER}\n<solution>SELECT 1</solution>"])
        self.assertEqual(total, 0.0)

    def test_first_turn_answer_with_neither_block_costs_only_one_lambda(self):
        total, _ = self.penalty(["<solution>SELECT 1</solution>"])
        self.assertAlmostEqual(total, 0.1)

    def test_no_external_knowledge_requires_neither_block(self):
        total, v = self.penalty(["<solution>SELECT 1</solution>"], evidence="")
        self.assertEqual(total, 0.0)
        self.assertFalse(v["applicable"])

    def test_requirement_penalty_applies_without_a_solution(self):
        """Checked before the model commits, so it applies either way."""
        total, v = self.penalty(["thinking", "still thinking"])
        self.assertAlmostEqual(total, 0.1)
        self.assertTrue(v["requirement"])
        self.assertFalse(v["verification"])  # never committed a solution

    def test_the_weight_scales_the_penalty_and_zero_keeps_the_record(self):
        total, _ = self.penalty(["thinking", "<solution>SELECT 1</solution>"], weight=0.25)
        self.assertAlmostEqual(total, 0.5)
        total, v = self.penalty(["thinking", "<solution>SELECT 1</solution>"], weight=0.0)
        self.assertEqual(total, 0.0)
        self.assertTrue(v["requirement"] and v["verification"])

    def test_the_prompt_is_not_the_policy(self):
        """The demonstration's assistant turn does not satisfy, or count as, the policy's first turn."""
        messages = [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "demo question"},
            {"role": "assistant", "content": self.REQ},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": f"{self.VER}\n<solution>SELECT 1</solution>"},
        ]
        task = {"evidence": "Austen refers to author = 'Jane Austen'"}
        turns = reward.assistant_turns(messages, prompt_message_count=4)
        self.assertEqual(turns, [f"{self.VER}\n<solution>SELECT 1</solution>"])
        self.assertEqual(reward.process_penalty(task, turns, weight=0.1)[0], 0.0)  # first-turn relaxation
        # Counted from the start, the demonstration would be the policy's first turn.
        self.assertEqual(reward.process_violations(task, reward.assistant_turns(messages))["requirement"], False)
        self.assertEqual(len(reward.assistant_turns(messages)), 2)

    def test_only_text_parts_are_the_turn(self):
        turn = [{"type": "thinking", "thinking": self.REQ}, {"type": "text", "text": "<solution>SELECT 1</solution>"}]
        messages = [{"role": "assistant", "content": "exploring"}, {"role": "assistant", "content": turn}]
        turns = reward.assistant_turns(messages)
        self.assertEqual(turns, ["exploring", "<solution>SELECT 1</solution>"])


# --- Truncated rollouts -------------------------------------------------------------------------------------------


class TruncatedRolloutTest(unittest.TestCase):
    def test_only_the_answer_turn_ending_on_the_turn_limit_commits_an_answer(self):
        self.assertTrue(reward.commits_answer_on_truncation("turns", turns=6, max_turns=5))
        self.assertFalse(reward.commits_answer_on_truncation("turns", turns=5, max_turns=5))
        for cause in ("context", "generated_tokens", "final_answer", "empty_sample"):
            with self.subTest(cause=cause):
                self.assertFalse(reward.commits_answer_on_truncation(cause, turns=6, max_turns=5))


# --- Re-drafts ------------------------------------------------------------------------------------------------------


class RedraftPenaltyTest(unittest.TestCase):
    def apply(self, value, redrafts, *, loop_stopped=False, penalty=0.05, cap=0.2):
        return reward.apply_redraft_penalty(
            value, redrafts=redrafts, loop_stopped=loop_stopped, penalty=penalty, cap=cap
        )

    def test_each_verbatim_redraft_costs_the_penalty_up_to_the_cap(self):
        self.assertEqual(self.apply(1.0, 2.0), (1.0 - 0.1, 0.1))
        self.assertEqual(self.apply(1.0, 10.0), (1.0 - 0.2, 0.2))
        self.assertEqual(self.apply(0.8, 1.0), (0.8 - 0.05, 0.05))

    def test_the_floor_is_the_no_answer_reward(self):
        value, amount = self.apply(-0.9, 4.0)
        self.assertEqual(value, -1.0)
        self.assertEqual(amount, 0.2)  # the nominal deduction is recorded
        for floor_or_below in (-1.0, -1.2):
            self.assertEqual(self.apply(floor_or_below, 4.0), (floor_or_below, None))

    def test_a_loop_stopped_rollout_is_not_charged_again(self):
        self.assertEqual(self.apply(-1.0, 2.0, loop_stopped=True), (-1.0, None))
        self.assertEqual(self.apply(0.0, 2.0, loop_stopped=True), (0.0, None))

    def test_off_without_a_penalty_or_a_redraft(self):
        self.assertEqual(self.apply(1.0, 0.0), (1.0, None))
        self.assertEqual(self.apply(1.0, 3.0, penalty=0.0), (1.0, None))

    def test_a_zero_cap_still_records_the_term(self):
        self.assertEqual(self.apply(1.0, 3.0, cap=0.0), (1.0, 0.0))


# --- Tool-call format -----------------------------------------------------------------------------------------------


def native(args, name="execute_sql_query"):
    return {"tool_calls": [{"id": "call1", "function": {"name": name, "arguments": args}}]}


def xml(query="SELECT 1"):
    return {
        "content": "<tool_call><function=execute_sql_query><parameter=query>"
        + query
        + "</parameter></function></tool_call>"
    }


class ToolProtocolTests(unittest.TestCase):
    """Behavioral contract for the malformed tool-call cost."""

    def test_observed_missing_delimiter_and_empty_args(self):
        for raw_message in [
            xml()["content"].replace("query>", "query"),
            native("{}"),
            native({"query": ""}),
            native({"query": "  "}),
            native({"query": None}),
        ]:
            message = {"content": raw_message} if isinstance(raw_message, str) else raw_message
            with self.subTest(message=message):
                valid, errors = reward.inspect_calls(message)
                self.assertFalse(valid)
                self.assertEqual(len(errors), 1)

    def test_valid_formats_and_sql_errors_are_opaque(self):
        for query in [
            "SELECT 1",
            "SELECT missing_column FROM missing_table",
            "SELECT 1 WHERE 0",
            "SELECT '<parameter=query>'",
            "SELEC invalid",
        ]:
            for message in [
                xml(query),
                native({"query": query}),
                native('{"query": "' + query + '"}'),
                {
                    "content": '<tool_call>{"name":"execute_sql_query","arguments":{"query":"'
                    + query
                    + '"}}</tool_call>'
                },
            ]:
                valid, errors = reward.inspect_calls(message)
                self.assertEqual(len(valid), 1)
                self.assertFalse(errors)
        self.assertEqual(reward.inspect_calls(native({"query": "SELECT 1"}))[0][0].call_id, "call1")

    def test_invalid_protocol_forms(self):
        for message in [
            native("{broken"),
            native("[]"),
            native("null"),
            native(7),
            native({"query": 7}),
            native({"query": "SELECT 1"}, "unknown"),
            {"tool_calls": [{}]},
            {"tool_calls": "oops"},
            {"content": "<tool_call>{not json}</tool_call>"},
            {"content": "<tool_call><function=execute_sql_query>"},
        ]:
            with self.subTest(message=message):
                valid, errors = reward.inspect_calls(message)
                self.assertFalse(valid)
                self.assertEqual(len(errors), 1)

    def test_mixed_calls_preserve_valid_and_count_invalid_once(self):
        msg = native({"query": "SELECT 1"})
        msg["tool_calls"] += native({})["tool_calls"]
        valid, errors = reward.inspect_calls(msg)
        self.assertEqual(len(valid), 1)
        self.assertEqual(errors, ["missing_query"])

    def test_final_turn_calls_but_not_normal_answers(self):
        valid, errors = reward.inspect_calls(xml(), tools_allowed=False)
        self.assertFalse(valid)
        self.assertEqual(errors, ["tool_after_limit"])
        for message in [
            {"content": "<solution>SELECT 1</solution>"},
            {"content": "ordinary prose"},
            {"reasoning_content": "<tool_call>broken", "content": "<solution>SELECT 1</solution>"},
        ]:
            self.assertEqual(reward.inspect_calls(message, tools_allowed=False), ([], []))

    def test_a_kimi_section_marker_without_a_call_is_an_attempt(self):
        valid, errors = reward.inspect_calls({"content": "<|tool_calls_section_begin|> nothing"})
        self.assertEqual((valid, errors), ([], ["invalid_envelope"]))

    def test_snapshot_and_same_turn_replay(self):
        events = []
        reward.observe_tool_calls(events, native({}), turn=1, tools_allowed=True)
        events = json.loads(json.dumps(events))  # saved and restored with the rollout
        reward.observe_tool_calls(events, native({}), turn=1, tools_allowed=True)
        reward.observe_tool_calls(events, xml(), turn=2, tools_allowed=True)
        value, receipt, metrics = reward.apply_tool_format_penalty(1, events, per_call=0.05, cap=0.2)
        self.assertEqual(metrics["tool_protocol/invalid_calls"], 1)
        self.assertEqual(metrics["tool_protocol/valid_calls"], 1)
        self.assertAlmostEqual(value, 0.95)
        self.assertEqual(receipt["errors"], {"missing_query": 1})

    def test_cap_floor_disabled_and_other_rewards(self):
        events = [{"turn": 1, "valid_calls": 0, "errors": ["missing_query"] * 10}]
        for base, expected in [
            (1, 0.8),
            (0.4, 0.2),
            (0, -0.2),
            (-0.95, -1),
            (-1, -1),
            (-1.2, -1.2),
        ]:
            value, receipt, _ = reward.apply_tool_format_penalty(base, events, per_call=0.05, cap=0.2)
            self.assertAlmostEqual(value, expected)
            self.assertAlmostEqual(receipt["penalty"], base - expected)
        value, receipt, metrics = reward.apply_tool_format_penalty(1, events, per_call=0.0, cap=0.2)
        self.assertEqual(value, 1)
        self.assertEqual(metrics["tool_protocol/penalty"], 0.0)
        self.assertEqual(metrics["tool_protocol/invalid_calls"], 10.0)  # still observed

    def test_group_centering_signal_and_uniform_penalty_cancellation(self):
        rewards = []
        for errors in [1] + [0] * 15:
            events = [{"turn": 1, "valid_calls": 0, "errors": ["missing_query"] * errors}]
            rewards.append(reward.apply_tool_format_penalty(1, events, per_call=0.05, cap=0.2)[0])
        mean = sum(rewards) / 16
        self.assertLess(rewards[0] - mean, 0)
        self.assertTrue(all(r - mean > 0 for r in rewards[1:]))

    def test_receipt_and_metrics(self):
        events = [
            {"turn": 1, "valid_calls": 1, "errors": []},
            {"turn": 2, "valid_calls": 0, "errors": ["missing_query", "unknown_tool"]},
        ]
        value, receipt, metrics = reward.apply_tool_format_penalty(0.5, events, per_call=0.05, cap=0.2)
        self.assertEqual(
            receipt,
            {"policy": "tool-protocol-v1", "invalid_calls": 2, "errors": {"missing_query": 1, "unknown_tool": 1},
             "penalty_per_call": 0.05, "penalty_cap": 0.2, "reward_before": 0.5, "penalty": 0.1, "reward_after": value},
        )
        self.assertEqual(
            set(metrics),
            {"tool_protocol/invalid_calls", "tool_protocol/invalid_trajectory", "tool_protocol/valid_calls",
             "tool_protocol/penalty"} | {f"tool_protocol/{key}" for key in reward.TOOL_PROTOCOL_ERRORS},
        )
        self.assertEqual(metrics, {**reward.tool_protocol_metrics(events), "tool_protocol/penalty": 0.1})
        self.assertEqual(metrics["tool_protocol/invalid_trajectory"], 1.0)
        self.assertEqual(metrics["tool_protocol/valid_calls"], 1.0)


class TurnProtocolObservationTest(unittest.TestCase):
    """How the environment's turns are observed: the turn as the tool-call parser reads it."""

    def test_malformed_call_then_a_correct_answer(self):
        raw = (
            "<tool_call><function=execute_sql_query><parameter=querySELECT n FROM t</parameter></function></tool_call>"
        )
        events = []
        valid, errors = reward.observe_tool_calls(events, {"content": raw}, turn=1, tools_allowed=True)
        self.assertEqual((valid, errors), ([], ["missing_query"]))
        result = rollout({"grade": "CORRECT", "format_compliant": True}, tool_protocol_events=events,
                         settings=RewardConfig(tool_format_penalty=0.05))
        self.assertAlmostEqual(result.reward, 0.95)
        self.assertEqual(result.metrics["tool_protocol/invalid_calls"], 1)
        self.assertEqual(result.grade["tool_protocol"]["errors"], {"missing_query": 1})

    def test_exploratory_sql_errors_have_no_format_cost(self):
        for query in ["SELECT n FROM t WHERE 0", "SELECT nonexistent FROM t"]:
            events = []
            arguments = json.dumps({"query": query})
            message = {"tool_calls": [{"function": {"name": "execute_sql_query", "arguments": arguments}}]}
            reward.observe_tool_calls(events, message, turn=1, tools_allowed=True)
            result = rollout(
                {"grade": "CORRECT"}, tool_protocol_events=events, settings=RewardConfig(tool_format_penalty=0.05)
            )
            self.assertEqual(result.reward, 1)
            self.assertEqual(result.metrics["tool_protocol/penalty"], 0)

    def test_answer_turn_tool_call_is_observed_but_the_floor_holds(self):
        events = []
        message = {"tool_calls": [{"function": {"name": "execute_sql_query", "arguments": '{"query":"SELECT 7"}'}}]}
        reward.observe_tool_calls(events, message, turn=6, tools_allowed=False)
        self.assertEqual(events[-1]["errors"], ["tool_after_limit"])
        result = rollout(
            {"grade": "NO_ANSWER"}, tool_protocol_events=events, settings=RewardConfig(tool_format_penalty=0.05)
        )
        self.assertEqual(result.reward, -1)
        self.assertEqual(result.grade["tool_protocol"]["penalty"], 0.0)

    def test_renderer_rejected_calls_are_checked_through_the_execution_message(self):
        raw = "<tool_call><function=execute_sql_query><parameter=querySELECT 7</parameter></function></tool_call>"
        message = {"content": "", "unparsed_tool_calls": [{"raw_text": raw, "error": "rejected"}]}
        self.assertEqual(reward.inspect_calls(execution_message(message)), ([], ["missing_query"]))
        # The renderer message alone no longer holds the rejected syntax.
        self.assertEqual(reward.inspect_calls(message), ([], []))


# --- The rollout's reward -------------------------------------------------------------------------------------------


def rollout(grade, **overrides):
    """`rollout_reward` with neutral defaults: no evidence, no turns, no re-drafts, no tool events."""
    arguments = {
        "settings": RewardConfig(),
        "task": {"id": "t", "evidence": ""},
        "messages": [],
        "prompt_message_count": 0,
        "unsafe": False,
        "refuted": False,
        "redrafts": 0.0,
        "loop_stopped": False,
        "tool_protocol_events": [],
    }
    arguments.update(overrides)
    settings = arguments.pop("settings")
    return reward.rollout_reward(grade, settings, **arguments)


class RolloutRewardTest(unittest.TestCase):
    EVIDENCE: ClassVar[dict] = {"id": "t", "evidence": "Austen refers to author = 'Jane Austen'"}
    TURNS: ClassVar[list] = [
        {"role": "assistant", "content": "exploring"},
        {"role": "assistant", "content": "<solution>SELECT 1</solution>"},
    ]  # breaks both process rules

    def test_terms_apply_in_order(self):
        events = [{"turn": 1, "valid_calls": 0, "errors": ["empty_query"]}]
        result = rollout(
            {"grade": "CORRECT"},
            settings=RewardConfig(tool_format_penalty=0.05),
            task=self.EVIDENCE,
            messages=self.TURNS,
            refuted=True,
            redrafts=1.0,
            tool_protocol_events=events,
        )
        outcome = 1.0 - 0.2
        expected = ((outcome - 0.1 * 2) - 0.05) - 0.05
        self.assertEqual(result.reward, expected)
        self.assertEqual(result.grade["tool_protocol"]["reward_before"], (outcome - 0.1 * 2) - 0.05)
        self.assertEqual(result.grade["redraft_penalty"], 0.05)
        self.assertEqual(result.grade["process_penalty"], 0.1 * 2)
        self.assertEqual(
            result.grade["process_violations"], {"requirement": True, "verification": True, "applicable": True}
        )
        self.assertEqual(
            {k: result.metrics[k] for k in ("redraft_penalty", "process_penalty", "process_violations",
                                            "tool_protocol/penalty")},
            {"redraft_penalty": 0.05, "process_penalty": 0.1 * 2, "process_violations": 2.0,
             "tool_protocol/penalty": 0.05},
        )

    def test_no_answer_is_left_at_the_floor(self):
        events = [{"turn": 1, "valid_calls": 0, "errors": ["empty_query"]}]
        result = rollout({"grade": "NO_ANSWER"}, settings=RewardConfig(tool_format_penalty=0.05), redrafts=3.0,
                         tool_protocol_events=events)
        self.assertEqual(result.reward, -1.0)
        self.assertNotIn("redraft_penalty", result.grade)
        self.assertEqual(result.metrics["redraft_penalty"], 0.0)
        self.assertEqual(result.grade["tool_protocol"]["penalty"], 0.0)

    def test_process_penalty_can_go_below_the_floor_and_stays_there(self):
        result = rollout({"grade": "INCORRECT"}, unsafe=True, task=self.EVIDENCE, messages=self.TURNS, redrafts=2.0,
                         tool_protocol_events=[{"turn": 1, "valid_calls": 0, "errors": ["unknown_tool"]}],
                         settings=RewardConfig(tool_format_penalty=0.05))
        self.assertAlmostEqual(result.reward, -1.2)
        self.assertNotIn("redraft_penalty", result.grade)
        self.assertEqual(result.grade["tool_protocol"]["penalty"], 0.0)

    def test_a_loop_stopped_rollout_pays_only_the_no_answer_reward(self):
        result = rollout({"grade": "NO_ANSWER"}, redrafts=2.0, loop_stopped=True)
        self.assertEqual(result.reward, -1.0)
        self.assertNotIn("redraft_penalty", result.grade)

    def test_evaluation_settings_score_the_outcome_only(self):
        """Evaluation turns every shaping weight off; the terms are still recorded."""
        evaluation = replace(RewardConfig(tool_format_penalty=0.05), process_penalty_weight=0.0, verifier_weight=0.0,
                             redraft_penalty=0.0, tool_format_penalty=0.0)
        result = rollout({"grade": "CORRECT"}, settings=evaluation, task=self.EVIDENCE, messages=self.TURNS,
                         refuted=True, redrafts=2.0,
                         tool_protocol_events=[{"turn": 1, "valid_calls": 0, "errors": ["empty_query"]}])
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.metrics["process_violations"], 2.0)
        self.assertEqual(result.metrics["tool_protocol/invalid_calls"], 1.0)

    def test_inputs_are_not_modified(self):
        grade = {"grade": "CORRECT"}
        events = [{"turn": 1, "valid_calls": 0, "errors": ["empty_query"]}]
        messages = copy.deepcopy(self.TURNS)
        rollout(grade, task=self.EVIDENCE, messages=messages, tool_protocol_events=events, redrafts=1.0)
        self.assertEqual(grade, {"grade": "CORRECT"})
        self.assertEqual(events, [{"turn": 1, "valid_calls": 0, "errors": ["empty_query"]}])
        self.assertEqual(messages, self.TURNS)

    def test_metric_keys(self):
        self.assertEqual(
            set(rollout({"grade": "CORRECT"}).metrics),
            {"redraft_penalty", "process_penalty", "process_violations", "tool_protocol/penalty",
             "tool_protocol/invalid_calls", "tool_protocol/invalid_trajectory", "tool_protocol/valid_calls"}
            | {f"tool_protocol/{key}" for key in reward.TOOL_PROTOCOL_ERRORS},
        )


class GoldenRewardTest(unittest.TestCase):
    """Golden rewards of synthetic rollouts."""

    def test_every_case_is_reproduced_exactly(self):
        cases = json.loads((FIXTURES / "golden_rewards.json").read_text())["cases"]
        self.assertGreaterEqual(len(cases), 200)
        for index, case in enumerate(cases):
            with self.subTest(case=index):
                result = reward.rollout_reward(
                    case["grade"],
                    RewardConfig(**case["settings"]),
                    task=case["task"],
                    messages=case["messages"],
                    prompt_message_count=case["prompt_message_count"],
                    unsafe=case["unsafe"],
                    refuted=case["refuted"],
                    redrafts=case["redrafts"],
                    loop_stopped=case["loop_stopped"],
                    tool_protocol_events=case["tool_protocol_events"],
                )
                expected = case["expected"]
                self.assertEqual(float(result.reward).hex(), expected["reward_hex"])
                self.assertEqual(result.grade, expected["grade"])
                self.assertEqual(result.metrics, expected["metrics"])


if __name__ == "__main__":
    unittest.main()
