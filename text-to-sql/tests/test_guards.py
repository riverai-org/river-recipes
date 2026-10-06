import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from text_to_sql_recipe import guards
from text_to_sql_recipe.config import GuardConfig
from text_to_sql_recipe.parsing import message_text

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "guards"


def decode(ids):
    """Stand-in tokenizer: one token per UTF-8 byte."""
    return bytes(ids).decode(errors="replace")


def parse(text):
    """Stand-in renderer parse: a turn that starts with "call" carries a tool call."""
    message = {"role": "assistant", "content": text}
    if text.startswith("call"):
        message["tool_calls"] = [
            {"type": "function", "id": "1", "function": {"name": "lookup", "arguments": '{"value": 3}'}}
        ]
    return message


def run_turn(text, *, reason="length", settings=None, cause=None, earlier=0, metrics=None, message=None):
    """Scan one turn and apply the guards the way the environment does.

    ``cause`` None is the ordinary turn path; otherwise the engine ended the
    rollout with that truncation cause. ``earlier`` generated tokens precede the turn.
    """
    settings = settings or GuardConfig()
    tokens = list(text.encode())
    metrics = {} if metrics is None else metrics
    scan = guards.scan_turn(
        text,
        tokens,
        metrics,
        settings,
        decode=decode,
        budget_truncated=cause in guards.BUDGET_TRUNCATION_CAUSES,
        last_stop_reason=reason,
    )
    stop = None
    if cause is None:
        stop = guards.apply_loop_guards(
            scan,
            settings,
            turn_tokens=tokens,
            message=parse(text) if message is None else message,
            decode=decode,
            generated_tokens=earlier + len(tokens),
            last_stop_reason=reason,
        )
    elif cause in guards.LOOP_GUARD_TRUNCATION_CAUSES:
        stop = guards.apply_loop_guards(
            scan,
            settings,
            turn_tokens=tokens,
            message=parse(text),
            decode=decode,
            generated_tokens=earlier + len(tokens),
            budget_truncated=True,
            last_stop_reason=reason,
        )
    if stop is not None:
        metrics.update(stop.metrics)
    return scan, stop, metrics


# --- Verbatim re-draft detection ------------------------------------------------------------------------------------

LOOP = (
    "<think>We need the Italian names.\n\n"
    "Draft:\n\nSELECT a FROM t WHERE x = 1\n\n"
    "Draft:\n\nSELECT a FROM t2 WHERE x = 1\n\n"
    "Draft:\n\nSELECT a FROM t WHERE x = 1\n\n"
    "Draft:\n\nSELECT a FROM t2 WHERE x = 1\n\n"
    "Draft:\n\n"
)


class DraftDetectionTests(unittest.TestCase):
    def test_finds_every_draft_and_the_first_verbatim_repeat(self):
        found = guards.drafts(LOOP)
        self.assertEqual(len(found), 5)
        self.assertEqual(found[0].normalized, "SELECT a FROM t WHERE x = 1")
        self.assertEqual(found[4].normalized, "")  # marker with no body
        self.assertEqual(guards.repeats(found), [2, 3])
        cut = guards.loop_cut(found)
        self.assertEqual(cut, found[2].body_end)
        # Everything through the first repeated draft is kept; the loop is not.
        self.assertTrue(LOOP[:cut].endswith("SELECT a FROM t WHERE x = 1"))
        self.assertEqual(LOOP[:cut].count("Draft:"), 3)

    def test_changed_drafts_are_revisions_not_repeats(self):
        text = "Draft:\n\nSELECT a FROM t\n\nHmm, wrong table.\n\nDraft:\n\nSELECT a FROM u\n"
        found = guards.drafts(text)
        self.assertEqual([d.normalized for d in found], ["SELECT a FROM t", "SELECT a FROM u"])
        self.assertEqual(guards.repeats(found), [])
        self.assertIsNone(guards.loop_cut(found))

    def test_whitespace_differences_do_not_break_identity(self):
        text = "Draft:\n\nSELECT a\n  FROM t\n\nDraft:\n\nSELECT   a FROM t\n"
        self.assertEqual(guards.repeats(guards.drafts(text)), [1])

    def test_fenced_drafts_and_commentary_between_drafts(self):
        text = (
            "Draft:\n\n```sql\nSELECT a\nFROM t\n```\n\nLet me check the join.\n\n"
            "Revised draft:\n\n```sql\nSELECT a\nFROM t\n```\n"
        )
        found = guards.drafts(text)
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0].normalized, "SELECT a FROM t")
        self.assertEqual(guards.repeats(found), [1])
        self.assertTrue(text[: guards.loop_cut(found)].endswith("```"))

    def test_text_without_markers_has_no_drafts(self):
        self.assertEqual(guards.drafts("SELECT a FROM t\n\nSELECT a FROM t\n"), [])
        self.assertEqual(guards.drafts(""), [])

    def test_empty_bodies_never_count_as_repeats(self):
        found = guards.drafts("Draft:\n\nDraft:\n\nDraft:\n")
        self.assertEqual([d.normalized for d in found], ["", "", ""])
        self.assertEqual(guards.repeats(found), [])


class TokenCutTests(unittest.TestCase):
    def test_smallest_prefix_covering_the_character_offset(self):
        text = "Draft:\n\nSELECT a FROM t\n\nDraft:\n\nSELECT a FROM t\n\nDraft:\n\n"
        tokens = list(text.encode())

        found = guards.drafts(text)
        cut = guards.loop_cut(found)
        keep = guards.token_cut(tokens, decode, cut)
        self.assertEqual(keep, cut)  # one byte per character here
        self.assertEqual(decode(tokens[:keep]), text[:cut])
        self.assertEqual(guards.token_cut(tokens, decode, 0), 0)
        self.assertEqual(guards.token_cut(tokens, decode, 10**6), len(tokens))

    def test_multibyte_tokens_round_up_to_cover_the_offset(self):
        # Two-character tokens: the cut must not split a token short of the offset.
        text = "abcdefgh"
        tokens = [0, 1, 2, 3]

        def pairs(ids):
            return "".join(text[2 * i : 2 * i + 2] for i in ids)

        self.assertEqual(guards.token_cut(tokens, pairs, 3), 2)
        self.assertEqual(guards.token_cut(tokens, pairs, 4), 2)
        self.assertEqual(guards.token_cut(tokens, pairs, 5), 3)


# --- Repetition detectors -------------------------------------------------------------------------------------------


def ids(text: str) -> list[int]:
    """A stand-in tokenizer: one id per whitespace-separated word (or punctuation run)."""
    vocab: dict[str, int] = {}
    return [vocab.setdefault(w, len(vocab)) for w in re.findall(r"\w+|[^\w\s]+", text)]


class PeriodicRunTests(unittest.TestCase):
    def test_single_word_loop(self):
        text = "Let me verify with SQL " + "thinking " * 60
        hit = guards.periodic_run(ids(text))
        self.assertIsNotNone(hit)
        start, period = hit
        self.assertEqual(period, 1)
        self.assertEqual(start, 5)  # right after the five prefix words

    def test_two_word_loop(self):
        text = "The check succeeded. " + "reserved correction " * 40
        start, period = guards.periodic_run(ids(text))
        self.assertEqual(period, 2)
        self.assertEqual(start, 4)

    def test_eight_token_period(self):
        text = "prefix words here " + "a b c d e f g h " * 12
        self.assertEqual(guards.periodic_run(ids(text))[1], 8)

    def test_below_threshold_is_silent(self):
        self.assertIsNone(guards.periodic_run(ids("wait " * 20)))  # 20 tokens < 48
        self.assertIsNone(guards.periodic_run(ids("previous " * 47)))

    def test_legitimate_sql_and_prose(self):
        union = " UNION ALL ".join(
            f"SELECT {i} AS id, 'name{i}' AS name FROM t{i} WHERE x > {i}" for i in range(12)
        )
        in_list = "SELECT * FROM t WHERE id IN (" + ", ".join(str(i) for i in range(300, 380)) + ")"
        case = (
            "SELECT "
            + ", ".join(f"SUM(CASE WHEN col = {i} THEN amount ELSE 0 END) AS c{i}" for i in range(20))
            + " FROM t"
        )
        table = (
            "| " + " | ".join(f"col{i}" for i in range(10)) + " |\n|" + "---|" * 10 + "\n"
            + "\n".join("| " + " | ".join(str(r * 10 + c) for c in range(10)) + " |" for r in range(30))
        )
        prose = ("First I need to check the schema. The customers table has id, name and country. Then the orders "
                 "table links by customer_id. The question asks for the average order value per country, so I group "
                 "by country and average the amount. Let me confirm the join key and the amount column with a quick "
                 "query. ") * 6
        for text in (union, in_list, case, table, prose):
            self.assertIsNone(guards.periodic_run(ids(text)), text[:60])


class SpanLoopTests(unittest.TestCase):
    def test_sentence_loop(self):
        text = "Now the final answer. " + (
            "The previous reserved correction response was reserved correction response after correction? " * 6
        )
        hit = guards.span_loop(text)
        self.assertIsNotNone(hit)
        start, first_end = hit
        self.assertEqual(start, len("Now the final answer. "))
        self.assertGreater(first_end, start + 40)

    def test_three_copies_are_silent(self):
        text = "Header. " + ("A perfectly ordinary sentence that just happens to repeat itself. " * 3)
        self.assertIsNone(guards.span_loop(text))

    def test_legitimate_repeated_structures(self):
        union = "\n".join(
            f"SELECT customer_id, SUM(amount) AS total FROM orders_{i} GROUP BY customer_id" for i in range(6)
        )
        schema = "\n".join(
            f"CREATE TABLE t{i} (\n  id INTEGER PRIMARY KEY,\n  name TEXT,\n  value REAL\n);" for i in range(8)
        )
        steps = "".join(
            f"Step {i}: run the query, inspect the rows, adjust the filter if the count is off.\n" for i in range(8)
        )
        for text in (union, schema, steps):
            self.assertIsNone(guards.span_loop(text), text[:60])


class DetectRepetitionTests(unittest.TestCase):
    SENTENCE = "The previous reserved correction response was reserved correction response after correction? "

    def test_without_a_decoder_only_token_runs_are_found(self):
        text = "x" * 60
        self.assertEqual(guards.detect_repetition(list(text.encode()), text, None), (1, 1, 0))
        text = self.SENTENCE * 6
        self.assertIsNone(guards.detect_repetition([1, 2, 3], text, None))

    def test_the_earliest_loop_wins_even_when_a_later_one_would_cut_sooner(self):
        # A sentence loop from offset 0 whose first copy holds a two-token run starting at 8: the first copy is kept.
        text = ("Answer: " + "ab" * 30 + ".") * 4
        tokens = list(text.encode())
        self.assertEqual(guards.periodic_run(tokens), (8, 2))
        self.assertEqual(guards.span_loop(text), (0, 69))
        self.assertEqual(guards.detect_repetition(tokens, text, decode), (2, 69, 69))

    def test_settings_reach_the_detectors(self):
        loop = "x" * 60
        self.assertIsNotNone(run_turn(loop)[0].repetition)
        self.assertIsNone(run_turn(loop, settings=GuardConfig(repetition_min_tokens=100))[0].repetition)
        two = "Header. " + self.SENTENCE * 2
        self.assertIsNone(run_turn(two)[0].repetition)
        self.assertEqual(run_turn(two, settings=GuardConfig(repetition_span_repeats=2))[0].repetition[0], 2)
        word_loop = "ab" * 40
        self.assertEqual(run_turn(word_loop)[0].repetition, (1, 2, 2))
        self.assertIsNone(run_turn(word_loop, settings=GuardConfig(repetition_max_period=1))[0].repetition)


# --- The guards on a turn -------------------------------------------------------------------------------------------


class RepetitionGuardTests(unittest.TestCase):
    """Repetition masking on cap-hit turns (the environment's turn and truncation paths)."""

    SENTENCE = "The previous reserved correction response was reserved correction response after correction? "

    def test_earlier_sentence_loop_wins_over_later_word_loop(self):
        text = self.SENTENCE * 6 + "x" * 100
        _, stop, metrics = run_turn(text)
        expected = guards.span_loop(text)[1]
        self.assertEqual(stop.loss_mask_end, expected)
        self.assertEqual(metrics["repetition_kind"], 2)
        # One token per character: exactly the first copy keeps its advantage.
        self.assertEqual(stop.keep_tokens, expected)
        self.assertEqual(metrics["repetition_masked_tokens"], len(text) - expected)

    def test_earlier_token_loop_wins_over_later_sentence_loop(self):
        text = "x" * 100 + self.SENTENCE * 6
        _, stop, metrics = run_turn(text)
        self.assertEqual(stop.loss_mask_end, 1)
        self.assertEqual(metrics["repetition_kind"], 1)

    def test_budget_truncation_runs_guard_even_with_stale_stop_reason(self):
        for cause in ("context", "generated_tokens"):
            with self.subTest(cause=cause):
                _, stop, metrics = run_turn("x" * 60, reason="stop", cause=cause)
                self.assertEqual(stop.loss_mask_end, 1)
                self.assertEqual(stop.keep_tokens, 1)
                self.assertEqual(metrics["cap_hit"], 1.0)

    def test_budget_truncation_guard_off_and_legitimate_output_keep_loss(self):
        for enabled, text in (
            (False, "x" * 60),
            (True, "Consider each column and inspect its datatype."),
        ):
            with self.subTest(enabled=enabled):
                settings = GuardConfig(repetition_guard=enabled)
                _, stop, _ = run_turn(text, reason="stop", cause="context", settings=settings)
                self.assertIsNone(stop)

    def test_budget_truncation_preserves_no_tool_call_restriction(self):
        # The engine stores the raw text after a budget truncation, so the guard parses it again.
        _, stop, metrics = run_turn("call " + "x" * 60, cause="context")
        self.assertIsNone(stop)
        self.assertEqual(metrics["repetition"], 1.0)  # still detected and recorded

    def test_completed_tool_calls_at_turn_limit_are_not_token_cap_hits(self):
        for cause in ("turns", "final_answer"):
            with self.subTest(cause=cause):
                _, stop, metrics = run_turn('call {"query":"SELECT 1"}', reason="stop", cause=cause)
                self.assertIsNone(stop)
                self.assertNotIn("cap_hit", metrics)
                self.assertNotIn("repetition", metrics)

    def test_non_cap_turns_are_only_observed(self):
        _, stop, metrics = run_turn("x" * 60, reason="stop")
        self.assertIsNone(stop)
        self.assertNotIn("repetition", metrics)
        self.assertNotIn("cap_hit", metrics)

    def test_truncation_causes(self):
        self.assertEqual(guards.BUDGET_TRUNCATION_CAUSES, {"context", "generated_tokens"})
        self.assertEqual(
            guards.LOOP_GUARD_TRUNCATION_CAUSES, {"context", "generated_tokens", "turns", "final_answer"}
        )


class LoopStopTests(unittest.TestCase):
    """The guard on a cap-hit turn that repeats a draft verbatim."""

    LOOP = (
        "<think>plan\n\nDraft:\n\nSELECT a FROM t\n\nDraft:\n\nSELECT b FROM t\n\n"
        "Draft:\n\nSELECT a FROM t\n\nDraft:\n\nSELECT b FROM t\n\nDraft:\n\n"
    )
    ONLY_DRAFTS = GuardConfig(loop_stop=True, repetition_guard=False)

    def test_cap_hit_repeat_masks_the_loop_and_ends_with_no_answer(self):
        _scan, stop, metrics = run_turn(self.LOOP, settings=self.ONLY_DRAFTS)
        cut = guards.loop_cut(guards.drafts(self.LOOP))
        self.assertEqual(stop.kind, "draft")
        self.assertEqual(stop.loss_mask_end, cut)  # byte tokens: one per character
        self.assertEqual(stop.char_end, cut)
        self.assertEqual(metrics["loop"], 1.0)
        self.assertEqual(metrics["loop_stopped"], 1.0)
        self.assertEqual(metrics["redrafts"], 2.0)
        self.assertEqual(metrics["drafts"], 5.0)
        self.assertEqual(metrics["loop_masked_tokens"], len(self.LOOP) - cut)
        # The kept prefix ends with the first repeated draft, the loop is context.
        self.assertTrue(self.LOOP[:cut].endswith("SELECT a FROM t"))

    def test_guard_off_only_records_metrics(self):
        scan, stop, metrics = run_turn(self.LOOP, settings=GuardConfig(loop_stop=False, repetition_guard=False))
        self.assertEqual(len(scan.drafts), 5)
        self.assertEqual(scan.repeated, [2, 3])
        self.assertEqual(metrics["loop"], 1.0)
        self.assertIsNone(stop)

    def test_metrics_accumulate_over_turns(self):
        metrics = {}
        run_turn(self.LOOP, reason="stop", metrics=metrics, settings=self.ONLY_DRAFTS)
        self.assertEqual((metrics["drafts"], metrics["redrafts"]), (5.0, 2.0))
        self.assertNotIn("loop", metrics)  # repeats on a turn that ended normally are counted, not a loop
        second = "Draft:\n\nSELECT 1\n\nDraft:\n\nSELECT 1\n"
        run_turn(second, reason="stop", metrics=metrics, settings=self.ONLY_DRAFTS)
        self.assertEqual((metrics["drafts"], metrics["redrafts"]), (5.0, 3.0))

    def test_the_mask_offset_counts_earlier_turns(self):
        _, stop, _ = run_turn(self.LOOP, earlier=40, settings=self.ONLY_DRAFTS)
        self.assertEqual(stop.loss_mask_end, 40 + stop.keep_tokens)

    def test_a_tool_call_turn_is_never_cut(self):
        message = {**parse(self.LOOP), "tool_calls": [{"function": {"name": "execute_sql_query", "arguments": "{}"}}]}
        _, stop, metrics = run_turn(self.LOOP, settings=self.ONLY_DRAFTS, message=message)
        self.assertIsNone(stop)
        self.assertEqual(metrics["loop"], 1.0)

    def test_the_answer_is_the_text_before_the_cut(self):
        text = "Draft:\n\nSELECT a FROM t\n\nDraft:\n\nSELECT a FROM t\n\nDraft:\n\nSELECT a FROM t\n\n"
        _, stop, _ = run_turn(text, settings=self.ONLY_DRAFTS)
        self.assertEqual(message_text(parse(text[: stop.char_end])).strip(), text[: stop.char_end].strip())
        self.assertTrue(text[: stop.char_end].endswith("SELECT a FROM t"))


class CutChoiceTests(unittest.TestCase):
    TEXT = LOOP

    def stop_with(self, repetition, settings=None):
        settings = GuardConfig() if settings is None else settings
        found = guards.drafts(self.TEXT)
        scan = guards.TurnScan(found, guards.repeats(found), repetition)
        tokens = list(self.TEXT.encode())
        return guards.apply_loop_guards(
            scan, settings, turn_tokens=tokens, message={}, decode=decode,
            generated_tokens=len(tokens), last_stop_reason="length",
        )

    def test_the_earlier_cut_wins_and_a_tie_goes_to_the_draft(self):
        cut = guards.loop_cut(guards.drafts(self.TEXT))
        self.assertEqual(self.stop_with((1, cut, cut)).kind, "draft")
        self.assertEqual(self.stop_with((1, cut - 1, cut - 1)).kind, "repetition")
        self.assertEqual(self.stop_with((2, cut + 1, cut + 1)).kind, "draft")

    def test_each_guard_has_its_own_switch(self):
        cut = guards.loop_cut(guards.drafts(self.TEXT))
        early = (1, cut - 1, cut - 1)
        self.assertEqual(self.stop_with(early, GuardConfig(repetition_guard=False)).kind, "draft")
        self.assertEqual(self.stop_with(early, GuardConfig(loop_stop=False)).kind, "repetition")
        self.assertIsNone(self.stop_with(early, GuardConfig(loop_stop=False, repetition_guard=False)))

    def test_a_repetition_cut_is_clamped_to_the_turn(self):
        tokens = [120] * 10
        stop = guards.repetition_loop_stop((1, 10**6, 5), turn_tokens=tokens, generated_tokens=25)
        self.assertEqual((stop.keep_tokens, stop.loss_mask_end), (10, 25))
        self.assertEqual(stop.metrics["repetition_masked_tokens"], 0.0)

    def test_no_decoder_or_no_tokens_means_no_guard(self):
        found = guards.drafts(self.TEXT)
        scan = guards.TurnScan(found, guards.repeats(found), None)
        for tokens, decoder in ((list(self.TEXT.encode()), None), ([], decode)):
            self.assertIsNone(
                guards.apply_loop_guards(
                    scan, GuardConfig(), turn_tokens=tokens, message={}, decode=decoder,
                    generated_tokens=len(tokens), last_stop_reason="length",
                )
            )


class GoldenTurnTest(unittest.TestCase):
    """Golden scores of synthetic turns (two stand-in tokenizers)."""

    def test_every_case_is_reproduced(self):
        cases = json.loads((FIXTURES / "golden_turns.json").read_text())["cases"]
        kinds = {case["expected"]["kind"] for case in cases}
        self.assertEqual(kinds, {None, "draft", "repetition"})
        for index, case in enumerate(cases):
            with self.subTest(case=index):
                if case["tokenizer"] == "bytes":
                    tokens, decoder = list(case["text"].encode()), decode
                else:
                    vocab = case["vocab"]
                    tokens = case["tokens"]

                    def decoder(ids, vocab=vocab):
                        return "".join(vocab[i] for i in ids)

                settings = replace(GuardConfig(), **case["settings"])
                metrics = dict(case["prior_metrics"])
                path = case["path"]
                scan = guards.scan_turn(
                    case["text"], tokens, metrics, settings, decode=decoder,
                    budget_truncated=path in guards.BUDGET_TRUNCATION_CAUSES,
                    last_stop_reason=case["last_stop_reason"],
                )
                if path == "on_turn":
                    message = parse(case["text"])
                    if case["message_tool_calls"]:
                        message["tool_calls"] = [{"function": {"name": "execute_sql_query", "arguments": "{}"}}]
                    stop = guards.apply_loop_guards(
                        scan, settings, turn_tokens=tokens, message=message, decode=decoder,
                        generated_tokens=case["earlier_generated_tokens"] + len(tokens),
                        last_stop_reason=case["last_stop_reason"],
                    )
                else:
                    stop = guards.apply_loop_guards(
                        scan, settings, turn_tokens=tokens, message=parse(case["text"]), decode=decoder,
                        generated_tokens=case["earlier_generated_tokens"] + len(tokens),
                        budget_truncated=True, last_stop_reason=case["last_stop_reason"],
                    )
                if stop is not None:
                    metrics.update(stop.metrics)
                expected = case["expected"]
                self.assertEqual([[d.marker_start, d.body_end, d.normalized] for d in scan.drafts], expected["drafts"])
                self.assertEqual(scan.repeated, expected["repeated"])
                self.assertEqual(list(scan.repetition) if scan.repetition else None, expected["repetition"])
                self.assertEqual(stop is not None, expected["stopped"])
                self.assertEqual(stop.kind if stop else None, expected["kind"])
                self.assertEqual(stop.loss_mask_end if stop else None, expected["loss_mask_end"])
                self.assertEqual(metrics, expected["metrics"])
                if stop is not None:
                    answer = message_text(parse(case["text"][: stop.char_end])).strip()
                    self.assertEqual(answer, expected["answer"])


# --- Batch monitors -------------------------------------------------------------------------------------------------


def trajectory(correct, answer, **metrics):
    return SimpleNamespace(metrics={"correct": correct, **metrics}, answer_signature=answer)


class GuardMetricsTests(unittest.TestCase):
    KEYS = frozenset({
        "guard/loop_rate", "guard/loop_stopped_rate", "guard/cap_hit_rate", "guard/drafts_mean",
        "guard/redrafts_mean", "guard/redraft_penalty_mean", "guard/process_penalty_mean",
        "guard/process_violations_mean", "guard/repetition_rate", "guard/repetition_stopped_rate",
        "guard/repetition_masked_tokens_mean", "sampled/accuracy",
    })
    DIVERSITY = frozenset({
        "diversity/distinct_answers_mean", "diversity/vote_correct_rate", "diversity/pass_rate",
        "diversity/vote_lift",
    })

    def batch(self):
        return [
            trajectory(1.0, "x", loop=1.0, cap_hit=1.0),
            trajectory(1.0, "x"),
            trajectory(0.0, "y"),
            trajectory(1.0, "x"),
            trajectory(0.0, "", loop=0.0),
            trajectory(0.0, "y"),
            trajectory(0.0, "z"),
            trajectory(1.0, "w"),
        ]

    def test_guard_metrics_summarise_rates_and_diversity(self):
        metrics = guards.guard_metrics(self.batch(), 4)
        self.assertEqual(set(metrics), self.KEYS | self.DIVERSITY)
        self.assertEqual(metrics["guard/loop_rate"], 1 / 8)
        self.assertEqual(metrics["guard/cap_hit_rate"], 1 / 8)
        self.assertEqual(metrics["sampled/accuracy"], 4 / 8)
        # Group 1 answers x, x, y, x; group 2 answers nothing, y, z, w.
        self.assertEqual(metrics["diversity/distinct_answers_mean"], 2.5)
        # Majority x is correct in group 1; the three-way tie in group 2 goes to the earliest answer, y (wrong).
        self.assertEqual(metrics["diversity/vote_correct_rate"], 0.5)
        self.assertEqual(metrics["diversity/pass_rate"], 1.0)
        self.assertAlmostEqual(metrics["diversity/vote_lift"], 0.5 - 0.5)

    def test_every_member_counts(self):
        metrics = guards.guard_metrics(self.batch()[:4], 4)
        self.assertEqual(metrics["sampled/accuracy"], 3 / 4)
        self.assertEqual(metrics["diversity/vote_correct_rate"], 1.0)

    def test_groups_must_tile_the_batch(self):
        self.assertEqual(set(guards.guard_metrics(self.batch(), 3)), self.KEYS)
        self.assertEqual(set(guards.guard_metrics(self.batch())), self.KEYS)
        self.assertEqual(guards.guard_metrics([], 4), {})

    def test_a_group_without_answers_votes_wrong(self):
        batch = [trajectory(0.0, ""), SimpleNamespace(metrics={"correct": 0.0})]
        metrics = guards.guard_metrics(batch, 2)
        self.assertEqual(metrics["diversity/distinct_answers_mean"], 0.0)
        self.assertEqual(metrics["diversity/vote_correct_rate"], 0.0)


class CollapsePauseTests(unittest.TestCase):
    BAD = [SimpleNamespace(metrics={"cap_hit": 1.0, "repetition": 1.0})] * 4
    CAP_ONLY = [SimpleNamespace(metrics={"cap_hit": 1.0})] * 4
    GOOD = [SimpleNamespace(metrics={})] * 4

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name) / "run"

    def tearDown(self):
        self.folder.cleanup()

    def pause(self, **settings):
        return guards.CollapsePause(self.root, replace(GuardConfig(), **settings))

    def state(self):
        return json.loads((self.root / "collapse-state.json").read_text())

    def test_consecutive_bad_steps_pause_and_block_every_restart(self):
        pause = self.pause(collapse_arm_after_steps=0)
        metrics = {}
        pause.check(1, self.BAD, metrics)
        self.assertEqual(metrics, {"guard/pause_cap_rate": 1.0, "guard/pause_repetition_rate": 1.0,
                                   "guard/pause_armed": 1.0})
        self.assertFalse(pause.marker.exists())
        with self.assertRaisesRegex(RuntimeError, "Training paused"):
            pause.check(2, self.BAD, {})
        marker = json.loads(pause.marker.read_text())
        self.assertEqual((marker["batch"], marker["streak"], marker["armed"]), (2, 2, True))
        with self.assertRaisesRegex(RuntimeError, "Training paused"):
            self.pause(collapse_arm_after_steps=0)
        with self.assertRaisesRegex(RuntimeError, "Training paused"):
            pause.check(3, self.GOOD, {})

    def test_no_pause_before_it_is_armed(self):
        pause = self.pause(collapse_arm_after_steps=3)
        for step in (1, 2, 3):
            metrics = {}
            pause.check(step, self.BAD, metrics)
            self.assertEqual(metrics["guard/pause_armed"], 0.0)
            self.assertEqual((self.state()["streak"], self.state()["armed"]), (step, False))
        self.assertFalse(pause.marker.exists())
        # The streak was tracked while disarmed, so the first armed bad step pauses at once.
        with self.assertRaisesRegex(RuntimeError, "Training paused"):
            pause.check(4, self.BAD, {})

    def test_default_arms_after_one_epoch(self):
        pause = self.pause()
        self.assertEqual(GuardConfig().collapse_arm_after_steps, 33)
        for step in range(1, 34):
            pause.check(step, self.BAD, {})
        self.assertFalse(pause.marker.exists())
        with self.assertRaisesRegex(RuntimeError, "Training paused"):
            pause.check(34, self.BAD, {})

    def test_both_rates_must_exceed_their_limits(self):
        pause = self.pause(collapse_arm_after_steps=0)
        pause.check(1, self.CAP_ONLY, {})
        pause.check(2, self.CAP_ONLY, {})
        self.assertEqual(self.state()["streak"], 0)
        # Strictly above: exactly at the limit is not bad.
        at_limit = [SimpleNamespace(metrics={"cap_hit": 1.0, "repetition": 1.0})] + [self.GOOD[0]] * 19
        metrics = {}
        self.pause(collapse_arm_after_steps=0, collapse_cap_rate=0.05).check(3, at_limit, metrics)
        self.assertEqual(metrics["guard/pause_cap_rate"], 0.05)
        self.assertEqual(self.state()["streak"], 0)

    def test_a_good_or_skipped_step_resets_the_streak(self):
        pause = self.pause(collapse_arm_after_steps=0, collapse_consecutive_steps=3)
        pause.check(1, self.BAD, {})
        pause.check(2, self.GOOD, {})
        self.assertEqual(self.state()["streak"], 0)
        pause.check(3, self.BAD, {})
        pause.check(5, self.BAD, {})  # step 4 missing: the streak restarts
        self.assertEqual(self.state()["streak"], 1)

    def test_rechecking_a_step_starts_from_the_streak_before_it(self):
        pause = self.pause(collapse_arm_after_steps=0, collapse_consecutive_steps=3)
        pause.check(1, self.BAD, {})
        pause.check(2, self.BAD, {})
        self.assertEqual((self.state()["previous_streak"], self.state()["streak"]), (1, 2))
        pause.check(2, self.GOOD, {})
        self.assertEqual((self.state()["previous_streak"], self.state()["streak"]), (1, 0))
        pause.check(2, self.BAD, {})
        self.assertEqual(self.state()["streak"], 2)
        self.assertFalse(pause.marker.exists())

    def test_state_ahead_of_the_trainer_is_refused(self):
        pause = self.pause(collapse_arm_after_steps=0)
        pause.check(5, self.GOOD, {})
        with self.assertRaisesRegex(RuntimeError, "ahead of trainer"):
            pause.check(4, self.GOOD, {})

    def test_an_empty_step_records_nothing(self):
        pause = self.pause(collapse_arm_after_steps=0)
        metrics = {}
        pause.check(1, [], metrics)
        self.assertEqual(metrics, {})
        self.assertFalse(pause.state_path.exists())

    def test_state_record(self):
        pause = self.pause(collapse_arm_after_steps=0)
        pause.check(1, self.BAD[:1] + self.GOOD[:3], {})
        state = self.state()
        self.assertEqual(
            set(state), {"batch", "previous_streak", "armed", "streak", "cap_rate", "repetition_rate", "time"}
        )
        self.assertEqual((state["cap_rate"], state["repetition_rate"]), (0.25, 0.25))
        self.assertFalse(pause.state_path.with_suffix(".tmp").exists())


if __name__ == "__main__":
    unittest.main()
