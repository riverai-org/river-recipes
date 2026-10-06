"""What the model sees, checked byte for byte against expected outputs.

`fixtures/prompts/opening_messages.json` holds expected opening messages for representative tasks: evaluation rows
with and without evidence, rows staged at each curriculum level (evidence withheld, shown, shown with a training
hint), rescues re-staged from an already staged row, rows placed in a lane by the evidence-mixture scheduler, and the
SQL tool schema under several limit settings. Staged rows are stored as inputs (`staged_row`, `scheduled_row`,
`rescued_row`, with the stage under "assistance_curriculum"); this module renders them and re-composes their task
text from `compose_problem` and `TRAINING_HINT_PREFIX`.
"""
import json
import unittest
from pathlib import Path

from text_to_sql_recipe import prompts

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "prompts" / "opening_messages.json").read_text(encoding="utf-8")
)
STAGE = "assistance_curriculum"


def staged_rows():
    """Every staged row in the fixture with the hint it was staged with: (label, row, hint)."""
    for case in FIXTURE["cases"]:
        if case["staged_row"] is not None:
            yield case["name"], case["staged_row"], FIXTURE["hints"][case["row"]]
    for item in FIXTURE["mixture"]:
        yield f"{item['lane']} lane", item["scheduled_row"], FIXTURE["hints"][item["row"]]
        yield f"{item['lane']} lane, hint rescue", item["rescued_row"], FIXTURE["hints"][item["row"]]


class OpeningMessagesParityTest(unittest.TestCase):
    def test_cases_match_expected_messages(self):
        self.assertGreaterEqual(len(FIXTURE["cases"]), 12)
        for case in FIXTURE["cases"]:
            with self.subTest(case=case["name"]):
                row = case["staged_row"] or FIXTURE["rows"][case["row"]]
                messages = prompts.opening_messages(row, few_shot=case["few_shot"])
                self.assertEqual(messages, case["messages"])
                self.assertEqual(len(messages), case["prompt_message_count"])
                self.assertEqual(prompts.external_knowledge_pieces(row), case["evidence_pieces"])

    def test_scheduled_and_rescued_rows_match_expected_messages(self):
        self.assertEqual({item["lane"] for item in FIXTURE["mixture"]}, {"none", "trusted"})
        for item in FIXTURE["mixture"]:
            with self.subTest(row=item["row"], lane=item["lane"]):
                self.assertEqual(prompts.opening_messages(item["scheduled_row"]), item["scheduled_messages"])
                self.assertEqual(prompts.opening_messages(item["rescued_row"]), item["rescued_messages"])

    def test_staged_text_is_composed_from_the_shared_pieces(self):
        levels = set()
        for label, row, hint in staged_rows():
            with self.subTest(row=label):
                stage = row[STAGE]
                original = stage["original"]
                shown = original["evidence"] if stage["level"] else ""
                text = prompts.compose_problem(original["schema"], evidence=shown, question=row["question"])
                if stage["level"] == 2:
                    text += prompts.TRAINING_HINT_PREFIX + hint
                self.assertEqual(row["problem"], text)
                self.assertEqual(row["evidence"], shown)
                levels.add(stage["level"])
        self.assertEqual(levels, {0, 1, 2})

    def test_stored_text_is_the_composition_of_its_fields(self):
        for label, row, _ in staged_rows():
            with self.subTest(row=label):
                original = row[STAGE]["original"]
                stored = next(r for r in FIXTURE["rows"].values() if r["id"] == row["id"])
                self.assertEqual(
                    prompts.compose_problem(original["schema"], evidence=original["evidence"],
                                            question=stored["question"]),
                    stored["problem"],
                )

    def test_tool_schema_matches(self):
        # Every preset runs the executor's defaults: 50 rows, 1,000,000 bytes, no further limits.
        schema = prompts.sql_tool_schema({"sql_timeout": 30, "sql_tool_max_rows": 50})
        self.assertEqual(schema, {"type": "function", "function": FIXTURE["tool_schema_function"]})
        self.assertEqual(prompts.sql_tool_schema({}), schema)
        for item in FIXTURE["tool_descriptions"]:
            with self.subTest(cfg=item["cfg"]):
                self.assertEqual(prompts.sql_tool_max_rows(item["cfg"]), item["max_rows"])
                self.assertEqual(prompts.sql_tool_description(item["cfg"]), item["description"])
                self.assertEqual(prompts.sql_tool_schema(item["cfg"]), item["schema"])

    def test_protocol_text_matches(self):
        text = FIXTURE["text"]
        self.assertEqual(prompts.system_prompt(), text["system_prompt"])
        self.assertEqual(prompts.few_shot_messages(), text["few_shot_messages"])
        self.assertEqual(prompts.tool_result_prompt(), text["tool_result_prompt"])
        for name in (
            "REVISQL_SYSTEM_PROMPT",
            "EVIDENCE_DIRECTIVES",
            "REQUIREMENT_MARKER",
            "VERIFICATION_MARKER",
            "SQL_LIMIT_PROMPT",
            "SQL_TOOL_RESULT_PROMPT",
            "SQL_TOOL_CONTEXT_ERROR_MAX_BYTES",
            "SQL_TOOL_CONTEXT_ERROR_GUIDANCE",
            "SQL_TOOL_NAME",
        ):
            with self.subTest(name=name):
                self.assertEqual(getattr(prompts, name), text[name])


class SystemPromptTest(unittest.TestCase):
    def test_semicolon_split_matches_what_the_prompt_promises(self):
        pieces = prompts.external_knowledge_pieces({"evidence": "a means x; b means y ; "})
        self.assertEqual(pieces, ["a means x", "b means y"])
        self.assertEqual(prompts.external_knowledge_pieces({"evidence": ""}), [])

    def test_evidence_prompt_extends_rather_than_replaces_the_base(self):
        base = prompts.REVISQL_SYSTEM_PROMPT
        ev = prompts.system_prompt()
        self.assertTrue(ev.startswith(base.rstrip()))
        self.assertIn(prompts.REQUIREMENT_MARKER, ev)
        self.assertIn(prompts.VERIFICATION_MARKER, ev)
        self.assertNotIn(prompts.REQUIREMENT_MARKER, base)

    def test_markers_are_the_formats_the_directives_name(self):
        for marker in (prompts.REQUIREMENT_MARKER, prompts.VERIFICATION_MARKER):
            self.assertIn(f'"{marker} {{i}} ({{raw external knowledge text}}): ...".', prompts.EVIDENCE_DIRECTIVES)

    def test_demonstration_is_a_fresh_copy(self):
        first = prompts.few_shot_messages()
        first[0]["content"] = "changed"
        self.assertNotEqual(prompts.few_shot_messages()[0]["content"], "changed")

    def test_demonstration_calls_the_sql_tool_and_answers_in_a_solution_block(self):
        roles = [m["role"] for m in prompts.few_shot_messages()]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant"])
        call, result, answer = (m["content"] for m in prompts.few_shot_messages()[1:])
        self.assertIn(f"<function={prompts.SQL_TOOL_NAME}>", call)
        self.assertTrue(result.endswith("\n\n" + prompts.tool_result_prompt()))
        self.assertIn("<solution>", answer)


class TaskTextTest(unittest.TestCase):
    def row(self, evidence="a means x; b means y"):
        return {
            "id": "q",
            "question": "Which item?",
            "evidence": evidence,
            "problem": prompts.compose_problem("CREATE TABLE t(n INTEGER);", evidence=evidence, question="Which item?"),
        }

    def test_sections_are_stripped_and_empty_evidence_reads_none(self):
        text = prompts.compose_problem("\n  CREATE TABLE t(n INTEGER);\n", evidence="  ", question=" Which item?\n")
        self.assertEqual(
            text,
            "Database Engine:\nSQLite\n\nDatabase Schema:\nCREATE TABLE t(n INTEGER);\n\n"
            "External Knowledge:\n(none)\n\nQuestion:\nWhich item?",
        )

    def test_hint_follows_the_task_text_under_its_header(self):
        self.assertEqual(prompts.TRAINING_HINT_PREFIX, "\n\nTraining hint:\n")
        hinted = self.row()["problem"] + prompts.TRAINING_HINT_PREFIX + "Use t."
        self.assertTrue(hinted.endswith("Question:\nWhich item?\n\nTraining hint:\nUse t."))

    def test_task_text_is_the_last_message_unchanged(self):
        row = {**self.row(), "problem": "  " + self.row()["problem"] + "\n\n"}   # passed through, not stripped
        for few_shot in (True, False):
            messages = prompts.opening_messages(row, few_shot=few_shot)
            self.assertEqual(messages[0], {"role": "system", "content": prompts.system_prompt()})
            self.assertEqual(messages[-1], {"role": "user", "content": row["problem"]})
            self.assertEqual(messages[1:-1], prompts.few_shot_messages() if few_shot else [])

    def test_messages_are_independent_copies(self):
        row = self.row()
        first = prompts.opening_messages(row)
        first[1]["content"] = "changed"
        self.assertEqual(prompts.opening_messages(row)[1], prompts.few_shot_messages()[0])


class SqlToolLimitTest(unittest.TestCase):
    def test_defaults_name_only_the_row_cap(self):
        self.assertEqual(prompts.sql_tool_max_rows({}), 50)
        self.assertEqual(
            prompts.sql_tool_description({"sql_tool_max_bytes": 1_000_000}),
            "Execute one read-only SQLite SELECT query against the task database. Results are capped at 50 rows.",
        )

    def test_phrases_join_in_a_fixed_order(self):
        self.assertEqual(prompts._join_limit_phrases(["a"]), "a")
        self.assertEqual(prompts._join_limit_phrases(["a", "b"]), "a or b")
        self.assertEqual(prompts._join_limit_phrases(["a", "b", "c"]), "a, b, or c")

    def test_malformed_limits_fail_in_the_shared_resolver(self):
        with self.assertRaises(TypeError):
            prompts.sql_tool_max_rows({"sql_tool_max_rows": True})
        for value in (0, -5, "many", 2.5j):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prompts.sql_tool_description({"sql_tool_max_cells": value})
        self.assertEqual(prompts._positive_sql_tool_limit({"sql_tool_max_cells": "7"}, "sql_tool_max_cells"), 7)
        self.assertIsNone(prompts._positive_sql_tool_limit({}, "sql_tool_max_cells"))


if __name__ == "__main__":
    unittest.main()
