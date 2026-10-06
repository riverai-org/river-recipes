import json
import random
import re
import time
import unittest
from dataclasses import asdict
from pathlib import Path

from text_to_sql_recipe import parsing
from text_to_sql_recipe.parsing import ToolCall, ToolSpec, parse_tool_calls

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "parsing"
PRESETS = {
    "glm-5.3-flash",
    "kimi-k2.6",
    "qwen3.8-27b",
    "qwen3.6-35b-a3b",
    "deepseek-v4.1-flash",
    "nemotron-3.5-lightning",
}


def kimi_call(call_id: str, arguments: str, *, end: bool = True) -> str:
    return (
        f"<|tool_call_begin|>{call_id}<|tool_call_argument_begin|>{arguments}"
        + ("<|tool_call_end|>" if end else "")
    )


def kimi_section(*calls: str) -> str:
    return "<|tool_calls_section_begin|>" + "".join(calls) + "<|tool_calls_section_end|>"


def as_dicts(calls: list[ToolCall]) -> list[dict]:
    return [asdict(call) for call in calls]


class ToolCallParsingTest(unittest.TestCase):
    def test_openai_tool_calls_array(self) -> None:
        message = {
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "execute_sql_query",
                        "arguments": '{"query": "SELECT 1"}',
                    }
                }
            ],
        }
        calls = parse_tool_calls(message)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "execute_sql_query")
        self.assertEqual(calls[0].arguments, {"query": "SELECT 1"})

    def test_qwen_xml_block(self) -> None:
        message = {
            "content": 'Let me check.\n<tool_call>\n{"name": "execute_sql_query", '
            '"arguments": {"query": "SELECT 2"}}\n</tool_call>'
        }
        calls = parse_tool_calls(message)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments["query"], "SELECT 2")

    def test_malformed_block_is_ignored(self) -> None:
        message = {"content": "<tool_call>not json</tool_call>"}
        self.assertEqual(parse_tool_calls(message), [])

    def test_qwen_named_xml_form(self) -> None:
        # The named-XML form the Qwen models emit, and the one the demonstration shows.
        message = {
            "content": "I will check.\n<tool_call>\n<function=execute_sql_query>\n"
            "<parameter=query>SELECT COUNT(*) FROM animals;</parameter>\n"
            "</function>\n</tool_call>"
        }
        calls = parse_tool_calls(message)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "execute_sql_query")
        self.assertEqual(calls[0].arguments, {"query": "SELECT COUNT(*) FROM animals;"})

    def test_kimi_call_id_is_preserved_for_the_tool_observation(self) -> None:
        message = {
            "content": (
                "<|tool_calls_section_begin|><|tool_call_begin|>"
                "functions.execute_sql_query:7<|tool_call_argument_begin|>"
                '{"query":"SELECT 1"}<|tool_call_end|><|tool_calls_section_end|>'
            )
        }
        calls = parse_tool_calls(message)
        self.assertEqual(calls[0].call_id, "functions.execute_sql_query:7")


class HostileToolCallTextTest(unittest.TestCase):
    """Model text that once made parsing slow or crash: an unclosed block, deeply nested JSON."""

    def test_unclosed_blocks_parse_in_linear_time(self):
        for text in ("<tool_call>" + " " * 64_000, "<tool_call>" * 8_192, "x</tool_call><tool_call>" * 4_096):
            started = time.perf_counter()
            parsing.parse_tool_calls({"content": text})
            self.assertLess(time.perf_counter() - started, 1.0)

    def test_the_block_scan_finds_what_the_lazy_regex_finds(self):
        pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        pieces = ["<tool_call>", "</tool_call>", "a", " ", "\n", "<", ">", "tool_call"]
        generator = random.Random(7)
        for _ in range(5_000):
            text = "".join(generator.choice(pieces) for _ in range(generator.randrange(12)))
            self.assertEqual(parsing._tool_call_blocks(text), pattern.findall(text), text)

    def test_deeply_nested_json_is_an_unparseable_call(self):
        deep = "[" * 100_000
        self.assertEqual(parsing.parse_tool_calls({"content": f"<tool_call>{deep}</tool_call>"}), [])
        native = {"content": "", "tool_calls": [{"function": {"name": "execute_sql_query", "arguments": deep}}]}
        self.assertEqual(parsing.parse_tool_calls(native), [parsing.ToolCall(name="execute_sql_query", arguments={})])


class MessageTextTest(unittest.TestCase):
    """A chat API returns string content; a renderer returns a list of typed parts.

    Anything reading a message must accept both shapes, or it works on one path
    and raises on the other.
    """

    def test_string_content(self):
        self.assertEqual(parsing.message_text({"content": "hello"}), "hello")

    def test_list_content_is_joined(self):
        message = {
            "content": [
                {"type": "text", "text": "a"},
                {"type": "image", "url": "ignored"},
                {"type": "text", "text": "b"},
            ]
        }
        self.assertEqual(parsing.message_text(message), "ab")

    def test_missing_content(self):
        self.assertEqual(parsing.message_text({}), "")

    def test_tool_calls_parse_from_list_content(self):
        message = {
            "content": [
                {
                    "type": "text",
                    "text": "<tool_call>\n<function=execute_sql_query>"
                    "<parameter=query>SELECT 1</parameter></function>\n</tool_call>",
                }
            ]
        }
        calls = parsing.parse_tool_calls(message)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "execute_sql_query")
        self.assertEqual(calls[0].arguments, {"query": "SELECT 1"})

    def test_reasoning_parts_are_not_text(self):
        message = {
            "content": [
                {"type": "thinking", "thinking": "draft <solution>SELECT 0</solution>"},
                {"type": "text", "text": "<solution>SELECT 1</solution>"},
            ]
        }
        self.assertEqual(parsing.message_text(message), "<solution>SELECT 1</solution>")


class KimiToolCallTest(unittest.TestCase):
    """Kimi's marker-delimited tool-call section.

    The wire format is asserted literally because it is a protocol, not prose:
    if the markers drift the model stops being able to call tools at all, and a
    semantic assertion could not detect that.
    """

    SECTION = (
        "I will look at the schema first.\n"
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.execute_sql_query:0<|tool_call_argument_begin|>"
        '{"query": "SELECT 1"}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    def test_a_kimi_section_parses_to_one_call(self):
        calls = parsing.parse_tool_calls({"content": self.SECTION})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "execute_sql_query")
        self.assertEqual(calls[0].arguments, {"query": "SELECT 1"})

    def test_two_calls_in_one_section(self):
        text = self.SECTION.replace(
            "<|tool_calls_section_end|>",
            "<|tool_call_begin|>functions.execute_sql_query:1"
            '<|tool_call_argument_begin|>{"query": "SELECT 2"}<|tool_call_end|>'
            "<|tool_calls_section_end|>",
        )
        calls = parsing.parse_tool_calls({"content": text})
        self.assertEqual(
            [c.arguments["query"] for c in calls], ["SELECT 1", "SELECT 2"]
        )

    def test_a_bare_numeric_call_id_is_not_invented_into_a_name(self):
        text = (
            "<|tool_calls_section_begin|><|tool_call_begin|>0"
            '<|tool_call_argument_begin|>{"query": "SELECT 1"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        # No name is recoverable without the tool specs, and guessing one would
        # dispatch to an arbitrary tool. Better to report no call.
        self.assertEqual(parsing.parse_tool_calls({"content": text}), [])

    def test_malformed_json_arguments_yield_an_empty_dict_not_a_crash(self):
        text = self.SECTION.replace('{"query": "SELECT 1"}', "{not json")
        calls = parsing.parse_tool_calls({"content": text})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments, {})

    def test_qwen_xml_still_parses(self):
        """The Kimi branch must not displace the format that already worked."""
        text = (
            "<tool_call>\n<function=execute_sql_query>"
            "<parameter=query>SELECT 1</parameter></function>\n</tool_call>"
        )
        calls = parsing.parse_tool_calls({"content": text})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "execute_sql_query")

    def test_the_structured_field_still_wins(self):
        message = {
            "content": self.SECTION,
            "tool_calls": [{"function": {"name": "other", "arguments": "{}"}}],
        }
        self.assertEqual(parsing.parse_tool_calls(message)[0].name, "other")

    def test_text_with_no_section_is_no_calls(self):
        self.assertEqual(parsing.parse_tool_calls({"content": "just prose"}), [])


class KimiSectionGrammarTest(unittest.TestCase):
    """The first complete section is read with the Kimi renderer's grammar; the rest with a looser one.

    Expected values are golden: produced with the Kimi renderer importable, as it
    is during training. Cases marked "section grammar" return something else under
    the looser grammar alone.
    """

    def parse(self, text):
        return as_dicts(parsing.parse_kimi_tool_calls(text))

    def test_an_invalid_call_does_not_shadow_a_valid_one_in_the_section(self):
        # Section grammar: the looser one would return the invalid call first, with empty arguments.
        text = kimi_section(
            kimi_call("functions.execute_sql_query:0", "{bad"),
            kimi_call("functions.execute_sql_query:1", '{"query": "SELECT 2"}'),
        )
        self.assertEqual(
            self.parse(text),
            [{"name": "execute_sql_query", "arguments": {"query": "SELECT 2"},
              "call_id": "functions.execute_sql_query:1"}],
        )

    def test_a_name_and_arguments_object_names_the_call(self):
        # Section grammar: the looser one would name the call "other" and keep the whole object.
        text = kimi_section(
            kimi_call("functions.other:0", '{"name": "execute_sql_query", "arguments": {"query": "SELECT 3"}}')
        )
        self.assertEqual(
            self.parse(text),
            [{"name": "execute_sql_query", "arguments": {"query": "SELECT 3"}, "call_id": "functions.other:0"}],
        )

    def test_non_object_arguments_in_a_name_and_arguments_object_become_empty(self):
        # Section grammar: the looser one would name the call "x" and keep the whole object.
        text = kimi_section(
            kimi_call("functions.x:0", '{"name": "execute_sql_query", "arguments": "SELECT 12"}')
        )
        self.assertEqual(
            self.parse(text), [{"name": "execute_sql_query", "arguments": {}, "call_id": "functions.x:0"}]
        )

    def test_a_dotted_name_is_kept_whole(self):
        # Section grammar: the looser one would name the call "run".
        text = kimi_section(kimi_call("functions.sql.run:0", '{"query": "SELECT 4"}'))
        self.assertEqual(
            self.parse(text),
            [{"name": "sql.run", "arguments": {"query": "SELECT 4"}, "call_id": "functions.sql.run:0"}],
        )

    def test_calls_outside_the_section_are_ignored_when_the_section_has_one(self):
        # Section grammar: the looser one would return the call before the section first.
        text = kimi_call("functions.execute_sql_query:0", '{"query": "SELECT 10"}') + kimi_section(
            kimi_call("functions.execute_sql_query:1", '{"query": "SELECT 11"}')
        )
        self.assertEqual(
            self.parse(text),
            [{"name": "execute_sql_query", "arguments": {"query": "SELECT 11"},
              "call_id": "functions.execute_sql_query:1"}],
        )

    def test_a_section_without_its_end_marker_still_yields_its_call(self):
        text = "Query:<|tool_calls_section_begin|>" + kimi_call(
            "functions.execute_sql_query:0", '{"query": "SELECT 8"}'
        )
        self.assertEqual(
            self.parse(text),
            [{"name": "execute_sql_query", "arguments": {"query": "SELECT 8"},
              "call_id": "functions.execute_sql_query:0"}],
        )

    def test_a_call_cut_off_in_its_arguments_has_empty_arguments(self):
        text = "<|tool_calls_section_begin|>" + kimi_call(
            "functions.execute_sql_query:0", '{"query": "SELECT na', end=False
        )
        self.assertEqual(
            self.parse(text),
            [{"name": "execute_sql_query", "arguments": {}, "call_id": "functions.execute_sql_query:0"}],
        )

    def test_an_id_without_an_index_falls_back_to_its_name(self):
        text = kimi_section(kimi_call("functions.execute_sql_query", '{"query": "SELECT 5"}'))
        self.assertEqual(
            self.parse(text),
            [{"name": "execute_sql_query", "arguments": {"query": "SELECT 5"},
              "call_id": "functions.execute_sql_query"}],
        )

    def test_a_bare_id_is_skipped_but_a_named_call_after_it_is_kept(self):
        text = kimi_section(
            kimi_call("0", '{"query": "SELECT 6"}'),
            kimi_call("functions.execute_sql_query:1", '{"query": "SELECT 7"}'),
        )
        self.assertEqual(
            self.parse(text),
            [{"name": "execute_sql_query", "arguments": {"query": "SELECT 7"},
              "call_id": "functions.execute_sql_query:1"}],
        )

    def test_non_object_arguments_become_empty(self):
        text = kimi_section(kimi_call("functions.execute_sql_query:0", "[1, 2]"))
        self.assertEqual(
            self.parse(text),
            [{"name": "execute_sql_query", "arguments": {}, "call_id": "functions.execute_sql_query:0"}],
        )

    def test_an_empty_name_falls_back_to_the_id(self):
        text = kimi_section(
            kimi_call("functions.execute_sql_query:0", '{"name": "", "arguments": {"query": "SELECT 13"}}')
        )
        self.assertEqual(
            self.parse(text),
            [{"name": "execute_sql_query", "arguments": {"name": "", "arguments": {"query": "SELECT 13"}},
              "call_id": "functions.execute_sql_query:0"}],
        )

    def test_text_without_a_section_marker_has_no_kimi_calls(self):
        self.assertEqual(self.parse(kimi_call("functions.execute_sql_query:0", '{"query": "SELECT 1"}')), [])


class MalformedActionTest(unittest.TestCase):
    """A turn with call syntax but no parseable call is re-prompted, not graded as an answer."""

    def test_the_prompt_text_is_stable(self):
        # Model-visible text.
        self.assertEqual(
            parsing.MALFORMED_ACTION_PROMPT,
            "The tool call could not be parsed. Re-emit one valid tool call in the required format.",
        )

    def test_an_unparseable_tool_call_is_an_intent(self):
        message = {"content": "<tool_call>not valid JSON</tool_call>"}
        self.assertEqual(parsing.parse_tool_calls(message), [])
        self.assertTrue(parsing.has_tool_call_intent(message))

    def test_a_function_tag_without_a_block_is_an_intent(self):
        self.assertTrue(parsing.has_tool_call_intent({"content": "<function=execute_sql_query>"}))

    def test_structured_calls_are_an_intent(self):
        message = {"content": "", "tool_calls": [{"function": {"name": "list_tables", "arguments": "{}"}}]}
        self.assertTrue(parsing.has_tool_call_intent(message))

    def test_prose_is_not_an_intent(self):
        self.assertFalse(parsing.has_tool_call_intent({"content": "<solution>SELECT 1</solution>"}))

    def test_a_kimi_marker_counts_only_when_asked(self):
        message = {
            "content": "<|tool_calls_section_begin|>broken<|tool_calls_section_end|>\n"
            "<solution>SELECT 1</solution>"
        }
        self.assertEqual(parsing.parse_tool_calls(message), [])
        self.assertFalse(parsing.has_tool_call_intent(message))
        self.assertTrue(parsing.has_tool_call_intent(message, kimi_markers=True))

    def test_a_tool_call_is_found_beside_a_premature_solution(self):
        message = {
            "content": "<tool_call>\n<function=execute_sql_query>\n<parameter=query>SELECT 2</parameter>\n"
            "</function>\n</tool_call>\n<solution>SELECT 2</solution>"
        }
        self.assertEqual(as_dicts(parsing.parse_tool_calls(message)),
                         [{"name": "execute_sql_query", "arguments": {"query": "SELECT 2"}, "call_id": None}])


class ExecutionMessageTest(unittest.TestCase):
    REJECTED = (
        "<tool_call>\n<function=execute_sql_query>\n<parameter=query>SELECT 1</parameter>\n"
        "</function>\n</tool_call>"
    )

    def test_a_message_without_rejected_calls_is_returned_unchanged(self):
        message = {"role": "assistant", "content": [{"type": "text", "text": "a"}], "tool_calls": []}
        self.assertIs(parsing.execution_message(message), message)

    def test_rejected_call_text_follows_the_visible_text(self):
        message = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "reasoning"},
                {"type": "text", "text": "Before.\n"},
                {"type": "text", "text": "After."},
            ],
            "unparsed_tool_calls": [{"raw_text": self.REJECTED, "error": "Invalid GLM function name"}],
        }
        restored = parsing.execution_message(message)
        self.assertEqual(restored["content"], "Before.\nAfter." + self.REJECTED)
        self.assertEqual(restored["unparsed_tool_calls"], message["unparsed_tool_calls"])
        self.assertEqual(restored["role"], "assistant")
        self.assertIsInstance(message["content"], list, "the renderer's message is not modified")

    def test_a_restored_call_is_parsed(self):
        message = {
            "role": "assistant",
            "content": "Checking.\n",
            "unparsed_tool_calls": [{"raw_text": self.REJECTED, "error": "Invalid GLM function name"}],
        }
        self.assertEqual(parsing.parse_tool_calls(message), [])
        self.assertEqual(
            as_dicts(parsing.parse_tool_calls(parsing.execution_message(message))),
            [{"name": "execute_sql_query", "arguments": {"query": "SELECT 1"}, "call_id": None}],
        )

    def test_rejected_calls_are_restored_in_order(self):
        message = {
            "content": "",
            "unparsed_tool_calls": [{"raw_text": "<a>", "error": "x"}, {"raw_text": "<b>", "error": "y"}],
        }
        self.assertEqual(parsing.execution_message(message)["content"], "<a><b>")


class ToolSpecTest(unittest.TestCase):
    def test_defaults(self):
        spec = ToolSpec(name="execute_sql_query", schema={}, run=lambda arguments: [])
        self.assertEqual(spec.terminal_errors, ())
        self.assertIsNone(spec.context_error_max_bytes)
        self.assertEqual(spec.context_error_guidance, "")


class RendererTurnFixtureTest(unittest.TestCase):
    """Turns as each preset's renderer parses them, with the golden values read from them.

    `sampled_text` is the decoded turn; `message` is the renderer's parse of it;
    `expected` holds the golden results for that message.
    """

    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((FIXTURES / "renderer_turns.json").read_text())

    def test_every_preset_is_covered(self):
        self.assertEqual({case["preset"] for case in self.cases}, PRESETS)

    def test_turns_parse_to_their_golden_values(self):
        for case in self.cases:
            with self.subTest(preset=case["preset"], case=case["name"]):
                message, expected = case["message"], case["expected"]
                execution = parsing.execution_message(message)
                self.assertEqual(parsing.message_text(message), expected["text"])
                self.assertEqual(execution, expected["execution_message"])
                self.assertEqual(as_dicts(parsing.parse_tool_calls(execution)), expected["calls"])
                self.assertEqual(parsing.has_tool_call_intent(execution), expected["intent"])
                self.assertEqual(
                    parsing.has_tool_call_intent(message, kimi_markers=True), expected["intent_kimi_markers"]
                )
                self.assertEqual(parsing.message_text(execution).strip(), expected["answer"])


if __name__ == "__main__":
    unittest.main()
