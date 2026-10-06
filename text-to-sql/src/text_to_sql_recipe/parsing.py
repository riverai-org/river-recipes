"""Read a model turn: its visible text, its tool calls, and whether it tried to call a tool.

A turn arrives as the message a chat-template renderer parsed from the sampled text, a plain dict:

    {
        "role": "assistant",
        "content": str, or a list of typed parts such as {"type": "text", "text": ...} and
                   {"type": "thinking", "thinking": ...},
        "tool_calls": [                                   # optional: calls the renderer parsed
            {"type": "function", "id": str | None,
             "function": {"name": str, "arguments": str (a JSON object) | dict}},
        ],
        "unparsed_tool_calls": [                          # optional: call syntax the renderer rejected
            {"raw_text": str, "error": str},
        ],
    }

The turn's text (`message_text`) is string content as it is, or, for a list, its text parts joined,
which leaves reasoning parts out.

`parse_tool_calls` returns the calls from the first of these sources that has any:

1. the structured `tool_calls` field, which carries every call in the model's native syntax that its
   renderer could parse;
2. a Kimi tool-call section in the text (`<|tool_calls_section_begin|>` ...);
3. `<tool_call>` blocks in the text, either in the named-XML form
   (`<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`) or holding
   a JSON object `{"name": ..., "arguments": {...}}`.

The text forms cover call syntax a renderer rejected, for example a model whose native syntax differs
writing the named-XML form shown in the demonstration. `execution_message` puts that rejected syntax
back into the text before the turn is parsed. A turn with call syntax but no parseable call is a
malformed action (`has_tool_call_intent`): the environment asks for one valid call again
(`MALFORMED_ACTION_PROMPT`) instead of grading the turn as an answer.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_TOOL_CALL_OPEN, _TOOL_CALL_CLOSE = "<tool_call>", "</tool_call>"


def _tool_call_blocks(text: str) -> list[str]:
    """The bodies of the complete `<tool_call>` blocks, in order: the matches of `<tool_call>(.*?)</tool_call>`.

    A scan rather than a regex, so text full of unclosed openers costs one pass: once an opener has no closer after
    it, no later one can.
    """
    blocks, start = [], 0
    while (opener := text.find(_TOOL_CALL_OPEN, start)) >= 0:
        body = opener + len(_TOOL_CALL_OPEN)
        closer = text.find(_TOOL_CALL_CLOSE, body)
        if closer < 0:
            break
        blocks.append(text[body:closer])
        start = closer + len(_TOOL_CALL_CLOSE)
    return blocks
_NAMED_FUNCTION_RE = re.compile(
    r"<function=([A-Za-z_]\w*)\s*>(.*?)</function>", re.DOTALL
)
_PARAMETER_RE = re.compile(r"<parameter=([A-Za-z_]\w*)\s*>(.*?)</parameter>", re.DOTALL)

# Kimi K2.6 writes its tool calls as a marker-delimited section in the text:
# <|tool_calls_section_begin|><|tool_call_begin|>functions.<name>:<index>
# <|tool_call_argument_begin|>{json}<|tool_call_end|>...<|tool_calls_section_end|>
KIMI_TOOL_SECTION_BEGIN = "<|tool_calls_section_begin|>"
KIMI_TOOL_SECTION_END = "<|tool_calls_section_end|>"
# The first complete section, with its body captured.
_KIMI_SECTION_BODY_RE = re.compile(
    re.escape(KIMI_TOOL_SECTION_BEGIN) + r"(.*?)" + re.escape(KIMI_TOOL_SECTION_END),
    re.DOTALL,
)
# One call; it may also end where the next begins or at the end of the text.
_KIMI_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>(.*?)<\|tool_call_argument_begin\|>(.*?)"
    r"(?:<\|tool_call_end\|>|(?=<\|tool_call_begin\|>)|$)",
    re.DOTALL,
)
# Inside a complete section a call id must be `[functions.]<name>:<index>`.
_KIMI_SECTION_CALL_ID_RE = re.compile(r"^(?:functions\.)?(?P<name>[\w.\-]+):(?P<index>\d+)$")
# Elsewhere the name is the id's trailing identifier; bare numeric ids name no tool.
_KIMI_CALL_ID_RE = re.compile(r"(?:functions\.)?([A-Za-z_]\w*)(?::\d+)?\Z")

MALFORMED_ACTION_PROMPT = (
    "The tool call could not be parsed. Re-emit one valid tool call in the "
    "required format."
)


@dataclass
class ToolSpec:
    """One tool the model may call.

    ``schema`` is sent to the API verbatim as an OpenAI function definition.
    ``run`` receives the parsed arguments and returns result objects exposing
    ``format() -> str``. Raising one of ``terminal_errors`` ends the task
    instead of returning an error the model can retry.
    """

    name: str
    schema: dict
    run: Callable[[dict], list[Any]]
    terminal_errors: tuple[type[BaseException], ...] = ()
    context_error_max_bytes: int | None = None
    context_error_guidance: str = ""


@dataclass
class ToolCall:
    name: str
    arguments: dict
    call_id: str | None = None


def message_text(message: dict) -> str:
    """The assistant's text, however the producer represented it.

    ``content`` is a plain string from a chat API, but a renderer parsing a
    sampled completion returns it as a list of typed parts, of which only the
    text parts are joined. A string is returned whole, inline reasoning included.
    Anything reading a message has to accept both.
    """
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def execution_message(parsed: dict) -> dict:
    """The turn as the tool-call parser reads it: rejected call syntax restored to the text.

    A renderer moves call syntax it cannot parse out of the text and into
    ``unparsed_tool_calls``. ``parse_tool_calls`` still accepts some of it (the
    named-XML ``<tool_call>`` form, for a model whose native syntax differs),
    and ``has_tool_call_intent`` still sees the attempt, so the rejected text is
    appended after the turn's text and the content becomes a string. A message
    without rejected calls is returned unchanged (the same object).
    """
    if not parsed.get("unparsed_tool_calls"):
        return parsed
    return {
        **parsed,
        "content": message_text(parsed)
        + "".join(c["raw_text"] for c in parsed["unparsed_tool_calls"]),
    }


def _kimi_section_calls(text: str) -> list[ToolCall]:
    """The named calls in the first complete Kimi section, read as the Kimi renderer reads them.

    Each call's arguments must be valid JSON; a JSON object of the form
    ``{"name": ..., "arguments": ...}`` names the call itself, otherwise the
    name comes from a ``[functions.]<name>:<index>`` call id. Calls that fail
    either test are skipped, and a text with no complete section has none.
    """
    section = _KIMI_SECTION_BODY_RE.search(text)
    if section is None:
        return []
    calls = []
    for call_id_raw, args_str in _KIMI_TOOL_CALL_RE.findall(section.group(1)):
        call_id = call_id_raw.strip()
        try:
            arguments = json.loads(args_str.strip())
        except (json.JSONDecodeError, RecursionError):
            continue
        if (
            isinstance(arguments, dict)
            and "name" in arguments
            and "arguments" in arguments
        ):
            name = arguments["name"]
            arguments = arguments["arguments"]
        else:
            named = _KIMI_SECTION_CALL_ID_RE.fullmatch(call_id)
            name = named.group("name") if named is not None else None
        if not isinstance(name, str) or not name:
            continue
        calls.append(
            ToolCall(
                name=name,
                arguments=arguments if isinstance(arguments, dict) else {},
                call_id=call_id or None,
            )
        )
    return calls


def parse_kimi_tool_calls(text: str) -> list[ToolCall]:
    """Kimi's marker-delimited tool-call section.

    The first complete section is read with the Kimi renderer's own grammar
    (``_kimi_section_calls``). When that yields no named call, because the
    section is unterminated or none of its calls parse, every call in the text
    is read with a looser grammar: a call may run to the end of the text, its
    name is the trailing identifier of its id, and arguments that are not a
    JSON object become ``{}``.
    """
    if KIMI_TOOL_SECTION_BEGIN not in text:
        return []
    try:
        out = _kimi_section_calls(text)
    except Exception:  # noqa: BLE001 - e.g. JSON nested too deeply; the grammar below decides
        out = []
    if out:
        return out
    calls = []
    for call_id, raw in _KIMI_TOOL_CALL_RE.findall(text):
        named = _KIMI_CALL_ID_RE.search(call_id.strip())
        if named is None:
            continue
        try:
            arguments = json.loads(raw.strip())
        except (json.JSONDecodeError, RecursionError):
            arguments = {}
        calls.append(
            ToolCall(
                name=named.group(1),
                arguments=arguments if isinstance(arguments, dict) else {},
                call_id=call_id.strip(),
            )
        )
    return calls


def parse_tool_calls(message: dict) -> list[ToolCall]:
    """A turn's tool calls, from the first source that has any (see the module docstring)."""
    calls: list[ToolCall] = []
    for entry in message.get("tool_calls") or []:
        function = entry.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, RecursionError):
                arguments = {}
        calls.append(
            ToolCall(
                name=function.get("name") or "",
                arguments=arguments or {},
                call_id=str(entry.get("id")) if entry.get("id") else None,
            )
        )
    if calls:
        return calls
    text = message_text(message)
    kimi = parse_kimi_tool_calls(text)
    if kimi:
        return kimi
    for block in _tool_call_blocks(text):
        block = block.strip()
        named = _NAMED_FUNCTION_RE.search(block)
        if named:
            arguments = {
                key: value.strip()
                for key, value in _PARAMETER_RE.findall(named.group(2))
            }
            calls.append(ToolCall(name=named.group(1), arguments=arguments))
            continue
        try:
            payload = json.loads(block)
        except (json.JSONDecodeError, RecursionError):
            continue
        if isinstance(payload, dict) and payload.get("name"):
            arguments = payload.get("arguments")
            calls.append(
                ToolCall(
                    name=str(payload["name"]),
                    arguments=arguments if isinstance(arguments, dict) else {},
                    call_id=str(payload.get("id")) if payload.get("id") else None,
                )
            )
    return calls


def has_tool_call_intent(message: dict, *, kimi_markers: bool = False) -> bool:
    """Whether a response attempted a tool call even if parsing failed.

    Structured calls, ``<tool_call`` and ``<function=`` count as attempts;
    ``kimi_markers`` also counts a Kimi section marker.
    """
    if message.get("tool_calls"):
        return True
    text = message_text(message)
    return (
        "<tool_call" in text
        or "<function=" in text
        or (kimi_markers and KIMI_TOOL_SECTION_BEGIN in text)
    )
