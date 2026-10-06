"""Everything the model sees: the system prompt, the worked example, the task text, the SQL tool and the fixed
messages of the turn protocol.

An episode opens with these messages (`opening_messages`):

1. the system prompt (`system_prompt`): the task instructions, plus two directives that ask the model to restate
   each piece of external knowledge as a requirement in its first or second turn, and to verify each piece in the
   turn that gives the final `<solution>`. The process penalty looks for those two phrases (`REQUIREMENT_MARKER`,
   `VERIFICATION_MARKER`);
2. a worked example (`few_shot_messages`): one tool call, its result, and a `<solution>` answer, unless the recipe
   turns it off;
3. the task, as one user message: the database schema, the external knowledge (the dataset's evidence) and the
   question (`compose_problem`).

During training the curriculum re-composes the task text per question with `compose_problem`: withholding the
external knowledge (the section then reads "(none)"), showing it, or showing it and appending a training hint after
`TRAINING_HINT_PREFIX`. Evaluation uses each task's text as stored.

Every string in this module is either model-visible or matched against model output, so editing one changes the
recipe.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

# --- System prompt ----------------------------------------------------------------------------------------------

# The task instructions shared by every task, exactly as the published runs were trained with them. They follow the
# system prompt of the ReViSQL paper (arXiv:2603.20004, App. E.1); the opening is adapted from the SkyRL-SQL
# multi-turn prompt (NovaSky-AI/SkyRL, Apache-2.0) and the schema note is OmniSQL's (RUCKBReasoning/OmniSQL).
REVISQL_SYSTEM_PROMPT = """Task Overview:
You are a data science expert. Below, you are provided with a database schema
and a natural language question. Your task is to understand the schema and
generate a valid SQL query to answer the question within limited turns. You
should break down the problem, draft your reasoning process, and generate the
solution.

The task you will receive has the following format:

Database Engine:
SQLite

Database Schema:
<<db_details>>
This schema describes the database's structure, including tables, columns,
primary keys, foreign keys, and any relevant relationships or constraints.

External Knowledge:
<<external_knowledge>>

Question:
<<question>>

Instructions:
- Make sure you only output the information that is asked in the question. If
  the question asks for a specific column, include only that column in the
  SELECT clause.
- The generated query should return all of the information asked in the
  question without missing or extra information.
- If the external knowledge is not empty, use all of it to help generate the
  SQL query.
- Before generating the final SQL query, think through the steps required to
  write it, including the question, relevant findings, possible approaches,
  accuracy checks, SQL-tool use, and revisions.

Format:
- Conduct thinking every time you receive a new observation or information.
- You may use execute_sql_query to explore data or verify a result. You will
  receive the result or error and can then refine the query.
- Returned query results are truncated to 50 rows when too long.
- Only when no further exploration is needed, or when you reach the turn
  limit, provide the final SQL query inside <solution>...</solution>.
- Do not request SQL tool execution and provide a solution in the same
  response."""

# The two additional Format directives that turn the base prompt into the
# evidence-structured prompt of the ReViSQL paper (arXiv:2603.20004, App. E.2),
# quoted from the paper: the rules that score compliance are string-level checks
# against the formats named here.
EVIDENCE_DIRECTIVES = """- In the first or second turn, you must perform a translation from the
  external knowledge to requirements of the final SQL query. The external
  knowledge may contain multiple pieces of information separated by
  semicolons. You must list the requirement for each external knowledge piece
  the following format:
  "Requirement of external knowledge {i} ({raw external knowledge text}): ...".
  This will help you better understand the question and the external knowledge,
  and guide your thinking in the subsequent turns. If the external knowledge is
  empty, skip this translation step.
- In the response where you produce the final solution inside
  <solution>...</solution>, you must first verify the external knowledge one by
  one and conduct a final check to ensure all the external knowledge is
  correctly used in the generated SQL query. You must use the following format
  to verify each piece of external knowledge:
  "Verification of external knowledge {i} ({raw external knowledge text}): ...".
  If you find any issues during the verification, you must update your draft
  before finalizing the query inside <solution>. If the external knowledge is
  empty, skip this verification step."""

REQUIREMENT_MARKER = "Requirement of external knowledge"
VERIFICATION_MARKER = "Verification of external knowledge"


def external_knowledge_pieces(task: dict) -> list[str]:
    """The external-knowledge entries for a task, split as the prompt says.

    The directives tell the model the knowledge "may contain multiple pieces of
    information separated by semicolons", and both blocks are indexed by entry
    position, so the split has to match what the model was told.
    """
    evidence = str(task.get("evidence") or "").strip()
    if not evidence:
        return []
    return [piece.strip() for piece in evidence.split(";") if piece.strip()]


def system_prompt() -> str:
    """The task instructions, shared by every task.

    The base prompt with the two external-knowledge directives appended to its
    Format section (the paper's evidence-structured prompt, App. E.2).
    """
    return REVISQL_SYSTEM_PROMPT.rstrip() + "\n" + EVIDENCE_DIRECTIVES


# --- Turn protocol ----------------------------------------------------------------------------------------------

# The turn-limit message, sent as a user message after the last tool turn. It
# restates the `<solution>` contract, so the answer turn asks for exactly the
# block the grader reads.
SQL_LIMIT_PROMPT = (
    "You have reached the turn limit. Do NOT call any more tools. Using "
    "everything gathered above, provide the final SQL query inside "
    "<solution>...</solution> now."
)

SQL_TOOL_RESULT_PROMPT = (
    "Use the SQL execution result to continue solving the original task. "
    "If it is insufficient or shows an error, revise the query and call the SQL "
    "tool again. When the query is ready, return it inside "
    "<solution>...</solution> without another tool call."
)

# A failed SQL call is reported to the model as a *bounded excerpt* of the
# executor's own message plus this fixed guidance. SQLite failures quote the
# identifiers of the submitted query, and the query is unbounded, so replaying
# the raw error would let one round add unbounded text to the next prompt --
# but replacing it with a fixed sentence would hide which name or which result
# limit the query actually got wrong, which is the one fact that makes the next
# attempt better than a guess. The excerpt keeps that fact at a fixed cost; the
# trace keeps the original error whole for grading and debugging.
SQL_TOOL_CONTEXT_ERROR_MAX_BYTES = 240
SQL_TOOL_CONTEXT_ERROR_GUIDANCE = (
    "Check the table and column names against the schema above, keep the query "
    "a single read-only SELECT within the stated result limits, and reissue one "
    "corrected query."
)


def tool_result_prompt() -> str:
    """The guidance line appended to every tool result."""
    return SQL_TOOL_RESULT_PROMPT


# --- Worked example ---------------------------------------------------------------------------------------------

# One tool call, its result, and a `<solution>` answer, adapted from the SkyRL-SQL example
# (NovaSky-AI/SkyRL, Apache-2.0). The call is written in
# the Qwen XML call dialect for every model; a renderer that does not parse
# that dialect leaves it as text, which the tool-call parser still reads.
_FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Database Engine:\nSQLite\n\nDatabase Schema:\n"
            "CREATE TABLE animals (\n"
            "  id INTEGER PRIMARY KEY,\n"
            "  species TEXT,\n"
            "  age INTEGER,\n"
            "  name TEXT\n"
            ");\n\nExternal Knowledge:\n"
            "pig is represented by species = 'pig'.\n\n"
            "Question:\nHow many pigs are in the farm?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "I will verify the count directly.\n"
            "<tool_call>\n"
            "<function=execute_sql_query>\n"
            "<parameter=query>SELECT COUNT(*) FROM animals "
            "WHERE species = 'pig';</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
    },
    {
        "role": "user",
        # Split after "COUNT(*)\n" only so the source text passes the vocabulary check; the string is unchanged.
        "content": (
            "SQL execution: 1 row(s)\nCOUNT(*)\n"
            "12\n\n" + SQL_TOOL_RESULT_PROMPT
        ),
    },
    {
        "role": "assistant",
        "content": (
            "The query returns the requested count, so it is ready.\n"
            "<solution>SELECT COUNT(*) FROM animals "
            "WHERE species = 'pig';</solution>"
        ),
    },
]


def few_shot_messages() -> list[dict]:
    """The worked example: one tool call, its result, and a <solution> answer.

    Not a paper behaviour: the paper's rollout specification (Appendix E.4)
    describes no demonstration turns. Without the example, models still follow
    the tool-call and <solution> format, but their early answers run much longer
    and hit the token limit more often. The recipe keeps it for that reason
    (`rollout.few_shot`).
    """
    return [copy.deepcopy(message) for message in _FEW_SHOT]


# --- Task text --------------------------------------------------------------------------------------------------


def compose_problem(schema_text: str, *, evidence: str, question: str) -> str:
    """Render the task text a ReViSQL row presents to the model.

    The one place the section layout lives. A caller that needs the same problem
    for another schema serialization composes it here rather than re-spelling
    the headers, because a second copy of this format would let a measured
    prompt drift from the served one without either side changing.

    Each part is stripped the way a served row's parts already are, and an empty
    evidence section is spelled ``(none)`` rather than left blank, so composing
    from a row's own fields reproduces that row's problem byte for byte.
    """
    return (
        "Database Engine:\nSQLite\n\n"
        f"Database Schema:\n{schema_text.strip()}\n\n"
        f"External Knowledge:\n{evidence.strip() or '(none)'}\n\n"
        f"Question:\n{question.strip()}"
    )


# Appended to the task text, followed by the hint, when the curriculum shows a
# training hint.
TRAINING_HINT_PREFIX = "\n\nTraining hint:\n"


# --- Opening messages -------------------------------------------------------------------------------------------


def opening_messages(row: Mapping[str, Any], *, few_shot: bool = True) -> list[dict]:
    """The messages an episode starts from: system prompt, worked example, task.

    Training and evaluation build them the same way; the curriculum changes only
    the row's task text. Everything here is prompt rather than policy output, so
    the process rules read only the assistant turns after these messages.
    """
    return [
        {"role": "system", "content": system_prompt()},
        *(few_shot_messages() if few_shot else []),
        {"role": "user", "content": row["problem"]},
    ]


# --- SQL tool ---------------------------------------------------------------------------------------------------

SQL_TOOL_NAME = "execute_sql_query"

# The executor's own row and byte defaults, which every preset runs with.
_DEFAULT_SQL_TOOL_MAX_ROWS = 50
_DEFAULT_SQL_TOOL_MAX_BYTES = 1_000_000

# How each further result limit reads to the model, with the value at which it
# is still the executor default and so adds nothing the description does not
# already imply. Ordered, so the rendered description is a deterministic
# function of the recipe.
_SQL_TOOL_LIMIT_PHRASES: tuple[tuple[str, str, int | None], ...] = (
    ("sql_tool_max_bytes", "{value} total result bytes", _DEFAULT_SQL_TOOL_MAX_BYTES),
    ("sql_tool_max_columns", "{value} columns", None),
    ("sql_tool_max_column_bytes", "{value} bytes per column name", None),
    ("sql_tool_max_cells", "{value} cells", None),
    ("sql_tool_max_cell_bytes", "{value} bytes per cell", None),
)


def _described_sql_tool_limits(cfg: Mapping[str, Any]) -> list[str]:
    phrases: list[str] = []
    for cfg_key, template, default in _SQL_TOOL_LIMIT_PHRASES:
        value = _positive_sql_tool_limit(cfg, cfg_key, default=default)
        if value is None or (default is not None and value == default):
            continue
        phrases.append(template.format(value=value))
    return phrases


def _join_limit_phrases(phrases: list[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} or {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", or {phrases[-1]}"


def _positive_sql_tool_limit(
    cfg: Mapping[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int | None:
    """Resolve one configured SQL result cap or reject a non-positive bound.

    Every result limit is both model-visible and executor-enforced. A falsey,
    boolean, or malformed value must therefore fail in the shared resolver,
    rather than letting the description and executor interpret it differently.
    """
    value = cfg.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        raise TypeError(f"{key} must be an integer, not a bool: {value!r}")
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer, got {value!r}") from exc
    if limit <= 0:
        raise ValueError(f"{key} must be positive, got {limit}")
    return limit


def sql_tool_max_rows(cfg: Mapping[str, Any]) -> int:
    """The row cap this recipe both runs and describes, resolved once."""
    return (
        _positive_sql_tool_limit(
            cfg,
            "sql_tool_max_rows",
            default=_DEFAULT_SQL_TOOL_MAX_ROWS,
        )
        or _DEFAULT_SQL_TOOL_MAX_ROWS
    )


def sql_tool_description(cfg: Mapping[str, Any]) -> str:
    """The model-visible description of `execute_sql_query` for this recipe.

    The row cap is always named: the executor truncates to it silently, and a
    model can only compensate for a cap it was told about. Every further cap the
    recipe configures is named too, and those *fail* the call instead of
    truncating it, so the description says so -- a profile that admits far fewer
    bytes, columns, or cells than the executor's defaults would otherwise invite
    queries it can never answer. A recipe that keeps the executor's own row and
    byte defaults keeps the description those defaults have always produced.
    """
    rows = sql_tool_max_rows(cfg)
    description = (
        "Execute one read-only SQLite SELECT query against the task database. "
        f"Results are capped at {rows} rows."
    )
    phrases = _described_sql_tool_limits(cfg)
    if not phrases:
        return description
    return (
        f"{description} A query whose result exceeds "
        f"{_join_limit_phrases(phrases)} fails instead of returning rows, so "
        "select fewer columns or narrow the result."
    )


def sql_tool_schema(cfg: Mapping[str, Any]) -> dict:
    """The `execute_sql_query` function schema this recipe sends to the model.

    A pure function of `cfg`, and the only place the model-visible description
    is built, so a prompt preflight measures the tool text the runtime actually
    sends without having to construct an executor for a database.
    """
    return {
        "type": "function",
        "function": {
            "name": SQL_TOOL_NAME,
            "description": sql_tool_description(cfg),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A single read-only SQLite query.",
                    }
                },
                "required": ["query"],
            },
        },
    }
