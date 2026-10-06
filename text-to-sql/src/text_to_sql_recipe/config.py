"""Recipe configuration: one schema, conventional key names, strict loading.

The dataclass defaults are the recipe core that every model shares. A preset (one per model, installed with the
package in `text_to_sql_recipe/recipes/`) sets only what is specific to its model; `load()` merges the preset over
the defaults and rejects unknown keys and wrongly typed values, so a misspelled setting fails loudly instead of
silently keeping its default.

Key names follow the Hugging Face Trainer / TRL conventions where one exists (`learning_rate`, `warmup_steps`,
`max_grad_norm`, `adam_beta2`, `num_train_epochs`, `epsilon_high`, `seed`, `eval_steps`). The method itself is
fixed and has no setting: CISPO loss, linear warmup then cosine decay.

Connection settings come from the environment, never from a preset:
  RIVER_API_KEY           required to train or evaluate
  RIVER_BASE_URL          optional, defaults to the public API
  TEXT_TO_SQL_DATA_DIR    optional, where datasets are read from (default ./data)
  TEXT_TO_SQL_OUTPUT_DIR  optional, where runs are written (default ./runs)
  TEXT_TO_SQL_VERIEQL_MANIFEST, TEXT_TO_SQL_VERIEQL_MANIFEST_SHA256
                          the VeriEQL runtime and its pin; required only with reward.verifier = "verieql"
"""
from __future__ import annotations

import dataclasses
import math
import os
import tomllib
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import data

DEFAULT_BASE_URL = "https://api.river.ai"


@dataclass(frozen=True)
class ModelConfig:
    name: str = ""                            # base model id served by River
    tokenizer_revision: str | None = None     # pinned tokenizer revision on the Hugging Face Hub
    lora_rank: int = 32
    routing_replay: bool = False              # replay MoE expert routing between sampling and training
    keep_thinking_in_history: bool = False    # keep earlier turns' reasoning in the prompt
    reasoning_effort: str | None = None       # for models that expose an effort setting


@dataclass(frozen=True)
class TrainingConfig:
    num_train_epochs: int = 5
    batch_size: int = 64                      # questions per update
    group_size: int = 16                      # rollouts per question
    learning_rate: float = 5e-5
    warmup_steps: int = 10                    # linear warmup, then cosine decay
    min_lr_ratio: float = 0.1                 # cosine floor, as a fraction of learning_rate
    adam_beta2: float = 0.95
    max_grad_norm: float = 20000.0            # clip ceiling; the adaptive clip below never exceeds it
    grad_clip_median_multiple: float | None = 2.0  # clip at this multiple of the recent median norm; None = fixed
    grad_clip_window: int = 20                # recent steps the median is taken over
    epsilon_high: float = 1.2                 # CISPO upper importance-ratio clip
    seed: int = 42


@dataclass(frozen=True)
class RolloutConfig:
    max_turns: int = 5                        # SQL tool turns; one answer turn follows
    max_tokens_per_turn: int = 3072
    max_context_length: int = 29696
    few_shot: bool = True                     # include the tool-use demonstration
    max_staleness: int = 2                    # updates a rollout's policy may lag the trainer


@dataclass(frozen=True)
class RewardConfig:
    process_penalty_weight: float = 0.1
    verifier: str = "verieql"                 # "verieql" or "none"
    verifier_weight: float = 0.2
    verifier_gate: bool = False               # count a refutation only with an executed counterexample
    redraft_penalty: float = 0.05
    redraft_penalty_cap: float = 0.2
    tool_format_penalty: float = 0.0
    tool_format_penalty_cap: float = 0.2


@dataclass(frozen=True)
class CurriculumConfig:
    evidence_keep_probability: float = 1.0    # chance a question is shown its evidence; 1.0 = always
    hints: bool = True                        # rescue: after all attempts fail with evidence, retry with a hint
    evidence_weighting: str = "lane"          # weight a draw by the record of its lane, or "unaided": the unaided record
    evidence_failed_weight: float = 0.05      # practice weight of a question last scored 0/16 in that record
    evidence_solved_weight: float = 0.0625    # practice weight of a question last scored 16/16 in that record
    evidence_fade_probability: float = 0.0    # once solved 16/16 with evidence, chance each later draw withholds it
    evidence_draws_include_empty: bool = True  # evidence draws may also pick a question whose evidence is empty


@dataclass(frozen=True)
class GuardConfig:
    loop_stop: bool = True
    repetition_guard: bool = True
    repetition_min_tokens: int = 48
    repetition_max_period: int = 8
    repetition_span_repeats: int = 4
    collapse_pause: bool = True
    collapse_cap_rate: float = 0.05
    collapse_repetition_rate: float = 0.01
    collapse_consecutive_steps: int = 2
    collapse_arm_after_steps: int = 33        # the pause can trigger only after this many steps (one epoch)


@dataclass(frozen=True)
class EvaluationConfig:
    eval_steps: int = 5                       # evaluate every N updates, and at the end
    arcwise: bool = True                      # also evaluate on the Arcwise set (its file must be in the data directory)
    arcwise_grading: str = "authors"          # "authors": the ReViSQL authors' 1% single-value tolerance; "strict"
    concurrency: int = 64


@dataclass(frozen=True)
class DataConfig:
    train: str = "revisql-verified"           # dataset name, resolved under TEXT_TO_SQL_DATA_DIR
    validation: str = "revisql-verified-validation"


@dataclass(frozen=True)
class ExecutionConfig:
    max_concurrent_rollouts: int = 128
    sample_batch_size: int = 128
    tool_concurrency: int = 32
    grading_concurrency: int = 16
    sql_concurrency: int = 16
    verifier_workers: int = 16


@dataclass(frozen=True)
class RecipeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    guards: GuardConfig = field(default_factory=GuardConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def validate(self) -> RecipeConfig:
        problems = []
        if not self.model.name:
            problems.append("model.name is required")
        for name, low in _MINIMUMS.items():
            section, key = name.split(".")
            if getattr(getattr(self, section), key) < low:
                problems.append(f"{name} must be at least {low}")
        for name in _UNIT_INTERVAL:
            section, key = name.split(".")
            if not 0 <= getattr(getattr(self, section), key) <= 1:
                problems.append(f"{name} must be within [0, 1]")
        if self.training.learning_rate <= 0:
            problems.append("training.learning_rate must be positive")
        if not 0 < self.training.adam_beta2 < 1:
            problems.append("training.adam_beta2 must be within (0, 1)")
        if self.training.max_grad_norm <= 0:
            problems.append("training.max_grad_norm must be positive")
        if self.training.epsilon_high < 1:
            problems.append("training.epsilon_high must be at least 1 (it caps the importance ratio)")
        if self.model.reasoning_effort is not None and not self.model.reasoning_effort:
            problems.append("model.reasoning_effort must be a non-empty string, or unset")
        problems += data.check_names(self.data.train, self.data.validation)
        if self.training.grad_clip_median_multiple is not None and self.training.grad_clip_median_multiple <= 1:
            problems.append("training.grad_clip_median_multiple must be greater than 1, or unset")
        if self.guards.collapse_arm_after_steps < 0:
            problems.append("guards.collapse_arm_after_steps must be non-negative")
        if self.training.grad_clip_window < 3:
            problems.append("training.grad_clip_window must be at least 3")
        if self.reward.verifier not in ("verieql", "none"):
            problems.append("reward.verifier must be 'verieql' or 'none'")
        if self.reward.verifier == "verieql" and not 0 < self.reward.verifier_weight <= 1:
            problems.append("reward.verifier = 'verieql' needs reward.verifier_weight within (0, 1]")
        if self.evaluation.eval_steps < 1:
            problems.append("evaluation.eval_steps must be at least 1")
        if self.evaluation.arcwise_grading not in ("authors", "strict"):
            problems.append("evaluation.arcwise_grading must be 'authors' or 'strict'")
        if not 0 <= self.curriculum.evidence_keep_probability <= 1:
            problems.append("curriculum.evidence_keep_probability must be within [0, 1]")
        if self.curriculum.evidence_weighting not in ("unaided", "lane"):
            problems.append("curriculum.evidence_weighting must be 'unaided' or 'lane'")
        if not 0 < self.curriculum.evidence_failed_weight <= 1:
            problems.append("curriculum.evidence_failed_weight must be within (0, 1]")
        if not 0 < self.curriculum.evidence_solved_weight <= 1:
            problems.append("curriculum.evidence_solved_weight must be within (0, 1]")
        if not 0 <= self.curriculum.evidence_fade_probability <= 1:
            problems.append("curriculum.evidence_fade_probability must be within [0, 1]")
        if self.rollout.max_tokens_per_turn > self.rollout.max_context_length:
            problems.append("rollout.max_tokens_per_turn exceeds rollout.max_context_length")
        if problems:
            raise ValueError("invalid recipe: " + "; ".join(problems))
        return self

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


SECTIONS = {f.name: f.type for f in dataclasses.fields(RecipeConfig)}

# Counts and sizes, with the smallest value that makes sense; every other integer setting is at least 1.
_MINIMUMS = {
    "model.lora_rank": 1,
    "training.num_train_epochs": 1, "training.batch_size": 1, "training.group_size": 2, "training.warmup_steps": 0,
    "rollout.max_turns": 1, "rollout.max_tokens_per_turn": 1, "rollout.max_context_length": 1,
    "rollout.max_staleness": 0,
    "reward.process_penalty_weight": 0, "reward.redraft_penalty": 0, "reward.redraft_penalty_cap": 0,
    "reward.tool_format_penalty": 0, "reward.tool_format_penalty_cap": 0,
    "guards.repetition_min_tokens": 1, "guards.repetition_max_period": 1, "guards.repetition_span_repeats": 2,
    "guards.collapse_consecutive_steps": 1,
    "evaluation.concurrency": 1,
    "execution.max_concurrent_rollouts": 1, "execution.sample_batch_size": 1, "execution.tool_concurrency": 1,
    "execution.grading_concurrency": 1, "execution.sql_concurrency": 1, "execution.verifier_workers": 1,
}
_UNIT_INTERVAL = ("training.min_lr_ratio", "guards.collapse_cap_rate", "guards.collapse_repetition_rate")


def _accepts(annotation, value) -> bool:
    """Whether a TOML value fits a setting's annotation. Booleans are never numbers, numbers are finite, and an
    integer is accepted where a float is expected."""
    if isinstance(annotation, types.UnionType):
        return any(_accepts(member, value) for member in typing.get_args(annotation))
    if annotation is type(None):
        return value is None
    if annotation is bool:
        return isinstance(value, bool)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if annotation is str:
        return isinstance(value, str)
    raise TypeError(f"unsupported setting type {annotation!r}")


def _section(cls, values: dict[str, Any], where: str):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f"unknown key(s) in [{where}]: {', '.join(unknown)}")
    hints = typing.get_type_hints(cls)
    wrong = [f"{where}.{key} = {value!r}" for key, value in values.items() if not _accepts(hints[key], value)]
    if wrong:
        raise ValueError(f"wrong type or non-finite value for {', '.join(wrong)}")
    return cls(**{key: float(value) if hints[key] is float else value for key, value in values.items()})


def from_dict(raw: dict[str, Any]) -> RecipeConfig:
    unknown = sorted(set(raw) - set(SECTIONS))
    if unknown:
        raise ValueError(f"unknown section(s): {', '.join(unknown)}")
    parts = {}
    for f in dataclasses.fields(RecipeConfig):
        cls = type(f.default_factory())
        parts[f.name] = _section(cls, raw.get(f.name, {}), f.name)
    return RecipeConfig(**parts).validate()


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `section.key=value` overrides; values are parsed as TOML scalars."""
    merged = {k: dict(v) for k, v in raw.items()}
    for item in overrides:
        key, sep, text = item.partition("=")
        section, dot, name = key.strip().partition(".")
        if not sep or not dot:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        try:
            value = tomllib.loads(f"v = {text.strip()}")["v"]
        except tomllib.TOMLDecodeError:
            raise ValueError(f"override {item!r}: the value must be a TOML value; quote a string, as in "
                             f"{key.strip()}='\"{text.strip()}\"'") from None
        merged.setdefault(section, {})[name] = value
    return merged


PRESET_DIR = Path(__file__).resolve().parent / "recipes"


def presets() -> list[str]:
    """The names of the presets installed with the package."""
    return sorted(path.stem for path in PRESET_DIR.glob("*.toml"))


def resolve(recipe: str | Path) -> Path:
    """A path to a recipe TOML: the file itself, or the installed preset of that name (`glm-5.3-flash`)."""
    path = Path(recipe)
    if path.is_file():
        return path
    preset = PRESET_DIR / f"{path.name.removesuffix('.toml')}.toml"
    if path.parent == Path(".") and preset.is_file():
        return preset
    raise FileNotFoundError(f"no recipe file {str(recipe)!r} and no preset of that name; presets: "
                            + ", ".join(presets()))


def load(recipe: str | Path, overrides: list[str] | None = None) -> RecipeConfig:
    raw = tomllib.loads(resolve(recipe).read_text())
    return from_dict(apply_overrides(raw, overrides or []))


@dataclass(frozen=True)
class Connection:
    api_key: str
    base_url: str
    data_dir: Path
    output_dir: Path


def connection(require_key: bool = True) -> Connection:
    key = os.environ.get("RIVER_API_KEY", "")
    if require_key and not key:
        raise SystemExit("RIVER_API_KEY is not set")
    return Connection(api_key=key, base_url=os.environ.get("RIVER_BASE_URL", DEFAULT_BASE_URL),
                      data_dir=Path(os.environ.get("TEXT_TO_SQL_DATA_DIR", "data")),
                      output_dir=Path(os.environ.get("TEXT_TO_SQL_OUTPUT_DIR", "runs")))
