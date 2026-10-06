import dataclasses
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text_to_sql_recipe import config

PRESETS = sorted((ROOT / "src/text_to_sql_recipe/recipes").glob("*.toml"))


class PresetTest(unittest.TestCase):
    def test_every_preset_loads_and_validates(self):
        self.assertEqual(len(PRESETS), 6)
        for path in PRESETS:
            with self.subTest(preset=path.name):
                recipe = config.load(path)
                self.assertTrue(recipe.model.name)
                self.assertTrue(recipe.model.tokenizer_revision)

    def test_presets_differ_only_where_the_model_needs_it(self):
        # the shared core stays at its defaults in every preset
        core = {"training": ("batch_size", "group_size", "learning_rate", "warmup_steps", "min_lr_ratio",
                             "adam_beta2", "max_grad_norm", "epsilon_high", "seed"),
                "rollout": ("max_turns", "max_context_length", "few_shot", "max_staleness"),
                "reward": ("process_penalty_weight", "verifier", "verifier_weight", "redraft_penalty")}
        default = config.RecipeConfig()
        for path in PRESETS:
            recipe = config.load(path)
            for section, names in core.items():
                for name in names:
                    with self.subTest(preset=path.name, key=f"{section}.{name}"):
                        self.assertEqual(getattr(getattr(recipe, section), name),
                                         getattr(getattr(default, section), name))

    def test_presets_load_by_name_or_path(self):
        self.assertEqual(config.presets(), sorted(path.stem for path in PRESETS))
        by_name = config.load("glm-5.3-flash")
        self.assertEqual(by_name, config.load(PRESETS[[p.stem for p in PRESETS].index("glm-5.3-flash")]))
        self.assertEqual(config.load("glm-5.3-flash.toml"), by_name)
        with self.assertRaisesRegex(FileNotFoundError, "presets: deepseek"):
            config.load("no-such-model")
        with self.assertRaises(FileNotFoundError):  # a path never falls back to a preset
            config.load("elsewhere/glm-5.3-flash.toml")

    def test_glm_runs_seven_epochs(self):
        self.assertEqual(config.load("glm-5.3-flash").training.num_train_epochs, 7)

    def test_lightning_runs_seven_epochs_with_routing_replay(self):
        lightning = config.load("nemotron-3.5-lightning")
        self.assertEqual(lightning.training.num_train_epochs, 7)
        self.assertTrue(lightning.model.routing_replay)

    def test_deepseek_runs_max_effort_with_room_to_think(self):
        deepseek = config.load("deepseek-v4.1-flash")
        self.assertEqual(deepseek.model.reasoning_effort, "max")
        self.assertEqual(deepseek.rollout.max_tokens_per_turn, 8192)

    def test_verieql_refutations_count_without_the_counterexample_gate(self):
        self.assertFalse(config.RewardConfig().verifier_gate)

    def test_every_model_shares_the_curriculum_and_the_data(self):
        endorsed = {
            "evidence_keep_probability": 1.0, "hints": True, "evidence_weighting": "lane",
            "evidence_failed_weight": 0.05, "evidence_solved_weight": 0.0625, "evidence_fade_probability": 0.0,
            "evidence_draws_include_empty": True,
        }
        self.assertEqual(dataclasses.asdict(config.CurriculumConfig()), endorsed)
        self.assertEqual(dataclasses.asdict(config.DataConfig()),
                         {"train": "revisql-verified", "validation": "revisql-verified-validation"})
        for path in PRESETS:
            with self.subTest(preset=path.name):
                raw = tomllib.loads(path.read_text())
                self.assertFalse({"curriculum", "data"} & set(raw))
                recipe = config.load(path)
                self.assertEqual(recipe.curriculum, config.CurriculumConfig())
                self.assertEqual(recipe.data, config.DataConfig())

    def test_reasoning_effort_is_pinned_where_the_model_has_one(self):
        efforts = {path.stem: config.load(path).model.reasoning_effort for path in PRESETS}
        self.assertEqual({name: effort for name, effort in efforts.items() if effort is not None},
                         {"glm-5.3-flash": "max", "qwen3.8-27b": "xhigh", "deepseek-v4.1-flash": "max"})

    def test_unknown_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown key"):
            config.from_dict({"model": {"name": "m", "learnign_rate": 1}})
        with self.assertRaisesRegex(ValueError, "unknown section"):
            config.from_dict({"model": {"name": "m"}, "trainer": {}})

    def test_overrides_parse_toml_values(self):
        recipe = config.load(PRESETS[0], ["training.learning_rate=1e-5", "reward.verifier=\"none\""])
        self.assertEqual(recipe.training.learning_rate, 1e-5)
        self.assertEqual(recipe.reward.verifier, "none")

    def test_invalid_values_fail(self):
        with self.assertRaisesRegex(ValueError, "evidence_keep_probability"):
            config.from_dict({"model": {"name": "m"}, "curriculum": {"evidence_keep_probability": 1.5}})

    def test_wrong_types_and_non_finite_numbers_fail_before_any_work(self):
        cases = {
            "training.learning_rate = nan": ("training", "learning_rate", float("nan")),
            "training.learning_rate = inf": ("training", "learning_rate", float("inf")),
            'reward.verifier_gate = "false"': ("reward", "verifier_gate", "false"),
            "reward.verifier_gate = 0": ("reward", "verifier_gate", 0),
            "model.lora_rank = true": ("model", "lora_rank", True),
            "training.batch_size = 64.0": ("training", "batch_size", 64.0),
            "model.reasoning_effort = 3": ("model", "reasoning_effort", 3),
        }
        for label, (section, key, value) in cases.items():
            with self.subTest(label), self.assertRaisesRegex(ValueError, "wrong type or non-finite"):
                config.from_dict({"model": {"name": "m"}, section: {key: value}} if section != "model"
                                 else {"model": {"name": "m", key: value}})
        # TOML spells booleans without quotes; an override follows the same rules as a preset.
        self.assertFalse(config.load(PRESETS[0], ["reward.verifier_gate=false"]).reward.verifier_gate)
        with self.assertRaisesRegex(ValueError, "wrong type"):
            config.load(PRESETS[0], ['reward.verifier_gate="false"'])
        # An unquoted string is not a TOML value: the error says how to quote it.
        with self.assertRaisesRegex(ValueError, "quote a string"):
            config.load(PRESETS[0], ["reward.verifier=none"])

    def test_arcwise_is_graded_the_authors_way_by_default(self):
        self.assertEqual(config.load(PRESETS[0], []).evaluation.arcwise_grading, "authors")
        self.assertEqual(config.load(PRESETS[0], ['evaluation.arcwise_grading="strict"']).evaluation.arcwise_grading,
                         "strict")
        with self.assertRaisesRegex(ValueError, "arcwise_grading"):
            config.load(PRESETS[0], ['evaluation.arcwise_grading="paper"'])

    def test_out_of_range_values_fail(self):
        cases = {
            "training.batch_size": 0, "training.num_train_epochs": -1, "training.group_size": 1,
            "execution.grading_concurrency": 0, "evaluation.concurrency": 0, "rollout.max_turns": 0,
            "training.learning_rate": 0.0, "training.adam_beta2": 1.0, "training.min_lr_ratio": 1.5,
            "training.epsilon_high": 0.5, "reward.process_penalty_weight": -0.1, "guards.collapse_cap_rate": 2.0,
        }
        for name, value in cases.items():
            section, key = name.split(".")
            raw = {"model": {"name": "m"}, section: {key: value}}
            with self.subTest(name), self.assertRaisesRegex(ValueError, f"invalid recipe: .*{key}"):
                config.from_dict(raw)

    def test_curriculum_practice_options_are_validated(self):
        options = {"evidence_weighting": "unaided", "evidence_failed_weight": 0.02, "evidence_solved_weight": 0.0625,
                   "evidence_fade_probability": 0.0, "evidence_draws_include_empty": False}
        recipe = config.from_dict({"model": {"name": "m"}, "curriculum": options})
        self.assertEqual({key: getattr(recipe.curriculum, key) for key in options}, options)
        # Overrides follow the same rules; the bounds are allowed, and an integer is read as a float.
        recipe = config.load(PRESETS[0], ['curriculum.evidence_weighting="unaided"',
                                          "curriculum.evidence_failed_weight=1",
                                          "curriculum.evidence_solved_weight=1",
                                          "curriculum.evidence_fade_probability=0",
                                          "curriculum.evidence_draws_include_empty=false"])
        self.assertEqual(recipe.curriculum.evidence_weighting, "unaided")
        self.assertIsInstance(recipe.curriculum.evidence_failed_weight, float)
        self.assertIsInstance(recipe.curriculum.evidence_solved_weight, float)
        self.assertEqual((recipe.curriculum.evidence_failed_weight, recipe.curriculum.evidence_solved_weight,
                          recipe.curriculum.evidence_fade_probability), (1.0, 1.0, 0.0))
        self.assertFalse(recipe.curriculum.evidence_draws_include_empty)
        self.assertEqual(config.load(PRESETS[0], ["curriculum.evidence_fade_probability=1"])
                         .curriculum.evidence_fade_probability, 1.0)
        out_of_range = {
            "evidence_weighting": ("Lane", "none", "evidence", ""),
            "evidence_failed_weight": (0, 0.0, -0.05, 1.5),
            "evidence_solved_weight": (0, 0.0, -0.05, 1.5),
            "evidence_fade_probability": (-0.1, 1.5),
        }
        for key, values in out_of_range.items():
            for value in values:
                raw = {"model": {"name": "m"}, "curriculum": {key: value}}
                with self.subTest(key=key, value=value), self.assertRaisesRegex(
                    ValueError, f"invalid recipe: .*curriculum.{key} must be"
                ):
                    config.from_dict(raw)
        wrong_type = {"evidence_weighting": 1, "evidence_failed_weight": "0.05", "evidence_solved_weight": True,
                      "evidence_fade_probability": True, "evidence_draws_include_empty": 1}
        for key, value in [*wrong_type.items(), ("evidence_fade_probability", float("nan")),
                           ("evidence_solved_weight", float("nan"))]:
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "wrong type or non-finite"):
                config.from_dict({"model": {"name": "m"}, "curriculum": {key: value}})

    def test_an_integer_is_accepted_for_a_float_setting(self):
        recipe = config.from_dict({"model": {"name": "m"}, "training": {"max_grad_norm": 5000}})
        self.assertEqual(recipe.training.max_grad_norm, 5000.0)
        self.assertIsInstance(recipe.training.max_grad_norm, float)

    def test_connection_needs_the_api_key(self):
        import os
        saved = os.environ.pop("RIVER_API_KEY", None)
        try:
            with self.assertRaises(SystemExit):
                config.connection()
            self.assertEqual(config.connection(require_key=False).base_url,
                             os.environ.get("RIVER_BASE_URL", config.DEFAULT_BASE_URL))
        finally:
            if saved is not None:
                os.environ["RIVER_API_KEY"] = saved

    def test_schema_sections_are_frozen_dataclasses(self):
        for f in dataclasses.fields(config.RecipeConfig):
            self.assertTrue(type(f.default_factory()).__dataclass_params__.frozen)


if __name__ == "__main__":
    unittest.main()
