# Text-to-SQL RL recipe

Teaches a language model to answer questions over a SQLite database by exploring it with a SQL tool, then
committing one final query. Training is reinforcement learning on River: LoRA fine-tuning with CISPO, rewarded by
executing the model's query and comparing its result with the gold query's.

## Quickstart

You need Python 3.12 or newer, a River API key and about 90 GB of free disk.

```bash
pip install '.[data]'
export RIVER_API_KEY=...
eval "$(python -m text_to_sql_recipe setup data)"      # the data release, about 34 GB, checked file by file
python -m text_to_sql_recipe train --recipe nemotron-3.5-lightning --max-steps 1 --set 'reward.verifier="none"'
```

That trial trains one batch without VeriEQL and evaluates the model before and after it; the scores land in
`runs/<run name>/eval/<suite>-step-<batch>/summary.json`. Leave out `--max-steps` for a full run. Every preset uses
VeriEQL by default: [docs/verieql.md](docs/verieql.md) installs it, and [docs/kubernetes.md](docs/kubernetes.md) runs
the recipe as a Kubernetes Job.

## How it works

1. The model gets the database schema, the question and, where the dataset has it, supporting evidence.
2. It may call `execute_sql_query` up to five times. Each call runs read-only against the database, and the
   result comes back as the next message.
3. It then gives its final SQL in a `<solution>` block. The reward compares that query's result with the gold
   result. Small deductions apply when the VeriEQL equivalence checker proves a matching query differs from the gold
   query on another database, when the model skips the evidence checks the prompt asks for, and for each answer it
   re-drafts word for word. The Qwen presets also deduct for invalid tool calls.

Around the episode:

- **Curriculum.** Every question is shown its evidence. The first epoch covers every question once; later epochs draw
  a question by how informative its last results were: one solved about half the time most often, one last solved
  16/16 or failed 0/16 rarely (`curriculum.evidence_solved_weight`, `curriculum.evidence_failed_weight`). A question
  that every rollout fails is retried with its evidence and then, with `curriculum.hints`, with a training hint; only
  the final stage trains. The `curriculum` module documents the schedule in full. With `data.hint_mask` (on by
  default), 31 training questions whose gold queries do not answer them as written are left out, so neither
  their reward nor their hints teach a wrong answer; `data.py` lists them and why.
- **Stability.** A turn that hits its token cap while repeating itself is cut where the loop began and scored as no
  answer, with the loss masked past the cut. The gradient clip follows the run's recent gradient norms (twice the
  median of the last 20). Two rules stop a run:
  - **Collapse pause** (`guards`): after `guards.collapse_arm_after_steps` steps (33, about one epoch of the
    default training set), two steps in a row in which more than 5% of rollouts hit the token cap and more than 1% repeat
    themselves pause the run. It writes `collapse-pause.json` to the run directory and refuses to start again while
    that file exists: look at the run's recent rollouts, then delete the file and run the same command, with the same
    `--run-name`, to resume.
  - **Format floor:** a training batch in which fewer than half of the rollouts give a well-formed answer stops the
    run with an error, since the reward no longer measures the task.
- **Evaluation.** At the start, every `evaluation.eval_steps` steps, and at the end, the weights are evaluated
  greedily on the validation split and the Arcwise set (`evaluation.arcwise = false` skips Arcwise). Arcwise is
  graded as the ReViSQL authors grade it: an answer that is a single value within 1% of a single-value gold counts
  as correct. Each Arcwise summary also reports `eval/accuracy_strict`, without that tolerance, and
  `evaluation.arcwise_grading = "strict"` grades Arcwise that way throughout. Training and validation are always
  graded strictly.

## Presets

One recipe per model, installed with the package (`src/text_to_sql_recipe/recipes/`); pass a preset by name, or
pass the path to your own TOML file. Each preset sets only what differs for its model; everything else comes from the
defaults in `src/text_to_sql_recipe/config.py`, which are the settings every model shares.

| Preset | Model |
|---|---|
| `glm-5.3-flash` | GLM-5.3 Flash |
| `kimi-k2.6` | Kimi K2.6 |
| `qwen3.8-27b` | Qwen3.8-27B |
| `qwen3.6-35b-a3b` | Qwen3.6-35B-A3B |
| `deepseek-v4.1-flash` | DeepSeek V4.1 Flash |
| `nemotron-3.5-lightning` | Nemotron 3.5 Lightning |

Print the full resolved settings for a preset, with any overrides applied:

```bash
python -m text_to_sql_recipe plan --recipe glm-5.3-flash --set training.learning_rate=3e-5
```

## Data

`setup data` downloads the data release, the Hugging Face dataset `scottfitz/text-to-sql-recipe-data` at the commit
the package pins, into `TEXT_TO_SQL_DATA_DIR` (default `./data`). It then checks every file against the package's
pins and every database against the rows that name it. The release holds:

- `revisql-verified`: the BIRD-Platinum tasks every preset trains on, the ReViSQL authors' verified release of 2,064
  training and 398 validation tasks, with their training hints and BIRD's 69 training databases (a run trains
  on 2,033 of the training tasks unless `data.hint_mask = false`);
- `bird-platinum-hard`: a named subset of 689 harder training tasks (`data.train = "bird-platinum-hard"`);
- `arcwise`: the 498 Arcwise-Plat-SQL evaluation tasks, with the gold queries the ReViSQL paper evaluated, and the 11
  BIRD Mini-Dev databases.

The task data is licensed CC BY-SA 4.0, like BIRD; the dataset's card has the attributions and describes how each
part was made. `--source DIR` copies a local copy of the release instead of downloading it. Running the command again
resumes an interrupted download, and a file that fails its check is named so it can be deleted and fetched again.
The layout and the fields of a row are documented in `data.py`.

Besides the release itself (about 34 GB), a run keeps a private, read-only copy of each database it queries (about
34 GB more, under `TEXT_TO_SQL_SQLITE_SNAPSHOT_ROOT`; later runs reuse the copies), and grading the ten questions with
very large results streams them through temporary indexes under `TMPDIR`, which must have 20 GB free when a run
starts.

## Configuration

Settings use the Hugging Face Trainer and TRL names where one exists (`learning_rate`, `warmup_steps`,
`max_grad_norm`, `num_train_epochs`, `epsilon_high`, `eval_steps`). Unknown keys are rejected, so a typo fails
instead of silently keeping a default.

Connection and paths come from the environment:

| Variable | Meaning | Default |
|---|---|---|
| `RIVER_API_KEY` | River API key | required |
| `RIVER_BASE_URL` | River API endpoint | `https://api.river.ai` |
| `TEXT_TO_SQL_DATA_DIR` | where datasets are read from | `./data` |
| `TEXT_TO_SQL_OUTPUT_DIR` | where runs are written | `./runs` |
| `TEXT_TO_SQL_SQLITE_SNAPSHOT_ROOT` | where the private database copies are kept | a directory under the system temp directory |
| `TEXT_TO_SQL_VERIEQL_MANIFEST`, `TEXT_TO_SQL_VERIEQL_MANIFEST_SHA256` | the VeriEQL runtime and its pin ([docs/verieql.md](docs/verieql.md)) | required with VeriEQL |

## Running

```bash
python -m text_to_sql_recipe train --recipe <preset>                                  # with VeriEQL
python -m text_to_sql_recipe train --recipe <preset> --set 'reward.verifier="none"'   # without it
```

The process drives River over its API, so a laptop is enough. It needs river-client 0.13 or newer; `train` names any
client feature it lacks before it does any work, and also checks that the data directory holds the named datasets
and, with VeriEQL, that its runtime is set. `--max-steps N` stops after N training batches for a quick trial (a batch
whose groups all carry no learning signal makes no optimizer step), and `--run-name` picks the run directory under
`TEXT_TO_SQL_OUTPUT_DIR`.

Everything a run needs to continue is in its run directory, so running the same command again with the same
`--run-name` resumes it. The process also restarts itself from the saved state when the River service fails
transiently (a replaced inference worker, a dropped connection, a timeout), waiting longer each time and giving up
after six restarts without progress. An error that stops the run is also written to `last-failure.json` in the run
directory.

To run on a cluster, see [docs/kubernetes.md](docs/kubernetes.md).

## Outputs

A run directory (`runs/<run name>/`) holds:

- `eval/<suite>-step-<batch>/summary.json`: the scores of each evaluation, for the suites `validation` and `arcwise`,
  at the start, every `evaluation.eval_steps` batches and at the end. `eval/accuracy` is the headline number; on
  Arcwise it is graded the authors' way and `eval/accuracy_strict` without their tolerance. Next to it,
  `tasks/` holds one record per question (the conversation, the answer and its grade).
- `steps.jsonl`: one record per training batch, with its reward, accuracy, guard and optimizer metrics.
- `plan.json`: the resolved recipe and what the run is pinned to (the datasets' task digests and the code, client,
  tokenizer and verifier identities); a resumed run must match it.
- `checkpoint-<batch>.json` and `eval/<suite>-step-<batch>/checkpoint.json`: the River checkpoints saved at each
  evaluation, the training state and the inference weights. The weights stay on River; sample from them with
  `river_client.rl.CheckpointSampler` and a `river_client.Checkpoint` built from that file.
- `training-complete.json` when the run finishes, `last-failure.json` when an error stopped it, and
  `collapse-pause.json` when the collapse pause did.

The layout is documented in full in `train.py` and `evaluate.py`.

## Tests

```bash
pip install -e '.[test]'
python -m pytest
```

The suite runs offline and needs no API key: it covers each module on small fixture databases, and an end-to-end
test trains on a scripted model.

## Credits

- The system prompt is the one the recipe's published runs were trained with. It follows the prompt of the ReViSQL
  paper (Human-Level Text-to-SQL via Reinforcement Learning on Verified Data, Without Pipeline Engineering,
  arXiv:2603.20004), whose two evidence directives it quotes; its opening and worked example are adapted from
  SkyRL-SQL (NovaSky-AI/SkyRL, Apache-2.0), and its schema note comes from OmniSQL (RUCKBReasoning/OmniSQL).
- The training and validation tasks are the ReViSQL authors' verified BIRD tasks (BIRD-Platinum), and the evaluation
  set is Arcwise-Plat-SQL with the gold queries their paper evaluated; the dataset card has the full attributions.
