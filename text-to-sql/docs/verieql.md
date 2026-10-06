# VeriEQL

Every preset uses the verifier deduction (`reward.verifier = "verieql"`): a matching answer that
[VeriEQL](https://github.com/VeriEQL/VeriEQL), a bounded SQL equivalence checker, proves different from the gold
query scores `1 - reward.verifier_weight` (0.8) instead of 1. The counterexample gate (`reward.verifier_gate = true`,
off in every preset) makes a refutation count only when the database VeriEQL constructs to show the difference is
replayed in SQLite and the two queries really return different results there; otherwise the answer keeps its full
reward.

VeriEQL is licensed separately, under CC BY-NC-SA 4.0 (non-commercial use only), and is not part of this package. To
train without it, add `--set 'reward.verifier="none"'` to `train` or `render-job`.

## Installing it

One command installs it once you have accepted that licence:

```bash
pip install '.[verieql]'                     # sqlglot, needed only with reward.verifier_gate = true
python -m text_to_sql_recipe setup verieql
```

`setup verieql` shows the licence and asks you to accept it before it downloads anything; `--accept-license` accepts
without the prompt. It then:

1. fetches the VeriEQL revision the recipe pins from GitHub and checks every file against the pin;
2. builds a Python 3.11 environment of its own with z3-solver and VeriEQL's other dependencies, each pinned by version
   and SHA-256;
3. writes the runtime manifest the recipe validates at startup;
4. runs a self-check of known verdicts through the recipe's own verifier;
5. prints the two variables to set:

```bash
export TEXT_TO_SQL_VERIEQL_MANIFEST=<prefix>/runtime-manifest.json
export TEXT_TO_SQL_VERIEQL_MANIFEST_SHA256=<SHA-256 of the manifest>
```

The variables go to standard output and everything else to standard error, so
`eval "$(python -m text_to_sql_recipe setup verieql --accept-license)"` sets them in the current shell.

## Requirements and limits

- The command needs [uv](https://docs.astral.sh/uv/), which also supplies the Python 3.11 interpreter, or `--python`
  naming a CPython 3.11 interpreter (pip then installs).
- It installs under your user cache directory unless you pass `--prefix`, and reuses what is already there when run
  again.
- It supports Linux on aarch64 and x86_64 and macOS on Apple silicon. The manifest records the platform, and the
  recipe refuses a manifest written on another one.
- The manifest also pins the system libraries VeriEQL loads, so run the command again after an operating-system
  update. A run records the manifest's SHA-256 in its `plan.json`, so a run interrupted across such an update cannot
  resume with the new runtime.
- Each check runs in a process of its own, and a hard query can grow past 1.4 GB of memory before its 30-second
  deadline. A check that is killed, by the out-of-memory killer for example, is an error rather than a verdict, so
  leave memory for `execution.verifier_workers` of them at once.

## On Kubernetes

A Job rendered for a recipe that uses VeriEQL needs `--verieql-manifest` and `--verieql-manifest-sha256` (paths inside
the image or volume); `render-job` refuses to render it without them. The manifest names the paths of the installation
it describes, so run `setup verieql` where the Job runs: while building the image (see [kubernetes.md](kubernetes.md)),
or with `--prefix` on the volume.
