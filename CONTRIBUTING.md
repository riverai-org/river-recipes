# Contributing

Thanks for improving River's examples. Two minutes of setup keeps this repo
safe to work in.

## Setup

```bash
brew install gitleaks pre-commit   # or your platform's equivalent
make setup                         # installs the pre-commit hooks
```

This installs two hooks: a commit-time scan of staged content, and a
push-time re-scan of every outgoing commit (the backstop if a commit was made
before setup or with `--no-verify`). `make check` runs the same scans over
the whole repository and its full git history.

## The one hard rule: nothing private, ever, in any commit

This is a public repository. Git history is permanent — on GitHub, even
commits on deleted branches and closed pull requests remain fetchable
forever. There is no such thing as "committed briefly."

Concretely, that means no:

- credentials of any kind — API keys, tokens, kubeconfigs, certificates.
  Examples read `RIVER_API_KEY` from the environment; Kubernetes manifests
  reference Secrets via `secretKeyRef`, with a committed `*.example.yaml`
  placeholder template the user fills in.
- non-public infrastructure vocabulary: hostnames, cluster or context names,
  container registries, personal namespaces, absolute home-directory paths.
  Use `*.river.ai`, `localhost`, cluster-local names, RFC 2606 example
  domains, and documentation IP ranges (`192.0.2.0/24`).

CI enforces both with a secret scan over full history and an
allowlist-based vocabulary guard (`scripts/check_public_vocabulary.py`). If
the guard blocks something genuinely public, extend its allowlist in the same
PR and say so. Maintainers additionally run a supplemental internal pattern
scan that is intentionally not part of this repository.

Write content fresh in this repository rather than copying from private
codebases and cleaning up afterward — pre-cleanup states have a way of
getting committed. Use the untracked `scratch/` directory for staging
material.

## If a secret does get pushed

Treat it as compromised from the moment the push succeeded — harvesting bots
watch public repos in real time, and history rewrites don't unpublish
anything on GitHub.

1. **Rotate the credential immediately.** This is the only real remediation.
2. Remove it from the tree in a new commit.
3. Open an issue (or contact the maintainers privately if sensitive) so the
   exposure is tracked; maintainers will request cache purging via GitHub
   Support.

## Pull requests

- Keep examples self-contained: data preparation, training, evaluation, and
  a README that runs end to end with only a River API key.
- CI must be green — including the guard workflow — before review.
