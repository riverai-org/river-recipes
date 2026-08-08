#!/usr/bin/env bash
# Pre-push backstop. The pre-commit hook can be skipped (--no-verify, a clone
# that never ran `make setup`); this re-scans every commit that is about to
# leave the machine — the boundary after which a leaked value is permanently
# public and must be rotated. Scans all local commits not already on a remote.
set -uo pipefail

PRIVATE_CONFIG="${RIVER_EXAMPLES_GITLEAKS_PRIVATE:-$HOME/.config/river-examples/gitleaks-private.toml}"
RANGE="--all --not --remotes"

if ! command -v gitleaks >/dev/null 2>&1; then
    echo "scan-push: gitleaks is not installed (brew install gitleaks)" >&2
    exit 1
fi

fail=0

if ! gitleaks git --redact --no-banner --max-archive-depth 2 --log-opts "$RANGE"; then
    fail=1
fi

if [[ -f "$PRIVATE_CONFIG" ]]; then
    if ! gitleaks git --redact --no-banner --max-archive-depth 2 --config "$PRIVATE_CONFIG" --log-opts "$RANGE"; then
        fail=1
    fi
fi

if ! python3 scripts/check_public_vocabulary.py; then
    fail=1
fi

if [[ "$fail" -ne 0 ]]; then
    cat >&2 <<'EOF'

scan-push: BLOCKED — an outgoing commit contains a secret or internal
identifier. Nothing has left this machine yet, so this is fully recoverable:
rewrite the offending commit(s) locally (git rebase -i / git reset), then push
again. Do NOT bypass this hook — once pushed, the value is permanently
fetchable and must be rotated. See CONTRIBUTING.md.
EOF
    exit 1
fi
