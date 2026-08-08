#!/usr/bin/env bash
# Pre-commit secret scan. Two passes:
#   1. gitleaks with the repository config (.gitleaks.toml).
#   2. gitleaks with a supplemental pattern file, if one is present on this
#      machine (maintainers only; see CONTRIBUTING.md). Its absence is fine.
#
# Runs against STAGED content only, so it is fast and checks exactly what the
# commit would contain. Findings are redacted in output.
set -uo pipefail

PRIVATE_CONFIG="${RIVER_EXAMPLES_GITLEAKS_PRIVATE:-$HOME/.config/river-examples/gitleaks-private.toml}"

if ! command -v gitleaks >/dev/null 2>&1; then
    echo "scan-staged: gitleaks is not installed (brew install gitleaks)" >&2
    exit 1
fi

fail=0

if ! gitleaks git --pre-commit --staged --redact --no-banner; then
    fail=1
fi

if [[ -f "$PRIVATE_CONFIG" ]]; then
    if ! gitleaks git --pre-commit --staged --redact --no-banner --config "$PRIVATE_CONFIG"; then
        fail=1
    fi
else
    echo "scan-staged: no supplemental pattern file (optional; maintainers see CONTRIBUTING.md)"
fi

if [[ "$fail" -ne 0 ]]; then
    cat >&2 <<'EOF'

scan-staged: BLOCKED — staged changes look like they contain a secret or an
internal identifier. Do not commit. If this is a false positive, adjust
.gitleaks.toml allowlists in this change and explain in the PR. If it is real,
the value must never be committed: keep it in the environment or a *.example.*
placeholder. A secret that has already been PUSHED must be rotated —
see CONTRIBUTING.md.
EOF
    exit 1
fi
