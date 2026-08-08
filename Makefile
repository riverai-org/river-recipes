.PHONY: setup check

# One-time per clone: install the pre-commit hooks that scan staged content.
setup:
	@command -v gitleaks >/dev/null || { echo "install gitleaks first: brew install gitleaks"; exit 1; }
	@command -v pre-commit >/dev/null || { echo "install pre-commit first: brew install pre-commit"; exit 1; }
	pre-commit install
	@echo "hooks installed"

# Full sweep: gitleaks over the entire git history + the public-vocabulary
# guard over all tracked files. CI runs the same two checks.
check:
	gitleaks git --redact --no-banner
	python3 scripts/check_public_vocabulary.py
	@echo "check passed"
