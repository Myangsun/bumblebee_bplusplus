#!/bin/bash
# Pre-commit hook template: run thesis-claim verifier and block commit on FAIL.
#
# To install:
#   ln -s ../../scripts/pre-commit-verify-thesis.sh .git/hooks/pre-commit
# Or append this logic to an existing pre-commit hook.
#
# Only runs if a thesis-relevant file is staged — commits that don't touch
# the thesis or its numerical sources skip this check.
#
# The verifier reads docs/thesis_main.md and docs/final_metrics.md, so any
# staged change to either one must trigger verification. The legacy
# docs/thesis_draft.md is still included for backward compatibility with
# pre-split branches.

set -e

if ! git diff --cached --name-only \
    | grep -qE "^docs/(thesis_main|thesis_draft|final_metrics)\.md$"; then
  exit 0
fi

echo "pre-commit: verifying thesis claims against source data..."

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# Prefer the project venv if it exists; fall back to system python.
if [ -x "venv/bin/python" ]; then
  PY="venv/bin/python"
else
  PY="python3"
fi

if ! "$PY" scripts/verify_thesis.py --strict; then
  echo ""
  echo "❌ Thesis verifier found FAIL claims. See docs/verify_report.md."
  echo "   Fix the flagged claims or regenerate sources (e.g."
  echo "   python scripts/dump_final_metrics.py) before committing."
  exit 1
fi

echo "✓ thesis verification passed"
