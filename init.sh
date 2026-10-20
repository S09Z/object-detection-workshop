#!/usr/bin/env bash
# Session start check (harness): env in sync, lint clean, tests green, recent history.
# Run at the start of every session:  ./init.sh
set -euo pipefail

cd "$(dirname "$0")"
echo "== repo: $(pwd)"

if [ ! -d .git ]; then
  echo "!! not a git repo yet — run: make setup"
  exit 1
fi

echo "== uv sync"
uv sync --all-groups --quiet

echo "== lint"
uv run ruff check . --quiet
uv run ruff format --check . --quiet

echo "== tests (CPU)"
uv run pytest -m "not gpu" -q

echo "== recent commits"
git log --oneline -10 || true

echo "== current state (PROGRESS.md)"
grep -E "^\| (Current sub-task|Status|Blockers) " PROGRESS.md || true

echo "== latest decisions (DECISIONS.md)"
grep -E "^\| D-[0-9]+" DECISIONS.md | tail -3 || true

echo "== next sub-task in PLAN.md"
grep -n -m1 -- "- \[ \] \*\*phase-" PLAN.md || echo "all sub-tasks done"

echo "== OK — environment is clean"
