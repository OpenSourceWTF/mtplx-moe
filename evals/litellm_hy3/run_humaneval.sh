#!/usr/bin/env bash
# Run HumanEval pass@1 against the running hy3 q2 proxy.
#
# Run this in a SECOND shell while serve.sh holds the guarded window. This is
# a plain HTTP client -- it does NOT take the GPU lock itself.
#
# NOTE: --base-url has NO /v1 suffix. code_eval_gate appends
# "/v1/chat/completions" itself (endpoint_url); adding /v1 here would double it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
CAMPAIGN_VENV="${MTPLX_CAMPAIGN_VENV:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv}"
VENV_PY="$CAMPAIGN_VENV/bin/python"

PORT="${MTPLX_HY3_PORT:-18183}"
LIMIT="${HUMANEVAL_LIMIT:-20}"
# The one HumanEval jsonl present on this box with the fields the loader needs
# (task_id, prompt, canonical_solution, test, entry_point). Override with a
# canonical HumanEval.jsonl via HUMANEVAL_DATASET if you have one.
DATASET="${HUMANEVAL_DATASET:-$HOME/Library/Caches/evalplus/HumanEvalPlus-v0.1.10.jsonl}"
OUTPUT_JSON="${HUMANEVAL_OUTPUT:-$REPO_ROOT/evals/humaneval_hy3_q2.json}"

if [[ ! -f "$DATASET" ]]; then
  echo "HumanEval dataset not found at $DATASET (set HUMANEVAL_DATASET)" >&2
  exit 1
fi

exec "$VENV_PY" "$REPO_ROOT/scripts/code_eval_gate.py" \
  --base-url "http://127.0.0.1:${PORT}" \
  --model hy3-q2 \
  --suite humaneval \
  --dataset-path "$DATASET" \
  --limit "$LIMIT" \
  --endpoint chat \
  --max-tokens "${HUMANEVAL_MAX_TOKENS:-1024}" \
  --temperature "${HUMANEVAL_TEMPERATURE:-0.0}" \
  --seed "${HUMANEVAL_SEED:-42}" \
  --output-json "$OUTPUT_JSON" \
  --progress \
  --allow-code-execution
