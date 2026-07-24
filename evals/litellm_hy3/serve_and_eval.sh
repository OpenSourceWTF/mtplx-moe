#!/usr/bin/env bash
# INNER command for the guarded HumanEval window.
#
# The CALLER (run_humaneval_guarded.sh -> run_with_qwen_stopped.py) already
# holds the exclusive MLX flock and has stopped qwen. This script therefore
# runs entirely INSIDE the lock: it starts the litellm proxy, waits for it to
# be up (proxy startup does NOT load the model), then runs code_eval_gate whose
# first request triggers the ONE model load + generation (GPU work, lock held),
# then tears the proxy down. No GPU work escapes the lock.
#
# Serve env = isolated litellm venv python + PYTHONPATH(eval worktree :
# campaign site-packages). Validated import stack (TEST B): litellm 1.93.0 +
# mlx + transformers + mtplx + generate_mtpk, campaign venv untouched.
set -uo pipefail

ISOPY="${MTPLX_LITELLM_PY:-/private/tmp/claude-501/-Users-davidtai-projects-OpenSourceWTF/5bc9942c-42b5-4300-bc57-4b4f43ee476e/scratchpad/litellm-serve-venv/bin/python}"
EVALWT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
CAMPSP="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/lib/python3.12/site-packages"
HERE="$EVALWT/evals/litellm_hy3"
export PYTHONPATH="$EVALWT:$CAMPSP"

PORT="${MTPLX_HY3_PORT:-18183}"
DATASET="${HUMANEVAL_DATASET:-$HOME/Library/Caches/evalplus/HumanEvalPlus-v0.1.10.jsonl}"
LIMIT="${HUMANEVAL_LIMIT:-20}"
OUT="${HUMANEVAL_OUTPUT:-$EVALWT/evals/tier2/humaneval_hy3_q2.json}"
PROXY_LOG="${MTPLX_HY3_PROXY_LOG:-$EVALWT/evals/tier2/humaneval_proxy.log}"

# Champion decode env; memory knob stays inside ExpertStreamingConfig (103GiB),
# never a second env override.
unset MTPLX_MEMORY_LIMIT_BYTES || true
export MTPLX_SUSTAINED_PREFILL="${MTPLX_SUSTAINED_PREFILL:-1}"
export MTPLX_DEFERRED_PIN_RELEASE="${MTPLX_DEFERRED_PIN_RELEASE:-1}"
export MTPLX_HY3_SUBMIT_CADENCE="${MTPLX_HY3_SUBMIT_CADENCE:-8}"

mkdir -p "$(dirname "$OUT")"
[[ -f "$DATASET" ]] || { echo "[serve_and_eval] dataset missing: $DATASET" >&2; exit 4; }

echo "[serve_and_eval] starting litellm proxy on 127.0.0.1:$PORT (num_workers=1)" >&2
# litellm has no __main__; use the console script (smoke-validated).
LITELLM_BIN="$(dirname "$ISOPY")/litellm"
"$LITELLM_BIN" --config "$HERE/config.yaml" --host 127.0.0.1 --port "$PORT" --num_workers 1 \
  > "$PROXY_LOG" 2>&1 &
PROXY=$!
trap 'kill "$PROXY" 2>/dev/null; wait "$PROXY" 2>/dev/null' EXIT

# Wait for the proxy to list the model (no model load yet).
up=0
for _ in $(seq 1 90); do
  if curl -s -m3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q 'hy3-q2'; then up=1; break; fi
  kill -0 "$PROXY" 2>/dev/null || { echo "[serve_and_eval] proxy died during startup; log:" >&2; tail -30 "$PROXY_LOG" >&2; exit 3; }
  sleep 2
done
[[ "$up" == 1 ]] || { echo "[serve_and_eval] proxy never listed hy3-q2; log:" >&2; tail -30 "$PROXY_LOG" >&2; exit 5; }
echo "[serve_and_eval] proxy up; running HumanEval limit=$LIMIT (first request loads the model)" >&2

"$ISOPY" "$EVALWT/scripts/code_eval_gate.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --model hy3-q2 \
  --suite humaneval \
  --dataset-path "$DATASET" \
  --limit "$LIMIT" \
  --endpoint chat \
  --max-tokens "${HUMANEVAL_MAX_TOKENS:-1024}" \
  --temperature "${HUMANEVAL_TEMPERATURE:-0.0}" \
  --seed "${HUMANEVAL_SEED:-42}" \
  --output-json "$OUT" \
  --progress \
  --allow-code-execution
RC=$?
echo "[serve_and_eval] code_eval_gate exit=$RC" >&2
exit $RC
