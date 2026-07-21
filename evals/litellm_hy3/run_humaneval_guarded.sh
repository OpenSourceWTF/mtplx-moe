#!/usr/bin/env bash
# OUTER launcher: run HumanEval against hy3 q2 inside a guarded GPU window.
#
# Wraps serve_and_eval.sh with scripts/run_with_qwen_stopped.py, which acquires
# the exclusive MLX flock (BLOCKING until free) and stops/restores qwen. The
# flock is held for the entire serve+eval lifetime, so the model load never
# happens outside the lock.
#
# Launch this via the Bash tool with run_in_background:true (NEVER shell `&` --
# a reaped process group leaves qwen dead). It queues behind any active lane.
set -uo pipefail

ISOPY="${MTPLX_LITELLM_PY:-/private/tmp/claude-501/-Users-davidtai-projects-OpenSourceWTF/5bc9942c-42b5-4300-bc57-4b4f43ee476e/scratchpad/litellm-serve-venv/bin/python}"
EVALWT="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.worktrees/eval-hy3-q2-2p6bit"
CAMPSP="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/lib/python3.12/site-packages"
HERE="$EVALWT/evals/litellm_hy3"
export PYTHONPATH="$EVALWT:$CAMPSP"

PLIST="${QWEN_PLIST:-$HOME/Library/LaunchAgents/com.tea.qwen.plist}"
LOCK_PATH="${MTPLX_GPU_LOCK:-/tmp/mtplx-gpu-exclusive.lock}"
QWEN_API_URL="${QWEN_API_URL:-http://127.0.0.1:8080/v1/models}"
LOCK_TIMEOUT="${MTPLX_LOCK_TIMEOUT:-7200}"

exec "$ISOPY" "$EVALWT/scripts/run_with_qwen_stopped.py" \
  --plist "$PLIST" \
  --api-url "$QWEN_API_URL" \
  --lock-path "$LOCK_PATH" \
  --lock-timeout-seconds "$LOCK_TIMEOUT" \
  -- \
  bash "$HERE/serve_and_eval.sh"
