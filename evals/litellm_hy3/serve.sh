#!/usr/bin/env bash
# Launch the hy3 q2 LiteLLM proxy INSIDE a guarded GPU window.
#
# This stops qwen + takes the exclusive MLX flock via
# scripts/run_with_qwen_stopped.py, then runs the litellm proxy from the
# CAMPAIGN venv (which has mtplx + mlx). The proxy holds the window until you
# stop it (Ctrl-C); qwen is restored automatically on exit.
#
# ONLY run this when the box is free (no live benchmark holding the lock).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"                       # eval worktree root
CAMPAIGN_VENV="${MTPLX_CAMPAIGN_VENV:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv}"
VENV_PY="$CAMPAIGN_VENV/bin/python"

HOST="${MTPLX_HY3_HOST:-127.0.0.1}"
PORT="${MTPLX_HY3_PORT:-18183}"                              # NOT 8080 (qwen)
PLIST="${QWEN_PLIST:-$HOME/Library/LaunchAgents/com.tea.qwen.plist}"
LOCK_PATH="${MTPLX_GPU_LOCK:-/tmp/mtplx-gpu-exclusive.lock}"
QWEN_API_URL="${QWEN_API_URL:-http://127.0.0.1:8080/v1/models}"

if [[ ! -x "$VENV_PY" ]]; then
  echo "campaign venv python not found at $VENV_PY" >&2
  exit 1
fi

# Champion decode environment (from the campaign handoff / docs reference).
# MTPLX_MEMORY_LIMIT_BYTES stays UNSET: the memory knob is enforced via the
# ExpertStreamingConfig memory_limit (103GiB), never via a second env override.
unset MTPLX_MEMORY_LIMIT_BYTES || true
export MTPLX_SUSTAINED_PREFILL="${MTPLX_SUSTAINED_PREFILL:-1}"
export MTPLX_DEFERRED_PIN_RELEASE="${MTPLX_DEFERRED_PIN_RELEASE:-1}"
export MTPLX_HY3_SUBMIT_CADENCE="${MTPLX_HY3_SUBMIT_CADENCE:-8}"

# 1. Install litellm[proxy] into the campaign venv. Only valid once the box is
#    free -- pip must not run while another lane holds the GPU under this venv.
echo "[serve] installing litellm[proxy] into campaign venv ..." >&2
"$VENV_PY" -m pip install 'litellm[proxy]'

# 2. Launch the proxy in the guarded window. --num_workers 1 is mandatory: one
#    uvicorn worker => exactly ONE MTPLXRuntime (memory guard).
echo "[serve] starting guarded window -> litellm proxy on ${HOST}:${PORT}" >&2
exec "$VENV_PY" "$REPO_ROOT/scripts/run_with_qwen_stopped.py" \
  --plist "$PLIST" \
  --api-url "$QWEN_API_URL" \
  --lock-path "$LOCK_PATH" \
  -- \
  "$VENV_PY" -m litellm \
    --config "$HERE/config.yaml" \
    --host "$HOST" \
    --port "$PORT" \
    --num_workers 1
