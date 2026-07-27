#!/bin/zsh
# Supported Laguna-S-2.1 oQ4e dual-lane server launcher.
#
# Blackwellboy's operational changes were applied to make the Laguna serving path work correctly.
# This attribution covers the startup and serving guards, not the fixed-M2
# kernel or its benchmark result.
set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_ROOT=${MTPLX_REPO_ROOT:-${SCRIPT_DIR:h}}

PYTHON=${MTPLX_PYTHON:-python3}
MODEL=${MTPLX_LAGUNA_MODEL:-mlx-community/Laguna-S-2.1-oQ4e}
HOST=${MTPLX_LAGUNA_HOST:-127.0.0.1}
PORT=${MTPLX_LAGUNA_PORT:-8080}
MIN_AVAIL_GIB=${MTPLX_LAGUNA_MIN_AVAIL_GIB:-60}
MEM_WAIT_S=${MTPLX_LAGUNA_MEM_WAIT_S:-90}
READY_TIMEOUT_S=${MTPLX_LAGUNA_READY_TIMEOUT_S:-300}

log() {
  print -u2 -- "[start-laguna-s21] $(date '+%F %T') $*"
}

fatal() {
  log "FATAL: $*"
  exit 1
}

require_positive_integer() {
  local name=$1
  local value=$2
  [[ "$value" == <-> ]] && (( value > 0 )) ||
    fatal "$name must be a positive integer"
}

require_positive_integer MTPLX_LAGUNA_PORT "$PORT"
require_positive_integer MTPLX_LAGUNA_MIN_AVAIL_GIB "$MIN_AVAIL_GIB"
require_positive_integer MTPLX_LAGUNA_MEM_WAIT_S "$MEM_WAIT_S"
require_positive_integer MTPLX_LAGUNA_READY_TIMEOUT_S "$READY_TIMEOUT_S"
[[ -n "$HOST" ]] || fatal "MTPLX_LAGUNA_HOST must not be empty"
[[ -n "$MODEL" ]] || fatal "MTPLX_LAGUNA_MODEL must not be empty"

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Exact full-stack routes. Measured-off routes are pinned so ambient shell
# state cannot change the supported construction contract.
export MTPLX_LAGUNA_FUSED_GATE_UP=1
export MTPLX_LAGUNA_FUSED_SHARED_GATE_UP=1
export MTPLX_LAGUNA_KERNEL_ROUTER=1
export MTPLX_LAGUNA_KERNEL_ATTN_GATE=1
export MTPLX_LAGUNA_KERNEL_QK_ROPE=1
export MTPLX_LAGUNA_KERNEL_COMBINE=1
export MTPLX_LAGUNA_KERNEL_ROUTER_GEMV=1
export MTPLX_LAGUNA_FUSED_RESIDUAL_NORM=0
export MTPLX_LAGUNA_CACHED_LHS=0
export MTPLX_LAGUNA_FUSED_QKVG=0

# Promoted after the exact parity, paired timing, and dispatch-census gates.
# Installation errors are construction-time fatal errors and remain uncaught.
export MTPLX_LAGUNA_FIXED_M2_ROUTER=1

# Native tool calls can carry large hidden argument payloads.
export MTPLX_STREAM_HIDDEN_TOOL_GUARD_TOKENS=${MTPLX_STREAM_HIDDEN_TOOL_GUARD_TOKENS:-16384}
export MTPLX_STREAM_HIDDEN_TOOL_GUARD_S=${MTPLX_STREAM_HIDDEN_TOOL_GUARD_S:-600}

COMMAND=(
  "$PYTHON" "-m" "mtplx.server.openai"
  "--model" "$MODEL"
  "--backend-id" "laguna_ar"
  "--host" "$HOST" "--port" "$PORT"
  "--context-window" "131072"
  "--reasoning" "off"
  "--reasoning-parser" "poolside_v1"
  "--tool-prompt-mode" "native"
  "--generation-mode" "ar" "--no-load-mtp"
  "--rate-limit" "0" "--stream-interval" "1"
  "--scheduler-mode" "ar_batch"
  "--batching-preset" "latency"
  "--max-active-requests" "2"
  "--decode-batch-max" "2"
  "--prefill-chunk-tokens" "1024"
  "--batch-wait-ms" "0"
  "--warmup-tokens" "16"
  "--model-id" "mtplx-laguna-s21-oq4e"
  "--no-stats-footer"
)

if (( $# > 1 )) || (( $# == 1 )) && [[ "$1" != "--print-config" ]]; then
  print -u2 -- "usage: ${0:t} [--print-config]"
  exit 2
fi

if (( $# == 1 )); then
  print -- "repository_root=$REPO_ROOT"
  print -- "python=$PYTHON"
  print -- "model=$MODEL"
  print -- "bind=$HOST:$PORT"
  print -- "minimum_available_gib=$MIN_AVAIL_GIB"
  print -- "MTPLX_LAGUNA_FIXED_M2_ROUTER=1"
  printf "command="
  printf "%q " "${COMMAND[@]}"
  print
  exit 0
fi

# Guard 1: refuse both a bound duplicate and an unbound loading duplicate.
if lsof -nP -iTCP@"${HOST}:${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  HOLDER=$(lsof -nP -t -iTCP@"${HOST}:${PORT}" -sTCP:LISTEN | head -1)
  fatal "${HOST}:${PORT} already has a listener (pid ${HOLDER:-unknown})"
fi

SERVER_PATTERN="[m]tplx.server.openai .*--port ${PORT}"
EXISTING=$(pgrep -f "$SERVER_PATTERN" || true)
if [[ -n "$EXISTING" ]]; then
  fatal "an unbound mtplx server already targets port ${PORT}: ${EXISTING//$'\n'/,}"
fi

# Guard 2: wait a bounded interval for reclaimable unified memory.
avail_gib() {
  vm_stat | awk -v ps="$(pagesize)" '
    /Pages free/        {free=$3}
    /Pages inactive/    {inact=$3}
    /Pages speculative/ {spec=$3}
    /Pages purgeable/   {purg=$3}
    END {
      gsub(/\./,"",free); gsub(/\./,"",inact)
      gsub(/\./,"",spec); gsub(/\./,"",purg)
      print int((free+inact+spec+purg)*ps/1073741824)
    }'
}

MEM_DEADLINE=$(( SECONDS + MEM_WAIT_S ))
AVAIL=$(avail_gib)
while (( AVAIL < MIN_AVAIL_GIB && SECONDS < MEM_DEADLINE )); do
  sleep 5
  AVAIL=$(avail_gib)
done
if (( AVAIL < MIN_AVAIL_GIB )); then
  fatal "only ${AVAIL} GiB reclaimable after ${MEM_WAIT_S}s (< ${MIN_AVAIL_GIB})"
fi

# Guard 3: prove this checkout wins import resolution before loading weights.
if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
  fatal "Python executable not found: $PYTHON"
fi
RESOLVED=$("$PYTHON" -c 'import mtplx; print(mtplx.__file__)')
case "$RESOLVED" in
  "$REPO_ROOT"/*) ;;
  *) fatal "mtplx resolved to $RESOLVED, not $REPO_ROOT" ;;
esac

log "guards passed: model=$MODEL bind=$HOST:$PORT available=${AVAIL}GiB"
log "scheduler=ar_batch active=2 decode_batch=2 prefill_chunk=1024 wait_ms=0 fixed_m2=1"

"${COMMAND[@]}" &
SERVER_PID=$!

stop_child() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
}
trap stop_child TERM INT HUP

# Guard 4: /health becomes reachable only after load and warmup. A fixed-M2
# construction error exits the child and is returned as a fatal startup result.
READY_DEADLINE=$(( SECONDS + READY_TIMEOUT_S ))
until curl -sf -m 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    if wait "$SERVER_PID"; then
      RC=0
    else
      RC=$?
    fi
    log "server died during startup (rc=$RC)"
    exit "$RC"
  fi
  if (( SECONDS >= READY_DEADLINE )); then
    log "not ready after ${READY_TIMEOUT_S}s; terminating pid $SERVER_PID"
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    exit 1
  fi
  sleep 5
done
log "ready on ${HOST}:${PORT} (pid $SERVER_PID)"

if wait "$SERVER_PID"; then
  RC=0
else
  RC=$?
fi
log "server exited (rc=$RC)"
exit "$RC"
