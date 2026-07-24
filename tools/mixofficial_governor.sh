#!/bin/bash
# Governor-gated full mixed-official conversion burn (issue #51, chain item 2).
#
# LAW (campaign handoff): the ~87.4 GiB burn runs ONLY while qwen serves AND
# no benchmark holds the GPU lock. This wrapper enforces it: it pauses the
# converter (SIGTERM; the resume journal makes the restart byte-identical)
# whenever a benchmark window opens, qwen stops serving, or free disk drops
# under the floor, and resumes when the box is green again.
#
# CPU + disk only. The GPU lock is checked READ-ONLY via lsof -- this script
# never acquires it and never signals any process other than its own child.
set -u

WT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python"
OUT="$HOME/.cache/huggingface/hy3-expert-only-mlx-mixofficial"
LOG_DIR="$WT/research/mixofficial_burn"
LOG="$LOG_DIR/burn-$(date +%Y%m%d-%H%M%S).log"
GPU_LOCK="/tmp/mtplx-gpu-exclusive.lock"
MIN_FREE_GIB=60
POLL_RUN=30
POLL_WAIT=60

mkdir -p "$LOG_DIR"
say() { echo "$(date '+%F %T') governor: $*" >> "$LOG"; }

conditions_ok() {
    # 1. no benchmark window: nobody holds an fd on the GPU lock
    if lsof -t "$GPU_LOCK" >/dev/null 2>&1; then return 1; fi
    # 2. qwen serves (normal box state; a stopped qwen means a guarded window)
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' \
        http://127.0.0.1:8080/v1/models)" = "200" ] || return 1
    # 3. disk floor
    local free
    free=$(df -g /Users/davidtai | awk 'NR==2{print $4}')
    [ "$free" -ge "$MIN_FREE_GIB" ] || return 1
    return 0
}

if [ "${1:-}" = "--check" ]; then
    if conditions_ok; then echo "conditions: GREEN"; exit 0; fi
    echo "conditions: RED"; exit 1
fi

if pgrep -f convert_expert_mixed_official.py >/dev/null; then
    say "another converter is already running; refusing double launch"
    echo "refusing double launch (see $LOG)" >&2
    exit 2
fi

say "burn start (governor pid $$, output $OUT)"
while [ ! -f "$OUT/expert-manifest.json" ]; do
    until conditions_ok; do
        say "box not green (benchmark window / qwen down / disk floor); waiting"
        sleep "$POLL_WAIT"
    done
    say "launching converter (--resume)"
    PYTHONPATH="$WT" nice -n 10 "$PY" \
        "$WT/scripts/convert_expert_mixed_official.py" \
        --resume --verify-sample 8 >> "$LOG" 2>&1 &
    CHILD=$!
    KILLED=0
    while kill -0 "$CHILD" 2>/dev/null; do
        sleep "$POLL_RUN"
        if ! conditions_ok && kill -0 "$CHILD" 2>/dev/null; then
            say "conditions went red; pausing converter (SIGTERM $CHILD)"
            KILLED=1
            kill -TERM "$CHILD" 2>/dev/null
            for _ in $(seq 1 30); do
                kill -0 "$CHILD" 2>/dev/null || break
                sleep 1
            done
            if kill -0 "$CHILD" 2>/dev/null; then
                say "converter ignored TERM for 30s; SIGKILL (torn tail is resume-safe)"
                kill -KILL "$CHILD" 2>/dev/null
            fi
        fi
    done
    wait "$CHILD"
    RC=$?
    [ -f "$OUT/expert-manifest.json" ] && break
    if [ "$KILLED" = "0" ]; then
        say "converter exited rc=$RC without publishing and without a pause; NOT retrying"
        exit 1
    fi
    say "paused at rc=$RC; will resume when green"
done
say "burn complete: $OUT/expert-manifest.json published"
exit 0
