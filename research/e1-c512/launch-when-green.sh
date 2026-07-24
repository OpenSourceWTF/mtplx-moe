#!/bin/bash
# Green-gate retry launcher for the E1 window. Race-classification fixed:
# only lines written AFTER the current attempt started count.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/e1-run-20260721.log"
GPU_LOCK=/tmp/mtplx-gpu-exclusive.lock
MODEL_ID="mtplx-qwen36-27b-optimized-speed"

green() {
    lsof -t "$GPU_LOCK" >/dev/null 2>&1 && return 1
    curl -s -m 4 http://127.0.0.1:8080/v1/models | grep -q "$MODEL_ID" || return 1
    return 0
}

for attempt in $(seq 1 10); do
    until green && sleep 10 && green; do sleep 30; done
    MARK=$(wc -l < "$LOG")
    echo "$(date '+%F %T') launcher: box green (attempt $attempt); starting E1 window" >> "$LOG"
    bash "$DIR/run-e1-candidate-only.sh" >> "$LOG" 2>&1
    RC=$?
    if [ $RC -eq 0 ]; then
        echo "$(date '+%F %T') launcher: E1 window completed rc=0" >> "$LOG"
        exit 0
    fi
    if tail -n +$((MARK + 1)) "$LOG" | grep -q "ambiguous or does not expose"; then
        echo "$(date '+%F %T') launcher: wrapper refused (raced a window); backing off 120s" >> "$LOG"
        sleep 120
        continue
    fi
    echo "$(date '+%F %T') launcher: E1 failed rc=$RC for a non-race reason; NOT retrying" >> "$LOG"
    exit "$RC"
done
echo "$(date '+%F %T') launcher: gave up after 10 attempts" >> "$LOG"
exit 3
