#!/usr/bin/env bash
#
# Model-scale gate for issue #113: streamed rANS miss-read bandwidth win.
#
# Runs the SAME strict-90g generation twice -- once reading records
# uncompressed, once reading per-record rANS containers (streamed_codec=
# rans32x-v1) and decoding them in-kernel -- and reports decode tok/s and
# bytes-read/token for both. Everything else (islands, cache, prefetch,
# memory limit, prompt, tokens) is whatever you pass through after `--`, so
# the two arms differ ONLY in the codec toggle.
#
# This is a GUARDED-WINDOW script: it is David's to run, not the agent's. It
# never backgrounds a run (panic-protocol: a shell-& guarded wrapper gets its
# process group reaped) and refuses to start if another streamed run is live.
#
# Usage:
#   scripts/gate_streamed_rans_90g.sh \
#       <MODEL_ROOT> <BASE_MANIFEST> <CODEC_MANIFEST> \
#       -- <your exact 90g benchmark_streamed_generation args...>
#
# Example (fill in your real strict-90g invocation after the --):
#   scripts/gate_streamed_rans_90g.sh \
#       ~/models/hy3-q2 ~/models/hy3-q2/expert-manifest.json \
#       ~/models/hy3-q2/expert-streamed-codec-rans32x.json \
#       -- --model-key hy3-q2 --memory-limit 90GiB --max-live-kv-tokens 4096 \
#          --slot-layout component-banks --cache-scope layer \
#          --max-tokens 128 --prompt "Explain why the sky is blue." \
#          --resource-telemetry
#
# If <CODEC_MANIFEST> does not exist yet it is built first with
# scripts/convert_streamed_rans.py (the FULL sweep -- run it inside the window
# too, it re-reads the whole sidecar once).
set -euo pipefail

PY="${MTPLX_PYTHON:-/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 4 ]]; then
  sed -n '2,40p' "${BASH_SOURCE[0]}"
  exit 2
fi

MODEL_ROOT="$1"; BASE_MANIFEST="$2"; CODEC_MANIFEST="$3"; shift 3
if [[ "${1:-}" != "--" ]]; then
  echo "error: separate the pass-through benchmark args with '--'" >&2
  exit 2
fi
shift
BENCH_ARGS=("$@")

# --- run-lock: never run two ~90 GiB streamed jobs at once (kernel-panic risk).
if pgrep -f "benchmark_streamed_generation.py" >/dev/null 2>&1; then
  echo "error: a benchmark_streamed_generation.py run is already live; refusing "\
"to start a second ~90 GiB job (panic protocol)." >&2
  exit 1
fi
echo ">> memory protocol: this is a strict-90g window. Confirm wired stays under"
echo "   the 100 GiB knob and no other large run is active before proceeding."

OUT_DIR="${MTPLX_GATE_OUT:-$(mktemp -d -t rans_stream_gate)}"
mkdir -p "$OUT_DIR"
NONE_JSON="$OUT_DIR/arm_uncompressed.json"
RANS_JSON="$OUT_DIR/arm_rans32x.json"
echo ">> gate artifacts: $OUT_DIR"

# --- build the compressed streamed sidecar if it is not there yet.
if [[ ! -f "$CODEC_MANIFEST" ]]; then
  echo ">> building compressed streamed sidecar (full sweep) ..."
  "$PY" "$ROOT/scripts/convert_streamed_rans.py" \
    --source-root "$MODEL_ROOT" \
    --manifest "$BASE_MANIFEST" \
    --output-manifest "$CODEC_MANIFEST" \
    --resume
fi

run_arm() {
  local label="$1"; local out_json="$2"; shift 2
  echo ">> ARM [$label] -> $out_json"
  # NB: run in the foreground; do NOT background a guarded run.
  "$PY" "$ROOT/scripts/benchmark_streamed_generation.py" \
    "$MODEL_ROOT" "$BASE_MANIFEST" \
    "$@" \
    --resource-telemetry \
    --output-json "$out_json" \
    "${BENCH_ARGS[@]}"
}

run_arm uncompressed "$NONE_JSON" --streamed-codec none
run_arm rans32x-v1   "$RANS_JSON" --streamed-codec rans32x-v1 \
  --streamed-codec-manifest "$CODEC_MANIFEST"

# --- report decode tok/s + bytes-read/token for both arms.
"$PY" - "$NONE_JSON" "$RANS_JSON" <<'PYEOF'
import json, sys

def walk(obj):
    """Yield every dict in a nested JSON payload."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)

def summarize(path):
    payload = json.loads(open(path).read())
    # Best decode tok/s across runs.
    decode_tps = 0.0
    completion_tokens = 0
    for d in walk(payload):
        for key in ("decode_tokens_per_second", "aggregate_completion_tokens_per_second",
                    "completion_tokens_per_second"):
            v = d.get(key)
            if isinstance(v, (int, float)):
                decode_tps = max(decode_tps, float(v))
        v = d.get("completion_tokens")
        if isinstance(v, int):
            completion_tokens = max(completion_tokens, v)
    # Terminal reader io counters (largest observed = cumulative).
    read_bytes = 0
    saved = 0
    decoded_records = 0
    for d in walk(payload):
        if "read_bytes" in d and "requested_bytes" in d:  # an io metrics block
            read_bytes = max(read_bytes, int(d.get("read_bytes", 0)))
            saved = max(saved, int(d.get("bytes_read_saved", 0)))
            decoded_records = max(decoded_records, int(d.get("decoded_records", 0)))
    return {
        "decode_tps": decode_tps,
        "completion_tokens": completion_tokens,
        "read_bytes": read_bytes,
        "bytes_read_saved": saved,
        "decoded_records": decoded_records,
        "codec": (payload.get("config") or {}).get("streamed_codec"),
    }

none = summarize(sys.argv[1])
rans = summarize(sys.argv[2])

def per_tok(s):
    return s["read_bytes"] / s["completion_tokens"] if s["completion_tokens"] else 0.0

print("\n================ issue #113 streamed rANS gate ================")
for name, s in (("uncompressed", none), ("rans32x-v1", rans)):
    print(f"[{name:12s}] codec={s['codec']!s:11s} decode={s['decode_tps']:.3f} tok/s "
          f"read={s['read_bytes']/1024**2:.1f} MiB "
          f"read/token={per_tok(s)/1024:.1f} KiB "
          f"decoded_records={s['decoded_records']} "
          f"saved={s['bytes_read_saved']/1024**2:.1f} MiB")
if per_tok(none) and per_tok(rans):
    drop = 1.0 - per_tok(rans) / per_tok(none)
    print(f"\nbytes-read/token drop: {drop*100:.1f}%  "
          f"(expected ~24% at the rANS ratio)")
if none["decode_tps"] and rans["decode_tps"]:
    speed = rans["decode_tps"] / none["decode_tps"]
    print(f"decode tok/s change:   {speed:.3f}x "
          f"({rans['decode_tps']:.2f} vs {none['decode_tps']:.2f})")
print("==============================================================\n")
PYEOF
