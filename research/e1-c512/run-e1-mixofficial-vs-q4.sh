#!/bin/bash
# E1 cell 1: mixed-official candidate vs q4 control, llama.cpp-exact WikiText-2
# c512 independent chunks. Gate: <=5% relative PPL vs q4 (ladder: q4 4.98,
# shipped q2 8.40). Hardened after 3 burned windows: (1) CPU-side preflight
# runs the EXACT lane validator (verify_expert_manifest) on BOTH roots before
# the guarded window opens; (2) output json is timestamped so a stale receipt
# can never File-exist the final write.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python
CACHE=$HOME/.cache/huggingface
OUT="$WT/research/e1-c512/e1-mixofficial-vs-q4-$(date +%Y%m%d-%H%M%S).json"
cd "$WT"

PYTHONPATH="$WT" "$PY" - << 'PYEOF' || { echo "PREFLIGHT FAILED — not opening a window"; exit 2; }
from pathlib import Path
from mtplx.expert_manifest import load_expert_manifest, verify_expert_manifest
for root in ("hy3-expert-only-mlx-q4", "hy3-expert-only-mlx-mixofficial"):
    r = Path.home() / ".cache/huggingface" / root
    m = load_expert_manifest(r / "expert-manifest.json")
    print("preflight", root, verify_expert_manifest(m, r))
PYEOF

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --child-timeout-seconds 10800 \
  -- \
  "$PY" scripts/compare_streamed_quality.py \
  --q4-root "$CACHE/hy3-expert-only-mlx-q4" \
  --q4-manifest "$CACHE/hy3-expert-only-mlx-q4/expert-manifest.json" \
  --q4-model-key hy3-expert-only-q4 \
  --q2-root "$CACHE/hy3-expert-only-mlx-mixofficial" \
  --q2-manifest "$CACHE/hy3-expert-only-mlx-mixofficial/expert-manifest.json" \
  --q2-model-key hy3-expert-mixofficial \
  --memory-limit 78GiB --expert-cache-limit 56GiB --runtime-reserve 12GiB \
  --max-live-kv-tokens 2048 --cache-policy lru --cache-scope layer \
  --slot-layout component-banks --transient-slots 32 --f-nocache \
  --corpus-file "$WT/benchmarks/fixtures/corpus/wikitext-2-raw-test.raw" \
  --evaluation-tokens 65536 --chunk-tokens 512 \
  --c512-independent-chunks --c512-n-ctx 512 --c512-n-chunks 128 \
  --prompt-file "$WT/benchmarks/fixtures/hy3-q2-quality-prompts.jsonl" \
  --greedy-max-tokens 64 \
  --max-relative-perplexity-regression 0.05 \
  --output-json "$OUT"
