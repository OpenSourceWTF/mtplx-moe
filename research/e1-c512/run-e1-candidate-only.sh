#!/bin/bash
# E1 candidate-only guarded window: mixofficial lane ONLY (control is banked;
# never re-run — David 2026-07-21). CPU preflight before the window opens.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python
OUT="$WT/research/e1-c512/e1-candidate-only-$(date +%Y%m%d-%H%M%S).json"
cd "$WT"

PYTHONPATH="$WT" "$PY" - << 'PYEOF' || { echo "PREFLIGHT FAILED — not opening a window"; exit 2; }
from pathlib import Path
from mtplx.expert_manifest import load_expert_manifest, verify_expert_manifest
r = Path.home() / ".cache/huggingface/hy3-expert-only-mlx-mixofficial"
m = load_expert_manifest(r / "expert-manifest.json")
print("preflight mixofficial:", verify_expert_manifest(m, r))
PYEOF

exec env PYTHONPATH="$WT" "$PY" scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --child-timeout-seconds 7200 \
  -- \
  "$PY" "$WT/research/e1-c512/e1_candidate_only.py" "$OUT"
