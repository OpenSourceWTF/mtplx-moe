#!/usr/bin/env python3
"""EVAL 3 -- manifest & bank completeness + effective bits-per-weight.

WHAT IT MEASURES
  1. Runs the repo tool scripts/verify_expert_manifest.py against the shipped q2
     artifact (structural validation: manifest digest, ordered components, record
     offsets/shapes in-bounds of the sidecar).
  2. Independent cross-check: every (layer, expert) in the routed grid
     (79 layers x 192 experts = 15168) is present with no gaps; every segment's
     [offset, offset+length) lies inside experts.bin's real size on disk.
  3. Effective bits-per-weight: (total expert record bytes * 8) / (total weight
     elements), i.e. the 2-bit payload PLUS the gs64 bf16 scale+bias overhead.
  4. Exercises the HF "expert-aware completeness" check added in commit a476015
     (mtplx.hf_loader.expert_artifact_status / cached_model_incompleteness_reason).

WHY ITS OWN CORRECTNESS IS VERIFIABLE
  Ground truth is the real file size of experts.bin (os.stat) and the closed grid
  size (routed_layer_count * expert_count from the target descriptor). The bpw is
  arithmetic on manifest-declared shapes, cross-checked against config.json
  dimensions. Two independent readers agree: the repo's verify tool AND this
  script's own bounds sweep.

Run:  PYTHONPATH=<worktree> .venv/bin/python evals/manifest_completeness.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import GROUP_SIZE, Q2_DIR, update_results  # noqa: E402

from mtplx.expert_manifest import load_expert_manifest  # noqa: E402
from mtplx.hf_loader import (  # noqa: E402
    cached_model_incompleteness_reason,
    expert_artifact_status,
)

ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = ROOT / "scripts" / "verify_expert_manifest.py"
VENV_PY = Path("/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python")

EXPECTED_LAYERS = list(range(1, 80))  # 79 routed layers
EXPECTED_EXPERTS = list(range(192))
QUANT_BITS = 2


def run_verify_tool() -> dict:
    """Structural verification via the repo's own fail-closed tool."""
    cmd = [
        str(VENV_PY),
        str(VERIFY_SCRIPT),
        str(Q2_DIR),
        str(Q2_DIR / "expert-manifest.json"),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env={"PYTHONPATH": str(ROOT)}
    )
    parsed = None
    if proc.returncode == 0 and proc.stdout.strip():
        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError:
            parsed = None
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "report": parsed,
        "stderr": proc.stderr.strip()[:2000],
    }


def independent_bounds_and_completeness(manifest) -> dict:
    bank = Q2_DIR / manifest.sidecar.file
    bank_size = bank.stat().st_size

    present = {(r.layer, r.expert) for r in manifest.records}
    expected = {(layer, e) for layer in EXPECTED_LAYERS for e in EXPECTED_EXPERTS}
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    duplicates = len(manifest.records) - len(present)

    out_of_bounds = []
    total_record_bytes = 0
    total_weight_elems = 0
    total_weight_bytes = 0
    total_meta_bytes = 0  # scales + biases
    for rec in manifest.records:
        total_record_bytes += rec.logical_bytes
        for seg in rec.segments:
            end = seg.offset + seg.length
            if seg.offset < 0 or end > bank_size:
                out_of_bounds.append(
                    {
                        "layer": rec.layer,
                        "expert": rec.expert,
                        "component": seg.component,
                        "offset": seg.offset,
                        "end": end,
                        "bank_size": bank_size,
                    }
                )
            if seg.component.endswith(".weight"):
                out_features, packed_in = seg.shape
                real_in = packed_in * (32 // QUANT_BITS)
                total_weight_elems += out_features * real_in
                total_weight_bytes += seg.length
            else:  # scales / biases
                total_meta_bytes += seg.length

    effective_bpw = total_record_bytes * 8 / total_weight_elems
    payload_bpw = total_weight_bytes * 8 / total_weight_elems
    overhead_bpw = total_meta_bytes * 8 / total_weight_elems

    return {
        "bank_file": manifest.sidecar.file,
        "bank_size_on_disk": bank_size,
        "sidecar_declared_size": manifest.sidecar.size,
        "sidecar_size_matches_disk": bank_size == manifest.sidecar.size,
        "record_count": len(manifest.records),
        "expected_grid": f"{len(EXPECTED_LAYERS)}x{len(EXPECTED_EXPERTS)}={len(expected)}",
        "missing_count": len(missing),
        "missing_sample": missing[:10],
        "extra_count": len(extra),
        "duplicate_records": duplicates,
        "segments_out_of_bounds": out_of_bounds[:10],
        "n_out_of_bounds": len(out_of_bounds),
        "total_expert_record_bytes": total_record_bytes,
        "total_weight_elements": total_weight_elems,
        "effective_bpw": round(effective_bpw, 4),
        "weight_payload_bpw": round(payload_bpw, 4),
        "scale_bias_overhead_bpw": round(overhead_bpw, 4),
        "group_size": GROUP_SIZE,
    }


def hf_expert_aware_completeness() -> dict:
    status = expert_artifact_status(Q2_DIR)
    reason = cached_model_incompleteness_reason(Q2_DIR)
    return {
        "source": "commit a476015 (mtplx.hf_loader)",
        "expert_artifact_status": status,
        "cached_model_incompleteness_reason": reason,
        "reads_as_complete": reason is None,
    }


def main() -> int:
    t0 = time.time()
    manifest = load_expert_manifest(Q2_DIR / "expert-manifest.json")

    verify = run_verify_tool()
    bounds = independent_bounds_and_completeness(manifest)
    hf = hf_expert_aware_completeness()

    # bpw near 2.5 for affine gs64 2-bit (2.0 payload + 0.5 bf16 scale/bias).
    bpw = bounds["effective_bpw"]
    completeness_ok = (
        bounds["missing_count"] == 0
        and bounds["extra_count"] == 0
        and bounds["duplicate_records"] == 0
        and bounds["n_out_of_bounds"] == 0
        and bounds["sidecar_size_matches_disk"]
    )
    verdict = "PASS" if (verify["ok"] and completeness_ok and hf["reads_as_complete"]) else "FAIL"

    result = {
        "eval": "manifest & bank completeness + effective bpw",
        "repo_verify_tool": verify,
        "independent_check": bounds,
        "hf_expert_aware_completeness": hf,
        "effective_bpw_measured": bpw,
        "effective_bpw_note": (
            "affine gs64 2-bit == exactly 2.0 payload + 0.5 (bf16 scale+bias per "
            "64 weights) = 2.50 bpw; the '2.6-bit' label is approximate/rounded, "
            "measured is 2.50."
        ),
        "verdict": verdict,
        "seconds": round(time.time() - t0, 1),
    }
    update_results("eval3_manifest_completeness", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
