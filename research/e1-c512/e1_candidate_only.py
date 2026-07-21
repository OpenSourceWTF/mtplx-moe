#!/usr/bin/env python
"""E1 candidate-only cell: mixofficial c512 WikiText PPL, NO control re-run.

The q4 control is BANKED: 4.981381178726977, reproduced bit-identically across
two independent guarded windows (receipts e1-attempt1-0243-q2lane-failed.json
and e1-attempt5-0747-q2lane-failed.json — same corpus sha, same protocol, same
config). Re-measuring it a third time wastes a window (David, 2026-07-21).

This runs ONLY the mixed-official lane through the harness's own machinery
(`_lane` + `_evaluate_lane` + C512Params — the exact code path the pairwise
script uses), then applies the gate against the banked control.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_WT = Path(__file__).resolve().parents[2]
if str(_WT) not in sys.path:
    sys.path.insert(0, str(_WT))

from scripts.compare_streamed_quality import (  # noqa: E402
    C512Params,
    LaneConfig,
    _corpus_receipt,
    _evaluate_lane,
    _lane,
)

BANKED_CONTROL = {
    "perplexity": 4.981381178726977,
    "model_key": "hy3-expert-only-q4",
    "protocol": "wikitext-c512-independent-chunks n_ctx=512 n_chunks=128",
    "corpus_sha256": "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08",
    "receipts": [
        "e1-attempt1-0243-q2lane-failed.json",
        "e1-attempt5-0747-q2lane-failed.json",
    ],
    "note": "bit-identical across two independent guarded windows; not re-run "
    "per David's directive 2026-07-21",
}
GATE_MAX_RELATIVE_REGRESSION = 0.05


def main() -> int:
    out_path = Path(sys.argv[1])
    corpus_path = _WT / "benchmarks/fixtures/corpus/wikitext-2-raw-test.raw"
    config = LaneConfig(
        label="q2",
        model_root=Path.home() / ".cache/huggingface/hy3-expert-only-mlx-mixofficial",
        manifest_path=Path.home()
        / ".cache/huggingface/hy3-expert-only-mlx-mixofficial/expert-manifest.json",
        model_key="hy3-expert-mixofficial",
        memory_limit="78GiB",
        expert_cache_limit="56GiB",
        runtime_reserve="12GiB",
        max_live_kv_tokens=2048,
        cache_policy="lru",
        cache_scope="layer",
        slot_layout="component-banks",
        transient_slots=32,
        read_chunk="8MiB",
        f_nocache=True,
        trust_sidecar=False,
    )
    corpus, corpus_texts = _corpus_receipt([corpus_path])
    if corpus["files"][0]["sha256"] != BANKED_CONTROL["corpus_sha256"]:
        raise SystemExit("corpus sha does not match the banked control's corpus")

    result, errors, ok = _evaluate_lane(
        _lane(config),
        corpus_texts=corpus_texts,
        prompts=[],
        evaluation_tokens=65536,
        chunk_tokens=512,
        greedy_max_tokens=64,
        c512=C512Params(n_ctx=512, n_chunks=128),
    )

    loss = result.get("loss") or {}
    candidate_ppl = loss.get("perplexity")
    verdict: dict[str, object] = {
        "schema": "mtplx-e1-candidate-only-c512-v1",
        "banked_control": BANKED_CONTROL,
        "corpus": corpus,
        "candidate": result,
        "errors": errors,
        "lane_ok": ok,
        "max_relative_perplexity_regression": GATE_MAX_RELATIVE_REGRESSION,
    }
    if ok and isinstance(candidate_ppl, float) and loss.get("finite"):
        relative = candidate_ppl / BANKED_CONTROL["perplexity"] - 1.0
        verdict["candidate_perplexity"] = candidate_ppl
        verdict["relative_perplexity_regression"] = relative
        verdict["passed"] = relative <= GATE_MAX_RELATIVE_REGRESSION
    else:
        verdict["candidate_perplexity"] = candidate_ppl
        verdict["passed"] = False
        verdict["gate_error"] = "candidate lane did not produce finite loss"

    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_path)
    print(
        "E1 candidate-only:",
        "ppl=", candidate_ppl,
        "rel=", verdict.get("relative_perplexity_regression"),
        "passed=", verdict.get("passed"),
    )
    return 0 if verdict.get("passed") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
