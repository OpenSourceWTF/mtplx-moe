"""O8 steps 2-4: mimic-fidelity matrix, trunk spot-check, allocation table.

For each sampled (layer, proj) cell whose official type is IQ1_M / IQ2_XXS /
IQ3_XXS we compute weight cosine + rel-err vs local bf16 for THREE arms, all
measured against bf16 ground truth:
  (a) official IQx dequant           (range-fetched from AngelSlim)
  (b) our MLX-servable mimic from bf16  (t158 / q2 / q3 by tier)
  (c) one-tier-up affine from bf16      (q2 / q3 / q4 reference)
Two full gate/up/down triplets add SwiGLU output cosine (32 draws, seed 51).
A trunk spot-check confirms the affine ladder at q8/q6/q5 (no fetch).
Then the routed-bank blended-bpw allocation table is assembled.
CPU-only.
"""
from __future__ import annotations

import json
import os
import sys
import time
import statistics as stats

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

import am_lib as A  # noqa: E402
import gguf_range as G  # noqa: E402
import st_range as ST  # noqa: E402
import pilot_lib as P  # noqa: E402

OUT_JSON = os.path.join(HERE, "mimic-pilot-20260721.json")
RECIPE_JSON = os.path.join(HERE, "official-recipe-map.json")

# ---- cell plan (spread across layers; experts varied) ----
# tier -> list of (layer, proj, expert). gate/up for IQ1_M & IQ2_XXS, down for IQ3.
FLAT_CELLS = {
    "IQ1_M": [(1, "gate", 0), (13, "up", 40), (20, "gate", 80),
              (27, "up", 120), (56, "gate", 150), (66, "up", 180)],
    "IQ2_XXS": [(2, "gate", 10), (7, "up", 50), (34, "gate", 90),
                (44, "gate", 96), (55, "up", 130), (78, "gate", 170)],
    "IQ3_XXS": [(3, "down", 5), (20, "down", 80), (30, "down", 125),
                (44, "down", 96), (60, "down", 165), (75, "down", 25)],
}
# triplets reuse flat gate+down fetches, add only the missing up fetch.
TRIPLETS = [
    {"name": "IQ1_M-layer", "layer": 20, "expert": 80, "gateup_tier": "IQ1_M"},
    {"name": "IQ2_XXS-layer", "layer": 44, "expert": 96, "gateup_tier": "IQ2_XXS"},
]
TRUNK_CELLS = [  # (bf16 name, official ggml type, mimic codec) - no fetch
    ("model.layers.5.self_attn.q_proj.weight", "Q8_0", "q8_gs64"),
    ("model.layers.5.self_attn.o_proj.weight", "Q5_K", "q5_gs64"),
    ("model.layers.5.mlp.shared_mlp.down_proj.weight", "Q6_K", "q6_gs64"),
]
SEED = 51
N_DRAWS = 32


def agg(xs):
    xs = [x for x in xs if x == x]
    return {"mean": stats.fmean(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def bf16_for(L, proj, expert):
    bpath, bname = P.bf16_shard_path(L, expert, proj)
    arr, meta = ST.read_tensor_fp32(bpath, bname)
    return arr, bname, os.path.basename(bpath), meta


def main():
    t0 = time.time()
    session = G.new_session()
    idx = A.AngelSlimIndex(session)

    dl = {"anchor_header": idx.header_bytes, "official_slices": 0}
    receipt = {
        "task": "O8 AngelSlim-recipe mimic pilot",
        "date": "2026-07-21",
        "repo": A.REPO, "anchor_file": A.ANCHOR_FILE,
        "ground_truth": {"model": "tencent/Hy3 bf16", "dir": P.BF16_DIR},
        "codec_defs": {
            "official": "gguf.quants.dequantize of the range-fetched IQx expert slice",
            "t158": "mtplx.expert_shadow t158 ternary, group 64, bf16 scale (1.875 bpw)",
            "qN_gs64": "mx.quantize/dequantize affine bits=N group_size=64 (N+0.5 bpw, fp16 scale+bias)",
            "arms_measured_against": "local bf16 (tencent/Hy3) ground truth",
            "flat_slice_note": "cosine on ONE dequantized expert slice (6.29M weights, ~25MB fp32, <1.5GB); fused 3D tensor NOT fully decoded",
        },
        "bpw_table": A.BPW,
        "seed": SEED, "n_draws": N_DRAWS,
        "cell_plan": {"flat": FLAT_CELLS, "triplets": TRIPLETS, "trunk": TRUNK_CELLS},
        "tiers": {}, "triplet_swiglu": [], "trunk_spotcheck": [],
    }

    # cache of fetched official deq + bf16 by (L,proj,expert) for triplet reuse
    cache = {}

    def get_cell(L, proj, E):
        key = (L, proj, E)
        if key in cache:
            return cache[key]
        deq, gmeta = idx.fetch_expert(L, proj, E)
        dl["official_slices"] += gmeta["byte_len"]
        bf16, bname, bshard, bmeta = bf16_for(L, proj, E)
        assert bf16.shape == deq.shape, (bf16.shape, deq.shape)
        cache[key] = (bf16, deq, gmeta, bname, bshard, bmeta)
        return cache[key]

    # ---- fidelity matrix ----
    for tier, cells in FLAT_CELLS.items():
        mimic_codec, tierup_codec = A.TIER_CODECS[tier]
        cols = {"cos_official": [], "relerr_official": [],
                "cos_mimic": [], "relerr_mimic": [],
                "cos_tierup": [], "relerr_tierup": []}
        samples = []
        for (L, proj, E) in cells:
            bf16, deq, gmeta, bname, bshard, bmeta = get_cell(L, proj, E)
            assert gmeta["ggml_type"] == tier, (L, proj, gmeta["ggml_type"], tier)
            w_mimic = A.apply_codec(mimic_codec, bf16)
            w_up = A.apply_codec(tierup_codec, bf16)
            c_off, re_off = P.cosine(bf16, deq), P.rel_err(bf16, deq)
            c_mi, re_mi = P.cosine(bf16, w_mimic), P.rel_err(bf16, w_mimic)
            c_up, re_up = P.cosine(bf16, w_up), P.rel_err(bf16, w_up)
            cols["cos_official"].append(c_off); cols["relerr_official"].append(re_off)
            cols["cos_mimic"].append(c_mi); cols["relerr_mimic"].append(re_mi)
            cols["cos_tierup"].append(c_up); cols["relerr_tierup"].append(re_up)
            samples.append({
                "layer": L, "proj": proj, "expert": E,
                "bf16_tensor": bname, "bf16_shard": bshard,
                "bf16_abs_begin": bmeta["abs_begin"], "bf16_nbytes": bmeta["nbytes"],
                "bf16_sha256": bmeta["sha256"], "official": gmeta,
                "cos_official": c_off, "relerr_official": re_off,
                "mimic_codec": mimic_codec, "cos_mimic": c_mi, "relerr_mimic": re_mi,
                "tierup_codec": tierup_codec, "cos_tierup": c_up, "relerr_tierup": re_up,
            })
            print(f"[{tier:8s}] L{L:>2} {proj:>4} e{E:<3} "
                  f"off={c_off:.4f} {mimic_codec}={c_mi:.4f} {tierup_codec}={c_up:.4f}")
        receipt["tiers"][tier] = {
            "official_bpw": A.BPW[tier], "mimic_codec": mimic_codec,
            "mimic_bpw": A.BPW[mimic_codec], "tierup_codec": tierup_codec,
            "tierup_bpw": A.BPW[tierup_codec],
            "cos_official": agg(cols["cos_official"]),
            "cos_mimic": agg(cols["cos_mimic"]),
            "cos_tierup": agg(cols["cos_tierup"]),
            "relerr_official": agg(cols["relerr_official"]),
            "relerr_mimic": agg(cols["relerr_mimic"]),
            "relerr_tierup": agg(cols["relerr_tierup"]),
            "samples": samples,
        }

    # ---- SwiGLU triplets ----
    rng = np.random.default_rng(SEED)
    for tri in TRIPLETS:
        L, E, gu_tier = tri["layer"], tri["expert"], tri["gateup_tier"]
        gate_bf, gate_off, gmeta_g, *_ = get_cell(L, "gate", E)
        up_bf, up_off, gmeta_u, *_ = get_cell(L, "up", E)
        down_bf, down_off, gmeta_d, *_ = get_cell(L, "down", E)
        gu_mimic = A.TIER_CODECS[gu_tier][0]
        down_mimic = A.TIER_CODECS["IQ3_XXS"][0]
        gate_mi = A.apply_codec(gu_mimic, gate_bf)
        up_mi = A.apply_codec(gu_mimic, up_bf)
        down_mi = A.apply_codec(down_mimic, down_bf)
        X = rng.standard_normal((N_DRAWS, gate_bf.shape[1])).astype(np.float64)
        y_bf = A.swiglu(gate_bf, up_bf, down_bf, X)
        y_off = A.swiglu(gate_off, up_off, down_off, X)
        y_mi = A.swiglu(gate_mi, up_mi, down_mi, X)
        rec = {
            "name": tri["name"], "layer": L, "expert": E,
            "gateup_tier": gu_tier, "gate_up_codec_mimic": gu_mimic,
            "down_tier": "IQ3_XXS", "down_codec_mimic": down_mimic,
            "official_types": {"gate": gmeta_g["ggml_type"],
                               "up": gmeta_u["ggml_type"], "down": gmeta_d["ggml_type"]},
            "swiglu_cos_official_vs_bf16": P.cosine(y_bf, y_off),
            "swiglu_cos_mimic_vs_bf16": P.cosine(y_bf, y_mi),
            "swiglu_relerr_official_vs_bf16": P.rel_err(y_bf, y_off),
            "swiglu_relerr_mimic_vs_bf16": P.rel_err(y_bf, y_mi),
        }
        receipt["triplet_swiglu"].append(rec)
        print(f"[triplet {tri['name']:14s}] L{L} e{E} "
              f"swiglu off={rec['swiglu_cos_official_vs_bf16']:.4f} "
              f"mimic={rec['swiglu_cos_mimic_vs_bf16']:.4f}")

    # ---- trunk spot-check (affine ladder, no fetch) ----
    for bname, off_type, codec in TRUNK_CELLS:
        bpath = P._WM = P._WM  # ensure wm loaded
        wm = P._weight_map()
        shard = os.path.join(P.BF16_DIR, wm[bname])
        w, meta = ST.read_tensor_fp32(shard, bname)
        w_q = A.apply_codec(codec, w)
        c = P.cosine(w, w_q)
        receipt["trunk_spotcheck"].append({
            "tensor": bname, "shape": list(w.shape), "official_type": off_type,
            "mimic_codec": codec, "mimic_bpw": A.BPW[codec], "cos_mimic_vs_bf16": c,
            "relerr_mimic_vs_bf16": P.rel_err(w, w_q),
        })
        print(f"[trunk {off_type:5s}->{codec}] {bname.split('.')[-2]:12s} cos={c:.5f}")

    # ---- allocation table: routed-bank blended bpw ----
    recipe = json.load(open(RECIPE_JSON))
    # weight count per routed tensor (one gate/up/down _exps) = 192*1536*4096
    per_tensor_w = 192 * 1536 * 4096
    tiers_layers = {"IQ1_M_gateup": 0, "IQ2_XXS_gateup": 0, "IQ3_XXS_down": 0}
    for L, d in recipe["per_layer_routed"].items():
        if d["gate"] == "IQ1_M":
            tiers_layers["IQ1_M_gateup"] += 2  # gate + up tensors
        elif d["gate"] == "IQ2_XXS":
            tiers_layers["IQ2_XXS_gateup"] += 2
        tiers_layers["IQ3_XXS_down"] += 1  # down tensor
    n_iq1 = tiers_layers["IQ1_M_gateup"]
    n_iq2 = tiers_layers["IQ2_XXS_gateup"]
    n_iq3 = tiers_layers["IQ3_XXS_down"]
    W = {"IQ1_M": n_iq1 * per_tensor_w, "IQ2_XXS": n_iq2 * per_tensor_w,
         "IQ3_XXS": n_iq3 * per_tensor_w}
    total_w = sum(W.values())

    def blended(bpw_map):
        return sum(W[t] * bpw_map[t] for t in W) / total_w

    official_bpw_map = {"IQ1_M": A.BPW["IQ1_M"], "IQ2_XXS": A.BPW["IQ2_XXS"],
                        "IQ3_XXS": A.BPW["IQ3_XXS"]}
    mimic_bpw_map = {"IQ1_M": A.BPW["t158"], "IQ2_XXS": A.BPW["q2_gs64"],
                     "IQ3_XXS": A.BPW["q3_gs64"]}
    # variant: gate/up tiers as mimic, but down dropped q3->... decided from data (filled in report)
    variant_down_q4 = {"IQ1_M": A.BPW["t158"], "IQ2_XXS": A.BPW["q2_gs64"],
                       "IQ3_XXS": A.BPW["q4_gs64"]}
    uniform_q2 = {"IQ1_M": A.BPW["q2_gs64"], "IQ2_XXS": A.BPW["q2_gs64"],
                  "IQ3_XXS": A.BPW["q2_gs64"]}

    routed_bank_bytes = recipe["role_rollup"]["routed_expert"]["nbytes"]
    receipt["allocation"] = {
        "routed_tensor_counts": {"IQ1_M_gateup_tensors": n_iq1,
                                 "IQ2_XXS_gateup_tensors": n_iq2,
                                 "IQ3_XXS_down_tensors": n_iq3},
        "weights_per_tier": W, "total_routed_weights": total_w,
        "official_routed_bank_bytes": routed_bank_bytes,
        "official_routed_bank_gib": round(routed_bank_bytes / (1 << 30), 3),
        "blended_bpw": {
            "official_exact": round(blended(official_bpw_map), 4),
            "mimic_exact_official_assignment": round(blended(mimic_bpw_map), 4),
            "uniform_q2_gs64": round(blended(uniform_q2), 4),
            "variant_mimic_gateup_down_q4": round(blended(variant_down_q4), 4),
        },
        "projected_bank_gib": {
            "official_exact": round(total_w * blended(official_bpw_map) / 8 / (1 << 30), 3),
            "mimic_exact": round(total_w * blended(mimic_bpw_map) / 8 / (1 << 30), 3),
            "uniform_q2": round(total_w * blended(uniform_q2) / 8 / (1 << 30), 3),
            "variant_down_q4": round(total_w * blended(variant_down_q4) / 8 / (1 << 30), 3),
        },
    }

    dl["total"] = dl["anchor_header"] + dl["official_slices"]
    receipt["download_bytes"] = dl
    receipt["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(OUT_JSON, "w") as f:
        json.dump(receipt, f, indent=2)
    print("\nwrote", OUT_JSON)
    print("download MB:", round(dl["total"] / 1e6, 2),
          "| blended bpw:", receipt["allocation"]["blended_bpw"],
          "| elapsed", receipt["elapsed_seconds"], "s")
    return receipt


if __name__ == "__main__":
    main()
