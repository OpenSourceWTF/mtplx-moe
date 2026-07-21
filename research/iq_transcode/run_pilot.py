"""IQ-transcode pilot: does transcoding published imatrix GGUF i-quants into
our MLX-servable formats (t158, affine q2/q4 gs64) inherit their calibration
quality, or does the transcode loss eat it?

Source: bartowski/Hy3-GGUF (imatrix), formats IQ1_M / IQ2_XXS / IQ3_XXS.
Ground truth: local bf16 tencent/Hy3 shards. CPU-only.
"""
from __future__ import annotations

import json
import os
import sys
import time
import statistics as stats

import numpy as np

WORKTREE = "/Users/davidtai/projects/OpenSourceWTF/.worktrees/51-c6-mmap-band"
sys.path.insert(0, WORKTREE)
sys.path.insert(0, os.path.dirname(__file__))

import mlx.core as mx  # noqa: E402
mx.set_default_device(mx.cpu)

import gguf_range as G  # noqa: E402
import st_range as ST  # noqa: E402
import pilot_lib as P  # noqa: E402
from mtplx.expert_shadow import encode_t158, decode_t158  # noqa: E402

OUT_JSON = os.path.join(os.path.dirname(__file__), "iq-transcode-pilot-20260721.json")

# spread across layers, mixed gate/up, diverse experts (8 samples/format)
SAMPLE_LAYERS = [1, 12, 23, 34, 45, 56, 67, 78]
SAMPLE_PROJS = ["gate", "up", "gate", "up", "gate", "up", "gate", "up"]
SAMPLE_EXPERTS = [0, 40, 80, 120, 160, 24, 96, 168]

TARGETS_BY_FORMAT = {
    "IQ1_M": ["t158", "q2_gs64"],
    "IQ2_XXS": ["t158", "q2_gs64"],
    "IQ3_XXS": ["t158", "q2_gs64", "q4_gs64"],
}


def target_from(name: str, w: np.ndarray) -> np.ndarray:
    """Encode w with target codec and return the dequantized fp32 array."""
    if name == "t158":
        packed, scales = encode_t158(w.astype(np.float32))
        return decode_t158(packed, scales, w.shape[1]).astype(np.float32)
    if name in ("q2_gs64", "q4_gs64"):
        bits = 2 if name == "q2_gs64" else 4
        wm = mx.array(w.astype(np.float32))
        wq, sc, bi = mx.quantize(wm, group_size=64, bits=bits)
        deq = mx.dequantize(wq, sc, bi, group_size=64, bits=bits)
        return np.array(deq.astype(mx.float32), dtype=np.float32)
    raise ValueError(name)


def imatrix_kv(fi: P.FormatIndex) -> dict:
    kv = fi.splits[0].header.kv
    return {k: kv[k] for k in kv if k.startswith("quantize.imatrix")
            or k in ("general.file_type", "general.quantization_version",
                     "general.name", "general.architecture")}


def main():
    t0 = time.time()
    session = G.new_session()
    receipt = {
        "task": "O7 IQ-transcode pilot",
        "date": "2026-07-21",
        "question": ("does transcoding published imatrix-calibrated GGUF "
                     "i-quants into MLX-servable formats inherit calibration "
                     "quality, or does transcode loss eat it?"),
        "source_repo": P.REPO,
        "source_substitution_note": (
            "SUBSTITUTION: LordNeel/Hy3-GGUF (the repo cited in the task/notes) "
            "EXISTS and is imatrix-calibrated, but contains NO IQ1_M/IQ2_XXS/"
            "IQ3_XXS quants -- its only IQ-family quant is IQ2_M; the rest are "
            "K-quants (Q3_K_L, Q4_K_M, Q6_K). The cited recipe (40 layers IQ1_M "
            "/ 39 IQ2_XXS gate-up / IQ3_XXS down) does NOT match that repo. "
            "Substituted bartowski/Hy3-GGUF, the canonical i-quant producer, "
            "which carries exactly IQ1_M, IQ2_XXS, IQ3_XXS -- all imatrix "
            "(tag 'imatrix', embedded quantize.imatrix.* metadata below)."),
        "ground_truth": {
            "model": "tencent/Hy3 bf16 rev 716aa724",
            "dir": P.BF16_DIR,
            "index": P.BF16_INDEX,
        },
        "config": {
            "shapes": {"gate/up": [1536, 4096], "down": [4096, 1536]},
            "expert_count": P.N_EXPERT,
            "sample_plan": {
                "layers": SAMPLE_LAYERS, "projs": SAMPLE_PROJS,
                "experts": SAMPLE_EXPERTS,
                "note": "gate/up_exps only; each sampled tensor's actual ggml "
                        "type is verified == source format before use",
            },
            "targets": {
                "t158": "mtplx.expert_shadow encode_t158/decode_t158 (ternary "
                        "1.58b, group 64, bf16 scale) -- 15 B/64w",
                "q2_gs64": "mx.quantize/dequantize affine bits=2 group_size=64 "
                           "(CPU, mode=affine -- the runtime path)",
                "q4_gs64": "mx.quantize/dequantize affine bits=4 group_size=64 "
                           "(IQ3_XXS sources only)",
            },
        },
        "decoder_validation": {},
        "formats": {},
        "download_bytes": {"headers": 0, "iq_slices": 0, "hand_validate": 0},
    }

    # ---- decoder hand-validation (IQ2_XXS, real block) ----
    import hand_validate as HV
    fi_v = P.FormatIndex("IQ2_XXS", session)
    raw_v, _, _, per_v, sha_v = fi_v.splits[0].fetch_expert_slice(
        "blk.1.ffn_gate_exps.weight", 0, P.N_EXPERT)
    receipt["download_bytes"]["hand_validate"] = per_v
    hv = HV.validate(raw_v[:66])
    receipt["decoder_validation"]["IQ2_XXS_hand_block"] = {
        "method": ("independent scalar reimpl; sign from even-parity rule "
                   "(not gguf ksigns), scale from top-4 bits, grid decoded "
                   "from grid_hex; matched vs gguf.quants on a real block"),
        **hv,
    }

    header_bytes_total = fi_v.header_bytes  # reused below via cache-free re-parse

    for fmt in ["IQ1_M", "IQ2_XXS", "IQ3_XXS"]:
        fi = fi_v if fmt == "IQ2_XXS" else P.FormatIndex(fmt, session)
        header_bytes_total += 0 if fmt == "IQ2_XXS" else fi.header_bytes
        tmap = fi.exps_type_map()

        # build validated sample coords for this format
        coords = []
        for L, proj, E in zip(SAMPLE_LAYERS, SAMPLE_PROJS, SAMPLE_EXPERTS):
            if tmap.get(L, {}).get(proj) != fmt:
                # fall back to nearest layer whose proj matches
                alt = next((l for l in sorted(tmap)
                            if tmap[l].get(proj) == fmt and l not in
                            [c[0] for c in coords]), None)
                if alt is None:
                    continue
                L = alt
            coords.append((L, proj, E))

        per_target = {t: {"cos_bf16_from_iqx": [], "cos_bf16_from_bf16": [],
                          "cos_src_from_iqx": [], "delta_cos": [],
                          "relerr_bf16_from_iqx": [], "relerr_bf16_from_bf16": [],
                          "relerr_src_from_iqx": []}
                      for t in TARGETS_BY_FORMAT[fmt]}
        src_cos_list, src_relerr_list = [], []
        samples = []

        for (L, proj, E) in coords:
            deq, gmeta = fi.fetch_expert(L, proj, E)
            receipt["download_bytes"]["iq_slices"] += gmeta["byte_len"]
            bpath, bname = P.bf16_shard_path(L, E, proj)
            bf16, bmeta = ST.read_tensor_fp32(bpath, bname)
            assert bf16.shape == deq.shape, (bf16.shape, deq.shape)

            cos_src = P.cosine(bf16, deq)
            re_src = P.rel_err(bf16, deq)
            src_cos_list.append(cos_src)
            src_relerr_list.append(re_src)

            srec = {
                "layer": L, "proj": proj, "expert": E,
                "bf16_tensor": bname,
                "bf16_shard": os.path.basename(bpath),
                "bf16_abs_begin": bmeta["abs_begin"],
                "bf16_nbytes": bmeta["nbytes"],
                "bf16_sha256": bmeta["sha256"],
                "gguf": gmeta,
                "cos_bf16_src": cos_src,
                "relerr_bf16_src": re_src,
                "targets": {},
            }

            for tname in TARGETS_BY_FORMAT[fmt]:
                t_bf16 = target_from(tname, bf16)
                t_iqx = target_from(tname, deq)
                c_ff = P.cosine(bf16, t_bf16)
                c_fi = P.cosine(bf16, t_iqx)
                c_si = P.cosine(deq, t_iqx)
                re_ff = P.rel_err(bf16, t_bf16)
                re_fi = P.rel_err(bf16, t_iqx)
                re_si = P.rel_err(deq, t_iqx)
                delta = c_fi - c_ff
                per_target[tname]["cos_bf16_from_iqx"].append(c_fi)
                per_target[tname]["cos_bf16_from_bf16"].append(c_ff)
                per_target[tname]["cos_src_from_iqx"].append(c_si)
                per_target[tname]["delta_cos"].append(delta)
                per_target[tname]["relerr_bf16_from_iqx"].append(re_fi)
                per_target[tname]["relerr_bf16_from_bf16"].append(re_ff)
                per_target[tname]["relerr_src_from_iqx"].append(re_si)
                srec["targets"][tname] = {
                    "cos_bf16_from_bf16": c_ff,
                    "cos_bf16_from_iqx": c_fi,
                    "cos_src_from_iqx": c_si,
                    "delta_cos": delta,
                    "relerr_bf16_from_bf16": re_ff,
                    "relerr_bf16_from_iqx": re_fi,
                    "relerr_src_from_iqx": re_si,
                }
            samples.append(srec)
            print(f"[{fmt}] L{L:>2} {proj:>4} e{E:<3} src_cos={cos_src:.4f} "
                  + " ".join(f"{t}:d={per_target[t]['delta_cos'][-1]:+.4f}"
                             for t in TARGETS_BY_FORMAT[fmt]))

        def agg(xs):
            xs = [x for x in xs if x == x]  # drop nan
            return {"mean": stats.fmean(xs), "min": min(xs), "max": max(xs),
                    "n": len(xs)}

        verdict = {}
        for tname, cols in per_target.items():
            verdict[tname] = {k: agg(v) for k, v in cols.items()}

        receipt["formats"][fmt] = {
            "split_files": P.FORMAT_SPLITS[fmt],
            "header_bytes": fi.header_bytes,
            "imatrix_metadata": imatrix_kv(fi),
            "exps_gate_up_type_note": (
                "gate/up_exps uniformly this format across 79 layers "
                "(+1 layer Q4_0, excluded by type filter)"),
            "n_samples": len(samples),
            "source_quality": {
                "cos_bf16_src": agg(src_cos_list),
                "relerr_bf16_src": agg(src_relerr_list),
            },
            "verdict": verdict,
            "samples": samples,
        }

    receipt["download_bytes"]["headers"] = header_bytes_total
    receipt["download_bytes"]["total"] = (
        receipt["download_bytes"]["headers"]
        + receipt["download_bytes"]["iq_slices"]
        + receipt["download_bytes"]["hand_validate"])
    receipt["elapsed_seconds"] = round(time.time() - t0, 1)

    with open(OUT_JSON, "w") as f:
        json.dump(receipt, f, indent=2)
    print("\nwrote", OUT_JSON, "in", receipt["elapsed_seconds"], "s")
    print("download total MB:", round(receipt["download_bytes"]["total"] / 1e6, 1))
    return receipt


if __name__ == "__main__":
    main()
