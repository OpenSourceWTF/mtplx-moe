# AngelSlim/Hy3-GGUF recipe extraction + mimic fidelity pilot (2026-07-21)

Official artifact: `AngelSlim/Hy3-GGUF` / `Hy3-IQ1_M.gguf`, 89.446 GB, anchor-exact
vs prior parse (+0.000%). Single file, 1278 tensors, 9 quant types — "IQ1_M" is a
recipe name, not a uniform format.

## Official assignment (complete map in official-recipe-map.json)
- Routed experts 76.333 GiB @ 2.290 bpw: `down_exps` IQ3_XXS in ALL 79 MoE layers;
  gate/up share one tier per layer — IQ1_M {1,8-27,29-31,51-54,56-67},
  IQ2_XXS {2-7,28,32-50,55,68-79}.
- Trunk is HIGH: attn QKV Q8_0 (all 80), attn_out Q5_K, shared gate/up Q5_K,
  shared down Q6_K, embed Q4_K, head Q6_K. Attention role = 7.17 bpw.
- MTP block (blk.80): experts Q4_K/Q5_K, eh_proj Q8_0 — the draft head is kept far
  ABOVE the bank (independent confirmation of our q4-head / acceptance law).

## Mimic fidelity (cos vs local bf16 ground truth; 6 cells/tier; mean [min])
| official tier (bpw) | official | mimic (bpw) | mimic | one-up (bpw) | one-up |
|---|---|---|---|---|---|
| IQ1_M (1.75) | 0.8990 [0.8718] | t158 (1.875) | 0.8986 [0.8836] | q2 (2.5) | 0.9217 |
| IQ2_XXS (2.0625) | 0.9419 [0.9360] | q2 (2.5) | 0.9230 [0.9177] | q3 (3.5) | 0.9817 |
| IQ3_XXS (3.0625) | 0.9827 [0.9818] | q3 (3.5) | 0.9820 [0.9819] | q4 (4.5) | 0.9959 |

SwiGLU output cosine (one full triplet, 32 draws): IQ1_M layer official 0.8159 /
mimic 0.8102; IQ2_XXS layer official 0.8795 / mimic 0.8436.
Trunk affine ladder: q8 0.99999 / q6 0.99976 / q5 0.99903 — lossless in practice.

## Blended routed-bank cost
| scheme | blended bpw | bank |
|---|---|---|
| official exact | 2.290 | 76.33 GiB |
| mimic exact assignment (t158/q2/q3) | 2.622 | 87.40 GiB |
| our current uniform q2 | 2.500 | 83.32 GiB |
| variant down→q4 | 2.956 | 98.51 GiB (rejected: +1 bpw buys +0.014) |

## Findings
1. t158 TIES official IQ1_M (Δ −0.0004, and beats it on the worst cell) at +0.125 bpw.
2. q3 TIES official IQ3_XXS on down (Δ −0.0007) at +0.44 bpw; q4 is not worth it.
3. IQ2_XXS is the one non-mimicable tier: official 0.9419 beats our q2 0.9230
   DESPITE costing less (2.06 vs 2.5 bpw) — the imatrix E8-grid edge is real and
   concentrated exactly at ~2 bpw. Affine can only undershoot (q2) or overshoot (q3).
4. Layer 1 is a hard outlier every tier (first MoE block is quantization-sensitive);
   layers 13-78 are near-homogeneous.
5. Faithful mimic costs +14.5% bytes vs official, purely format overhead.

Per-cell data: mimic-pilot-20260721.json. Harness: recipe_extract.py, am_lib.py,
run_fidelity.py (reuses ../iq_transcode tooling). 52.2 MB fetched; bf16 read locally.
