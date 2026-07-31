# Qwen 3.6 quantization eval matrix — HumanEval+/MBPP+ with bf16 references

Measured 2026-07-30/31 on an M5 Max (100 GiB wired limit), MTPLX serving stacks,
one model resident at a time (flock-guarded windows). Full EvalPlus suites —
HumanEval 164 and MBPP 378, greedy (`temperature 0`), `enable_thinking: false` —
via `evalplus.codegen --backend openai` against each live serving config, scored
with `evalplus.evaluate` (base + plus pass@1). Rates are server-reported
(`mtplx_stats`), never client-wall-derived.

## Results

| config | HumanEval / + | MBPP / + | solo tok/s | conc2 agg | prefill tok/s |
|---|---|---|---|---|---|
| A3B 35B bf16 (reference) | 91.5 / 88.4 | 89.7 / 75.9 | 62 | 55.1 | 3177 |
| A3B 35B 6-bit "Balance" | 91.5 / 89.0 | 90.5 / 76.5 | 106–115 | 82.5–104 | 3070 |
| A3B 35B 4-bit "Speed" | 91.5 / 87.2 | 91.8 / 77.8 | 111.6–112.0 | 99.7 | 3319 |
| 27B bf16 (reference) | 93.9 / 90.2 | 92.1 / 77.8 | 9.8 | 9.0 | 846 |
| 27B 8-bit "Quality" | 93.3 / 90.9 | 89.9 / 75.7 | 36.8 | 58.2 | 724 |
| 27B 4-bit "Speed" | 92.7 / 90.9 | 90.5 / 76.2 | ~36–50 | 64.7 | 734 |

Models: `Youssofal/Qwen3.6-27B-MTPLX-Optimized-{Speed,Quality}`,
`Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance` (6-bit gs64 affine, gate
layers 8-bit), `Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed` (4-bit gs64
affine body, router/gate 8-bit gs64, MTP numbered-expert draft head 4-bit
gs32 cyankiwi-prequantized), and the official `Qwen/Qwen3.6-35B-A3B` /
`Qwen/Qwen3.6-27B` bf16 releases as references.

## Verdicts

- **The A3B 6-bit quantization is free; 4-bit is not — but it's a trade, not
  a broad loss.** 6-bit still matches or beats its own bf16 reference on all
  four tiers, so it remains the sweet spot of the A3B ladder. 4-bit is the
  first real degradation step: HumanEval+ drops to 87.2, 1.8 points below
  6-bit (89.0) and 1.2 below bf16 (88.4). But the loss is one-sided — MBPP+
  actually climbs to 77.8, the *best* score anywhere in the A3B family (6-bit
  76.5, bf16 75.9), and MBPP base does the same (91.8 vs 90.5 vs 89.7). 4-bit
  A3B trades HumanEval for MBPP rather than failing across the board, and it
  posts the highest prefill throughput measured in this whole matrix (3319
  tok/s vs 3070 at 6-bit, 3177 at bf16); solo/conc2 decode sit within the
  6-bit config's own measured range (111.6–112.0 vs 106–115 solo; 99.7 vs
  82.5–104 conc2), so call decode a tie and prefill the clear win. Net: 6-bit
  is still the one to default to for quality-sensitive HumanEval-shaped work;
  4-bit is the one to reach for when MBPP-shaped code or raw prefill/decode
  throughput matters more than HumanEval+.
- **The 27B quants give up ~2 points of MBPP** against bf16 (92.1 → 89.9/90.5
  base; 77.8 → 75.7/76.2 plus) and 0.6–1.2 of HumanEval base. Both quants
  score *above* bf16 on HumanEval+ (90.9 vs 90.2).
- **4-bit ≈ 8-bit across every tier.** The 8-bit build's quality premium does
  not appear on these suites.
- Highest absolute scores on the box: 27B bf16 on MBPP (92.1/77.8) — at an
  unusable 9.8 tok/s (dense bf16 is memory-bandwidth-bound; ~54 GB weights
  against ~614 GB/s lands almost exactly on the measured rate).
- Prefill is compute-bound, not weight-bound: ~724–850 tok/s for every 27B
  precision; ~3.1–3.3k tok/s for the A3B (MoE, ~3B active params).

## Comparison with published numbers

Community EvalPlus references for the 35B-A3B (Q4_K_M leaderboard submission;
UD-Q6_K_XL) report 93.3/90.2 HumanEval and 90.2/75.4 MBPP — our 6-bit MLX
build lands within 2–3 HumanEval problems and above on MBPP. No official bf16
EvalPlus numbers exist for either model (vendor cards publish agentic
benchmarks only), which is why the references here were measured locally.
Our own 4-bit build's HumanEval+ (87.2) lands ~3 points below that same
Q4_K_M leaderboard reference (90.2) — most plausibly a quant-recipe
difference (affine gs64 body + a separately-sourced cyankiwi AWQ draft head
vs. their calibration mix), not a harness difference, since our bf16 and
6-bit rows track the community numbers consistently on the same suite.

## Serving configs measured

- 27B builds: mtplx 2.3.0-src, `--generation-mode mtp --depth 2`,
  `capture_commit` / `linear-gdn-from-conv-tape`, turbo profile, headquarter
  GDN tape kernel, width-2 lockstep cohort (`--scheduler-mode
  mtp_cohort_experimental`, 8-bit lane via the q8 M6 + qmv_wide kernels).
- A3B Balance: `--scheduler-mode ar_batch`, MTP depth 2 solo, turbo profile.
- A3B Speed (4-bit): same `ar_batch` / MTP depth 2 / turbo profile as
  Balance, `--verify-strategy target_prefix`, draft LM head at 4-bit gs64
  affine (`--draft-lm-head-bits 4 --draft-lm-head-group-size 64
  --draft-lm-head-mode affine`).
- bf16 references: `--generation-mode ar`, sustained profile (no MTP sidecars
  in the official repos).

Raw receipts (samples, `eval_results.json`, probe logs, chain logs) and the
matrix assembler live in the OpenSourceWTF repo:
`bench/evals-a3b6-vs-27b8-20260730/` (`scripts/assemble_matrix.py` regenerates
the table; its JSON-recount differs ≤0.6 pt from the EvalPlus console numbers
above, which are canonical, with one borderline exception: a3b4-speed
HumanEval+ recounts 87.8 vs. the console's 87.2, a 0.605 pt gap — flagged
here, not resolved).
