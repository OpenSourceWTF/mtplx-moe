# Hy3 q2 (gs64, 2-bit) — Tier-1 correctness evals

Branch `eval/hy3-q2-2p6bit` @ merged `main` `3e1ef9d`. Artifact under test:
`~/.cache/huggingface/hy3-expert-only-mlx-q2` (2-bit, group-size 64, derived
bf16 → q4 → q2; source tencent/Hy3 @716aa724).

All four evals run **CPU-first, one record at a time**, via the repo's own
hash-verifying reader — **no GPU flock, no benchmark, no full-model load**
(eval 2 does a minimal per-expert Metal spot-check, <200 MB, freed between
experts). Each eval was chosen because its **own correctness is verifiable
against a ground-truth anchor**.

| # | Eval | Verdict | Headline | Correctness anchor |
|---|------|---------|----------|--------------------|
| 1 | Weight-space fidelity vs bf16 | **PASS** | median cosine **0.9148** (prior 0.9197) | decode path bitwise-identical to `mx.dequantize`; reproduces documented prior |
| 2 | Dequant round-trip / quant integrity | **PASS** | recipe reproduces shipped q2 **bit-exactly on Metal** (7/7) | q4 source sha256 == manifest's recorded source hash; binary byte match |
| 3 | Manifest & bank completeness | **PASS** | **15168/15168** records, 0 gaps, bpw = **2.50** | counts + byte-offsets are exact; on-disk size == declared |
| 4 | pass@k estimator unit test | **PASS** | **440/440** analytic grid match | closed-form `1 − C(n−c,k)/C(n,k)` |

## Eval 1 — Fidelity vs bf16 → PASS
Sampled 45 experts (low/mid/high layers × 5 ids = 135 projection tensors),
dequantized q2 with `mx.dequantize`, compared to the original tencent/Hy3 bf16.
- **Anchor (a):** decode path is bitwise identical to the MLX reference
  (round-trip a random `mx.quantize` tensor through the bank byte layout →
  `max_abs_diff 0.0`).
- **Anchor (b):** documented prior median cosine ≈ 0.9197, min ≈ 0.9112.
- **Result:** per-tensor cosine min 0.8804 / **median 0.9148** / mean 0.9133 /
  max 0.9302; rel-Frobenius error median 0.415. Median lands within 0.5 % of
  the prior → loader + tensor alignment validated.
- **Caveat:** per-tensor min 0.88 sits below the prior's *aggregate* min 0.9112
  because this enumerates individual gate/up/down tensors (gate/up are the low
  ones; down_proj is tight at 0.913–0.930) rather than a per-expert aggregate.
  The comparable *median* matches.

## Eval 2 — Dequant round-trip / quant integrity → PASS
Fed the shipped **q4** record through the repo's recorded q4→q2 quantizer
(`_convert_one_record`) and compared to the shipped **q2** bytes.
- **q4 source identity hash-confirmed:** `sha256(q4/expert-manifest.json)` ==
  the q2 conversion-manifest's `source.manifest_file_sha256`; producer
  mlx 0.31.2 = ours.
- **Root-caused finding:** `mx.quantize`/`mx.dequantize` round **differently on
  CPU vs Metal**. On **Metal the recipe reproduces the shipped q2 BIT-EXACTLY**
  — 7/7 experts, all 9 components (weight/scales/biases sha256 all match). On
  **CPU it diverges deterministically** (~19.5 % of packed 2-bit weight bytes
  differ; bf16 scales/biases still ~99.96 % identical; identical run-to-run).
- **Caveat:** bit-exact reproduction is **backend-specific**. A strictly-CPU
  reproduction lands ~19.5 % off even though the recipe is deterministic and
  exact on its production backend. Without Metal this eval would be **BLOCKED,
  not FAIL** — CPU divergence alone does not impugn the artifact.

## Eval 3 — Manifest & bank completeness → PASS
- Repo `scripts/verify_expert_manifest.py`: `valid: true` (19 shards).
- Independent sweep: **15168 records = 79 × 192**, missing 0, extra 0,
  duplicates 0, segments out-of-bounds 0; on-disk `experts.bin`
  89,464,504,320 B == manifest-declared size exactly.
- HF **expert-aware completeness** (commit `a476015`): reads complete
  (`incompleteness_reason = None`, bank present at declared 89.46 GB).
- **Honest correction:** effective width measures **2.50 bpw, not 2.6**
  (2.0 payload + exactly 0.5 bf16 scale+bias per gs64 group; records are
  16384-aligned and tile the bank with zero padding waste). The "2.6-bit"
  label rounds up.

## Eval 4 — pass@k estimator unit test → PASS
- `pass_at_k(n,c,k)` matches the closed form `1 − C(n−c,k)/C(n,k)` on the full
  440/440 (n≤10) grid + 6 named anchors.
- `summarize()` (fix `5ce0576`) equals the mean of per-task analytic pass@k for
  k∈{1,2,3}, skips under-sampled tasks, and the old-bug corpus (150 passes,
  n=5/task) summarizes to 1.0 without raising `c>n`.
- Repo pytest `tests/test_code_eval.py` and `tests/test_code_eval_gate.py` both
  pass.

## Most important caveats (across the set)
1. **Bit-exact q4→q2 reproduction is Metal-specific** — CPU FP rounding alone
   puts a strictly-CPU reproduction ~19.5 % off, though the recipe is
   deterministic and exact on its production backend.
2. The artifact is **2.50 bpw, not 2.6** — the label rounds up.
3. This is a **weight-space** fidelity harness. High weight cosine (median
   0.9148) is consistent with the prior but is **NOT** a quality-gate pass — it
   does not speak to the separately-recorded **15.4 % perplexity-gate failure**
   of this quant. That behavioral gate is Tier-2 (needs a guarded ~90 GiB
   window) and was intentionally not run here.
