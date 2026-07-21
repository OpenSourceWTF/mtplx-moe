# Tier-2 — WikiText-2 perplexity, shipped hy3 q2 vs q4

Run 2026-07-20 in a guarded GPU window (`run_with_qwen_stopped.py` → `compare_streamed_quality.py`).
Output: `ppl_shipped_q2_vs_q4_wikitext2.json`.

## Result — FAIL (severe)

| Lane | model_key | root | PPL | mean_nll |
|------|-----------|------|-----|----------|
| q4 (control) | `hy3-expert-only-q4` | `hy3-expert-only-mlx-q4` | **2.860** | 1.051 |
| q2 (shipped) | `hy3-expert-q2` | `hy3-expert-only-mlx-q2` | **6.747** | 1.909 |

- **relative_perplexity_regression = 1.359 (135.9%)** vs the 5% gate → **FAIL**.
- NLL gap = 0.858 nats. Greedy token agreement = **3.7%** (19/512), first divergence at token 0–6 on all 4 prompts.
- Both lanes finite, `errors=[]`, full `token_count=4095/4096`, distinct roots/keys → genuine two-model comparison, **not** control-vs-control.

## Validity

- Load is **spec-correct**: `HY3_EXPERT_Q2` (`mtplx/expert_streaming_models.py:363`) = quant_bits 2, group_size 64, source `local/hy3-expert-only-mlx-q4` — matches the shipped bank and Tier-1's measured quant. Not a mis-load.
- Consistent with **Tier-1**: median rel-Frobenius error 0.415 (41.5% per-tensor) predicts catastrophic PPL; the 0.858-nat gap fits.
- `EXIT=2` came from the **wrapper's qwen-restore losing the flock race** to the next lane (`51-freq-alloc`), NOT from the comparison, which ran clean to completion and wrote this full JSON. qwen being down afterward = the other lane's window, not this run.

## Reconciliation gap (open)

This 135.9% is ~9× the **15.43%** recorded in `hy3-q2-never-quality-validated.md` for the same bank (that was a 0.143-nat gap). The `-direct` variant scored **125.7%** on this same `wiki.test.raw` corpus. So two independent runs + Tier-1 all agree the bank is severely degraded here, and **15.43% does not reproduce on this config** — likely a different corpus/chunking. A config-matched re-run is needed to explain the 15.43% figure; the *direction* (well over the 5% bar) is robust either way.

## Config used

`--memory-limit 103GiB --expert-cache-limit 60GiB --runtime-reserve 16GiB --max-live-kv-tokens 8192 --evaluation-tokens 4096 --chunk-tokens 64 --greedy-max-tokens 128`, corpus `mlx-kld/corpora/wiki.test.raw`, `--cache-policy frequency --cache-scope layer --slot-layout direct-slots`, full per-record hash verification. Pre-flight fit: q4 94.24 / q2 94.72 GiB of 103, `fits_fixed=True`, ~8–9 GiB headroom, memory knob never raised.

---

# Tier-2 — HumanEval via LiteLLM (2026-07-20): pass@1=0 is a HARNESS ARTIFACT

The guarded run went fully end-to-end — proxy served, `code_eval_gate` connected,
requests reached the handler — but **every request 500'd before any generation**:
`ModuleNotFoundError: No module named 'mtplx.benchmarks.resource_telemetry'`.
So `humaneval_hy3_q2.json` pass@1=0.0 (20/20 `request_error`) says **nothing about
the model** — zero tokens were generated.

Root cause: the campaign venv ships a **PEP660 editable install of `mtplx 2.0.2`**
whose `sys.meta_path` finder points at the stale primary checkout
`mtplx-hy3-ssd/mtplx` (branch codex/…), which lacks `benchmarks/resource_telemetry.py`.
That finder wins over `sys.path`, so the handler's `import mtplx.*` (via
`benchmark_q2_mtp_depth_matrix.py`) resolved to the stale worktree.

Fix (handler.py `_import_benchmark_module`, GPU-free-verified under the campaign
python where the finder is active): drop the mtplx package finder (keep the
native-extension finder), force the eval worktree to the front of `sys.path`,
evict stale cached `mtplx`. Verified: `mtplx` → eval worktree, `resource_telemetry`
imports, native ext still loads.

NOT re-run end-to-end: the box kernel-panicked at 23:10 (GLM freq-alloc buffered
arm — a different lane, not this run), and HumanEval is expected ≈0 from the
Tier-2 PPL (135.9% regression, 3.7% token agreement). The LiteLLM serving path is
proven functional GPU-free except the model-load+generate call itself.

---

# Tier-2 — HumanEval COMPLETED (2026-07-20 23:49): pass@1 = 0.80 (16/20) — "expect ≈0" was WRONG

Two further guarded attempts after the harness-artifact fix:

1. **23:41 attempt — Metal GPU watchdog timeout** during the cold ~90 GiB
   island-fill load on the freshly-panicked box
   (`kIOGPUCommandBufferCallbackErrorTimeout`, proxy SIGABRT). Archived as
   `humaneval_hy3_q2.gpu-timeout.json`. No generation occurred.
2. **23:47 warm retry — CLEAN END-TO-END SUCCESS.** Page cache warm from the
   aborted load; first request (load + generate) 58.9 s, run total 134.8 s
   wall. `EXIT=0`, flock released, qwen restored. No island-count reduction
   needed (champion config as baked).

## Result

- **pass@1 = 0.80 (16/20)**, HumanEvalPlus-v0.1.10 first 20 tasks, 1 sample/task,
  chat endpoint, temp 0.0, seed 42, max_tokens 1024, `no_think`.
- All 20 completions finished at natural `stop`: 25–152 completion tokens
  (median 79, total 1619). No truncation, no empties, no request errors.
- Failures (4): `HumanEval/6` NameError `current`; `/7` NameError `strings`;
  `/10` NameError `is_palindrome` (the prompt-provided helper — possibly a
  chat-extraction artifact dropping prompt context, which would only UNDERcount
  passes); `/12` AssertionError. These are coherent code-shaped mistakes,
  not gibberish.
- Provenance: model `hy3-q2` via LiteLLM proxy :18183, handler default root
  `~/.cache/huggingface/hy3-expert-only-mlx-q2` (no env override in
  `serve_and_eval.sh` — verified), mtplx 2.0.2, dataset sha256 `42526ec0…`.

## Interpretation — revises the lane's headline

The PPL FAIL (135.9% rel regression, 3.7% greedy token agreement vs q4) is
real, but the inference "therefore pass@1 ≈ 0" was wrong: PPL 6.747 absolute
is degraded-but-functional, not gibberish. The bank writes mostly-correct
short Python. Token agreement measures divergence from q4's exact path, not
task competence.

Caveats: n=20 (95% CI on 0.80 is roughly [0.58, 0.93]); first-20 task IDs,
not a random sample; NO q4/bf16 HumanEval baseline exists on this harness
(memory: no published Hy3 HumanEval anchor either) — so the DELTA attributable
to q2 quantization is unmeasured. The 5% quality gate verdict stays FAIL on
PPL; HumanEval says the failure mode is mild-on-code, not catastrophic.

Completions are NOT persisted by `code_eval_gate.py` (rows carry status/usage
only) — capturing sample generations for inspection needs either a gate flag
or a one-off request in a future GPU window.

---

# oQ2e campaign (2026-07-21): mlx-community/Hy3-oQ2e on the mtplx runtime

David's directive: serve as close to stock as possible, requantize nothing.
Integration: spec `hy3-expert-oq2e` (2-bit gs128 experts, q8-gs64 residents via
the config-driven pre-quantized loader path, NO proj-quant), manifest into the
stock shards, experts.bin sidecar (byte-identical extraction, sha
c72fb8c0…). MTP head = our external bf16 layer-80 artifacts (same base
revision).

Local checkpoint corrections (originals kept as *.orig-published):
- index total_size 89,871,151,524 -> 89,870,806,272 (publisher metadata
  overstates true header inventory by 345,252 B; fail-closed check caught it).
- config num_nextn_predict_layers 0 -> 1 (quantizer stripped MTP tensors and
  rewrote the count; base tencent/Hy3@716aa724 declares 1 — restoring the
  architecture-true value lets the trained external head attach).

Script fix en route: depth-matrix `_requests_from_args` binary hy3/glm dispatch
sent any new campaign key down the GLM branch (crossed MTP path). Fixed +
regression tests (71a9f03).

## Decode results (1024/1024, guarded windows, zero requantization)

| envelope | AR | K2 | K3 | notes |
|---|---|---|---|---|
| 96 GiB (limit 103, islands 79, fully resident) | 30.17 | **36.82**† | **31.81** | 0 misses; load peak 90.7 GiB, hard peak 91.9 |
| 88 GiB (limit 95, islands 74 + 5 streamed) | 22.35 | 25.93† | 22.15 | hit rate .26-.33, 101-131 GiB read/cell; hard peak 87.7 |

† K2 shows ONE token-divergence event per envelope vs the AR reference (96:
token 893; both "unclassified" attribution). K3 is bit-exact both envelopes.
Conservative headlines are the K3 numbers. Acceptance identical across
envelopes (1.637@K2 / 2.024@K3) — a model property, unaffected by streaming.

Reference points: our shipped q2 champion 22.74 (K3, islands 60); oMLX
publishes 29.4 tok/s for this checkpoint on the same hardware class.
Fully-resident pre-flight at limit 95 fails by 4.21 GiB (fixed footprint) —
islands 74 is the honest 88 GiB configuration; the >30-at-88 goal is NOT met
by this bank on the current streaming stack (best exact: 22.15).

## Weight fidelity vs bf16 (evals/fidelity_oq2e_vs_bf16.py, CPU, committed)

Median cosine **0.9212** / rel-Frobenius median **0.397** on the exact Tier-1
sample — BEATS the shipped q2 bank (0.9148 / 0.415) at smaller size (2.44 vs
2.50 bpw). Per-projection medians uniform (~0.920-0.9215): the imatrix rescues
gate/up, which our plain affine recipe hurt most (they dipped to ~0.88).

## Open

- K2 divergence attribution (one flip/run; near-tie logit under batched
  verify is the suspect — unproven).
- gs128 disables the fixed-M4 island fast wave path (hardcoded 64; falls back
  to generic gather) — island decode has headroom if a gs128 variant lands.
- HumanEval on oQ2e via the LiteLLM lane (env switch, needs a window).

## oQ2e WikiText-2 PPL (2026-07-21 03:11, config-matched to Tier-2)

**oQ2e 6.443 vs q4 2.860 = 125.3% relative regression** (q4 control reproduced
Tier-2's 2.860 exactly; greedy agreement vs q4 3.9%, diverges at token 2).
Compare: shipped q2 135.9%, `-direct` bf16-derived q2 125.7% — three
independently derived 2-bit banks cluster at 6.4-6.7 on this config while all
evidence says task competence survives (q2 HumanEval 0.80; oQ2e published
benches ≈ its 2.68bpw sibling; oQ2e weight cosine 0.9212). Conclusion
sharpened: the 5%-PPL gate at this config is structurally unpassable at 2-bit
expert precision and mostly measures calibration loss, not task damage. Any
future bank verdicts need a task eval alongside PPL.

Ops notes: shard sha256 provenance is REQUIRED by compare_streamed_quality's
resident check — build manifests with `--hash-shards` (rebuilt + sidecar
--overwrite, byte-identical sha c72fb8c0…). Wrapper exits 1 on the qwen-restore
flock race even after writing full results — launchers must treat the output
artifact as the success signal (attempt-269 window was burned re-learning
this; attempt-1's q4-only receipt kept as *.attempt1-q4-control-only.json).

## oQ2e HumanEval (2026-07-21): pass@1 = 0.95 (19/20)

Same 20-task HumanEvalPlus gate as the shipped-q2 run (greedy, seed 42,
chat endpoint, no_think), served via LiteLLM with env overrides
(MTPLX_HY3_MODEL_KEY=hy3-expert-oq2e, islands 79 fully resident,
proj_quant=none). All completions natural-stop, 77-238 tokens (median 147 —
q2's was 79). Sole failure: HumanEval/10 AssertionError (make_palindrome —
the task q2 also failed). Provenance note: a silent fallback to the q2 bank
under these overrides is structurally impossible (bf16 residents unquantized
+ 79 q2-record islands ≈ 101 GiB > the 103 limit with KV/reserve — pre-flight
would reject); behavioral fingerprint (token profile, score) also differs.

Quality triple for oQ2e vs shipped q2: cosine 0.9212 vs 0.9148; PPL 6.443 vs
6.747 (both far over the 5% gate); HumanEval 0.95 vs 0.80. The bank-quality
ordering is consistent across all three; PPL magnitude remains the outlier
measure (calibration, not competence).

## Wave-port K3 re-measure (2026-07-21): NULL result under AR control

gs128 wave eligibility (489790b) re-measured at the champion envelope:
AR 28.92 / K3 30.54 (both bit-exact, hard peak 91.9 unchanged) vs pre-wave
AR 30.17 / K3 31.81. Raw drop ~4% in BOTH cells — but AR shares no wave code,
so it is the window-drift control: K3/AR ratio 1.0560 post vs 1.0544 pre =
+0.15%, sub-noise. **The wave port recovers ~nothing at K3.** Consistent with
the gather_qmm microbench (gs128 0.92x the time of gs64 at wave shapes —
committed as bench_gather_qmm_gs128.json): both paths call the same kernel;
the wave only restructures surrounding ops. Attribution revision: the q8
residents carry essentially the entire oq2e-vs-q2-champion decode deficit;
wave ineligibility was worth ~0. The port stays (bitwise-locked, correct,
extends to GLM dims) as eligibility hygiene, not as a perf claim.
Cross-window comparisons on this box carry ~4% drift under sustained load —
single-window paired arms only.

## K2 compile-island A/B (2026-07-21, single window, paired arms): NULL

base AR 30.54 / K2 38.00; MTPLX_HY3_COMPILE_ISLAND=1 AR 30.70 / K2 38.31.
K2/AR ratio 1.2443 vs 1.2479 = +0.3%, sub-noise. Compile-island is worth
nothing at full residency and is non-bitwise by design — verdict: leave OFF,
permanently, at this operating point. Third op-restructuring null in a row
(wave, compile, historic compile-the-forward): the gather kernel is the cost
and it is ALU-bound; surrounding-op work does not move it. K2's persistent
single-divergence signature reproduced in both arms (acc/verify 1.637
identical). Window itself ran ~3% faster than last night's (drift confirmed);
paired arms agreed internally.

## oQ2e per-component roofline (2026-07-21, MTPLX_ROOFLINE_PROFILE, K2 window)

Ceiling 502 GB/s measured. attention 84.4 MB/call @313 GB/s (62%) = 21.6
ms/tok DOMINANT; moe batch-1 gather 192 GB/s (38%, occupancy); router 5.5
ms/tok @5% ceiling (occupancy — future lever); shared 469 (93%, saturated);
32-assign wave 463 (92%, saturated). T0a: MoE inefficiency is batch=1-ONLY
(192 vs 463 GB/s at 32 assignments). Attribution: q8 ATTENTION owns the
4 tps deficit vs the q4-attention champion (2x bytes/call on the largest
component); proj-requant q8->q4 over proj_quant_covers scope is aimed
correctly — expected ~9-10 ms/tok raw before overlap. Instrumented tok/s
diagnostic-only.

## proj_requant quality gates (2026-07-21): BOTH PASS — candidate confirmed

Gate 1 HumanEval: 0.95 (19/20) identical to stock-q8 (same lone HumanEval/10
failure, same token profile). Gate 2 WikiText-2 (config-matched): requant
6.547 vs stock 6.443 = +1.6% relative (q4 control 2.8596 reproduced exactly,
third time). Both inside David's "don't lose much" bar; the requant arm still
beats shipped q2 (6.747) on PPL. Remaining: paired K2 speed A/B (stock-q8 vs
requant-q4) to price the win — roofline predicts ~9-10 ms/tok raw attention
savings.

## proj_requant speed A/B (2026-07-21, paired single window): +8.6% K2, +26.3% AR

stock-q8 AR 26.89 / K2 32.63 vs requant-q4 AR 33.97 / K2 35.44; hard peak
91.9 -> 88.4 GiB. Window globally slow (drift) — ratios are the evidence;
scaled to fast-window baselines the projections are ~41 K2 (above the 40.59
championship) and ~38 AR. Acceptance 1.637 -> 1.564 (trunk perturbation costs
the MTP head slightly; net K2 still +8.6%). Candidate scorecard complete:
HumanEval identical, wiki +1.6%, K2 +8.6%, AR +26.3%, -3.5 GiB wired.
Adoption as serving default = David's decision.

## Champion-41 reproduction (2026-07-21): K2 42.33 — NEW ALL-TIME CHAMPION

oq2e + proj-requant q4, 96 envelope, islands 79, championship shape (AR+K2):
AR 37.79 / **K2 42.33** (hard peak 88.4 GiB). Beats the q2-bank championship
40.59 by +4.3% on a smaller, higher-quality bank with quality gates passed
(HumanEval 0.95 identical / wiki +1.6%). Projection from the paired-ratio
scaling (~41) confirmed. Known caveats carry: K2 single-divergence signature
(parity False), acceptance 1.564.

## Router M1 scope A/B (2026-07-21 13:10, paired): +4.6% AR; K2 DIVERGENCE ATTRIBUTED

scope-mtp AR 30.60 / K2 37.96 (parity False) vs scope-all AR 32.01 / K2 38.15
(**parity True**). AR +4.6% as predicted (trunk M1 stock->kernel); K2 +0.5% =
control held. The persistent K2 single-token divergence is ATTRIBUTED: a
router-numerics mismatch between lanes (AR reference routed via the stock
host path at M1 while K2 routed via the fp32 split-K kernel; one near-tie
logit forks the sequence). Same kernel numerics both lanes -> parity
restored. Follow-up (one window, when the box frees): champion + requant +
MTPLX_HY3_ROUTER_SPLITK_M1=all — expect 42.33-class K2 with parity True.

## PARITY-STAMPED CHAMPION (2026-07-21): K2 42.18 / AR 40.18, parity True

oq2e + proj_requant q4 + islands 79 + MTPLX_HY3_ROUTER_SPLITK_M1=all at the
96 envelope: AR 40.18 (parity True) / K2 42.18 (parity True), hard peak 88.4
GiB. Reproduces the 42.33 champion within noise WITH the divergence resolved
(router-numerics attribution confirmed by construction: same kernel both
lanes -> exact parity). This is the complete champion config: every caveat
closed — quality gates passed, bit-exact, one envelope tier below the old
champion's memory.
