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
savings. FULL-164 CONFIRMATION: see "proj_requant full-164 HumanEval" below —
the 20-task gate's saturation resolved, McNemar p=1.0, verdict unchanged.

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

## q4 anchor (2026-07-21): kernel EXONERATED; serving infeasibility QUANTIFIED

Kernel microbench (bench_gather_qmm_q4.json, wave shapes, queued lane):
2-bit/gs64 6.16 µs, 2-bit/gs128 5.99, **4-bit/gs64 7.79 = 2x the bytes for
1.26x the time — per byte the MOST efficient arm measured.** MLX's q4 gather
needs no custom work; every q2-lane optimization (wave, router kernels,
scope-all, proj-quant, cadence, census islands) applies to q4 as-is.

q4 TPS, 96 envelope (islands 42 — ran on the q2-order placement, arms
straddled the census revert): **AR 3.39 / K2 3.00 tok/s**, hit 8-9%,
2.6-2.7 TiB read/cell, hard peak 95.6 GiB. Placement-corrected numbers would
improve marginally; the binding cost is 37 streamed layers x 10.1 MiB
records. **q4 is ~13x slower than oq2e-requant (42.33/40.18) at the same
envelope** — the anchor row justifying the 2-bit program. 80-envelope arm
re-armed with census-explicit islands (its count-resolution exposed a
pre-flight gap for census-only specs — follow-up: resolve_island_placement
before the runtime.py pre-flight plan).

Ops: census-vs-spec precedence matters — q4's own island-placement.json
ranks a DIFFERENT layer set than the q2-derived order (David's catch);
spec pin orders must not shadow per-bank census artifacts.

## q4 anchor complete (2026-07-21): 80-envelope AR 2.74 / K2 2.60

Census islands 34 (own placement, census-first live), hard peak 79.8 GiB,
hit 4-5%, 3.4-3.6 TiB read/cell. Microbench re-run strengthened the kernel
verdict (4-bit 7.19 µs = 1.15x the 2-bit time for 2x bytes). Q4 GOAL CLOSED:
kernel exonerated twice, optimization parity complete, TPS anchored at 96
(3.39/3.00) and 80 (2.74/2.60) — q4 serving is 13-16x under the
oq2e-requant champion at comparable envelopes; the 2-bit program is the only
serveable path on this box.

## q4-OPTIMAL found (2026-07-21, one window, 3 ratio arms x AR/K1/K2/K3)

Same ~80 GiB expert budget, three arrangements (census islands + frequency
cache), full lever stack (proj-quant q4, headers-only, deferred+pin, 8MiB
chunks, cadence, scope-all):

| arm | AR | K1 | K2 | K3 | hit |
|---|---|---|---|---|---|
| island-heavy 40+4GiB | 3.90 | 3.92 | 3.17 | 2.58 | .14-.17 |
| balanced 28+26GiB | 5.02 | 5.18 | 4.61 | 3.73 | .33-.41 |
| **cache-heavy 12+57GiB** | 5.54 | **6.29** | 5.63 | 4.67 | .41-.51 |

VERDICTS: (1) On a bank that doesn't fit, CACHE BEATS ISLANDS per byte
(dynamic frequency adaptation > static census pinning) — inverts the q2-era
C5 conclusion, which held only because that bank nearly fit. (2) K1 > AR in
every arm; K2+ monotonically worse (verify batches amplify misses faster
than acceptance pays — reads grow 1.1->1.5 TiB/cell with depth). Best q4
operating point: cache-heavy K1 = 6.29 tok/s, above the 5-6.4 historical
band. Extrapolation: zero-island max-cache might add ~5%; diminishing.
FINAL ANCHOR: q4-optimal 6.29 vs oq2e-requant champion 42.18 = 6.7x — the
2-bit program's justification at q4's own best configuration.

## proj_requant full-164 HumanEval (2026-07-21, paired one window): NO MEASURABLE COST — McNemar p=1.0

The saturated 20-task gate resolved on HumanEvalPlus-164, both arms in ONE
guarded window (attempt 1, no retries; conditions identical to the 0.95
baselines — MTPLX_HY3_ROUTER_SPLITK_M1 deliberately NOT set):

- stock-q8 **0.8720** (143/164) vs proj-requant-q4 **0.8659** (142/164);
  request_errors 0 both arms.
- Discordant pairs **9 (5:4)** — q8-only passes {83, 102, 130, 147, 148},
  rq4-only passes {26, 32, 100, 108}. **McNemar exact two-sided p = 1.0000**:
  the split is exactly what coin-flip generation divergence predicts; no
  directional quality signal.
- 17 both-fail tasks incl. HumanEval/10 (the lone 20-task failure — screen
  consistent with the full run).
- All 5 rq4-only failures are complete generations (finish=stop) failing
  logic assertions (102/147/83 AssertionError, 130 IndexError, 148 planet-
  ordering) — wrong-answer flips, not truncation or harness artifacts.

VERDICT vs David's "don't lose much" bar: PASS — net −1 task (−0.6 pp),
symmetric flips, p=1.0. The requant candidate's scorecard is now complete at
full task-eval resolution: HumanEval −0.6pp (p=1.0), wiki +1.6%, K2 +8.6%,
AR +26.3%, −3.5 GiB wired. Artifacts:
`humaneval_oq2e_full164_{q8,rq4}.json` (+ per-arm proxy logs
`full164_{q8,rq4}_proxy.log`).

## R1 (per-layer frequency allocation) KILLED at the CPU gate (2026-07-21): the 8x table was leakage

David picked R1 from the q4->10 brainstorm; its zero-GPU kill-test ran the
route-analysis script's issue-#9 HELD-OUT gate (chronological train/eval
split) on the same trace behind the famous 8x claim
(`hy3-q4-route-trace-1k-64.json`, 8058 slots ~= today's 80 GiB budget):

| policy (held-out, split 0.5) | hit | miss/tok |
|---|---|---|
| uniform_per_layer_lru | .8881 | 70.72 |
| **trained_dynamic_quota_lru** | .8880 | **70.78** |
| global_pool_lru | .8917 | 68.47 |
| uniform_per_layer_belady | .9189 | 51.25 |

The deployable trained allocator captures ZERO (identical to uniform; same
at split 0.75: 88.00 vs 87.88). The rebalancer moved 15-18 slots but its own
TRAINING curves offered only +35-47 hits of ~18-27k (~0.2%) — no signal, not
insufficient data. Root cause measured directly on the prefill phase (8184
samples/layer): per-layer top-102 coverage median .862 (concentrated — 53.1%
if uniform) but CROSS-LAYER stdev just .0274 — every layer concentrates the
same amount, so per-layer slot counts have nothing to trade. The legacy
`oracle_decode_frequency_allocation` 8.5 miss/tok (the "8x, beats Belady"
headline in docs/MMAP_VS_PREAD_FINDINGS.md and the mmap memory) is a pure
evaluation-oracle artifact: on a 64-step trace it just pins whatever appears.

Belady ceiling RECOMPUTED (handoff ask): perfect eviction buys 20-28% fewer
misses than LRU at this budget (~25-35 ms/tok at 1.5 ms/miss) — the hard cap
on ALL cache-policy work, unreachable in practice; global pooling adds ~3%.
Receipt: `r1_alloc_killtest_heldout.json`. Consequence for the 10-tps
program: miss COUNT is near its practical floor at this budget — the live
levers are hiding miss latency (R3 overlap), cheapening service (R2
page-cache L2, R4 read_chunk), or batch amortization (R5). Also cautions
GLM W1: the hy3 half of its evidence base is now dead; GLM's own
frequency-vs-uniform A/B remains unmeasured.

## R2-band/R4 replay window (2026-07-21): R4 CONFIRMED ~13-15 ms/tok; band DEMOTED to ~5

Model-free pread replay against the real q4 sidecar (wired ~0, cache fill
19.9 GiB, receipt `r2r4_band_replay_receipt.json`; first launch self-crashed
6x on a watchdog RSS cap that miscounted clean mmap pages as anon memory —
fixed cap 30 GiB + launcher no longer retries watchdog aborts):

- **A1 cold single pread 10.1 MiB: 0.93 ms (10.8 GiB/s).** A2 production
  8 MiB-chunk mirror: 1.05 ms — the 2-preads-per-record tax is 0.13 ms/miss
  = **R4 ≈ 13-15 ms/token for a read_chunk >= record-size change.**
- **B1 warm band via mmap 0.66 ms vs deep tail 0.91 ms alongside it** (no
  interference; band 200/200 resident after replay). Save 0.24 ms/hit x ~20
  band-hits/tok = **~5 ms/token — band DEMOTED** (the ~22 estimate assumed
  1.5 ms cold reads; raw cold is 0.91). Only worth revisiting as zero-copy
  serving that also skips per-miss runtime overhead.
- **Key discovery: ~0.6 ms/miss (~60 ms/token) of NON-READ miss overhead** —
  production wall budget 1.5 ms/miss vs 0.91 raw read, corroborated by raw
  read time 0.98 GiB/tok / 10.8 GiB/s = 91 ms of the 159 ms step. This pool
  (admission/slot/host-sync + serialization) is R3's target; fully-hidden
  reads ceiling ~11 tok/s.
- Cold band populate: 19.9 GiB in 2.1 s (sorted offsets ~sequential);
  warm sweeps 33-40 GiB/s.

Program re-rank: R4 (trivial, confirmed) -> R3 decomposition (hideable vs
deletable split of the 60 ms pool) -> R5 batch (aggregate only) -> band
(parked) . R1/page-cache-LRU remain dead. Awaiting David's pick.

## R4 INVERTED by paired A/B (2026-07-21): chunk16 LOSES −2.1% K1; chunking IS the fanout

One window, receipt-exact cache-heavy config, deterministic-identical routing
across arms (103.4 miss/tok, 0.975 GiB/tok both): chunk8 AR 5.709 / K1 6.270
(champion 6.29 reproduced) vs chunk16 AR 5.635 / K1 6.136 (−1.3%/−2.1%).
The replay's +13 ms/tok prediction for single-pread records was an artifact
of SERIAL microbenching: telemetry (r3_telemetry_chunk16.json, diagnostic
cell −10% overhead) shows the live reader averages **1.15 concurrent reads**
(8-reader pool 14% utilized), I/O active 100% of intervals, realized 5.38
GiB/s vs 10.8 isolated single-stream. Decode misses arrive ~1.5/layer
serially — record-chunk splitting is the ONLY queue depth the read path
gets, so 2 chunks/record beats 1. Attribution field:
'synchronous_fence_or_evaluation' (incomplete); GPU/DRAM counters
unavailable in this lane.

Verdict: read_chunk >= record is DEAD; the live lever is chunk-DESCENT
(more intra-record fanout). Sweep armed: chunk8 control vs 4MiB vs 2MiB,
same window discipline. If realized BW climbs toward the ~12.5 GiB/s
plateau, the ~91 ms/tok I/O share compresses toward ~65-75 ms.
Artifacts: r4_ab_chunk8.json / r4_ab_chunk16.json / r3_telemetry_chunk16.json.
Lesson (method): I/O microbenches must mirror the production SUBMISSION
model (concurrency), not just the syscall pattern.

## Chunk-descent sweep (2026-07-21): FLAT — the chunk lever is exhausted at ~2% total span

Paired one-window (identical routing all arms): chunk8 AR 5.611 / K1 6.282;
chunk4 5.678 / **6.323** (+0.65%); chunk2 5.713 / 6.322. Descent past 4 MiB
buys nothing; the whole read_chunk knob spans 16MiB 6.136 -> 4MiB 6.323
(~2%, chunk4 marginal best). Leans T2 (null-at-IO-layer): live realized read
bandwidth barely responds to chunk shape, so the dynamic-splitter idea is
LOW-EV at K1 (David's test-the-theory-first call — validated before any
build). Discriminating evidence queued in one window: HOL microbench through
the PRODUCTION reader (QD x chunk grid, predictions pre-registered in the
receipt), 2x5.06 MiB even-split arm, and David's pure-mmap probe
(slot_layout metal-mmap, no islands — "islands only work when the bank
nearly fits" — 256-token order-of-magnitude cell, non-gating).
Artifacts: r4_sweep_chunk{8,4,2}.json.

## I/O layer CLOSED (2026-07-21 eve): drive saturates at QD2; the lever is submission, not shape

**HOL theory test** (production PositionalExpertReader, native backend,
F_NOCACHE, predictions pre-registered; receipt hol_theory_test.json):
QD1 whole 0.951 ms (serial chunk8 tax +0.15); at QD2 ALL shapes — whole /
chunk8 / even-halves / even-thirds — within ±2.5% at the aggregate CEILING
12.7-12.8 GiB/s; QD3 same. T2 (null-at-IO-layer) CONFIRMED: no quantum/HOL
effect worth engineering; **two concurrent 10 MiB preads saturate this SSD.**
Live even-split serving A/B agrees: 2x5.06 MiB 6.302 vs chunk8 6.304 (dead
even; r4_even_*.json).

**NVMe research (sonnet web agent, cited in #130):** Apple ANS2/ANS3 exposes
ONE I/O queue (linear submission, per Linux nvme-apple) — no hardware
multi-queue to unlock; 4 independent Apple-silicon MoE-streaming projects
(ds4, hypura, mac-code, SwiftLM) all converged on N-thread sync pread +
F_NOCACHE; MTLIO/dispatch_io/aio: no evidence, do-not-port (MTLIO's GPU-
timeline event gating unmeasured, parked as a 50-line curiosity). F_NOCACHE
is a HINT needing 16 KiB alignment — our offsets/chunks are exact multiples.

**Synthesis:** live decode realizes ~5.4-6.1 GiB/s at mean 1.15 in-flight
reads — between QD1 (10.4) and QD2 (12.7) — so the entire remaining I/O gap
is SUBMISSION-SIDE serialization (the layer loop waits on 1-2 misses), not
read shape, pool size, chunking, or API. read_chunk verdict: keep 8MiB (4MiB
+0.65% marginal, within pairing noise; 16MiB genuinely worse at QD1-ish
depths). The 10-tps program's remaining levers: R3 overlap/host-overhead
(~60 ms/tok pool), batch (R5), fewer bytes (2-bit program).

**mmap probe:** first launch rejected at configuration — metal-mmap requires
--expert-integrity at-open (no per-record hashing on the mapped path);
relaunched with at-open (one-time full-sidecar hash ~15 min). Probe pending.

## mmap probe outcome (2026-07-21): metal-mmap is UNBENCHMARKABLE in this lane today

With at-open integrity the config gate passed; the run then failed closed at
WARMUP: "hy3-q4 d0 decode expert-cache has no routed assignments" — all
expert_streaming_counters zero. Load peak 13.9 GiB (trunk-only; the
mapped-store unwired accounting works). The mapped path executes outside the
expert-cache counters this lane hard-gates on, so pure-mmap serving cannot
produce a valid cell without integration work (wire counters/gates through
MappedExpertStore). Given streamed-mmap physics is independently condemned
(demand-fault 1.4 GiB/s flat; kernel-LRU = worst simulated policy; pread
12.9 GiB/s), further investment is low-EV — David's call whether the
integration is worth doing just to close the measurement.
Artifact: q4_mmap_probe.json (failure record).

## R3 PRICED (2026-07-21 late): 159 ms = ~100 read-wait + ~37 GPU + ~22 host; 10 tok/s is REACHABLE

Instruments: per-layer replay (r3_perlayer_replay.json — production reader,
production per-layer submission; 48 steps, 8173 misses) + roofline-profiled
K1 cell (r3_roofline_cell.{json,log}, d1 6.229 diagnostic ~= headline 6.28).
Caveat: oQ4e download ran concurrently (~1% drive BW; singles 1.06 ms vs
0.93 clean — direction unaffected).

DECOMPOSITION of the 159 ms K1 step:
- **Read-wait ~100 ms** (replay: 165.7 ms/tok at 170 synthetic miss/tok,
  scaled x103.4/170.3). Group walls scale LINEARLY with miss count
  (k=1 1.16 / k=2 2.01 / k=3 2.92 / k=4 3.73 — ~0.97 ms per extra read).
- **GPU ~37 ms**: attention 15.3 (46.7 MB/call, 45% of 539 GB/s ceiling),
  routed experts ~14 (est. via shared-expert calibration 476 GB/s), router
  4.3 (compute/occupancy-bound), shared 1.8 (88% ceiling), MTP head ~2.
- **Host residual ~22 ms** (admission/bookkeeping/dispatch).

KEY MECHANISM — BURST-READ SERIALIZATION (new, reproducible): a burst of k
reads submitted together through the warm production executor completes in
~sum time (pair 1.90 ms, triple 2.71) not ~max time (1.65/2.5 ideal), while
steady-state looping workers achieve true bandwidth-sharing (HOL qd2 1.548
ms/record at 12.7 GiB/s aggregate). Fresh-thread spawn exonerated (0.06 ms).
The live 1.15 mean-active-readers despite 1.5-miss groups is this effect.
Suspects for the fix: chunk-loop GIL crossings on the burst path, or route
misses down the EXISTING batch scatter-preadv path (expert_banked) = ONE
native call per layer group, kernel-level concurrency, no GIL.

LEVER PRICES (stacking):
- **(A) Fix burst concurrency**: group walls sum->max ~= read-wait 100 ->
  ~50-60 ms => ~110-120 ms/tok = **8.4-9.1 tok/s**. Software artifact, not
  hardware (drive+reader both proven QD2-capable).
- **(B) Resident-first same-layer overlap** (David's design; canonical
  accumulation order preserves bit-exactness): hide min(GPU ~0.5 ms/layer,
  read/layer) => additional ~15-25 ms => **~10-11 tok/s** after (A).
- (C) Early submission at route time: few ms, folds into (A)/(B).
VERDICT: q4->10 is priced as reachable via (A)+(B). No implementation
without David's pick.

## CORRECTION (2026-07-21 ~21:50): fix-A re-priced ~9 ms (was 40-50); (B) overlap carries the program

Quiet-disk burst re-probe (sidecar build finished, no background traffic):
single 1.110 ms; bursts k=2/3/4 = 1.880/2.698/3.523 ms vs concurrent-ideal
1.56/2.34/3.11 and serial 2.22/3.33/4.44 — **51/64/69% overlap efficiency
already present**. The earlier "sum-not-max" serialization read used a
download-contaminated single baseline; the R1 lesson (verify before build)
caught it pre-implementation. GIL exonerated separately (spinner probe:
native read releases it).

Re-priced against the REAL-trace burst distribution (per token: 22.0 k=1 /
14.6 k=2 / 6.6 k=3 / 2.4 k=4 / 1.2 k>=5 groups): perfect batching recovers
~9 ms/token. Read-wait stands at ~83-99 ms/token but is mostly IRREDUCIBLE
read time near the drive ceiling, not serialization.

REVISED PROGRAM: **(B) resident-first same-layer overlap is the primary
lever** — hide reads behind ~37 GPU + ~22 host ms => floor max(read, rest)
≈ 100 ms => ~10 tok/s. (A) batched non-blocking group reads DEMOTED to
(B)'s foundation (~9 ms standalone; required so miss reads stop blocking
the layer loop). Design: submit layer's misses at route time (batch
scatter-preadv, non-blocking) -> compute resident experts + shared ->
accumulate partials in canonical expert order on arrival (bit-exactness
preserved) -> parity-gated paired A/B.
Also: oQ4e sidecar COMPLETE (experts.bin 161,036,107,776 B — byte-identical
geometry to old q4; manifest w/ record hashes; spec + benchmark registered
5df7d7d). Eval ladder pending a window.

## FIX (B) IMPLEMENTED (2026-07-21): --moe-overlap = single batched miss part + run-coalesced reads + measured overlap telemetry (commit 9865f0b)

Design per David's pick (resident-first same-layer overlap; batched group
reads as its foundation). Config-gated: `overlap_miss_reads` /
`--moe-overlap`, default OFF; OFF leaves the per-expert split-part decode
path untouched (behavior-locked by test). Requires component-banks.

WHAT CHANGED (ON):
- `begin_split_route` submits a decode layer's misses as ONE part -> one
  future, one admission pass ahead of the wait, one policy commit, one
  miss-gather dispatch. (Same part shape prefill always used; only the
  phase gate is new.) Per-expert parts each paid their own executor task,
  lifecycle claim, admission, pin pass, ReadyRoute, and per-part
  iter/claim/commit bookkeeping — the ~0.6 ms/miss non-read overhead pool
  (~60 ms/token at 103 miss/tok) this lever targets.
- Pool partitions the part's loads into sidecar-adjacency runs: run >= 2
  -> one scatter preadv batch (`_fill_batch`); scattered records keep
  their own reads on the 8-worker pool, so batching never serializes
  non-adjacent misses. (Near-uniform routing makes adjacent same-layer
  miss pairs rare — the concurrency-preserving fallback is the common
  case; coalescing is opportunistic.)
- Hit + shared dispatch before the miss wait was already in place under
  split-route-release=deferred; the availability split and canonical
  accumulation were already partition-invariant — LOCKED by a new
  bitwise test: gather_qmm subset dispatch + position-order reassembly
  reproduces the fused 8-expert wave BIT-EXACTLY on real shapes (hidden
  4096, expert_hidden 1536, gs64 affine q4; partitions 8/0, 6/2, 0/8,
  interleaved 3/5; run-to-run deterministic). The [rows,1,K] gather
  calling-convention shape is asserted inside the test harness.
- Telemetry (the acceptance counter, measured, never tok/s-inferred):
  slot metrics `batched_miss_parts`/`batched_miss_records` prove the
  batch path executed; `overlap_split_routes`,
  `overlap_gpu_dispatch_ns` (host ns of GPU work built+submitted while a
  miss read was open), `overlap_exposed_wait_ns` (residual blocking wait
  after dispatch = the read-wait the overlap could NOT hide). Every
  benchmark observation row carries them as `overlap_telemetry`.

TDD EVIDENCE (tests/test_expert_overlap_split.py, 9 tests, all green):
batch == N single reads bitwise via per-record sha256 verification on a
sidecar-backed component-banks runtime; adjacency coalescing observed via
reader read_operations (adjacent [1,2,3] -> 1 op, scattered [5,7] -> 2);
short-read and pre-set-cancellation FAIL CLOSED; knob OFF keeps 4
per-expert parts and zero overlap counters; telemetry counters recorded
under a slowed reader with a warm hit + cold miss route. Benchmark flag
test: --moe-overlap -> runtime config -> per-row overlap_telemetry.

FULL SUITE: touched-area files 309 passed. Full tests/ shows 2
pre-existing test_settings_audit failures (MTPLX_HY3_ROUTER_SPLITK_M1
unclassified — fails identically at HEAD b9c28c3, another lane's env
var) plus sdpa_gqa_packed/streamed_rans failures that appear ONLY under
the full-suite run and pass in isolation both at HEAD and with this
change (suite-order/GPU-state flakes, statically untouched by this diff;
second full-suite run pending as corroboration).

Acceptance window (paired, one window, champion cache-heavy config,
armA control OFF expect ~6.28 / armB ON, gates + telemetry split): NEXT.
