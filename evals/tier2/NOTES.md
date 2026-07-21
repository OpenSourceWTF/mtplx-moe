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
