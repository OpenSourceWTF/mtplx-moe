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
