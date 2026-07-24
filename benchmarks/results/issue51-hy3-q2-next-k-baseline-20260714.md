# Issue 51 Hy3-Q2 Next-K sustained baseline

## Reproducibility

- UTC completion: 2026-07-15T01:16:01Z
- Commit: `ddac4f252cd45e7681562971c5e89ad621935b83`
- Hardware: Apple M5 Max MacBook Pro, 18 CPU cores, 128 GB unified memory
- OS: macOS 26.5.2 (25F84), arm64
- Fan mode: system automatic; no manual override recorded
- Model: Hy3 expert-only MLX Q2 with BF16 MTP weights
- Prompt suite: deterministic `coding_agent_tail_v2`, raw non-thinking prompts at exactly 1,024, 2,048, and 4,096 tokens
- Sampling: temperature 0, top-k 1, top-p 1, seed 0, no stop-token or repetition stop
- Generation: exactly 1,028 output tokens per retained cell
- Repeats: one retained cell per context/depth after an 8-token discarded warmup for every cell
- Runtime: one resident model load, interactive QoS, component-bank slots, F_NOCACHE, 56 transient reader slots, 64 GiB expert cache, 112 GiB memory limit, 8 MiB reads, 8,192 live KV tokens
- Verifier: capture/commit, compiled verification off, persistent expert cache, committed MTP history
- Telemetry: resource instrumentation enabled; route tracing disabled
- Isolation: Qwen launch agent unloaded for the exclusive benchmark window

Command:

```text
MTPLX_SUSTAINED_PREFILL=1 MTPLX_COMPILED_VERIFY=off \
python scripts/benchmark_q2_mtp_depth_matrix.py \
  --model hy3-q2 \
  --contexts 1024,2048,4096 \
  --output-tokens 1028 \
  --hy3-depths 1,2,3,4,5,6 \
  --transient-slots 56 \
  --max-live-kv-tokens 8192 \
  --verify-strategy capture_commit \
  --compiled-verify-mode off \
  --no-trace-routes \
  --resource-telemetry
```

## Results

| Context | K | Ingest tok/s | Prefill s | Decode tok/s | vs K0 | Cache hit | MTP accepted/evaluated | MTP accuracy | Verify calls | Readers mean/peak | Saturation | AR parity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1024 | 0 | 38.441 | 26.638 | 6.046 | control | 96.085% | n/a | n/a | 1027 | 0.411 / 7 | 0.734% | yes |
| 1024 | 1 | 38.017 | 26.935 | 7.225 | +19.5% | 97.084% | 500 / 527 | 94.88% | 527 | 0.377 / 7 | 0.673% | no at 353 |
| 1024 | 2 | 37.628 | 27.214 | 8.063 | +33.4% | 96.841% | 641 / 754 | 85.01% | 387 | 0.512 / 7 | 0.914% | no at 353 |
| 1024 | 3 | 38.308 | 26.731 | 8.230 | +36.1% | 96.611% | 703 / 878 | 80.07% | 324 | 0.651 / 8 | 1.162% | no at 353 |
| 1024 | 4 | 38.017 | 26.935 | 5.989 | -0.9% | 95.097% | 661 / 963 | 68.64% | 367 | 0.948 / 11 | 1.693% | no at 406 |
| 1024 | 5 | 38.115 | 26.866 | 5.727 | -5.3% | 95.175% | 670 / 988 | 67.81% | 358 | 1.055 / 11 | 1.884% | no at 406 |
| 1024 | 6 | 38.090 | 26.884 | 5.470 | -9.5% | 95.340% | 673 / 1020 | 65.98% | 355 | 1.136 / 12 | 2.029% | no at 406 |
| 2048 | 0 | 52.423 | 39.067 | 5.681 | control | 95.255% | n/a | n/a | 1027 | 0.463 / 5 | 0.827% | yes |
| 2048 | 1 | 53.398 | 38.353 | 5.713 | +0.6% | 96.738% | 366 / 661 | 55.37% | 661 | 0.413 / 7 | 0.737% | no at 5 |
| 2048 | 2 | 50.736 | 40.366 | 5.709 | +0.5% | 96.985% | 454 / 880 | 51.59% | 573 | 0.497 / 9 | 0.887% | no at 5 |
| 2048 | 3 | 50.608 | 40.468 | 5.338 | -6.0% | 97.132% | 485 / 967 | 50.16% | 542 | 0.578 / 11 | 1.032% | no at 5 |
| 2048 | 4 | 50.889 | 40.244 | 4.807 | -15.4% | 97.093% | 498 / 1007 | 49.45% | 529 | 0.642 / 10 | 1.147% | no at 5 |
| 2048 | 5 | 50.708 | 40.388 | 4.528 | -20.3% | 97.189% | 499 / 1023 | 48.78% | 528 | 0.706 / 12 | 1.261% | no at 5 |
| 2048 | 6 | 50.824 | 40.296 | 4.311 | -24.1% | 97.247% | 499 / 1026 | 48.64% | 528 | 0.774 / 12 | 1.382% | no at 5 |
| 4096 | 0 | 76.323 | 53.667 | 5.834 | control | 95.859% | n/a | n/a | 1027 | 0.413 / 7 | 0.737% | yes |
| 4096 | 1 | 76.379 | 53.627 | 5.480 | -6.1% | 96.038% | 389 / 639 | 60.88% | 639 | 0.461 / 7 | 0.823% | no at 4 |
| 4096 | 2 | 76.383 | 53.624 | 5.683 | -2.6% | 96.281% | 495 / 856 | 57.83% | 532 | 0.570 / 9 | 1.017% | no at 4 |
| 4096 | 3 | 76.181 | 53.766 | 5.378 | -7.8% | 96.464% | 531 / 963 | 55.14% | 496 | 0.662 / 12 | 1.183% | no at 4 |
| 4096 | 4 | 76.264 | 53.708 | 5.142 | -11.9% | 97.403% | 533 / 1009 | 52.82% | 494 | 0.569 / 11 | 1.016% | no at 4 |
| 4096 | 5 | 76.522 | 53.527 | 4.850 | -16.9% | 97.478% | 534 / 1024 | 52.15% | 493 | 0.628 / 13 | 1.122% | no at 4 |
| 4096 | 6 | 76.065 | 53.849 | 4.597 | -21.2% | 97.534% | 534 / 1027 | 52.00% | 493 | 0.688 / 13 | 1.228% | no at 4 |

## Interpretation

Current unfused Next-K is strongly beneficial only at 1,024 prompt tokens, where K3 reaches 8.230 decode tok/s, 36.1% over K0. K1/K2 are effectively neutral at 2,048, and every speculative depth loses at 4,096; K2 is the least-negative 4,096 row at -2.6%.

K4-K6 are no-go depths at every context. Their additional accepted-token yield flattens while expert traffic and decode time rise. The reader pool is not saturated: mean occupancy never exceeds 2.029% of 56 slots, peak occupancy is 13, and full-capacity time is zero in every row.

All event, cache-offset, and final-state hard gates pass. Token parity remains diagnostic and blocks promotion: K1-K3 share one deterministic multi-row output trajectory and K4-K6 another at every context. First divergence occurs at output 353, 5, and 4 for the 1,024, 2,048, and 4,096 prompts. The next gates are cross-shape parity classification, NAX-only measurement, fused K-row expert execution, and fused+NAX measurement.
