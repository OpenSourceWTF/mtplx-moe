# Qwen4 M=3 Whole-MoE Target Lane

## Goal

Improve the production Qwen3.8 Flash-Next workload by extending the existing Qwen4 whole-MoE target kernel from logical M=2 to logical M=3. The implementation must preserve the validated model arithmetic while reducing the cost of depth-2 verification. The owner explicitly accepts sampled near-boundary flips from the measured BF16-equivalent lane; exact token-digest identity is not a promotion requirement.

The production promotion workload is 16,384 input tokens and 1,024 output tokens with thinking enabled, `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, and `repetition_penalty=1.0`.

## Evidence and boundary

The current M=2 production control averages 53.62 tok/s. Depth 2 reduces verifier calls from 603 to 479, but falls to 41.13 tok/s because each M=3 target call is substantially more expensive. Matched MLX traces attribute that increase to stock M=3 graph work and copies: GPU busy time grows from 25.76 ms to 43.97 ms per verifier call, dispatches from 3,951 to 5,498, and command buffers from 138.5 to 243.5. Explicit completion waits are effectively zero, so runner-side or kernel-drain waiting is not the primary problem.

Depth 3 is slower still at 37.21 tok/s and provides no acceptance benefit sufficient to justify an M=4 lane. This design is therefore limited to M=3.

## Design

Parameterize the existing exact Qwen4 whole-MoE kernel sources and bindings by a construction-time row count of either 2 or 3. Both variants use the same stage arithmetic, ownership, tiling, data layout, compilation behavior, and routing semantics already established for M=2:

- hidden size 2,560, 512 routed experts, top 10 experts, intermediate size 640, and 11 activation slots;
- per-row precise router softmax and unchanged top-10 ordering;
- unchanged routed-expert and shared-expert arithmetic;
- stage 1 geometry of 128 x 32 groups and stage 2 geometry of 440 x 128 groups;
- stage 3 geometry derived as `rows * 160 x 128` and router work derived as `rows * 256`.

The row count is a compile-time source constant for each bound variant, not a dynamic kernel argument. The construction path compiles and binds explicit M=2 and M=3 callables, runs exact self-checks for each, and installs the route only if both pass.

The installed execution route branches only on the genuinely dynamic logical M:

- M=2 calls the prebound M=2 whole-MoE lane directly.
- M=3 calls the prebound M=3 whole-MoE lane directly.
- Other widths use the explicit stock route selected during construction.

There are no eligibility checks, metadata validation, environment reads, engagement counters, or try-custom-then-fallback behavior in either enabled hot path. Any model, shape, layout, dtype, compilation, or self-check failure prevents installation and fails once before measured generation.

## Correctness and failure handling

Focused tests lock down generated source constants and geometry for rows 2 and 3, exact top-10 routing for three rows, construction-time self-checking of both variants, direct routing for M=2 and M=3, and the explicit stock route for other widths.

The candidate is rejected if any of the following occurs:

- M=3 exceeds the construction self-check tolerance, produces invalid output, or changes arithmetic beyond the explicitly accepted BF16 near-boundary behavior.
- M=3 occupancy or per-row cost does not offset the reduction in verifier calls.
- The candidate depends on a hot-path validation, fallback, or proof counter.

## Benchmark and promotion gate

Benchmark under `/tmp/mtplx-gpu-exclusive.lock` against an unchanged depth-2 control, using the exact production workload and settings above. Run matched warmups and repeated measured samples, record throughput, verifier calls, acceptance by depth, repair cost, target evaluation time, target-forward time, output digest, and profiler evidence when attribution is needed.

Promote and commit only if the M=3 candidate passes both construction self-checks, produces valid output with zero repair, and repeatably improves the matched production depth-2 control. Record any digest and acceptance change explicitly. A production win may then be measured on the short greedy palindrome prompt as a discussion-only result; that number is not a promotion gate. If the production result still does not yield a repeatable 90+ tok/s palindrome decode, continue with measured acceptance or target-forward work rather than redefining success.

## Alternatives rejected

- Duplicating separate M=3 kernel sources would invite arithmetic and maintenance drift from the proven M=2 implementation.
- Optimizing the stock piecewise M=3 path is unlikely to remove the systemic graph-dispatch and copy overhead identified by profiling.
- Building M=4 is not justified by the measured depth-3 acceptance and throughput.

## Non-goals

This change does not alter speculative acceptance, repair semantics, sampling settings, the n-gram cache interface, or kernel behavior for logical widths other than 2 and 3.

## Measured outcome

The exact-parent control is commit `c226e2ae`; the candidate is `85d91258`. Both used the 16,384-input/1,024-output production workload, depth 2, `capture_commit`, one warmup, and the production sampler documented above.

| Lane | Run 1 | Run 2 | Mean | Acceptance | Verify calls | Repair |
|---|---:|---:|---:|---|---:|---:|
| M=2-only control | 41.0469 tok/s | 40.7780 tok/s | 40.9125 tok/s | `[331, 170]` | 479 | 0.0 s |
| M=2/M=3 candidate | 42.2599 tok/s | 43.4294 tok/s | 42.8446 tok/s | `[326, 158]` | 516 | 0.0 s |

The candidate improves the matched two-run mean by 4.72%. Its construction self-check maximum error is `0.001953125` BF16 for both M=2 and M=3. Both candidate runs produced the same digest, `deb2d15f...`, while both controls produced `9a52872f...`; this is the explicitly accepted near-boundary numerical change and is not represented as exact-output parity.

Candidate target evaluation time was 18.6278/18.4518 seconds and verify-forward time was 2.7993/2.5667 seconds, versus control target evaluation time 19.4006/19.5067 seconds and verify-forward time 3.1110/3.1469 seconds. The accepted-prefix path remained active: captured recurrent state was committed at `1 + accepted_count`, the rejected KV suffix was trimmed, corrections were deferred as the next primary, and measured repair time remained zero.

The discussion-only greedy palindrome pair measured 71.7910/72.1307 tok/s, mean 71.9608 tok/s, with the 100-token prompt and 32 generated tokens. This is below the existing depth-1 discussion result of approximately 82.05 tok/s and does not meet the 90 tok/s goal. It is not a production benchmark or a reason to select depth 2 for short greedy requests.
