# M2 contract — runtime consumption of mixed-official-v1 (issue #51)

Orchestrator-decided 2026-07-21, from the consumer-surface map (Explore agent,
session df2d7609). Build to this. M1 (`794000a`) produces the bank; M2 makes
`ExpertStreamingRuntime` serve it. Single source of truth for tier layouts is
`mtplx/expert_mixed_official.py` — the runtime IMPORTS it, never re-derives.

## Decisions (D1–D8)

- **D1 — tier knowledge is manifest-exact.** The served tier map comes from
  the loaded manifest's `quantization.layer_tier_map`, never from the spec or
  the recipe file. Spec carries only `expert_codec="mixed-official-v1"`.
  Fail closed if any routed layer lacks a tier entry.
- **D2 — per-layer record bytes at BOTH memory gates.** `plan_expert_memory`
  replaces `streamed_layer_count * spec.expert_record_bytes` with a sum of
  per-layer record bytes (uniform slot COUNT per layer is kept; byte size per
  layer differs: t158 layers 5,701,632 B / affine2 layers 6,684,672 B for the
  real bank). Both gates (runtime.py preflight + `ExpertStreamingRuntime.open`)
  must use the manifest-derived per-layer sizes.
- **D3 — per-layer bank geometry.** `make_mlx_component_bank_allocator` gets a
  per-layer expected signature for mixed mode (derived from that layer's tier
  via `expert_mixed_official.TIERS` + the layer's records), enforced within
  the layer; uniform modes keep the existing single-exemplar check untouched.
- **D4 — per-projection-group dispatch.** `HotExpertSwitchGLU` is already a
  per-layer instance: at bind it resolves `(gate_up_tier, down_tier)` for its
  layer. Forward for t158-tier layers: gate/up via `shadow_gather_mm`, down
  via `mx.gather_qmm(bits=3)`. For affine2-tier layers: gate/up
  `gather_qmm(bits=2)`, down `gather_qmm(bits=3)`. `group_size=64` everywhere.
- **D5 — manifest parsing.** `ExpertManifest` learns mode
  `"mixed-official-v1"`: `quant_bits` is None for mixed (there is no scalar
  bits); `TensorSegment` accepts + preserves `quant_tier`; `ExpertRecord`
  accepts the 7-segment (t158 gate/up + affine3 down) and 9-segment (affine2
  gate/up + affine3 down) component orders, exactly as
  `expert_mixed_official.plan_record_segments` emits them.
  `validate_expert_manifest_spec` validates per-layer logical bytes and
  component order against the tier map, fail closed.
- **D6 — MVP serving lane.** Mixed mode forces `slot_layout=component-banks`
  and FORBIDS at open(): direct-slots, metal-mmap, miss_shadow, dense islands
  (same shape as the t158 lane's restrictions). Quality gate E1 does not need
  islands; perf lanes can lift restrictions later with their own receipts.
- **D7 — fail loud on missed uniformity consumers.**
  `spec.expert_record_bytes` RAISES for mixed-official specs. Every consumer
  the mixed lane actually reaches must be routed to per-layer sizes; lanes
  forbidden by D6 keep their uniform code untouched and unreachable.
- **D8 — registry.** New spec `HY3_EXPERT_MIXOFFICIAL`
  (`model_key="hy3-expert-mixofficial"`) via `dataclasses.replace` of
  `HY3_EXPERT_Q2`; joins `MODEL_SPECS` + the `expert_cli.py` choices list.
  `optimization_profiles` registration NOT required (get_profile no-ops).

## Hard laws
- M1 files are READ-ONLY: `mtplx/expert_mixed_official.py`,
  `scripts/convert_expert_mixed_official.py`, `mtplx/expert_shadow.py`
  encode/decode. If integration seems to require changing them, STOP and
  report — do not improvise.
- Tests live at the CONSUMER layer (campaign law). Mirror
  `tests/test_expert_q1_serve.py`: tiny bank built through the REAL M1
  converter, opened through the REAL runtime, spy-asserted dispatch, numeric
  parity vs a `decode_projection` reference, memory-plan arithmetic with two
  record sizes, and every D6/D5 rejection.
- CPU-only. No GPU, no Metal kernels executed (shadow_gather_mm spy/stub as
  the q1 serve test does), no touching `/tmp/mtplx-gpu-exclusive.lock`, no
  reads of the real bank or bf16 tree.
- Pytest form: `PYTHONPATH=<worktree> <venv python> -m pytest <files> > log
  2>&1; echo EXIT=$?` — never pipe through tail/grep. Targeted suites only
  (new tests + test files of every touched module); the full suite has known
  pre-existing noise.
