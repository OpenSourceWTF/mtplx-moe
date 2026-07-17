# Streamed-path codec-awareness gap analysis (q1 lane, issue #51)

2026-07-17. What must change before the streamed runtime can serve
q1 (shadow-codec) expert records — `{proj}.packed` + `{proj}.scales`
pairs — instead of affine `{weight, scales, biases}` triples. Line
references are against `feat/miss-shadow` after the q1 registry entries
landed.

**State today.** The converter lane exists (`mtplx/expert_q1.py`,
`scripts/convert_expert_q1.py`; bitwise-verified against fresh source
encodes) and the registry prices q1 artifacts (`glm52-expert-q1t`,
`glm52-expert-q1b1`; `ExpertStreamingModelSpec.expert_codec`). The probe
(`research/glm52-shadow-codec-probe-20260717.json`) gates the burn to
**t158** (combine-cosine 0.922; b1 collapses to 0.017 from Q2 sources —
the Q2 grid's exact-zero level carries ~50% of the weight mass and a
1-bit sign code cannot represent it). Execution of q1 records already
has a working reference implementation: the miss-shadow lane
(`_run_shadow_bank` + `shadow_gather_mm`) executes exactly this
`{packed, scales}` layout today, just from resident banks rather than
streamed slots.

The affine assumption is expressed as four independent invariants:
**9 components** (3 projections x 3 leaves), leaf names
`weight/scales/biases`, dtypes `U32`/`BF16`, and byte math
`params*bits/8 + groups*2*parameter_bytes`. Each gate below enforces at
least one of them.

## Gate 1 — manifest schema (hard-fails first)

`mtplx/expert_manifest.py`:

- `_PROJECTIONS`/`_LEAVES`/`_COMPONENTS` (L36-41): closed 9-name set.
  `TensorSegment.from_dict` L377-378 rejects `gate_proj.packed` as an
  "unsupported expert component" before dtype is even considered.
- `ExpertRecord.from_dict` L469-477: "must contain the nine ordered
  quantized components" — a q1 record has six.
- `validate_structure` L702: `quant_bits not in {2, 4} or quant_mode !=
  "affine"` — needs a `quant_mode` value for shadow codecs (proposal:
  `quant_mode == codec name`, `quant_bits` nominal 1/2).
- `_expected_component_shape` (L1135-1159): per-leaf geometry hardcodes
  weight→U32 `(out, in*bits/32)` and params→BF16 `(out, in/group)`.
  Needs a codec branch: packed→U32 `(out, groups*2)` (b1) / U8
  `(out, groups*13)` (t158), scales→U16 `(out, groups)`.
- `_classify_expert_name` (L1162-1186) + `_expert_segments`
  (L1196-1285): build-side discovery recognizes only the affine triple
  and guards `quant_parameter_bytes == 2` (L1200-1203). Not needed for
  the q1 lane if the q1 manifest is produced by the converter rather
  than `build_expert_manifest` — but `load_expert_manifest` still runs
  the schema gates above.
- Good news: `_DTYPE_BYTES` (L46-60) already whitelists `U8`/`U16`, so
  only names/count/geometry/mode need extension, not the dtype table.

Two viable routes: (a) extend `ExpertManifest` with codec-conditional
component sets (one schema, both codecs), or (b) teach the runtime to
open the self-describing `Q1Manifest` directly and bypass
`ExpertManifest` for q1 artifacts. (a) keeps one manifest surface for
integrity/verification tooling and is the recommended route; the q1
converter's manifest already mirrors the segment vocabulary to make
that unification mechanical.

## Gate 2 — spec byte math (CLOSED)

`mtplx/expert_streaming_models.py`: `expert_record_bytes` is now
codec-aware via `expert_codec`; all plan consumers (`plan_expert_memory`
L573-645, transient/island/prefetch/slot sizing) inherit it. The
whole-byte check (L158) and group-divisibility check (L153-156) hold for
b1/t158 (10 and 15 bytes per g64). `packed_weight_bytes`/
`scale_bias_bytes` remain affine-only helpers; nothing on the q1 path
reads them when `expert_codec != "affine"`.

## Gate 3 — slot pool and bank allocators

- `mtplx/expert_slots.py` L639-643: pool guard compares
  `manifest.quant_bits/quant_group_size` to the spec — passes once the
  manifest carries the nominal q1 values. `ExpertSlotBinding
  .component_view` (L258-272) is already codec-agnostic (segment-driven).
- `mtplx/models/expert_mlx.py` `_component_array` (L149-175): dtype
  branch handles only U32/BF16 → add U8/U16.
- `make_mlx_component_bank_allocator` (L883-1095): `expected_signature`
  (L937-976) hardcodes the affine triple per projection → needs the
  codec branch mirroring Gate 1's geometry. `MlxComponentBank`
  (L245-314) builds arrays straight from segment shape/dtype and only
  needs the dtype-map extension (L269-274).
- `make_mlx_slot_buffer_allocator` (L178-242) sizes raw byte slots from
  `spec.expert_record_bytes` — already correct via Gate 2.

## Gate 4 — execution

- Streamed component-bank path: `_run_component_bank_q4` →
  `_gather_component_bank` (L1152-1212) calls
  `mx.gather_qmm(..., mode="affine")` on the `.weight/.scales/.biases`
  triple. q1 needs a codec dispatch to the existing
  `shadow_gather_mm(..., codec=...)` on `.packed/.scales` — the shadow
  bank path (`_run_shadow_bank`, L1215-1241) is the template; the only
  new work is feeding it slot-bank rows (slot indices) instead of
  expert-id rows.
- Direct-slot path `_run_q4_expert` (L1105-1149) and mapped path
  `_run_mapped_q4`: same triple/`mode="affine"` assumption; q1 initial
  scope can require `slot_layout == "component-banks"` (like miss_shadow
  does) and reject the others loudly.
- Dense islands / mmap band: `DenseIslandSwitchGLU` (bits/group from
  spec), `hy3_expert_wave_m4._COMPONENT_LAYOUT` (9-component validation,
  `HY3_M4_BITS=2`), and `expert_banked._DTYPE_ITEM_SIZE` (L31, U32/BF16
  only) are all affine-pinned. Islands under q1 are optional at first —
  a q1 config can simply run island-free (the whole point of smaller
  records is more slots) — but the banked-mmap C6 band would need the
  dtype-map + layout extension to carry q1 layers.
- Kernel perf: `shadow_gather_mm` is a parity-grade one-thread-per-output
  kernel. Prefill (row counts >> 8) and sustained decode need a tuned
  variant (simdgroup reduction, vectorized unpack) before q1 can be the
  primary format; benchmark under the queued lane per the
  queued-vs-eager rule.

## Gate 5 — I/O (already codec-agnostic)

`mtplx/expert_io.py` `PositionalExpertReader` (`read_record_into`
L507-601, `read_component_records_into`, `_readv_range_into`) iterates
`record.segments` generically — no change beyond trusting the new
segment geometry. Record sha256 integrity carries over unchanged (the
q1 converter already hashes per record).

## Gate 6 — registry/CLI/config plumbing

- `expert_cli.py` L46-56: `--expert-model-key` choices list — add the q1
  keys when the runtime can open them (adding earlier gives a
  loud-but-late manifest error). `_read_model_key` (L133-147) never
  auto-infers expert-only keys; explicit flag remains the contract.
- `ExpertStreamingConfig`: no new field strictly needed (the codec rides
  the spec/manifest), but `miss_shadow` atop a q1t artifact would shadow
  t158 with t158 — validation should reject `miss_shadow` when
  `spec.expert_codec != "affine"`.
- Benchmark: model-key plumbing already generic; q1 rows land as new
  `--models` entries once open works.

## Ordered close-out (estimate: the runtime work is Gates 1+3+4)

1. Manifest: codec-conditional component schema + `quant_mode`
   extension (Gate 1a) and converter emitting the unified schema.
2. Allocators: dtype map + codec-aware `expected_signature`.
3. Execution: codec dispatch in the component-bank path to
   `shadow_gather_mm`; require component-banks layout initially.
4. Tuned q1 gather kernel (prefill-capable) + benchmark matrix rows.
5. Optional: banked-mmap band + island support for q1 layers;
   `miss_shadow`/q1 mutual exclusion.

Until 1-3 land, q1 artifacts are priced (registry), produced
(converter), and quality-gated (probe) but not servable.
