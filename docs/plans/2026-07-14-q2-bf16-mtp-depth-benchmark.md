# Q2 BF16 MTP Depth Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers-optimized:subagent-driven-development` to implement this plan
> task-by-task. Use `superpowers-optimized:test-driven-development` for every
> behavior change and `superpowers-optimized:verification-before-completion`
> before any completion claim. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Correctly attach the BF16 Hy3 layer-80 and GLM-5.2 layer-78 MTP
heads to their expert-only affine-Q2 trunks, then measure 512-input/128-output
qualification and repeated 1,024-input/128-output depth sweeps with ingestion
TPS, prefill attribution, decode TPS, exact AR equivalence, and per-depth MTP
acceptance.

**Architecture:** Keep both Q2 trunks and all default serving behavior
unchanged. Add only the missing GLM-Q2 descriptor, an exact-bit external GLM
MTP artifact, model-family-specific streamed MTP injection, and one canonical
`generate_mtpk` path. GLM owns request-scoped recurrent state: D1 computes
IndexShare top-k and advances committed MLA/indexer caches, while D2-D5 reuse
the D1 selection and write only discardable main-KV scratch bounded at D1.
The campaign loads each target/MTP pair once, runs correctness gates before
timing, uses fixed mirrored schedules, retains every memory high-water mark,
and collects resource telemetry only in the final diagnostic pair.

The GLM target continues to use the existing local `glm52_mlx` overlay that
has already run AR. Do not route GLM-5.2 through the generic
`deepseek_mtp_patch.py` fallback: that fallback constructs a stock decoder,
does not model GLM's DSA indexer, and loads non-strictly. Shared mlx-lm tensor
primitives used internally by the existing GLM overlay do not change the model
family or checkpoint semantics.

**Tech Stack:** Python 3.12, MLX 0.31.x, mlx-lm, NumPy, safetensors container
I/O, pytest, Ruff, JSON/JSONL, POSIX `pread`/`pwrite`/`fsync`, `fcntl.flock`,
macOS launchd, and the existing streamed-expert runtime.

**Assumptions:**

- Work stays in
  `/Users/davidtai/projects/OpenSourceWTF/.worktrees/q2-bf16-mtp-bench` on
  `codex/q2-bf16-mtp-bench`, starting at design commit `2656cf9`.
- Do not merge or wholesale cherry-pick `codex/glm52-expert-q2`. Its tree
  predates the completed Hy3-Q2 work. Port only additive GLM descriptor/CLI
  hunks from `1112f78` and `413e533`; current commits `f18c14d` and `d80d37b`
  already contain the shared affine-Q2 manifest and bit-aware QMM support.
- The completed external trunks are
  `/Users/davidtai/.cache/huggingface/hy3-expert-only-mlx-q2` and
  `/Users/davidtai/.cache/huggingface/glm52-expert-only-mlx-q2`. Both remain
  experimental because their independent quality gates regressed by roughly
  22 percent; this work does not promote either trunk.
- The existing Hy3 BF16 MTP artifact is
  `/Users/davidtai/.cache/huggingface/hy3-mtp-layer80/layer80-bf16.safetensors`.
- The staged GLM source is
  `/Users/davidtai/.cache/huggingface/glm52-mtp-layer78-source`, pinned to
  `zai-org/GLM-5.2@b4734de4facf877f85769a911abafc5283eab3d9`.
- The GLM output is external at
  `/Users/davidtai/.cache/huggingface/glm52-mtp-layer78`; no model bytes,
  extraction workspaces, or raw benchmark payloads enter Git.
- Raw campaign payloads live under ignored
  `benchmarks/raw/q2-bf16-mtp-depth/<run-id>/`; only a reviewed Markdown report
  belongs under `benchmarks/results/`.
- Qwen is managed by
  `$HOME/Library/LaunchAgents/com.tea.qwen.plist`, serves
  `mtplx-qwen36-27b-optimized-speed` at
  `http://127.0.0.1:8080/v1/models`, and must be restored to its exact initial
  state after every exclusive MLX window.
- The feature remains explicit and off by default. A correct negative
  performance result is a complete result.

---

## Dependency Order

| Stage | Depends on | Unlocks |
| --- | --- | --- |
| GLM-Q2 registration | approved design | artifact/runtime CLI selection |
| GLM artifact builder | pinned source invariants | strict BF16 loader |
| GLM layer/cache adapter | local GLM DSA geometry | streamed GLM MTP |
| generation accounting | runtime cycle hook | exact depth metrics |
| exclusive MLX guard | existing Qwen guard | safe real-model runs |
| campaign/summarizer | all schemas above | qualification and standard sweep |
| real campaign | clean code, artifact, all gates | curated benchmark report |

Generation accounting and the exclusive MLX guard may be developed in
parallel. Real MLX work begins only after their tests and the artifact/runtime
tests pass.

## File Structure

### Create

- `mtplx/glm52_mtp_artifact.py`: bounded exact-bit layer-78 extraction and
  authenticated artifact verification.
- `scripts/extract_glm52_mtp_layer78.py`: `preflight`, `extract`, and `verify`
  commands for the external GLM artifact.
- `tests/test_glm52_mtp_artifact.py`: synthetic extraction, provenance,
  publication, and tamper tests.
- `mtplx/glm52_mtp_patch.py`: strict BF16 loader, recurrent cache, and streamed
  GLM MTP injector.
- `tests/test_glm52_mtp.py`: module construction, router, IndexShare/KVShare,
  rollback, and snapshot tests.
- `tests/test_glm52_streamed_mtp.py`: streamed runtime and generation
  integration tests.
- `mtplx/benchmarks/q2_mtp_depth_campaign.py`: fixed schedules, schema
  validation, paired statistics, and candidate selection.
- `scripts/summarize_q2_mtp_depth_campaign.py`: pure-data campaign verifier and
  summary CLI.
- `tests/test_q2_mtp_depth_campaign.py`: schedule, gate, pairing, noise-floor,
  and summary tests.
- `benchmarks/results/q2-bf16-mtp-depth-2026-07-14.md`: reviewed final results;
  create only after both valid campaigns finish.

### Modify

- `mtplx/expert_streaming_models.py`
- `mtplx/expert_cli.py`
- `scripts/verify_streamed_parity.py`
- `tests/test_expert_streaming_models.py`
- `tests/test_expert_cli_runtime.py`
- `tests/test_verify_streamed_parity_cli.py`
- `mtplx/models/glm52_mlx.py`
- `tests/test_streamed_models.py`
- `mtplx/runtime.py`
- `mtplx/generation.py`
- `tests/test_generation_sustained.py`
- `tests/test_hy3_streamed_mtp.py`
- `mtplx/qwen_guard.py`
- `scripts/run_with_qwen_stopped.py`
- `scripts/run_issue30_starvation_attribution.py`
- `tests/test_qwen_guard.py`
- `tests/test_issue30_starvation_campaign.py`
- `scripts/benchmark_streamed_generation.py`
- `tests/test_benchmark_streamed_generation_cli.py`

---

### Task 1: Register the GLM-5.2 Expert-Q2 Target Additively

**Files:**

- Modify: `mtplx/expert_streaming_models.py`
- Modify: `mtplx/expert_cli.py`
- Modify: `scripts/benchmark_streamed_generation.py`
- Modify: `scripts/verify_streamed_parity.py`
- Test: `tests/test_expert_streaming_models.py`
- Test: `tests/test_expert_cli_runtime.py`
- Test: `tests/test_benchmark_streamed_generation_cli.py`
- Test: `tests/test_verify_streamed_parity_cli.py`

**Security flag:** `none`

**Does NOT cover:** MTP loading, GLM artifact extraction, trunk conversion, or
changes to any existing Hy3/GLM production descriptor.

- [ ] **Step 1: Write failing additive registry tests**

Add `test_glm52_expert_q2_exact_expert_and_indexshare_layout()` and extend
registry/CLI parameterizations without removing any Hy3 key:

```python
spec = get_model_spec("glm52-expert-q2")
assert spec.total_layers == 78
assert spec.routed_layer_indices == tuple(range(3, 78))
assert spec.expert_count == 256
assert spec.top_k == 8
assert spec.quant_bits == 2
assert spec.quant_group_size == 64
assert spec.expert_record_bytes == 11_796_480
assert spec.routed_expert_bytes == 226_492_416_000
assert spec.resident_bytes == 10_634_546_688
assert spec.total_tensor_bytes == 237_126_962_688
assert spec.router_storage == "bfloat16 with fp32 correction bias"
assert spec.router_matmul_dtype == "float32"
assert spec.mtp_layer_index == 78
assert spec.mtp_included is False
assert spec.full_indexer_layers == (0, 1, 2, *range(6, 75, 4))
```

Snapshot the existing `HY3_Q4`, `HY3_EXPERT_ONLY_Q4`, `HY3_EXPERT_Q2`, and
`GLM52_Q4` dataclass values and assert registry expansion does not change them.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest -q \
  tests/test_expert_streaming_models.py \
  tests/test_expert_cli_runtime.py \
  tests/test_verify_streamed_parity_cli.py \
  tests/test_benchmark_streamed_generation_cli.py \
  -k 'glm52_expert_q2 or model_registry or model_key'
```

Expected: FAIL because `glm52-expert-q2` is absent from the current registry
and CLI choices.

- [ ] **Step 3: Add the exact descriptor and additive CLI keys**

Define `GLM52_EXPERT_Q2` with the tested values, source revision
`b4734de4facf877f85769a911abafc5283eab3d9`, and quant source
`mlx-community/GLM-5.2-4bit@6b347a6472d46bf55de65ee34032136a3929d778`.
Register it alongside all current descriptors. Add the key to expert CLI,
benchmark CLI, parity CLI, and `_MODEL_DEFAULTS` using `_GLM52_AR_DEFAULTS`.
Keep model-type inference unchanged: `glm_moe_dsa` still infers `glm52-q4`
unless the caller explicitly selects the Q2 key.

- [ ] **Step 4: Run GREEN and commit**

```bash
.venv/bin/python -m pytest -q \
  tests/test_expert_streaming_models.py \
  tests/test_expert_cli_runtime.py \
  tests/test_verify_streamed_parity_cli.py \
  tests/test_benchmark_streamed_generation_cli.py
git diff --check
git add \
  mtplx/expert_streaming_models.py mtplx/expert_cli.py \
  scripts/benchmark_streamed_generation.py scripts/verify_streamed_parity.py \
  tests/test_expert_streaming_models.py tests/test_expert_cli_runtime.py \
  tests/test_benchmark_streamed_generation_cli.py \
  tests/test_verify_streamed_parity_cli.py
git commit -m "feat(streaming): register GLM-5.2 expert Q2"
```

Expected: all selected tests pass and every pre-existing descriptor remains
byte-for-byte equivalent as a dataclass value.

### Task 2: Build a Strict Exact-Bit GLM Layer-78 Artifact

**Files:**

- Create: `mtplx/glm52_mtp_artifact.py`
- Create: `scripts/extract_glm52_mtp_layer78.py`
- Create: `tests/test_glm52_mtp_artifact.py`

**Security flag:** `security`

**Does NOT cover:** MLX allocation, model injection, network download, or
modification of the staged source files.

- [ ] **Step 1: Write failing exact-inventory and adversarial I/O tests**

Define the 23 fixed tensor expectations exactly; generate the remaining 768
names as 256 experts times `gate_proj`, `up_proj`, and `down_proj`:

```python
FIXED_LAYER78 = {
    "eh_proj.weight": ("BF16", (6144, 12288)),
    "enorm.weight": ("BF16", (6144,)),
    "hnorm.weight": ("BF16", (6144,)),
    "input_layernorm.weight": ("BF16", (6144,)),
    "mlp.gate.e_score_correction_bias": ("F32", (256,)),
    "mlp.gate.weight": ("BF16", (256, 6144)),
    "mlp.shared_experts.down_proj.weight": ("BF16", (6144, 2048)),
    "mlp.shared_experts.gate_proj.weight": ("BF16", (2048, 6144)),
    "mlp.shared_experts.up_proj.weight": ("BF16", (2048, 6144)),
    "post_attention_layernorm.weight": ("BF16", (6144,)),
    "self_attn.indexer.k_norm.bias": ("BF16", (128,)),
    "self_attn.indexer.k_norm.weight": ("BF16", (128,)),
    "self_attn.indexer.weights_proj.weight": ("BF16", (32, 6144)),
    "self_attn.indexer.wk.weight": ("BF16", (128, 6144)),
    "self_attn.indexer.wq_b.weight": ("BF16", (4096, 2048)),
    "self_attn.kv_a_layernorm.weight": ("BF16", (512,)),
    "self_attn.kv_a_proj_with_mqa.weight": ("BF16", (576, 6144)),
    "self_attn.kv_b_proj.weight": ("BF16", (28672, 512)),
    "self_attn.o_proj.weight": ("BF16", (6144, 16384)),
    "self_attn.q_a_layernorm.weight": ("BF16", (2048,)),
    "self_attn.q_a_proj.weight": ("BF16", (2048, 6144)),
    "self_attn.q_b_proj.weight": ("BF16", (16384, 2048)),
    "shared_head.norm.weight": ("BF16", (6144,)),
}
```

Tests must assert:

- exactly 791 names and 19,905,841,664 payload bytes;
- exactly 790 BF16 tensors and one F32 correction bias;
- source shard tensor distribution `4,213,213,213,148`;
- missing, extra, duplicate, wrong-dtype, wrong-shape, overlapping range,
  trailing data, short read, and malformed JSON rejection;
- config/index/shard digest mismatch and source replacement rejection;
- exact raw-byte copying and per-tensor/output digest verification;
- authenticated manifest key exactness and tamper rejection;
- failure leaves no final output directory;
- an existing or racing destination is never overwritten;
- symlink, hardlink, non-regular file, unsafe ownership/mode, and dirty producer
  rejection.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest -q tests/test_glm52_mtp_artifact.py
```

Expected: FAIL because the artifact module and command do not exist.

- [ ] **Step 3: Implement bounded preflight, extraction, and verification**

Expose `Glm52MtpArtifactConfig(source_root: Path, output_root: Path,
producer_root: Path)` as a frozen dataclass and these public functions with the
stated return types:

- `expected_glm52_layer78_inventory(config: Mapping[str, object]) ->
  dict[str, TensorExpectation]`;
- `preflight_glm52_mtp_layer78(config: Glm52MtpArtifactConfig) ->
  ExtractionPlan`;
- `extract_glm52_mtp_layer78(config: Glm52MtpArtifactConfig) -> Path`;
- `verify_glm52_mtp_layer78(root: Path, *, deep: bool = True) ->
  dict[str, object]`.

The implementation must use strict duplicate-key JSON parsing, bounded header
sizes/counts, no-follow held descriptors, exact dtype/shape byte arithmetic,
chunked `pread`/`pwrite`, per-tensor SHA-256, source identity rechecks, file and
directory `fsync`, deep verification of the sibling staging directory, and an
exclusive atomic rename. It must never decode BF16 through NumPy.

Pin these source facts in code and verify them before copying:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 3,732 | `185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a` |
| `model.safetensors.index.json` | 5,408,032 | `5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e` |
| `model-00270-of-00282.safetensors` | 5,366,430,968 | `d74106256f061e73000e9660d157bd22254d2a5692cf9466d76dfea6985c0924` |
| `model-00271-of-00282.safetensors` | 5,360,347,304 | `90ba74c758309888b9d3f17adc189e32e77cee77ca8f7892ad12a9b38956cd43` |
| `model-00272-of-00282.safetensors` | 5,360,347,320 | `d5c9dbfba6aff2be069079cf39c8991393a2b69469a4f0ac4e246af25d519a06` |
| `model-00273-of-00282.safetensors` | 5,360,347,208 | `1344c75f27e5564baa46641ebcdf19a2a13ac1b2cfc52e01744b6dda1127aa94` |
| `model-00274-of-00282.safetensors` | 5,359,997,688 | `1943b335a5aa626389e819fa5a7339c361844c425389b35328eba0142935fbf6` |

Publish exactly `layer78-bf16.safetensors` and
`mtp-artifact-manifest.json`. The manifest schema is
`mtplx-glm52-mtp-layer78-v1` and binds source repo/revision, all source
digests, all tensor names/shapes/dtypes/ranges/digests, output header/file
digests, exact byte counts, producer commit, clean-tree status, and a canonical
manifest digest excluding only `manifest_sha256`.

- [ ] **Step 4: Add the three command modes**

```bash
.venv/bin/python scripts/extract_glm52_mtp_layer78.py preflight --help
.venv/bin/python scripts/extract_glm52_mtp_layer78.py extract --help
.venv/bin/python scripts/extract_glm52_mtp_layer78.py verify --help
```

Each command accepts explicit `--source-root`, `--output-root`, and
`--producer-root` where applicable. `verify` accepts `--deep` and fails closed
by default if the authenticated manifest is absent.

- [ ] **Step 5: Run GREEN, static checks, and commit**

```bash
.venv/bin/python -m pytest -q tests/test_glm52_mtp_artifact.py
.venv/bin/ruff check \
  mtplx/glm52_mtp_artifact.py scripts/extract_glm52_mtp_layer78.py \
  tests/test_glm52_mtp_artifact.py
.venv/bin/ruff format --check \
  mtplx/glm52_mtp_artifact.py scripts/extract_glm52_mtp_layer78.py \
  tests/test_glm52_mtp_artifact.py
git diff --check
git add mtplx/glm52_mtp_artifact.py scripts/extract_glm52_mtp_layer78.py \
  tests/test_glm52_mtp_artifact.py
git commit -m "feat(mtp): package exact GLM-5.2 layer 78"
```

Expected: all synthetic corruption/publication cases pass without creating
external artifacts during tests.

### Task 3: Extend the Existing GLM-5.2 Overlay for Its Layer-78 Head

**Files:**

- Modify: `mtplx/models/glm52_mlx.py`
- Modify: `tests/test_streamed_models.py`

**Security flag:** `none`

**Does NOT cover:** Artifact parsing, generation cache commit policy, or target
model behavior when the new keyword arguments are omitted. It does not replace
the already-working GLM target or route GLM through the generic DeepSeek MTP
fallback.

- [ ] **Step 1: Prove the existing generic fallback is not the exact GLM path**

Write `test_glm52_layer78_requires_every_dsa_indexer_weight()` against a
synthetic layer-78 inventory containing
`self_attn.indexer.{k_norm,weights_proj,wk,wq_b}`. Pass it through the current
generic `glm_moe_dsa` MTP construction and require strict consumption of every
leaf. The current path must fail because it constructs a stock decoder and
uses non-strict loading.

If the current adapter unexpectedly consumes the full exact inventory with the
local GLM DSA layer and strict loading, stop and reuse it; delete the custom
construction work below. Do not maintain two exact GLM-5.2 adapters.

- [ ] **Step 2: Write failing target-parity and MTP-geometry tests**

Add tests proving:

- default target calls still compute/share top-k exactly as before;
- layer 78 can be built with `indexer_type="full"` despite the target's
  78-entry `indexer_types` list;
- resident mode creates 256 resident BF16 experts and no
  `UnboundExpertSwitch(78)`;
- resident and streamed gates select identical experts/scores on fixed hidden
  states and near-tie logits, with FP32 routing arithmetic;
- `compute_topk=False` never calls or advances the indexer;
- `kv_read_boundary=C1` slices MLA, KPE, sparse indices/mask, and the dense
  `topk=None` path so D2+ cannot read beyond D1.

- [ ] **Step 3: Run RED**

```bash
.venv/bin/python -m pytest -q tests/test_streamed_models.py \
  -k 'glm52 and (resident or recurrent or indexshare or read_boundary or router)'
```

Expected: FAIL because layer construction is fixed to streamed experts and
attention has no recurrent controls.

- [ ] **Step 4: Add opt-in GLM controls while preserving target defaults**

Extend `GlmMoeDsaDecoderLayer.__init__` with keyword-only
`expert_mode: Literal["streamed", "resident"] = "streamed"` and
`indexer_type: Literal["full", "shared"] | None = None` arguments.
The production body must construct a local `GlmMoeDsaResidentMoE` using the
same stacked expert tensor primitive already used by mlx-lm's GLM DSA model,
then replace its gate with the existing GLM `FP32MoEGate`. Do not instantiate
`DeepseekV32DecoderLayer` or call `inject_deepseek_mtp_support`.

Extend attention and decoder calls with keyword-only `compute_topk` and
`kv_read_boundary`. When `compute_topk` is `None`, retain current target
behavior. When false, reuse `prev_topk_indices`, do not touch indexer KV, and
cap every attention read/mask at the supplied boundary before sparse gather or
dense attention.

Add `Glm52MTPLayer` and `Glm52MTP`. The layer performs `enorm`, `hnorm`,
concatenation, `eh_proj`, local DSA decoder block, shared-head norm, and the
call-time target LM head in checkpoint order. Pass target embedding and target
LM head into `__call__`; do not register duplicate modules.

- [ ] **Step 5: Run GREEN and commit**

```bash
.venv/bin/python -m pytest -q tests/test_streamed_models.py
.venv/bin/ruff check mtplx/models/glm52_mlx.py tests/test_streamed_models.py
git diff --check
git add mtplx/models/glm52_mlx.py tests/test_streamed_models.py
git commit -m "feat(mtp): add GLM recurrent layer geometry"
```

Expected: all GLM target tests remain green and the new resident/recurrent
tests pass.

### Task 4: Implement the Strict GLM BF16 Loader and Cache Lifecycle

**Files:**

- Create: `mtplx/glm52_mtp_patch.py`
- Create: `tests/test_glm52_mtp.py`

**Security flag:** `security`

**Does NOT cover:** Generic DeepSeek MTP behavior, trunk quantization, or
global/model-owned recurrent state.

- [ ] **Step 1: Write failing strict-load and cache-state tests**

Cover:

- authenticated artifact/revision/digest validation happens before MLX tensor
  allocation;
- exact 791 raw tensors map to exactly 27 module leaves;
- missing, extra, dtype, shape, or mapped-leaf mismatch fails strict loading;
- MTP experts remain BF16 and the F32 correction bias remains F32;
- target embedding and LM-head objects are shared by identity at call time;
- D1 advances main MLA and indexer-key offsets from `O` to `O+1`, computes
  top-k once, and records `C1=O+1`;
- D2-D5 advance only main scratch, keep indexer-key at `C1`, reuse top-k, and
  cannot read beyond `C1`;
- `finish_cycle()` trims main scratch to `C1`, leaves indexer at `C1`, and
  clears top-k;
- `rollback_to(target)` trims each child independently rather than applying
  one trim count to divergent offsets;
- committed append advances both children by the aligned target-grounded row
  count and clears cycle state;
- snapshots contain only committed MLA/indexer state and exclude top-k and
  D2+ scratch;
- stop, exception, rejection, and adaptive exit all leave no cycle state.

Also assert the new adapter consumes every GLM-specific indexer tensor that
the old generic fallback left unmatched. This is the gate that justifies the
new file; merely producing tokens is insufficient.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest -q tests/test_glm52_mtp.py
```

Expected: FAIL because `mtplx.glm52_mtp_patch` does not exist.

- [ ] **Step 3: Implement a request/cache-scoped recurrent cache**

Use this public surface:

```python
@dataclass(frozen=True)
class GLM52MTPTensorSpec:
    dtype: str
    shape: tuple[int, ...]

@dataclass
class GLM52MTPCycleState:
    base_offset: int
    d1_boundary: int | None = None
    topk_indices: Any | None = None
    topk_computed: bool = False

class GLM52MTPCache:
    main_kv: KVCache
    indexer_kv: KVCache
    cycle: GLM52MTPCycleState | None

    @property
    def offset(self) -> int:
        return int(self.main_kv.offset)

    def rollback_to(self, target: int) -> None:
        for child in (self.main_kv, self.indexer_kv):
            excess = max(0, int(child.offset) - int(target))
            if excess:
                child.trim(excess)
        self.cycle = None

    def trim(self, count: int) -> int:
        before = self.offset
        self.rollback_to(max(0, before - int(count)))
        return before - self.offset
```

Also implement `__len__`/`__getitem__` for attention access, committed-only
`state`/`meta_state` properties, `begin_cycle`, `finish_d1`, `recurrent_view`,
and `finish_cycle`. `make_mtp_cache()` returns exactly one flat entry:
`[GLM52MTPCache(KVCache(), KVCache())]`, so generic generation observes a real
logical `.offset`.

- [ ] **Step 4: Implement strict artifact mapping and injection**

Expose `expected_glm52_mtp_inventory(args) -> dict[str,
GLM52MTPTensorSpec]`, `load_glm52_mtp_bf16_weights(artifact_dir, args, *,
expected_revision=GLM52_SOURCE_REVISION) -> dict[str, Any]`,
`build_glm52_mtp_module(artifact_dir, args, *,
expected_revision=GLM52_SOURCE_REVISION)`, and
`inject_glm52_streamed_mtp_support(model, artifact_dir, config, contract=None,
*, expected_revision=GLM52_SOURCE_REVISION) -> bool`.

Map head-local leaves to `layers.0.*`, decoder leaves to
`layers.0.mtp_block.*`, and `shared_head.norm` to `shared_head_norm`. Stack
each expert projection into one 256-row BF16 tensor. Split
`self_attn.kv_b_proj.weight` into `embed_q.weight` and
`unembed_out.weight` using the target overlay's exact transformation. Call
`mtp.load_weights(list(mapped.items()), strict=True)` on exactly 27 mapped
leaves. Never call the
generic trunk-driven MTP quantizer and never construct a second shared output
head.

`mtp_forward` begins D1 state on recurrent depth one, records its top-k/C1,
and uses the recurrent view for later depths. `mtp_update_cache` is a separate
committed-append path that updates both child caches from target-grounded
hidden/token pairs and invalidates cycle state.

- [ ] **Step 5: Run GREEN and commit**

```bash
.venv/bin/python -m pytest -q tests/test_glm52_mtp.py
.venv/bin/ruff check mtplx/glm52_mtp_patch.py tests/test_glm52_mtp.py
.venv/bin/ruff format --check mtplx/glm52_mtp_patch.py tests/test_glm52_mtp.py
git diff --check
git add mtplx/glm52_mtp_patch.py tests/test_glm52_mtp.py
git commit -m "feat(mtp): add strict GLM streamed adapter"
```

Expected: strict loading and every recurrent lifecycle test pass.

### Task 5: Dispatch Streamed MTP by Model Family

**Files:**

- Modify: `mtplx/runtime.py`
- Modify: `tests/test_hy3_streamed_mtp.py`
- Create: `tests/test_glm52_streamed_mtp.py`

**Security flag:** `none`

**Does NOT cover:** Enabling MTP by default, GLM Q4 MTP, or accepting a
quantized MTP head for either expert-Q2 benchmark lane.

- [ ] **Step 1: Write failing dispatch and precision tests**

Assert this exact support matrix before model allocation:

| Model key | Injector | Accepted MTP precision |
| --- | --- | --- |
| `hy3-q4` | Hy3 | `bf16`, existing diagnostic `q4` |
| `hy3-expert-q2` | Hy3 | `bf16` only |
| `glm52-expert-q2` | GLM-5.2 | `bf16` only |

Add tests that unsupported keys, missing artifacts, quantized MTP for a Q2
lane, and invalid provenance fail before resident target allocation. Prove an
MTP-enabled AR call still invokes only target forward and that a no-MTP load is
unchanged.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest -q \
  tests/test_hy3_streamed_mtp.py tests/test_glm52_streamed_mtp.py \
  -k 'dispatch or precision or expert_q2 or ar_preservation'
```

Expected: FAIL because runtime currently accepts streamed MTP only for
`hy3-q4` and always imports the Hy3 injector.

- [ ] **Step 3: Add family-aware dispatch and a cycle-finalization hook**

Replace the single Hy3 check with an explicit mapping. Use the descriptor's
source revision when calling the strict injector and make artifact errors
model-family neutral. Set GLM `mtp_verify_width=6` and leave the Hy3 width
unchanged.

Add to `MTPLXRuntime`:

```python
def finish_mtp_cycle(self, mtp_cache) -> None:
    finish = getattr(self.model, "finish_mtp_cycle", None)
    if callable(finish):
        finish(mtp_cache)
```

The GLM injected model delegates to every `GLM52MTPCache.finish_cycle()`;
other backends remain no-op compatible.

- [ ] **Step 4: Run GREEN and commit**

```bash
.venv/bin/python -m pytest -q \
  tests/test_hy3_streamed_mtp.py tests/test_glm52_streamed_mtp.py \
  tests/test_mtp_patch.py tests/test_session_bank.py
.venv/bin/ruff check \
  mtplx/runtime.py tests/test_hy3_streamed_mtp.py \
  tests/test_glm52_streamed_mtp.py
git diff --check
git add mtplx/runtime.py tests/test_hy3_streamed_mtp.py \
  tests/test_glm52_streamed_mtp.py
git commit -m "feat(mtp): dispatch streamed Q2 draft heads"
```

Expected: both Q2 families load only their strict BF16 adapter, and existing
MTP/session tests remain green.

### Task 6: Make `generate_mtpk` Correct and Measurable at Every Depth

**Files:**

- Modify: `mtplx/generation.py`
- Modify: `tests/test_generation_sustained.py`

**Security flag:** `none`

**Does NOT cover:** Adaptive production policy, sampling changes, or use of
the legacy `generate_mtp1` path in the campaign.

- [ ] **Step 1: Write the accepted-bonus/rejection regression first**

Add `test_generate_mtpk_pending_bonus_then_rejection_matches_ar()` using a
stateful tiny runtime where cycle one accepts a full block and emits a bonus,
then cycle two accepts one draft and rejects the next. Assert:

```python
assert mtpk.tokens == ar.tokens
assert mtpk.finish_reason == ar.finish_reason == "length"
assert mtpk.stats.generated_tokens == ar.stats.generated_tokens
assert runtime.target_cache_offset == prompt_tokens + generated_tokens
assert runtime.mtp_committed_offset == prompt_tokens + generated_tokens - 1
```

Run both generators with final-state capture enabled so the final emitted
token is committed before comparing cache offsets.

If the RED result reproduces divergence, repair only the pending-primary and
target/MTP commit seam before adding metrics.

- [ ] **Step 2: Add failing evaluated-depth and history-accounting tests**

For a D3 cycle that rejects at D2, require:

```python
assert stats.drafted_by_depth == [1, 1, 1]
assert stats.evaluated_by_depth == [1, 1, 0]
assert stats.accepted_by_depth == [1, 0, 0]
assert stats.evaluated_drafts == 2
assert stats.fully_accepted_verify_calls == 0
assert stats.mean_accept_probability_by_depth == [1.0, 0.0, None]
```

Extend committed-history tests to capture every `(hidden, token)` append and
assert a cold prompt `p[0:N]` uses `(h[i], p[i+1])` for `i=0..N-2`, reports
`prompt_mtp_history_tokens == N-1`, reaches cache offset `N-1`, records
position base zero, and calculates history TPS with `N-1` as the numerator.

- [ ] **Step 3: Run RED**

```bash
.venv/bin/python -m pytest -q tests/test_generation_sustained.py \
  -k 'pending_bonus or evaluated_by_depth or committed_mtp_history or cycle_cleanup'
```

Expected: new accounting fields/lifecycle assertions fail; token divergence,
if present, is visible before performance work.

- [ ] **Step 4: Add raw counters and correct denominators**

Extend `GenerationStats`, `PromptState`, and `_DecodeTrace` with:

```python
evaluated_drafts: int = 0
evaluated_by_depth: list[int] = field(default_factory=list)
fully_accepted_verify_calls: int = 0
prompt_mtp_history_tokens: int = 0
```

Increment `evaluated_by_depth[d]` immediately before the acceptance decision
for depth `d`; drafts after the first rejection remain drafted but unevaluated.
Divide acceptance-probability sums by evaluated counts. Propagate totals and
deltas through trace rows. Count a fully accepted verification call only when
every drafted proposal in that call was evaluated and accepted.

Return the actual history-row count from committed prompt prefill and use that
count for `prompt_mtp_history_tok_s`. Preserve `new_prefill_tokens /
prompt_eval_time_s` as the ingestion-TPS definition.

- [ ] **Step 5: Guarantee cycle cleanup around recurrent drafting**

Wrap the recurrent draft loop, not target verification, in:

```python
try:
    run_recurrent_draft_steps()
finally:
    rt.finish_mtp_cycle(step_mtp_cache)
```

Use the existing loop body in place of `run_recurrent_draft_steps()`; do not
introduce a second generation implementation. Tests must cover rejection,
stop, exception, and adaptive early exit. GLM cleanup retains D1 and discards
D2+ before target verification; Hy3 remains a no-op.

- [ ] **Step 6: Run GREEN, regression tests, and commit**

```bash
.venv/bin/python -m pytest -q \
  tests/test_generation_sustained.py tests/test_penalties.py \
  tests/test_loop_guard.py tests/test_graphbank_compiled_verify.py \
  tests/test_session_bank.py
.venv/bin/ruff check mtplx/generation.py tests/test_generation_sustained.py
git diff --check
git add mtplx/generation.py tests/test_generation_sustained.py
git commit -m "fix(mtp): account and clean recurrent verification"
```

Expected: canonical `generate_mtpk` is token-exact through the retained failure
shape, and conditional/cumulative depth denominators are distinguishable.

### Task 7: Compose the Shared Hardware Lock with Exact Qwen Restoration

**Files:**

- Modify: `mtplx/qwen_guard.py`
- Modify: `scripts/run_with_qwen_stopped.py`
- Modify: `scripts/run_issue30_starvation_attribution.py`
- Modify: `tests/test_qwen_guard.py`
- Modify: `tests/test_issue30_starvation_campaign.py`

**Security flag:** `security`

**Does NOT cover:** Stopping arbitrary MLX processes, killing unrelated
daemons, or deleting an ambiguous pre-existing lock.

- [ ] **Step 1: Write failing lock-order and restoration tests**

Test that:

- an exclusive `flock` is held before the first Qwen state mutation;
- a second holder waits and then acquires, or times out without mutation;
- success, exception, child timeout, `SIGINT`, `SIGTERM`, and
  `KeyboardInterrupt` restore Qwen before unlock;
- an initially unloaded Qwen remains unloaded;
- symlink, foreign-owner, multi-link, non-regular, and group/world-writable
  lock files fail closed;
- full argv fixtures for `runningboardd`, `lsd`, and `containermanagerd` never
  count as benchmark competitors;
- the child wrapper and issue-30 runner call the same context manager.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest -q \
  tests/test_qwen_guard.py tests/test_issue30_starvation_campaign.py \
  -k 'exclusive or lock or restore or daemon or signal'
```

Expected: FAIL because Qwen stop/restore and the old directory/process lane are
not one atomic context.

- [ ] **Step 3: Implement one composed context**

Add `exclusive_mlx_window(*, plist: Path, api_url: str,
timeout_seconds: float, lock_path: Path =
Path("/tmp/mtplx-gpu-exclusive.lock"), lock_timeout_seconds: float | None =
None) -> Iterator[MlxWindowReceipt]` as a context manager. Its production body
must:

1. open/create a user-owned no-follow mode-0600 single-link regular lock file;
2. acquire `LOCK_EX` while respecting the monotonic deadline;
3. capture the exact launchctl/API/process state;
4. stop only `com.tea.qwen` after lock acquisition;
5. yield a receipt recording lock and Qwen transitions;
6. restore the captured state inside termination-signal shielding;
7. verify restoration, record it, then release/close the lock.

Replace the issue-30 mkdir plus broad `pgrep -f` authority with this context.
If process diagnostics remain, parse complete argv and allow only exact MTPLX
benchmark module/script names.

- [ ] **Step 4: Route both wrappers through the composed guard**

Add `--lock-path` and `--lock-timeout-seconds` to
`scripts/run_with_qwen_stopped.py`. Make its default guard
`exclusive_mlx_window`. Refactor `scripts/run_issue30_starvation_attribution.py`
to use the same context and receipt instead of its private acquire/capture/
stop/restore sequence.

- [ ] **Step 5: Run GREEN and commit**

```bash
.venv/bin/python -m pytest -q \
  tests/test_qwen_guard.py tests/test_issue30_starvation_campaign.py
.venv/bin/ruff check \
  mtplx/qwen_guard.py scripts/run_with_qwen_stopped.py \
  scripts/run_issue30_starvation_attribution.py tests/test_qwen_guard.py \
  tests/test_issue30_starvation_campaign.py
git diff --check
git add mtplx/qwen_guard.py scripts/run_with_qwen_stopped.py \
  scripts/run_issue30_starvation_attribution.py tests/test_qwen_guard.py \
  tests/test_issue30_starvation_campaign.py
git commit -m "fix(bench): serialize exclusive MLX windows"
```

Expected: all restoration and contention paths pass, with no daemon name used
as lock authority.

### Task 8: Implement the Pure Campaign Contract and Statistics

**Files:**

- Create: `mtplx/benchmarks/q2_mtp_depth_campaign.py`
- Create: `scripts/summarize_q2_mtp_depth_campaign.py`
- Create: `tests/test_q2_mtp_depth_campaign.py`

**Security flag:** `none`

**Does NOT cover:** Model loading, Qwen control, resource sampling, or writing
raw campaign rows.

- [ ] **Step 1: Write failing fixed-schedule and pairing tests**

Pin these constants:

```python
QUALIFICATION = {
    "hy3-expert-q2": ("AR", "D1", "D2", "D3", "D4"),
    "glm52-expert-q2": ("AR", "D1", "D2", "D3", "D4", "D5"),
}
STANDARD = {
    "hy3-expert-q2": (
        ("AR", "D1", "D2", "D3", "D4"),
        ("D4", "D3", "D2", "D1", "AR"),
        ("D2", "D4", "AR", "D3", "D1"),
    ),
    "glm52-expert-q2": (
        ("AR", "D1", "D2", "D3", "D4", "D5"),
        ("D5", "D4", "D3", "D2", "D1", "AR"),
        ("D2", "D4", "D1", "AR", "D5", "D3"),
    ),
}
```

Tests must reject wrong row counts/order/replicate/position, cross-replicate AR
pairing, warmups entering headline aggregates, inconsistent prompt/artifact/
configuration identities, non-committed history, requested/effective depth
mismatch, counter algebra errors, non-128 output, token divergence, non-length
finish, inherited memory peaks, cap breaches, headline telemetry, absent
diagnostic telemetry, and incomplete Qwen restoration.

- [ ] **Step 2: Write failing statistic and selection tests**

For every depth, assert paired decode and end-to-end deltas, median,
minimum/maximum/range, median absolute deviation, and positive-pair count.
Define the AR noise floor as:

```python
ar_median = statistics.median(ar_values)
noise_floor = max(abs(value - ar_median) for value in ar_values) / ar_median
```

A candidate needs decode and end-to-end median gains above their respective
noise floors and at least two positive pairs. Select highest decode gain;
within one percentage point, use end-to-end gain and then shallower depth. If
all depths lose, select the least-negative diagnostic depth and set
`none_won=True`. Never emit a production-promotion result.

- [ ] **Step 3: Run RED**

```bash
.venv/bin/python -m pytest -q tests/test_q2_mtp_depth_campaign.py
```

Expected: FAIL because the pure campaign module does not exist.

- [ ] **Step 4: Implement strict schema validation and summarization**

Expose `summarize_q2_mtp_depth_campaign(payload: Mapping[str, Any]) ->
dict[str, Any]`.

Use schema `mtplx-q2-bf16-mtp-depth-benchmark-v1`. Derive and report:

- ingestion TPS = `new_prefill_tokens / prompt_eval_time_s`;
- target-only prefill TPS and total/target/MTP-history seconds;
- decode TPS = completion tokens / decode elapsed seconds;
- end-to-end completion TPS separately;
- conditional hit = accepted/evaluated;
- cumulative prefix yield = accepted/drafted;
- accepted drafts per verify call;
- whole-block acceptance = fully accepted calls / verify calls;
- first-rejection depth distribution and mean acceptance probability.

All derived values must be recomputed from raw counts/times and compared to
any pre-rendered field with a tight floating tolerance.

- [ ] **Step 5: Add the summary CLI, run GREEN, and commit**

```bash
.venv/bin/python -m pytest -q tests/test_q2_mtp_depth_campaign.py
.venv/bin/ruff check \
  mtplx/benchmarks/q2_mtp_depth_campaign.py \
  scripts/summarize_q2_mtp_depth_campaign.py \
  tests/test_q2_mtp_depth_campaign.py
.venv/bin/ruff format --check \
  mtplx/benchmarks/q2_mtp_depth_campaign.py \
  scripts/summarize_q2_mtp_depth_campaign.py \
  tests/test_q2_mtp_depth_campaign.py
git diff --check
git add mtplx/benchmarks/q2_mtp_depth_campaign.py \
  scripts/summarize_q2_mtp_depth_campaign.py \
  tests/test_q2_mtp_depth_campaign.py
git commit -m "feat(bench): validate Q2 MTP depth campaigns"
```

Expected: pure synthetic campaigns prove every schedule, gate, and selection
rule without MLX.

### Task 9: Add the Model-Aware Depth Campaign to the Streamed Harness

**Files:**

- Modify: `scripts/benchmark_streamed_generation.py`
- Modify: `tests/test_benchmark_streamed_generation_cli.py`

**Security flag:** `security`

**Does NOT cover:** Changing the continuous-batch AR lane, production sampling,
or mixing diagnostic resource telemetry into headline timings.

- [ ] **Step 1: Write failing parser and pre-allocation validation tests**

Add:

- `--mtp-draft-length` for a paired AR/MTP diagnostic at one exact depth;
- `--mtp-depth-sweep` for the fixed full campaign;
- `--mtp-history-policy` with sweep requiring `committed`;
- `--qwen-plist`, `--qwen-api-url`, `--mlx-lock-path`,
  `--mlx-lock-timeout-seconds`;
- model-aware depth ranges Hy3 `1..4`, GLM `1..5`;
- family-aware MTP artifact identity.

Tests must prove every Q2 MTP depth, including D1, dispatches
`generate_mtpk`, while legacy non-Q2 `generate_mtp1` remains untouched.
Reject conflicting sweep overrides rather than silently correcting them.
Record the option tokens present in the original argv and reject explicit
`--context-tokens`, `--max-tokens`, `--repeats`, window-telemetry, sampling,
or resource-telemetry overrides in sweep mode; ordinary non-sweep defaults
remain unchanged.

- [ ] **Step 2: Write failing prompt, reset, peak, and row-schema tests**

With fake tokenizer/runtime/MLX objects, assert:

- prompt fixtures have exact SHA-256 values
  `4825f939cd4c7228f1c2cd164515cc4541f71d4f3a6a6a180b31587991a582a4`
  (Hy3) and
  `14530bcb61a1c7c0b56ec41104ed968977ae9f92c73bb10efc8884170f83df90`
  (GLM);
- `_prompt_build_for_context` returns immutable exact 512/1,024 ID sequences
  using coding-agent/raw/thinking-off and preserves the fixture tail;
- target KV, committed MTP history, cycle state, expert cache, diagnostics, and
  row results reset between observations while compiled programs remain;
- setup, smoke, qualification, warmup, standard, and diagnostic peaks all feed
  a retained `hard_peak_bytes` even after row-local resets;
- an over-cap stage aborts immediately;
- unavailable peak reset is labeled process-wide and invalidates row-level
  memory comparison;
- rows include schedule, replicate, one-based position, pair ID, requested and
  effective depth, token IDs, finish reason, timings/counts, memory, and
  exploratory Hy3 D3/D4 labels.

- [ ] **Step 3: Run RED**

```bash
.venv/bin/python -m pytest -q tests/test_benchmark_streamed_generation_cli.py \
  -k 'mtp_depth or q2_campaign or prompt_identity or peak or qwen or schedule'
```

Expected: FAIL because the existing harness supports only a single legacy Hy3
MTP1 reference row.

- [ ] **Step 4: Implement model-aware artifact identity and flag validation**

Refactor `build_mtp_artifact_identity` to require keyword-only `model_key` and
`precision` beside `artifact_root`, returning `dict[str, Any]`. For Hy3,
retain existing file/digest behavior and require the pinned BF16
revision for the expert-Q2 lane. For GLM, call
`verify_glm52_mtp_layer78(artifact_root, deep=True)` and report its
authenticated manifest/file identity. Validate all sweep flags and the
complete memory plan before target allocation.

- [ ] **Step 5: Implement one-load campaign orchestration**

Add helpers `_campaign_cells`, `_run_q2_mtp_observation`,
`_reset_campaign_row_state`, and `_run_q2_mtp_depth_campaign`. The campaign
must enter `exclusive_mlx_window` before applying the Metal cap/loading the
runtime and leave it only after the runtime closes.

Enforce this exact runtime configuration:

```python
EXPECTED_Q2_MTP_RUNTIME = {
    "memory_limit_bytes": 112 * 1024**3,
    "runtime_reserve_bytes": 12 * 1024**3,
    "expert_cache_limit_bytes": 64 * 1024**3,
    "max_live_kv_tokens": 4096,
    "cache_policy": "frequency",
    "cache_scope": "layer",
    "slot_layout": "component-banks",
    "transient_slots": 8,
    "max_read_chunk_bytes": 8 * 1024**2,
    "bypass_page_cache": True,
}
```

Load target plus BF16 MTP once per model. AR calls `generate_ar` on that same
runtime without executing MTP. MTP calls `generate_mtpk` with fixed depth,
empty stop set, `max_tokens=128`, temperature 0, top-p 1, top-k 1, seed 0,
persistent cache, committed history, no SessionBank, no prefix restore,
repetition stop false, and loop guard false.

Run these retained stages in order:

1. setup/load peak;
2. 512-input/1-output smoke for AR and every depth;
3. 512-input/32-output AR equivalence for every depth;
4. 512-input/128-output qualification in ascending cell order;
5. discarded 1,024-input/8-output warmup in ascending cell order;
6. three 1,024-input/128-output standard schedules from Task 8;
7. pure summary to select diagnostic depth;
8. `AR,best,best,AR` diagnostic with resource telemetry.

Abort before the next stage if any prior exactness, provenance, memory,
history, count, or Qwen gate fails. Qualification and standard headline rows
must have resource and rolling-window telemetry disabled. Diagnostic rows must
capture expert bytes/read operations/cache hits/misses/evictions, SSD
throughput, reader occupancy, queue depth, completion fences, and peak memory.

- [ ] **Step 6: Add stage-local peak tracking**

Implement `_MlxPeakTracker` around `mx.reset_peak_memory`, `mx.synchronize`,
`mx.get_active_memory`, `mx.get_cache_memory`, and `mx.get_peak_memory`.
Reset before setup and every row, record active before/after/delta and local
peak, and update a software maximum that can never decrease. Apply the 112 GiB
limit before any model allocation and abort immediately if any stage or
`hard_peak_bytes` exceeds it.

- [ ] **Step 7: Emit the dedicated payload and validate it before publish**

The payload must include commit/harness/model/trunk/MTP/tokenizer identities,
both prompt identities, sampler, complete memory plan, machine/OS/Python/MLX,
applied Metal limit, lock/Qwen transitions, start/end provenance timestamps,
all schedules, every stage peak, smoke/qualification/warmup/standard/
diagnostic rows, and gate results. Timestamps are never used as performance
metrics.

Call `summarize_q2_mtp_depth_campaign` before atomically committing the JSON
evidence target. A validation error must leave no final payload.

- [ ] **Step 8: Run GREEN, relevant regressions, and commit**

```bash
.venv/bin/python -m pytest -q \
  tests/test_benchmark_streamed_generation_cli.py \
  tests/test_q2_mtp_depth_campaign.py \
  tests/test_qwen_guard.py \
  tests/test_generation_sustained.py
.venv/bin/ruff check \
  scripts/benchmark_streamed_generation.py \
  tests/test_benchmark_streamed_generation_cli.py
git diff --check
git add scripts/benchmark_streamed_generation.py \
  tests/test_benchmark_streamed_generation_cli.py
git commit -m "feat(bench): run Q2 BF16 MTP depth sweeps"
```

Expected: fake-runtime tests cover every campaign row and gate without loading
the real models.

### Task 10: Extract and Deep-Verify the Real GLM Artifact

**Files:**

- External source:
  `/Users/davidtai/.cache/huggingface/glm52-mtp-layer78-source`
- External output:
  `/Users/davidtai/.cache/huggingface/glm52-mtp-layer78`
- Git files: none

**Security flag:** `security`

**Does NOT cover:** MLX model loading or benchmark timing. Qwen stays in its
initial state because exact-bit extraction is CPU/storage-only.

- [ ] **Step 1: Require a clean producer commit and absent destination**

```bash
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test ! -e /Users/davidtai/.cache/huggingface/glm52-mtp-layer78
git rev-parse --verify HEAD
```

Expected: clean branch and no final output path. If a prior failed staging
directory exists, inspect it and remove it only after confirming it is owned by
this extractor and is not the final artifact.

- [ ] **Step 2: Run pinned preflight**

```bash
PYTHONNOUSERSITE=1 .venv/bin/python \
  scripts/extract_glm52_mtp_layer78.py preflight \
  --source-root /Users/davidtai/.cache/huggingface/glm52-mtp-layer78-source \
  --output-root /Users/davidtai/.cache/huggingface/glm52-mtp-layer78 \
  --producer-root "$PWD"
```

Expected: 791 tensors, 19,905,841,664 payload bytes, 790 BF16 plus one F32,
five exact source shards, and all pinned digests pass.

- [ ] **Step 3: Extract, publish, and deep-verify**

```bash
PYTHONNOUSERSITE=1 .venv/bin/python \
  scripts/extract_glm52_mtp_layer78.py extract \
  --source-root /Users/davidtai/.cache/huggingface/glm52-mtp-layer78-source \
  --output-root /Users/davidtai/.cache/huggingface/glm52-mtp-layer78 \
  --producer-root "$PWD"
PYTHONNOUSERSITE=1 .venv/bin/python \
  scripts/extract_glm52_mtp_layer78.py verify \
  --output-root /Users/davidtai/.cache/huggingface/glm52-mtp-layer78 \
  --deep
find /Users/davidtai/.cache/huggingface/glm52-mtp-layer78 \
  -maxdepth 1 -type f -print | sort
```

Expected: only `layer78-bf16.safetensors` and
`mtp-artifact-manifest.json`; deep verification reports exact inventory,
payload bytes, source revision, per-tensor digests, and output digest.

- [ ] **Step 4: Confirm Git remains clean**

```bash
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

Expected: external artifact creation changes no repository file.

### Task 11: Run Hy3 and GLM Qualification, Standard, and Diagnostic Campaigns

**Files:**

- Write ignored raw evidence under
  `benchmarks/raw/q2-bf16-mtp-depth/<run-id>/`
- Git files: none

**Security flag:** `security`

**Does NOT cover:** Production rollout, quality-gate reconsideration, or manual
weakening of a failed campaign gate.

- [ ] **Step 1: Run the complete pre-model verification suite**

```bash
PYTHONNOUSERSITE=1 .venv/bin/python -m pytest -q \
  tests/test_glm52_mtp_artifact.py \
  tests/test_streamed_models.py \
  tests/test_glm52_mtp.py \
  tests/test_hy3_streamed_mtp.py \
  tests/test_glm52_streamed_mtp.py \
  tests/test_generation_sustained.py \
  tests/test_qwen_guard.py \
  tests/test_benchmark_streamed_generation_cli.py \
  tests/test_q2_mtp_depth_campaign.py
```

Expected: all tests pass. Do not enter an MLX window on failure.

- [ ] **Step 2: Create one exact ignored run directory**

```bash
RUN_ID="q2-bf16-mtp-depth-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short=10 HEAD)"
RAW_ROOT="benchmarks/raw/q2-bf16-mtp-depth/$RUN_ID"
mkdir -p "$RAW_ROOT"
git check-ignore -q "$RAW_ROOT/hy3.json"
git check-ignore -q "$RAW_ROOT/glm52.json"
printf '%s\n' "$RAW_ROOT" > /tmp/mtplx-q2-mtp-current-run
```

Expected: both evidence paths are ignored and the exact run root is retained
for the following commands.

- [ ] **Step 3: Run the complete Hy3 sweep inside one exclusive window**

```bash
RAW_ROOT="$(cat /tmp/mtplx-q2-mtp-current-run)"
PYTHONNOUSERSITE=1 .venv/bin/python \
  scripts/benchmark_streamed_generation.py \
  /Users/davidtai/.cache/huggingface/hy3-expert-only-mlx-q2 \
  /Users/davidtai/.cache/huggingface/hy3-expert-only-mlx-q2/expert-manifest.json \
  --model-key hy3-expert-q2 \
  --memory-limit 112GiB \
  --runtime-reserve 12GiB \
  --expert-cache-limit 64GiB \
  --max-live-kv-tokens 4096 \
  --cache-policy frequency \
  --cache-scope layer \
  --slot-layout component-banks \
  --transient-slots 8 \
  --read-chunk 8MiB \
  --f-nocache \
  --enable-mtp \
  --mtp-artifacts /Users/davidtai/.cache/huggingface/hy3-mtp-layer80 \
  --mtp-precision bf16 \
  --mtp-depth-sweep \
  --mtp-history-policy committed \
  --qwen-plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --qwen-api-url http://127.0.0.1:8080/v1/models \
  --mlx-lock-path /tmp/mtplx-gpu-exclusive.lock \
  --mlx-lock-timeout-seconds 1800 \
  --output-json "$RAW_ROOT/hy3.json" \
  --run-label hy3_q2_bf16_mtp_depth
```

Expected: smoke and qualification exactness pass at D1-D4; standard produces
three rows per cell; diagnostic runs `AR,best,best,AR`; every row emits exactly
128 tokens except the declared smoke/warmup stages; Qwen is restored before
the command exits.

- [ ] **Step 4: Verify Hy3 payload before starting GLM**

```bash
RAW_ROOT="$(cat /tmp/mtplx-q2-mtp-current-run)"
.venv/bin/python scripts/summarize_q2_mtp_depth_campaign.py \
  "$RAW_ROOT/hy3.json" --output "$RAW_ROOT/hy3-summary.json"
curl -fsS http://127.0.0.1:8080/v1/models | \
  jq -e '.data[] | select(.id == "mtplx-qwen36-27b-optimized-speed")'
```

Expected: strict summary succeeds and Qwen serves the exact expected model. If
Qwen was initially unloaded, replace the curl assertion with the payload's
verified `initially_unloaded_and_remained_unloaded` receipt.

- [ ] **Step 5: Run the complete GLM sweep inside one exclusive window**

```bash
RAW_ROOT="$(cat /tmp/mtplx-q2-mtp-current-run)"
PYTHONNOUSERSITE=1 .venv/bin/python \
  scripts/benchmark_streamed_generation.py \
  /Users/davidtai/.cache/huggingface/glm52-expert-only-mlx-q2 \
  /Users/davidtai/.cache/huggingface/glm52-expert-only-mlx-q2/expert-manifest.json \
  --model-key glm52-expert-q2 \
  --memory-limit 112GiB \
  --runtime-reserve 12GiB \
  --expert-cache-limit 64GiB \
  --max-live-kv-tokens 4096 \
  --cache-policy frequency \
  --cache-scope layer \
  --slot-layout component-banks \
  --transient-slots 8 \
  --read-chunk 8MiB \
  --f-nocache \
  --enable-mtp \
  --mtp-artifacts /Users/davidtai/.cache/huggingface/glm52-mtp-layer78 \
  --mtp-precision bf16 \
  --mtp-depth-sweep \
  --mtp-history-policy committed \
  --qwen-plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --qwen-api-url http://127.0.0.1:8080/v1/models \
  --mlx-lock-path /tmp/mtplx-gpu-exclusive.lock \
  --mlx-lock-timeout-seconds 1800 \
  --output-json "$RAW_ROOT/glm52.json" \
  --run-label glm52_q2_bf16_mtp_depth
```

Expected: smoke and qualification exactness pass at D1-D5; standard produces
three rows per cell; diagnostic runs `AR,best,best,AR`; all GLM IndexShare,
KVShare, memory, and restoration gates pass.

- [ ] **Step 6: Strictly summarize GLM and recheck Qwen**

```bash
RAW_ROOT="$(cat /tmp/mtplx-q2-mtp-current-run)"
.venv/bin/python scripts/summarize_q2_mtp_depth_campaign.py \
  "$RAW_ROOT/glm52.json" --output "$RAW_ROOT/glm52-summary.json"
curl -fsS http://127.0.0.1:8080/v1/models | \
  jq -e '.data[] | select(.id == "mtplx-qwen36-27b-optimized-speed")'
```

Expected: strict summary succeeds and Qwen restoration is exact. On any failed
gate, retain the invalid raw payload, debug the cause, and rerun the affected
model under a new run ID; never splice rows from incompatible commits or
artifact identities.

### Task 12: Curate Results and Run Final Verification

**Files:**

- Create: `benchmarks/results/q2-bf16-mtp-depth-2026-07-14.md`
- Test: all changed code

**Security flag:** `none`

**Does NOT cover:** Creating a PR, changing the default branch, enabling MTP in
serving, or claiming that three benchmark replicates establish production
readiness.

- [ ] **Step 1: Write the reviewed comparison report**

Read the two strict summaries and record, for each model and depth:

- 512 qualification exactness and memory;
- 1,024 standard ingestion TPS, total prefill seconds, target prefill seconds,
  MTP-history seconds, decode TPS, and end-to-end completion TPS;
- drafted/evaluated/accepted counts, conditional hit, cumulative prefix yield,
  accepted drafts per verify, whole-block acceptance, and first rejection;
- all three paired decode/end-to-end deltas, median, range, MAD, positive-pair
  count, and AR noise floor;
- selected diagnostic depth and whether `none_won` is true;
- diagnostic storage/cache/queue/fence evidence and retained hard peak;
- exact trunk/MTP/commit/prompt identities and the Qwen restoration receipt;
- the explicit conclusion that neither Q2 trunk nor MTP mode is promoted by
  this campaign.

Do not quote a lone best repeat. Keep qualification, standard, and diagnostic
numbers in separate tables.

- [ ] **Step 2: Run the complete targeted suite and repository static checks**

```bash
PYTHONNOUSERSITE=1 .venv/bin/python -m pytest -q \
  tests/test_expert_streaming_models.py \
  tests/test_expert_manifest.py \
  tests/test_expert_cli_runtime.py \
  tests/test_verify_streamed_parity_cli.py \
  tests/test_glm52_mtp_artifact.py \
  tests/test_streamed_models.py \
  tests/test_glm52_mtp.py \
  tests/test_hy3_streamed_mtp.py \
  tests/test_glm52_streamed_mtp.py \
  tests/test_generation_sustained.py \
  tests/test_session_bank.py \
  tests/test_qwen_guard.py \
  tests/test_issue30_starvation_campaign.py \
  tests/test_benchmark_streamed_generation_cli.py \
  tests/test_q2_mtp_depth_campaign.py
.venv/bin/ruff check \
  mtplx/glm52_mtp_artifact.py mtplx/glm52_mtp_patch.py \
  mtplx/models/glm52_mlx.py mtplx/runtime.py mtplx/generation.py \
  mtplx/qwen_guard.py mtplx/benchmarks/q2_mtp_depth_campaign.py \
  scripts/extract_glm52_mtp_layer78.py \
  scripts/benchmark_streamed_generation.py \
  scripts/summarize_q2_mtp_depth_campaign.py \
  scripts/run_with_qwen_stopped.py \
  scripts/run_issue30_starvation_attribution.py \
  tests/test_glm52_mtp_artifact.py tests/test_glm52_mtp.py \
  tests/test_glm52_streamed_mtp.py tests/test_generation_sustained.py \
  tests/test_qwen_guard.py tests/test_issue30_starvation_campaign.py \
  tests/test_benchmark_streamed_generation_cli.py \
  tests/test_q2_mtp_depth_campaign.py
.venv/bin/python -m compileall -q mtplx scripts tests
git diff --check
```

Expected: all targeted tests and static checks pass with fresh output.

- [ ] **Step 3: Verify scope, raw-artifact hygiene, and default behavior**

```bash
git status --short --branch
git check-ignore -q "$(cat /tmp/mtplx-q2-mtp-current-run)/hy3.json"
git check-ignore -q "$(cat /tmp/mtplx-q2-mtp-current-run)/glm52.json"
git diff --name-only HEAD | sort
rg -n 'default=.*False|default=False' scripts/benchmark_streamed_generation.py \
  | rg 'mtp|depth'
test ! -e /tmp/mtplx-gpu-exclusive.lock || \
  test -O /tmp/mtplx-gpu-exclusive.lock
```

Expected: only the curated report is uncommitted, raw payloads remain ignored,
MTP/sweep defaults remain off, and no active lock holder remains. The persistent
0600 lock inode may exist after unlock; its existence is not ownership.

- [ ] **Step 4: Commit the curated evidence**

```bash
git add benchmarks/results/q2-bf16-mtp-depth-2026-07-14.md
git diff --cached --check
git diff --cached --stat
git commit -m "docs(bench): report Q2 BF16 MTP depth results"
git status --short --branch
```

Expected: report commit succeeds and the worktree is clean. Record the fresh
test counts, exact raw run ID, and both strict-summary validity results in the
handoff; do not add raw JSON or external artifact files.

---

## Completion Gates

The work is complete only when all of the following are true:

1. GLM artifact verification proves the exact 791 tensors and
   19,905,841,664 payload bytes from the pinned revision.
2. Hy3 D1-D4 and GLM D1-D5 all use `generate_mtpk` with committed history.
3. Every retained MTP row emits exactly the paired deterministic AR token IDs,
   128-token length, and length finish reason.
4. `evaluated_by_depth` separates conditional hit from cumulative prefix
   yield, and every reported metric is recomputable from raw fields.
5. GLM D2-D5 never advance indexer-key state or attend beyond D1's boundary;
   all scratch/top-k state is cleared on every exit.
6. Every setup/smoke/qualification/warmup/standard/diagnostic peak and retained
   `hard_peak_bytes` stays below the applied 112 GiB limit.
7. Headline rows have telemetry disabled; diagnostic rows have the required
   resource counters.
8. Qwen is restored to its exact pre-window state before the shared lock is
   released for both model campaigns.
9. The fixed 512 and 1,024 schedules, pairing, noise-floor rule, and diagnostic
   selection pass the strict pure-data validator.
10. Raw evidence stays ignored, the curated report states negative results
    plainly, and no serving default or Q2 quality status changes.
