# Kimi K3 Q2_K to t158 Streaming Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers-optimized:subagent-driven-development (recommended) or
> superpowers-optimized:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the pinned 96-part Kimi K3 Q2_K GGUF into a resumable,
resident-preserving MTPLX t158 routed-expert artifact, run its exact K3 text
topology through MTPLX streaming, and collect basic no-MTP benchmarks.

**Architecture:** A local GGUF reader validates all source headers and exposes
contiguous tensor slices. Each routed expert projection is decoded from Q2_K,
encoded with the existing t158 codec, and written into a per-layer sidecar
part. BF16/F32 tensors bypass numerical conversion and are copied into
safetensors payloads. A K3-specific MLX text model preserves latent MoE, SITU,
q-LoRA MLA, safe-gated KDA, and attention-residual semantics. Construction
validates the artifact once and installs a prebound SITU t158 streamed switch;
the enabled hot path contains no invariant checks or silent stock fallback.

**Tech Stack:** Python 3.12, NumPy, MLX/MLX-LM, MTPLX expert
manifest/t158 codecs, safetensors framing, pytest, Hugging Face Hub/Xet.

**Assumptions:**

- Assumes GrEarl revision
  `0169245d3ea1473a3f9f03bca821d855df5fb2a3` retains the measured 96-shard
  inventory — conversion will refuse a changed revision or geometry.
- Assumes routed tensors remain contiguous by expert in GGUF — conversion will
  NOT accept interleaved or split expert slices.
- Assumes resident tensors are only BF16/F32 — conversion will NOT silently
  dequantize another resident type.
- Assumes “streaming like GLM” means K3 text generation. Vision and multimodal
  residents are copied byte-for-byte but excluded from the text runtime's
  construction-time resident allowlist.
- Assumes K3 runs autoregressively with MTP disabled because the pinned
  checkpoint declares zero next-token-prediction layers.
- Repository changes remain uncommitted for user review because the user asked
  for local artifact work, not a Git commit or publication.

---

## File structure

- Create `mtplx/kimi_k3_gguf.py`: strict GGUF v3 header parser, tensor sizing,
  Q2_K decoder, and pinned K3 inventory validation.
- Create `mtplx/kimi_k3_t158.py`: expert-slice conversion, resumable layer
  parts, resident safetensors copy, manifest/index/receipt assembly.
- Create `scripts/convert_kimi_k3_q2k_t158.py`: command-line orchestration only.
- Create `tests/test_kimi_k3_gguf.py`: parser and independent Q2_K golden tests.
- Create `tests/test_kimi_k3_t158.py`: record layout, resume, resident-copy,
  manifest, and construction-boundary tests.
- Create `mtplx/models/kimi_k3_mlx.py`: exact K3 text model, cache, sanitize, and
  construction-time tensor mapping.
- Create `tests/test_kimi_k3_model.py`: fixed-weight component parity, tiny-model
  load, cache, and streamed SITU parity.
- Modify `mtplx/expert_streaming_models.py`: exact K3 latent expert geometry,
  SITU activation route, and text-resident totals.
- Modify `mtplx/models/expert_mlx.py`: construction-time activation binding and
  direct SITU t158 execution.
- Modify `mtplx/resident_loader.py`: install the K3 overlay for `kimi_linear`.
- Modify `mtplx/runtime.py`: load K3's pinned local tiktoken implementation
  without a nonexistent `tokenizer.json`, install EOS 163586, and translate
  MTPLX chat controls to K3's `thinking`/`thinking_effort` arguments.
- Modify `mtplx/expert_cli.py`: accept the manifest's K3 model key.
- Modify `scripts/benchmark_streamed_generation.py`: accept the same streaming
  config as the service and identify multi-part sidecars without dereferencing
  the single-part-only `.file` property.
- Do not add a promoted K3 memory/optimization profile or public default-model
  alias until real measurements and a published package identity exist.
- Modify `scripts/verify_expert_manifest.py` only if its reporting cannot expose
  multi-part verification totals; no manifest semantics change is planned.

### Task 1: Strict local GGUF inventory and Q2_K decode

**Files:**

- Create: `mtplx/kimi_k3_gguf.py`
- Create: `tests/test_kimi_k3_gguf.py`

**Security flag:** `security` — parses untrusted local binary lengths, offsets,
shapes, and names.

**Does NOT cover:** Network fetching, resident serialization, t158 encoding, or
runtime installation.

- [ ] **Step 1: Write a failing Q2_K golden test**

Create a deterministic 84-byte block with known scale/min nibbles, 2-bit
lanes, FP16 `d`, and FP16 `dmin`. Assert:

```python
decoded = dequantize_q2_k(block, value_count=256)
assert decoded.shape == (256,)
np.testing.assert_array_equal(decoded, independent_scalar_decode(block))
```

The scalar reference in the test must decode each of the sixteen groups and
each 2-bit lane without calling production code.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -q tests/test_kimi_k3_gguf.py
```

Expected: collection fails because `mtplx.kimi_k3_gguf` does not exist.

- [ ] **Step 3: Implement the Q2_K decoder**

Define:

```python
Q2_K_VALUES = 256
Q2_K_BYTES = 84

def dequantize_q2_k(blob: bytes | memoryview, *, value_count: int) -> np.ndarray:
    ...
```

Require `value_count % 256 == 0` and exact byte length. Decode in bounded block
batches to FP32 using `value = d * low_nibble * q - dmin * high_nibble`.

- [ ] **Step 4: Add GGUF parser failure tests**

Generate tiny GGUF v3 files in tests and assert rejection of bad magic,
unsupported version, oversized counts, duplicate tensor names, out-of-file
tensor spans, non-power-of-two alignment, and unknown tensor types.

- [ ] **Step 5: Implement strict header parsing and inventory**

Define immutable `GGUFTensor` and `GGUFFile` records plus:

```python
def read_gguf(path: Path) -> GGUFFile: ...
def tensor_nbytes(tensor: GGUFTensor) -> int: ...
def inspect_kimi_k3_source(root: Path, revision: str) -> KimiK3Inventory: ...
```

The K3 validator requires shards 1..96, split numbers 0..95, exactly 276 Q2_K
tensors, layers 1..92, three expected expert names per layer, and only BF16/F32
resident tensors.

- [ ] **Step 6: Verify GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

### Task 2: One-expert t158 record and one-layer resumable part

**Files:**

- Create: `mtplx/kimi_k3_t158.py`
- Create: `tests/test_kimi_k3_t158.py`

**Security flag:** `security` — validates source/output separation and adopts
existing partial files only after identity and hash checks.

**Does NOT cover:** Resident safetensors, final model index, or network
downloads.

- [ ] **Step 1: Write a failing record-layout test**

Use a small injected layout whose three input axes divide by 64. Encode known
gate/up/down FP32 matrices and assert six ordered segments:

```python
assert [segment.component for segment in segments] == [
    "gate_proj.packed", "gate_proj.scales",
    "up_proj.packed", "up_proj.scales",
    "down_proj.packed", "down_proj.scales",
]
np.testing.assert_array_equal(
    decode_t158(gate_packed, gate_scales, gate.shape[1]), expected_gate_t158
)
```

- [ ] **Step 2: Run the targeted test and verify RED**

Run:

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -q tests/test_kimi_k3_t158.py -k record
```

Expected: import or symbol failure for `encode_expert_record`.

- [ ] **Step 3: Implement bounded record encoding**

Define:

```python
def encode_expert_record(
    projections: Mapping[str, np.ndarray],
    *,
    layer: int,
    expert: int,
    shard: str,
    record_offset: int,
) -> EncodedExpertRecord:
    ...
```

Call the existing `encode_t158` separately for gate, up, and down so peak
working memory never holds three decoded FP32 projections plus three encoder
workspaces.

- [ ] **Step 4: Write failing resume-integrity tests**

Cover: clean new part; completed journal adoption; torn tail truncation; wrong
source revision; changed tensor identity; altered completed record; and
source/output path equality.

- [ ] **Step 5: Implement per-layer conversion and journal**

Define:

```python
def convert_layer(
    source: GGUFFile,
    output_dir: Path,
    inventory: KimiK3Inventory,
    *,
    layer: int,
    resume: bool,
) -> ConvertedLayer:
    ...
```

Write `experts-t158-layer-NNN-of-092.bin.partial`, append one fsynced JSONL
journal record only after each complete expert record is written, validate
the exact 6,936,330,240-byte K3 part, then atomically rename it.

- [ ] **Step 6: Verify GREEN**

Run all `tests/test_kimi_k3_t158.py` tests selected by
`-k 'record or resume or layer'`. Expected: pass.

### Task 3: Byte-preserving resident safetensors

**Files:**

- Modify: `mtplx/kimi_k3_t158.py`
- Modify: `tests/test_kimi_k3_t158.py`

**Security flag:** `security` — writes binary tensor offsets and validates every
span against the source file.

**Does NOT cover:** Q2_K routed tensors, which must never enter resident files.

- [ ] **Step 1: Write a failing resident-copy test**

Create a tiny GGUF containing one F32 resident, one BF16 resident, and one Q2_K
expert tensor. After conversion, parse the safetensors header and assert that
only the two residents exist, shapes equal reversed GGUF dimensions, and their
payload bytes are byte-identical to the GGUF spans.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -q tests/test_kimi_k3_t158.py -k resident
```

Expected: missing `copy_resident_safetensors`.

- [ ] **Step 3: Implement raw safetensors framing**

Define:

```python
def copy_resident_safetensors(
    source: GGUFFile,
    output_path: Path,
) -> tuple[ResidentTensor, ...]:
    ...
```

Build a padded JSON header, stream-copy each BF16/F32 source span in chunks,
hash while copying, fsync, verify offsets and final size, and atomically rename.
Reject Q2_K from the resident selection rather than decoding it.

- [ ] **Step 4: Add interruption and overwrite tests**

Assert a valid completed resident shard is adopted only when its receipt hash
matches; a `.partial` is rebuilt; and an unrelated existing final file is
refused rather than overwritten.

- [ ] **Step 5: Verify GREEN**

Run `tests/test_kimi_k3_t158.py -k resident`. Expected: pass.

### Task 4: K3 streaming spec and construction-time SITU route

**Files:**

- Modify: `mtplx/expert_streaming_models.py`
- Modify: `mtplx/models/expert_mlx.py`
- Modify: `tests/test_kimi_k3_t158.py`
- Modify: `tests/test_streamed_models.py`

**Security flag:** `none`

**Does NOT cover:** The K3 transformer model itself. This task installs the
expert-side arithmetic that model will call.

- [ ] **Step 1: Write failing spec and SITU tests**

Assert `get_model_spec("kimi-k3-q1t")` has layers 1..92, 896 experts, top 16,
latent expert width 3584, intermediate width 3072, t158 record size 7,741,440,
codec `t158`, expert activation `situ`, text resident bytes
113,509,540,864, and text runtime tensor bytes 751,651,922,944. Compare the
streamed component-bank output against an independent dense decode using:

```python
4 * tanh(gate / 4) * sigmoid(gate) * 25 * tanh(up / 25)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -q tests/test_kimi_k3_t158.py tests/test_streamed_models.py \
  -k 'kimi or situ'
```

Expected: unknown model key or missing field.

- [ ] **Step 3: Add the K3 spec and construction-time activation route**

Add to `ExpertStreamingModelSpec`:

```python
expert_activation: str = "swiglu"
model_hidden_size: int | None = None
fixed_cache_bytes_per_batch: int = 0
```

Add the pinned K3 spec with `hidden_size=3584`,
`expert_hidden_size=3072`, `expert_codec="t158"`, and the measured routed
totals. Set `model_hidden_size=7168`, `kv_bytes_per_token=27_648` for the 24
latent MLA caches, and `fixed_cache_bytes_per_batch=449_372_160` for 69 FP32
KDA recurrent states plus their three BF16 convolution states. Price both
terms explicitly in memory planning; do not hide KDA state in a generic
reserve or promoted profile. Resolve `expert_activation` once while constructing
`HotExpertSwitchGLU`; install a prebound callable for either existing SwiGLU
or K3 SITU. The t158 component-bank route calls that callable directly.
Construction rejects layouts whose installed execution route cannot preserve
the selected activation. Do not branch on activation metadata inside a
per-token/per-expert loop and do not add a fallback to SwiGLU.

- [ ] **Step 4: Verify GREEN and regressions**

Run the Task 4 command plus:

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -q tests/test_expert_shadow.py tests/test_expert_sidecar_parts.py
```

Expected: pass, with existing optional fixture skips only.

### Task 5: Artifact assembly CLI and construction verification

**Files:**

- Modify: `mtplx/kimi_k3_t158.py`
- Create: `scripts/convert_kimi_k3_q2k_t158.py`
- Modify: `tests/test_kimi_k3_t158.py`
- Optionally modify: `scripts/verify_expert_manifest.py`

**Security flag:** `security` — CLI accepts source/output paths and emits an
authoritative model manifest.

**Does NOT cover:** Uploading, publishing, deleting sources, or starting a K3
server.

- [ ] **Step 1: Write failing assembly tests**

With a tiny injected layout, assert assembly writes:

```text
expert-manifest.json
model.safetensors.index.json
conversion-receipt.json
resident-*.safetensors
experts-t158-layer-*.bin
```

Assert the manifest is multi-part, resident tensors are absent from sidecar
parts, every routed record has a valid part-relative range, and no final
manifest appears when any part is incomplete.

- [ ] **Step 2: Verify RED**

Run `tests/test_kimi_k3_t158.py -k assembly`. Expected: missing assembly API.

- [ ] **Step 3: Implement final assembly**

Define:

```python
def assemble_artifact(
    source_root: Path,
    output_root: Path,
    *,
    source_revision: str,
    official_revision: str,
    resume: bool,
    layers: Collection[int] | None = None,
) -> AssemblyResult:
    ...
```

Build `ExpertManifest`, `SidecarInfo(parts=...)`, resident index, and receipt
from verified per-shard receipts. `layers` is a pilot-only selection; final
manifest creation requires all 92 layers.

- [ ] **Step 4: Implement the CLI**

Expose required `--source`, `--output`, `--source-revision`, optional
`--official-revision` pinned by default, `--resume`, `--layers`, `--inspect`,
and `--json`. The CLI prints source/output byte projections before writes and
refuses unpinned revisions.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -q tests/test_kimi_k3_gguf.py tests/test_kimi_k3_t158.py \
  tests/test_expert_shadow.py tests/test_expert_sidecar_parts.py
```

Expected: all available tests pass.

### Task 6: Restore Hy3, download Kimi, pilot, full conversion, and verify

**Files:**

- Source: `/Users/davidtai/.cache/huggingface/kimi-k3-q2-k-source`
- Output:
  `/Users/davidtai/.mtplx/models/GrEarl--Kimi-K3-Q2K-t158-MTPLX-streaming`
- Receipt: output `conversion-receipt.json`

**Security flag:** `none`

**Does NOT cover:** Remote publication or runtime benchmarks.

- [ ] **Step 1: Complete and verify endorsed Hy3 restoration**

Wait for the pinned Xet download, remove only the two incomplete files created
by the failed pull, then run:

```bash
PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/.worktrees/merge-main-230 \
  /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/merge-main-230/scripts/verify_expert_manifest.py \
  /Users/davidtai/.mtplx/models/OpensourceWTF--Hy3-oQ2e-MTPLX-streaming \
  /Users/davidtai/.mtplx/models/OpensourceWTF--Hy3-oQ2e-MTPLX-streaming/expert-manifest.json \
  --sidecar
```

Expected: `valid: true`, 79 routed layers, 15,168 records, and the pinned
`experts.bin` SHA-256.

- [ ] **Step 2: Download the pinned Q2_K source**

Run:

```bash
HF_XET_HIGH_PERFORMANCE=1 hf download GrEarl/Kimi-K3-GGUF \
  --revision 0169245d3ea1473a3f9f03bca821d855df5fb2a3 \
  --include '*.gguf' \
  --local-dir /Users/davidtai/.cache/huggingface/kimi-k3-q2-k-source \
  --max-workers 8 --format agent
```

Expected: 96 GGUF files and the measured 1,007,803,593,728 tensor payload bytes.

- [ ] **Step 3: Download pinned official metadata without weights**

Run:

```bash
hf download moonshotai/Kimi-K3 \
  --revision 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
  --include '*.py' --include '*.json' --include '*.model' \
  --include LICENSE --include README.md --include .gitattributes \
  --local-dir /Users/davidtai/.cache/huggingface/kimi-k3-official-metadata \
  --max-workers 4 --format agent
```

Expected: config, tokenizer, processor, and model-code files with no source
safetensors.

- [ ] **Step 4: Run a real one-expert pilot**

Convert expert 0 of layer 1 to a pilot directory, independently decode the
written t158 segments, and retain JSON metrics for gate/up/down/combined:
RMSE, relative RMSE, cosine, finite count, and exact shapes.

- [ ] **Step 5: Review the pilot**

Confirm the pilot measures t158 against decoded Q2_K, source/output hashes are
present, all cosines are finite, and the output shapes are
`(3072,3584)`, `(3072,3584)`, `(3584,3072)`. If any invariant fails, stop
before the full burn.

- [ ] **Step 6: Run the resumable full conversion**

Run:

```bash
PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/.worktrees/kimi-k3-t158 \
  /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  scripts/convert_kimi_k3_q2k_t158.py \
  --source /Users/davidtai/.cache/huggingface/kimi-k3-q2-k-source \
  --output /Users/davidtai/.mtplx/models/GrEarl--Kimi-K3-Q2K-t158-MTPLX-streaming \
  --source-revision 0169245d3ea1473a3f9f03bca821d855df5fb2a3 \
  --official-revision 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
  --resume --json
```

- [ ] **Step 7: Verify the full artifact**

Run unit/regression tests, strict manifest verification with all sidecar part
hashes, safetensors index coverage, conversion-receipt hash verification, and
`df -h`. Expected: 92 sidecar parts, 82,432 expert records, 638,142,382,080
routed bytes, 114,404,258,816 byte-preserved resident tensor bytes, an exact
text resident allowlist, and no unverified partial files.

### Task 7: Exact K3 MLX component arithmetic

**Files:**

- Create: `mtplx/models/kimi_k3_mlx.py`
- Create: `tests/test_kimi_k3_model.py`

**Security flag:** `none`

**Does NOT cover:** Full checkpoint loading or service registration.

- [ ] **Step 1: Write failing fixed-weight SITU, router, and latent-MoE tests**

Use small deterministic tensors and independent NumPy formulas to assert:

- SITU computes in FP32 and casts back to its input dtype;
- route selection uses correction bias but returned weights come from unbiased
  sigmoid scores, top-k weights are renormalized, and no grouping is applied
  for K3's one expert group;
- routing is computed from the original model hidden state, while only the
  routed experts receive the 3,584-wide latent projection;
- routed output is normalized before latent up-projection and is added to
  shared-expert output computed from the original hidden state.

Run:

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -q tests/test_kimi_k3_model.py -k 'situ or router or latent_moe'
```

Expected: import or symbol failures.

- [ ] **Step 2: Implement SITU, dense/shared MLP, router, and latent MoE**

Expose `switch_mlp` as the routed-expert seam consumed by
`bind_streamed_switches`. The initial object may contain a construction-only
placeholder, but after streaming installation no resident routed weights or
stock fallback may remain reachable.

- [ ] **Step 3: Write failing MLA and KDA recurrence tests**

Compare MLX outputs against independent formulas on small fixed shapes:

- q-LoRA projection and RMSNorm, compressed-KV MLA, the published no-RoPE
  behavior, output sigmoid gate, one latent cache head of width 576, q-a/kv-a
  RMS epsilon exactly `1e-5`, and causal cache update;
- KDA depthwise short convolution, Q/K L2 normalization, full-rank forget
  projection, beta sigmoid, recurrent state update, full-rank output gate,
  and safe forget activation
  `-5*sigmoid(exp(A_log)*(g+dt_bias))`.
- the pinned vLLM K3 construction mapping at
  `658f2f56e557e8141f8726b4626c51496d78d914`, which accepts the checkpoint's
  exact `[128]` `A_log` and installs its first 96 values for the configured 96
  heads; reject every other short/surplus shape.

Test both multi-token prefill and one-token cached decode from the same initial
state. Do not use the production helper as the reference formula.

- [ ] **Step 4: Implement MLA and KDA**

Adapt the installed MLX-LM Kimi code only where its arithmetic matches the
pinned official K3 source. Replace the incompatible q projection, KDA gate,
output gate, and cache semantics explicitly. Resolve configuration choices in
constructors; hot calls execute the installed K3 route directly.

- [ ] **Step 5: Verify component GREEN**

Run all `tests/test_kimi_k3_model.py` tests selected by
`-k 'situ or router or latent_moe or mla or kda'`. Expected: pass on the
Metal-capable local lane.

### Task 8: K3 decoder, attention residuals, cache, and strict loading

**Files:**

- Modify: `mtplx/models/kimi_k3_mlx.py`
- Modify: `mtplx/resident_loader.py`
- Modify: `mtplx/runtime.py`
- Modify: `tests/test_kimi_k3_model.py`
- Modify: `tests/test_streamed_models.py`
- Modify the focused runtime/server tokenizer tests.

**Security flag:** `security` — checkpoint keys and tensor ownership are
validated before model installation.

**Does NOT cover:** Model catalog registration or performance tuning.

- [ ] **Step 1: Write failing decoder/residual tests**

For a tiny 13-layer alternating fixture, independently calculate the
attention-residual prefix sum, per-layer block state, normalized projection,
12-layer block reset, and final output residual. Assert the decoder chooses
KDA/full-attention layers exactly from the pinned one-based lists.

- [ ] **Step 2: Implement decoder layers and final residual route**

Preserve the official layer ordering, pre/post norms, attention-residual
state, block size 12, and final residual application. Construct causal masks
and KDA/full-attention caches once per request shape using the standard
MLX-LM cache protocol.

- [ ] **Step 3: Write failing sanitize and strict ownership tests**

Build a tiny checkpoint with official `language_model.*` names and assert
construction maps every expected K3 text tensor exactly once, excludes
byte-preserved vision/multimodal residents from the text load, and rejects
missing, duplicate, wrong-shape, or unexpected text tensors before generation.
Cover GGUF convolution layout, q-LoRA, MLA KV decomposition if required,
latent-MoE projection names, KDA `A_log`/`dt_bias`, and attention-residual
weights.

- [ ] **Step 4: Implement model, sanitize, and resident-loader registration**

Register the MTPLX overlay for `model_type="kimi_linear"` in
`get_streaming_model_classes` and `construct_resident_model`. Flatten the
pinned official `text_config` during artifact assembly while retaining the
original top-level K3 configuration separately. Load only the exact text
resident allowlist and bind all streamed switches after strict model
construction.

- [ ] **Step 5: Tiny end-to-end parity and regression gate**

Before the gate, add K3-specific tokenizer tests proving that the pinned
`tokenization_kimi.TikTokenTokenizer` loads from local audited code with
`trust_remote_code=True` despite the absence of `tokenizer.json`, uses
`eos_token_ids=[163586]`, translates public `enable_thinking` to the K3
`thinking` argument, and forwards `thinking_effort` only when explicitly
provided. Cover thinking-on/off rendering and one OpenAI chat request reaching
the translated template.

Run:

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  -m pytest -q tests/test_kimi_k3_model.py tests/test_streamed_models.py \
  tests/test_expert_streaming_models.py
```

Expected: K3 tiny-model prefill and cached decode pass with no regressions to
Hy3 or GLM.

### Task 9: Full-model construction, real streamed-layer parity, and CLI

**Files:**

- Modify: `mtplx/expert_cli.py`
- Modify: `scripts/benchmark_streamed_generation.py`
- Modify: `tests/test_kimi_k3_model.py`
- Modify the matching CLI/benchmark tests.

**Security flag:** `security` — local path resolution and startup verification
must fail closed.

**Does NOT cover:** Remote publication or enabling MTP.

- [ ] **Step 1: Add a real-layer streamed parity gate**

Load one completed layer part through `ExpertStreamingRuntime`, route a fixed
latent activation through selected real experts, and compare the installed
streamed SITU output to an independent dense evaluation of the same decoded
t158 record.

- [ ] **Step 2: Register the manifest key and multi-part benchmark identity**

Accept `kimi-k3-q1t` in the expert CLI. Keep the manifest `model_key`
authoritative and use the artifact's direct local path for the first launch.
Teach the benchmark runner to accept `--expert-streaming-config` and hash/list
all `sidecar.parts` without touching the single-part-only `.file` property.
Pin `mtp=false`; do not expose an MTP toggle for a checkpoint with zero MTP
layers. Do not create a public catalog alias or promoted memory/optimization
profile from unmeasured values.

- [ ] **Step 3: Construct the complete model**

Run the normal MTPLX loader against the full artifact and require:

- all resident text tensors load exactly once;
- all 92 routed layers bind to t158 SITU streaming;
- layer 0 remains a resident dense SITU MLP;
- the cache list has exactly the official 69 KDA and 24 MLA entries;
- no vision tensor is allocated by the text route;
- no stock routed-expert fallback remains installed.

- [ ] **Step 4: Deterministic generation smoke**

Generate at least 16 deterministic tokens from a short prompt through the
normal MTPLX runtime. Retain prompt, token IDs, decoded text, construction
receipt, and streaming statistics. Reject non-finite logits, repeated
construction failures, or a route with zero sidecar reads.

### Task 10: Basic K3 streaming benchmarks and completion verification

**Files:**

- Write benchmark results under the existing local benchmark-results convention
  identified by the GLM/Hy3 route.

**Security flag:** `none`

**Does NOT cover:** Quality claims against the original MXFP4 checkpoint,
remote publication, or optimization work.

- [ ] **Step 1: Run a fixed no-MTP benchmark**

Using the normal MTPLX runner and one fixed prompt, collect:

- cold construction time;
- prefill tokens/second and prompt length;
- autoregressive tokens/second for at least 32 generated tokens;
- routed bytes read, cache hits/misses, and stream wait time from existing
  runtime statistics;
- process/Metal resident memory from `vmmap -summary` or `footprint`;
- artifact bytes and remaining disk space.

- [ ] **Step 2: Run a repeat/warm measurement**

Repeat the same prompt/config after caches are warm. Do not mix the cold and
warm samples and do not compare against MTP-enabled GLM numbers.

- [ ] **Step 3: Verify endorsed models remain runnable**

Run construction or a minimal generation smoke for the preserved GLM t158
package and the restored endorsed Hy3 oQ2e package without modifying their
catalog defaults.

- [ ] **Step 4: Fresh completion gate**

Run the focused K3 suite, existing expert-streaming regressions, strict
artifact verification, deterministic K3 smoke, and the benchmark commands
again from the final source state. Report exact commands, results, source and
artifact paths, revisions, conversion metrics, model limitations, and disk
state. Do not claim completion without this fresh evidence.
