# GLM-5.2 Q4 target + BF16 MTP D+1 qualification

Date: 2026-07-14

Issue: `davidtai/MTPLX#52`

Qualification code commit: `9db07573c3dd193d5cc2f0c935d07f939f2cd16c`

Stacked base: `codex/q2-bf16-mtp-bench` at
`68c9c09` (including greedy verifier-correction fix `7a3bbb4`)

Machine lane: exclusive local MLX lane, with Qwen stopped and restored by
`scripts/run_with_qwen_stopped.py`

## Outcome

**PASS for the D+1/D1 correctness contract.** The single retained D1
observation passed every hard verifier-event, counter, committed-history,
cache-offset, generated-token, and safe-final-state gate. This was the one
complete D+1 observation requested by the user; execution stopped afterward,
so this result makes no repeated-observation determinism claim.

The AR-versus-D1 comparison found token divergence beginning at output index
12, with 116 of 128 output tokens different. The validator classified the
divergence as `unclassified`. AR parity is diagnostic in this qualification
contract, but this result must not be presented as a correctness equivalence or
performance-promotion result. The measured throughput is qualification-only.

## Reproducibility identities

| Item | Value |
| --- | --- |
| Q4 model root | `models--mlx-community--GLM-5.2-4bit/snapshots/6b347a6472d46bf55de65ee34032136a3929d778` |
| Q4 model key | `glm52-q4` |
| Q4 expert manifest SHA-256 | `38bfd15988ca21461f08c2ca59a1cbcd3cc7be97a225da7807596c3d46e2acce` |
| Q4 manifest records | 19,200 |
| Q4 routed logical bytes | 407,686,348,800 |
| Q4 manifest verification | valid; 91 shards checked; 0 record payload hashes requested; sidecar verification disabled |
| BF16 MTP source | `zai-org/GLM-5.2` at `b4734de4facf877f85769a911abafc5283eab3d9` |
| BF16 MTP artifact | `glm52-mtp-layer78/layer78-bf16.safetensors` |
| BF16 MTP artifact SHA-256 | `56a19e9c0328f3b8f9ec32569f17d76aef7c20081334971d8855514e409746a6` |
| BF16 MTP artifact bytes | 19,905,942,064 (100,392-byte header; 19,905,841,664-byte payload) |
| BF16 MTP header SHA-256 | `7475bb46b5744f35a4938d8d3edd18e6293a82eb6d90b1439904bafd56979d21` |
| BF16 MTP manifest SHA-256 | `e6fbcb0a673b5072080eb9cf8efb0a0b7e3a9355f9d7fd1d198197b052902c67` |
| BF16 MTP tensors | 791 total: 790 BF16, 1 F32 |
| Validator file SHA-256, before and after | `92fab54a7bcc86f025d607e023ae97545f9795ff565a11fe2470461332e6f27b` |
| Validator-specific diff SHA-256, before and after | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty diff) |
| Raw result SHA-256 | `ec8e30a0fd287f89f66221b35b23b059947341c86d726831fe4e5b0da4a456e4` |
| Raw result path | `benchmarks/raw/2026-07-14-glm52-q4-bf16-mtp-d1-c512-o128-smoke-a.json` (local, gitignored) |

Deep MTP verification also confirmed that the 791 tensors came from source
shards `model-00270-of-00282.safetensors` through
`model-00274-of-00282.safetensors`, with all tensor hashes verified.

## Request and runtime configuration

| Setting | Value |
| --- | --- |
| Context / output / warmup output tokens | 512 / 128 / 8 |
| Retained replicates | 1 |
| Requested cells | AR and D1 |
| Seed / temperature / top-p / top-k | 0 / 0.0 / 1.0 / 1 |
| Stop token IDs | none |
| Repetition stop / loop guard | disabled / disabled |
| Final-state capture | enabled |
| MTP cache / history policy | persistent / committed |
| Measurement lane | `headline-uninstrumented` |
| Memory limit / runtime reserve | 112 GiB / 12 GiB |
| Expert cache limit | 64 GiB |
| Maximum live KV tokens | 4,096 |
| Cache policy / scope | frequency / layer |
| Frequency decay | 0.995 |
| Slot layout / transient slots | component-banks / 8 |
| Maximum read chunk | 8 MiB |
| Maximum open files | 16 |
| Bypass page cache | true |
| Resource telemetry / route tracing | false / false |
| Prefill admission | false |
| Verify artifact headers / record hashes | true / true |
| Verify sidecar hash at open | false |
| Prefer sidecar | true |
| Sustained prefill environment | `MTPLX_SUSTAINED_PREFILL=1`; other late-depth/layout controls unset |

Runtime-resolved byte limits were 120,259,084,288 bytes total,
12,884,901,888 bytes reserve, 68,719,476,736 bytes expert cache, and
8,388,608 bytes per read chunk. Execution workspace and I/O staging were both
0 bytes; maximum in-flight I/O was unset.

The raw coding-agent prompt contained exactly 512 tokens. Its token SHA-256 was
`74dd24557a2bf6b516cc442a5632deb42438f3fd6dd2f9f074b0baa326d001ca`.
The preserved 13-token tail SHA-256 was
`4e4d4f25b132f19f6314295af3612b188e0e78333338872beea70a79a9dd053e`;
the 499-token filler SHA-256 was
`69b2be3dbd45365504b6dd0f0caff283b04905e96e0b905fa670260e359f8dcc`.
No head or tail tokens were trimmed, thinking was disabled, and the prompt
release-valid/tail-preserved checks passed.

## Complete retained timing, throughput, and memory statistics

| Statistic | AR | D1 |
| --- | ---: | ---: |
| Prompt tokens | 512 | 512 |
| New prefill tokens | 512 | 512 |
| Generated tokens | 128 | 128 |
| Finish reason | `length` | `length` |
| Prompt evaluation time (s) | 109.35727845801739 | 244.7414066249912 |
| Target prefill time (s) | 109.35727845801739 | 235.25991712498944 |
| Target prefill throughput (tok/s) | 4.681901444690382 | 2.176316332407715 |
| Total ingestion throughput (tok/s) | 4.681901444690382 | 2.092003993359897 |
| Prompt MTP-history time (s) | 0.0 | 9.481489500001771 |
| Prompt MTP-history tokens | 0 | 511 |
| Decode time (s) | 559.0354989579937 | 556.6791236250137 |
| Raw decode time (s) | 559.0354989579937 | 556.6791236250137 |
| Total elapsed time (s) | 668.3927774160111 | 801.420530250005 |
| Raw total elapsed time (s) | 668.3927774160111 | 801.420530250005 |
| Decode throughput (tok/s, qualification only) | 0.2289657816696503 | 0.2299349743286268 |
| Reported decode throughput (tok/s) | 0.2289657816696503 | 0.2299349743286268 |
| End-to-end throughput (tok/s) | 0.19150416390620592 | 0.15971639753235434 |
| Peak memory (bytes) | 99,876,055,152 | 99,956,895,802 |
| Final-state capture time (s) | 0.0 | 0.0 |
| Replicate | 1 | 1 |

The model loaded once. Load peak memory was 99,189,807,112 bytes, hard peak
memory was 99,956,895,802 bytes, and two warmup observations were discarded.

## Speculation and token statistics

| Statistic | AR | D1 |
| --- | ---: | ---: |
| Requested / effective depth | 0 / 0 | 1 / 1 |
| Drafted tokens | 0 | 64 |
| Evaluated drafts | 0 | 64 |
| Accepted drafts | 0 | 63 |
| Rejected drafts | 0 | 1 |
| Verify calls | 127 | 64 |
| Fully accepted verify calls | 0 | 63 |
| Fully accepted verify ratio | 0.0 | 0.984375 |
| Accepted per verify | 0.0 | 0.984375 |
| Depth-1 conditional hit rate | n/a | 0.984375 |
| Depth-1 cumulative accepted/drafted yield | n/a | 0.984375 |
| Depth-1 mean accept probability | n/a | 0.984375 |
| Output token SHA-256 | `4c722834db8ce8b1d679b2842294ba2d240edf6eb1eeec0cc00bf9b4ebc244b4` | `c32ac86dd97732bcab484c36c554bbbf6235151fa99b652bb0fcaf4ab723e847` |
| AR token parity | reference | false |
| First divergence | n/a | output index 12 |
| Token at first divergence | n/a | AR 15658; D1 13 |
| Differing output tokens | 0 | 116 |
| Divergence attribution | n/a | `unclassified` |

The D1 event ledger contained 64 total events and 64 verify events, with 64
drafted records, 64 evaluated records, and 63 accepted records. These totals
agree with the generation counters and the one rejection correction.

## Final-state contract and hard gates

The D1 final state reported `safe_to_commit=true`, matching finish reasons,
matching generated token IDs between the returned result and captured final
state, committed MTP cache offset `[639]`, all 80 target-layer cache offsets at
`640`, MTP history position base `0`, and 511 prompt MTP-history tokens.

Every hard gate was `true` for both retained cells:

- committed history
- decode expert-cache metrics
- exact effective depth
- final-state contract
- generated-count consistency
- disabled generation guards
- length finish
- exact new-prefill count
- exact output count
- exact prompt length
- exact requested depth
- speculative-event contract

## Expert-streaming counters

### Overall

| Counter | AR | D1 |
| --- | ---: | ---: |
| Bytes read | 1,266,927,796,224 | 1,044,377,763,840 |
| Evictions | 10,616 | 11,257 |
| Expert hits | 35,235 | 46,745 |
| Expert misses | 348,165 | 338,455 |
| Expert requests | 383,400 | 385,200 |
| Hit rate | 0.09190140845070423 | 0.12135254413291796 |
| Persistent loads | 13,841 | 14,482 |
| Route calls | 11,900 | 12,125 |
| Shared-expert assignments | 288,499 | 294,869 |
| Transient loads | 45,825 | 34,703 |
| Unique expert requests | 94,901 | 90,331 |

### Decode phase

| Counter | AR | D1 |
| --- | ---: | ---: |
| Bytes read | 875,400,265,728 | 652,850,233,344 |
| Evictions | 10,616 | 11,257 |
| Expert hits | 34,973 | 46,483 |
| Expert misses | 41,227 | 31,517 |
| Expert requests | 76,200 | 78,000 |
| Hit rate | 0.45896325459317583 | 0.5959358974358975 |
| Persistent loads | 10,616 | 11,257 |
| Route calls | 9,525 | 9,750 |
| Shared-expert assignments | 0 | 6,370 |
| Transient loads | 30,611 | 19,489 |
| Unique expert requests | 76,200 | 71,630 |

### Prefill phase

The AR and D1 prefill counters were identical:

| Counter | Value |
| --- | ---: |
| Bytes read | 391,527,530,496 |
| Evictions | 0 |
| Expert hits | 262 |
| Expert misses | 306,938 |
| Expert requests | 307,200 |
| Hit rate | 0.0008528645833333333 |
| Persistent loads | 3,225 |
| Route calls | 2,375 |
| Shared-expert assignments | 288,499 |
| Transient loads | 15,214 |
| Unique expert requests | 18,701 |

## Commands and lane restoration

The target manifest was verified without allocating the model with:

```bash
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python
shasum -a 256 \
  /Users/davidtai/.cache/huggingface/hub/models--mlx-community--GLM-5.2-4bit/snapshots/6b347a6472d46bf55de65ee34032136a3929d778/expert-manifest.json
PYTHONPATH=. "$PY" - <<'PY'
from pathlib import Path
from mtplx.expert_manifest import load_expert_manifest, verify_expert_manifest

root = Path("/Users/davidtai/.cache/huggingface/hub/models--mlx-community--GLM-5.2-4bit/snapshots/6b347a6472d46bf55de65ee34032136a3929d778")
manifest = load_expert_manifest(root / "expert-manifest.json")
assert manifest.model_key == "glm52-q4"
assert len(manifest.records) == 19_200
assert sum(record.logical_bytes for record in manifest.records) == 407_686_348_800
receipt = verify_expert_manifest(manifest, root)
assert receipt == {
    "valid": True,
    "model_key": "glm52-q4",
    "checked_shards": 91,
    "checked_records": 0,
    "sidecar_verified": False,
}
print(receipt)
PY
```

Immediately before the MLX window, the BF16 MTP was deep-verified with:

```bash
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python
PYTHONPATH=. "$PY" scripts/extract_glm52_mtp_layer78.py verify \
  --output-root /Users/davidtai/.cache/huggingface/glm52-mtp-layer78 \
  --deep > /tmp/glm52-q4-mtp-deep-verify-rerun.json
jq -e '.source.revision == "b4734de4facf877f85769a911abafc5283eab3d9" and
  .inventory.tensor_count == 791 and
  .inventory.payload_bytes == 19905841664' \
  /tmp/glm52-q4-mtp-deep-verify-rerun.json
shasum -a 256 \
  /Users/davidtai/.cache/huggingface/glm52-mtp-layer78/layer78-bf16.safetensors
```

The single observation was run through the exclusive Qwen guard with:

```bash
PYTHONPATH=. PYTHONNOUSERSITE=1 MTPLX_SUSTAINED_PREFILL=1 \
  /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  scripts/run_with_qwen_stopped.py \
  --plist "$HOME/Library/LaunchAgents/com.tea.qwen.plist" \
  --api-url http://127.0.0.1:8080/v1/models \
  --lock-path /tmp/mtplx-gpu-exclusive.lock \
  --lock-timeout-seconds 1800 -- \
  /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  /tmp/run_glm52_q4_d1_qualification.py
```

After observation A completed, execution was intentionally stopped before a
second observation. The guard restored Qwen, and the immediate post-run check
confirmed that the advisory lock was unheld and the exact captured model
`mtplx-qwen36-27b-optimized-speed` was served by `/v1/models`. A later,
independent benchmark acquired the shared lane; it was not interrupted and is
not part of this result.

The gitignored raw JSON is the source of truth for the full token-ID arrays,
all 64 D1 verifier events and their timings, captured final state, and every
counter recorded above.
