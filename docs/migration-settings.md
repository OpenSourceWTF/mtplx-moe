# Migrating Flags and Environment Variables

Canonical dotted settings are the primary reusable configuration interface.
The first settings release keeps existing individual CLI flags, reviewed
environment aliases, and flat TOML keys working through compatibility adapters.
They are not removed by this migration.

Prefer explicit scopes:

- Replace reusable flag clusters with `mtplx settings user set NAME=VALUE`.
- Replace one-run overrides with repeatable `--set NAME=VALUE`.
- Replace versioned groups with a data-only `--settings PATH` bundle.
- Use `mtplx settings live show/set` only for a running daemon.
- Use `mtplx settings explain NAME` to confirm which source won.

The old bare `mtplx settings get/set` daemon forms remain accepted. New
automation should spell the `live` scope so a future user-setting operation is
never confused with daemon mutation.

<!-- BEGIN GENERATED SETTINGS ALIASES -->

## CLI flag aliases

| Compatibility name | Canonical setting |
|---|---|
| `--api-key-file` | `server.api_key_file` |
| `--batch-wait-ms` | `runtime.batching.wait_ms` |
| `--batching-preset` | `runtime.batching.preset` |
| `--cache-dir` | `model.cache_dir` |
| `--context-window` | `model.context_window` |
| `--decode-batch-max` | `runtime.batching.decode_max` |
| `--default-temperature` | `generation.temperature` |
| `--default-top-k` | `generation.top_k` |
| `--default-top-p` | `generation.top_p` |
| `--depth` | `runtime.mtp.depth` |
| `--experimental-mtp-cohorts` | `runtime.mtp.cohorts.enabled` |
| `--kv-quant` | `memory.kv.quantization` |
| `--max-active-requests` | `runtime.requests.max_active` |
| `--model` | `model.ref` |
| `--mtp` | `runtime.mtp.enabled` |
| `--no-mtp` | `runtime.mtp.enabled` |
| `--paged-kv-quant` | `memory.kv.quantization` |
| `--paged-kv-quantization` | `memory.kv.quantization` |
| `--prefill-chunk-tokens` | `runtime.prefill.chunk_tokens` |
| `--profile` | `runtime.profile` |
| `--ram-session-block-prefix-restore` | `cache.session.ram.block_prefix_restore` |
| `--ram-session-cache-max-entries` | `cache.session.ram.max_entries` |
| `--ram-session-cache-max-size` | `cache.session.ram.max_size` |
| `--ram-session-cache-per-session-max-size` | `cache.session.ram.per_session_max_size` |
| `--ram-session-cache-policy` | `cache.session.ram.policy` |
| `--reasoning` | `generation.reasoning` |
| `--reasoning-effort` | `generation.reasoning_effort` |
| `--scheduler-mode` | `runtime.scheduler.mode` |
| `--ssd-session-cache` | `cache.session.ssd.mode` |
| `--ssd-session-cache-dir` | `cache.session.ssd.directory` |
| `--ssd-session-cache-max-size` | `cache.session.ssd.max_size` |
| `--ssd-session-cache-min-prefix-tokens` | `cache.session.ssd.min_prefix_tokens` |
| `--temperature` | `generation.temperature` |
| `--thermal-control` | `thermal.control` |
| `--top-k` | `generation.top_k` |
| `--top-p` | `generation.top_p` |

## Environment aliases

| Compatibility name | Canonical setting |
|---|---|
| `MTPLX_BATCHING_PRESET` | `runtime.batching.preset` |
| `MTPLX_COMPILED_VERIFY` | `verify.compiled.mode` |
| `MTPLX_GQA_PACKED_SDPA` | `attention.gqa_packed_sdpa.enabled` |
| `MTPLX_NAX_VERIFY` | `verify.nax.enabled` |
| `MTPLX_PAGED_KV_QUANT` | `memory.kv.quantization` |
| `MTPLX_SCHEDULER_MODE` | `runtime.scheduler.mode` |
| `MTPLX_SESSION_BANK_MAX_BYTES` | `cache.session.ram.max_size` |
| `MTPLX_SESSION_BANK_MAX_ENTRIES` | `cache.session.ram.max_entries` |
| `MTPLX_SESSION_BANK_PER_SESSION_BYTES` | `cache.session.ram.per_session_max_size` |
| `MTPLX_SESSION_BLOCK_PREFIX_RESTORE` | `cache.session.ram.block_prefix_restore` |

<!-- END GENERATED SETTINGS ALIASES -->

Internal experiment and diagnostic environment variables are not promoted by
this table. They remain classified in the generated
[experiment inventory](experiments/inventory.md) and should be migrated only
with an owner, test coverage, and lifecycle decision.
