# Qwen3.8 Flash-Next resident n-gram cache

`OpensourceWTF/Qwen3.8-Flash-Next-MTPLX-oQ4-MTP` contains a 29.80 GiB learned
n-gram embedding table. The model hashes recent 2-token and 3-token histories
into exact 100-byte affine-Q4 rows from that table.

MTPLX keeps the target model and embedded MTP model resident, but keeps only a
bounded set of n-gram rows in unified memory. A cache hit reuses the resident
row. A miss reads the authoritative row from the original safetensors shard.
Eviction changes residency only; cache size does not change model values or
correctness.

## Use it

```bash
mtplx pull OpensourceWTF/Qwen3.8-Flash-Next-MTPLX-oQ4-MTP

mtplx serve \
  --model OpensourceWTF/Qwen3.8-Flash-Next-MTPLX-oQ4-MTP \
  --ngram-cache-limit 4GiB \
  --port 8000
```

`--ngram-cache-limit` is the requested row-payload ceiling. It accepts any
positive size understood by Pydantic's `ByteSize`, including:

- `1GB` — 1,000,000,000 bytes
- `1GiB` — 1,073,741,824 bytes
- `4096MiB` — 4,294,967,296 bytes
- `1.5 GB` — 1,500,000,000 bytes
- raw byte counts such as `1073741824`

The default is `1GiB`. There is no fixed maximum. A large requested value does
not bypass memory safety: construction selects the largest exact-row cache that
fits after resident weights, KV/MTP state, Metal working memory, cache metadata,
transients, and the safety reserve are charged against the 82 GiB runtime
target. A value too small to hold one exact row plus required overhead fails at
startup.

The environment equivalent is:

```bash
export MTPLX_NGRAM_CACHE_LIMIT=4GiB
mtplx serve \
  --model OpensourceWTF/Qwen3.8-Flash-Next-MTPLX-oQ4-MTP
```

An explicit `--ngram-cache-limit` takes precedence over the environment.

## Measured example

The guarded 16,384-input / 1,024-output validation run used the earlier 10 GiB
request and the 82 GiB runtime target. Construction selected:

| Component | Size |
|---|---:|
| Exact-row payload | 1,677,721,600 bytes (1.5625 GiB) |
| Cached rows | 16,777,216 |
| Metadata, route table, and transient overhead | 892,502,016 bytes (0.831 GiB) |
| Total cache reservation | 2,570,223,616 bytes (2.394 GiB) |

The requested value was only a ceiling; the complete memory plan determined the
smaller allocation.
