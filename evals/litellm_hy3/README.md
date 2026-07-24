# hy3 q2 → LiteLLM → HumanEval

A LiteLLM `CustomLLM` attachment that exposes the **hy3 q2** MTPLX runtime as an
OpenAI-compatible endpoint, so `scripts/code_eval_gate.py` (and any OpenAI
tooling) can run HumanEval / MBPP against it.

Validated against **litellm==1.93.0** (Python 3.12, matching the campaign venv).

```
evals/litellm_hy3/
├── handler.py                 # Hy3StreamedLLM(litellm.CustomLLM) + module-level `instance`
├── config.yaml                # proxy config: custom_provider_map -> handler.instance, model `hy3-q2`
├── serve.sh                   # guarded-window launcher (stops qwen, takes flock, runs proxy)
├── run_humaneval.sh           # code_eval_gate.py invocation against the proxy
├── test_handler_offline.py    # offline unit tests (no model load)
└── README.md
```

## What it does

* Builds **one** `MTPLXRuntime` for hy3 q2 lazily on the first request (a
  module-level singleton, thread-locked), reusing the campaign benchmark's own
  config builder: `scripts/benchmark_q2_mtp_depth_matrix.py` →
  `DEFAULT_RUNTIME_OPTIONS` + `_runtime_config` + `_default_apis`. The runtime
  is therefore byte-identical to the benchmarked configuration, not a
  reimplementation.
* `completion()` / `acompletion()`:
  * **chat** endpoint → applies hy3's `chat_template.jinja` to `messages`
    (`reasoning_effort=no_think` so the model emits code directly);
  * **completions** endpoint → tokenizes the raw prompt verbatim (detected via
    `litellm_params["text_completion"] is True`);
  * tokenizes → `prompt_ids`, calls
    `generate_mtpk(rt, prompt_ids, max_tokens, sampler=SamplerConfig(...),
    speculative_depth=2, stop_token_ids=…)`, decodes, and returns a valid
    `litellm.ModelResponse` (OpenAI chat schema: `choices[0].message.content`,
    `finish_reason`, `usage`).
  * Honors `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`, and string
    `stop` from the request.
* Streaming is **not** implemented (non-streaming is sufficient for HumanEval).
  A `stream=true` request will get a `CustomLLMError` from the base class.

## Champion configuration (baked defaults, all env-overridable)

| Setting | Default | Env override |
| --- | --- | --- |
| model dir | `~/.cache/huggingface/hy3-expert-only-mlx-q2` | `MTPLX_HY3_MODEL_ROOT` |
| expert manifest | `<model>/expert-manifest.json` | `MTPLX_HY3_MANIFEST` |
| MTP artifacts | `~/.cache/huggingface/hy3-bf16-and-mtp-layer80` (bf16) | `MTPLX_HY3_MTP_ARTIFACTS` |
| memory-limit | `103GiB` | `MTPLX_HY3_MEMORY_LIMIT` |
| runtime-reserve | `7GiB` | `MTPLX_HY3_RUNTIME_RESERVE` |
| expert-cache-limit | `2GiB` | `MTPLX_HY3_EXPERT_CACHE_LIMIT` |
| islands | `1-35,44,46,49-51,55-58,60,63-76,79` | `MTPLX_HY3_ISLANDS` |
| mmap-islands | `38,45,47-48,52-54,59,61-62,77-78` | `MTPLX_HY3_MMAP_ISLANDS` |
| banked-manifest | `<model>/experts-banked-mmaparm-manifest.json` | `MTPLX_HY3_BANKED_MANIFEST` |
| proj-quant | `q4` | `MTPLX_HY3_PROJ_QUANT` |
| expert-integrity | `headers-only` | `MTPLX_HY3_EXPERT_INTEGRITY` |
| split-route-release | `deferred` | `MTPLX_HY3_SPLIT_ROUTE_RELEASE` |
| speculative depth K | `2` | `MTPLX_HY3_SPECULATIVE_DEPTH` |
| verify-strategy | `batched` (safe) | `MTPLX_HY3_VERIFY_STRATEGY` |
| draft-core | `stock` (safe) | `MTPLX_HY3_DRAFT_CORE` |
| reasoning_effort | `no_think` | `MTPLX_HY3_REASONING_EFFORT` |
| port | `18183` | `MTPLX_HY3_PORT` |

`serve.sh` also exports `MTPLX_SUSTAINED_PREFILL=1`,
`MTPLX_DEFERRED_PIN_RELEASE=1`, `MTPLX_HY3_SUBMIT_CADENCE=8` and leaves
`MTPLX_MEMORY_LIMIT_BYTES` unset (the memory knob is enforced only via the
config's `memory_limit`, never a second env override).

**The `banked-manifest` was chosen by matching layer coverage:**
`experts-banked-mmaparm-manifest.json` covers **exactly** the mmap-island layer
set `{38,45,47,48,52,53,54,59,61,62,77,78}` (codec `none`, bin present). The
mmap band requires a banked manifest — `ExpertStreamingConfig` rejects
mmap-islands without one at config-construction time.

> Note: the docs' own reference config
> (`docs/HY3_SSD_EXPERT_STREAMING.md`) uses the simpler fully-resident
> `--island-layer-count 79` (nothing streams, no mmap band, no banked
> manifest). If the explicit island/mmap split is rejected at pre-flight, that
> fully-resident mode is the built-in fallback: set
> `MTPLX_HY3_ISLAND_LAYER_COUNT=79` before `serve.sh`. The handler then pins the
> 79 worst-streaming layers via the model's measured pin order and clears the
> explicit island_layers / mmap band / banked manifest automatically.

## Guarded-window runbook

**Pre-flight:** the box must be free — no other lane holding
`/tmp/mtplx-gpu-exclusive.lock`, and peak memory must stay under the **100 GiB**
wired knob. If the pre-flight rejects the islands config (over budget), **that
rejection is the verdict** — do not raise the memory knob.

1. **Confirm the box is free.**
   ```sh
   pgrep -fl benchmark_q2_mtp_depth_matrix || echo "no benchmark running"
   ls -l /tmp/mtplx-gpu-exclusive.lock   # exists but should be unheld
   ```

2. **Install litellm into the campaign venv + start the proxy (guarded).**
   `serve.sh` does the `pip install 'litellm[proxy]'` into the campaign venv
   first, then launches the proxy inside `run_with_qwen_stopped.py` (stops qwen,
   takes the flock). It holds the window until you Ctrl-C it.
   ```sh
   evals/litellm_hy3/serve.sh
   ```
   Leave this running in shell #1.

3. **Wait until the model is loaded and listed.** In shell #2:
   ```sh
   curl -s http://127.0.0.1:18183/v1/models | python3 -m json.tool
   # expect an entry with "id": "hy3-q2"
   ```
   The FIRST request triggers the model load (tens of seconds to minutes).

4. **Smoke one request.**
   ```sh
   curl -s http://127.0.0.1:18183/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"hy3-q2","messages":[{"role":"user","content":"Write a Python function is_even(n)."}],"max_tokens":128,"temperature":0}' \
     | python3 -m json.tool
   ```

5. **Run HumanEval pass@1.** Still in shell #2:
   ```sh
   evals/litellm_hy3/run_humaneval.sh
   ```
   Writes `evals/humaneval_hy3_q2.json`. `--allow-code-execution` is included
   (scoring executes model-written code in a sandboxed subprocess).

6. **Collect pass@1.**
   ```sh
   python3 -c "import json;s=json.load(open('evals/humaneval_hy3_q2.json'))['summary'];print('pass@1', s['pass@1'], '|', s['passed'],'/',s['tasks'])"
   ```

7. **Shut down.** Ctrl-C shell #1 (`serve.sh`). The guard restores qwen and
   releases the flock automatically.

### base-url gotcha

`code_eval_gate.py` appends `/v1/chat/completions` itself, so `--base-url` must
be `http://127.0.0.1:18183` **without** a `/v1` suffix. `run_humaneval.sh`
already gets this right.

## Offline tests

Runs in the throwaway probe venv (litellm installed, mtplx NOT):
```sh
<probe-venv>/bin/python evals/litellm_hy3/test_handler_offline.py
```
Covers: import without mtplx, chat-template application → token ids, raw-prompt
path, sampler mapping, `ModelResponse` shaping, string-stop truncation,
finish-reason mapping, a full round-trip through `litellm.completion`, and that
`config.yaml`'s `custom_handler` resolves via the proxy's own `get_instance_fn`.

## Not validated until the guarded window

The `generate_mtpk` path and the model load itself cannot be exercised without
the GPU. Specifically unproven until the window:

* that the champion `ExpertStreamingConfig` (103GiB / islands+mmap band /
  proj-q4 / 2GiB expert cache) loads **under the 100 GiB wired knob** — the
  `expert-cache-limit` value in particular is an estimate, not a measured
  campaign number;
* that `generate_mtpk` accepts this exact kwarg set on the loaded hy3 runtime
  and returns non-empty completions;
* that the worktree's `mtplx` source is import-compatible with the campaign
  venv's compiled `mtplx_native_expert_io` / `mlx 0.31.2`;
* end-to-end HumanEval pass@1.
