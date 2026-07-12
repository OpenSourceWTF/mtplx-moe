# Changelog

All notable user-facing changes to MTPLX. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [2.0.2] - 2026-07-09

The agent-reliability release: multi-turn agent sessions now render
reasoning history on the model's trained contract (the fix for the
plan-execution repetition marathons), LAN serving works from the app,
and every agent client gets warm prefix reuse.

### Changed

- Reasoning history is now scoped to the active agent round by default
  on Qwen3.6/3.5 models, matching Qwen's trained multi-turn contract.
  The Qwen chat template keeps `<think>` blocks only for assistant
  messages after the last real user query (its built-in "rolling
  checkpoint"); MTPLX used to override that with `preserve_thinking`,
  which rendered an off-contract empty `<think>` scaffold on every
  completed assistant turn and replayed stale inline reasoning across
  turns, while silently dropping the structured `reasoning_content`
  fields agent clients such as OpenCode send. Scoped mode lets the
  template's own checkpoint govern: completed turns render with no
  think scaffold, and the active round (assistant -> tool -> assistant
  chains) keeps its reasoning, now including structured
  `reasoning_content` - strictly better in-round continuity than
  before. `--preserve-thinking on` restores the previous behavior
  byte-for-byte (including cache identity), `off` still strips
  everything, and the new explicit `scoped` value pins the scoped mode.
  Templates without the rolling checkpoint (Gemma 4, custom templates)
  keep the previous preserve-all behavior. The resolved policy is shown
  at startup ("Reasoning history: scoped (active round only)") and as
  `reasoning_history_mode` in `/v1/mtplx/settings` and the snapshot.

### Added

- Loop Guard (opt-in): `MTPLX_LOOP_GUARD=1` enables a loop-armed
  anti-repetition steering mode for models prone to verbatim repetition
  marathons (for example repetition-damaged third-party quants). Unlike
  a static presence penalty, the guard is completely inert until a real
  loop is detected mid-response (bit-exact sampling otherwise, MTP
  acceptance math unchanged), then penalizes only the tokens that would
  extend a verbatim repeat and disarms once the loop is broken. Content
  inside tool calls is never steered (code legitimately repeats short
  token runs, so tool-call spans are masked token-exactly). The default
  is OFF: no synthetic steering touches sampling unless you ask for it,
  and the repetition fix that matters for MTPLX's own models is the
  scoped reasoning history change above. Detector/steering knobs are
  documented in `mtplx/loop_guard.py`; per-request guard activity is
  visible under `loop_guard` in `/v1/mtplx/snapshot`.

### Fixed

- Hidden reasoning no longer leaks into visible chat content when a
  long reasoning tag (e.g. `</reasoning>`) splits across a streaming
  chunk boundary; the splitter held back too few bytes for tags longer
  than `</think>` (PR #149 by @Osamaali313).
- `mtplx` quickstart onboarding screens now display the requested
  `--host`/`--port` in the Web UI and dashboard URLs instead of a
  hardcoded `http://127.0.0.1:8000` (PR #148 by @hasegaw).
- Ctrl-C now returns control to the terminal within a bounded delay
  even when a browser tab holds an open chat/dashboard stream. The
  server previously waited forever for infinite SSE generators to
  finish; shutdown now drains in-flight requests with a 5-second
  deadline, and thermal/fan cleanup still runs (#124).
- Serving on any host and port now works from the app. Setting host
  0.0.0.0 (LAN serving) used to misreport free ports as occupied, bump
  the port, and kill the healthy daemon after a health-wait timeout,
  because the app probed the bind address verbatim; all app-side
  connections now resolve wildcard binds to a connectable loopback
  address (#109). The app also surfaces the "LAN serving requires an
  API key" rule before launch instead of a generic Degraded state, an
  API-key mismatch reads as a live-but-unauthorized daemon instead of
  a lost one, and the port-in-use preflight tests the address family
  the daemon will actually bind.
- The app now passes the SSD session cache setting to the daemon
  explicitly, including "Off". Since 2.0.0 flipped the serve default to
  on, an explicit Settings "Off" was silently re-enabled with default
  limits on app-launched daemons, and `~/.mtplx/session-bank` kept
  growing (#140; also the "session-bank came back after I deleted it"
  half of #138). Generated `mtplx start` server commands carry the
  explicit `--ssd-session-cache off` for the same reason.
- The app's runtime venv now self-heals after app updates. A venv whose
  base-interpreter symlink pointed into a replaced app bundle made every
  reinstall fail with "[Errno 2] No such file or directory:
  .../runtime-venv/bin/python3" and no DMG reinstall could fix it; venv
  creation now rebuilds with `--clear` when the existing venv python is
  broken or creation fails (#139).
- Warm prefix reuse no longer freezes on the oldest short prefix for
  agent harnesses outside OpenCode (Pi, little-coder, Hermes, custom
  clients). The block-prefix restore lane was still gated to OpenCode's
  compact tool contract even though kvcache-v2's boundary-true restores
  made it exact for every client, so transcripts whose turns diverge
  more than a few tokens before the stored end re-prefilled a growing
  suffix every turn (#138). The lane now engages for all clients while
  boundary-true restore is on; `MTPLX_SESSION_BLOCK_PREFIX_RESTORE=0`
  still disables it.

## [2.0.1] - 2026-07-07

Turbo for every Mac. The v2 turbo default now covers every dense catalog
model on every Apple Silicon generation, with a load-time kernel
self-validation safety net.

### Added

- 6-bit affine verify kernels (split-K hexpack family): the 6-bit 9B tier
  gains 33-62% decode and 43% 2k-prefill under turbo (M5 Max, verified
  arms). Qwen 3.5 9B and 9B FP16 now default to turbo.
- New model: `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality-FP16`, the
  missing M1/M2 quality artifact. Wired into the app picker, CLI catalog,
  and chip-aware routing (a Quality pick on M1/M2 resolves the FP16
  sibling, mirroring the speed lane). Validated against the bf16 parent
  and measured at 2.5x over true AR under turbo.
- Load-time kernel self-validation: every turbo lane checks itself against
  stock MLX in the model's exact dtype/quantization at boot; a mismatching
  lane falls back to the stock path for the session and the verdicts are
  surfaced as `kernel_selfcheck` in `/health`. Worst case is v2.0.0 speed,
  never wrong output.
- `MTPLX_FORCE_GPU_FAMILY_FALLBACK=1` rehearses the exact M1-M4 code path
  on newer machines; `MTPLX_KERNEL_SELFCHECK=0` disables the probe.
- CI kernel matrix on real M1 runners: kernel exactness across
  {bf16, fp16} x {4, 6, 8}-bit plus a live 4B turbo boot smoke.

### Changed

- 27B Optimized-Speed-FP16 (the M1/M2 routing target) defaults to turbo:
  19-31% faster decode than the v2.0.0 sustained default across 0.5k-32k
  context; true-AR multiplier 1.34x to ~2x. First-ever e2e measurement of
  this artifact; exactness gated (30/30 hot-shape logit-diff cases on real
  weights, greedy match turbo vs stock).
- The published Speed-FP16 model card now recommends the turbo profile.

### Unchanged on purpose

- 35B A3B MoE and Gemma 4 keep sustained (expert layers / assistant-pair
  architecture bypass these kernels; named 2.0.2 lanes). The 4B keeps
  sustained (turbo measured slightly slower at matched depth). Compiled
  verify stays off for 6-bit models.

## [2.0.0] - 2026-07-06

MTPLX v2: the coding-agent release. Session-cache v2 (RAM + SSD), the
turbo profile with NAX verify kernels and compiled verify, a new verify
attention kernel wave for long context, and a long campaign of
OpenCode/agent-bridge fixes measured on real sessions.

### Added

- Session-cache v2: boundary-true GDN restores, O(1) RAM restores, SSD
  cold tier (default on) that survives daemon restarts — a 100k-token
  session restores in ~2s after a restart instead of a five-minute cold
  prefill. Prompt-cache reuse now chains across agent tool rounds.
- Turbo profile: verify-specialized quantized-matmul kernels
  (`MTPLX_NAX_VERIFY`, vk_k/vk-q8 families) plus context-routed compiled
  verify with a per-model quantization gate. Measured on M5 Max: 27B
  Optimized-Speed 44.7 -> 58-60 tok/s chat lane; Optimized-Quality (q8)
  31-36 -> 43-44 tok/s.
- The quantized 27B flagships (Optimized-Speed, Optimized-Quality, and
  the legacy Optimized hybrid) now default to the turbo profile on the
  CLI and the bare OpenAI API — the same launch rule the macOS app
  applies. Explicit `--profile` flags and wizard picks still win; other
  models keep the sustained default.
- `POST /admin/cache/clear` resets the MLX peak-memory counter after
  dropping the session bank, so per-request `peak_memory_bytes` reports
  the current phase instead of a process-lifetime ratchet (benchmark
  harnesses that clear between context rows now chart honest per-row
  peaks).
- `--scheduler-mode ar_batch` now genuinely admits anonymous OpenAI
  clients into the concurrency-adaptive batch lane (lone requests keep
  solo MTP; real concurrency shares the batched AR decode lane), and
  the batched lane samples on the GPU (decode-heavy batch-8 aggregate
  70.9 -> 79.2 tok/s). Serial remains the default: measured end to end,
  serialized solo-MTP still beats batched AR on prefill-heavy
  concurrent loads because MTP decode is ~4x faster per stream.
- Long-context decode wave: a packed-GQA verify attention kernel plus
  commit-first KV donation in compiled verify. Measured on M5 Max
  (Optimized-Speed): 64k decode +12%, 128k decode 17 -> 20+ tok/s, and
  peak memory down 8 GB at 64k / 16 GB at 128k.
- Startup warming without the wait: the daemon is ready in ~2s and the
  deeper kernel/shape warmup continues silently in the background,
  yielding instantly to real requests. First messages hit warm kernels
  without a slow boot.
- RAM session-cache budget now scales to the machine (roughly half the
  RAM headroom above the model) instead of a flat cap, and the app's
  Settings tab exposes explicit RAM and SSD cache limits.
- Per-request presence/frequency penalties end-to-end (server, CLI
  flags, dashboard slider, app dial), with MLX on-device penalty math.
- App chat: markdown renders live during streaming at zero per-token
  cost; each turn gets one compact activity strip with grouped tool
  rounds and a sources footer for web results; turbo is a first-class
  mode in Settings.
- Tool contracts are date-anchored (web-search answers stop regressing
  to the training cutoff) and post-search answers are no longer
  clipped to one sentence.
- Vision: images flow through the OpenAI API into MTP decode; MoE
  multimodal checkpoints that store the tower under `model.visual.*`
  (for example Ornith 1.0) are now recognised (community PR #134).
- Gemma 4 assistant-pair models default to their measured-best MTP
  depth; explicit `--depth` still wins.

### Fixed

- Fresh installs no longer crash at model load: transformers 5.13.0
  broke mlx-lm's import (`AutoTokenizer.register` string key), which
  killed every new install and DMG first run. mtplx now pins
  `transformers<5.13` (#135, #136, community PR #137).
- The app no longer kills a healthy engine. The health watchdog
  treated a response it could not parse the same as a dead server and
  terminated the daemon mid-session — the main driver of "Stream
  offline" / "server dies mid-session during agent workloads" reports
  (#105). Liveness is now transport truth: if the daemon answers, it
  lives.
- SSD session-cache restores are boundary-true for recurrent (GDN)
  layers, fixing corrupted agent output after a prefix restore (prompt
  recitation, phantom tool calls, argument leakage) (#130).
- Vision + MTP: the draft head's committed history now consumes the
  spliced vision rows instead of image-pad embeddings, fixing
  fabricated visual differences between similar screenshots (#103).
- Smart fan control ramps at request arrival, verifies actual RPM, and
  holds through the post-response cache work instead of dropping to
  auto while the GPU is still pinned (#127).
- The app no longer rewrites the Hermes profile config on every
  launch; user sections (memory/providers/delegation/auxiliary) are
  merge-preserved (#131).
- Served model identity is contract-match-only: third-party builds no
  longer get coerced onto official `mtplx-*` model ids (#57).
- OpenCode plan -> build mode switch no longer breaks the prompt cache
  or hides file tools. The bridge misread OpenCode's build-mode
  reminder ("no longer in read-only mode") as a read-only instruction,
  hid write/edit exactly when the user said "execute the plan", and the
  model spiralled re-planning files it could not create. OpenCode
  toolsets now pass through byte-stable; the negation is parsed
  correctly for other clients.
- Agent transcripts render prefix-stable across rounds (historical
  bytes never rewrite), force-answer and Pi-convergence contracts ride
  as pure suffixes, and warm prefills inherit recurrent boundaries —
  together these take mid-session tool rounds from multi-second cold
  re-prefills to sub-2s warm restores.
- One busy OpenCode conversation no longer evicts every other project
  from the RAM session cache (prefix-superseded entries + a wider
  high-memory entry budget); multitasking across projects keeps each
  project's cache warm.
- `mtplx start`'s live-dashboard handoff no longer crashes on an
  ImportError (community PR #133).
- The batch scheduler no longer accumulates every finished request
  forever (community PR #132).
- Prefill disconnect-cancel: closing an agent client mid-prefill frees
  the engine immediately instead of finishing a 48k-token orphan.
- CJK and dead-key input no longer drops composed characters in the
  app's chat composer (community PR #119).
- Dense-layout prefill chunk cache-cleanup cadence relaxed 1 -> 4:
  5-21% prefill TPS, memory byte-identical.

### Changed

- Bumped to 2.0.0. The default OpenCode/agent daemon profile is turbo
  with the compiled-verify per-model gate; q8 Quality stays on the
  eager verify path it measures best on.
- Removed the vestigial "required MLX fork" metadata from all profiles.
  MTPLX runs on stock PyPI MLX and always has in the shipped product;
  the speed stack (NAX verify kernels, packed-GQA verify attention,
  compiled verify) ships as in-package Metal kernels, not a patched
  MLX/qmm build. Profile payloads no longer carry
  `required_mlx_fork_commit`/`required_mlx_fork_fragment`, `/health`
  now reports a plain `mlx_runtime` diagnostic instead of a fork
  expectation, and `--strict-fast-path` /
  `--strict-mlx-fork-assert` are accepted as deprecated no-ops.

## [1.0.4] - 2026-06-12

Same-day hotfix: 1.0.3 broke coding agents on their first tool turn,
and pasting a GGUF repo got a wrong answer.

### Fixed

- Coding agents crashed on 1.0.3. Any tool-using client (Pi, Hermes,
  OpenCode, or anything speaking the OpenAI tools protocol) hit
  "unexpected keyword argument 'vision_splice'" on its first tool
  turn after a cache miss: non-streaming callers got a 500, streaming
  agents lost the stream, and the app reported "Stream offline." One
  stray argument left behind by the vision work, in a diagnostics
  call only agent tool turns reach. Removed, with tests that drive
  the exact path and an audit test that fails if any call ever passes
  an argument its target does not accept (#99, #100).
- Pasting a GGUF repo into "Add a model from Hugging Face" claimed
  the repository did not exist. The app now says what is actually
  going on: GGUF is llama.cpp's format and MTPLX runs MLX models. It
  names the source repo the GGUF was made from so Forge can convert
  it, and genuine typos get "check the name" instead.
- The "Add a model" repo check now follows the configured Hugging
  Face download mirror instead of always probing huggingface.co, so
  it works on networks where huggingface.co is blocked.

## [1.0.3] - 2026-06-12

The app can see. Vision lands across the Qwen models, and the
compatibility gate stops blocking models that run fine.

### Added

- Vision support in chat and the API. Attach PNG, JPEG, or WebP images
  in the app composer, or send OpenAI image_url content parts to
  /v1/chat/completions, and the model describes what it sees with MTP
  speculative decoding still running on top. Works on Qwen 3.6 27B
  (Speed and Quality), Qwen 3.6 35B, and Qwen 3.5 9B. The 9B repo on
  Hugging Face regained its vision weights; an explicit mtplx pull now
  syncs such repo updates into existing local copies automatically.
- /health reports whether the loaded model supports vision, and the
  composer adapts to it.

### Fixed

- Models that run fine are no longer refused for paperwork. The
  compatibility gate treated unverified runtime contracts (including
  the official Optimized Quality build) and even "slower than AR"
  speed evidence as reasons not to load. Verification is now a label:
  unverified models load with an honest note, and refusals are
  reserved for models that genuinely cannot execute (#98).
- The gate's explanation message crashed with a traceback instead of
  printing since 1.0.0. It prints again, including the hint that was
  supposed to unblock you (#98).
- Image attachments preview their actual pixels in the composer and
  the transcript instead of a "Could not read" placeholder.

## [1.0.2] - 2026-06-11

Bug-fix release with one small feature.

### Fixed

- Choosing the Auto or Sustained Max profile in the app's Settings left
  the engine unable to start, showing Degraded on every launch until
  the profile was changed back. Both values now resolve to real
  profiles (Sustained Max keeps its pinned-fans intent as the fan mode
  setting), existing configurations heal themselves on load, and the
  picker only offers values the engine accepts. `mtplx serve --profile
  auto` works from the command line too.
- Parallel requests from agent tools that do not send session ids could
  fail with "session anon-... is already in flight" when they shared a
  prompt prefix. Busy sessions now fork to a fresh session instead of
  erroring, and anonymous session ids are random rather than clock
  derived. Reported and fixed by Frank Denis (@jedisct1) in #95.
- A daemon launch that lost its port (another server bound it between
  checks, or a listener invisible to the local probe held it) now
  remediates and retries once before reporting a failure, and the
  failure message names the occupant when it can.

### Added

- Optional Hugging Face download mirror for networks where
  huggingface.co is blocked (requested from mainland China in #96). Set
  it inline in the onboarding download step or later in Settings under
  Advanced; downloads and the engine then use the mirror endpoint. The
  stored HF token is never sent to a mirror, so gated repos stay on the
  official endpoint.

## [1.0.1] - 2026-06-11

Bug-fix release.

### Fixed

- First-run tuning no longer fails on Macs where fan control cannot
  verify a max ramp (for example when the passwordless helper grant is
  not in place yet). Tuning now runs with fans on automatic, the
  results are labeled accordingly, and `--require-max-fans` keeps the
  strict behavior for benchmarking.
- The `mtplx` CLI accepts the official Gemma 4 assistant-pair repos
  directly from Hugging Face. The app already ran them; the CLI's
  preflight now reaches the same verdict.

## [1.0.0] - 2026-06-10

The first full release: the native macOS app and the `mtplx` command line
working as one product. Full notes:
[mtplx.com/releases/notes/v1.0.0](https://mtplx.com/releases/notes/v1.0.0.html).

### Added

- Native macOS app with onboarding (hardware check, model pick, guided
  setup, tuning), a live speed dashboard (decode gauge, acceptance by
  depth, verify waterfall, activity), native chat with attachments and
  web search, Forge, the AIME benchmark, and agent launchers for
  OpenCode, Pi, Hermes, and Open WebUI.
- New models: Gemma 4 (assistant-pair drafting tuned by draft block
  size) and Qwen 3.6 MoE 35B-A3B (prequantized expert sidecars,
  normalized expert layouts, hard blocks on unrunnable layouts),
  alongside Qwen 3.5 4B and 9B for smaller machines.
- KV cache reuse on two layers: warm-prefix reuse in RAM across turns
  and requests (multi-turn chats and agents like OpenCode hit the cache
  instead of re-processing the conversation), and an SSD session cache
  that persists KV state to disk with enforced size caps and restores
  near-instantly across server restarts.
- Concurrency: continuous batching with presets, a scheduler mode, and
  explicit caps (`--max-active-requests`, `--decode-batch-max`,
  `--batch-wait-ms`).
- Smart fan mode across the app, CLI, and server API: ramps while the
  model works, restores on idle, survives client handoffs, and keeps the
  crash-safe restore watchdog.
- Forge: convert any Hugging Face repo to MLX (AWQ, compressed-tensors,
  NVFP4, BF16 sources), calibrate and train the MTP adapter, verify with
  quality gates that reject speed wins that degrade output, and publish
  with provenance. Vision towers are preserved through conversion. In
  the app and as `mtplx forge`.
- Agent-grade serving: hardened tool contracts and dedicated lanes for
  OpenCode, Pi, and Hermes; long-context depth policy; client identity
  tagging; a live server-sent metrics stream plus snapshot, thermal, and
  prefill-history endpoints; honest cancellation that stops decode.
- Automatic runtime setup during onboarding: the app installs its own
  Python engine, fan control (ThermalForge), and the `mtplx` terminal
  command without requiring Homebrew. Release builds bundle a pinned
  CPython interpreter, the engine environment ignores user pip
  configuration, and the interpreter is signed so installed packages
  load on macOS 14 and 15. A stale `mtplx` on PATH is updated
  automatically; a newer one is left alone.
- Official Apple Silicon model catalog (Qwen 3.5/3.6, Gemma 4 in speed,
  balance, and quality builds) with device-aware defaults shared by the
  app and the CLI: chip generation picks precision and machines under
  32 GiB route to the 9B model automatically.
- App-aware `mtplx start`: detects a running MTPLX server and attaches
  instead of loading a second copy, lists installed models first, and
  adds a "Same as the MTPLX app" option. `mtplx stop` knows the app's
  persisted port.
- New commands: `mtplx stop`, `mtplx settings get/set`, and
  `mtplx bench aime` for running the app's AIME benchmark from the
  terminal.
- Sparkle automatic app updates with signed appcasts; the app verifies
  the installed engine against the shipped wheel and refreshes it after
  each update.

### Changed

- Busy ports are now handled gracefully everywhere: the app moves to the
  next free port with a banner (and persists it), and the CLI explains
  exactly who owns a busy port and how to stop it.
- The OpenAI-compatible server honors `stop` sequences (chat,
  completions, and Anthropic `stop_sequences`) and `/v1/completions`
  streams tokens as they are generated with real finish reasons.

[1.0.0]: https://github.com/youssofal/mtplx/releases/tag/v1.0.0
