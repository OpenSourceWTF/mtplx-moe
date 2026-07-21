"""LiteLLM ``CustomLLM`` attachment exposing the hy3 q2 MTPLX runtime.

This module is loaded by the LiteLLM proxy (``config.yaml`` ->
``custom_provider_map`` -> ``handler.instance``). It builds ONE
``MTPLXRuntime`` for hy3 q2 lazily on the first request and answers OpenAI
chat/text completions with ``generate_mtpk``.

Design constraints (see the campaign handoff / README):

* The heavy MLX / mtplx imports live INSIDE ``_build_state`` so the module can
  be imported (and unit-tested) without MLX present. The proxy loads this file
  as a standalone module named ``handler`` -- there is no package context, so
  everything here is import-light at module scope.
* The champion runtime configuration is baked as defaults and is fully
  overridable through ``MTPLX_HY3_*`` environment variables so the guarded
  window operator can tune without editing code.
* The runtime is built by reusing the campaign benchmark's own config builder
  (``scripts/benchmark_q2_mtp_depth_matrix.py``: ``DEFAULT_RUNTIME_OPTIONS`` +
  ``_runtime_config`` + ``_default_apis``) rather than re-deriving
  ``ExpertStreamingConfig`` by hand, so this path stays byte-identical to the
  benchmarked configuration.

Validated offline against litellm==1.93.0. The ``generate_mtpk`` path itself
CANNOT be validated without loading the model on the GPU; see README.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable, Iterable

from litellm import CustomLLM
from litellm.types.utils import Usage


# ---------------------------------------------------------------------------
# Champion configuration (all overridable via env). These mirror the campaign
# handoff for hy3 q2. Layer sets are passed to the benchmark's
# ``parse_island_layers`` verbatim (string form, e.g. "1-35,44,...").
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


MODEL_KEY = _env("MTPLX_HY3_MODEL_KEY", "hy3-expert-q2")  # MODEL_SPECS["hy3-q2"]["model_key"]


def _expand(path: str) -> str:
    return str(Path(path).expanduser())


def _default_model_root() -> str:
    return _expand(_env("MTPLX_HY3_MODEL_ROOT", "~/.cache/huggingface/hy3-expert-only-mlx-q2"))


def _default_manifest(model_root: str) -> str:
    return _expand(_env("MTPLX_HY3_MANIFEST", str(Path(model_root) / "expert-manifest.json")))


def _default_mtp_artifacts() -> str:
    return _expand(_env("MTPLX_HY3_MTP_ARTIFACTS", "~/.cache/huggingface/hy3-bf16-and-mtp-layer80"))


def _default_banked_manifest(model_root: str) -> str:
    return _expand(
        _env(
            "MTPLX_HY3_BANKED_MANIFEST",
            str(Path(model_root) / "experts-banked-mmaparm-manifest.json"),
        )
    )


# Sampling / decode defaults.
DEFAULT_MAX_TOKENS = int(_env("MTPLX_HY3_DEFAULT_MAX_TOKENS", "1024"))
DEFAULT_TEMPERATURE = float(_env("MTPLX_HY3_DEFAULT_TEMPERATURE", "1.0"))
DEFAULT_TOP_P = float(_env("MTPLX_HY3_DEFAULT_TOP_P", "1.0"))
DEFAULT_TOP_K = int(_env("MTPLX_HY3_DEFAULT_TOP_K", "0"))  # 0 => unrestricted
DEFAULT_SEED = int(_env("MTPLX_HY3_DEFAULT_SEED", "0"))
SPECULATIVE_DEPTH = int(_env("MTPLX_HY3_SPECULATIVE_DEPTH", "2"))  # champion K=2
VERIFY_STRATEGY = _env("MTPLX_HY3_VERIFY_STRATEGY", "batched")  # safe default
DRAFT_CORE = _env("MTPLX_HY3_DRAFT_CORE", "stock")  # safe default
MTP_PRECISION = _env("MTPLX_HY3_MTP_PRECISION", "bf16")

# The hy3 chat template controls reasoning via ``reasoning_effort`` (it ignores
# ``enable_thinking``). Valid values are 'no_think' | 'low' | 'high'; anything
# else makes the template raise. 'no_think' closes the <think> block in the
# generation prompt so the model emits the answer directly -- what HumanEval
# wants. Empty string => omit the kwarg entirely (template defaults to
# 'no_think').
_VALID_REASONING = {"no_think", "low", "high", ""}
REASONING_EFFORT = _env("MTPLX_HY3_REASONING_EFFORT", "no_think")
if REASONING_EFFORT not in _VALID_REASONING:
    REASONING_EFFORT = "no_think"

# Fallback eos if the tokenizer exposes none (hy3 config eos_token_id).
_FALLBACK_EOS = int(_env("MTPLX_HY3_FALLBACK_EOS", "120025"))


def _champion_overrides() -> dict[str, Any]:
    """Runtime-option overrides layered over ``DEFAULT_RUNTIME_OPTIONS``.

    Keys match the schema the benchmark's ``_runtime_config`` consumes.
    """

    model_root = _default_model_root()
    overrides: dict[str, Any] = {
        "memory_limit": _env("MTPLX_HY3_MEMORY_LIMIT", "103GiB"),
        "runtime_reserve": _env("MTPLX_HY3_RUNTIME_RESERVE", "7GiB"),
        "expert_cache_limit": _env("MTPLX_HY3_EXPERT_CACHE_LIMIT", "2GiB"),
        "max_live_kv_tokens": int(_env("MTPLX_HY3_MAX_LIVE_KV_TOKENS", "4096")),
        "cache_policy": _env("MTPLX_HY3_CACHE_POLICY", "frequency"),
        "cache_scope": _env("MTPLX_HY3_CACHE_SCOPE", "layer"),
        "slot_layout": _env("MTPLX_HY3_SLOT_LAYOUT", "component-banks"),
        "banked_codec": _env("MTPLX_HY3_BANKED_CODEC", "none"),
        # "none" sentinel: pre-quantized-resident checkpoints (e.g. oq2e's q8)
        # must NOT set proj_quant — the loader raises when it matches nothing.
        "proj_quant": (
            None
            if _env("MTPLX_HY3_PROJ_QUANT", "q4") == "none"
            else _env("MTPLX_HY3_PROJ_QUANT", "q4")
        ),
        "expert_integrity": _env("MTPLX_HY3_EXPERT_INTEGRITY", "headers-only"),
        "split_route_release": _env("MTPLX_HY3_SPLIT_ROUTE_RELEASE", "deferred"),
        "deferred_pin_release": True,
        "hy3_router_kernel": _env("MTPLX_HY3_ROUTER_KERNEL", "mpp-r1-fused-r2"),
    }

    # Fallback mode: MTPLX_HY3_ISLAND_LAYER_COUNT (e.g. 79) pins the N worst-
    # streaming layers as dense islands via the model's measured pin order --
    # the docs' fully-resident reference. It is mutually exclusive with an
    # explicit island_layers list, so clear the explicit split + mmap band when
    # it is set.
    island_count_raw = os.environ.get("MTPLX_HY3_ISLAND_LAYER_COUNT")
    if island_count_raw:
        overrides["island_layer_count"] = int(island_count_raw)
        overrides["island_layers"] = ""
        overrides["mmap_island_layers"] = ""
        overrides["banked_manifest"] = ""
        return overrides

    # Champion mode: explicit island split + banked mmap band.
    overrides["island_layers"] = _env(
        "MTPLX_HY3_ISLANDS", "1-35,44,46,49-51,55-58,60,63-76,79"
    )
    overrides["mmap_island_layers"] = _env(
        "MTPLX_HY3_MMAP_ISLANDS", "38,45,47-48,52-54,59,61-62,77-78"
    )
    # A banked manifest is required by ExpertStreamingConfig whenever the mmap
    # band is non-empty; drop it (and the manifest) if the operator cleared it.
    if overrides["mmap_island_layers"].strip():
        overrides["banked_manifest"] = _default_banked_manifest(model_root)
    else:
        overrides["banked_manifest"] = ""
    return overrides


# ---------------------------------------------------------------------------
# Runtime singleton state.
# ---------------------------------------------------------------------------


class _State:
    """Everything ``completion`` needs, injectable for offline tests."""

    def __init__(
        self,
        *,
        runtime: Any,
        tokenizer: Any,
        generate_mtpk: Callable[..., Any],
        sampler_factory: Callable[..., Any],
        stop_token_ids: set[int],
        speculative_depth: int = SPECULATIVE_DEPTH,
        verify_strategy: str = VERIFY_STRATEGY,
        draft_core: str = DRAFT_CORE,
    ) -> None:
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.generate_mtpk = generate_mtpk
        self.sampler_factory = sampler_factory
        self.stop_token_ids = set(int(t) for t in stop_token_ids)
        self.speculative_depth = int(speculative_depth)
        self.verify_strategy = verify_strategy
        self.draft_core = draft_core


_STATE: _State | None = None
_STATE_LOCK = threading.Lock()


def _import_benchmark_module() -> Any:
    """Import ``scripts/benchmark_q2_mtp_depth_matrix.py`` by file path.

    ``scripts`` is not a package, so load it explicitly. The repo root is put
    on ``sys.path`` first so the module's own ``import mtplx.*`` works.
    """

    import importlib.util
    import sys

    # The campaign venv ships a PEP660 editable install of `mtplx` whose
    # sys.meta_path finder points at a DIFFERENT (stale) worktree lacking newer
    # submodules (e.g. mtplx.benchmarks.resource_telemetry). That finder wins
    # over sys.path, so inserting the eval worktree alone is not enough: drop the
    # mtplx package finder (keep the native-extension finder), force this
    # worktree to the front of sys.path, and evict any stale cached mtplx so
    # `import mtplx.*` resolves from THIS worktree.
    repo = str(_REPO_ROOT)
    sys.meta_path[:] = [
        f
        for f in sys.meta_path
        if not (
            type(f).__module__.startswith("__editable___mtplx_")
            and "native" not in type(f).__module__
        )
    ]
    while repo in sys.path:
        sys.path.remove(repo)
    sys.path.insert(0, repo)
    for _name in [n for n in list(sys.modules) if n == "mtplx" or n.startswith("mtplx.")]:
        _file = getattr(sys.modules.get(_name), "__file__", None) or ""
        if not _file.startswith(repo):
            del sys.modules[_name]
    module_path = _REPO_ROOT / "scripts" / "benchmark_q2_mtp_depth_matrix.py"
    spec = importlib.util.spec_from_file_location(
        "mtplx_benchmark_q2_mtp_depth_matrix", str(module_path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_state() -> _State:
    """Load the hy3 q2 runtime ONCE using the campaign's own config builder."""

    bench = _import_benchmark_module()
    apis = bench._default_apis()  # SamplerConfig, load, generate_mtpk, ExpertStreamingConfig

    options = {
        **bench.DEFAULT_RUNTIME_OPTIONS,
        **_champion_overrides(),
        "trace_routes": False,
    }
    config = bench._runtime_config(apis, MODEL_KEY, options)

    model_root = _default_model_root()
    runtime = apis.load(
        model_root,
        mtp=True,
        expert_streaming_config=config,
        expert_manifest=_default_manifest(model_root),
        mtp_artifacts=_default_mtp_artifacts(),
        mtp_precision=MTP_PRECISION,
    )

    from mtplx.generation import _default_stop_tokens  # local: needs mlx

    stop_ids = set(_default_stop_tokens(runtime.tokenizer))
    if not stop_ids:
        stop_ids = {_FALLBACK_EOS}

    return _State(
        runtime=runtime,
        tokenizer=runtime.tokenizer,
        generate_mtpk=apis.generate_mtpk,
        sampler_factory=apis.sampler_factory,
        stop_token_ids=stop_ids,
    )


def _ensure_state() -> _State:
    global _STATE
    if _STATE is not None:
        return _STATE
    with _STATE_LOCK:
        if _STATE is None:
            _STATE = _build_state()
        return _STATE


# ---------------------------------------------------------------------------
# Prompt construction / tokenization (pure, tokenizer-injected -> testable).
# ---------------------------------------------------------------------------


def _coerce_token_ids(encoded: Any) -> list[int]:
    """Normalize any tokenizer output shape to a flat token-id list.

    Mirrors ``mtplx.server.openai._coerce_token_ids`` so HF slow/fast
    tokenizers and mlx-lm's ``TokenizerWrapper`` all normalize identically.
    """

    if encoded is None:
        return []
    if hasattr(encoded, "ids"):
        return [int(t) for t in encoded.ids]
    if hasattr(encoded, "tolist"):
        return _coerce_token_ids(encoded.tolist())
    if isinstance(encoded, dict):
        if "input_ids" in encoded:
            return _coerce_token_ids(encoded["input_ids"])
        return []
    if isinstance(encoded, (list, tuple)):
        tokens: list[int] = []
        for item in encoded:
            if isinstance(item, int):
                tokens.append(int(item))
            elif hasattr(item, "ids") or hasattr(item, "tolist") or isinstance(
                item, (list, tuple, dict)
            ):
                tokens.extend(_coerce_token_ids(item))
            else:
                tokens.append(int(item))
        return tokens
    return [int(encoded)]


def _raw_prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    """Raw-prompt path: the completions endpoint wraps ``prompt`` into a single
    user message. Recover the concatenated text content verbatim."""

    parts: list[str] = []
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
                    parts.append(chunk["text"])
    return "".join(parts)


def _render_chat_text(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
    reasoning_effort: str = REASONING_EFFORT,
) -> str:
    """Apply the model's chat template to ``messages`` -> a prompt string.

    Passes ``reasoning_effort`` (hy3's real thinking control). Falls back to a
    plain render if a tokenizer rejects the kwarg.
    """

    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    if not isinstance(rendered, str):
        raise TypeError("apply_chat_template did not return a string with tokenize=False")
    return rendered


def render_prompt_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    text_completion: bool,
    add_generation_prompt: bool = True,
    reasoning_effort: str = REASONING_EFFORT,
) -> list[int]:
    """Return prompt token ids for either the chat or raw-completion path."""

    if text_completion:
        # Base-model continuation: tokenize the raw prompt verbatim (with the
        # tokenizer's default special-token handling, matching the server's
        # ``_encode_plain_text``).
        raw = _raw_prompt_from_messages(messages)
        return _coerce_token_ids(tokenizer.encode(raw))

    rendered = _render_chat_text(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
        reasoning_effort=reasoning_effort,
    )
    # The template already emits BOS/special sentinels, so encode without
    # adding them again (matches the server's ``_encode_rendered_chat_text``).
    try:
        return _coerce_token_ids(tokenizer.encode(rendered, add_special_tokens=False))
    except TypeError:
        return _coerce_token_ids(tokenizer.encode(rendered))


# ---------------------------------------------------------------------------
# Sampling / decode helpers.
# ---------------------------------------------------------------------------


def _build_sampler(sampler_factory: Callable[..., Any], optional_params: dict[str, Any]) -> Any:
    op = optional_params or {}

    def _get(name: str, default: Any) -> Any:
        value = op.get(name)
        return default if value is None else value

    return sampler_factory(
        temperature=float(_get("temperature", DEFAULT_TEMPERATURE)),
        top_p=float(_get("top_p", DEFAULT_TOP_P)),
        top_k=int(_get("top_k", DEFAULT_TOP_K)),
        presence_penalty=float(_get("presence_penalty", 0.0)),
        frequency_penalty=float(_get("frequency_penalty", 0.0)),
    )


def _strip_terminal_stop(tokens: list[int], stop_token_ids: set[int]) -> list[int]:
    stripped = list(tokens)
    while stripped and int(stripped[-1]) in stop_token_ids:
        stripped.pop()
    return stripped


def _apply_string_stops(text: str, stop: Any) -> tuple[str, bool]:
    """Truncate ``text`` at the earliest requested stop string.

    Returns (text, truncated). ``generate_mtpk`` stops on token ids only, so
    honor OpenAI-style string ``stop`` here on the decoded text.
    """

    if not stop:
        return text, False
    stops: Iterable[str]
    stops = [stop] if isinstance(stop, str) else [s for s in stop if isinstance(s, str)]
    cut = None
    for needle in stops:
        if not needle:
            continue
        index = text.find(needle)
        if index != -1 and (cut is None or index < cut):
            cut = index
    if cut is None:
        return text, False
    return text[:cut], True


def _extract_text(out: Any, tokenizer: Any, stop_token_ids: set[int]) -> tuple[str, list[int]]:
    """Return (completion_text, completion_token_ids_without_terminal_stops)."""

    tokens = [int(t) for t in getattr(out, "tokens", []) or []]
    stripped = _strip_terminal_stop(tokens, stop_token_ids)
    text = getattr(out, "text", None)
    if not isinstance(text, str):
        text = tokenizer.decode(stripped)
    return text, stripped


def _finish_reason(out: Any, *, completion_tokens: int, max_tokens: int, string_stopped: bool) -> str:
    if string_stopped:
        return "stop"
    reason = getattr(out, "finish_reason", None)
    if reason in ("stop", "length"):
        return reason
    return "length" if completion_tokens >= max_tokens else "stop"


def _shape_response(
    model_response: Any,
    *,
    model: str | None,
    text: str,
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Any:
    """Populate the litellm-provided ``ModelResponse`` (OpenAI chat schema)."""

    choice = model_response.choices[0]
    choice.message.content = text
    choice.message.role = "assistant"
    choice.finish_reason = finish_reason
    if model:
        model_response.model = model
    model_response.usage = Usage(
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        total_tokens=int(prompt_tokens) + int(completion_tokens),
    )
    return model_response


# ---------------------------------------------------------------------------
# The CustomLLM handler.
# ---------------------------------------------------------------------------


class Hy3StreamedLLM(CustomLLM):
    """OpenAI-compatible hy3 q2 handler backed by ``generate_mtpk``."""

    def _run(
        self,
        *,
        model: str | None,
        messages: list[dict[str, Any]],
        model_response: Any,
        optional_params: dict[str, Any] | None,
        litellm_params: dict[str, Any] | None,
    ) -> Any:
        state = _ensure_state()
        optional_params = optional_params or {}
        litellm_params = litellm_params or {}

        text_completion = bool(litellm_params.get("text_completion"))
        prompt_ids = render_prompt_ids(
            state.tokenizer, messages, text_completion=text_completion
        )

        sampler = _build_sampler(state.sampler_factory, optional_params)
        max_tokens = int(optional_params.get("max_tokens") or DEFAULT_MAX_TOKENS)
        seed_value = optional_params.get("seed")
        seed = int(seed_value) if seed_value is not None else DEFAULT_SEED

        out = state.generate_mtpk(
            state.runtime,
            list(prompt_ids),
            max_tokens=max_tokens,
            sampler=sampler,
            speculative_depth=state.speculative_depth,
            seed=seed,
            stop_token_ids=set(state.stop_token_ids),
            mtp_hidden_variant="post_norm",
            mtp_cache_policy="persistent",
            mtp_history_policy="committed",
            verify_strategy=state.verify_strategy,
            draft_core=state.draft_core,
        )

        text, completion_ids = _extract_text(out, state.tokenizer, state.stop_token_ids)
        text, string_stopped = _apply_string_stops(text, optional_params.get("stop"))
        finish_reason = _finish_reason(
            out,
            completion_tokens=len(completion_ids),
            max_tokens=max_tokens,
            string_stopped=string_stopped,
        )
        return _shape_response(
            model_response,
            model=model,
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(completion_ids),
        )

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str = "",
        custom_prompt_dict: dict | None = None,
        model_response: Any = None,
        print_verbose: Callable | None = None,
        encoding: Any = None,
        api_key: Any = None,
        logging_obj: Any = None,
        optional_params: dict | None = None,
        acompletion: Any = None,
        litellm_params: Any = None,
        logger_fn: Any = None,
        headers: dict | None = None,
        timeout: Any = None,
        client: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self._run(
            model=model,
            messages=messages,
            model_response=model_response,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str = "",
        custom_prompt_dict: dict | None = None,
        model_response: Any = None,
        print_verbose: Callable | None = None,
        encoding: Any = None,
        api_key: Any = None,
        logging_obj: Any = None,
        optional_params: dict | None = None,
        acompletion: Any = None,
        litellm_params: Any = None,
        logger_fn: Any = None,
        headers: dict | None = None,
        timeout: Any = None,
        client: Any = None,
        **kwargs: Any,
    ) -> Any:
        # The generate path is synchronous and holds the MLX runtime; run it
        # inline. LiteLLM awaits this coroutine, so returning the ModelResponse
        # directly is correct (the proxy serializes requests anyway).
        return self._run(
            model=model,
            messages=messages,
            model_response=model_response,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )


# Module-level singleton referenced by config.yaml as ``handler.instance``.
instance = Hy3StreamedLLM()
