"""Pi coding-agent integration helpers.

The public CLI uses this module to make ``mtplx start pi`` a real connection
flow: merge an MTPLX provider into Pi's ``models.json`` and then start the
OpenAI-compatible MTPLX server with matching settings.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PI_PROVIDER_ID = "mtplx"
PI_LOCAL_API_KEY = "mtplx-local"
PI_NPM_PACKAGE = "@earendil-works/pi-coding-agent"
PI_DEFAULT_CONTEXT_WINDOW = 131_072
PI_DEFAULT_MAX_TOKENS: int | None = None
# Pi serializes a 16,384 output ceiling for models whose metadata omits
# maxTokens. The extension strips exactly this value; any other cap is a
# deliberate client choice and must reach MTPLX intact.
PI_INJECTED_DEFAULT_MAX_TOKENS = 16_384
PI_REQUEST_POLICY_EXTENSION_NAME = "mtplx-request-policy.ts"
# Connection identity MTPLX must keep correct for the integration to work at
# all (ports move between launches). Everything else belongs to the user once
# they edit it (#282: silent clobber of user edits in models.json).
PI_OWNED_PROVIDER_CONNECTION_KEYS = ("baseUrl", "api", "apiKey", "authHeader")
PI_EXTENSION_MANAGED_MARKER = "MTPLX-managed"


def pi_install_command() -> str:
    return f"npm install -g {PI_NPM_PACKAGE}"


def pi_models_json_path(path: str | Path | None = None) -> Path:
    """Return Pi's custom models config path.

    ``MTPLX_PI_MODELS_JSON`` exists only for tests and power-user overrides.
    Normal users get Pi's documented ``~/.pi/agent/models.json`` path.
    """

    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get("MTPLX_PI_MODELS_JSON")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".pi" / "agent" / "models.json"


def pi_request_policy_extension_path(path: str | Path | None = None) -> Path:
    """Return the MTPLX-owned Pi extension next to ``models.json``."""

    return pi_models_json_path(path).parent / "extensions" / PI_REQUEST_POLICY_EXTENSION_NAME


def build_pi_request_policy_extension_source(
    model_id: str,
    *,
    uncapped: bool,
) -> str:
    """Build Pi's request/session bridge for the configured MTPLX model.

    Pi defaults omitted ``maxTokens`` metadata to 16,384 and serializes that
    default on every request. The extension removes only Pi's generated output
    ceiling for the exact MTPLX model while leaving explicit user caps alone.
    It also gives MTPLX Pi's real session id so prompt-cache reuse is stable.
    """

    model_literal = json.dumps(str(model_id))
    uncapped_literal = "true" if uncapped else "false"
    return f"""// {PI_EXTENSION_MANAGED_MARKER} Pi extension. MTPLX keeps this file up to
// date on every sync. To take ownership (or disable it), edit it and delete
// this marker line: MTPLX never touches the file again once the marker and
// the mtplx identifiers below are gone from it.
const mtplxModelID = {model_literal};
const mtplxUncapped = {uncapped_literal};
const mtplxPiInjectedDefaultMaxTokens = {PI_INJECTED_DEFAULT_MAX_TOKENS};

export default function (pi: any) {{
  pi.on("before_provider_headers", (event: any, ctx: any) => {{
    const headers = event?.headers;
    if (!headers || typeof headers !== "object") return;
    const client = Object.entries(headers).find(
      ([key]) => key.toLowerCase() === "x-mtplx-client",
    )?.[1];
    if (client !== "pi") return;
    event.headers["x-mtplx-session-id"] = String(
      ctx.sessionManager.getSessionId(),
    );
  }});

  pi.on("before_provider_request", (event: any) => {{
    const payload = event?.payload;
    if (!mtplxUncapped || !payload || typeof payload !== "object") return;
    if (payload.model !== mtplxModelID) return;
    // Strip only Pi's serialized default ceiling; an explicit user cap (any
    // other value) is honored end to end.
    const request = {{ ...payload }};
    let changed = false;
    if (request.max_tokens === mtplxPiInjectedDefaultMaxTokens) {{
      delete request.max_tokens;
      changed = true;
    }}
    if (request.max_completion_tokens === mtplxPiInjectedDefaultMaxTokens) {{
      delete request.max_completion_tokens;
      changed = true;
    }}
    if (!changed) return;
    return request;
  }});
}}
"""


def write_pi_request_policy_extension(
    *,
    model_id: str,
    uncapped: bool,
    path: str | Path | None = None,
) -> Path:
    """Install the small Pi bridge owned by the MTPLX provider config."""

    extension_path = pi_request_policy_extension_path(path)
    source = build_pi_request_policy_extension_source(model_id, uncapped=uncapped)
    if extension_path.exists():
        try:
            current = extension_path.read_text(encoding="utf-8")
        except OSError:
            current = ""
        managed = (
            PI_EXTENSION_MANAGED_MARKER in current
            or "mtplxPiInjectedDefaultMaxTokens" in current
        )
        if not managed:
            # The user replaced the extension with their own content: it is
            # theirs now. Never overwrite a user-owned file (#282).
            return extension_path
        if current == source:
            return extension_path
    extension_path.parent.mkdir(parents=True, exist_ok=True)
    extension_path.write_text(source, encoding="utf-8")
    try:
        extension_path.chmod(0o600)
    except OSError:
        pass
    return extension_path


def pi_model_ref(model_id: str, *, provider_id: str = PI_PROVIDER_ID) -> str:
    return f"{provider_id}/{model_id}"


def pi_launch_command(model_id: str, *, provider_id: str = PI_PROVIDER_ID) -> str:
    return f"pi --model {pi_model_ref(model_id, provider_id=provider_id)}"


def launch_pi_in_terminal(command: str, *, model_ref: str | None = None) -> dict[str, Any]:
    """Open Pi in a macOS Terminal window/tab without blocking MTPLX.

    Pi is an interactive terminal client, so spawning it as a silent background
    process would be worse UX than doing nothing. Always try to open it: a
    false "already running" is much worse than an extra Pi tab. On non-macOS
    systems, return a clear fallback payload.
    """

    _ = model_ref  # kept for call-site clarity and future platform-specific launchers.
    if sys.platform != "darwin":
        return {
            "ok": False,
            "status": "unsupported_platform",
            "command": command,
            "error": "automatic Pi launch currently requires macOS Terminal",
        }
    script = "\n".join(
        [
            'tell application "Terminal"',
            "  activate",
            f"  do script {json.dumps(command)}",
            "end tell",
        ]
    )
    try:
        subprocess.Popen(
            ["osascript", "-e", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        return {"ok": False, "status": "launch_failed", "command": command, "error": str(exc)}
    return {"ok": True, "status": "launched", "command": command}


def build_pi_provider_config(
    *,
    base_url: str,
    model_id: str,
    model_name: str | None = None,
    api_key: str = PI_LOCAL_API_KEY,
    context_window: int = PI_DEFAULT_CONTEXT_WINDOW,
    max_tokens: int | None = PI_DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Build the Pi provider block MTPLX needs.

    Pi's OpenAI-compatible transport currently needs the Chat Completions API
    name, a dummy-or-real API key, and compatibility flags so it sends
    ``system`` instead of ``developer`` and ``max_tokens`` instead of the newer
    OpenAI field. The Qwen thinking format wires Pi's thinking-level picker to
    the server's ``enable_thinking``/``reasoning_effort`` request fields.
    """

    model_config: dict[str, Any] = {
        "id": str(model_id),
        "name": model_name or f"MTPLX {model_id}",
        "reasoning": True,
        # Pi's effort ladder is off/minimal/low/medium/high/xhigh/max; the
        # MTPLX vocabulary is low/medium/high/xhigh (mtplx/reasoning_effort.py)
        # and the server narrows to the loaded family's declared tiers.
        # "minimal": null hides Pi's duplicate below-low tier; "xhigh" must be
        # mapped to appear in Pi's picker at all (Qwen 3.8's top tier); "max"
        # stays unmapped, so hidden. Unmapped levels pass through verbatim.
        "thinkingLevelMap": {
            "minimal": None,
            "xhigh": "xhigh",
        },
        "input": ["text"],
        "contextWindow": int(context_window),
        "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
        },
    }
    # Pi requires output metadata and otherwise silently substitutes 16,384.
    # Advertise the real context ceiling; the MTPLX-owned request extension
    # omits the generated wire cap when the user did not explicitly request one.
    model_config["maxTokens"] = int(
        context_window if max_tokens is None else max_tokens
    )

    return {
        "baseUrl": str(base_url).rstrip("/"),
        "api": "openai-completions",
        "apiKey": str(api_key),
        "authHeader": True,
        "headers": {
            "x-mtplx-client": "pi",
        },
        # Pi 0.84.x with thinkingFormat "qwen" serializes exactly the fields
        # the MTPLX server accepts: top-level ``enable_thinking`` (true when a
        # thinking level is selected, false for Pi's "off" level) plus
        # ``reasoning_effort`` mapped through thinkingLevelMap
        # (pi-ai openai-completions buildParams). Pi's default level is
        # "medium" — the Qwen 3.8 family coding default.
        "compat": {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": True,
            "thinkingFormat": "qwen",
            "maxTokensField": "max_tokens",
        },
        "models": [model_config],
    }


def _backup_invalid_config(path: Path) -> Path:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.invalid-{stamp}.bak")
    counter = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.invalid-{stamp}-{counter}.bak")
        counter += 1
    path.replace(backup)
    return backup


def _fill_missing_deep(existing: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Existing user values win; defaults only fill gaps, recursively."""

    merged = dict(existing)
    for key, default_value in defaults.items():
        if key not in merged:
            merged[key] = default_value
        elif isinstance(merged[key], dict) and isinstance(default_value, dict):
            merged[key] = _fill_missing_deep(merged[key], default_value)
    return merged


def merge_pi_provider_config(
    existing_provider: Any,
    fresh: dict[str, Any],
) -> dict[str, Any]:
    """User-preserving merge of the MTPLX provider block (#282 clobber fix).

    MTPLX owns the connection identity (``baseUrl``/``api``/``apiKey``/
    ``authHeader`` and the ``x-mtplx-client`` header) because ports move
    between launches and the integration must keep working. Every other key
    the user edited wins: our values only fill missing keys, recursively.
    Model entries merge by ``id`` the same way, and user-added models or
    fields (``input: ["text", "image"]``, custom ``thinkingLevelMap``,
    explicit ``maxTokens``) survive a sync untouched.
    """

    if not isinstance(existing_provider, dict):
        return fresh
    merged = _fill_missing_deep(
        existing_provider,
        {
            key: value
            for key, value in fresh.items()
            if key not in ("models", "headers", "compat")
        },
    )
    for key in PI_OWNED_PROVIDER_CONNECTION_KEYS:
        if key in fresh:
            merged[key] = fresh[key]
    # ``compat`` is MTPLX's transport contract with Pi (which wire fields the
    # server supports), not a user preference: a stale block written by an
    # older MTPLX must not outlive the engine that wrote it. Receipt
    # 2026-08-28: a 2.9.x-era ``supportsReasoningEffort: false`` survived
    # every re-sync and silently killed Pi's effort dial after upgrade. Our
    # keys win; user-added extra compat keys still survive.
    existing_compat = (
        dict(existing_provider.get("compat"))
        if isinstance(existing_provider.get("compat"), dict)
        else {}
    )
    existing_compat.update(fresh.get("compat") or {})
    if existing_compat:
        merged["compat"] = existing_compat
    headers = {
        key: value
        for key, value in (
            existing_provider.get("headers") or {}
        ).items()
        if str(key).lower() != "x-mtplx-client"
    } if isinstance(existing_provider.get("headers"), dict) else {}
    headers.update(fresh.get("headers") or {})
    merged["headers"] = headers

    fresh_models = fresh.get("models") or []
    existing_models = existing_provider.get("models")
    if not isinstance(existing_models, list):
        merged["models"] = fresh_models
        return merged
    fresh_ids = {str(model.get("id")) for model in fresh_models}
    # Stale MTPLX-owned entries (our own previous model ids, always
    # "mtplx-"-prefixed) are pruned so switching models does not pile up
    # dead picker rows; user-added models never match the prefix and stay.
    result_models = [
        entry
        for entry in existing_models
        if not (
            isinstance(entry, dict)
            and str(entry.get("id", "")).startswith("mtplx-")
            and str(entry.get("id")) not in fresh_ids
        )
    ]
    for fresh_model in fresh_models:
        fresh_id = str(fresh_model.get("id"))
        for index, entry in enumerate(result_models):
            if isinstance(entry, dict) and str(entry.get("id")) == fresh_id:
                result_models[index] = _fill_missing_deep(entry, fresh_model)
                break
        else:
            result_models.append(fresh_model)
    merged["models"] = result_models
    return merged


def merge_pi_models_config(
    existing: dict[str, Any] | None,
    *,
    provider_config: dict[str, Any],
    provider_id: str = PI_PROVIDER_ID,
) -> dict[str, Any]:
    """Merge or create a Pi ``models.json`` payload.

    MTPLX owns only the ``providers.mtplx`` block, and inside it only the
    connection identity: user edits within the block are preserved via
    :func:`merge_pi_provider_config`. Other providers are untouched.
    """

    payload = dict(existing or {})
    providers = payload.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    else:
        providers = dict(providers)
    providers[str(provider_id)] = merge_pi_provider_config(
        providers.get(str(provider_id)),
        provider_config,
    )
    payload["providers"] = providers
    return payload


def write_pi_models_config(
    *,
    base_url: str,
    model_id: str,
    model_name: str | None = None,
    api_key: str = PI_LOCAL_API_KEY,
    path: str | Path | None = None,
    provider_id: str = PI_PROVIDER_ID,
    context_window: int = PI_DEFAULT_CONTEXT_WINDOW,
    max_tokens: int | None = PI_DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Write the MTPLX provider into Pi's config and return a handoff payload."""

    config_path = pi_models_json_path(path)
    backup_path: Path | None = None
    existing: dict[str, Any] | None = None
    if config_path.exists():
        try:
            parsed = json.loads(config_path.read_text(encoding="utf-8"))
            existing = parsed if isinstance(parsed, dict) else {}
        except (OSError, json.JSONDecodeError):
            backup_path = _backup_invalid_config(config_path)
            existing = {}

    provider_config = build_pi_provider_config(
        base_url=base_url,
        model_id=model_id,
        model_name=model_name,
        api_key=api_key,
        context_window=context_window,
        max_tokens=max_tokens,
    )
    merged = merge_pi_models_config(
        existing,
        provider_config=provider_config,
        provider_id=provider_id,
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    try:
        config_path.chmod(0o600)
    except OSError:
        pass
    request_policy_extension_path = write_pi_request_policy_extension(
        model_id=model_id,
        uncapped=max_tokens is None,
        path=config_path,
    )
    return {
        "config_path": str(config_path),
        "backup_path": str(backup_path) if backup_path is not None else None,
        "provider_id": provider_id,
        "base_url": provider_config["baseUrl"],
        "model_id": model_id,
        "model_ref": pi_model_ref(model_id, provider_id=provider_id),
        "launch_command": pi_launch_command(model_id, provider_id=provider_id),
        "api_key": api_key,
        "context_window": int(context_window),
        "max_tokens": None if max_tokens is None else int(max_tokens),
        "no_hidden_max_tokens": max_tokens is None,
        "request_policy_extension_path": str(request_policy_extension_path),
        "uncapped_request_policy": max_tokens is None,
        "written": True,
    }
