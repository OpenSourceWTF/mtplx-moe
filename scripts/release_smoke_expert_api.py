#!/usr/bin/env python3
"""Exercise the installed expert-serving API through OpenAI and LiteLLM."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping
from typing import Any, TypeVar
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


EXPECTED_REVISION = "d33ce31c0605fc571c374cdf0aa0f085ec50ff88"
EXPECTED_BANK_SHA256 = (
    "c72fb8c0a66020439f4a78591ab9a79d8da3d38412635a531d604ffbf0d2e7d4"
)
EXPECTED_PROFILE_EVIDENCE = {
    "hy3-oq2e-64": "14c8b57fff358bee3da2d10968a855b955b86847",
    "hy3-oq2e-88": "191ed9aa362e645f48f1a105a6ec024ea4fd5cf4",
    "hy3-oq2e-96": "191ed9aa362e645f48f1a105a6ec024ea4fd5cf4",
}
_T = TypeVar("_T")


class SmokeFailure(RuntimeError):
    """A release-smoke assertion with a concise operator-facing message."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a JSON object")
    return value


def _nonempty_text(value: object, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} was empty")
    return value


def _hex_digest(value: object, length: int, name: str) -> str:
    text = _nonempty_text(value, name)
    _require(
        len(text) == length and re.fullmatch(r"[0-9a-f]+", text) is not None,
        f"{name} must be a {length}-character lowercase hexadecimal value",
    )
    return text


def _redacted_error(exc: BaseException, api_key: str) -> str:
    message = str(exc)
    if api_key:
        message = message.replace(api_key, "<redacted>")
    return re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+",
        r"\1<redacted>",
        message,
    )


def _run_step(
    label: str,
    operation: Callable[[], _T],
    *,
    api_key: str,
) -> _T:
    try:
        return operation()
    except SmokeFailure:
        raise
    except BaseException as exc:
        detail = _redacted_error(exc, api_key)
        raise SmokeFailure(
            f"{label} failed ({type(exc).__name__}): {detail}"
        ) from None


def _health_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    _require(
        parsed.scheme in {"http", "https"} and bool(parsed.netloc),
        "--base-url must be an absolute HTTP(S) URL",
    )
    _require(
        not parsed.query and not parsed.fragment,
        "--base-url must not contain a query string or fragment",
    )
    path = parsed.path.rstrip("/")
    _require(
        path.endswith("/v1"),
        "--base-url must end in the /v1 API path",
    )
    root_path = path.removesuffix("/v1").rstrip("/")
    health_path = f"{root_path}/health" if root_path else "/health"
    return urlunsplit(
        (parsed.scheme, parsed.netloc, health_path, "", "")
    )


def _fetch_health(base_url: str, api_key: str) -> dict[str, Any]:
    request = Request(
        _health_url(base_url),
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310
        payload = json.load(response)
    _require(isinstance(payload, dict), "/health did not return a JSON object")
    return payload


def _model_dump(value: object) -> Mapping[str, Any]:
    dump = getattr(value, "model_dump", None)
    _require(callable(dump), "client response does not support model_dump()")
    return _mapping(dump(), "client response")


def _assert_request_ar(response: object, name: str) -> None:
    payload = _model_dump(response)
    stats = _mapping(payload.get("mtplx_stats"), f"{name}.mtplx_stats")
    _require(
        stats.get("generation_mode") == "ar",
        f"{name} did not report generation_mode='ar'",
    )


def _profile_evidence(
    health: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        health.get("generation_mode") == "ar",
        "/health generation_mode must be 'ar'",
    )
    _require(
        health.get("available_generation_modes") == ["ar"],
        "/health must advertise only the installed AR route",
    )
    profile = _mapping(health.get("expert_profile"), "expert_profile")
    admission = _mapping(health.get("expert_admission"), "expert_admission")
    streaming = _mapping(health.get("expert_streaming"), "expert_streaming")

    profile_name = _nonempty_text(profile.get("name"), "expert_profile.name")
    _require(
        profile_name in EXPECTED_PROFILE_EVIDENCE,
        f"unexpected expert profile {profile_name!r}",
    )
    model_key = _nonempty_text(
        profile.get("model_key"), "expert_profile.model_key"
    )
    _require(
        model_key == "hy3-expert-oq2e",
        f"unexpected expert model key {model_key!r}",
    )
    backend = _nonempty_text(
        profile.get("backend"), "expert_profile.backend"
    )
    _require(
        profile.get("generation_mode") == "ar",
        "expert_profile.generation_mode must be 'ar'",
    )

    revision = _hex_digest(
        admission.get("revision"), 40, "expert_admission.revision"
    )
    _require(
        revision == EXPECTED_REVISION,
        f"unexpected admitted revision {revision!r}",
    )
    manifest_sha256 = _hex_digest(
        admission.get("manifest_sha256"),
        64,
        "expert_admission.manifest_sha256",
    )
    bank_sha256 = _hex_digest(
        admission.get("bank_sha256"),
        64,
        "expert_admission.bank_sha256",
    )
    _require(
        bank_sha256 == EXPECTED_BANK_SHA256,
        f"unexpected admitted bank SHA-256 {bank_sha256!r}",
    )
    _require(
        streaming.get("model_key") == model_key,
        "expert_streaming.model_key disagrees with expert_profile.model_key",
    )
    _require(
        streaming.get("manifest_sha256") == manifest_sha256,
        "expert_streaming manifest digest disagrees with expert_admission",
    )
    _mapping(streaming.get("memory_plan"), "expert_streaming.memory_plan")
    cache_by_phase = _mapping(
        streaming.get("cache_by_phase"), "expert_streaming.cache_by_phase"
    )
    _mapping(cache_by_phase.get("decode"), "expert_streaming decode route")

    customized = profile.get("customized") is True
    evidence_commit = profile.get("evidence_commit")
    if customized:
        _require(
            evidence_commit is None,
            "customized profile must not claim promoted evidence",
        )
        _require(
            bool(_mapping(profile.get("effective"), "expert_profile.effective")),
            "customized profile must report its effective configuration",
        )
    else:
        expected_evidence = EXPECTED_PROFILE_EVIDENCE[profile_name]
        _require(
            evidence_commit == expected_evidence,
            "uncustomized profile evidence commit does not match its "
            "promoted profile",
        )

    return {
        "profile": profile_name,
        "customized": customized,
        "evidence_commit": evidence_commit,
        "backend": backend,
        "generation_mode": "ar",
        "revision": revision,
        "manifest_sha256": manifest_sha256,
        "bank_sha256": bank_sha256,
    }


def _assert_health_consistent(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    for field in (
        "profile",
        "customized",
        "evidence_commit",
        "backend",
        "generation_mode",
        "revision",
        "manifest_sha256",
        "bank_sha256",
    ):
        _require(
            before.get(field) == after.get(field),
            f"health field {field!r} changed across API calls",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test an MTPLX expert-serving release candidate."
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible API base URL ending in /v1.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Exact served model ID returned by /v1/models.",
    )
    parser.add_argument("--api-key", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        from openai import OpenAI
        import litellm
    except ImportError as exc:
        raise SmokeFailure(
            "release smoke requires the openai and litellm packages"
        ) from exc

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    models = _run_step(
        "OpenAI model listing",
        client.models.list,
        api_key=args.api_key,
    )
    model_ids = [
        item.id
        for item in getattr(models, "data", [])
        if isinstance(getattr(item, "id", None), str)
    ]
    _require(bool(model_ids), "/v1/models returned no models")
    _require(
        args.model in model_ids,
        f"--model {args.model!r} is absent from /v1/models",
    )

    nonstream = _run_step(
        "OpenAI non-streaming Chat Completions",
        lambda: client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply with exactly OK"}],
            temperature=0,
        ),
        api_key=args.api_key,
    )
    _nonempty_text(
        nonstream.choices[0].message.content,
        "OpenAI non-streaming response text",
    )
    _assert_request_ar(nonstream, "OpenAI non-streaming response")

    stream = _run_step(
        "OpenAI streaming Chat Completions",
        lambda: client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply with exactly OK"}],
            temperature=0,
            stream=True,
        ),
        api_key=args.api_key,
    )
    stream_parts: list[str] = []
    stream_final: object | None = None
    for chunk in stream:
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if isinstance(content, str):
                stream_parts.append(content)
        payload = _model_dump(chunk)
        if isinstance(payload.get("mtplx_stats"), Mapping):
            stream_final = chunk
    _nonempty_text("".join(stream_parts), "OpenAI streaming response text")
    _require(stream_final is not None, "stream omitted final mtplx_stats")
    _assert_request_ar(stream_final, "OpenAI streaming response")

    litellm_response = _run_step(
        "LiteLLM Chat Completions",
        lambda: litellm.completion(
            model=f"openai/{args.model}",
            api_base=args.base_url,
            api_key=args.api_key,
            messages=[{"role": "user", "content": "Reply with exactly OK"}],
            temperature=0,
        ),
        api_key=args.api_key,
    )
    _nonempty_text(
        litellm_response.choices[0].message.content,
        "LiteLLM response text",
    )

    first_health = _run_step(
        "health after short requests",
        lambda: _profile_evidence(
            _fetch_health(args.base_url, args.api_key)
        ),
        api_key=args.api_key,
    )

    long_response = _run_step(
        "OpenAI long Chat Completions",
        lambda: client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": "Summarize in one sentence: "
                    + ("MTPLX streams experts from SSD. " * 512),
                }
            ],
            max_tokens=32,
            temperature=0,
        ),
        api_key=args.api_key,
    )
    _nonempty_text(
        long_response.choices[0].message.content,
        "OpenAI long response text",
    )
    _assert_request_ar(long_response, "OpenAI long response")

    final_health = _run_step(
        "health after long request",
        lambda: _profile_evidence(
            _fetch_health(args.base_url, args.api_key)
        ),
        api_key=args.api_key,
    )
    _assert_health_consistent(first_health, final_health)

    print(
        json.dumps(
            {
                "ok": True,
                "model": args.model,
                **final_health,
                "requests": [
                    "openai-nonstream",
                    "openai-stream",
                    "litellm",
                    "openai-long",
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeFailure as exc:
        print(f"release smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
