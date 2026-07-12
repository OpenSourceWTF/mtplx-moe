#!/usr/bin/env python3
"""Mid-form serve-path gate for NAX m4 kernel candidates.

This is the cheap gate from OVERNIGHT_KERNEL_CAMPAIGN_20260613.md. It runs a
real streamed OpenAI-compatible request against either an existing MTPLX server
or a fresh branch server per kernel implementation. The point is to catch the
co-residency regime that isolated qmm microbenches miss.

Typical use:

  python scripts/midform_gate.py --start-server --impls legacy,auto --max-tokens 768

The script records full JSON plus generated text under
outputs/kernel-campaign-20260613/midform-gate/<run-id>/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
DEFAULT_MODEL_ID = "qwen3.6-27b-mtplx-optimized-speed"
DEFAULT_PROMPT = """Create a single-file HTML5 Canvas flappy bird game.
All visuals drawn procedurally. Animated bird with up-stroke and down-stroke
wing shapes, body tilt, feather particles, shaded pipes, parallax background,
start screen, game-over screen, localStorage best score, and delta-time
physics. Think carefully, then write the complete file."""


def _now_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _http_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _wait_for_health(
    base_url: str,
    *,
    api_key: str | None,
    timeout_s: float,
    proc: subprocess.Popen[str] | None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited early with rc={proc.returncode}: {last_error}")
        try:
            health = _http_json(
                "GET",
                base_url.rstrip("/") + "/health",
                api_key=api_key,
                timeout_s=10.0,
            )
            if health:
                return health
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2.0)
    raise TimeoutError(f"server did not become healthy within {timeout_s:.1f}s: {last_error}")


def _extract_delta_text(chunk: dict[str, Any]) -> tuple[str, str]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        content = delta.get("content")
        reasoning = delta.get("reasoning_content")
        if isinstance(content, str):
            content_parts.append(content)
        if isinstance(reasoning, str):
            reasoning_parts.append(reasoning)
    return "".join(content_parts), "".join(reasoning_parts)


def _rate(times: list[float], *, first: bool, window: int) -> float | None:
    if len(times) < 2:
        return None
    subset = times[:window] if first else times[-window:]
    if len(subset) < 2:
        return None
    elapsed = subset[-1] - subset[0]
    if elapsed <= 0:
        return None
    return (len(subset) - 1) / elapsed


def _per_call(value: Any, calls: int) -> float | None:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        return None
    if calls <= 0:
        return None
    return numeric / calls


def _acceptance_by_depth(stats: dict[str, Any]) -> list[float | None]:
    accepted = stats.get("accepted_by_depth") or []
    drafted = stats.get("drafted_by_depth") or []
    rates: list[float | None] = []
    for index, accepted_value in enumerate(accepted):
        try:
            drafted_value = drafted[index]
        except IndexError:
            rates.append(None)
            continue
        rates.append((accepted_value / drafted_value) if drafted_value else None)
    return rates


def _stream_chat(
    *,
    base_url: str,
    api_key: str | None,
    model_id: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    heartbeat_s: float,
) -> dict[str, Any]:
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": True,
        "metadata": {"client": "kernel_midform_gate", "seed": seed},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-MTPLX-Client": "kernel-midform-gate",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers=headers,
    )

    started_wall = time.time()
    started = time.perf_counter()
    last_heartbeat = started
    first_token_s: float | None = None
    token_times: list[float] = []
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    finish_reasons: list[str] = []
    chunks = 0

    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            chunk = json.loads(payload)
            chunks += 1
            content_delta, reasoning_delta = _extract_delta_text(chunk)
            if content_delta or reasoning_delta:
                now = time.perf_counter()
                token_times.append(now)
                if first_token_s is None:
                    first_token_s = now
                if content_delta:
                    content_parts.append(content_delta)
                if reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                if heartbeat_s > 0 and now - last_heartbeat >= heartbeat_s:
                    elapsed = now - started
                    print(
                        "heartbeat "
                        f"elapsed_s={elapsed:.1f} chunks={len(token_times)} "
                        f"client_rate={len(token_times) / elapsed if elapsed > 0 else 0.0:.2f}",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_heartbeat = now
            for choice in chunk.get("choices") or []:
                reason = choice.get("finish_reason")
                if isinstance(reason, str) and reason:
                    finish_reasons.append(reason)
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            if isinstance(chunk.get("mtplx_stats"), dict):
                stats = chunk["mtplx_stats"]

    finished = time.perf_counter()
    wall_s = finished - started
    completion_tokens = int(
        stats.get("completion_tokens") or usage.get("completion_tokens") or len(token_times)
    )
    verify_calls = int(stats.get("verify_calls") or 0)
    return {
        "started_wall": started_wall,
        "wall_s": wall_s,
        "ttft_client_s": None if first_token_s is None else first_token_s - started,
        "chunks_with_text": len(token_times),
        "raw_sse_chunks": chunks,
        "client_chunk_rate": len(token_times) / wall_s if wall_s > 0 else 0.0,
        "client_first_32_chunk_rate": _rate(token_times, first=True, window=32),
        "client_first_128_chunk_rate": _rate(token_times, first=True, window=128),
        "client_last_128_chunk_rate": _rate(token_times, first=False, window=128),
        "finish_reasons": finish_reasons,
        "usage": usage,
        "mtplx_stats": stats,
        "derived": {
            "completion_tokens": completion_tokens,
            "verify_calls": verify_calls,
            "tokens_per_verify_call": (
                completion_tokens / verify_calls if verify_calls > 0 else None
            ),
            "verify_hidden_eval_s_per_call": _per_call(
                stats.get("verify_hidden_eval_time_s"), verify_calls
            ),
            "verify_eval_s_per_call": _per_call(stats.get("verify_eval_time_s"), verify_calls),
            "verify_forward_s_per_call": _per_call(
                stats.get("verify_forward_time_s"), verify_calls
            ),
            "draft_s_per_call": _per_call(stats.get("draft_time_s"), verify_calls),
            "acceptance_by_depth": _acceptance_by_depth(stats),
        },
        "reasoning_text": "".join(reasoning_parts),
        "content_text": "".join(content_parts),
    }


def _start_server(
    *,
    python_exe: str,
    model: str,
    model_id: str,
    host: str,
    port: int,
    profile: str,
    warmup_tokens: int,
    impl: str,
    log_path: Path,
    extra_env: dict[str, str],
) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env.update(extra_env)
    env["PYTHONPATH"] = str(ROOT)
    if impl == "stock":
        # Baseline arm: verify kernels fully off (stock mx.quantized_matmul).
        env["MTPLX_NAX_VERIFY"] = "0"
        env["MTPLX_NAX_M4_IMPL"] = "legacy"
    else:
        env["MTPLX_NAX_VERIFY"] = "1"
        env["MTPLX_NAX_M4_IMPL"] = impl
    command = [
        python_exe,
        "-m",
        "mtplx.cli",
        "serve",
        "--model",
        model,
        "--model-id",
        model_id,
        "--profile",
        profile,
        "--host",
        host,
        "--port",
        str(port),
        "--generation-mode",
        "mtp",
        "--depth",
        "3",
        "--warmup-tokens",
        str(warmup_tokens),
        "--no-stats-footer",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    # Keep the file descriptor owned by the child; parent does not need it open.
    log_handle.close()
    return proc


def _stop_server(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=20)


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    if args.prompt:
        return str(args.prompt)
    return DEFAULT_PROMPT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("MTPLX_MODEL", DEFAULT_MODEL))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--profile", default="turbo")
    parser.add_argument("--impls", default="legacy,auto")
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:18083")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--port-step", type=int, default=1)
    parser.add_argument("--server-timeout-s", type=float, default=1200.0)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--heartbeat-s", type=float, default=30.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "kernel-campaign-20260613" / "midform-gate"),
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra KEY=VALUE environment override for launched servers.",
    )
    return parser.parse_args(argv)


def _parse_env(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key] = value
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompt = _load_prompt(args)
    impls = [item.strip() for item in str(args.impls).split(",") if item.strip()]
    if not impls:
        raise SystemExit("--impls must name at least one implementation")
    run_id = args.run_id or _now_run_id()
    run_dir = Path(args.output_dir).expanduser() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    extra_env = _parse_env(args.env)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "root": str(ROOT),
        "args": vars(args),
        "impls": {},
    }

    for index, impl in enumerate(impls):
        port = int(args.port) + index * int(args.port_step)
        base_url = f"http://{args.host}:{port}" if args.start_server else args.base_url
        proc: subprocess.Popen[str] | None = None
        impl_dir = run_dir / impl
        impl_dir.mkdir(parents=True, exist_ok=True)
        print(f"== {impl}: base_url={base_url} start_server={args.start_server}", flush=True)
        try:
            if args.start_server:
                proc = _start_server(
                    python_exe=args.python,
                    model=args.model,
                    model_id=args.model_id,
                    host=args.host,
                    port=port,
                    profile=args.profile,
                    warmup_tokens=args.warmup_tokens,
                    impl=impl,
                    log_path=impl_dir / "server.log",
                    extra_env=extra_env,
                )
            health = _wait_for_health(
                base_url,
                api_key=args.api_key,
                timeout_s=args.server_timeout_s,
                proc=proc,
            )
            result = _stream_chat(
                base_url=base_url,
                api_key=args.api_key,
                model_id=args.model_id,
                prompt=prompt,
                max_tokens=args.max_tokens,
                seed=args.seed,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                heartbeat_s=args.heartbeat_s,
            )
            result["health_before"] = health
            result["impl"] = impl
            (impl_dir / "result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            (impl_dir / "content.txt").write_text(result["content_text"], encoding="utf-8")
            (impl_dir / "reasoning.txt").write_text(
                result["reasoning_text"], encoding="utf-8"
            )
            stats = result.get("mtplx_stats") or {}
            derived = result.get("derived") or {}
            summary["impls"][impl] = {
                "ok": True,
                "decode_tok_s": stats.get("decode_tok_s"),
                "request_tok_s": stats.get("request_tok_s"),
                "completion_tokens": derived.get("completion_tokens"),
                "verify_calls": derived.get("verify_calls"),
                "tokens_per_verify_call": derived.get("tokens_per_verify_call"),
                "verify_hidden_eval_s_per_call": derived.get(
                    "verify_hidden_eval_s_per_call"
                ),
                "sliding_first_128": stats.get("sliding_decode_tok_s_first_128"),
                "sliding_last_128": stats.get("sliding_decode_tok_s_last_128"),
                "client_first_128_chunk_rate": result.get("client_first_128_chunk_rate"),
                "client_last_128_chunk_rate": result.get("client_last_128_chunk_rate"),
                "acceptance_by_depth": derived.get("acceptance_by_depth"),
                "finish_reasons": result.get("finish_reasons"),
            }
            print(
                json.dumps(summary["impls"][impl], indent=2, sort_keys=True),
                flush=True,
            )
        except Exception as exc:
            summary["impls"][impl] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"{impl} failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        finally:
            _stop_server(proc)

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0 if all(item.get("ok") for item in summary["impls"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
