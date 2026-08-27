#!/usr/bin/env python3
"""Guarded resident-oQ4 smoke and exact 16K-in/1K-out harness."""

from __future__ import annotations

import argparse
from collections.abc import MutableMapping
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[1]
DEFAULT_MODEL = (
    Path.home() / ".mtplx/models/Vontra--Qwen3.8-Flash-Next-MLX-oQ4-MTP"
)
DEFAULT_PROMPT_FILE = (
    ROOT / "mtplx/benchmarks/prompts/qwen38_naturalistic_generation_patch.jsonl"
)
DEFAULT_CONTEXT_FILE = ROOT / "mtplx/generation.py"
DEFAULT_GUARD = WORKSPACE_ROOT / "bench/laguna/run_guarded.py"
DEFAULT_PLIST = Path.home() / "Library/LaunchAgents/com.tea.qwen.plist"
DEFAULT_LOCK = Path("/tmp/mtplx-gpu-exclusive.lock")
_CONTENT_SENTINEL = "MTPLX_QWEN38_RESIDENT_OQ4_CONTENT_17E3A1"
_SMOKE_INSTRUCTION = "Write a Python function that adds two integers."
_MAX_ATTESTATION_BYTES = 16 * 1024
_GIB = 1024**3


def _resident_load_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Normal resident loader arguments; the n-gram cache is the only lane."""

    return {
        "mtp": args.mode == "mtp",
        "ngram_cache_limit_bytes": int(args.ngram_cache_gib) * _GIB,
        "ngram_context_tokens": int(args.prompt_tokens) + int(args.max_tokens),
        "ngram_target_residency_bytes": int(args.runtime_target_gib) * _GIB,
    }


def _token_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(int(token)) for token in tokens).encode("ascii")
    ).hexdigest()


def build_exact_python_prompt(
    tokenizer: Any,
    *,
    context: str,
    instruction: str,
    target_tokens: int,
    reasoning_effort: str,
) -> tuple[str, list[int]]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": _CONTENT_SENTINEL}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort=reasoning_effort,
    )
    if not isinstance(rendered, str) or rendered.count(_CONTENT_SENTINEL) != 1:
        raise ValueError("chat template did not preserve the workload sentinel")
    prefix, suffix = rendered.split(_CONTENT_SENTINEL)
    prefix_ids = list(tokenizer.encode(prefix))
    suffix_ids = list(tokenizer.encode(suffix))
    instruction_ids = list(tokenizer.encode("\n\n" + instruction.strip()))
    fixed = len(prefix_ids) + len(instruction_ids) + len(suffix_ids)
    if fixed >= target_tokens:
        raise ValueError("instruction does not fit inside the prompt token target")
    context_ids = list(tokenizer.encode(context.rstrip() + "\n"))
    if not context_ids:
        raise ValueError("Python context must encode to at least one token")
    budget = target_tokens - fixed
    repeats = (budget + len(context_ids) - 1) // len(context_ids)
    tokens = prefix_ids + (context_ids * repeats)[:budget] + instruction_ids + suffix_ids
    # The explicit IDs are the workload contract. BPE decode is not a canonical
    # inverse of encode: adjacent newline tokens, for example, may re-encode as
    # one merged token even though they decode to identical text.
    prompt = str(tokenizer.decode(tokens))
    return prompt, tokens


def _consume_guard_attestation(
    environment: MutableMapping[str, str], *, expected_lock: Path
) -> dict[str, Any]:
    """Consume run_guarded.py's one-shot proof before any MLX import."""

    raw_fd = environment.pop("MTPLX_GUARD_ATTEST_FD", None)
    nonce = environment.pop("MTPLX_GUARD_ATTEST_NONCE", None)
    if raw_fd is None or nonce is None:
        raise RuntimeError("harness must run through the canonical guarded wrapper")
    try:
        descriptor = int(raw_fd)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("guard supplied an invalid attestation pipe") from exc
    payload = bytearray()
    try:
        while len(payload) <= _MAX_ATTESTATION_BYTES:
            chunk = os.read(descriptor, _MAX_ATTESTATION_BYTES + 1 - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
    finally:
        os.close(descriptor)
    if not payload or len(payload) > _MAX_ATTESTATION_BYTES:
        raise RuntimeError("guard attestation is malformed")
    receipt = json.loads(payload)
    lock_path = expected_lock.expanduser().resolve(strict=True)
    lock_status = lock_path.lstat()
    now = time.monotonic_ns()
    expected = {
        "schema_version": 1,
        "nonce": nonce,
        "child_pid": os.getpid(),
        "guard_pid": os.getppid(),
        "lock_path": str(lock_path),
        "lock_device": lock_status.st_dev,
        "lock_inode": lock_status.st_ino,
    }
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise RuntimeError(f"guard attestation mismatch: {name}")
    issued = receipt.get("issued_monotonic_ns")
    expires = receipt.get("expires_monotonic_ns")
    if type(issued) is not int or type(expires) is not int or not issued <= now <= expires:
        raise RuntimeError("guard attestation has expired")
    if (
        not stat.S_ISREG(lock_status.st_mode)
        or lock_status.st_nlink != 1
        or stat.S_IMODE(lock_status.st_mode) != 0o600
        or lock_status.st_uid != os.getuid()
    ):
        raise RuntimeError("canonical GPU lock identity is unsafe")
    return receipt


def _prompt_instruction(path: Path) -> str:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or not isinstance(rows[0].get("prompt"), str):
        raise RuntimeError("Python workload fixture must contain exactly one prompt")
    return str(rows[0]["prompt"])


def _run_instruction(path: Path, *, smoke: bool) -> str:
    return _SMOKE_INSTRUCTION if smoke else _prompt_instruction(path)


def _source_revision(model: Path) -> str:
    manifest = json.loads((model / "ngram-manifest.json").read_text(encoding="utf-8"))
    revision = manifest.get("source_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise RuntimeError("ngram manifest has no pinned source revision")
    return revision


def _static_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Run the exact CPU-only inventory and budget solver without MLX."""

    from mtplx.artifacts import load_config
    from mtplx.qwen4_ngram import load_ngram_manifest
    from mtplx.qwen4_preflight import (
        plan_qwen4_resident_preflight,
        scan_qwen4_weight_bytes,
        validate_qwen4_oq4_contract,
    )

    model = args.model.expanduser().resolve(strict=True)
    config = load_config(model)
    manifest = load_ngram_manifest(model / "ngram-manifest.json")
    validate_qwen4_oq4_contract(config, manifest)
    inventory = scan_qwen4_weight_bytes(model)
    plan = plan_qwen4_resident_preflight(
        resident_weight_bytes=inventory.resident_bytes,
        manifest=manifest,
        context_tokens=args.prompt_tokens + args.max_tokens,
        payload_ceiling_bytes=args.ngram_cache_gib * _GIB,
        target_residency_bytes=args.runtime_target_gib * _GIB,
    )
    return {
        **inventory.__dict__,
        **plan.__dict__,
    }


def _run_guarded_child(args: argparse.Namespace) -> int:
    guard = _consume_guard_attestation(os.environ, expected_lock=args.lock)
    model = args.model.expanduser().resolve(strict=True)
    if not args.prompt_file.is_file() or not args.context_file.is_file():
        raise RuntimeError("Python workload fixture or context file is missing")

    # Install the existing MTPLX product profile once, before generation/runtime
    # imports and construction. runtime.load then performs the header-only gate
    # before importing MLX.
    from mtplx.profiles import apply_profile_env, restore_profile_env

    previous_profile_env = apply_profile_env(args.profile)
    runtime = None
    output = None
    started = time.perf_counter()
    try:
        from mtplx.generation import generate_ar, generate_mtpk
        from mtplx.runtime import load
        from mtplx.sampling import SamplerConfig

        runtime = load(model, **_resident_load_kwargs(args))
        _prompt, prompt_ids = build_exact_python_prompt(
            runtime.tokenizer,
            context=args.context_file.read_text(encoding="utf-8"),
            instruction=_run_instruction(args.prompt_file, smoke=args.smoke),
            target_tokens=args.prompt_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        sampler = SamplerConfig(1.0, 0.95, 20)
        if args.mode == "ar":
            output = generate_ar(
                runtime,
                prompt_ids,
                max_tokens=args.max_tokens,
                sampler=sampler,
                seed=42,
                stop_token_ids=set(),
            )
        else:
            output = generate_mtpk(
                runtime,
                prompt_ids,
                max_tokens=args.max_tokens,
                sampler=sampler,
                speculative_depth=args.depth,
                seed=42,
                stop_token_ids=set(),
                mtp_hidden_variant="post_norm",
                mtp_cache_policy="persistent",
                mtp_history_policy="committed",
                verify_strategy="batched",
                verify_core="stock",
            )
        wall_s = time.perf_counter() - started
        stats = output.stats
        generated = int(stats.generated_tokens)
        if generated != args.max_tokens or len(output.tokens) != args.max_tokens:
            raise RuntimeError(
                f"generation count {generated}/{len(output.tokens)} != {args.max_tokens}"
            )
        receipt = {
            "schema": "mtplx-qwen38-resident-oq4-harness-v1",
            "status": "passed",
            "mode": args.mode,
            "profile": args.profile,
            "source_revision": _source_revision(model),
            "prompt_tokens": len(prompt_ids),
            "prompt_token_sha256": _token_sha256(prompt_ids),
            "generated_tokens": generated,
            "output_token_sha256": _token_sha256(list(output.tokens)),
            "wall_s": wall_s,
            "prefill_tps": float(getattr(stats, "prompt_tps", 0.0) or 0.0),
            "decode_tps": float(getattr(stats, "tok_s", 0.0) or 0.0),
            "peak_memory_bytes": int(getattr(stats, "peak_memory_bytes", 0) or 0),
            "accepted_by_depth": list(getattr(stats, "accepted_by_depth", []) or []),
            "preflight": runtime.ngram_preflight_report,
            "ngram_cache": runtime.ngram_memory_report,
            "guard": {
                "lock_path": guard.get("lock_path"),
                "guard_pid": guard.get("guard_pid"),
                "child_pid": guard.get("child_pid"),
            },
            "output_text": str(output.text),
        }
    finally:
        if runtime is not None:
            runtime.close()
        restore_profile_env(previous_profile_env)
    if output is None:
        raise RuntimeError("generation returned no output")
    output_path = args.output.expanduser().resolve() if args.output else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


def _outer_command(args: argparse.Namespace) -> list[str]:
    forwarded = [
        str(args.guard),
        "--plist", str(args.plist),
        "--lock-path", str(args.lock),
        "--lock-timeout-seconds", str(args.lock_timeout_seconds),
        "--child-timeout-seconds", str(args.child_timeout_seconds),
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        "--inner-guarded",
        "--model", str(args.model),
        "--mode", args.mode,
        "--prompt-file", str(args.prompt_file),
        "--context-file", str(args.context_file),
        "--prompt-tokens", str(args.prompt_tokens),
        "--max-tokens", str(args.max_tokens),
        "--reasoning-effort", args.reasoning_effort,
        "--profile", args.profile,
        "--depth", str(args.depth),
        "--ngram-cache-gib", str(args.ngram_cache_gib),
        "--runtime-target-gib", str(args.runtime_target_gib),
        "--lock", str(args.lock),
    ]
    if args.smoke:
        forwarded.append("--smoke")
    if args.output:
        forwarded.extend(("--output", str(args.output)))
    return [sys.executable, *forwarded]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=("mtp", "ar"), default="mtp")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--context-file", type=Path, default=DEFAULT_CONTEXT_FILE)
    parser.add_argument("--prompt-tokens", type=int, default=16_384)
    parser.add_argument("--max-tokens", type=int, default=1_024)
    parser.add_argument("--reasoning-effort", choices=("low", "xhigh"), default="low")
    parser.add_argument("--profile", default="sustained")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--ngram-cache-gib", type=int, default=10)
    parser.add_argument("--runtime-target-gib", type=int, default=82)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--guard", type=Path, default=DEFAULT_GUARD)
    parser.add_argument("--plist", type=Path, default=DEFAULT_PLIST)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--lock-timeout-seconds", type=int, default=21_600)
    parser.add_argument("--child-timeout-seconds", type=int, default=7_200)
    parser.add_argument("--inner-guarded", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.smoke:
        if (args.prompt_tokens, args.max_tokens) == (16_384, 1_024):
            args.prompt_tokens, args.max_tokens = 128, 8
        if args.prompt_tokens > 128 or args.max_tokens > 32:
            raise ValueError("smoke mode is limited to 128 prompt and 32 output tokens")
    elif (args.prompt_tokens, args.max_tokens) != (16_384, 1_024):
        raise ValueError("production mode requires exactly 16,384 input and 1,024 output")
    if not 1 <= args.depth <= 8:
        raise ValueError("depth must be in [1, 8]")
    if not 1 <= args.ngram_cache_gib <= 10:
        raise ValueError("ngram cache payload must be in [1, 10] GiB")
    if not 1 <= args.runtime_target_gib <= 82:
        raise ValueError("runtime target must be in [1, 82] GiB")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.preflight_only:
        print(json.dumps(_static_preflight(args), indent=2, sort_keys=True))
        return 0
    if args.inner_guarded:
        return _run_guarded_child(args)
    if not args.guard.is_file():
        raise RuntimeError(f"canonical guard wrapper is missing: {args.guard}")
    if not args.plist.is_file():
        raise RuntimeError(f"Qwen launchd plist is missing: {args.plist}")
    return subprocess.call(_outer_command(args), cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
