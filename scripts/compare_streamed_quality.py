#!/usr/bin/env python3
"""Compare sequential Q4/Q2 GLM streamed quality with deterministic gates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import stat
import sys
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SCHEMA = "mtplx-streamed-quality-v1"
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)
_MAX_EVIDENCE_FILE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class LaneConfig:
    """Immutable runtime identity and memory contract for one quality lane."""

    label: str
    model_root: Path
    manifest_path: Path
    model_key: str
    memory_limit: str | None = None
    expert_cache_limit: str | None = None
    runtime_reserve: str = "16GiB"
    max_live_kv_tokens: int = 8192

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_root", Path(self.model_root))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        if self.label not in {"q4", "q2"}:
            raise ValueError("lane label must be 'q4' or 'q2'")
        if not self.model_key:
            raise ValueError("lane model_key must be non-empty")


@dataclass(frozen=True)
class QualityLane:
    """A lane loader plus its mandatory post-close MLX cache cleanup."""

    config: LaneConfig
    load_runtime: Callable[[], Any]
    clear_cache: Callable[[], None]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_file_bytes(path: Path, *, label: str) -> tuple[Path, bytes]:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {source}")
    try:
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {source}: {exc}") from exc
    descriptor = os.open(
        resolved,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file: {resolved}")
        if before.st_size > _MAX_EVIDENCE_FILE_BYTES:
            raise ValueError(f"{label} exceeds its size bound: {resolved}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"{label} ended before its declared size: {resolved}")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"{label} changed while being read: {resolved}")
    return resolved, b"".join(chunks)


def _manifest_receipt(path: Path) -> dict[str, Any]:
    resolved, payload = _stable_file_bytes(path, label="expert manifest")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"expert manifest is malformed: {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expert manifest must be an object: {resolved}")
    declared = value.get("manifest_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError(f"expert manifest has no valid manifest_sha256: {resolved}")
    return {
        "path": str(resolved),
        "file_sha256": _sha256(payload),
        "declared_sha256": declared,
    }


def _tokenizer_receipt(root: Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve(strict=True)
    entries = []
    digest = hashlib.sha256()
    for name in _TOKENIZER_FILES:
        candidate = root / name
        if not candidate.exists():
            continue
        resolved, payload = _stable_file_bytes(candidate, label=f"tokenizer {name}")
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        entries.append(
            {
                "name": name,
                "path": str(resolved),
                "bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )
    if not entries or entries[0]["name"] != "tokenizer.json":
        raise ValueError(f"tokenizer.json is required under {root}")
    return {
        "algorithm": "sha256-name-and-length-prefixed-tokenizer-files-v1",
        "sha256": digest.hexdigest(),
        "files": entries,
    }


def _corpus_receipt(paths: Sequence[Path]) -> tuple[dict[str, Any], list[str]]:
    if not paths:
        raise ValueError("at least one corpus file is required")
    digest = hashlib.sha256()
    entries = []
    texts = []
    for index, path in enumerate(paths):
        resolved, payload = _stable_file_bytes(path, label="corpus file")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"corpus file is not UTF-8: {resolved}: {exc}") from exc
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        entries.append(
            {
                "index": index,
                "path": str(resolved),
                "bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )
        texts.append(text)
    return (
        {
            "algorithm": "sha256-length-prefixed-file-bytes-v1",
            "sha256": digest.hexdigest(),
            "file_order": [entry["path"] for entry in entries],
            "files": entries,
        },
        texts,
    )


def _load_prompts(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    resolved, payload = _stable_file_bytes(path, label="prompt JSONL")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"prompt JSONL is not UTF-8: {resolved}: {exc}") from exc
    prompts = []
    names = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{resolved}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{resolved}:{line_number}: prompt must be an object")
        row = {}
        for field in ("name", "category", "prompt"):
            item = value.get(field)
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"{resolved}:{line_number}: {field} must be a non-empty string"
                )
            row[field] = item
        if row["name"] in names:
            raise ValueError(f"{resolved}:{line_number}: duplicate prompt name")
        names.add(row["name"])
        prompts.append(row)
    if not prompts:
        raise ValueError("prompt JSONL is empty")
    return (
        {
            "path": str(resolved),
            "sha256": _sha256(payload),
            "bytes": len(payload),
            "prompt_count": len(prompts),
            "categories": sorted({prompt["category"] for prompt in prompts}),
        },
        prompts,
    )


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if isinstance(encoded, np.ndarray):
        encoded = encoded.tolist()
    if not isinstance(encoded, (list, tuple)):
        raise TypeError("tokenizer.encode must return a token sequence")
    result = []
    for token in encoded:
        if isinstance(token, bool) or not isinstance(token, (int, np.integer)):
            raise TypeError("tokenizer returned a non-integer token")
        value = int(token)
        if value < 0:
            raise ValueError("tokenizer returned a negative token")
        result.append(value)
    return result


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token in token_ids:
        digest.update(int(token).to_bytes(8, "big", signed=False))
    return digest.hexdigest()


def _tokenize_corpus(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    evaluation_tokens: int,
) -> tuple[list[int], list[int]]:
    per_file = [_encode(tokenizer, text) for text in texts]
    all_tokens = [token for tokens in per_file for token in tokens]
    selected = all_tokens[:evaluation_tokens]
    if len(selected) < 2:
        raise ValueError("quality corpus must yield at least two evaluation tokens")
    return selected, [len(tokens) for tokens in per_file]


def _runtime_input(runtime: Any, token_ids: Sequence[int]) -> Any:
    maker = getattr(runtime, "quality_input_array", None)
    if callable(maker):
        return maker([int(token) for token in token_ids])
    import mlx.core as mx

    return mx.array([[int(token) for token in token_ids]], dtype=mx.int32)


def _logits_array(logits: Any, *, expected_tokens: int) -> np.ndarray:
    module_name = type(logits).__module__
    if module_name == "mlx.core" or module_name.startswith("mlx."):
        import mlx.core as mx

        logits = logits.astype(mx.float32)
        mx.eval(logits)
    array = np.asarray(logits, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[0] != expected_tokens or array.shape[1] <= 0:
        raise ValueError(
            "runtime logits must have shape [1, token_count, vocabulary_size]"
        )
    return array


def _attention_phase(name: str):
    try:
        from mtplx.attention_context import attention_phase
    except ImportError:
        return nullcontext()
    return attention_phase(name)


def _admission(runtime: Any, tokens: int):
    admit = getattr(runtime, "admit_kv_tokens", None)
    return admit(tokens) if callable(admit) else nullcontext()


def _reset_streaming(runtime: Any) -> None:
    streaming = getattr(runtime, "expert_streaming", None)
    reset = getattr(streaming, "reset", None)
    if callable(reset):
        reset()


def teacher_forced_loss(
    runtime: Any,
    token_ids: Sequence[int],
    *,
    chunk_tokens: int,
) -> dict[str, Any]:
    """Score every next token with float32 logits through one sequential cache."""

    tokens = [int(token) for token in token_ids]
    if len(tokens) < 2:
        raise ValueError("teacher-forced evaluation requires at least two tokens")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    inputs = tokens[:-1]
    targets = tokens[1:]
    cache = runtime.make_cache()
    nll_sum = 0.0
    nan_count = 0
    nonfinite_count = 0
    chunks = 0
    _reset_streaming(runtime)
    with _admission(runtime, len(tokens)), _attention_phase("prefill"):
        for start in range(0, len(inputs), chunk_tokens):
            input_chunk = inputs[start : start + chunk_tokens]
            target_chunk = np.asarray(
                targets[start : start + chunk_tokens], dtype=np.int64
            )
            logits = runtime.forward_ar(
                _runtime_input(runtime, input_chunk), cache=cache
            )
            rows = _logits_array(logits, expected_tokens=len(input_chunk))
            if np.any(target_chunk >= rows.shape[1]):
                raise ValueError("teacher-forced target token exceeds vocabulary")
            nan_count += int(np.isnan(rows).sum())
            nonfinite_count += int((~np.isfinite(rows)).sum())
            maximum = np.max(rows, axis=-1).astype(np.float32, copy=False)
            shifted = (rows - maximum[:, None]).astype(np.float32, copy=False)
            exponentials = np.exp(shifted).astype(np.float32, copy=False)
            totals = np.sum(exponentials, axis=-1, dtype=np.float32)
            logsumexp = (maximum + np.log(totals)).astype(np.float32, copy=False)
            selected = rows[np.arange(len(target_chunk)), target_chunk]
            losses = (logsumexp - selected).astype(np.float32, copy=False)
            nan_count += int(np.isnan(losses).sum())
            nonfinite_count += int((~np.isfinite(losses)).sum())
            nll_sum += float(np.sum(losses, dtype=np.float64))
            chunks += 1
    token_count = len(targets)
    mean_nll = nll_sum / token_count
    try:
        perplexity = math.exp(mean_nll)
    except OverflowError:
        perplexity = math.inf
    finite = bool(
        nan_count == 0
        and nonfinite_count == 0
        and math.isfinite(nll_sum)
        and math.isfinite(mean_nll)
        and math.isfinite(perplexity)
    )
    return {
        "input_token_count": len(tokens),
        "token_count": token_count,
        "chunk_tokens": chunk_tokens,
        "chunk_count": chunks,
        "nll_sum": nll_sum,
        "mean_nll": mean_nll,
        "perplexity": perplexity,
        "finite": finite,
        "nan_count": nan_count,
        "nonfinite_count": nonfinite_count,
    }


def _stop_token_ids(tokenizer: Any) -> set[int]:
    values = getattr(tokenizer, "eos_token_ids", None)
    if values is None:
        value = getattr(tokenizer, "eos_token_id", None)
        values = [] if value is None else [value]
    if isinstance(values, (int, np.integer)):
        values = [values]
    return {
        int(value)
        for value in values
        if not isinstance(value, bool) and isinstance(value, (int, np.integer))
    }


def greedy_outputs(
    runtime: Any,
    prompts: Sequence[dict[str, str]],
    *,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Generate bounded greedy token arrays for diagnostics only."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    tokenizer = runtime.tokenizer
    stops = _stop_token_ids(tokenizer)
    results = []
    for index, prompt in enumerate(prompts):
        prompt_ids = _encode(tokenizer, prompt["prompt"])
        if not prompt_ids:
            raise ValueError(f"prompt {prompt['name']!r} tokenized to an empty input")
        _reset_streaming(runtime)
        cache = runtime.make_cache()
        generated: list[int] = []
        nan_count = 0
        nonfinite_count = 0
        finish_reason = "length"
        with _admission(runtime, len(prompt_ids) + max_tokens):
            with _attention_phase("prefill"):
                logits = runtime.forward_ar(
                    _runtime_input(runtime, prompt_ids), cache=cache
                )
            row = _logits_array(logits, expected_tokens=len(prompt_ids))[-1]
            for step in range(max_tokens):
                nan_count += int(np.isnan(row).sum())
                nonfinite_count += int((~np.isfinite(row)).sum())
                if not bool(np.all(np.isfinite(row))):
                    finish_reason = "nonfinite"
                    break
                token = int(np.argmax(row))
                generated.append(token)
                if token in stops:
                    finish_reason = "eos"
                    break
                if step + 1 < max_tokens:
                    with _attention_phase("ar_decode"):
                        logits = runtime.forward_ar(
                            _runtime_input(runtime, [token]), cache=cache
                        )
                    row = _logits_array(logits, expected_tokens=1)[-1]
        try:
            text = tokenizer.decode(generated)
        except TypeError:
            text = tokenizer.decode(generated, skip_special_tokens=False)
        results.append(
            {
                "prompt_index": index,
                "name": prompt["name"],
                "category": prompt["category"],
                "prompt_sha256": _sha256(prompt["prompt"].encode("utf-8")),
                "prompt_token_count": len(prompt_ids),
                "token_ids": generated,
                "generated_token_count": len(generated),
                "text": str(text),
                "finish_reason": finish_reason,
                "finite": nan_count == 0 and nonfinite_count == 0,
                "nan_count": nan_count,
                "nonfinite_count": nonfinite_count,
            }
        )
    return results


def greedy_diagnostics(
    q4_outputs: Sequence[dict[str, Any]],
    q2_outputs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compare deterministic greedy arrays without making them a quality gate."""

    if len(q4_outputs) != len(q2_outputs):
        raise ValueError("greedy output prompt counts differ")
    prompt_results = []
    agreement_tokens = 0
    compared_positions = 0
    first_divergence = None
    for index, (q4, q2) in enumerate(zip(q4_outputs, q2_outputs, strict=True)):
        if q4["name"] != q2["name"]:
            raise ValueError("greedy output prompt order differs")
        q4_tokens = [int(token) for token in q4["token_ids"]]
        q2_tokens = [int(token) for token in q2["token_ids"]]
        positions = max(len(q4_tokens), len(q2_tokens))
        agreements = 0
        prompt_first = None
        for token_index in range(positions):
            q4_token = q4_tokens[token_index] if token_index < len(q4_tokens) else None
            q2_token = q2_tokens[token_index] if token_index < len(q2_tokens) else None
            if q4_token == q2_token:
                agreements += 1
                continue
            if prompt_first is None:
                prompt_first = {
                    "token_index": token_index,
                    "q4_token": q4_token,
                    "q2_token": q2_token,
                }
            if first_divergence is None:
                first_divergence = {
                    "prompt_index": index,
                    "prompt_name": q4["name"],
                    **prompt_first,
                }
        agreement_tokens += agreements
        compared_positions += positions
        prompt_results.append(
            {
                "prompt_index": index,
                "prompt_name": q4["name"],
                "agreement_tokens": agreements,
                "compared_positions": positions,
                "agreement_fraction": agreements / positions if positions else 1.0,
                "first_divergence": prompt_first,
            }
        )
    return {
        "agreement_tokens": agreement_tokens,
        "compared_positions": compared_positions,
        "agreement_fraction": (
            agreement_tokens / compared_positions if compared_positions else 1.0
        ),
        "first_divergence": first_divergence,
        "prompts": prompt_results,
    }


def quality_gate(
    q4_perplexity: float,
    q2_perplexity: float,
    *,
    finite: bool,
    max_relative_perplexity_regression: float,
) -> dict[str, Any]:
    if (
        math.isfinite(q4_perplexity)
        and q4_perplexity > 0.0
        and math.isfinite(q2_perplexity)
    ):
        relative = q2_perplexity / q4_perplexity - 1.0
    else:
        relative = None
    quality_passed = bool(
        finite
        and relative is not None
        and math.isfinite(relative)
        and relative <= max_relative_perplexity_regression
    )
    return {
        "relative_perplexity_regression": relative,
        "max_relative_perplexity_regression": max_relative_perplexity_regression,
        "quality_passed": quality_passed,
    }


def _error(
    stage: str, exc: BaseException, *, lane: str | None = None
) -> dict[str, str]:
    result = {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
    }
    if lane is not None:
        result["lane"] = lane
    return result


def _evaluate_lane(
    lane: QualityLane,
    *,
    corpus_texts: Sequence[str],
    prompts: Sequence[dict[str, str]],
    evaluation_tokens: int,
    chunk_tokens: int,
    greedy_max_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, str]], bool]:
    config = lane.config
    result: dict[str, Any] = {
        "label": config.label,
        "model_root": str(config.model_root.expanduser().resolve()),
        "model_key": config.model_key,
    }
    errors: list[dict[str, str]] = []
    runtime = None
    load_attempted = False
    evaluation_ok = False
    cleanup_ok = True
    try:
        result["manifest"] = _manifest_receipt(config.manifest_path)
        result["tokenizer"] = _tokenizer_receipt(config.model_root)
        load_attempted = True
        runtime = lane.load_runtime()
        token_ids, per_file_token_counts = _tokenize_corpus(
            runtime.tokenizer,
            corpus_texts,
            evaluation_tokens=evaluation_tokens,
        )
        result["corpus"] = {
            "token_count": len(token_ids),
            "per_file_token_counts_before_truncation": per_file_token_counts,
            "token_ids_sha256": _token_ids_sha256(token_ids),
        }
        result["loss"] = teacher_forced_loss(
            runtime,
            token_ids,
            chunk_tokens=chunk_tokens,
        )
        result["greedy_outputs"] = greedy_outputs(
            runtime,
            prompts,
            max_tokens=greedy_max_tokens,
        )
        evaluation_ok = True
    except Exception as exc:
        errors.append(_error("lane_evaluation", exc, lane=config.label))
        result["error"] = errors[-1]
    finally:
        if runtime is not None:
            try:
                runtime.close(timeout=10.0)
            except Exception as exc:
                cleanup_ok = False
                errors.append(_error("runtime_close", exc, lane=config.label))
        if load_attempted:
            runtime = None
            gc.collect()
            try:
                lane.clear_cache()
            except Exception as exc:
                cleanup_ok = False
                errors.append(_error("mlx_cache_clear", exc, lane=config.label))
    return result, errors, evaluation_ok and cleanup_ok


def compare_quality(
    q4_lane: QualityLane,
    q2_lane: QualityLane,
    *,
    corpus_files: Sequence[Path],
    prompt_file: Path,
    evaluation_tokens: int,
    chunk_tokens: int,
    greedy_max_tokens: int,
    max_relative_perplexity_regression: float = 0.05,
) -> dict[str, Any]:
    """Evaluate Q4 to completion, close it, then evaluate Q2."""

    for name, value in (
        ("evaluation_tokens", evaluation_tokens),
        ("chunk_tokens", chunk_tokens),
        ("greedy_max_tokens", greedy_max_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        not math.isfinite(max_relative_perplexity_regression)
        or max_relative_perplexity_regression < 0.0
    ):
        raise ValueError(
            "max_relative_perplexity_regression must be finite and non-negative"
        )
    corpus, corpus_texts = _corpus_receipt(corpus_files)
    prompt_receipt, prompts = _load_prompts(prompt_file)
    errors: list[dict[str, str]] = []
    q4_result, q4_errors, q4_lane_ok = _evaluate_lane(
        q4_lane,
        corpus_texts=corpus_texts,
        prompts=prompts,
        evaluation_tokens=evaluation_tokens,
        chunk_tokens=chunk_tokens,
        greedy_max_tokens=greedy_max_tokens,
    )
    errors.extend(q4_errors)
    if q4_lane_ok:
        q2_result, q2_errors, _q2_cleanup_ok = _evaluate_lane(
            q2_lane,
            corpus_texts=corpus_texts,
            prompts=prompts,
            evaluation_tokens=evaluation_tokens,
            chunk_tokens=chunk_tokens,
            greedy_max_tokens=greedy_max_tokens,
        )
        errors.extend(q2_errors)
    else:
        q2_result = {
            "label": q2_lane.config.label,
            "model_root": str(q2_lane.config.model_root.expanduser().resolve()),
            "model_key": q2_lane.config.model_key,
            "skipped": "q4 lane did not complete safely",
        }

    q4_corpus = q4_result.get("corpus")
    q2_corpus = q2_result.get("corpus")
    if isinstance(q4_corpus, dict) and isinstance(q2_corpus, dict):
        corpus["token_count"] = q4_corpus["token_count"]
        if q4_corpus["token_ids_sha256"] != q2_corpus["token_ids_sha256"]:
            errors.append(
                {
                    "stage": "token_alignment",
                    "type": "TokenizerMismatch",
                    "message": "Q4 and Q2 corpus token IDs differ",
                }
            )
    q4_tokenizer = q4_result.get("tokenizer")
    q2_tokenizer = q2_result.get("tokenizer")
    if isinstance(q4_tokenizer, dict) and isinstance(q2_tokenizer, dict):
        if q4_tokenizer["sha256"] != q2_tokenizer["sha256"]:
            errors.append(
                {
                    "stage": "tokenizer_identity",
                    "type": "TokenizerMismatch",
                    "message": "Q4 and Q2 tokenizer artifact hashes differ",
                }
            )

    greedy = None
    if "greedy_outputs" in q4_result and "greedy_outputs" in q2_result:
        try:
            greedy = greedy_diagnostics(
                q4_result["greedy_outputs"], q2_result["greedy_outputs"]
            )
        except Exception as exc:
            errors.append(_error("greedy_diagnostics", exc))

    lane_values = (q4_result, q2_result)
    nan_count = sum(
        int(lane.get("loss", {}).get("nan_count", 0))
        + sum(int(row.get("nan_count", 0)) for row in lane.get("greedy_outputs", []))
        for lane in lane_values
    )
    nonfinite_count = sum(
        int(lane.get("loss", {}).get("nonfinite_count", 0))
        + sum(
            int(row.get("nonfinite_count", 0)) for row in lane.get("greedy_outputs", [])
        )
        for lane in lane_values
    )
    finite = bool(
        not errors
        and nan_count == 0
        and nonfinite_count == 0
        and all(bool(lane.get("loss", {}).get("finite")) for lane in lane_values)
        and all(
            bool(row.get("finite"))
            for lane in lane_values
            for row in lane.get("greedy_outputs", [])
        )
    )
    q4_perplexity = float(q4_result.get("loss", {}).get("perplexity", math.nan))
    q2_perplexity = float(q2_result.get("loss", {}).get("perplexity", math.nan))
    gate = quality_gate(
        q4_perplexity,
        q2_perplexity,
        finite=finite,
        max_relative_perplexity_regression=max_relative_perplexity_regression,
    )
    passed = bool(gate["quality_passed"] and not errors)
    return {
        "schema": _SCHEMA,
        "passed": passed,
        "quality_passed": gate["quality_passed"],
        "finite": finite,
        "relative_perplexity_regression": gate["relative_perplexity_regression"],
        "max_relative_perplexity_regression": gate[
            "max_relative_perplexity_regression"
        ],
        "nan_count": nan_count,
        "nonfinite_count": nonfinite_count,
        "corpus": corpus,
        "prompt_file": prompt_receipt,
        "lanes": {"q4": q4_result, "q2": q2_result},
        "greedy_diagnostics": greedy,
        "errors": errors,
    }


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q4-root", type=Path, required=True)
    parser.add_argument("--q4-manifest", type=Path, required=True)
    parser.add_argument("--q4-model-key", default="glm52-q4")
    parser.add_argument("--q2-root", type=Path, required=True)
    parser.add_argument("--q2-manifest", type=Path, required=True)
    parser.add_argument("--q2-model-key", default="glm52-expert-q2")
    parser.add_argument("--memory-limit", required=True)
    parser.add_argument("--expert-cache-limit", required=True)
    parser.add_argument("--runtime-reserve", default="16GiB")
    parser.add_argument("--max-live-kv-tokens", type=_positive_int, required=True)
    parser.add_argument(
        "--corpus-file", type=Path, action="append", required=True, dest="corpus_files"
    )
    parser.add_argument("--evaluation-tokens", type=_positive_int, required=True)
    parser.add_argument("--chunk-tokens", type=_positive_int, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--greedy-max-tokens", type=_positive_int, required=True)
    parser.add_argument(
        "--max-relative-perplexity-regression",
        type=_nonnegative_float,
        default=0.05,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _load_lane_runtime(config: LaneConfig) -> Any:
    if config.memory_limit is None or config.expert_cache_limit is None:
        raise ValueError("runtime lane memory limits are required")
    from mtplx.expert_runtime import ExpertStreamingConfig, parse_memory_bytes
    from mtplx.runtime import load

    streaming = ExpertStreamingConfig(
        model_key=config.model_key,
        memory_limit_bytes=parse_memory_bytes(config.memory_limit),
        expert_cache_limit_bytes=parse_memory_bytes(config.expert_cache_limit),
        runtime_reserve_bytes=parse_memory_bytes(config.runtime_reserve),
        max_live_kv_tokens=config.max_live_kv_tokens,
    )
    return load(
        config.model_root,
        mtp=False,
        expert_streaming_config=streaming,
        expert_manifest=config.manifest_path,
    )


def _clear_mlx_cache() -> None:
    import mlx.core as mx

    mx.synchronize()
    mx.clear_cache()


def _lane(config: LaneConfig) -> QualityLane:
    return QualityLane(
        config=config,
        load_runtime=lambda: _load_lane_runtime(config),
        clear_cache=_clear_mlx_cache,
    )


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        written = 0
        while written < len(encoded):
            count = os.write(descriptor, encoded[written:])
            if count <= 0:
                raise OSError("short write while saving quality evidence")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(
    argv: Sequence[str] | None = None,
    *,
    _compare_quality: Callable[..., dict[str, Any]] = compare_quality,
) -> int:
    args = build_parser().parse_args(argv)
    q4_config = LaneConfig(
        "q4",
        args.q4_root,
        args.q4_manifest,
        args.q4_model_key,
        memory_limit=args.memory_limit,
        expert_cache_limit=args.expert_cache_limit,
        runtime_reserve=args.runtime_reserve,
        max_live_kv_tokens=args.max_live_kv_tokens,
    )
    q2_config = LaneConfig(
        "q2",
        args.q2_root,
        args.q2_manifest,
        args.q2_model_key,
        memory_limit=args.memory_limit,
        expert_cache_limit=args.expert_cache_limit,
        runtime_reserve=args.runtime_reserve,
        max_live_kv_tokens=args.max_live_kv_tokens,
    )
    try:
        payload = _compare_quality(
            _lane(q4_config),
            _lane(q2_config),
            corpus_files=args.corpus_files,
            prompt_file=args.prompt_file,
            evaluation_tokens=args.evaluation_tokens,
            chunk_tokens=args.chunk_tokens,
            greedy_max_tokens=args.greedy_max_tokens,
            max_relative_perplexity_regression=(
                args.max_relative_perplexity_regression
            ),
        )
    except Exception as exc:
        payload = {
            "schema": _SCHEMA,
            "passed": False,
            "quality_passed": False,
            "relative_perplexity_regression": None,
            "errors": [_error("operation", exc)],
        }
        try:
            _write_json_once(args.output_json, payload)
        except Exception as write_exc:
            print(f"compare_streamed_quality: {write_exc}", file=sys.stderr)
        return 1
    try:
        _write_json_once(args.output_json, payload)
    except Exception as exc:
        print(f"compare_streamed_quality: {exc}", file=sys.stderr)
        return 1
    if payload.get("errors"):
        return 1
    return 0 if payload.get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
