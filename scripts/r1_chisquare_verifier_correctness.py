#!/usr/bin/env python3
"""R1 verifier correctness gate for the overnight kernel runbook.

The full gate is intentionally expensive: flappy + python-modules, three
prompt variants each, five fixed seeds, and 10k generated tokens per cell.
This script also supports tiny smoke settings so the harness itself can be
tested before starting the overnight run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterator, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_MODEL = "models/Qwen3.6-27B-MTPLX-GDN8-Speed4-CyanKiwiMTP"
DEFAULT_OUTPUT = (
    "outputs/overnight/kernel-verify-cycle/r1-chisquare-20260503.json"
)
DEFAULT_SEEDS = (42, 1337, 2024, 31415, 271828)
DEFAULT_SUITES = ("flappy", "python-modules")
DEFAULT_PAIR_PREFIX_CUTS = (64, 2048, 6144, 10240)
DEFAULT_VARIANT_SUFFIXES = (
    "",
    "\n\nUse clear structure and deterministic helper names.",
    "\n\nPrefer compact abstractions and include edge-case handling.",
)

PAGED_ENV_KEYS = (
    "MTPLX_VLLM_METAL_PAGED_ATTN",
    "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE",
    "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS",
    "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN",
    "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL",
    "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD",
    "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE",
    "MTPLX_VLLM_METAL_PAGED_ATTN_EXACT_GATHER_LAST_N",
    "MTPLX_VLLM_METAL_PAGED_ATTN_EXACT_GATHER_INDICES",
    "MTPLX_SPLIT_FULL_ATTN",
    "MTPLX_BLOCKWISE_ATTN",
    "MTPLX_SDPA_2PASS",
    "MTPLX_SDPA_2PASS_BLOCKS",
    "MTPLX_SDPA_DYNAMIC_OFFSET_ACTIVE_BLOCKS",
)


@dataclass(frozen=True)
class PromptSpec:
    suite: str
    source_id: str
    prompt_id: str
    category: str
    prompt: str
    variant_index: int
    derived: bool


@contextmanager
def _profile_env(profile: str) -> Iterator[dict[str, str | None]]:
    from mtplx.profiles import apply_profile_env, restore_profile_env

    previous = apply_profile_env(profile)
    try:
        yield previous
    finally:
        restore_profile_env(previous)


def _progress(args: argparse.Namespace, message: str) -> None:
    if getattr(args, "quiet", False):
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def _parse_ints(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def _parse_positive_ints(value: str) -> list[int]:
    out = _parse_ints(value)
    if any(item < 1 for item in out):
        raise argparse.ArgumentTypeError("all values must be positive")
    return out


def _parse_suites(value: str) -> list[str]:
    out = [part.strip() for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one suite")
    return out


def _resolve_suite_name(name: str) -> str:
    if name in {"python-modules", "python_modules"}:
        return "python_modules_long"
    return name


def _load_prompt_specs(
    suites: Sequence[str],
    *,
    prompts_per_suite: int,
) -> list[PromptSpec]:
    from mtplx.benchmarks.schema import load_prompt_suite
    from mtplx.kpi.runtime_kpis import prompt_suite_path

    specs: list[PromptSpec] = []
    for suite in suites:
        resolved = _resolve_suite_name(suite)
        cases = load_prompt_suite(prompt_suite_path(resolved))
        if not cases:
            raise ValueError(f"prompt suite {suite!r} is empty")
        for index in range(prompts_per_suite):
            case = cases[index % len(cases)]
            suffix = DEFAULT_VARIANT_SUFFIXES[index % len(DEFAULT_VARIANT_SUFFIXES)]
            derived = index >= len(cases) or bool(suffix)
            prompt = case.prompt + suffix
            specs.append(
                PromptSpec(
                    suite=suite,
                    source_id=case.id,
                    prompt_id=f"{case.id}__r1v{index + 1}",
                    category=case.category,
                    prompt=prompt,
                    variant_index=index + 1,
                    derived=derived,
                )
            )
    return specs


def _top_token_ids(a: Sequence[int], b: Sequence[int], *, top_n: int) -> list[int]:
    counts: dict[int, int] = {}
    for token in list(a) + list(b):
        counts[int(token)] = counts.get(int(token), 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _count in ranked[:top_n]]


def _count_vector(tokens: Sequence[int], vocab: Sequence[int]) -> np.ndarray:
    positions = {int(token): index for index, token in enumerate(vocab)}
    counts = np.zeros(len(vocab) + 1, dtype=np.float64)
    other_index = len(vocab)
    for token in tokens:
        counts[positions.get(int(token), other_index)] += 1.0
    return counts


def _chi_square_stat(counts_a: np.ndarray, counts_b: np.ndarray) -> tuple[float, int]:
    total_a = float(counts_a.sum())
    total_b = float(counts_b.sum())
    pooled = counts_a + counts_b
    total = total_a + total_b
    if total_a <= 0 or total_b <= 0 or total <= 0:
        return 0.0, 0
    expected_a = pooled * (total_a / total)
    expected_b = pooled * (total_b / total)
    mask = (expected_a > 0) & (expected_b > 0)
    if int(mask.sum()) <= 1:
        return 0.0, 0
    stat = float(
        np.sum(((counts_a[mask] - expected_a[mask]) ** 2) / expected_a[mask])
        + np.sum(((counts_b[mask] - expected_b[mask]) ** 2) / expected_b[mask])
    )
    return stat, int(mask.sum()) - 1


def _chi_square_pvalue_approx(stat: float, dof: int) -> float:
    """Approximate chi-square survival function without SciPy.

    Wilson-Hilferty is accurate enough for this gate's coarse pass/fail signal;
    the block permutation p-value is the primary reported value.
    """
    if dof <= 0:
        return 1.0
    if stat <= 0:
        return 1.0
    z = ((stat / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / math.sqrt(
        2.0 / (9.0 * dof)
    )
    return float(max(0.0, min(1.0, 1.0 - NormalDist().cdf(z))))


def _block_count_vectors(
    tokens: Sequence[int],
    vocab: Sequence[int],
    *,
    block_size: int,
) -> list[np.ndarray]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    blocks = []
    for start in range(0, len(tokens), block_size):
        block = tokens[start : start + block_size]
        if block:
            blocks.append(_count_vector(block, vocab))
    return blocks


def _block_permutation_pvalue(
    tokens_a: Sequence[int],
    tokens_b: Sequence[int],
    vocab: Sequence[int],
    *,
    observed_stat: float,
    block_size: int,
    bootstrap_samples: int,
    seed: int,
) -> float | None:
    blocks_a = _block_count_vectors(tokens_a, vocab, block_size=block_size)
    blocks_b = _block_count_vectors(tokens_b, vocab, block_size=block_size)
    if len(blocks_a) < 2 or len(blocks_b) < 2 or bootstrap_samples <= 0:
        return None
    blocks = blocks_a + blocks_b
    n_a = len(blocks_a)
    n_b = len(blocks_b)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(int(bootstrap_samples)):
        order = rng.permutation(len(blocks))
        boot_a = np.sum([blocks[int(i)] for i in order[:n_a]], axis=0)
        boot_b = np.sum([blocks[int(i)] for i in order[n_a : n_a + n_b]], axis=0)
        stat, _dof = _chi_square_stat(boot_a, boot_b)
        if stat >= observed_stat:
            hits += 1
    return float((hits + 1) / (int(bootstrap_samples) + 1))


def _kl_top_tokens(
    ref_tokens: Sequence[int],
    mtplx_tokens: Sequence[int],
    vocab: Sequence[int],
) -> float:
    ref = _count_vector(ref_tokens, vocab)
    mtplx = _count_vector(mtplx_tokens, vocab)
    epsilon = 1e-12
    p = (ref + epsilon) / float(ref.sum() + epsilon * ref.size)
    q = (mtplx + epsilon) / float(mtplx.sum() + epsilon * mtplx.size)
    return float(np.sum(p * np.log(p / q)))


def token_frequency_metrics(
    ref_tokens: Sequence[int],
    mtplx_tokens: Sequence[int],
    *,
    top_n: int = 200,
    block_size: int = 64,
    bootstrap_samples: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    vocab = _top_token_ids(ref_tokens, mtplx_tokens, top_n=top_n)
    ref_counts = _count_vector(ref_tokens, vocab)
    mtplx_counts = _count_vector(mtplx_tokens, vocab)
    stat, dof = _chi_square_stat(ref_counts, mtplx_counts)
    block_p = _block_permutation_pvalue(
        ref_tokens,
        mtplx_tokens,
        vocab,
        observed_stat=stat,
        block_size=block_size,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    ref_top = set(_top_token_ids(ref_tokens, [], top_n=top_n))
    mtplx_top = set(_top_token_ids(mtplx_tokens, [], top_n=top_n))
    return {
        "top_n": int(top_n),
        "block_size": int(block_size),
        "bootstrap_samples": int(bootstrap_samples),
        "effective_ref_blocks": int(math.ceil(len(ref_tokens) / block_size)) if ref_tokens else 0,
        "effective_mtplx_blocks": int(math.ceil(len(mtplx_tokens) / block_size)) if mtplx_tokens else 0,
        "chi_square_stat": stat,
        "chi_square_dof": dof,
        "chi_square_pvalue_block_permutation": block_p,
        "chi_square_pvalue_approx": _chi_square_pvalue_approx(stat, dof),
        "kl_ref_to_mtplx_top": _kl_top_tokens(ref_tokens, mtplx_tokens, vocab),
        "top_token_overlap": len(ref_top & mtplx_top),
        "top_token_union": len(ref_top | mtplx_top),
    }


def _effective_pvalue(frequency: dict[str, Any]) -> float:
    block_p = frequency.get("chi_square_pvalue_block_permutation")
    return float(block_p if block_p is not None else frequency["chi_square_pvalue_approx"])


def _frequency_gate_fails(frequency: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        _effective_pvalue(frequency) <= float(args.pvalue_threshold)
        or float(frequency["kl_ref_to_mtplx_top"]) >= float(args.kl_threshold)
    )


@contextmanager
def _patched_env(updates: dict[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _paged_env(args: argparse.Namespace, *, enabled: bool) -> dict[str, str | None]:
    env = {key: None for key in PAGED_ENV_KEYS}
    if not enabled:
        return env
    env.update(
        {
            "MTPLX_VLLM_METAL_PAGED_ATTN": "1",
            "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE": str(args.attention_block_size),
            "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS": str(args.attention_num_blocks),
            "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL": args.attention_impl,
        }
    )
    if args.attention_partitioned:
        env["MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN"] = "1"
        env["MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD"] = str(
            args.attention_partition_threshold
        )
        env["MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE"] = str(
            args.attention_partition_size
        )
    return env


def _eval_result(value: Any) -> None:
    import mlx.core as mx

    if isinstance(value, tuple):
        mx.eval(*value)
    else:
        mx.eval(value)


def _last_logits_np(logits: Any) -> np.ndarray:
    import mlx.core as mx

    final_logits = logits[:, -1, :].astype(mx.float32)
    mx.eval(final_logits)
    return np.asarray(final_logits, dtype=np.float32).reshape(-1)


def _decode_last_logits_from_prefix(
    rt: Any,
    prefix_ids: Sequence[int],
    args: argparse.Namespace,
    *,
    paged: bool,
) -> tuple[np.ndarray, float]:
    """Return next-token logits after a paired prefix using stock or paged decode.

    The prefix is split into a stock prefill prefix and a one-token decode step.
    That keeps both sides on the same token history while isolating the active
    verifier decode path.
    """
    import mlx.core as mx

    from mtplx.attention_split import configure_split_full_attention
    from mtplx.cache_state import install_vllm_metal_paged_attention_kv_cache

    if len(prefix_ids) < 2:
        raise ValueError("paired-prefix rows need at least two prefix tokens")

    with _patched_env(_paged_env(args, enabled=False)):
        configure_split_full_attention(rt.model)
        cache = rt.make_cache()
        prefill = rt.forward_ar(
            mx.array([list(prefix_ids[:-1])], dtype=mx.int32),
            cache=cache,
            return_hidden=False,
        )
        _eval_result(prefill)

    if paged:
        install_vllm_metal_paged_attention_kv_cache(
            cache,
            block_size=args.attention_block_size,
            num_blocks=args.attention_num_blocks,
        )

    with _patched_env(_paged_env(args, enabled=paged)):
        configure_split_full_attention(rt.model)
        started = time.perf_counter()
        logits = rt.forward_ar(
            mx.array([[int(prefix_ids[-1])]], dtype=mx.int32),
            cache=cache,
            return_hidden=False,
        )
        row = _last_logits_np(logits)
        elapsed = time.perf_counter() - started
    return row, elapsed


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        out = np.zeros_like(logits, dtype=np.float64)
        out[int(np.argmax(logits))] = 1.0
        return out
    scaled = logits / float(temperature)
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    return exp / np.sum(exp)


def _distribution_from_logits(
    logits: np.ndarray,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> np.ndarray:
    probs = _softmax(logits, temperature)
    mask = np.ones(probs.shape[0], dtype=bool)
    if 0 < top_p < 1.0:
        order = np.argsort(-probs)
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        keep_sorted = cumulative <= float(top_p)
        if keep_sorted.size:
            keep_sorted[0] = True
            first_over = np.argmax(cumulative >= float(top_p))
            keep_sorted[: first_over + 1] = True
        nucleus_mask = np.zeros_like(mask)
        nucleus_mask[order[keep_sorted]] = True
        mask &= nucleus_mask
    if top_k and 0 < top_k < probs.shape[0]:
        scoped = np.where(mask, probs, 0.0)
        keep = np.argpartition(-scoped, int(top_k) - 1)[: int(top_k)]
        top_mask = np.zeros_like(mask)
        top_mask[keep] = True
        mask &= top_mask
    filtered = np.where(mask, probs, 0.0)
    total = float(filtered.sum())
    if total <= 0:
        filtered[int(np.argmax(probs))] = 1.0
        total = 1.0
    return filtered / total


def _top_ids(logits: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), int(logits.shape[0])))
    ids = np.argpartition(-logits, k - 1)[:k]
    order = np.argsort(-logits[ids])
    return ids[order].astype(np.int64)


def _sample_with_shared_uniforms(
    p: np.ndarray,
    q: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    p_cdf = np.cumsum(p)
    q_cdf = np.cumsum(q)
    p_cdf[-1] = 1.0
    q_cdf[-1] = 1.0
    matches = 0
    first_mismatch: dict[str, Any] | None = None
    for index, uniform in enumerate(rng.random(draws)):
        p_token = int(np.searchsorted(p_cdf, uniform, side="left"))
        q_token = int(np.searchsorted(q_cdf, uniform, side="left"))
        if p_token == q_token:
            matches += 1
        elif first_mismatch is None:
            first_mismatch = {
                "draw_index": int(index),
                "u": float(uniform),
                "stock_token": p_token,
                "candidate_token": q_token,
                "stock_prob": float(p[p_token]),
                "candidate_prob": float(q[q_token]),
            }
    return {
        "draws": int(draws),
        "matches": int(matches),
        "agreement": float(matches / max(1, draws)),
        "first_mismatch": first_mismatch,
    }


def paired_prefix_distribution_metrics(
    stock_logits: np.ndarray,
    candidate_logits: np.ndarray,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    top_k_compare: int,
    sample_seed: int,
    sample_draws: int,
) -> dict[str, Any]:
    diff = candidate_logits.astype(np.float32) - stock_logits.astype(np.float32)
    stock_argmax = int(np.argmax(stock_logits))
    candidate_argmax = int(np.argmax(candidate_logits))
    stock_top = _top_ids(stock_logits, top_k_compare)
    candidate_top = _top_ids(candidate_logits, top_k_compare)
    overlap = len(set(stock_top.tolist()) & set(candidate_top.tolist()))

    p = _distribution_from_logits(
        stock_logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    q = _distribution_from_logits(
        candidate_logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    p_support = p > 0
    q_support = q > 0
    support_union = p_support | q_support
    support_intersection = p_support & q_support
    eps = 1e-300
    return {
        "logits": {
            "max_abs_diff": float(np.max(np.abs(diff))),
            "mean_abs_diff": float(np.mean(np.abs(diff))),
            "rms_diff": float(math.sqrt(float(np.mean(diff.astype(np.float64) ** 2)))),
            "stock_argmax": stock_argmax,
            "candidate_argmax": candidate_argmax,
            "argmax_match": bool(stock_argmax == candidate_argmax),
        },
        "topk": {
            "k": int(top_k_compare),
            "stock": stock_top.tolist(),
            "candidate": candidate_top.tolist(),
            "overlap": int(overlap),
            "overlap_ratio": float(overlap / max(1, min(top_k_compare, stock_logits.shape[0]))),
        },
        "distribution": {
            "stock_support_size": int(p_support.sum()),
            "candidate_support_size": int(q_support.sum()),
            "support_intersection": int(support_intersection.sum()),
            "support_union": int(support_union.sum()),
            "support_jaccard": float(
                support_intersection.sum() / max(1, support_union.sum())
            ),
            "support_equal": bool(np.array_equal(p_support, q_support)),
            "kl_stock_to_candidate": float(
                np.sum(p[p_support] * np.log(p[p_support] / np.maximum(q[p_support], eps)))
            ),
            "kl_candidate_to_stock": float(
                np.sum(q[q_support] * np.log(q[q_support] / np.maximum(p[q_support], eps)))
            ),
            "total_variation": float(0.5 * np.sum(np.abs(p - q))),
        },
        "controlled_rng_sample": _sample_with_shared_uniforms(
            p,
            q,
            seed=sample_seed,
            draws=sample_draws,
        ),
    }


def _paired_prefix_row_passes(row: dict[str, Any], args: argparse.Namespace) -> bool:
    metrics = row["metrics"]
    return bool(
        metrics["logits"]["max_abs_diff"] <= float(args.paired_max_logit_diff)
        and metrics["logits"]["argmax_match"]
        and metrics["topk"]["overlap_ratio"] >= float(args.paired_min_topk_overlap)
        and metrics["distribution"]["support_equal"]
        and metrics["distribution"]["total_variation"] <= float(args.paired_max_total_variation)
        and metrics["distribution"]["kl_stock_to_candidate"] <= float(args.paired_max_kl)
        and metrics["controlled_rng_sample"]["agreement"] >= float(args.paired_min_sample_agreement)
    )


def sequence_metrics(ref_tokens: Sequence[int], mtplx_tokens: Sequence[int]) -> dict[str, Any]:
    compare_len = min(len(ref_tokens), len(mtplx_tokens))
    first_mismatch = None
    for index in range(compare_len):
        if int(ref_tokens[index]) != int(mtplx_tokens[index]):
            first_mismatch = {
                "index": index,
                "reference": int(ref_tokens[index]),
                "mtplx": int(mtplx_tokens[index]),
            }
            break
    return {
        "reference_tokens": len(ref_tokens),
        "mtplx_tokens": len(mtplx_tokens),
        "reference_sha256": _token_sha256(ref_tokens),
        "mtplx_sha256": _token_sha256(mtplx_tokens),
        "exact_match": list(ref_tokens) == list(mtplx_tokens),
        "first_100_exact": list(ref_tokens[:100]) == list(mtplx_tokens[:100]),
        "prefix_equal_tokens": compare_len if first_mismatch is None else int(first_mismatch["index"]),
        "first_mismatch": first_mismatch,
    }


def _token_sha256(tokens: Sequence[int]) -> str:
    payload = ",".join(str(int(token)) for token in tokens).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _encode_prompt(rt: Any, spec: PromptSpec, *, enable_thinking: bool | None) -> list[int]:
    from mtplx.benchmarks.schema import PromptCase, encode_prompt_case

    case = PromptCase(
        id=spec.prompt_id,
        category=spec.category,
        prompt=spec.prompt,
        max_tokens=0,
    )
    return encode_prompt_case(
        rt.tokenizer,
        case,
        chat_template=True,
        enable_thinking=enable_thinking,
    )


def _generation_row(out: Any, *, include_tokens: bool = False) -> dict[str, Any]:
    stats = out.stats.to_dict()
    stats.pop("events", None)
    row = {
        "generated_tokens": int(out.stats.generated_tokens),
        "tok_s": float(out.stats.tok_s),
        "elapsed_s": float(out.stats.elapsed_s),
        "target_forward_time_s": float(out.stats.target_forward_time_s),
        "verify_time_s": float(out.stats.verify_time_s),
        "verify_hidden_eval_time_s": float(out.stats.verify_hidden_eval_time_s),
        "verify_calls": int(out.stats.verify_calls),
        "accepted_by_depth": list(out.stats.accepted_by_depth),
        "drafted_by_depth": list(out.stats.drafted_by_depth),
        "correction_tokens": int(out.stats.correction_tokens),
        "bonus_tokens": int(out.stats.bonus_tokens),
        "tokens_sha256": _token_sha256(out.tokens),
        "stats": stats,
    }
    if include_tokens:
        row["tokens"] = list(out.tokens)
    return row


def _run_reference_ar(
    rt: Any,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    sampler: Any,
    seed: int,
) -> Any:
    from mtplx.generation import generate_ar

    return generate_ar(rt, prompt_ids, max_tokens=max_tokens, sampler=sampler, seed=seed)


def _run_mtplx(
    rt: Any,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    sampler: Any,
    seed: int,
    depth: int,
    verify_strategy: str,
    verify_core: str,
    mtp_history_policy: str,
) -> Any:
    from mtplx.generation import generate_mtpk

    return generate_mtpk(
        rt,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=sampler,
        speculative_depth=depth,
        seed=seed,
        verify_strategy=verify_strategy,
        verify_core=verify_core,
        mtp_history_policy=mtp_history_policy,
        mtp_cache_policy="persistent",
    )


def _install_profile_accelerators(rt: Any, profile_name: str) -> dict[str, Any] | None:
    from mtplx.profiles import get_profile

    profile = get_profile(profile_name)
    if profile.draft_lm_head is None:
        return None
    from mtplx.draft_lm_head import _install_draft_lm_head

    req = profile.draft_lm_head
    return _install_draft_lm_head(
        rt,
        bits=req.bits,
        group_size=req.group_size,
        mode=req.mode,
    )


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    from mtplx.mtp_patch import MTPContract
    from mtplx.runtime import load
    from mtplx.sampling import SamplerConfig

    suites = _parse_suites(args.suites)
    seeds = _parse_ints(args.seeds)
    prompt_specs = _load_prompt_specs(suites, prompts_per_suite=args.prompts_per_suite)
    if args.max_cells is not None:
        prompt_specs = prompt_specs[: max(1, int(args.max_cells))]

    started = time.perf_counter()
    with _profile_env(args.profile) as previous_env:
        _progress(args, f"loading model={args.model} profile={args.profile}")
        rt = load(args.model, mtp=True, contract=MTPContract())
        accelerator_report = _install_profile_accelerators(rt, args.profile)
        _progress(args, f"model loaded; draft_lm_head={accelerator_report is not None}")
        t0_sampler = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
        t06_sampler = SamplerConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        prompt_rows: list[dict[str, Any]] = []
        t0_rows: list[dict[str, Any]] = []
        t06_rows: list[dict[str, Any]] = []
        ar_null_rows: list[dict[str, Any]] = []
        reference_tokens_by_prompt: dict[str, list[tuple[int, list[int]]]] = {}
        t06_failure_count = 0
        early_stop_reason: str | None = None

        for prompt_index, spec in enumerate(prompt_specs):
            ids = _encode_prompt(rt, spec, enable_thinking=args.enable_thinking)
            _progress(
                args,
                f"prompt {prompt_index + 1}/{len(prompt_specs)} {spec.prompt_id} "
                f"tokens={len(ids)}",
            )
            prompt_rows.append(
                {
                    **asdict(spec),
                    "prompt_tokens": len(ids),
                }
            )
            t0_ref = _run_reference_ar(
                rt,
                ids,
                max_tokens=args.t0_length,
                sampler=t0_sampler,
                seed=seeds[0],
            )
            t0_mtplx = _run_mtplx(
                rt,
                ids,
                max_tokens=args.t0_length,
                sampler=t0_sampler,
                seed=seeds[0],
                depth=args.depth,
                verify_strategy=args.verify_strategy,
                verify_core=args.verify_core,
                mtp_history_policy=args.mtp_history_policy,
            )
            t0_rows.append(
                {
                    "prompt_id": spec.prompt_id,
                    "seed": int(seeds[0]),
                    "sequence": sequence_metrics(t0_ref.tokens, t0_mtplx.tokens),
                    "reference": _generation_row(t0_ref, include_tokens=args.include_tokens),
                    "mtplx": _generation_row(t0_mtplx, include_tokens=args.include_tokens),
                }
            )
            _progress(
                args,
                "T=0 "
                f"prompt={spec.prompt_id} exact={t0_rows[-1]['sequence']['exact_match']} "
                f"ref_tok_s={t0_rows[-1]['reference']['tok_s']:.3f} "
                f"mtplx_tok_s={t0_rows[-1]['mtplx']['tok_s']:.3f}",
            )
            for seed in seeds:
                if args.max_t06_cells is not None and len(t06_rows) >= int(args.max_t06_cells):
                    break
                _progress(
                    args,
                    f"T=0.6 start cell={len(t06_rows) + 1} "
                    f"prompt={spec.prompt_id} seed={int(seed)} length={args.length}",
                )
                ref = _run_reference_ar(
                    rt,
                    ids,
                    max_tokens=args.length,
                    sampler=t06_sampler,
                    seed=seed,
                )
                reference_tokens_by_prompt.setdefault(spec.prompt_id, []).append(
                    (int(seed), list(ref.tokens))
                )
                if args.ar_null_calibration and not any(
                    row["prompt_id"] == spec.prompt_id for row in ar_null_rows
                ):
                    refs = reference_tokens_by_prompt[spec.prompt_id]
                    if len(refs) >= 2:
                        seed_a, tokens_a = refs[0]
                        seed_b, tokens_b = refs[1]
                        null_freq = token_frequency_metrics(
                            tokens_a,
                            tokens_b,
                            top_n=args.top_n,
                            block_size=args.block_size,
                            bootstrap_samples=args.bootstrap_samples,
                            seed=args.bootstrap_seed + prompt_index * 100_000 + 9999,
                        )
                        ar_null_rows.append(
                            {
                                "prompt_id": spec.prompt_id,
                                "suite": spec.suite,
                                "seed_a": int(seed_a),
                                "seed_b": int(seed_b),
                                "frequency": null_freq,
                                "fails_gate": _frequency_gate_fails(null_freq, args),
                            }
                        )
                        _progress(
                            args,
                            "AR-null "
                            f"prompt={spec.prompt_id} seeds={seed_a},{seed_b} "
                            f"p={_effective_pvalue(null_freq):.6g} "
                            f"kl={null_freq['kl_ref_to_mtplx_top']:.6g} "
                            f"fails={ar_null_rows[-1]['fails_gate']}",
                        )
                mtplx = _run_mtplx(
                    rt,
                    ids,
                    max_tokens=args.length,
                    sampler=t06_sampler,
                    seed=seed,
                    depth=args.depth,
                    verify_strategy=args.verify_strategy,
                    verify_core=args.verify_core,
                    mtp_history_policy=args.mtp_history_policy,
                )
                freq = token_frequency_metrics(
                    ref.tokens,
                    mtplx.tokens,
                    top_n=args.top_n,
                    block_size=args.block_size,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed + prompt_index * 100_000 + int(seed),
                )
                t06_rows.append(
                    {
                        "prompt_id": spec.prompt_id,
                        "suite": spec.suite,
                        "seed": int(seed),
                        "sequence": sequence_metrics(ref.tokens, mtplx.tokens),
                        "frequency": freq,
                        "reference": _generation_row(ref, include_tokens=args.include_tokens),
                        "mtplx": _generation_row(mtplx, include_tokens=args.include_tokens),
                    }
                )
                last = t06_rows[-1]
                pvalue = _effective_pvalue(last["frequency"])
                row_fails = _frequency_gate_fails(last["frequency"], args)
                if row_fails:
                    t06_failure_count += 1
                _progress(
                    args,
                    "T=0.6 done "
                    f"cell={len(t06_rows)} prompt={spec.prompt_id} seed={int(seed)} "
                    f"p={float(pvalue):.6g} "
                    f"kl={last['frequency']['kl_ref_to_mtplx_top']:.6g} "
                    f"fails={row_fails} "
                    f"ref_tok_s={last['reference']['tok_s']:.3f} "
                    f"mtplx_tok_s={last['mtplx']['tok_s']:.3f}",
                )
                if (
                    int(args.stop_after_t06_failures) > 0
                    and t06_failure_count >= int(args.stop_after_t06_failures)
                ):
                    early_stop_reason = (
                        f"candidate_t06_failures_reached_{int(args.stop_after_t06_failures)}"
                    )
                    _progress(args, f"early stop: {early_stop_reason}")
                    break
            if args.max_t06_cells is not None and len(t06_rows) >= int(args.max_t06_cells):
                break
            if early_stop_reason is not None:
                break

    t0_failures = [row for row in t0_rows if not row["sequence"]["exact_match"]]
    t06_failures = [row for row in t06_rows if _frequency_gate_fails(row["frequency"], args)]
    ar_null_failures = [row for row in ar_null_rows if row["fails_gate"]]

    status = "PASS"
    if t0_failures:
        status = "FAIL_T0_ARGMAX"
    elif len(t06_failures) >= args.max_t06_failed_cells:
        status = (
            "INCONCLUSIVE_T06_AR_NULL_FAILED"
            if ar_null_failures
            else "FAIL_T06_DISTRIBUTION"
        )

    elapsed = time.perf_counter() - started
    return {
        "schema": "mtplx.r1_chisquare_verifier_correctness.v1",
        "status": status,
        "elapsed_s": elapsed,
        "config": {
            "model": args.model,
            "profile": args.profile,
            "reference": "mtplx.generate_ar stock target path",
            "mtplx": {
                "depth": args.depth,
                "verify_strategy": args.verify_strategy,
                "verify_core": args.verify_core,
                "mtp_history_policy": args.mtp_history_policy,
            },
            "suites": suites,
            "prompts_per_suite": int(args.prompts_per_suite),
            "seeds": seeds,
            "length": int(args.length),
            "t0_length": int(args.t0_length),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "top_n": int(args.top_n),
            "block_size": int(args.block_size),
            "bootstrap_samples": int(args.bootstrap_samples),
            "pvalue_threshold": float(args.pvalue_threshold),
            "kl_threshold": float(args.kl_threshold),
            "max_t06_failed_cells": int(args.max_t06_failed_cells),
            "max_cells": args.max_cells,
            "max_t06_cells": args.max_t06_cells,
            "stop_after_t06_failures": int(args.stop_after_t06_failures),
            "ar_null_calibration": bool(args.ar_null_calibration),
            "enable_thinking": args.enable_thinking,
            "include_tokens": bool(args.include_tokens),
        },
        "profile_env_previous": previous_env,
        "draft_lm_head": accelerator_report,
        "prompts": prompt_rows,
        "summary": {
            "t0_cells": len(t0_rows),
            "t0_failures": len(t0_failures),
            "t06_cells": len(t06_rows),
            "t06_failures": len(t06_failures),
            "ar_null_cells": len(ar_null_rows),
            "ar_null_failures": len(ar_null_failures),
            "early_stop_reason": early_stop_reason,
            "t06_min_effective_pvalue": min(
                [_effective_pvalue(row["frequency"]) for row in t06_rows]
                or [1.0]
            ),
            "t06_max_kl_top": max(
                [row["frequency"]["kl_ref_to_mtplx_top"] for row in t06_rows] or [0.0]
            ),
        },
        "t0_argmax_rows": t0_rows,
        "ar_null_rows": ar_null_rows,
        "t06_rows": t06_rows,
    }


def run_paired_prefix_gate(args: argparse.Namespace) -> dict[str, Any]:
    """R1b paired-prefix verifier distribution gate.

    Unlike the original free-running R1, this gate does not compare two
    independently branched stochastic programs. It uses reference AR only to
    create realistic shared prefixes, then compares stock decode logits against
    the active paged verifier decode path at the exact same prefix cuts.
    """
    import mlx.core as mx

    from mtplx.mtp_patch import MTPContract
    from mtplx.runtime import load
    from mtplx.sampling import SamplerConfig

    suites = _parse_suites(args.suites)
    seeds = _parse_ints(args.seeds)
    cuts = _parse_positive_ints(args.paired_prefix_cuts)
    prompt_specs = _load_prompt_specs(suites, prompts_per_suite=args.prompts_per_suite)
    if args.max_cells is not None:
        prompt_specs = prompt_specs[: max(1, int(args.max_cells))]

    started = time.perf_counter()
    with _profile_env(args.profile) as previous_env:
        _progress(args, f"R1b loading model={args.model} profile={args.profile}")
        rt = load(args.model, mtp=True, contract=MTPContract())
        accelerator_report = _install_profile_accelerators(rt, args.profile)
        _progress(args, f"R1b model loaded; draft_lm_head={accelerator_report is not None}")
        sampler = SamplerConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        prompt_rows: list[dict[str, Any]] = []
        paired_rows: list[dict[str, Any]] = []
        source_rows: list[dict[str, Any]] = []
        insufficient_rows: list[dict[str, Any]] = []
        cell_count = 0

        for prompt_index, spec in enumerate(prompt_specs):
            ids = _encode_prompt(rt, spec, enable_thinking=args.enable_thinking)
            prompt_rows.append({**asdict(spec), "prompt_tokens": len(ids)})
            required_generated = max(0, max(cuts) - len(ids))
            source_tokens = (
                required_generated
                if args.paired_prefix_source_tokens is None
                else int(args.paired_prefix_source_tokens)
            )
            source_tokens = max(0, int(source_tokens))

            for seed in seeds:
                if args.max_paired_cells is not None and cell_count >= int(args.max_paired_cells):
                    break
                cell_count += 1
                _progress(
                    args,
                    f"R1b source cell={cell_count} prompt={spec.prompt_id} "
                    f"seed={int(seed)} source_tokens={source_tokens}",
                )
                if source_tokens:
                    ref = _run_reference_ar(
                        rt,
                        ids,
                        max_tokens=source_tokens,
                        sampler=sampler,
                        seed=seed,
                    )
                    generated = list(ref.tokens)
                    source_row = _generation_row(ref, include_tokens=False)
                else:
                    generated = []
                    source_row = {"generated_tokens": 0, "tokens_sha256": _token_sha256([])}
                full_prefix = list(ids) + generated
                source_rows.append(
                    {
                        "prompt_id": spec.prompt_id,
                        "suite": spec.suite,
                        "seed": int(seed),
                        "prompt_tokens": len(ids),
                        "source_generated_tokens": len(generated),
                        "full_prefix_tokens": len(full_prefix),
                        "prefix_sha256": _token_sha256(full_prefix),
                        "reference_ar": source_row,
                    }
                )

                for cut in cuts:
                    if cut < 2 or cut > len(full_prefix):
                        insufficient_rows.append(
                            {
                                "prompt_id": spec.prompt_id,
                                "suite": spec.suite,
                                "seed": int(seed),
                                "cut": int(cut),
                                "available_prefix_tokens": len(full_prefix),
                            }
                        )
                        continue
                    prefix = full_prefix[: int(cut)]
                    stock_logits, stock_elapsed = _decode_last_logits_from_prefix(
                        rt,
                        prefix,
                        args,
                        paged=False,
                    )
                    mx.clear_cache()
                    candidate_logits, candidate_elapsed = _decode_last_logits_from_prefix(
                        rt,
                        prefix,
                        args,
                        paged=True,
                    )
                    mx.clear_cache()
                    metrics = paired_prefix_distribution_metrics(
                        stock_logits,
                        candidate_logits,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        top_k_compare=args.paired_top_k_compare,
                        sample_seed=int(args.bootstrap_seed) + prompt_index * 100_000 + int(seed) + int(cut),
                        sample_draws=args.paired_sample_draws,
                    )
                    row = {
                        "prompt_id": spec.prompt_id,
                        "suite": spec.suite,
                        "seed": int(seed),
                        "cut": int(cut),
                        "stock_elapsed_s": float(stock_elapsed),
                        "candidate_elapsed_s": float(candidate_elapsed),
                        "metrics": metrics,
                    }
                    row["passed"] = _paired_prefix_row_passes(row, args)
                    paired_rows.append(row)
                    _progress(
                        args,
                        "R1b row "
                        f"prompt={spec.prompt_id} seed={int(seed)} cut={int(cut)} "
                        f"pass={row['passed']} "
                        f"maxdiff={metrics['logits']['max_abs_diff']:.6g} "
                        f"tv={metrics['distribution']['total_variation']:.6g} "
                        f"kl={metrics['distribution']['kl_stock_to_candidate']:.6g} "
                        f"topk={metrics['topk']['overlap_ratio']:.3f}",
                    )
            if args.max_paired_cells is not None and cell_count >= int(args.max_paired_cells):
                break

    failed_rows = [row for row in paired_rows if not row["passed"]]
    status = "PASS"
    if insufficient_rows:
        status = "FAIL_PAIR_PREFIX_INSUFFICIENT_SOURCE"
    elif failed_rows:
        status = "FAIL_PAIR_PREFIX_DISTRIBUTION"

    elapsed = time.perf_counter() - started
    return {
        "schema": "mtplx.r1b_paired_prefix_verifier_correctness.v1",
        "status": status,
        "elapsed_s": elapsed,
        "config": {
            "model": args.model,
            "profile": args.profile,
            "reference": "stock decode from shared AR prefix",
            "candidate": {
                "attention_impl": args.attention_impl,
                "attention_block_size": int(args.attention_block_size),
                "attention_num_blocks": int(args.attention_num_blocks),
                "attention_partitioned": bool(args.attention_partitioned),
                "attention_partition_threshold": int(args.attention_partition_threshold),
                "attention_partition_size": int(args.attention_partition_size),
            },
            "suites": suites,
            "prompts_per_suite": int(args.prompts_per_suite),
            "seeds": seeds,
            "paired_prefix_cuts": cuts,
            "paired_prefix_source_tokens": args.paired_prefix_source_tokens,
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "thresholds": {
                "max_logit_diff": float(args.paired_max_logit_diff),
                "max_total_variation": float(args.paired_max_total_variation),
                "max_kl": float(args.paired_max_kl),
                "min_topk_overlap": float(args.paired_min_topk_overlap),
                "min_sample_agreement": float(args.paired_min_sample_agreement),
            },
            "max_cells": args.max_cells,
            "max_paired_cells": args.max_paired_cells,
            "enable_thinking": args.enable_thinking,
        },
        "profile_env_previous": previous_env,
        "draft_lm_head": accelerator_report,
        "prompts": prompt_rows,
        "source_rows": source_rows,
        "summary": {
            "source_cells": len(source_rows),
            "paired_rows": len(paired_rows),
            "paired_failures": len(failed_rows),
            "insufficient_rows": len(insufficient_rows),
            "max_logit_diff": max(
                [row["metrics"]["logits"]["max_abs_diff"] for row in paired_rows]
                or [0.0]
            ),
            "max_total_variation": max(
                [row["metrics"]["distribution"]["total_variation"] for row in paired_rows]
                or [0.0]
            ),
            "max_kl_stock_to_candidate": max(
                [row["metrics"]["distribution"]["kl_stock_to_candidate"] for row in paired_rows]
                or [0.0]
            ),
            "min_topk_overlap": min(
                [row["metrics"]["topk"]["overlap_ratio"] for row in paired_rows]
                or [1.0]
            ),
            "min_sample_agreement": min(
                [row["metrics"]["controlled_rng_sample"]["agreement"] for row in paired_rows]
                or [1.0]
            ),
        },
        "insufficient_rows": insufficient_rows,
        "paired_rows": paired_rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("free-running", "paired-prefix"),
        default="free-running",
        help="Run original R1 free-running gate or R1b paired-prefix gate.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", default="performance-cold")
    parser.add_argument("--suites", default=",".join(DEFAULT_SUITES))
    parser.add_argument("--prompts-per-suite", type=int, default=3)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--length", type=int, default=10000)
    parser.add_argument("--t0-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--verify-strategy", default="capture_commit")
    parser.add_argument("--verify-core", default="linear-gdn-from-conv-tape")
    parser.add_argument("--mtp-history-policy", default="committed")
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=20260503)
    parser.add_argument("--pvalue-threshold", type=float, default=0.01)
    parser.add_argument("--kl-threshold", type=float, default=1e-3)
    parser.add_argument("--max-t06-failed-cells", type=int, default=2)
    parser.add_argument("--stop-after-t06-failures", type=int, default=2)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--max-t06-cells", type=int)
    parser.add_argument(
        "--paired-prefix-cuts",
        default=",".join(str(item) for item in DEFAULT_PAIR_PREFIX_CUTS),
    )
    parser.add_argument("--paired-prefix-source-tokens", type=int)
    parser.add_argument("--max-paired-cells", type=int)
    parser.add_argument("--paired-top-k-compare", type=int, default=20)
    parser.add_argument("--paired-sample-draws", type=int, default=256)
    parser.add_argument("--paired-max-logit-diff", type=float, default=3e-2)
    parser.add_argument("--paired-max-total-variation", type=float, default=5e-3)
    parser.add_argument("--paired-max-kl", type=float, default=1e-3)
    parser.add_argument("--paired-min-topk-overlap", type=float, default=0.95)
    parser.add_argument("--paired-min-sample-agreement", type=float, default=0.995)
    parser.add_argument("--attention-impl", default="mlx_vector_paged")
    parser.add_argument("--attention-block-size", type=int, default=16)
    parser.add_argument("--attention-num-blocks", type=int, default=1024)
    parser.add_argument(
        "--attention-partitioned",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--attention-partition-threshold", type=int, default=2048)
    parser.add_argument("--attention-partition-size", type=int, default=512)
    parser.add_argument("--include-tokens", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ar-null-calibration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_paired_prefix_gate(args) if args.mode == "paired-prefix" else run_gate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output),
                "summary": result["summary"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
