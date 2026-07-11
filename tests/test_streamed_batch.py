"""Stage 5 streamed continuous batching: runner semantics on tiny fixtures.

Outputs of batches with two or more live streams are intentionally not
compared against single-stream runs of the same prompt: batched kernels see
different shapes, so ``B > 1`` results are only comparable at equal batch
sizes (see ``mtplx/streamed_batch.py``).  The identity guarantee tested here
is the ``B = 1`` one.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mtplx.expert_manifest import build_expert_manifest, save_expert_manifest
from mtplx.expert_runtime import ExpertStreamingConfig, ExpertStreamingRuntime
from mtplx.expert_streaming_models import ExpertStreamingModelSpec
from mtplx.generation import generate_ar
from mtplx.models.expert_mlx import make_mlx_slot_buffer_allocator
from mtplx.models.hy3_mlx import Model as Hy3Model
from mtplx.models.hy3_mlx import ModelArgs as Hy3Args
from mtplx.mtp_patch import MTPContract
from mtplx.resident_loader import construct_resident_model
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig
from mtplx.streamed_batch import (
    StreamedBatchError,
    StreamedBatchRequest,
    StreamedBatchRunner,
)


class _FixtureTokenizer:
    """Minimal tokenizer: no implicit stop tokens, positional decode."""

    eos_token_id = None

    def decode(self, token_ids):
        return " ".join(str(int(token)) for token in token_ids)


def _hy3_args() -> Hy3Args:
    return Hy3Args(
        model_type="hy_v3",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        num_shared_experts=1,
        first_k_dense_replace=1,
        rms_norm_eps=1e-5,
        vocab_size=128,
        max_position_embeddings=128,
        head_dim=16,
        router_scaling_factor=2.0,
    )


def _batched_hy3_artifact(tmp_path: Path):
    """Tiny Hy3 artifact whose two experts produce distinct outputs."""

    mx.random.seed(11)
    args = _hy3_args()
    model = Hy3Model(args)
    weights = dict(tree_flatten(model.parameters()))
    expert_shapes = {
        "gate_proj.weight": (2, 64, 8),
        "gate_proj.scales": (2, 64, 1),
        "gate_proj.biases": (2, 64, 1),
        "up_proj.weight": (2, 64, 8),
        "up_proj.scales": (2, 64, 1),
        "up_proj.biases": (2, 64, 1),
        "down_proj.weight": (2, 64, 8),
        "down_proj.scales": (2, 64, 1),
        "down_proj.biases": (2, 64, 1),
    }
    # Distinct non-zero Q4 payloads per expert so routing changes the output:
    # every nibble of expert 0 dequantizes to 1 * scale, expert 1 to 3 * scale.
    nibble_fill = {0: 0x11111111, 1: 0x33333333}
    for component, shape in expert_shapes.items():
        if component.endswith("weight"):
            rows = mx.stack(
                [
                    mx.full(shape[1:], nibble_fill[expert], dtype=mx.uint32)
                    for expert in range(shape[0])
                ]
            )
            weights[f"model.layers.1.mlp.switch_mlp.{component}"] = rows
        elif component.endswith("scales"):
            weights[f"model.layers.1.mlp.switch_mlp.{component}"] = mx.full(
                shape, 0.03125, dtype=mx.bfloat16
            )
        else:
            weights[f"model.layers.1.mlp.switch_mlp.{component}"] = mx.zeros(
                shape, dtype=mx.bfloat16
            )
    mx.eval(weights)
    root = tmp_path / "hy3"
    root.mkdir()
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    config = asdict(args)
    config["model_type"] = "hy_v3"
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    total_bytes = sum(int(value.nbytes) for value in weights.values())
    spec = ExpertStreamingModelSpec(
        key="tiny-hy3-q4",
        display_name="Tiny Hy3 Q4",
        source_model="test/tiny-hy3",
        source_revision="source",
        quant_model="test/tiny-hy3-q4",
        quant_revision="quant",
        total_tensor_bytes=total_bytes,
        total_layers=2,
        routed_layer_start=1,
        routed_layer_count=1,
        expert_count=2,
        top_k=1,
        hidden_size=64,
        expert_hidden_size=64,
        quant_bits=4,
        quant_group_size=64,
        quant_parameter_bytes=2,
        router_storage="float32",
        router_matmul_dtype="float32",
        router_bytes=2 * 64 * 4 + 2 * 4,
        kv_bytes_per_token=0,
        mtp_layer_index=2,
        mtp_included=False,
    )
    manifest = build_expert_manifest(root, spec)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    return root, config, spec, manifest_path


def _open_fixture_runtime(tmp_path: Path, *, max_live_kv_tokens: int):
    root, config, spec, manifest_path = _batched_hy3_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        max_live_kv_tokens=max_live_kv_tokens,
        runtime_reserve_bytes=0,
    )
    streaming = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_slot_buffer_allocator(
            stream_config.memory_plan(spec), spec
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    try:
        resident = construct_resident_model(root, streaming, config=config)
    except BaseException:
        streaming.close()
        raise
    rt = MTPLXRuntime(
        model=resident.model,
        tokenizer=_FixtureTokenizer(),
        model_path=root,
        mtp_enabled=False,
        contract=MTPContract(),
        expert_streaming=streaming,
    )
    return rt, streaming


def _request(
    request_id: str,
    prompt: list[int],
    *,
    max_tokens: int,
    sampler: SamplerConfig | None = None,
    seed: int = 0,
    stop_token_ids: frozenset[int] | None = frozenset(),
) -> StreamedBatchRequest:
    return StreamedBatchRequest(
        request_id=request_id,
        prompt_ids=tuple(prompt),
        max_tokens=max_tokens,
        sampler=sampler or SamplerConfig(temperature=0.0, top_p=1.0, top_k=1),
        seed=seed,
        stop_token_ids=stop_token_ids,
    )


@pytest.mark.parametrize(
    "sampler,seed",
    [
        (SamplerConfig(temperature=0.0, top_p=1.0, top_k=1), 0),
        (SamplerConfig(temperature=0.9, top_p=1.0, top_k=3), 11),
    ],
    ids=["greedy", "sampled"],
)
def test_single_stream_runner_matches_generate_ar_tokens(
    tmp_path: Path, sampler: SamplerConfig, seed: int
) -> None:
    rt, streaming = _open_fixture_runtime(tmp_path, max_live_kv_tokens=64)
    try:
        prompt = [1, 2, 3, 4]
        reference = generate_ar(
            rt,
            prompt,
            max_tokens=8,
            sampler=sampler,
            seed=seed,
            stop_token_ids=set(),
        )
        runner = StreamedBatchRunner(rt, max_concurrency=1)
        runner.submit(
            _request("solo", prompt, max_tokens=8, sampler=sampler, seed=seed)
        )
        results = runner.run()

        assert len(results) == 1
        assert list(results[0].tokens) == reference.tokens
        assert results[0].finish_reason == reference.finish_reason
        assert results[0].text == reference.text
        assert streaming.snapshot()["live_kv_tokens"] == 0
    finally:
        rt.close()


def test_two_streams_keep_outputs_and_kv_isolated_per_sequence(
    tmp_path: Path,
) -> None:
    rt, streaming = _open_fixture_runtime(tmp_path, max_live_kv_tokens=64)
    try:
        prompt_a = [1, 2, 3]
        prompt_b = [7, 8, 9, 10]
        states: list = []

        def run_batch(first: str) -> dict[str, tuple[int, ...]]:
            runner = StreamedBatchRunner(
                rt, max_concurrency=2, on_step=states.append
            )
            requests = {
                "a": _request("a", prompt_a, max_tokens=5),
                "b": _request("b", prompt_b, max_tokens=5),
            }
            second = "b" if first == "a" else "a"
            runner.submit(requests[first])
            runner.submit(requests[second])
            return {
                result.request_id: result.tokens for result in runner.run()
            }

        first_order = run_batch("a")
        second_order = run_batch("b")

        # Outputs follow the request, not the batch slot it happened to fill.
        assert first_order["a"] == second_order["a"]
        assert first_order["b"] == second_order["b"]
        assert len(first_order["a"]) == 5
        assert len(first_order["b"]) == 5

        # Each stream advances its own KV cache by exactly its own tokens:
        # after the prefill (prompt positions) every decode step appends one
        # position for the previously sampled token.
        prompt_tokens = {"a": len(prompt_a), "b": len(prompt_b)}
        seen_live = set()
        for state in states:
            for view in state.live:
                seen_live.add(view.request_id)
                assert view.cache_offset == (
                    prompt_tokens[view.request_id] + view.generated_tokens - 1
                )
        assert seen_live == {"a", "b"}
        assert streaming.snapshot()["live_kv_tokens"] == 0
    finally:
        rt.close()


def test_stream_finishing_midbatch_releases_kv_and_batch_continues(
    tmp_path: Path,
) -> None:
    rt, streaming = _open_fixture_runtime(tmp_path, max_live_kv_tokens=32)
    try:
        states: list = []
        runner = StreamedBatchRunner(rt, max_concurrency=2, on_step=states.append)
        runner.submit(_request("long", [1, 2, 3], max_tokens=6))
        runner.submit(_request("short", [4, 5, 6], max_tokens=3))
        results = {result.request_id: result for result in runner.run()}

        assert len(results["long"].tokens) == 6
        assert len(results["short"].tokens) == 3
        assert results["long"].finish_reason == "length"
        assert results["short"].finish_reason == "length"
        assert results["short"].finished_step < results["long"].finished_step

        both_live = [state for state in states if len(state.live) == 2]
        assert both_live, "streams never decoded concurrently"
        assert both_live[0].reserved_kv_tokens == (3 + 6) + (3 + 3)
        long_only = [
            state
            for state in states
            if [view.request_id for view in state.live] == ["long"]
        ]
        assert long_only, "batch did not continue after the short stream ended"
        assert all(state.reserved_kv_tokens == 3 + 6 for state in long_only)
        assert streaming.snapshot()["live_kv_tokens"] == 0
    finally:
        rt.close()


def test_continuous_admission_joins_at_a_later_step_boundary(
    tmp_path: Path,
) -> None:
    rt, streaming = _open_fixture_runtime(tmp_path, max_live_kv_tokens=64)
    try:
        states: list = []
        runner = StreamedBatchRunner(rt, max_concurrency=2, on_step=states.append)
        runner.submit(_request("r1", [1, 2, 3], max_tokens=3))
        runner.submit(_request("r2", [4, 5, 6], max_tokens=6))
        runner.submit(_request("r3", [7, 8, 9], max_tokens=3))
        results = {result.request_id: result for result in runner.run()}

        # r3 could not start with the batch full; it joined at a decode step
        # boundary after r1 finished, while r2 was still decoding.
        assert results["r1"].admitted_step == 0
        assert results["r2"].admitted_step == 0
        assert results["r3"].admitted_step > results["r1"].finished_step - 1
        overlapped = [
            state
            for state in states
            if {"r2", "r3"}
            <= {view.request_id for view in state.live}
        ]
        assert overlapped, "r3 never decoded next to a still-live stream"
        assert all(len(result.tokens) > 0 for result in results.values())
        assert streaming.snapshot()["live_kv_tokens"] == 0
    finally:
        rt.close()


def test_mid_run_submission_joins_from_the_step_hook(tmp_path: Path) -> None:
    rt, streaming = _open_fixture_runtime(tmp_path, max_live_kv_tokens=64)
    try:
        runner_box: list[StreamedBatchRunner] = []
        submitted: list[int] = []

        def on_step(state) -> None:
            if state.step == 2 and not submitted:
                submitted.append(state.step)
                runner_box[0].submit(_request("late", [9, 10], max_tokens=2))

        runner = StreamedBatchRunner(rt, max_concurrency=2, on_step=on_step)
        runner_box.append(runner)
        runner.submit(_request("early", [1, 2, 3], max_tokens=8))
        results = {result.request_id: result for result in runner.run()}

        assert set(results) == {"early", "late"}
        assert results["late"].admitted_step == 3
        assert results["late"].finish_reason == "length"
        assert len(results["late"].tokens) == 2
        assert streaming.snapshot()["live_kv_tokens"] == 0
    finally:
        rt.close()


def test_per_request_stop_tokens_finish_with_stop_reason(tmp_path: Path) -> None:
    rt, streaming = _open_fixture_runtime(tmp_path, max_live_kv_tokens=64)
    try:
        prompt_a = [1, 2, 3]
        prompt_b = [7, 8, 9, 10]

        def run_pair(stop_b: frozenset[int]):
            runner = StreamedBatchRunner(rt, max_concurrency=2)
            runner.submit(_request("a", prompt_a, max_tokens=6))
            runner.submit(
                _request("b", prompt_b, max_tokens=6, stop_token_ids=stop_b)
            )
            return {result.request_id: result for result in runner.run()}

        # Learn stream b's deterministic second token, then re-run with it as
        # b's stop token: b must end early with "stop" while a is unaffected
        # in length and reason.
        dry = run_pair(frozenset())
        stop_token = int(dry["b"].tokens[1])
        wet = run_pair(frozenset({stop_token}))

        assert wet["b"].finish_reason == "stop"
        assert wet["b"].tokens[-1] == stop_token
        assert len(wet["b"].tokens) <= len(dry["b"].tokens)
        assert wet["a"].finish_reason == "length"
        assert len(wet["a"].tokens) == 6
        # The terminal stop token never leaks into the decoded text.
        assert str(stop_token) not in wet["b"].text.split()[-1:]
        assert streaming.snapshot()["live_kv_tokens"] == 0
    finally:
        rt.close()


def test_kv_admission_is_fail_closed_for_impossible_requests(
    tmp_path: Path,
) -> None:
    rt, streaming = _open_fixture_runtime(tmp_path, max_live_kv_tokens=8)
    try:
        runner = StreamedBatchRunner(rt, max_concurrency=2)
        runner.submit(_request("too-big", [1, 2, 3], max_tokens=6))
        with pytest.raises(StreamedBatchError, match="KV"):
            runner.run()
        assert streaming.snapshot()["live_kv_tokens"] == 0
    finally:
        rt.close()


def test_kv_budget_serializes_streams_instead_of_overcommitting(
    tmp_path: Path,
) -> None:
    rt, streaming = _open_fixture_runtime(tmp_path, max_live_kv_tokens=8)
    try:
        states: list = []
        runner = StreamedBatchRunner(rt, max_concurrency=2, on_step=states.append)
        runner.submit(_request("first", [1, 2, 3], max_tokens=3))
        runner.submit(_request("second", [4, 5, 6], max_tokens=3))
        results = {result.request_id: result for result in runner.run()}

        # Each stream needs 6 of the 8 planned KV tokens, so they must never
        # decode concurrently, and both must still complete.
        assert all(len(state.live) <= 1 for state in states)
        assert results["second"].admitted_step > results["first"].finished_step - 1
        assert len(results["first"].tokens) == 3
        assert len(results["second"].tokens) == 3
        assert streaming.snapshot()["live_kv_tokens"] == 0
    finally:
        rt.close()


def test_runner_rejects_duplicate_and_empty_submissions(tmp_path: Path) -> None:
    rt, _streaming = _open_fixture_runtime(tmp_path, max_live_kv_tokens=16)
    try:
        runner = StreamedBatchRunner(rt, max_concurrency=2)
        with pytest.raises(StreamedBatchError, match="no requests"):
            runner.run()
        runner.submit(_request("dup", [1, 2], max_tokens=1))
        with pytest.raises(ValueError, match="duplicate"):
            runner.submit(_request("dup", [3, 4], max_tokens=1))
    finally:
        rt.close()
