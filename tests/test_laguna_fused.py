"""Contracts for the Laguna fused decode paths and the AR batched lane.

These run on the CPU device at toy geometry so they are cheap enough for CI and
do not need the 60 GB checkpoint or a GPU window.

The bar each fused path has to clear is stated per test rather than assumed:
a path that changes the greedy tokens is a defect; a float32 value delta well
under bfloat16 resolution is not, because it cannot survive the cast into the
dtype the real checkpoint runs in.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.models import laguna, laguna_fused
from mtplx.models.laguna import Model, ModelArgs

# A float32 delta this far below bf16's ~8e-3 resolution cannot change the
# model's own arithmetic.
BF16_SAFE_TOLERANCE = 1e-5

LAYER_TYPES = [
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
]


def _toy_args(**updates):
    config = dict(
        model_type="laguna",
        hidden_size=64,
        num_hidden_layers=len(LAYER_TYPES),
        intermediate_size=128,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=256,
        rms_norm_eps=1e-6,
        num_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        mlp_only_layers=[0],
        gating="per-head",
        sliding_window=8,
        layer_types=list(LAYER_TYPES),
        rope_parameters={
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 500_000.0,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 1.0,
            },
        },
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    config.update(updates)
    return ModelArgs(**config)


@pytest.fixture
def toy_model():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(3)
    model = Model(_toy_args())
    mx.eval(model.parameters())
    stock_moe = laguna.LagunaSparseMoeBlock.__call__
    stock_forward = laguna.LagunaModel.__call__
    try:
        yield model
    finally:
        laguna.LagunaSparseMoeBlock.__call__ = stock_moe
        laguna.LagunaModel.__call__ = stock_forward
        laguna.PER_HEAD_GATE_IMPL = laguna._stock_per_head_gate
        mx.set_default_device(previous)


PROMPTS = mx.array(
    [
        [3, 9, 14, 2, 7, 21, 5, 11, 30, 1, 18, 6],
        [17, 4, 25, 8, 13, 0, 29, 22, 10, 16, 27, 19],
    ],
    dtype=mx.uint32,
)


def _greedy(model, prompts, steps: int):
    cache = model.make_cache()
    logits = model(prompts, cache=cache, logits_keep=1)
    token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)[:, None]
    rows = [token]
    for _ in range(steps):
        logits = model(token, cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)[:, None]
        rows.append(token)
    stacked = mx.concatenate(rows, axis=1)
    mx.eval(stacked)
    return stacked.tolist()


def _last_logits(model, prompts):
    cache = model.make_cache()
    out = model(prompts, cache=cache, logits_keep=1)
    mx.eval(out)
    return out


@pytest.mark.parametrize(
    "installer, bit_exact",
    [
        (laguna_fused.install_compiled_router, True),
        (laguna_fused.install_fused_residual_norm, True),
        (laguna_fused.install_compiled_attention_gate, False),
        (laguna_fused.install_kernel_attention_gate, False),
    ],
)
def test_fused_path_preserves_greedy_output(toy_model, installer, bit_exact):
    """No fused path may change what the model actually generates."""

    reference_tokens = _greedy(toy_model, PROMPTS, 5)
    reference_logits = _last_logits(toy_model, PROMPTS)

    installer(toy_model)

    assert _greedy(toy_model, PROMPTS, 5) == reference_tokens
    delta = float(mx.abs(_last_logits(toy_model, PROMPTS) - reference_logits).max())
    assert delta <= BF16_SAFE_TOLERANCE
    if bit_exact:
        assert delta == 0.0, f"expected bit-exact, got {delta:.3e}"


def test_fused_gate_up_is_bit_exact_when_quantized(toy_model):
    """The gate/up concatenation must not perturb a quantized expert at all.

    Quantization groups run along the INPUT dimension, so concatenating output
    rows carries each row's own scales and biases untouched. That is the whole
    argument for the transform being safe, and it only holds for the quantized
    path — which is the one the real checkpoint uses.
    """

    nn.quantize(toy_model, group_size=32, bits=4)
    mx.eval(toy_model.parameters())

    reference_tokens = _greedy(toy_model, PROMPTS, 5)
    reference_logits = _last_logits(toy_model, PROMPTS)

    report = laguna_fused.install_fused_gate_up(toy_model)
    assert report["layers_converted"] == 3  # layer 0 is dense

    assert _greedy(toy_model, PROMPTS, 5) == reference_tokens
    delta = float(mx.abs(_last_logits(toy_model, PROMPTS) - reference_logits).max())
    assert delta == 0.0, f"gate/up fusion was not bit-exact: {delta:.3e}"


def test_fused_gate_up_is_idempotent(toy_model):
    """Installing twice must not concatenate an already-fused layer again."""

    nn.quantize(toy_model, group_size=32, bits=4)
    mx.eval(toy_model.parameters())
    assert laguna_fused.install_fused_gate_up(toy_model)["layers_converted"] == 3
    assert laguna_fused.install_fused_gate_up(toy_model)["layers_converted"] == 0


def test_residual_norm_install_reports_whether_the_kernel_engaged(toy_model):
    """A silent fallback must be visible, not read as 'the fusion bought nothing'."""

    report = laguna_fused.install_fused_residual_norm(toy_model)
    assert "kernel_engaged" in report
    assert isinstance(report["kernel_engaged"], bool)


def test_sort_pin_overrides_the_stock_heuristic():
    laguna_fused.SORT_DECISION = None
    small = mx.zeros((2, 4), dtype=mx.uint32)
    large = mx.zeros((32, 4), dtype=mx.uint32)
    try:
        assert laguna_fused.should_sort(small) is False
        assert laguna_fused.should_sort(large) is True
        laguna_fused.SORT_DECISION = True
        assert laguna_fused.should_sort(small) is True
        laguna_fused.SORT_DECISION = False
        assert laguna_fused.should_sort(large) is False
    finally:
        laguna_fused.SORT_DECISION = None


# ---------------------------------------------------------------------------
# the AR batched lane, opened to target-only runtimes
# ---------------------------------------------------------------------------
class _TargetOnlyRuntime:
    """A runtime with no MTP head, returning logits ONLY — like Laguna's."""

    def __init__(self, model):
        self.model = model
        self.mtp_enabled = False

    def forward_ar(self, input_ids, cache=None, logits_keep=None, **kwargs):
        if kwargs.get("return_hidden"):
            raise AssertionError(
                "the AR lane must not ask a target-only runtime for hidden "
                "states; it returns logits only"
            )
        return self.model(input_ids, cache=cache, logits_keep=logits_keep)

    def make_cache(self):
        return self.model.make_cache()


def test_ar_batched_decode_runs_without_an_mtp_head(toy_model):
    """decode_mode='ar' needs no draft head, so it must not require one."""

    from mtplx.batched_decode import generate_greedy_batched

    prompts = [row for row in PROMPTS.tolist()]
    result = generate_greedy_batched(
        _TargetOnlyRuntime(toy_model), prompts, max_new_tokens=4, decode_mode="ar"
    )
    assert len(result.streams) == len(prompts)
    assert all(len(stream.tokens) == 4 for stream in result.streams)
    # One forward per cycle serving every stream is the point of the lane.
    assert result.forwards <= result.cycles + 1


def test_spec_lane_still_requires_an_mtp_head(toy_model):
    """Opening the AR lane must not open the speculative one."""

    from mtplx.batched_decode import generate_greedy_batched

    with pytest.raises(RuntimeError, match="MTP-enabled"):
        generate_greedy_batched(
            _TargetOnlyRuntime(toy_model),
            [row for row in PROMPTS.tolist()],
            max_new_tokens=2,
            decode_mode="spec",
        )


def test_ar_batched_streams_match_running_each_prompt_alone(toy_model):
    """The correctness contract: batching must not change a stream's output.

    Held at a FIXED cohort shape so both runs use the same kernels — a
    difference then rests solely on per-row forward independence rather than on
    a batched matmul reducing in a different order.
    """

    from mtplx.batched_decode import generate_greedy_batched

    runtime = _TargetOnlyRuntime(toy_model)
    prompts = [row for row in PROMPTS.tolist()]
    slots = len(prompts)

    batched = generate_greedy_batched(
        runtime, prompts, max_new_tokens=4, decode_mode="ar", cohort_slots=slots
    )
    for index, prompt in enumerate(prompts):
        solo = generate_greedy_batched(
            runtime, [prompt], max_new_tokens=4, decode_mode="ar",
            cohort_slots=slots,
        )
        assert batched.streams[index].sha == solo.streams[0].sha, (
            f"stream {index} diverged when batched alongside other prompts"
        )
