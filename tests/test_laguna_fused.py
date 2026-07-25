"""Contracts for the Laguna fused decode paths and the AR batched lane.

These run on the CPU device at toy geometry so they are cheap enough for CI and
do not need the 60 GB checkpoint or a GPU window.

The bar each fused path has to clear is stated per test rather than assumed:
a path that changes the greedy tokens is a defect; a float32 value delta well
under bfloat16 resolution is not, because it cannot survive the cast into the
dtype the real checkpoint runs in.
"""

from __future__ import annotations

import math

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
    stock_attention = laguna.Attention.__call__
    try:
        yield model
    finally:
        laguna.LagunaSparseMoeBlock.__call__ = stock_moe
        laguna.LagunaModel.__call__ = stock_forward
        laguna.Attention.__call__ = stock_attention
        laguna.PER_HEAD_GATE_IMPL = laguna._stock_per_head_gate
        laguna.MOE_COMBINE_IMPL = laguna._stock_moe_combine
        laguna_fused.reset_cached_gather_indices()
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
        (laguna_fused.install_kernel_qk_rope, True),
        (laguna_fused.install_kernel_moe_combine, True),
        (laguna_fused.install_cached_gather_indices, True),
        # UNQUANTIZED float32 here, which is the one shape the q/k/v/g fusion is
        # NOT bit-exact in, for two reasons that are both MLX kernel selection
        # rather than arithmetic: the fused `x @ W.T` is a wider gemm and CPU
        # BLAS blocks it differently, and at float32 the gate's
        # `.astype(mx.float32)` is a no-op, so `logaddexp` sees a strided slice
        # and takes its scalar path instead of its SIMD one.  Both vanish at the
        # checkpoint's own dtype; `test_fused_qkvg_is_bit_exact_when_quantized`
        # pins delta == 0.0 there.
        (laguna_fused.install_fused_qkvg, False),
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


def test_qk_rope_install_reads_the_layer_rope_modules(toy_model):
    """The installer must capture each layer's own rope constants, not guess.

    On the toy geometry (head_dim 8) no layer is kernel-covered, but the specs
    still have to reflect the modules exactly: default-rope layers carry
    log2(theta) and the full-attention layers' partial rotary width.
    """

    report = laguna_fused.install_kernel_qk_rope(toy_model)
    assert report["layers_covered"] == 0  # head_dim 8 stays on the stock path
    assert report["layers_skipped"] == len(LAYER_TYPES)

    layers = toy_model.model.layers
    full = layers[0].self_attn._qk_rope_spec
    sliding = layers[1].self_attn._qk_rope_spec
    assert full.rot_dims == 4  # head_dim 8 * partial_rotary_factor 0.5
    assert full.base_log2 == pytest.approx(math.log2(500_000.0))
    assert full.mscale is None and full.freqs is None
    assert sliding.rot_dims == 8
    assert sliding.base_log2 == pytest.approx(math.log2(10_000.0))


def test_qk_rope_spec_for_yarn_captures_freqs_and_mscale():
    """A YaRN rope must contribute its own freqs buffer and attention factor."""

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        args = _toy_args(
            head_dim=128,
            hidden_size=256,
            num_attention_heads=2,
            num_key_value_heads=2,
            rope_parameters={
                "full_attention": {
                    "rope_type": "yarn",
                    "rope_theta": 500_000.0,
                    "factor": 128.0,
                    "original_max_position_embeddings": 8192,
                    "beta_fast": 32.0,
                    "beta_slow": 1.0,
                    "partial_rotary_factor": 0.5,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                    "partial_rotary_factor": 1.0,
                },
            },
        )
        model = Model(args)
        spec = laguna_fused._qk_rope_spec_for(model.model.layers[0].self_attn)
        assert spec is not None
        assert spec.rot_dims == 64
        assert spec.freqs is not None and int(spec.freqs.size) == 32
        assert spec.base_log2 is None
        # yarn_get_mscale(128, 1) — what mlx-lm computes for this config.
        assert spec.mscale == pytest.approx(0.1 * math.log(128.0) + 1.0)

        sliding_spec = laguna_fused._qk_rope_spec_for(
            model.model.layers[1].self_attn
        )
        assert sliding_spec is not None
        assert sliding_spec.rot_dims == 128
        assert sliding_spec.freqs is None and sliding_spec.mscale is None
    finally:
        mx.set_default_device(previous)


def test_moe_combine_fallback_matches_stock_expression(toy_model):
    """The CPU fallback inside fused_moe_combine IS the stock arithmetic."""

    from mtplx.kernels.laguna_decode import fused_moe_combine

    mx.random.seed(11)
    expert_out = mx.random.normal((3, 4, 16)).astype(mx.bfloat16)
    weights = mx.random.uniform(shape=(3, 4)).astype(mx.float32)
    shared = mx.random.normal((3, 16)).astype(mx.bfloat16)

    stock = (
        expert_out * weights.astype(mx.bfloat16)[..., None]
    ).sum(axis=-2) + shared
    fused = fused_moe_combine(expert_out, weights, shared)
    assert float(mx.abs(stock - fused).astype(mx.float32).max()) == 0.0


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


def test_fused_shared_gate_up_is_bit_exact_when_quantized(toy_model):
    """Concatenating the dense-MLP gate/up rows must not perturb anything.

    Covers both flavors the installer touches: the layer-0 dense block and the
    per-layer shared experts.
    """

    nn.quantize(toy_model, group_size=32, bits=4)
    mx.eval(toy_model.parameters())

    reference_tokens = _greedy(toy_model, PROMPTS, 5)
    reference_logits = _last_logits(toy_model, PROMPTS)

    report = laguna_fused.install_fused_shared_gate_up(toy_model)
    # layer 0 dense + three sparse layers' shared experts
    assert report["layers_converted"] == 4

    assert _greedy(toy_model, PROMPTS, 5) == reference_tokens
    delta = float(mx.abs(_last_logits(toy_model, PROMPTS) - reference_logits).max())
    assert delta == 0.0, f"shared gate/up fusion was not bit-exact: {delta:.3e}"


def test_cached_lhs_indices_is_bit_exact_when_quantized(toy_model):
    """The cached lhs_indices must reproduce MLX's default arange exactly.

    Covers the QuantizedSwitchLinear path the real checkpoint runs, on top of
    the gate/up-fused expert bank, at both sort settings.
    """

    nn.quantize(toy_model, group_size=32, bits=4)
    mx.eval(toy_model.parameters())
    laguna_fused.install_fused_gate_up(toy_model)

    reference_tokens = _greedy(toy_model, PROMPTS, 5)
    reference_logits = _last_logits(toy_model, PROMPTS)

    laguna_fused.install_cached_gather_indices(toy_model)
    for sort_decision in (False, True):
        laguna_fused.SORT_DECISION = sort_decision
        try:
            assert _greedy(toy_model, PROMPTS, 5) == reference_tokens
            delta = float(
                mx.abs(_last_logits(toy_model, PROMPTS) - reference_logits).max()
            )
            assert delta == 0.0, (
                f"cached lhs (sort={sort_decision}) not bit-exact: {delta:.3e}"
            )
        finally:
            laguna_fused.SORT_DECISION = None


def _as_shipped(model):
    """Put the toy in the shipped checkpoint's shape: quantized, bfloat16.

    Both matter for what the q/k/v/g fusion can be held to.  Quantized is the
    arithmetic the concatenation argument is about, and bfloat16 is what makes
    the attention gate's ``.astype(mx.float32)`` a REAL cast — at float32 it is
    a no-op, which leaves ``logaddexp`` looking at a strided slice and taking a
    scalar path that differs from its SIMD one in the last ulp.  That is MLX
    kernel selection, not the fusion, and it does not exist at the dtype the
    checkpoint runs in.
    """

    nn.quantize(model, group_size=32, bits=4)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def test_fused_qkvg_projection_slices_are_bit_exact(toy_model):
    """The concatenation itself, priced directly against the four projections.

    Quantization groups run along the INPUT dimension, so stacking output ROWS
    carries each row's own scales and biases untouched and every output element
    is the same products summed in the same order.  Checked per layer at both
    the decode shape and a prefill shape, and non-destructively (the module is
    built by hand, not installed), so this is a statement about the transform
    alone with no forward wrapped around it.
    """

    nn.quantize(toy_model, group_size=32, bits=4)
    mx.eval(toy_model.parameters())
    hidden = int(toy_model.args.hidden_size)

    for index, layer in enumerate(toy_model.model.layers):
        attention = layer.self_attn
        fused = laguna_fused.FusedQkvgProj(attention)
        mx.eval(fused.parameters())
        for shape in ((1, 1, hidden), (2, 12, hidden)):
            x = mx.random.normal(shape)
            mx.eval(x)
            want = (
                attention.q_proj(x),
                attention.k_proj(x),
                attention.v_proj(x),
                attention.g_proj(x),
            )
            got = fused(x)
            mx.eval(want, got)
            for name, a, b in zip(("q", "k", "v", "g"), got, want):
                delta = float(mx.abs(a - b).max())
                assert delta == 0.0, (
                    f"layer {index} {name} at {shape} was not bit-exact: {delta:.3e}"
                )


def test_fused_qkvg_is_bit_exact_when_quantized(toy_model):
    """No end-to-end effect on the model the checkpoint actually is.

    The bar is the strict one — identical greedy tokens AND a zero logits delta
    — because the transform is exact by construction and the shapes it feeds
    (norms, rope, attention, the gate) are the shipped ones unchanged.
    """

    _as_shipped(toy_model)

    reference_tokens = _greedy(toy_model, PROMPTS, 5)
    reference_logits = _last_logits(toy_model, PROMPTS)

    report = laguna_fused.install_fused_qkvg(toy_model)
    assert report["layers_converted"] == len(LAYER_TYPES)
    assert report["layers_skipped"] == 0

    assert _greedy(toy_model, PROMPTS, 5) == reference_tokens
    delta = float(mx.abs(_last_logits(toy_model, PROMPTS) - reference_logits).max())
    assert delta == 0.0, f"q/k/v/g fusion was not bit-exact: {delta:.3e}"


def test_fused_qkvg_drops_the_projections_it_concatenated(toy_model):
    """The install is destructive by design: one copy of the weights, not two.

    Pinned as a contract because it is also the hazard — after this install any
    code path that reaches for ``attention.q_proj`` is broken, which is why the
    forward is swapped in the same call.
    """

    _as_shipped(toy_model)
    laguna_fused.install_fused_qkvg(toy_model)

    for layer in toy_model.model.layers:
        attention = layer.self_attn
        assert attention._qkvg is not None
        for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
            assert name not in attention, f"{name} survived the fusion"


@pytest.mark.parametrize("rope_first", [False, True])
def test_fused_qkvg_composes_with_the_qk_rope_kernel(toy_model, rope_first):
    """Both attention installs must compose, in either install order.

    They both swap ``Attention.__call__``.  The qk-rope forward reads ``_qkvg``
    itself and falls back THROUGH the q/k/v/g dispatcher, so it covers strictly
    more; the rule is that it wins whichever order the two are installed in,
    and neither may capture the other's patch as "the pristine call".

    On this geometry (head_dim 8) the rope kernel never engages, so what is
    exercised is exactly the fallback wiring — the part that would recurse or
    call a dropped ``q_proj`` if the composition were wrong.
    """

    _as_shipped(toy_model)

    reference_tokens = _greedy(toy_model, PROMPTS, 5)
    reference_logits = _last_logits(toy_model, PROMPTS)

    if rope_first:
        laguna_fused.install_kernel_qk_rope(toy_model)
        laguna_fused.install_fused_qkvg(toy_model)
    else:
        laguna_fused.install_fused_qkvg(toy_model)
        laguna_fused.install_kernel_qk_rope(toy_model)

    assert laguna.Attention.__call__ is laguna_fused._kernel_attention_call
    assert laguna_fused._STOCK_ATTENTION_CALL is not None
    assert laguna_fused._STOCK_ATTENTION_CALL not in (
        laguna_fused._kernel_attention_call,
        laguna_fused._qkvg_attention_dispatch,
    )

    assert _greedy(toy_model, PROMPTS, 5) == reference_tokens
    delta = float(mx.abs(_last_logits(toy_model, PROMPTS) - reference_logits).max())
    assert delta == 0.0, f"composed installs were not bit-exact: {delta:.3e}"


def test_install_from_env_runs_qkvg_after_the_qk_rope_kernel(toy_model, monkeypatch):
    """The env entry point is where the destructive ordering has to hold.

    ``install_fused_qkvg`` drops the projections, so it runs last, and it must
    not downgrade the qk-rope forward that was installed before it.
    """

    _as_shipped(toy_model)
    reference_tokens = _greedy(toy_model, PROMPTS, 5)
    reference_logits = _last_logits(toy_model, PROMPTS)

    monkeypatch.setenv(laguna_fused.ENV_KERNEL_QK_ROPE, "1")
    monkeypatch.setenv(laguna_fused.ENV_FUSED_QKVG, "1")
    report = laguna_fused.install_from_env(toy_model)

    assert [entry["path"] for entry in report] == ["kernel_qk_rope", "fused_qkvg"]
    assert laguna.Attention.__call__ is laguna_fused._kernel_attention_call

    assert _greedy(toy_model, PROMPTS, 5) == reference_tokens
    delta = float(mx.abs(_last_logits(toy_model, PROMPTS) - reference_logits).max())
    assert delta == 0.0, f"env install was not bit-exact: {delta:.3e}"


def test_fused_qkvg_arrays_are_materialized_by_the_install(toy_model):
    """The fused weights are in the module tree but OUT of ``parameters()``.

    ``Module.valid_parameter_filter`` drops keys beginning with an underscore,
    so ``_qkvg`` is reachable as an attribute and holds real arrays while a
    later ``mx.eval(model.parameters())`` walks straight past it.  The install
    therefore evaluates the concatenation itself — which is also what lets the
    originals be freed instead of being held alive by an unevaluated graph.
    """

    _as_shipped(toy_model)
    laguna_fused.install_fused_qkvg(toy_model)

    attention = toy_model.model.layers[0].self_attn
    assert "_qkvg" in attention  # in the module tree
    tree = toy_model.parameters()["model"]["layers"][0]["self_attn"]
    assert "_qkvg" not in tree  # but not a parameter
    assert sorted(tree) == ["k_norm", "o_proj", "q_norm", "rope"]

    assert sorted(attention._qkvg.parameters()) == [
        "qkvg_biases",
        "qkvg_scales",
        "qkvg_weight",
    ]
    # The model still evaluates as a whole after the projections were dropped.
    mx.eval(toy_model.parameters())


def test_fused_qkvg_is_idempotent(toy_model):
    """Installing twice must not re-concatenate — the originals are gone."""

    _as_shipped(toy_model)
    first = laguna_fused.install_fused_qkvg(toy_model)
    assert first["layers_converted"] == len(LAYER_TYPES)
    assert laguna_fused.install_fused_qkvg(toy_model)["layers_converted"] == 0


def test_fused_qkvg_handles_a_model_with_no_attention_gate(toy_model):
    """Without gating there is no g_proj, so three projections concatenate.

    The fixture is taken for the CPU device and for restoring
    ``Attention.__call__``; the model is built here because gating is a
    construction-time choice.
    """

    mx.random.seed(5)
    model = _as_shipped(Model(_toy_args(gating=False)))
    reference_tokens = _greedy(model, PROMPTS, 5)
    reference_logits = _last_logits(model, PROMPTS)

    report = laguna_fused.install_fused_qkvg(model)
    assert report["layers_converted"] == len(LAYER_TYPES)

    attention = model.model.layers[0].self_attn
    assert attention._qkvg.gating is False
    assert attention._qkvg(mx.zeros((1, 1, model.args.hidden_size)))[3] is None

    assert _greedy(model, PROMPTS, 5) == reference_tokens
    delta = float(mx.abs(_last_logits(model, PROMPTS) - reference_logits).max())
    assert delta == 0.0, f"non-gating fusion was not bit-exact: {delta:.3e}"


def test_fused_qkvg_leaves_a_mixed_precision_layer_stock(toy_model):
    """A layer whose q/k/v/g widths differ cannot be concatenated, so it isn't.

    The shipped oQ4e table gives q, k, v and g the same width on every layer
    (layer 33 differs only in ``o_proj``, which is not part of this
    concatenation), but the installer refuses rather than assumes: a mixed
    layer keeps its four modules and runs the shipped forward, and the mixed
    model still generates exactly what it did before the install.
    """

    _as_shipped(toy_model)

    attention = toy_model.model.layers[1].self_attn
    rows = int(attention.k_proj.scales.shape[0])
    dims = int(toy_model.args.hidden_size)
    replacement = nn.QuantizedLinear.from_linear(
        nn.Linear(dims, rows, bias=False), group_size=32, bits=8
    )
    replacement.set_dtype(mx.bfloat16)
    attention.k_proj = replacement
    mx.eval(toy_model.parameters())

    reference_tokens = _greedy(toy_model, PROMPTS, 5)
    reference_logits = _last_logits(toy_model, PROMPTS)

    report = laguna_fused.install_fused_qkvg(toy_model)
    assert report["layers_converted"] == len(LAYER_TYPES) - 1
    assert report["layers_skipped"] == 1
    assert report["skip_reasons"] and report["skip_reasons"][0].startswith("layer 1:")

    # The refused layer is untouched, which is what lets it keep running stock.
    assert "q_proj" in attention
    assert getattr(attention, "_qkvg", None) is None

    assert _greedy(toy_model, PROMPTS, 5) == reference_tokens
    delta = float(mx.abs(_last_logits(toy_model, PROMPTS) - reference_logits).max())
    assert delta == 0.0, f"mixed-precision model was perturbed: {delta:.3e}"


def test_fused_qkvg_refuses_mismatched_quantization_directly(toy_model):
    """The guard is on the module, not only on the installer's bookkeeping."""

    nn.quantize(toy_model, group_size=32, bits=4)
    mx.eval(toy_model.parameters())

    attention = toy_model.model.layers[0].self_attn
    rows = int(attention.v_proj.scales.shape[0])
    dims = int(toy_model.args.hidden_size)
    attention.v_proj = nn.QuantizedLinear.from_linear(
        nn.Linear(dims, rows, bias=False), group_size=32, bits=8
    )
    with pytest.raises(ValueError, match="change the arithmetic"):
        laguna_fused.FusedQkvgProj(attention)


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
