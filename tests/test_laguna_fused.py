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
        (laguna_fused.install_kernel_router, True),
        # INEXACT on the GPU by construction (the folded gemv reassociates the
        # dot), but on the CPU device eligibility is false and every layer takes
        # the two-step stock path, so what this pins is that the FALLBACK is the
        # shipped arithmetic untouched.  The GPU divergence is measured in
        # bench/laguna/laguna_kernel_check.py::check_router_gemv instead.
        (laguna_fused.install_kernel_router_gemv, True),
        (laguna_fused.install_cached_gather_indices, True),
        # The dense-MLP gate/up concatenation, whose activation now runs through
        # `fused_glu`.  On the CPU device that kernel is ineligible and every
        # dense block evaluates the stock expression, so what this pins is the
        # FALLBACK plus the concatenation itself; the GPU kernel's own
        # bit-exactness is measured in
        # bench/laguna/laguna_kernel_check.py::check_fused_glu.
        (laguna_fused.install_fused_shared_gate_up, True),
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


# ---------------------------------------------------------------------------
# the router gemv folded into the router kernel
# ---------------------------------------------------------------------------
MOE_LAYERS = len(LAYER_TYPES) - 1  # layer 0 is dense


class _CountingGate:
    """Stands in for ``mlp.gate`` and records whether anything called it.

    Assigning a non-array object to a module attribute stores it as a plain
    Python attribute (``Module.__setattr__`` pops the dict entry), so the spy is
    invisible to ``parameters()`` and delegates everything it is asked for.
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self.inner(x)


def _spy_on_every_packed_router(model) -> list[_CountingGate]:
    spies: list[_CountingGate] = []
    for layer in model.model.layers:
        block = layer.mlp
        if getattr(block, "_router_gemv_pack", None) is None:
            continue
        spy = _CountingGate(block.gate)
        block.gate = spy
        spies.append(spy)
    return spies


def _stock_router_via_mx_ops(x, gate_weight, correction_bias, top_k, normalize, scale):
    """The two-step stock chain, written out in mx ops.

    Used as the body of a FAKE kernel so the call contract can be proved on the
    CPU: if ``_kernel_moe_call`` hands this the right arguments, the model has to
    generate exactly what it generated before the install.
    """

    logits = (x @ gate_weight.swapaxes(-1, -2)).astype(mx.float32)
    scores = mx.sigmoid(logits)
    choice = scores + correction_bias
    indices = mx.argpartition(-choice, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    if normalize:
        weights = weights / weights.sum(axis=-1, keepdims=True)
    return indices, weights * scale


def test_kernel_router_gemv_packs_the_routers_and_implies_the_kernel_router(toy_model):
    """The install has to bring the epilogue path with it, and pack every router.

    The pack holds a REFERENCE to the router's own weight — not a copy — and
    lives under an underscore, so it stays out of ``parameters()`` and cannot
    double-count 3 MB of BF16 per layer on the real checkpoint.
    """

    assert laguna.LagunaSparseMoeBlock.__call__ is not laguna_fused._kernel_moe_call

    report = laguna_fused.install_kernel_router_gemv(toy_model)

    assert report["path"] == "kernel_router_gemv"
    assert report["layers_packed"] == MOE_LAYERS
    assert report["layers_skipped"] == 0
    # The kernel-router forward owns the select/normalize/scale epilogue, so the
    # gemv install is required to put it in place.
    assert laguna.LagunaSparseMoeBlock.__call__ is laguna_fused._kernel_moe_call
    assert report["installed_kernel_router"]["path"] == "kernel_router"
    # CPU device: the kernel cannot engage, and that has to be visible.
    assert report["kernel_engaged"] is False

    for layer in toy_model.model.layers:
        block = layer.mlp
        if not isinstance(block, laguna.LagunaSparseMoeBlock):
            continue
        pack = block._router_gemv_pack
        assert pack.weight is block.gate.weight
        assert "_router_gemv_pack" not in block  # out of the module tree
        assert block._router_bias_f32 is not None  # the kernel-router box, too


def test_kernel_router_gemv_skips_a_quantized_router(toy_model):
    """A quantized router cannot be packed, so it is left on the stock path.

    The shipped oQ4e checkpoint leaves ``mlp.gate`` unquantized (it carries no
    entry in the per-path quantization map), but the installer refuses rather
    than assumes, and a skipped layer still routes correctly.
    """

    nn.quantize(toy_model, group_size=32, bits=4)
    mx.eval(toy_model.parameters())

    reference_tokens = _greedy(toy_model, PROMPTS, 5)
    reference_logits = _last_logits(toy_model, PROMPTS)

    report = laguna_fused.install_kernel_router_gemv(toy_model)
    assert report["layers_packed"] == 0
    assert report["layers_skipped"] == MOE_LAYERS
    assert report["kernel_engaged"] is False
    assert all("quantized" in reason for reason in report["skip_reasons"])

    for layer in toy_model.model.layers:
        block = layer.mlp
        if isinstance(block, laguna.LagunaSparseMoeBlock):
            assert block._router_gemv_pack is None

    assert _greedy(toy_model, PROMPTS, 5) == reference_tokens
    delta = float(mx.abs(_last_logits(toy_model, PROMPTS) - reference_logits).max())
    assert delta == 0.0, f"a skipped router was perturbed: {delta:.3e}"


def test_kernel_router_gemv_call_contract_holds_end_to_end(toy_model, monkeypatch):
    """The arguments the fused entry point is handed must BE the router.

    Eligibility is forced true and the kernel is replaced by the stock chain
    written in mx ops.  If ``_kernel_moe_call`` passes the right hidden states,
    the right weight (in ``nn.Linear``'s own ``[experts, dims]`` layout, no
    transpose), the right float32 bias and the right normalize/scale, then the
    model generates exactly what it did before — on the CPU, with no GPU.  It
    also has to STOP calling ``mlp.gate``, which is half the point of the fold.
    """

    from mtplx.kernels import laguna_decode as kernels

    reference_tokens = _greedy(toy_model, PROMPTS, 5)
    reference_logits = _last_logits(toy_model, PROMPTS)

    laguna_fused.install_kernel_router_gemv(toy_model)
    spies = _spy_on_every_packed_router(toy_model)
    assert len(spies) == MOE_LAYERS

    calls: list[dict] = []

    def fake_gemv(x, gate_weight, correction_bias, top_k, *, normalize, scale):
        calls.append(
            {
                "rows": int(x.shape[0]),
                "dims": int(x.shape[1]),
                "weight_shape": tuple(int(dim) for dim in gate_weight.shape),
                "bias_dtype": correction_bias.dtype,
                "top_k": top_k,
                "normalize": normalize,
                "scale": scale,
            }
        )
        return _stock_router_via_mx_ops(
            x, gate_weight, correction_bias, top_k, normalize, scale
        )

    monkeypatch.setattr(kernels, "is_router_gemv_eligible", lambda *a, **k: True)
    monkeypatch.setattr(kernels, "fused_router_gemv_topk", fake_gemv)

    assert _greedy(toy_model, PROMPTS, 5) == reference_tokens
    delta = float(mx.abs(_last_logits(toy_model, PROMPTS) - reference_logits).max())
    assert delta == 0.0, f"the fused call contract changed the model: {delta:.3e}"

    assert calls, "the fused router-gemv entry point was never reached"
    assert all(spy.calls == 0 for spy in spies), "the fused path still called mlp.gate"

    args = toy_model.args
    for call in calls:
        assert call["dims"] == int(args.hidden_size)
        assert call["weight_shape"] == (
            int(args.num_experts),
            int(args.hidden_size),
        )
        assert call["bias_dtype"] == mx.float32
        assert call["top_k"] == int(args.num_experts_per_tok)
        assert call["normalize"] is bool(args.norm_topk_prob)
        assert call["scale"] == pytest.approx(
            float(args.moe_routed_scaling_factor)
        )


def test_kernel_router_gemv_leaves_mlp_gate_running_when_ineligible(toy_model):
    """The fallback is the whole safety net, so pin that it really runs.

    Same install, real eligibility (false on the CPU device): every MoE layer
    has to go back through ``mlp.gate``.
    """

    laguna_fused.install_kernel_router_gemv(toy_model)
    spies = _spy_on_every_packed_router(toy_model)

    _greedy(toy_model, PROMPTS, 2)

    assert spies and all(spy.calls > 0 for spy in spies)


def test_kernel_router_gemv_is_idempotent(toy_model):
    """Installing twice must repack, not double-install or double-count."""

    first = laguna_fused.install_kernel_router_gemv(toy_model)
    second = laguna_fused.install_kernel_router_gemv(toy_model)

    assert first["layers_packed"] == second["layers_packed"] == MOE_LAYERS
    assert second["layers_skipped"] == 0
    # The kernel-router forward was already in place the second time around.
    assert "installed_kernel_router" in first
    assert "installed_kernel_router" not in second
    assert laguna.LagunaSparseMoeBlock.__call__ is laguna_fused._kernel_moe_call


def test_install_from_env_wires_the_router_gemv(toy_model):
    """The env entry point has to reach the installer and stay token-equal."""

    import os

    reference_tokens = _greedy(toy_model, PROMPTS, 5)

    os.environ[laguna_fused.ENV_KERNEL_ROUTER_GEMV] = "1"
    try:
        report = laguna_fused.install_from_env(toy_model)
    finally:
        del os.environ[laguna_fused.ENV_KERNEL_ROUTER_GEMV]

    assert [entry["path"] for entry in report] == ["kernel_router_gemv"]
    assert laguna.LagunaSparseMoeBlock.__call__ is laguna_fused._kernel_moe_call
    assert _greedy(toy_model, PROMPTS, 5) == reference_tokens


def test_router_gemv_eligibility_refuses_what_the_kernel_cannot_do():
    """The eligibility gate is the correctness branch; pin its shape rules.

    Written so it says the same thing on a CPU-only box and on this one: every
    malformed shape has to be refused whether or not the well-formed one would
    be accepted, and on a Metal box the well-formed one has to be accepted too,
    or the whole path is silently dead.
    """

    from mtplx.kernels import laguna_decode as kernels

    assert kernels._router_selection_shape_ok(256, 10) is True
    assert kernels._router_selection_shape_ok(256, 0) is False
    assert kernels._router_selection_shape_ok(256, 33) is False  # past the scratch
    assert kernels._router_selection_shape_ok(16, 4) is False  # below the floor
    assert kernels._router_selection_shape_ok(48, 4) is False  # not a 32-multiple
    assert kernels.MAX_ROUTER_GEMV_DIMS % 4 == 0

    weight = mx.zeros((256, 3072), dtype=mx.bfloat16)
    bias = mx.zeros((256,), dtype=mx.float32)
    x = mx.zeros((1, 3072), dtype=mx.bfloat16)

    if kernels._on_metal_device():
        assert kernels.is_router_gemv_eligible(x, weight, bias, 10) is True

    refused = {
        "3-D hidden states": (x[None], weight, bias, 10),
        "float32 hidden states": (x.astype(mx.float32), weight, bias, 10),
        "float32 weight": (x, weight.astype(mx.float32), bias, 10),
        "bfloat16 bias": (x, weight, bias.astype(mx.bfloat16), 10),
        "width mismatch": (mx.zeros((1, 3068), dtype=mx.bfloat16), weight, bias, 10),
        "bias/expert mismatch": (
            x,
            weight,
            mx.zeros((128,), dtype=mx.float32),
            10,
        ),
        "width not a multiple of four": (
            mx.zeros((1, 3070), dtype=mx.bfloat16),
            mx.zeros((256, 3070), dtype=mx.bfloat16),
            bias,
            10,
        ),
        "width past the threadgroup budget": (
            mx.zeros((1, kernels.MAX_ROUTER_GEMV_DIMS + 4), dtype=mx.bfloat16),
            mx.zeros((256, kernels.MAX_ROUTER_GEMV_DIMS + 4), dtype=mx.bfloat16),
            bias,
            10,
        ),
        "past the row gate": (
            mx.zeros((64, 3072), dtype=mx.bfloat16),
            weight,
            bias,
            10,
        ),
        "top_k past the scratch": (x, weight, bias, 33),
    }
    for label, arguments in refused.items():
        assert kernels.is_router_gemv_eligible(*arguments) is False, label


def test_router_gemv_logits_k_partition_covers_every_admissible_width():
    """The K split has to cover the row exactly once, at every width.

    A threadgroup's eight simdgroups share one expert's row, each taking a
    ceil-sized slice of the blocks of four.  Ceil is what makes a width that is
    a multiple of four but not of 32 safe — the last slice comes up short and a
    surplus one is empty — but ceil is also how a partition ends up covering an
    element twice or not at all, so walk the arithmetic the kernel uses and
    check the slices tile the row.  Off the GPU this is the only reachable proof
    of it.
    """

    from mtplx.kernels import laguna_decode as kernels

    simd = kernels._ROUTER_GEMV_SIMD
    split = kernels._ROUTER_GEMV_SPLIT
    assert kernels._ROUTER_GEMV_THREADS == simd * split

    widths = [4, 8, 28, 32, 3068, 3072, 3076, kernels.MAX_ROUTER_GEMV_DIMS]
    for dims in widths:
        assert dims % 4 == 0 and dims <= kernels.MAX_ROUTER_GEMV_DIMS
        blocks = dims // 4
        per_part = -(-blocks // split)  # the kernel's (BLOCKS + SPLIT - 1) / SPLIT
        covered: list[int] = []
        for part in range(split):
            begin, end = part * per_part, min((part + 1) * per_part, blocks)
            for lane in range(simd):
                covered.extend(range(begin + lane, end, simd))
        assert sorted(covered) == list(range(blocks)), dims


def test_router_gemv_fallback_matches_the_two_step_stock_chain():
    """The CPU fallback inside fused_router_gemv_topk IS the stock arithmetic."""

    from mtplx.kernels.laguna_decode import fused_router_gemv_topk

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(19)
        x = mx.random.normal((3, 64)).astype(mx.float32)
        weight = mx.random.normal((16, 64)).astype(mx.float32)
        bias = (mx.random.normal((16,)) * 0.05).astype(mx.float32)
        mx.eval(x, weight, bias)

        want_idx, want_w = _stock_router_via_mx_ops(x, weight, bias, 4, True, 2.5)
        got_idx, got_w = fused_router_gemv_topk(
            x, weight, bias, 4, normalize=True, scale=2.5
        )
        mx.eval(want_idx, want_w, got_idx, got_w)
        assert got_idx.tolist() == want_idx.tolist()
        assert float(mx.abs(want_w - got_w).max()) == 0.0
    finally:
        mx.set_default_device(previous)


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


def test_fused_glu_fallback_is_the_stock_activation():
    """Off the GPU, ``fused_glu`` has to BE ``nn.silu(gate) * up``.

    Pinned on the CPU device on purpose: the kernel's own arithmetic is checked
    where it runs (bench/laguna/laguna_kernel_check.py::check_fused_glu), and
    what is reachable here is the other half of the contract — the half a
    refactor is most likely to break.  The fallback must stay the shipped
    expression rather than drift into something merely equivalent-looking, and
    it must produce the same shape the kernel does at every rank the dense MLPs
    hand it.
    """

    from mtplx.kernels.laguna_decode import fused_glu

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        for shape, hidden in (((2 * 5,), 5), ((1, 2 * 7), 7), ((3, 1, 2 * 4), 4)):
            mx.random.seed(sum(shape) + hidden)
            fused = mx.random.normal(shape).astype(mx.float32)
            mx.eval(fused)

            want = nn.silu(fused[..., :hidden]) * fused[..., hidden:]
            got = fused_glu(fused, hidden)
            mx.eval(want, got)

            assert tuple(got.shape) == tuple(shape[:-1]) + (hidden,)
            assert float(mx.abs(want - got).max()) == 0.0, shape
    finally:
        mx.set_default_device(previous)


def test_fused_glu_takes_the_gate_from_the_first_half():
    """Which half is the gate is a silent, unrecoverable defect if it is wrong.

    ``silu(a) * b`` and ``silu(b) * a`` are both plausible-looking outputs of the
    right shape, and the concatenation order (gate rows first, then up) is a
    convention set in ``FusedGateUpMLP.__init__`` that the kernel's ``base`` vs
    ``base + HIDDEN`` addressing has to agree with.  So assert the halves are
    not interchangeable AND that it is the first one that goes through the
    sigmoid.

    Left on the DEFAULT device rather than pinned to the CPU: on a Metal box
    that makes this the one suite test that actually dispatches the kernel, and
    the halves contract is what a GPU dispatch is best spent on.
    """

    from mtplx.kernels.laguna_decode import fused_glu

    hidden = 4
    gate = mx.array([-2.0, -0.5, 0.5, 2.0])
    up = mx.array([1.0, 3.0, -4.0, 7.0])
    fused = mx.concatenate([gate, up])

    got = fused_glu(fused, hidden)
    swapped = fused_glu(mx.concatenate([up, gate]), hidden)
    mx.eval(got, swapped)

    assert float(mx.abs(got - nn.silu(gate) * up).max()) == 0.0
    assert float(mx.abs(got - swapped).max()) > 0.0


def test_fused_glu_addressing_tiles_the_concatenated_row():
    """Walk the kernel's flat index arithmetic and check it hits each half once.

    The kernel is one thread per OUTPUT element over a [rows, 2H] input, so it
    recovers ``(row, col)`` by division and reads ``base`` and ``base + HIDDEN``.
    That arithmetic is unreachable off the GPU, but it is arithmetic, and an
    off-by-a-row would read a neighbouring token's activation rather than crash.
    So reproduce it here and pin it against the flat positions the stock slices
    actually occupy.
    """

    for rows, hidden in ((1, 1024), (1, 12288), (4, 32), (7, 5)):
        flat = mx.arange(rows * 2 * hidden).reshape(rows, 2 * hidden)
        want_gate = flat[:, :hidden].reshape(-1).tolist()
        want_up = flat[:, hidden:].reshape(-1).tolist()

        got_gate, got_up = [], []
        for index in range(rows * hidden):
            row, col = index // hidden, index % hidden
            base = row * (2 * hidden) + col
            got_gate.append(base)
            got_up.append(base + hidden)

        assert got_gate == want_gate, (rows, hidden)
        assert got_up == want_up, (rows, hidden)


def test_glu_eligibility_refuses_what_the_kernel_cannot_do():
    """The eligibility gate is the correctness branch; pin its shape rules.

    Says the same thing on a CPU-only box and on this one: every malformed
    shape has to be refused either way, and on a Metal box the well-formed one
    has to be accepted or the path is silently dead.
    """

    from mtplx.kernels import laguna_decode as kernels

    fused = mx.zeros((1, 2048), dtype=mx.bfloat16)

    if kernels._on_metal_device():
        assert kernels.is_glu_eligible(fused, 1024) is True
        # No row gate: this kernel is strictly fewer bytes and strictly fewer
        # dispatches than the stock chain at every batch, so there is nothing a
        # threshold would protect.
        batched = mx.zeros((64, 2048), dtype=mx.bfloat16)
        assert kernels.is_glu_eligible(batched, 1024) is True

    refused = {
        "last axis is not 2H": (mx.zeros((1, 2047), dtype=mx.bfloat16), 1024),
        "hidden claims the whole row": (fused, 2048),
        "hidden is zero": (fused, 0),
        "hidden is negative": (fused, -1024),
        "integer projection": (mx.zeros((1, 2048), dtype=mx.int32), 1024),
    }
    for label, arguments in refused.items():
        assert kernels.is_glu_eligible(*arguments) is False, label


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
