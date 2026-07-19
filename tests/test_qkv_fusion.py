"""QKV fusion parity (issue #51 T1a).

Fusing q/k/v into one packed qkv_proj is only worth doing if it's BIT-EXACT vs
the three separate projections under the shipped q4-g64 affine quant. It is,
because the quant groups run along the INPUT axis (4096), so concatenating output
rows (q's 8192 + k's 1024 + v's 1024 = 10240) leaves every row's own scale/bias
untouched, and the batch=1 matvec is per-output-row independent. These assert
that end-to-end at real hy3 shapes: quantizing the fused weight then splitting the
result equals quantizing each weight and matmul-ing separately, to the bit.
"""

import mlx.core as mx

H, Q_DIM, KV_DIM = 4096, 8192, 1024  # hy3: 64x128 query, 8x128 kv


def _q4_matmul(x, weight):
    wq, scales, biases = mx.quantize(weight, group_size=64, bits=4)
    return mx.quantized_matmul(
        x, wq, scales, biases, transpose=True, group_size=64, bits=4
    )


def test_fused_qkv_bit_exact_under_q4() -> None:
    mx.random.seed(0)
    wq = mx.random.normal((Q_DIM, H)).astype(mx.bfloat16)
    wk = mx.random.normal((KV_DIM, H)).astype(mx.bfloat16)
    wv = mx.random.normal((KV_DIM, H)).astype(mx.bfloat16)
    x = mx.random.normal((1, 1, H)).astype(mx.bfloat16)

    separate = [_q4_matmul(x, w) for w in (wq, wk, wv)]

    fused_weight = mx.concatenate([wq, wk, wv], axis=0)  # [10240, 4096]
    fused = _q4_matmul(x, fused_weight)
    split = mx.split(fused, [Q_DIM, Q_DIM + KV_DIM], axis=-1)

    for name, got, ref in zip("qkv", split, separate):
        mx.eval(got, ref)
        assert float(mx.abs(got - ref).max()) == 0.0, f"{name} diverged under fusion"


def test_fused_split_shapes() -> None:
    fused = mx.zeros((1, 1, Q_DIM + 2 * KV_DIM))
    q, k, v = mx.split(fused, [Q_DIM, Q_DIM + KV_DIM], axis=-1)
    assert q.shape[-1] == Q_DIM
    assert k.shape[-1] == KV_DIM
    assert v.shape[-1] == KV_DIM


def test_sanitize_pack_engages() -> None:
    """The load-time pack must actually fold q/k/v into qkv_proj (engagement
    proof — a flat A/B is only creditable if the fusion truly ran)."""
    from mtplx.models.hy3_mlx import _pack_linear_projection, _projection_pack_plan

    weights = {}
    for layer in (0, 1):
        base = f"model.layers.{layer}.self_attn"
        weights[f"{base}.q_proj.weight"] = mx.zeros((Q_DIM, H), dtype=mx.bfloat16)
        weights[f"{base}.k_proj.weight"] = mx.zeros((KV_DIM, H), dtype=mx.bfloat16)
        weights[f"{base}.v_proj.weight"] = mx.zeros((KV_DIM, H), dtype=mx.bfloat16)
    for layer in (0, 1):
        base = f"model.layers.{layer}.self_attn"
        plan = _projection_pack_plan(
            weights,
            target=f"{base}.qkv_proj",
            sources=(f"{base}.q_proj", f"{base}.k_proj", f"{base}.v_proj"),
            equal_output_widths=False,
        )
        assert plan is not None
        _pack_linear_projection(weights, plan)
        assert f"{base}.qkv_proj.weight" in weights
        assert f"{base}.q_proj.weight" not in weights
        assert tuple(weights[f"{base}.qkv_proj.weight"].shape) == (Q_DIM + 2 * KV_DIM, H)
