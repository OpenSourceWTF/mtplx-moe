"""6-bit weights for the M6 K-split verify matmul (bn-templated variant).

The motivating consumer is the 35B-A3B Balance build: its dense projections are
quantized 6-bit affine, so solo-MTP verify on those shapes fell all the way to
the stock path while the kernel family stayed 4-bit only.

MLX packs 6-bit as a continuous little-endian bitstream: 3 uint32 words hold
exactly 16 values, and the two values that straddle a word boundary are field 5
((w0 >> 30) | (w1 << 2)) and field 10 ((w1 >> 28) | (w2 << 4)). Every group size
the lane accepts (32/64/128) is a whole number of those 3-word blocks, so the
scale/bias index is constant across a block -- all compile-time, no per-value
group math. The row stride is 3 * K/16 words instead of K/8.

Exactness contract. This family is argmax-validated, not bit-exact: the kernel
reassociates the fp32 reduction (K-split across simdgroups, then simd_sum) so
it does not reproduce stock's accumulation order at ANY width -- the 4-bit
kernel already differs from stock by up to ~0.25 absolute on these shapes. The
bar every case here holds to is therefore bounded relative deviation from stock
plus per-row argmax parity. The dequant is held to a stricter bar separately --
probed one value at a time it is bit-exact against mx.dequantize, which is what
pins the layout.

Both callers are covered: nax_qmm_m6 (k_parts=2, BN=4) and nax_qmm_m6_kp1_bn2
(k_parts=1, BN=2). The 6-bit unit indexes columns with the BN loop variable, so
it has to hold at both column tilings, not just the default one.
"""

import importlib.metadata

import mlx.core as mx
import pytest

from mtplx.nax_verify import (
    _M6_INNER_BY_BITS,
    _M6_PACK_RANGE_BY_BITS,
    _M6_UNIT_SETUP_BY_BITS,
    _m6_ksplit_source,
    _packed_bits_from_shape,
    m6_ksplit_eligible,
    nax_qmm_m6,
    nax_qmm_m6_kp1_bn2,
)

_MLX_VERSION = importlib.metadata.version("mlx")
pytestmark = pytest.mark.skipif(
    not _MLX_VERSION.startswith("0.32."),
    reason=f"kernels reproduce MLX 0.32 stock arithmetic; found {_MLX_VERSION}",
)

# Widths this lane builds a kernel for. 8-bit is deliberately absent: the m6
# ksplit has no 8-bit unit, and 8-bit shapes must keep falling to stock.
SUPPORTED_BITS = (4, 6)
UNSUPPORTED_BITS = (2, 3, 5, 8)

# Real verify shapes: A3B Balance dense projections (hidden 2048, 16 q-heads x
# head_dim 256, 2 kv-heads) plus a 27B layer shape for cross-family coverage.
SHAPES = [
    (2048, 4096),   # A3B q_proj (16*256)
    (2048, 512),    # A3B kv_proj (2*256)
    (4096, 2048),   # A3B o_proj
    (5120, 6144),   # 27B projection
]

# The kp1/bn2 caller is a fixed six-row lane; two shapes are enough to pin the
# BN=2 column tiling at both widths.
KP1_BN2_SHAPES = [(2048, 4096), (2048, 512)]

# Deviation from stock, relative to the largest stock magnitude in the tile.
REL_TOL = 1e-2


def _case(K, N, bits, group_size=64, seed=0):
    mx.random.seed(seed)
    w = mx.random.normal((N, K)).astype(mx.bfloat16)
    w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    x = mx.random.normal((6, K)).astype(mx.bfloat16)
    mx.eval(w_q, scales, biases, x)
    return x, w_q, scales, biases


def _stock(x, w_q, scales, biases, bits, group_size=64):
    return mx.quantized_matmul(x, w_q, scales=scales, biases=biases,
                               transpose=True, group_size=group_size, bits=bits)


def _assert_tracks_stock(got, want, label):
    """The family bar: bounded relative deviation + per-row argmax parity."""
    mx.eval(got, want)
    g = got.astype(mx.float32)
    w = want.astype(mx.float32)
    dmax = float(mx.max(mx.abs(g - w)))
    scale = float(mx.max(mx.abs(w)))
    rel = dmax / scale if scale else dmax
    assert rel <= REL_TOL, f"{label}: dmax={dmax} scale={scale} rel={rel}"
    got_am = mx.argmax(g, axis=-1).tolist()
    want_am = mx.argmax(w, axis=-1).tolist()
    assert got_am == want_am, f"{label}: argmax {got_am} != stock {want_am}"
    return dmax, rel


def _one_hot_probe(K, group_size):
    """Rows that each select a single k, hitting both straddles and a group hop."""
    idx = mx.array([0, 5, 10, 15, 16 + 5, group_size + 10])
    x = mx.zeros((6, K), dtype=mx.bfloat16)
    x[mx.arange(6), idx] = mx.array(1.0, dtype=mx.bfloat16)
    return mx.contiguous(x), idx


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
def test_eligible_for_supported_widths(bits):
    assert m6_ksplit_eligible(6, 2048, 4096, bits, 64, mx.bfloat16) is True


@pytest.mark.parametrize("bits", UNSUPPORTED_BITS)
def test_ineligible_for_unsupported_widths(bits):
    assert m6_ksplit_eligible(6, 2048, 4096, bits, 64, mx.bfloat16) is False


@pytest.mark.parametrize("bits", UNSUPPORTED_BITS)
def test_codegen_refuses_unsupported_widths(bits):
    """Ineligibility is not the only guard -- codegen refuses too, so an 8-bit
    tensor that somehow reached either caller raises instead of decoding
    garbage."""
    with pytest.raises(ValueError):
        _m6_ksplit_source(64, bits=bits)
    with pytest.raises(ValueError):
        _m6_ksplit_source(64, k_parts=1, bn=2, bits=bits)


def test_packed_bits_inferred_from_column_count():
    K = 2048
    for bits, cols in ((4, K // 8), (6, 3 * K // 16), (8, K // 4)):
        assert _packed_bits_from_shape(K, cols) == bits
    for w_q_cols in (0, -1, 7):
        with pytest.raises(ValueError):
            _packed_bits_from_shape(K, w_q_cols)


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
@pytest.mark.parametrize(("K", "N"), SHAPES)
def test_tracks_stock(bits, K, N, record_property):
    x, w_q, scales, biases = _case(K, N, bits)
    got = nax_qmm_m6(x, w_q, scales, biases, group_size=64)
    want = _stock(x, w_q, scales, biases, bits)
    dmax, rel = _assert_tracks_stock(got, want, f"bits={bits} K={K} N={N}")
    record_property("dmax", dmax)
    record_property("rel", rel)


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
@pytest.mark.parametrize(("K", "N"), KP1_BN2_SHAPES)
def test_kp1_bn2_tracks_stock(bits, K, N, record_property):
    """Second caller: one K partition, two columns per threadgroup."""
    x, w_q, scales, biases = _case(K, N, bits)
    got = nax_qmm_m6_kp1_bn2(x, w_q, scales, biases, group_size=64)
    want = _stock(x, w_q, scales, biases, bits)
    assert got.shape == (6, N)
    dmax, rel = _assert_tracks_stock(
        got, want, f"kp1bn2 bits={bits} K={K} N={N}"
    )
    record_property("dmax", dmax)
    record_property("rel", rel)


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize(("K", "N"), [(2048, 4096), (2048, 512)])
def test_six_bit_across_group_sizes(group_size, K, N):
    """gs is a compile-time constant in the kernel and the 3-word block must
    land inside one scale group at every gs the lane declares eligible."""
    assert m6_ksplit_eligible(6, K, N, 6, group_size, mx.bfloat16) is True
    x, w_q, scales, biases = _case(K, N, 6, group_size=group_size)
    got = nax_qmm_m6(x, w_q, scales, biases, group_size=group_size)
    want = _stock(x, w_q, scales, biases, 6, group_size=group_size)
    _assert_tracks_stock(got, want, f"bits=6 gs={group_size} K={K} N={N}")


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_six_bit_bitstream_decode(group_size):
    """Read the decoded weights out directly, free of accumulation order.

    One-hot rows turn the matmul into a lookup: y[r, n] is the kernel's own
    dequant of weight (n, k_r), with every other term an exact zero. The decode
    itself IS bit-exact against mx.dequantize -- only the K-split reduction
    reassociates -- so this one asserts equality, not tolerance.

    The probed k values are the first field of the first 3-word block, both of
    its straddles (5 and 10), its last field (15), the straddle in the NEXT
    block (21), and one a whole scale group away -- so a wrong shift, a wrong
    W_row stride or a wrong group index shows up as a raw value mismatch.
    Reading the neighbouring fields instead lands ~1.4 relative away, two
    orders of magnitude past even the loose tolerance bar.
    """
    K, N = 2048, 4096
    _, w_q, scales, biases = _case(K, N, 6, group_size=group_size)
    x, idx = _one_hot_probe(K, group_size)
    got = nax_qmm_m6(x, w_q, scales, biases, group_size=group_size)
    deq = mx.dequantize(w_q, scales, biases, group_size=group_size, bits=6)
    want = mx.take(deq, idx, axis=1).T  # [6, N]
    mx.eval(got, want)
    assert int(mx.sum(x != 0)) == 6
    dmax = float(mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32))))
    assert dmax == 0.0, f"6-bit decode gs={group_size} dmax={dmax}"


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_six_bit_bitstream_decode_kp1_bn2(group_size):
    """Same one-hot dequant probe through the BN=2 / K_PARTS=1 caller, whose
    column indexing walks the packed rows in a different tiling."""
    K, N = 2048, 4096
    _, w_q, scales, biases = _case(K, N, 6, group_size=group_size)
    x, idx = _one_hot_probe(K, group_size)
    got = nax_qmm_m6_kp1_bn2(x, w_q, scales, biases, group_size=group_size)
    deq = mx.dequantize(w_q, scales, biases, group_size=group_size, bits=6)
    want = mx.take(deq, idx, axis=1).T  # [6, N]
    mx.eval(got, want)
    dmax = float(mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32))))
    assert dmax == 0.0, f"6-bit kp1bn2 decode gs={group_size} dmax={dmax}"


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
def test_m5_padding_path(bits):
    x, w_q, scales, biases = _case(2048, 4096, bits)
    got = nax_qmm_m6(x[:5], w_q, scales, biases, group_size=64)
    want = _stock(x[:5], w_q, scales, biases, bits)
    assert got.shape == (5, 4096)
    _assert_tracks_stock(got, want, f"M=5 bits={bits}")


# Byte-for-byte rendering of the 4-bit-only kernel as it stood before 6-bit was
# added (gs=64, k_parts=2, bn=4). Splitting the source into per-width blocks
# must not perturb the 4-bit lane by so much as a space: it is the shipped
# verify kernel. The two whitespace-only lines are how the renderer's
# pre-indented interpolation lands; they are written as substitutions because
# trailing whitespace does not survive a source file.
_GOLDEN_4BIT_GS64_KP2_BN4 = """
        using namespace metal;
        constexpr int M = 6;
        constexpr int BN = 4;
        constexpr int K_PARTS = 2;
        constexpr int GS = 64;

        uint part = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_8 = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;
@PAD8@
        int packs_per_part = K_by_8 / K_PARTS;
        int pack_start = int(part) * packs_per_part;
        int pack_end = (int(part) == K_PARTS - 1)
            ? K_by_8
            : pack_start + packs_per_part;
@PAD4@

        float acc[BN * M];
        _Pragma("unroll")
        for (int i = 0; i < BN * M; ++i) {
            acc[i] = 0.0f;
        }

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        for (int pack = pack_start + int(lane); pack < pack_end; pack += 32) {
            int k_base = pack * 8;
            Vec8 v0 = xv[(0 * K + k_base) / 8];
            Vec8 v1 = xv[(1 * K + k_base) / 8];
            Vec8 v2 = xv[(2 * K + k_base) / 8];
            Vec8 v3 = xv[(3 * K + k_base) / 8];
            Vec8 v4 = xv[(4 * K + k_base) / 8];
            Vec8 v5 = xv[(5 * K + k_base) / 8];
            _Pragma("unroll")
            for (int j = 0; j < BN; ++j) {
                uint32_t packed = w_q[(n0 + j) * K_by_8 + pack];
                float s = float(scales[(n0 + j) * K_by_gs + (k_base / GS)]);
                float b = float(biases[(n0 + j) * K_by_gs + (k_base / GS)]);
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[j * M + 0] += float(v0[ki]) * wv;
                    acc[j * M + 1] += float(v1[ki]) * wv;
                    acc[j * M + 2] += float(v2[ki]) * wv;
                    acc[j * M + 3] += float(v3[ki]) * wv;
                    acc[j * M + 4] += float(v4[ki]) * wv;
                    acc[j * M + 5] += float(v5[ki]) * wv;
                }
            }
        }

        _Pragma("unroll")
        for (int i = 0; i < BN * M; ++i) {
            acc[i] = simd_sum(acc[i]);
        }

        threadgroup float partial[K_PARTS * BN * M];
        if (lane == 0) {
            _Pragma("unroll")
            for (int i = 0; i < BN * M; ++i) {
                partial[int(part) * BN * M + i] = acc[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (part == 0 && lane < BN * M) {
            float total = 0.0f;
            _Pragma("unroll")
            for (int p = 0; p < K_PARTS; ++p) {
                total += partial[p * BN * M + int(lane)];
            }
            int j = int(lane) / M;
            int row = int(lane) - j * M;
            int n_global = n0 + j;
            if (n_global < N) {
                y[row * N + n_global] = T(total);
            }
        }
@PAD4@""".replace("@PAD8@", " " * 8).replace(
    "@PAD4@", " " * 4
)


def test_four_bit_codegen_is_unchanged():
    assert _m6_ksplit_source(64, k_parts=2, bn=4, bits=4) == _GOLDEN_4BIT_GS64_KP2_BN4


@pytest.mark.parametrize("group_size", [32, 128])
def test_four_bit_codegen_unchanged_at_other_group_sizes(group_size):
    """gs only ever appeared as the GS constant; nothing else may move."""
    expected = _GOLDEN_4BIT_GS64_KP2_BN4.replace(
        "constexpr int GS = 64;", f"constexpr int GS = {group_size};"
    )
    assert _m6_ksplit_source(group_size, k_parts=2, bn=4, bits=4) == expected


def test_four_bit_codegen_unchanged_for_kp1_bn2():
    """The second caller's rendering is the same text with two constants
    swapped -- BN and K_PARTS never leak into the body."""
    expected = _GOLDEN_4BIT_GS64_KP2_BN4.replace(
        "constexpr int BN = 4;", "constexpr int BN = 2;"
    ).replace("constexpr int K_PARTS = 2;", "constexpr int K_PARTS = 1;")
    assert _m6_ksplit_source(64, k_parts=1, bn=2, bits=4) == expected


def test_six_bit_templating_is_bn_and_k_parts_clean():
    """The 6-bit unit must template the same way: only the two constants move,
    so the BN=2 caller gets the identical decode."""
    base = _m6_ksplit_source(64, k_parts=2, bn=4, bits=6)
    expected = base.replace(
        "constexpr int BN = 4;", "constexpr int BN = 2;"
    ).replace("constexpr int K_PARTS = 2;", "constexpr int K_PARTS = 1;")
    assert _m6_ksplit_source(64, k_parts=1, bn=2, bits=6) == expected


def test_four_bit_unit_markers():
    """The 4-bit unit still walks 8 values per uint32 on nibble boundaries."""
    assert "K_by_8 = K / 8" in _M6_UNIT_SETUP_BY_BITS[4]
    assert "K_by_8 / K_PARTS" in _M6_PACK_RANGE_BY_BITS[4]
    assert "float((packed >> (ki * 4)) & 0xFu)" in _M6_INNER_BY_BITS[4]
    assert "K_by_16" not in _M6_UNIT_SETUP_BY_BITS[4]


def test_six_bit_unit_markers():
    """The 6-bit unit walks 16 values per 3-word block, straddles included."""
    assert "W_row = 3 * K_by_16" in _M6_UNIT_SETUP_BY_BITS[6]
    assert "K_by_16 / K_PARTS" in _M6_PACK_RANGE_BY_BITS[6]
    assert "((w0 >> 30) | (w1 << 2)) & 0x3Fu" in _M6_INNER_BY_BITS[6]
    assert "((w1 >> 28) | (w2 << 4)) & 0x3Fu" in _M6_INNER_BY_BITS[6]
    assert _m6_ksplit_source(64, bits=6) != _GOLDEN_4BIT_GS64_KP2_BN4
