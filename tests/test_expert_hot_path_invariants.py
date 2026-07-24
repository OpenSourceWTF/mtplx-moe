from mtplx import generation
from mtplx.models import hy3_mlx


def test_sustained_prefill_is_bound_before_generation(monkeypatch):
    policy = generation.bind_generation_feature_policy(
        {"MTPLX_SUSTAINED_PREFILL": "1"}
    )
    monkeypatch.setattr(generation, "_GENERATION_FEATURE_POLICY", policy)
    monkeypatch.setattr(
        generation.os.environ,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hot-path environment read")
        ),
    )
    assert generation._sustained_prefill_enabled() is True


def test_hy3_submit_cadence_is_bound_before_forward(monkeypatch):
    policy = hy3_mlx.bind_hy3_execution_policy(
        {"MTPLX_HY3_SUBMIT_CADENCE": "8"}
    )
    monkeypatch.setattr(hy3_mlx, "_HY3_EXECUTION_POLICY", policy)
    monkeypatch.setattr(
        hy3_mlx.os.environ,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hot-path environment read")
        ),
    )
    assert hy3_mlx._decode_submit_cadence() == 8
