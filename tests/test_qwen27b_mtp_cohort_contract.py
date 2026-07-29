from __future__ import annotations

import inspect
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import pytest

import mtplx.qwen27b_mtp_cohort as cohort
from mtplx.qwen27b_mtp_cohort import (
    EXPECTED_MODEL_ID,
    EXPECTED_QLINEAR_GEOMETRY_HISTOGRAM,
    FixedQLinearRoute,
    Qwen27BK2DualLane,
    TargetForwardResult,
    install_qwen27b_k2_dual_lane,
)


ROOT = Path(__file__).resolve().parents[1]
BF16 = object()
FP16 = object()
CONTROL_GEOMETRY_HISTOGRAM = (
    (5120, 48, 96),
    (5120, 1024, 32),
    (5120, 6144, 48),
    (5120, 10240, 48),
    (5120, 12288, 16),
    (5120, 17408, 128),
    (5120, 248320, 1),
    (6144, 5120, 64),
    (17408, 5120, 64),
)


class _Array:
    def __init__(self, shape: tuple[int, ...], dtype: object = BF16, value: str = "x"):
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)
        self.value = value

    def reshape(self, *shape: int) -> "_Array":
        return _Array(tuple(shape), self.dtype, f"reshape({self.value})")

    def __add__(self, other: object) -> "_Array":
        return _Array(self.shape, self.dtype, f"{self.value}+{other!r}")


class _Tensor:
    def __init__(self, shape: tuple[int, ...], dtype: str = "uint32"):
        self.shape = shape
        self.dtype = dtype


class _QLinear:
    def __init__(
        self,
        k: int,
        n: int,
        *,
        bits: int = 4,
        group_size: int = 64,
        mode: str = "affine",
        output_bias: object | None = None,
    ):
        self.bits = bits
        self.group_size = group_size
        self.mode = mode
        self._values = {
            "weight": _Tensor((n, k * bits // 32)),
            "scales": _Tensor((n, k // group_size), "bfloat16"),
            "biases": _Tensor((n, k // group_size), "bfloat16"),
        }
        if output_bias is not None:
            self._values["bias"] = output_bias

    @property
    def weight(self) -> _Tensor:
        return self._values["weight"]

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __contains__(self, key: str) -> bool:
        return key in self._values


class _TextModel:
    def __init__(
        self,
        modules: list[tuple[str, _QLinear]],
        layers: list[object] | None = None,
    ):
        self.model = SimpleNamespace(
            layers=(
                [object() for _ in range(64)]
                if layers is None
                else layers
            )
        )
        self._modules = modules

    def named_modules(self):
        return iter(self._modules)


class _Runtime:
    def __init__(
        self,
        model_path: Path,
        modules: list[tuple[str, _QLinear]],
        *,
        layers: list[object] | None = None,
    ):
        self.model_path = model_path
        self.model = SimpleNamespace(
            language_model=_TextModel(modules, layers)
        )
        self.model._runtime = self
        self.mtp_enabled = True
        self.contract = SimpleNamespace(hidden_variant="post_norm")
        self.calls: list[dict[str, object]] = []
        self.qwen27b_k2_dual_lane = None

    def forward_ar_capture(self, input_ids, **kwargs):
        self.calls.append({"input_ids": input_ids, **kwargs})
        return "logits", "hidden", {"capture": True}


class _Scope:
    def __init__(self, seen: list[tuple[int, object]], execution: object):
        self.seen = seen
        self.execution = execution

    def __enter__(self):
        self.seen.append((self.execution.width, self.execution.routes))

    def __exit__(self, *_args):
        return False


class _PatchLease:
    def __init__(self, stock_call, acquire_calls: list[None] | None = None):
        self.stock_call = stock_call
        self.initially_dynamic = False
        self.active = False
        self.acquire_calls = acquire_calls

    def acquire(self):
        if self.acquire_calls is not None:
            self.acquire_calls.append(None)
        self.active = True

    def release(self):
        self.active = False


def _model_path(tmp_path: Path, *, dtype: str = "bfloat16") -> Path:
    path = tmp_path / "Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed"
    path.mkdir()
    config_text = (
        """
{
  "model_type": "qwen3_5",
  "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
  "text_config": {
    "dtype": "%s",
    "hidden_size": 5120,
    "model_type": "qwen3_5_text",
    "mtp_num_hidden_layers": 1,
    "num_hidden_layers": 64
  }
}
"""
        % dtype
    )
    (path / "config.json").write_text(
        config_text.strip() + "\n",
        encoding="utf-8",
    )
    (path / "MTPLX_PUBLISH_MANIFEST.json").write_text(
        f'{{"repo_id": "{EXPECTED_MODEL_ID}"}}\n',
        encoding="utf-8",
    )
    return path


def _selfcheck_comparison(
    path: str,
    *,
    tolerance: float,
) -> dict[str, object]:
    return {
        "path": path,
        "candidate_shape": [1],
        "reference_shape": [1],
        "candidate_dtype": "bfloat16",
        "reference_dtype": "bfloat16",
        "dmax": 0.0,
        "tolerance": tolerance,
    }


def _selfcheck_cache_layers(
    path: str,
    *,
    tolerance: float,
) -> list[dict[str, object]]:
    attention = set(range(3, 64, 4))
    return [
        {
            "row": row,
            "layer_index": layer_index,
            "kind": "attention" if layer_index in attention else "recurrent",
            "candidate_offset": 1100,
            "reference_offset": 1100,
            "state_comparisons": [
                _selfcheck_comparison(
                    f"{path}.row{row}.layer{layer_index}.state.0",
                    tolerance=tolerance,
                )
            ],
        }
        for row in range(2)
        for layer_index in range(64)
    ]


def _actual_selfcheck_report(
    qlinear_report: MappingProxyType | dict[str, object],
) -> dict[str, object]:
    attention = set(range(3, 64, 4))
    recurrent = [
        layer_index
        for layer_index in range(64)
        if layer_index not in attention
    ]
    return {
        "schema": "qwen27b-mtp-cohort-selfcheck-v1",
        "status": "pass",
        "prefill_chunk_tokens": 1024,
        "prefill_prompt_tokens": {"0": 1100, "1": 1101},
        "prefill_spans": {
            "0": [[0, 1024], [1024, 1100]],
            "1": [[0, 1024], [1024, 1101]],
        },
        "target_cache_reference": "two_owned_clones_per_prefilled_row",
        "qlinear": dict(qlinear_report),
        "target_cycle": {
            "input_shape": [2, 3],
            "acceptance_source": "GenerationOutput.stats.accepted_drafts",
            "output_comparisons": [
                _selfcheck_comparison("logits", tolerance=1.0),
                _selfcheck_comparison("hidden", tolerance=1.0),
                *[
                    _selfcheck_comparison(
                        f"captures.row{row}.{layer_index}.gate",
                        tolerance=1.0,
                    )
                    for row in range(2)
                    for layer_index in recurrent
                ],
            ],
            "starting_cache_layers": _selfcheck_cache_layers(
                "starting_cache",
                tolerance=0.0,
            ),
            "starting_cache_aliasing": [
                {
                    "row": row,
                    "candidate_reference_aliasing": False,
                    "candidate_sibling_aliasing": False,
                    "reference_sibling_aliasing": False,
                }
                for row in range(2)
            ],
            "cache_layers": _selfcheck_cache_layers(
                "cache",
                tolerance=1.0,
            ),
            "commit_order_layers": _selfcheck_cache_layers(
                "commit_order",
                tolerance=0.0,
            ),
            "rows": [
                {
                    "row": row,
                    "candidate_tokens": [row + 1, row + 2],
                    "reference_tokens": [row + 1, row + 2],
                    "candidate_accepted_drafts": row,
                    "reference_accepted_drafts": row,
                }
                for row in range(2)
            ],
            "isolation": [
                {
                    "row": row,
                    "sibling_row": 1 - row,
                    "extracted_aliasing": False,
                    "sibling_unchanged_after_mutation": True,
                    "sibling_unchanged_after_commit": True,
                }
                for row in range(2)
            ],
        },
    }


def _dependencies(
    *,
    eligible=None,
    stock_calls: list[tuple[object, object]] | None = None,
    m6_calls: list[dict[str, object]] | None = None,
    scopes: list[tuple[int, object]] | None = None,
    expected_qlinear_count: int = 1,
    expected_geometry_histogram: tuple[tuple[int, int, int], ...] | None = None,
    commit_calls: list[dict[str, object]] | None = None,
):
    stock_calls = [] if stock_calls is None else stock_calls
    m6_calls = [] if m6_calls is None else m6_calls
    scopes = [] if scopes is None else scopes
    commit_calls = [] if commit_calls is None else commit_calls

    def stock(module, x):
        stock_calls.append((module, x))
        return "stock"

    def record_m6(kernel, x, weight, scales, biases, *, group_size):
        m6_calls.append(
            {
                "kernel": kernel,
                "x": x,
                "weight": weight,
                "scales": scales,
                "biases": biases,
                "group_size": group_size,
            }
        )
        return _Array((6, weight.shape[0]), x.dtype, "m6")

    def m6(x, weight, scales, biases, *, group_size):
        return record_m6(
            "ksplit2_bn4",
            x,
            weight,
            scales,
            biases,
            group_size=group_size,
        )

    def m6_kp1_bn2(x, weight, scales, biases, *, group_size):
        return record_m6(
            "kp1_bn2",
            x,
            weight,
            scales,
            biases,
            group_size=group_size,
        )

    def resolve_capture_config(*, target_width, **_kwargs):
        return SimpleNamespace(
            target_width=target_width,
            attention_cache_type=(
                "KVCache" if target_width == 1 else "BatchKVCache"
            ),
            capture_backend="linear_gdn_from_conv_tape",
            projection_path="stock",
            linear_conv_path="stock",
            authoritative_state_path="contiguous",
            gdn_tail_path="stock",
            residual_path="stock",
            hidden_variant="post_norm",
            layer_eval_every=0,
            layer_eval_schedule=(),
            layer_eval_context_threshold=0,
            layer_eval_max_q=8,
            tape_replay_tgy=8,
        )

    def configured_capture(model, input_ids, cache, *, config):
        model._runtime.calls.append(
            {
                "input_ids": input_ids,
                "cache": cache,
                "return_hidden": True,
                "hidden_variant": config.hidden_variant,
                "capture_backend": config.capture_backend,
            }
        )
        return "logits", "hidden", {"capture": True}

    def bind_capture_commit_route(
        *,
        target_width,
        row,
        **_kwargs,
    ):
        def commit(cache, captures, *, steps):
            commit_calls.append(
                {
                    "width": target_width,
                    "row": row,
                    "cache": cache,
                    "captures": captures,
                    "steps": steps,
                }
            )
            return cache

        return commit

    def build_compiled_width2_target(**kwargs):
        target = cohort._target_callable(
            execution=kwargs["execution"],
            capture_forward=kwargs["capture_forward"],
            fixed_scope=kwargs["fixed_scope"],
        )
        target.release_construction_state = lambda: None
        return target

    dependencies = SimpleNamespace(
        quantized_linear_type=_QLinear,
        bfloat16=BF16,
        float16=FP16,
        install_patch=lambda: {"installed": True},
        patch_snapshot=lambda: SimpleNamespace(
            installed=False,
            stock_call=stock,
        ),
        prepare_patch_lease=lambda: _PatchLease(stock),
        inspect_model_contract=lambda _runtime, _path: {
            "backend_id": "qwen3_next",
            "architecture_id": "qwen3-next-mtp",
            "native_mtp_enabled": True,
            "native_mtp_model_depth_max": 3,
        },
        m6_eligible=(
            (lambda m, k, n, bits, group_size, dtype: True)
            if eligible is None
            else eligible
        ),
        nax_qmm_m6=m6,
        nax_qmm_m6_qmv_wide_vec6=m6_kp1_bn2,
        numeric_self_check=lambda **_kwargs: 0.0,
        execute_width1_ticket=lambda runtime, ticket: (
            "solo-width1",
            runtime,
            ticket,
        ),
        fixed_execution=lambda routes, width: SimpleNamespace(
            routes=routes, width=width
        ),
        fixed_scope=lambda execution: _Scope(scopes, execution),
        configured_capture=configured_capture,
        resolve_capture_config=resolve_capture_config,
        bind_capture_commit_route=bind_capture_commit_route,
        build_compiled_width2_target=build_compiled_width2_target,
        expected_qlinear_count=expected_qlinear_count,
        actual_model_self_check=(
            lambda _runtime, _lane, *, qlinear_report: (
                _actual_selfcheck_report(qlinear_report)
            )
        ),
    )
    if expected_geometry_histogram is not None:
        dependencies.expected_geometry_histogram = expected_geometry_histogram
    return dependencies


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _qlinear_structure_sha256(
    modules: list[tuple[str, _QLinear]],
) -> str:
    records = []
    for path, module in modules:
        weight = module["weight"]
        records.append(
            {
                "bits": module.bits,
                "group_size": module.group_size,
                "k": int(weight.shape[1]) * (32 // module.bits),
                "mode": module.mode,
                "n": int(weight.shape[0]),
                "path": path,
                "type": f"{type(module).__module__}.{type(module).__qualname__}",
                "weight_dtype": str(weight.dtype),
                "weight_shape": [int(value) for value in weight.shape],
            }
        )
    records.sort(key=lambda item: item["path"])
    return _sha256_json(records)


def _layer_structure_sha256(layers: list[object]) -> str:
    records = []
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        records.append(
            {
                "attention_type": (
                    None
                    if attention is None
                    else f"{type(attention).__module__}.{type(attention).__qualname__}"
                ),
                "is_linear": bool(getattr(layer, "is_linear", False)),
                "layer_index": index,
                "layer_type": (
                    f"{type(layer).__module__}.{type(layer).__qualname__}"
                ),
            }
        )
    return _sha256_json(records)


def test_contract_module_import_does_not_require_mlx(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "sitecustomize.py").write_text(
        """
import importlib.abc
import sys

class BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mlx", "mlx_lm"}:
            raise ModuleNotFoundError(f"blocked {fullname}")
        return None

sys.meta_path.insert(0, BlockMLX())
""".lstrip(),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(blocker), str(ROOT)))
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from mtplx.qwen27b_mtp_cohort import Qwen27BK2DualLane; print('ok')",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_install_builds_complete_immutable_exact_qwen_route_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    modules = [
        ("model.layers.0.self_attn.q_proj", _QLinear(512, 256)),
        ("model.layers.0.mlp.down_proj", _QLinear(1024, 512)),
        ("mtp.layers.0.mlp.down_proj", _QLinear(1024, 512)),
    ]
    runtime = _Runtime(_model_path(tmp_path), modules)
    deps = _dependencies(expected_qlinear_count=2)
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies", lambda: deps
    )

    lane = install_qwen27b_k2_dual_lane(
        runtime,
        backend_id="qwen3_next",
        depth=2,
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
    )

    assert isinstance(lane, Qwen27BK2DualLane)
    assert lane.backend_id == "qwen3_next"
    assert lane.depth == 2
    assert lane.bits == 4
    assert lane.group_size == 64
    assert lane.activation_dtype is BF16
    assert lane.hidden_variant == "post_norm"
    assert lane.verify_strategy == "capture_commit"
    assert lane.verify_core == "linear-gdn-from-conv-tape"
    assert lane.max_width == 2
    width1_ticket = object()
    assert lane.width1_execute_ticket(width1_ticket) == (
        "solo-width1",
        runtime,
        width1_ticket,
    )
    assert isinstance(lane.qlinear_routes, MappingProxyType)
    assert set(lane.qlinear_routes) == {id(modules[0][1]), id(modules[1][1])}
    assert {
        (route.k, route.n, route.activation_dtype)
        for route in lane.qlinear_routes.values()
    } == {(512, 256, BF16), (1024, 512, BF16)}
    assert lane.construction_receipt["qlinear_module_count"] == 2
    assert lane.construction_receipt["actual_model_qlinear_module_count"] == 497
    assert lane.construction_receipt["post_prefill_cache_types"] == (
        "ArraysCache",
        "VllmMetalPagedKVCache",
    )
    assert runtime.qwen27b_k2_dual_lane is lane
    with pytest.raises(TypeError):
        lane.qlinear_routes[id(modules[0][1])] = lane.qlinear_routes[id(modules[0][1])]


def test_fixed_routes_use_captured_stock_for_width1_and_prebound_m6_for_width2(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _QLinear(512, 256)
    runtime = _Runtime(
        _model_path(tmp_path), [("model.layers.0.self_attn.q_proj", module)]
    )
    stock_calls: list[tuple[object, object]] = []
    m6_calls: list[dict[str, object]] = []
    scopes: list[tuple[int, object]] = []
    construction_open = [True]
    eligible_calls: list[tuple[object, ...]] = []

    def eligible(*args):
        if not construction_open[0]:
            raise AssertionError("eligibility reached after construction")
        eligible_calls.append(args)
        return True

    deps = _dependencies(
        eligible=eligible,
        stock_calls=stock_calls,
        m6_calls=m6_calls,
        scopes=scopes,
    )
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies", lambda: deps
    )
    lane = install_qwen27b_k2_dual_lane(
        runtime,
        backend_id="qwen3_next",
        depth=2,
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
    )
    construction_open[0] = False
    route = lane.qlinear_routes[id(module)]
    x1 = _Array((1, 3, 512))
    x2 = _Array((2, 3, 512))

    assert route.execute(x1, width=1) == "stock"
    y2 = route.execute(x2, width=2)

    assert stock_calls == [(module, x1)]
    assert y2.shape == (2, 3, 256)
    assert len(m6_calls) == 1
    assert m6_calls[0]["x"].shape == (6, 512)
    assert m6_calls[0]["weight"] is module["weight"]
    assert m6_calls[0]["scales"] is module["scales"]
    assert m6_calls[0]["biases"] is module["biases"]
    assert m6_calls[0]["group_size"] == 64
    assert m6_calls[0]["kernel"] == "kp1_bn2"
    assert eligible_calls == [(6, 512, 256, 4, 64, BF16)]

    result1 = lane.width1_target(input_ids="ids1", cache=["cache1"])
    result2 = lane.width2_target(input_ids="ids2", cache=["cache2"])

    assert result1 == TargetForwardResult(
        logits="logits",
        hidden="hidden",
        captures={"capture": True},
        cache=["cache1"],
    )
    assert result2.cache == ["cache2"]
    assert [width for width, _routes in scopes] == [1, 2]
    assert runtime.calls == [
        {
            "input_ids": "ids1",
            "cache": ["cache1"],
            "return_hidden": True,
            "hidden_variant": "post_norm",
            "capture_backend": "linear_gdn_from_conv_tape",
        },
        {
            "input_ids": "ids2",
            "cache": ["cache2"],
            "return_hidden": True,
            "hidden_variant": "post_norm",
            "capture_backend": "linear_gdn_from_conv_tape",
        },
    ]


@pytest.mark.parametrize(
    "missing_dependency",
    [
        "resolve_capture_config",
        "configured_capture",
        "bind_capture_commit_route",
        "build_compiled_width2_target",
    ],
)
def test_missing_configured_capture_dependency_fails_before_patch_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    missing_dependency: str,
) -> None:
    runtime = _Runtime(
        _model_path(tmp_path),
        [("model.layers.0.self_attn.q_proj", _QLinear(512, 256))],
    )
    prepare_calls: list[None] = []
    deps = _dependencies()
    delattr(deps, missing_dependency)
    deps.prepare_patch_lease = lambda: (
        prepare_calls.append(None) or _PatchLease(lambda *_args: None)
    )
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies", lambda: deps
    )

    with pytest.raises(RuntimeError, match="configured GDN capture dependencies"):
        install_qwen27b_k2_dual_lane(
            runtime,
            backend_id="qwen3_next",
            depth=2,
            verify_strategy="capture_commit",
            verify_core="linear-gdn-from-conv-tape",
        )

    assert prepare_calls == []
    assert runtime.qwen27b_k2_dual_lane is None


def test_installed_capture_commit_routes_are_prebound_by_width_and_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _Runtime(
        _model_path(tmp_path),
        [("model.layers.0.self_attn.q_proj", _QLinear(512, 256))],
    )
    commit_calls: list[dict[str, object]] = []
    deps = _dependencies(commit_calls=commit_calls)
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies", lambda: deps
    )
    lane = install_qwen27b_k2_dual_lane(
        runtime,
        backend_id="qwen3_next",
        depth=2,
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
    )

    width1 = lane.capture_commit_for(1, 0)
    width2_row0 = lane.capture_commit_for(2, 0)
    width2_row1 = lane.capture_commit_for(2, 1)
    assert width1(["w1"], {"capture": 1}, steps=1) == ["w1"]
    assert width2_row0(["w2"], {"capture": 2}, steps=2) == ["w2"]
    assert width2_row1(["w2"], {"capture": 2}, steps=1) == ["w2"]
    assert [(call["width"], call["row"], call["steps"]) for call in commit_calls] == [
        (1, 0, 1),
        (2, 0, 2),
        (2, 1, 1),
    ]


def test_install_is_atomic_and_names_path_module_and_invariant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    good = _QLinear(512, 256)
    bad = _QLinear(512, 256, group_size=32)
    runtime = _Runtime(
        _model_path(tmp_path),
        [
            ("model.layers.0.self_attn.q_proj", good),
            ("model.layers.7.mlp.down_proj", bad),
        ],
    )
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies",
        lambda: _dependencies(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        install_qwen27b_k2_dual_lane(
            runtime,
            backend_id="qwen3_next",
            depth=2,
            verify_strategy="capture_commit",
            verify_core="linear-gdn-from-conv-tape",
        )

    message = str(exc_info.value)
    assert str(runtime.model_path) in message
    assert "model.layers.7.mlp.down_proj" in message
    assert "group_size" in message
    assert runtime.qwen27b_k2_dual_lane is None


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("backend_id", "glm", "backend_id"),
        ("depth", 3, "depth"),
        ("verify_strategy", "batched", "verify_strategy"),
        ("verify_core", "stock", "verify_core"),
    ],
)
def test_install_rejects_non_exact_execution_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    runtime = _Runtime(
        _model_path(tmp_path),
        [("model.layers.0.self_attn.q_proj", _QLinear(512, 256))],
    )
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies",
        lambda: _dependencies(),
    )
    kwargs: dict[str, object] = {
        "backend_id": "qwen3_next",
        "depth": 2,
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
    }
    kwargs[field] = value

    with pytest.raises(RuntimeError, match=match):
        install_qwen27b_k2_dual_lane(runtime, **kwargs)
    assert runtime.qwen27b_k2_dual_lane is None


def test_target_for_width_rejects_every_other_value() -> None:
    lane = Qwen27BK2DualLane(
        backend_id="qwen3_next",
        depth=2,
        bits=4,
        group_size=64,
        activation_dtype=BF16,
        hidden_variant="post_norm",
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
        max_width=2,
        width1_target=lambda **_kwargs: None,
        width2_target=lambda **_kwargs: None,
        cache_routes=(),
        qlinear_routes=MappingProxyType({}),
        construction_receipt=MappingProxyType({}),
    )

    assert lane.target_for_width(1) is lane.width1_target
    assert lane.target_for_width(2) is lane.width2_target
    for width in (-1, 0, 3, 9):
        with pytest.raises(
            ValueError,
            match=rf"Qwen27BK2DualLane width must be 1 or 2, got {width}",
        ):
            lane.target_for_width(width)


def test_fixed_route_execution_source_has_no_dynamic_gate() -> None:
    source = inspect.getsource(FixedQLinearRoute.execute)

    assert "os.environ.get" not in source
    assert "m6_ksplit_eligible" not in source
    assert "lane_disabled" not in source
    assert "except" not in source


def test_runtime_fixed_target_entrypoint_only_selects_installed_width() -> None:
    from mtplx.runtime import MTPLXRuntime

    calls: list[tuple[int, object, object]] = []

    class Lane:
        def target_for_width(self, width: int):
            def target(*, input_ids, cache):
                calls.append((width, input_ids, cache))
                return "result"

            return target

    result = MTPLXRuntime.forward_qwen27b_k2_target(
        SimpleNamespace(),
        Lane(),
        2,
        "ids",
        ["cache"],
    )

    assert result == "result"
    assert calls == [(2, "ids", ["cache"])]


def test_late_numeric_failure_leaves_lane_and_process_patch_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    modules = [
        ("model.layers.0.self_attn.q_proj", _QLinear(512, 256)),
        ("model.layers.0.mlp.down_proj", _QLinear(1024, 512)),
    ]
    runtime = _Runtime(_model_path(tmp_path), modules)
    patch_installed = [False]
    install_calls: list[None] = []
    numeric_calls: list[str] = []
    deps = _dependencies(expected_qlinear_count=2)

    def numeric_self_check(*, module, route):
        del module
        numeric_calls.append(route.module_path)
        if len(numeric_calls) == 2:
            raise RuntimeError("late numeric mismatch")
        return 0.0

    lease = _PatchLease(
        deps.patch_snapshot().stock_call,
        install_calls,
    )

    def acquire():
        lease.acquire_calls.append(None)
        patch_installed[0] = True
        lease.active = True

    lease.acquire = acquire
    deps.prepare_patch_lease = lambda: lease
    deps.numeric_self_check = numeric_self_check
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies", lambda: deps
    )

    with pytest.raises(RuntimeError, match="late numeric mismatch"):
        install_qwen27b_k2_dual_lane(
            runtime,
            backend_id="qwen3_next",
            depth=2,
            verify_strategy="capture_commit",
            verify_core="linear-gdn-from-conv-tape",
        )

    assert len(numeric_calls) == 2
    assert install_calls == []
    assert patch_installed == [False]
    assert runtime.qwen27b_k2_dual_lane is None


def test_non_exact_qlinear_selfcheck_fails_before_patch_acquire(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _Runtime(
        _model_path(tmp_path),
        [("model.layers.0.self_attn.q_proj", _QLinear(512, 256))],
    )
    deps = _dependencies()
    deps.numeric_self_check = lambda **_kwargs: 0.01
    lease = _PatchLease(deps.patch_snapshot().stock_call, [])
    deps.prepare_patch_lease = lambda: lease
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies",
        lambda: deps,
    )

    with pytest.raises(RuntimeError, match="numeric self-check exceeded 0.0"):
        install_qwen27b_k2_dual_lane(
            runtime,
            backend_id="qwen3_next",
            depth=2,
            verify_strategy="capture_commit",
            verify_core="linear-gdn-from-conv-tape",
        )

    assert lease.acquire_calls == []
    assert runtime.qwen27b_k2_dual_lane is None


def test_actual_model_selfcheck_failure_releases_patch_and_restores_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _Runtime(
        _model_path(tmp_path),
        [("model.layers.0.self_attn.q_proj", _QLinear(512, 256))],
    )
    deps = _dependencies()
    lease = _PatchLease(deps.patch_snapshot().stock_call)
    seen: list[tuple[object, object]] = []
    deps.prepare_patch_lease = lambda: lease

    def fail_selfcheck(runtime_arg, lane, *, qlinear_report):
        seen.append((runtime_arg, lane))
        assert runtime_arg.qwen27b_k2_dual_lane is None
        assert qlinear_report["tested_module_count"] == 1
        raise RuntimeError("target parity diverged")

    deps.actual_model_self_check = fail_selfcheck
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies",
        lambda: deps,
    )

    with pytest.raises(RuntimeError, match="target parity diverged"):
        install_qwen27b_k2_dual_lane(
            runtime,
            backend_id="qwen3_next",
            depth=2,
            verify_strategy="capture_commit",
            verify_core="linear-gdn-from-conv-tape",
        )

    assert len(seen) == 1
    assert lease.active is False
    assert runtime.qwen27b_k2_dual_lane is None


def test_validated_selfcheck_releases_width2_construction_state_before_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _Runtime(
        _model_path(tmp_path),
        [("model.layers.0.self_attn.q_proj", _QLinear(512, 256))],
    )
    deps = _dependencies()
    release_observations: list[object] = []
    original_builder = deps.build_compiled_width2_target

    def build_compiled_width2_target(**kwargs):
        target = original_builder(**kwargs)

        def release_construction_state():
            release_observations.append(runtime.qwen27b_k2_dual_lane)

        target.release_construction_state = release_construction_state
        return target

    deps.build_compiled_width2_target = build_compiled_width2_target
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies",
        lambda: deps,
    )

    lane = install_qwen27b_k2_dual_lane(
        runtime,
        backend_id="qwen3_next",
        depth=2,
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
    )

    assert release_observations == [None]
    assert runtime.qwen27b_k2_dual_lane is lane


def test_missing_actual_selfcheck_fails_before_patch_acquire(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _Runtime(
        _model_path(tmp_path),
        [("model.layers.0.self_attn.q_proj", _QLinear(512, 256))],
    )
    deps = _dependencies()
    del deps.actual_model_self_check
    acquire_calls: list[None] = []
    deps.prepare_patch_lease = lambda: _PatchLease(
        deps.patch_snapshot().stock_call,
        acquire_calls,
    )
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies",
        lambda: deps,
    )

    with pytest.raises(RuntimeError, match="actual-model B2 self-check dependency"):
        install_qwen27b_k2_dual_lane(
            runtime,
            backend_id="qwen3_next",
            depth=2,
            verify_strategy="capture_commit",
            verify_core="linear-gdn-from-conv-tape",
        )

    assert acquire_calls == []
    assert runtime.qwen27b_k2_dual_lane is None


def test_unvalidated_actual_selfcheck_report_cannot_publish_lane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _Runtime(
        _model_path(tmp_path),
        [("model.layers.0.self_attn.q_proj", _QLinear(512, 256))],
    )
    deps = _dependencies()
    lease = _PatchLease(deps.patch_snapshot().stock_call)
    deps.prepare_patch_lease = lambda: lease

    def inflated_report(_runtime, _lane, *, qlinear_report):
        report = _actual_selfcheck_report(qlinear_report)
        report["target_cycle"]["output_comparisons"][0]["tolerance"] = 99.0
        return report

    deps.actual_model_self_check = inflated_report
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies",
        lambda: deps,
    )

    with pytest.raises(RuntimeError, match="tolerance must be 1.0"):
        install_qwen27b_k2_dual_lane(
            runtime,
            backend_id="qwen3_next",
            depth=2,
            verify_strategy="capture_commit",
            verify_core="linear-gdn-from-conv-tape",
        )

    assert lease.active is False
    assert runtime.qwen27b_k2_dual_lane is None


def test_same_count_but_mutated_control_geometry_fails_before_patch_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert EXPECTED_QLINEAR_GEOMETRY_HISTOGRAM == CONTROL_GEOMETRY_HISTOGRAM
    assert sum(count for _k, _n, count in CONTROL_GEOMETRY_HISTOGRAM) == 497
    modules: list[tuple[str, _QLinear]] = []
    for k, n, count in CONTROL_GEOMETRY_HISTOGRAM:
        modules.extend(
            (f"model.layers.{len(modules)}.qlinear", _QLinear(k, n))
            for _ in range(count)
        )
    assert len(modules) == 497
    modules[0] = (modules[0][0], _QLinear(5120, 52))
    runtime = _Runtime(_model_path(tmp_path), modules)
    install_calls: list[None] = []
    deps = _dependencies(
        expected_qlinear_count=497,
        expected_geometry_histogram=CONTROL_GEOMETRY_HISTOGRAM,
    )
    deps.prepare_patch_lease = lambda: _PatchLease(
        deps.patch_snapshot().stock_call,
        install_calls,
    )
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies", lambda: deps
    )

    with pytest.raises(RuntimeError, match="geometry histogram"):
        install_qwen27b_k2_dual_lane(
            runtime,
            backend_id="qwen3_next",
            depth=2,
            verify_strategy="capture_commit",
            verify_core="linear-gdn-from-conv-tape",
        )

    assert install_calls == []
    assert runtime.qwen27b_k2_dual_lane is None


def test_exact_control_rejects_fp16_activation_before_patch_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _Runtime(
        _model_path(tmp_path, dtype="float16"),
        [("model.layers.0.self_attn.q_proj", _QLinear(512, 256))],
    )
    install_calls: list[None] = []
    deps = _dependencies()
    deps.prepare_patch_lease = lambda: _PatchLease(
        deps.patch_snapshot().stock_call,
        install_calls,
    )
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies", lambda: deps
    )

    with pytest.raises(RuntimeError, match="BF16"):
        install_qwen27b_k2_dual_lane(
            runtime,
            backend_id="qwen3_next",
            depth=2,
            verify_strategy="capture_commit",
            verify_core="linear-gdn-from-conv-tape",
        )

    assert install_calls == []
    assert runtime.qwen27b_k2_dual_lane is None


def test_width_execution_scope_objects_are_prebound_once_at_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _Runtime(
        _model_path(tmp_path),
        [("model.layers.0.self_attn.q_proj", _QLinear(512, 256))],
    )
    construction_calls: list[SimpleNamespace] = []
    scopes: list[tuple[int, object]] = []
    deps = _dependencies(scopes=scopes)

    def fixed_execution(*, routes, width):
        execution = SimpleNamespace(routes=routes, width=width)
        construction_calls.append(execution)
        return execution

    deps.fixed_execution = fixed_execution
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies", lambda: deps
    )
    lane = install_qwen27b_k2_dual_lane(
        runtime,
        backend_id="qwen3_next",
        depth=2,
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
    )

    assert [execution.width for execution in construction_calls] == [1, 2]
    lane.width1_target(input_ids="ids1", cache=[])
    lane.width2_target(input_ids="ids2", cache=[])
    lane.width2_target(input_ids="ids3", cache=[])

    assert [execution.width for execution in construction_calls] == [1, 2]
    assert [width for width, _routes in scopes] == [1, 2, 2]


def test_same_histogram_with_swapped_qlinear_paths_fails_before_patch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert (
        cohort.EXPECTED_QLINEAR_STRUCTURE_SHA256
        == "7c83a60bf2afaf71f4894bfa98a2fd22ab561c69d3d17b9fdc67fad258a5908e"
    )
    pristine: list[tuple[str, _QLinear]] = []
    for k, n, count in CONTROL_GEOMETRY_HISTOGRAM:
        for _ in range(count):
            index = len(pristine)
            pristine.append((f"model.slot.{index}", _QLinear(k, n)))
    expected_sha256 = _qlinear_structure_sha256(pristine)
    mutated = list(pristine)
    left_path, left_module = mutated[0]
    right_path, right_module = mutated[-1]
    mutated[0] = (right_path, left_module)
    mutated[-1] = (left_path, right_module)
    assert _qlinear_structure_sha256(mutated) != expected_sha256

    runtime = _Runtime(_model_path(tmp_path), mutated)
    install_calls: list[None] = []
    deps = _dependencies(
        expected_qlinear_count=497,
        expected_geometry_histogram=CONTROL_GEOMETRY_HISTOGRAM,
    )
    deps.expected_qlinear_structure_sha256 = expected_sha256
    deps.prepare_patch_lease = lambda: _PatchLease(
        deps.patch_snapshot().stock_call,
        install_calls,
    )
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies", lambda: deps
    )

    with pytest.raises(RuntimeError, match="qlinear structural fingerprint"):
        install_qwen27b_k2_dual_lane(
            runtime,
            backend_id="qwen3_next",
            depth=2,
            verify_strategy="capture_commit",
            verify_core="linear-gdn-from-conv-tape",
        )

    assert install_calls == []
    assert runtime.qwen27b_k2_dual_lane is None


def test_serving_draft_lm_head_alias_keeps_canonical_target_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lm_head = _QLinear(512, 256)
    canonical = [("lm_head", lm_head)]
    runtime = _Runtime(
        _model_path(tmp_path),
        [
            ("_mtplx_draft_lm_head", lm_head),
            *canonical,
        ],
    )
    deps = _dependencies(expected_qlinear_count=1)
    deps.expected_geometry_histogram = ((512, 256, 1),)
    deps.expected_qlinear_structure_sha256 = _qlinear_structure_sha256(
        canonical
    )
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies", lambda: deps
    )

    lane = install_qwen27b_k2_dual_lane(
        runtime,
        backend_id="qwen3_next",
        depth=2,
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
    )

    route = lane.qlinear_routes[id(lm_head)]
    assert route.module_path == "lm_head"


def test_wrong_layer_pattern_fails_structural_fingerprint_before_patch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert (
        cohort.EXPECTED_LAYER_STRUCTURE_SHA256
        == "3ff2c2c8ac7cc348801dfd0341fe8afa8985750b34f303f1b610dc7cdbddfdfc"
    )
    class Attention:
        pass

    class Layer:
        def __init__(self, *, is_linear: bool):
            self.is_linear = is_linear
            if not is_linear:
                self.self_attn = Attention()

    pristine = [
        Layer(is_linear=(index % 4 != 3))
        for index in range(64)
    ]
    expected_sha256 = _layer_structure_sha256(pristine)
    mutated = [
        Layer(is_linear=(index % 4 != 3))
        for index in range(64)
    ]
    mutated[2] = Layer(is_linear=False)
    mutated[3] = Layer(is_linear=True)
    assert _layer_structure_sha256(mutated) != expected_sha256

    runtime = _Runtime(
        _model_path(tmp_path),
        [("model.layers.0.qlinear", _QLinear(512, 256))],
        layers=mutated,
    )
    install_calls: list[None] = []
    deps = _dependencies()
    deps.expected_layer_structure_sha256 = expected_sha256
    deps.prepare_patch_lease = lambda: _PatchLease(
        deps.patch_snapshot().stock_call,
        install_calls,
    )
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies", lambda: deps
    )

    with pytest.raises(RuntimeError, match="layer structural fingerprint"):
        install_qwen27b_k2_dual_lane(
            runtime,
            backend_id="qwen3_next",
            depth=2,
            verify_strategy="capture_commit",
            verify_core="linear-gdn-from-conv-tape",
        )

    assert install_calls == []
    assert runtime.qwen27b_k2_dual_lane is None


def test_caller_backend_cannot_override_failed_actual_model_inspection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _Runtime(
        _model_path(tmp_path),
        [("model.layers.0.qlinear", _QLinear(512, 256))],
    )
    install_calls: list[None] = []
    deps = _dependencies()
    deps.inspect_model_contract = lambda _runtime, _path: {
        "backend_id": "glm",
        "architecture_id": "glm-mtp",
        "native_mtp_enabled": True,
        "native_mtp_model_depth_max": 3,
    }
    deps.prepare_patch_lease = lambda: _PatchLease(
        deps.patch_snapshot().stock_call,
        install_calls,
    )
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies", lambda: deps
    )

    with pytest.raises(RuntimeError, match="actual model inspection"):
        install_qwen27b_k2_dual_lane(
            runtime,
            backend_id="qwen3_next",
            depth=2,
            verify_strategy="capture_commit",
            verify_core="linear-gdn-from-conv-tape",
        )

    assert install_calls == []
    assert runtime.qwen27b_k2_dual_lane is None


def test_construction_receipt_does_not_claim_post_prefill_observation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _Runtime(
        _model_path(tmp_path),
        [("model.layers.0.qlinear", _QLinear(512, 256))],
    )
    deps = _dependencies()
    monkeypatch.setattr(
        "mtplx.qwen27b_mtp_cohort._construction_dependencies", lambda: deps
    )

    lane = install_qwen27b_k2_dual_lane(
        runtime,
        backend_id="qwen3_next",
        depth=2,
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
    )

    assert "activation_dtype_after_real_prefill" not in lane.construction_receipt
    assert lane.construction_receipt["activation_dtype_from_config"] == "bfloat16"
    assert lane.construction_receipt["numeric_probe_dtype"] == str(BF16)
