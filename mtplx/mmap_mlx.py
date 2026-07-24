"""JIT bridge for page-cache-backed MLX expert weights on Apple Silicon."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import ModuleType


_SOURCE = Path(__file__).resolve().parent / "native" / "mmap_mlx.cpp"


def _package_root(name: str) -> Path:
    module = importlib.import_module(name)
    paths = getattr(module, "__path__", None)
    if paths:
        return Path(next(iter(paths)))
    source = getattr(module, "__file__", None)
    if source:
        return Path(source).resolve().parent
    raise RuntimeError(f"cannot locate Python package {name!r}")


def _extension_path() -> Path:
    import mlx.core as mx

    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    identity = hashlib.sha256(
        _SOURCE.read_bytes()
        + str(getattr(mx, "__version__", "unknown")).encode("utf-8")
        + str(suffix).encode("utf-8")
    ).hexdigest()[:16]
    root = Path.home() / ".cache" / "mtplx" / "mmap-mlx" / identity
    root.mkdir(parents=True, exist_ok=True)
    return root / f"_mmap_mlx{suffix}"


def _build(output: Path) -> None:
    python_include = Path(sysconfig.get_paths()["include"])
    nanobind = _package_root("nanobind")
    mlx = _package_root("mlx")
    mlx_include = mlx / "include"
    mlx_library = mlx / "lib"
    nanobind_source = nanobind / "src" / "nb_combined.cpp"
    required = (
        python_include,
        nanobind_source,
        mlx_include / "mlx" / "array.h",
        mlx_library / "libmlx.dylib",
        _SOURCE,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing mmap MLX build inputs: {missing}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    command = [
        "clang++",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-O3",
        "-fvisibility=default",
        f"-I{python_include}",
        f"-I{nanobind / 'include'}",
        f"-I{nanobind / 'src'}",
        f"-I{nanobind / 'ext' / 'robin_map' / 'include'}",
        f"-I{mlx_include}",
        f"-I{mlx_include / 'metal_cpp'}",
        f"-L{mlx_library}",
        "-lmlx",
        "-framework",
        "Metal",
        "-framework",
        "Foundation",
        f"-Wl,-rpath,{mlx_library}",
        "-DNB_DOMAIN=mlx",
        "-D_METAL_",
        "-DACCELERATE_NEW_LAPACK",
        "-undefined",
        "dynamic_lookup",
        str(nanobind_source),
        str(_SOURCE),
        "-o",
        str(temporary),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            "failed to build file-backed MLX bridge:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    temporary.replace(output)


def load_mmap_extension() -> ModuleType:
    """Build once per source/MLX version and return the native module."""

    output = _extension_path()
    if not output.exists():
        _build(output)
    name = "mtplx._mmap_mlx"
    registered = sys.modules.get(name)
    if registered is not None:
        return registered
    loaded = importlib.util.find_spec(name)
    if loaded is not None:
        return importlib.import_module(name)
    spec = importlib.util.spec_from_file_location(name, output)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load mmap MLX extension from {output}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def mmap_u32(
    path: Path | str,
    offset: int,
    length: int,
    *,
    wired: bool = False,
):
    """Map one page-aligned file range into Metal without copying it."""

    return load_mmap_extension().mmap_u32(
        str(Path(path)), int(offset), int(length), bool(wired)
    )


def prefetch(arrays: list[object]) -> int:
    """Issue one native MADV_WILLNEED batch for mapped expert records."""

    return int(load_mmap_extension().prefetch(arrays))


def plan_region(
    path: Path | str,
    offset: int,
    length: int,
    *,
    shared_readonly: bool = False,
):
    """Validate and retain one fixed file range without mapping its pages."""

    return load_mmap_extension().MappedRegionPlan(
        str(Path(path)), int(offset), int(length), bool(shared_readonly)
    )


def activate_region(plan: object):
    """Map one construction-validated range for a bounded execution band."""

    return load_mmap_extension().MappedRegion(plan)


def metal_u32(region: object):
    """Create one transient no-copy Metal array over a mapped region."""

    return load_mmap_extension().metal_u32(region)


def metal_u32_slice(region: object, offset: int, length: int):
    """Create one no-copy Metal array over a page-aligned region subrange."""

    return load_mmap_extension().metal_u32_slice(
        region,
        int(offset),
        int(length),
    )


def allocate_metal_u8(length: int):
    """Allocate mutable shared Metal bytes without constructing an MLX zero op."""

    return load_mmap_extension().allocate_metal_u8(int(length))
