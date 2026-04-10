import ctypes
import os
from pathlib import Path
import sys


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _prepend_library_dirs(paths: list[str]) -> None:
    current = [
        p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p
    ]
    pending = []
    for path in paths:
        if not path or not os.path.isdir(path):
            continue
        if path in current or path in pending:
            continue
        pending.append(path)
    if pending:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(pending + current)


def _nvidia_wheel_library_dirs() -> list[str]:
    site_packages = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
    )
    if not site_packages.is_dir():
        return []
    return [
        str(lib_dir)
        for lib_dir in sorted(site_packages.glob("*/lib"))
        if lib_dir.is_dir()
    ]


def _preload_nvidia_wheel_libraries() -> None:
    root = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
    )
    libraries = [
        root / "nvjitlink/lib/libnvJitLink.so.12",
        root / "cuda_runtime/lib/libcudart.so.12",
        root / "cuda_nvrtc/lib/libnvrtc.so.12",
        root / "cublas/lib/libcublasLt.so.12",
        root / "cublas/lib/libcublas.so.12",
        root / "cudnn/lib/libcudnn.so.9",
        root / "cufft/lib/libcufft.so.11",
        root / "cusolver/lib/libcusolver.so.11",
        root / "cusparse/lib/libcusparse.so.12",
        root / "nccl/lib/libnccl.so.2",
    ]
    for library in libraries:
        if not library.is_file():
            continue
        try:
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue


def configure_jax_runtime(
    *,
    default_platform: str = "gpu",
    default_mujoco_gl: str = "egl",
    default_preallocate: bool = False,
    default_enable_triton_gemm: bool = False,
) -> None:
    """Configures env vars before importing JAX.

    Triton GEMM is left disabled by default because it can produce noisy
    compiler/timer logs or regressions on some setups. Users can opt in with
    `RSDR_ENABLE_TRITON_GEMM=1`.
    """
    _prepend_library_dirs(_nvidia_wheel_library_dirs())
    _preload_nvidia_wheel_libraries()
    os.environ.setdefault("MUJOCO_GL", default_mujoco_gl)
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "true" if default_preallocate else "false",
    )
    os.environ.setdefault("JAX_PLATFORM_NAME", default_platform)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "2")
    enable_triton = _env_flag_enabled(
        "RSDR_ENABLE_TRITON_GEMM",
        default=default_enable_triton_gemm,
    )
    if not enable_triton:
        return

    xla_flags = os.environ.get("XLA_FLAGS", "").strip()
    triton_flag = "--xla_gpu_triton_gemm_any=True"
    if triton_flag not in xla_flags.split():
        os.environ["XLA_FLAGS"] = f"{xla_flags} {triton_flag}".strip()
