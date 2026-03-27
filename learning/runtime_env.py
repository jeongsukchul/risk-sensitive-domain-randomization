import os


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
