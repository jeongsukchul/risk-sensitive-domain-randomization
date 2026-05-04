from __future__ import annotations

import argparse
import atexit
from datetime import datetime
import importlib
import logging
import os
from pathlib import Path
import sys
import warnings
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", module=r"jax(\.|$)")
warnings.filterwarnings("ignore", module=r"jaxlib(\.|$)")
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("jaxlib").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
try:
    import imageio
except ImportError:  # optional dependency, only needed when writing GIFs
    imageio = None
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import functools
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import trange
import wandb
from learning.module.gbs.gbs_loss_test import VP
from learning.module.gbs.gbs_trainer_test import gbs_history_keys, run_gbs
from learning.module.gbs.target_examples_bridge import (
    is_target_example,
    load_target_example,
    target_example_has_reference,
    sample_target_example_reference,
    target_example_bounds,
    target_example_names,
)
from learning.module.gbs.sinkhorn_metrics import (
    energy_wasserstein_1d,
    effective_sample_size_from_log_weights,
    interatomic_wasserstein_1d,
    should_compute_interatomic_w2,
    sinkhorn_distance,
)
from learning.module.gbs.targets.target_family import (
    add_target_cli_args,
    build_target_params_from_args,
    get_fixed_target_setup,
    target_effective_num_modes,
    target_energy_values,
)
from learning.module.gmmvi.network import create_gmm_network_and_state
from learning.module.gmmvi.network import GMMTrainingState


ALGORITHM_SPECS = {
    "gmmvi": {"kind": "gmmvi", "label": "GMMVI", "color": "tab:orange"},
    "dis": {"kind": "gbs", "loss_mode": "dis", "label": "DIS", "color": "tab:blue"},
    "dis_lv": {"kind": "gbs", "loss_mode": "dis_lv", "label": "DIS-LV", "color": "tab:cyan"},
    "dds": {"kind": "gbs", "loss_mode": "dds", "label": "DDS", "color": "tab:purple"},
    "dds_lv": {"kind": "gbs", "loss_mode": "dds_lv", "label": "DDS-LV", "color": "tab:green"},
    "dds_exponential_lv": {
        "kind": "gbs",
        "loss_mode": "dds_exponential_lv",
        "label": "DDS-LV Exp",
        "color": "tab:brown",
    },
    "tr_dds_lv": {
        "kind": "gbs",
        "loss_mode": "tr_dds_lv",
        "label": "TR-DDS-LV",
        "color": "tab:red",
    },
}


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except (ValueError, OSError):
                continue
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except (ValueError, OSError):
                continue

    def isatty(self) -> bool:
        return bool(getattr(self.streams[0], "isatty", lambda: False)())

    def __getattr__(self, name: str):
        return getattr(self.streams[0], name)


def _normalize_x_axis_mode(value: str) -> str:
    normalized = value.replace("-", "_").lower()
    aliases = {
        "function_evaluations": "function_evaluations",
        "function_evals": "function_evaluations",
        "function_eval": "function_evaluations",
        "fevals": "function_evaluations",
        "policy_update_steps": "policy_update_steps",
        "policy_updates": "policy_update_steps",
        "p_updates": "policy_update_steps",
        "updates": "policy_update_steps",
    }
    if normalized not in aliases:
        raise argparse.ArgumentTypeError(
            "x_axis must be one of: function_evaluations, function_evals, "
            "policy_update_steps, policy_updates"
        )
    return aliases[normalized]


def _policy_update_step_values(num_steps: int, p_update_freq: int) -> np.ndarray:
    steps = np.arange(1, num_steps + 1, dtype=np.float64)
    if p_update_freq <= 0:
        return steps
    return steps / float(p_update_freq)


def _plot_x_values(result: dict[str, object], x_axis: str) -> np.ndarray:
    if x_axis == "policy_update_steps":
        return np.asarray(result["policy_update_steps"], dtype=np.float64)
    return np.asarray(result["function_evals"], dtype=np.float64)


def _plot_x_label(x_axis: str) -> str:
    if x_axis == "policy_update_steps":
        return "policy update steps"
    return "function evaluations"


def _target_beta_for_dim(experiment_beta: float, dim: int) -> float:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    return float(experiment_beta) * float(dim)


def _args_target_beta(args: argparse.Namespace) -> float:
    return float(getattr(args, "beta_times_dim", _target_beta_for_dim(args.beta, args.dim)))


def _limited_samples(samples, sample_size: int | None):
    samples = np.asarray(samples)
    if sample_size is None or sample_size <= 0:
        return samples
    return samples[: min(int(sample_size), samples.shape[0])]


def tanh_box_bijector(z: jax.Array, low: jax.Array, high: jax.Array) -> jax.Array:
    half = 0.5 * (high - low)
    mid = 0.5 * (high + low)
    return mid + half * jnp.tanh(z)


def tanh_box_logabsdet(z: jax.Array, low: jax.Array, high: jax.Array) -> jax.Array:
    z = jnp.atleast_2d(z)
    half = 0.5 * (high - low)
    jac_diag = half * (1.0 - jnp.tanh(z) ** 2)
    return jnp.sum(jnp.log(jnp.clip(jac_diag, 1e-12)), axis=-1)


def build_latent_target_loggrad_fn(
    *,
    to_box,
    logabsdet_fn,
    target_logprob_box_fn,
):
    def _single_latent_logprob(z: jax.Array, lam: jax.Array, policy_p: jax.Array) -> jax.Array:
        z_batch = z[None, :]
        target_lp = jnp.asarray(target_logprob_box_fn(to_box(z_batch), lam, policy_p)).reshape(())
        logabsdet = jnp.asarray(logabsdet_fn(z_batch)).reshape(())
        return target_lp + logabsdet

    single_grad = jax.grad(_single_latent_logprob, argnums=0)

    @jax.jit
    def target_loggrad(z: jax.Array, lam: jax.Array, policy_p: jax.Array) -> jax.Array:
        z_batch = jnp.atleast_2d(z)
        grads = jax.vmap(single_grad, in_axes=(0, None, None))(z_batch, lam, policy_p)
        if z.ndim == 1:
            return grads[0]
        return grads

    return target_loggrad


def compute_outer_iterations(function_evaluations: int, buffer_size: int) -> int:
    if buffer_size <= 0:
        raise ValueError(f"buffer_size must be positive, got {buffer_size}")
    if function_evaluations <= 0:
        raise ValueError(
            f"function_evaluations must be positive, got {function_evaluations}"
        )
    if function_evaluations % buffer_size != 0:
        raise ValueError(
            "gbs_function_evaluations must be divisible by gbs_buffer_size, got "
            f"{function_evaluations=} and {buffer_size=}"
        )
    return function_evaluations // buffer_size


def compute_gmmvi_iterations(function_evaluations: int, num_envs: int) -> int:
    if num_envs <= 0:
        raise ValueError(f"gmmvi_num_envs must be positive, got {num_envs}")
    if function_evaluations <= 0:
        raise ValueError(
            f"gmmvi_function_evaluations must be positive, got {function_evaluations}"
        )
    if function_evaluations % num_envs != 0:
        raise ValueError(
            "gmmvi_function_evaluations must be divisible by gmmvi_num_envs, got "
            f"{function_evaluations=} and {num_envs=}"
        )
    return function_evaluations // num_envs


def _resolve_target_config(version: str) -> tuple[str, str]:
    version = str(version)
    if version in ("target1", "target2", "target3"):
        return version, version
    if is_target_example(version):
        return version, version
    if version == "target_C":
        return "C", "target_C"
    if version == "target_A":
        return "A", "target_A"
    if version == "target_B":
        return "B", "target_B"
    return version, "target_B"


def _target_label(version: str) -> str:
    _, resolved_family = _resolve_target_config(version)
    return resolved_family


def load_target_utils(version: str):
    if version in ("target1", "target2", "target3") or is_target_example(version):
        raise ValueError(f"Target {version} does not use target utils modules.")
    utils_version, _ = _resolve_target_config(version)
    module_name = f"learning.module.gbs.targets.target_{utils_version}_notebook_utils"
    return importlib.import_module(module_name)


def build_compare_target_params(args: argparse.Namespace, dim: int):
    if str(args.target_version) in ("target1", "target2", "target3") or is_target_example(args.target_version):
        return None
    return build_target_params_from_args(args, dim)


def _is_fixed_target(version: str) -> bool:
    return str(version) in ("target1", "target2", "target3")


def _is_target_example(version: str) -> bool:
    return is_target_example(str(version))


def _target_example_dim01_logprob(target_example, grid: jax.Array, sample_dim: int):
    if hasattr(target_example, "log_prob_2D"):
        return target_example.log_prob_2D(grid)
    if sample_dim != 2:
        embedded = jnp.zeros((grid.shape[0], sample_dim), dtype=grid.dtype)
        embedded = embedded.at[:, :2].set(grid[:, :2])
        return target_example.log_prob(embedded)
    return target_example.log_prob(grid)


def build_gmmvi_fns_dynamic(gmm_network, num_envs: int, target_params, target_logprob_fn):
    del num_envs

    def _target_scalar_logprob(sample: jax.Array, lam: jax.Array, policy_p: jax.Array) -> jax.Array:
        return target_logprob_fn(sample[None, :], lam, target_params=target_params, policy_p=policy_p).reshape(())

    @jax.jit
    def gather_samples(train_state: GMMTrainingState, key: jax.Array, lam: jax.Array, policy_p: jax.Array):
        target_value_and_grad = jax.value_and_grad(_target_scalar_logprob, argnums=0)
        key, subkey = jax.random.split(key)
        new_samples, mapping = gmm_network.sample_selector.select_samples(train_state.model_state, subkey)
        new_target_lnpdfs, new_target_grads = jax.vmap(
            lambda sample: target_value_and_grad(sample, lam, policy_p)
        )(new_samples)
        new_sample_db_state = gmm_network.sample_selector.save_samples(
            train_state.model_state,
            train_state.sample_db_state,
            new_samples,
            new_target_lnpdfs,
            new_target_grads,
            mapping,
        )
        return GMMTrainingState(
            temperature=train_state.temperature,
            model_state=train_state.model_state,
            component_adaptation_state=train_state.component_adaptation_state,
            num_updates=train_state.num_updates,
            sample_db_state=new_sample_db_state,
            weight_stepsize=train_state.weight_stepsize,
        )

    @jax.jit
    def train_iter(train_state: GMMTrainingState, key: jax.Array, lam: jax.Array, policy_p: jax.Array):
        target_value_and_grad = jax.value_and_grad(_target_scalar_logprob, argnums=0)
        key, subkey = jax.random.split(key)
        new_samples, mapping = gmm_network.sample_selector.select_samples(train_state.model_state, subkey)
        new_target_lnpdfs, new_target_grads = jax.vmap(
            lambda sample: target_value_and_grad(sample, lam, policy_p)
        )(new_samples)
        new_sample_db_state = gmm_network.sample_selector.save_samples(
            train_state.model_state,
            train_state.sample_db_state,
            new_samples,
            new_target_lnpdfs,
            new_target_grads,
            mapping,
        )
        samples, mapping, sample_dist_densities, target_lnpdfs, target_lnpdf_grads = (
            gmm_network.sample_selector.select_train_datas(new_sample_db_state)
        )

        new_component_stepsizes = gmm_network.component_stepsize_fn(train_state.model_state)
        new_model_state = gmm_network.model.update_stepsizes(train_state.model_state, new_component_stepsizes)
        expected_hessian_neg, expected_grad_neg = gmm_network.more_ng_estimator(
            new_model_state,
            samples,
            sample_dist_densities,
            target_lnpdfs,
            target_lnpdf_grads,
        )
        new_model_state = gmm_network.component_updater(
            new_model_state,
            expected_hessian_neg,
            expected_grad_neg,
            new_model_state.stepsizes,
        )

        new_model_state = gmm_network.weight_updater(
            new_model_state,
            samples,
            sample_dist_densities,
            target_lnpdfs,
            train_state.weight_stepsize,
        )
        new_num_updates = train_state.num_updates + 1
        key, subkey = jax.random.split(key)
        new_model_state, new_component_adapter_state, new_sample_db_state = gmm_network.component_adapter(
            train_state.component_adaptation_state,
            new_sample_db_state,
            new_model_state,
            new_num_updates,
            subkey,
        )
        return GMMTrainingState(
            temperature=train_state.temperature,
            model_state=new_model_state,
            component_adaptation_state=new_component_adapter_state,
            num_updates=new_num_updates,
            sample_db_state=new_sample_db_state,
            weight_stepsize=train_state.weight_stepsize,
        )

    @functools.partial(jax.jit, static_argnums=(2,))
    def sample_model(train_state: GMMTrainingState, key: jax.Array, n_samples: int):
        return gmm_network.model.sample(train_state.model_state.gmm_state, key, n_samples)[0]

    @jax.jit
    def model_log_density(train_state: GMMTrainingState, samples: jax.Array):
        return jax.vmap(
            functools.partial(gmm_network.model.log_density, gmm_state=train_state.model_state.gmm_state)
        )(sample=samples)

    return gather_samples, train_iter, sample_model, model_log_density


def _hide_initial(values: np.ndarray) -> np.ndarray:
    masked = values.astype(np.float64, copy=True)
    if masked.size:
        masked[0] = np.nan
    return masked


def sample_target_reference(
    key: jax.Array,
    low: jax.Array,
    high: jax.Array,
    logprob_fn,
    sample_shape: tuple[int, int],
    grid_size: int = 128,
) -> jax.Array:
    if len(sample_shape) != 2 or sample_shape[1] != 2:
        raise ValueError(f"Expected sample_shape [N,2], got {sample_shape}")
    x, y = jnp.meshgrid(
        jnp.linspace(low[0], high[0], grid_size),
        jnp.linspace(low[1], high[1], grid_size),
        indexing="xy",
    )
    grid = jnp.stack([x.reshape(-1), y.reshape(-1)], axis=-1)
    logp = jnp.asarray(logprob_fn(grid)).reshape(-1)
    idx = jax.random.categorical(key, logp, shape=(sample_shape[0],))
    return grid[idx]


def energy_wasserstein_against_target(
    samples: jax.Array,
    ref_samples: jax.Array,
    logprob_fn,
) -> float:
    sample_energy = -jnp.asarray(logprob_fn(samples)).reshape(-1)
    ref_energy = -jnp.asarray(logprob_fn(ref_samples)).reshape(-1)
    sample_energy = jnp.sort(sample_energy)
    ref_energy = jnp.sort(ref_energy)
    n = min(sample_energy.shape[0], ref_energy.shape[0])
    return float(jnp.mean((sample_energy[:n] - ref_energy[:n]) ** 2))


def _safe_n_particles(dim: int, n_particles: int | None, n_spatial_dim: int) -> int:
    if n_particles is not None:
        return n_particles
    if dim % n_spatial_dim != 0:
        raise ValueError(f"dim={dim} must be divisible by n_spatial_dim={n_spatial_dim}")
    return dim // n_spatial_dim

def _plot_curve(
    ax,
    x_values: np.ndarray,
    values: np.ndarray,
    label: str,
    color: str,
    hide_initial_point: bool,
    *,
    linestyle: str = "-",
    linewidth: float = 2.0,
    marker: str | None = None,
    markersize: float = 5.0,
    zorder: float = 2.0,
    alpha: float = 0.95,
) -> None:
    curve = _hide_initial(values) if hide_initial_point else values
    curve = np.asarray(curve, dtype=np.float64)
    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    if x.shape != curve.shape:
        raise ValueError("x_values must have the same shape as values")
    finite_mask = np.isfinite(curve)
    if not np.any(finite_mask):
        return
    ax.plot(
        x[finite_mask],
        curve[finite_mask],
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markersize=markersize,
        zorder=zorder,
        alpha=alpha,
    )


def _last_finite(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite[-1])


def _prepend_scalar(values: np.ndarray, initial_value: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.concatenate(
        [np.asarray([initial_value], dtype=np.float64), arr],
        axis=0,
    )


def _prepend_history_point(
    hist: dict[str, np.ndarray],
    initial_values: dict[str, float],
) -> dict[str, np.ndarray]:
    prepped = {}
    for key, values in hist.items():
        initial_value = initial_values.get(key, np.nan)
        prepped[key] = _prepend_scalar(values, initial_value)
    return prepped


def _write_numeric_csv(
    path: Path,
    columns: list[tuple[str, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([np.asarray(values, dtype=np.float64) for _, values in columns])
    header = ",".join(name for name, _ in columns)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def save_raw_metric_data(
    algorithm_results: dict[str, dict[str, object]],
    uniform_baseline: dict[str, np.ndarray],
    output_dir: Path,
    *,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for algo_name, result in algorithm_results.items():
        hist = result["hist"]
        seed_values = np.full(
            np.asarray(result["function_evals"], dtype=np.float64).shape,
            float(seed),
            dtype=np.float64,
        )
        columns = [
            ("function_evals", result["function_evals"]),
            ("policy_update_steps", result["policy_update_steps"]),
            ("seed", seed_values),
        ]
        columns.extend((key.replace("/", "__"), values) for key, values in hist.items())
        _write_numeric_csv(output_dir / f"{algo_name}_metrics.csv", columns)

    first_result = next(iter(algorithm_results.values()))
    baseline_seed_values = np.full(
        np.asarray(first_result["function_evals"], dtype=np.float64).shape,
        float(seed),
        dtype=np.float64,
    )
    baseline_columns = [
        ("function_evals", first_result["function_evals"]),
        ("policy_update_steps", first_result["policy_update_steps"]),
        ("seed", baseline_seed_values),
    ]
    baseline_columns.extend(
        (key.replace("/", "__"), values) for key, values in uniform_baseline.items()
    )
    _write_numeric_csv(output_dir / "uniform_baseline_metrics.csv", baseline_columns)


def _set_robust_ylim(ax, series: list[np.ndarray], *, lower_q: float = 2.0, upper_q: float = 98.0) -> None:
    finite_parts = []
    for values in series:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_parts.append(arr)
    if not finite_parts:
        return

    all_values = np.concatenate(finite_parts)
    lo = float(np.nanpercentile(all_values, lower_q))
    hi = float(np.nanpercentile(all_values, upper_q))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return
    if np.isclose(lo, hi):
        pad = 0.05 * max(abs(lo), 1.0)
        ax.set_ylim(lo - pad, hi + pad)
        return

    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)


def _build_snapshot_iters(total_iters: int, num_frames: int) -> list[int]:
    if total_iters <= 0 or num_frames <= 0:
        return []
    count = min(total_iters, num_frames)
    return sorted(set(int(v) for v in np.linspace(0, total_iters - 1, count)))


def compute_uniform_metric_baseline(
    p_values: np.ndarray,
    *,
    beta: float,
    dim: int,
    sinkhorn_num_samples: int,
    target_params,
    target_version: str,
    seed: int,
    n_spatial_dim: int = 1,
    eval_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    sinkhorn_vals = []
    energy_w2_vals = []
    key = jax.random.PRNGKey(seed)
    n_samples = max(1, int(sinkhorn_num_samples))
    fixed_target = _is_fixed_target(target_version)
    example_target = _is_target_example(target_version)
    if fixed_target:
        fixed_logprob, low, high, _, _, _ = get_fixed_target_setup(target_version, beta)
    elif example_target:
        example = load_target_example(target_version, dim, n_spatial_dim)
        low, high = target_example_bounds(target_version, dim)
    else:
        target_utils = load_target_utils(target_version)
        sample_truncated_exponential = target_utils.sample_truncated_exponential

    p_array = np.asarray(p_values, dtype=np.float64)
    if eval_mask is None:
        eval_mask_arr = np.ones_like(p_array, dtype=bool)
    else:
        eval_mask_arr = np.asarray(eval_mask, dtype=bool).reshape(-1)
        if eval_mask_arr.shape != p_array.shape:
            raise ValueError("eval_mask must have the same shape as p_values")

    for p, should_eval in zip(p_array, eval_mask_arr):
        if (not should_eval) or np.isnan(p):
            sinkhorn_vals.append(np.nan)
            energy_w2_vals.append(np.nan)
            continue
        lam = float(beta * p)
        key, k_uniform, k_target = jax.random.split(key, 3)
        uniform_samples = jax.random.uniform(
            k_uniform,
            shape=(n_samples, dim),
            minval=0.0,
            maxval=1.0,
        )
        if fixed_target:
            target_samples = sample_target_reference(
                k_target, low, high, fixed_logprob, (n_samples, dim)
            )
            sinkhorn_vals.append(float(sinkhorn_distance(uniform_samples, target_samples)))
            energy_w2_vals.append(
                energy_wasserstein_against_target(uniform_samples, target_samples, fixed_logprob)
            )
        elif example_target:
            if target_example_has_reference(example):
                target_samples = sample_target_example_reference(example, k_target, n_samples)
                sinkhorn_vals.append(float(sinkhorn_distance(uniform_samples, target_samples)))
                energy_w2_vals.append(
                    energy_wasserstein_against_target(uniform_samples, target_samples, example.log_prob)
                )
            else:
                sinkhorn_vals.append(np.nan)
                energy_w2_vals.append(np.nan)
        else:
            try:
                target_samples = sample_truncated_exponential(
                    k_target,
                    lam,
                    (n_samples, dim),
                    target_params=target_params,
                    policy_p=float(p),
                )
            except TypeError:
                target_samples = sample_truncated_exponential(
                    k_target,
                    lam,
                    (n_samples, dim),
                )
            sinkhorn_vals.append(float(sinkhorn_distance(uniform_samples, target_samples)))
            energy_w2_vals.append(
                float(
                    energy_wasserstein_1d(
                        uniform_samples,
                        target_samples,
                        lam,
                        target_params=target_params,
                        policy_p=float(p),
                    )
                )
            )

    return {
        "uniform_baseline/sinkhorn": np.asarray(sinkhorn_vals, dtype=np.float64),
        "uniform_baseline/energy_w2": np.asarray(energy_w2_vals, dtype=np.float64),
    }
def gmmvi_training(args: argparse.Namespace) -> dict[str, np.ndarray]:
    dim = args.dim
    target_beta = _args_target_beta(args)
    gmmvi_steps = compute_gmmvi_iterations(
        args.gmmvi_function_evaluations, args.gmmvi_num_envs
    )
    fixed_target = _is_fixed_target(args.target_version)
    example_target = _is_target_example(args.target_version)
    if fixed_target:
        if dim != 2:
            raise ValueError(f"{args.target_version} is only defined for dim=2, got dim={dim}")
        target_logprob_fixed, low, high, _, _, _ = get_fixed_target_setup(args.target_version, target_beta)
        target_logprob = lambda x, lam, target_params=None, policy_p=None: target_logprob_fixed(x)
        compute_target_metrics = None
        optimal_p_from_target_mean = None
        sample_truncated_exponential = None
        update_p_with_ema_and_jump = None
        low = jnp.asarray(low)
        high = jnp.asarray(high)
    elif example_target:
        target_example = load_target_example(args.target_version, dim, args.n_spatial_dim)
        low, high = target_example_bounds(args.target_version, dim)
        target_logprob = lambda x, lam, target_params=None, policy_p=None: target_example.log_prob(x)
        compute_target_metrics = None
        optimal_p_from_target_mean = None
        sample_truncated_exponential = None
        update_p_with_ema_and_jump = None
    else:
        target_utils = load_target_utils(args.target_version)
        compute_target_metrics = target_utils.compute_target_metrics
        optimal_p_from_target_mean = target_utils.optimal_p_from_target_mean
        sample_truncated_exponential = target_utils.sample_truncated_exponential
        target_logprob = target_utils.target_logprob
        update_p_with_ema_and_jump = target_utils.update_p_with_ema_and_jump
        low = jnp.zeros(dim)
        high = jnp.ones(dim)
    n_particles = _safe_n_particles(dim, args.n_particles, args.n_spatial_dim)

    key = jax.random.PRNGKey(args.seed + 1)
    key, k_init, k_p0 = jax.random.split(key, 3)
    target_params = build_compare_target_params(args, dim)
    gmmvi_component_count = None if target_params is None else max(20, target_effective_num_modes(target_params))
    state, gmm_network = create_gmm_network_and_state(
        dim,
        args.gmmvi_num_envs,
        args.gmmvi_batch_size,
        k_init,
        prior_scale=args.gmmvi_prior_scale,
        bound_info=(low, high),
        max_components=gmmvi_component_count,
        num_initial_components=gmmvi_component_count,
    )
    gather_samples, train_iter, sample_model, model_log_density = build_gmmvi_fns_dynamic(
        gmm_network, args.gmmvi_num_envs, target_params, target_logprob
    )

    if args.initial_p is None:
        p = float(jax.random.uniform(k_p0, minval=0.0, maxval=1.0))
    else:
        p = float(np.clip(args.initial_p, 0.0, 1.0))

    initial_lambda = target_beta * p
    key, k_init_eval, k_init_metric = jax.random.split(key, 3)
    initial_eval_sample_size = max(int(args.gmmvi_eval_samples), int(args.metric_eval_samples))
    initial_samples = np.asarray(sample_model(state, k_init_eval, initial_eval_sample_size))
    initial_metric_samples = _limited_samples(initial_samples, args.metric_eval_samples)
    if fixed_target or example_target:
        initial_sample_mean = float(np.mean(initial_metric_samples))
    else:
        initial_sample_mean = float(
            np.mean(np.asarray(target_energy_values(jnp.asarray(initial_metric_samples), target_params, policy_p=p)))
        )
    if fixed_target:
        initial_forward_kl = float("nan")
        initial_reverse_kl = float("nan")
        initial_wasserstein = float("nan")
    elif example_target:
        initial_forward_kl = float("nan")
        initial_reverse_kl = float("nan")
        initial_wasserstein = float("nan")
    else:
        initial_forward_kl, initial_reverse_kl, initial_wasserstein = compute_target_metrics(
            initial_metric_samples,
            initial_lambda,
            target_params=target_params,
            num_bins=args.metric_num_bins,
            key=k_init_metric,
            policy_p=p,
        )
    key, k_init_sink = jax.random.split(key)
    initial_samples_jax = jnp.asarray(initial_metric_samples)
    if fixed_target:
        initial_sinkhorn_target = sample_target_reference(
            k_init_sink, low, high, target_logprob_fixed, initial_metric_samples.shape
        )
    elif example_target:
        initial_sinkhorn_target = (
            sample_target_example_reference(target_example, k_init_sink, initial_metric_samples.shape[0])
            if target_example_has_reference(target_example)
            else None
        )
    else:
        initial_sinkhorn_target = sample_truncated_exponential(
            k_init_sink,
            initial_lambda,
            initial_metric_samples.shape,
            target_params=target_params,
            policy_p=p,
        )
    initial_n_sink = min(args.sinkhorn_num_samples, initial_metric_samples.shape[0])
    initial_sinkhorn = (
        float("nan")
        if initial_sinkhorn_target is None
        else float(sinkhorn_distance(initial_samples_jax[:initial_n_sink], initial_sinkhorn_target[:initial_n_sink]))
    )
    if fixed_target:
        initial_ess = float("nan")
        initial_energy_w2 = energy_wasserstein_against_target(
            initial_samples_jax[:initial_n_sink], initial_sinkhorn_target[:initial_n_sink], target_logprob_fixed
        )
    elif example_target:
        initial_ess = float("nan")
        if initial_sinkhorn_target is None:
            initial_sinkhorn = float("nan")
            initial_energy_w2 = float("nan")
        else:
            initial_energy_w2 = energy_wasserstein_against_target(
                initial_samples_jax[:initial_n_sink], initial_sinkhorn_target[:initial_n_sink], target_example.log_prob
            )
    else:
        initial_ess = effective_sample_size_from_log_weights(
            target_logprob(initial_samples_jax, initial_lambda, target_params=target_params, policy_p=p)
        )
        initial_energy_w2 = float(
            energy_wasserstein_1d(
                initial_samples_jax[:initial_n_sink],
                initial_sinkhorn_target[:initial_n_sink],
                initial_lambda,
                target_params=target_params,
                policy_p=p,
            )
        )
    if should_compute_interatomic_w2(n_particles) and initial_sinkhorn_target is not None:
        initial_interatomic_w2 = float(
            interatomic_wasserstein_1d(
                initial_samples_jax[:initial_n_sink],
                initial_sinkhorn_target[:initial_n_sink],
                n_particles=n_particles,
                n_spatial_dim=args.n_spatial_dim,
            )
        )
    else:
        initial_interatomic_w2 = float("nan")
    if fixed_target or example_target:
        initial_optimal_p, initial_target_mean = 1.0, float("nan")
    else:
        initial_optimal_p, initial_target_mean = optimal_p_from_target_mean(
            initial_lambda,
            args.tau,
            args.safe_q,
            target_params=target_params,
        )
    initial_hist_values = {
        "target/p": float(p),
        "target/lambda": float(initial_lambda),
        "target/sample_mean": float(initial_sample_mean),
        "target/forward_kl": float(initial_forward_kl),
        "target/reverse_kl": float(initial_reverse_kl),
        "target/wasserstein": float(initial_wasserstein),
        "target/sinkhorn": float(initial_sinkhorn),
        "target/ess": float(initial_ess),
        "target/energy_w2": float(initial_energy_w2),
        "target/interatomic_w2": float(initial_interatomic_w2),
        "target/target_mean": float(initial_target_mean),
        "target/optimal_p": float(initial_optimal_p),
        "target/p_updated": 0.0,
        "target/p_jumped": 0.0,
        "target/p_base": float(jax.nn.sigmoid(args.tau * (initial_sample_mean - args.safe_q))),
        "target/p_ema": float(p),
        "model/num_components": int(state.model_state.gmm_state.num_components),
    }

    hist: dict[str, list[float]] = {
        "target/p": [],
        "target/lambda": [],
        "target/sample_mean": [],
        "target/forward_kl": [],
        "target/reverse_kl": [],
        "target/wasserstein": [],
        "target/sinkhorn": [],
        "target/ess": [],
        "target/energy_w2": [],
        "target/interatomic_w2": [],
        "target/target_mean": [],
        "target/optimal_p": [],
        "target/p_updated": [],
        "target/p_jumped": [],
        "target/p_base": [],
        "target/p_ema": [],
        "model/num_components": [],
    }
    metric_eval_iters = set()
    if args.max_eval_points is None or gmmvi_steps <= args.max_eval_points:
        metric_eval_iters = set(range(gmmvi_steps))
    else:
        metric_eval_iters = set(
            np.unique(np.linspace(0, gmmvi_steps - 1, args.max_eval_points).astype(int)).tolist()
        )

    snapshot_iters = set(_build_snapshot_iters(gmmvi_steps, args.gif_num_frames) if args.save_dim01_gif else [])
    snapshots: list[dict[str, object]] = []

    current_lambda = target_beta * p
    for _ in range(max(args.gmmvi_batch_size // args.gmmvi_num_envs, 1)):
        key, subkey = jax.random.split(key)
        state = gather_samples(state, subkey, jnp.asarray(current_lambda), jnp.asarray(p, dtype=jnp.float32))

    for step in trange(gmmvi_steps, desc="GMMVI", leave=False):
        current_lambda = target_beta * p

        key, subkey = jax.random.split(key)
        state = train_iter(state, subkey, jnp.asarray(current_lambda), jnp.asarray(p, dtype=jnp.float32))

        key, k_eval, k_metric = jax.random.split(key, 3)
        eval_sample_size = max(int(args.gmmvi_eval_samples), int(args.metric_eval_samples))
        samples = np.asarray(sample_model(state, k_eval, eval_sample_size))
        _ = model_log_density(state, jnp.asarray(samples))
        metric_samples = _limited_samples(samples, args.metric_eval_samples)

        if fixed_target or example_target:
            sample_mean = float(np.mean(metric_samples))
        else:
            sample_mean = float(
                np.mean(np.asarray(target_energy_values(jnp.asarray(metric_samples), target_params, policy_p=p)))
            )
        if step in metric_eval_iters:
            if fixed_target:
                forward_kl = float("nan")
                reverse_kl = float("nan")
                wasserstein = float("nan")
            elif example_target:
                forward_kl = float("nan")
                reverse_kl = float("nan")
                wasserstein = float("nan")
            else:
                forward_kl, reverse_kl, wasserstein = compute_target_metrics(
                    metric_samples,
                    current_lambda,
                    target_params=target_params,
                    num_bins=args.metric_num_bins,
                    key=k_metric,
                    policy_p=p,
                )
            key, k_sink = jax.random.split(key)
            samples_jax = jnp.asarray(metric_samples)
            if fixed_target:
                sinkhorn_target = sample_target_reference(
                    k_sink, low, high, target_logprob_fixed, metric_samples.shape
                )
            elif example_target:
                sinkhorn_target = (
                    sample_target_example_reference(target_example, k_sink, metric_samples.shape[0])
                    if target_example_has_reference(target_example)
                    else None
                )
            else:
                sinkhorn_target = sample_truncated_exponential(
                    k_sink,
                    current_lambda,
                    metric_samples.shape,
                    target_params=target_params,
                    policy_p=p,
                )
            n_sink = min(args.sinkhorn_num_samples, metric_samples.shape[0])
            sinkhorn = (
                float("nan")
                if sinkhorn_target is None
                else sinkhorn_distance(samples_jax[:n_sink], sinkhorn_target[:n_sink])
            )
            if fixed_target:
                ess = float("nan")
                energy_w2 = energy_wasserstein_against_target(
                    samples_jax[:n_sink], sinkhorn_target[:n_sink], target_logprob_fixed
                )
            elif example_target:
                ess = float("nan")
                if sinkhorn_target is None:
                    sinkhorn = float("nan")
                    energy_w2 = float("nan")
                else:
                    energy_w2 = energy_wasserstein_against_target(
                        samples_jax[:n_sink], sinkhorn_target[:n_sink], target_example.log_prob
                    )
            else:
                ess = effective_sample_size_from_log_weights(
                    target_logprob(samples_jax, current_lambda, target_params=target_params, policy_p=p)
                )
                energy_w2 = float(
                    energy_wasserstein_1d(
                        samples_jax[:n_sink],
                        sinkhorn_target[:n_sink],
                        current_lambda,
                        target_params=target_params,
                        policy_p=p,
                    )
                )
            if should_compute_interatomic_w2(n_particles):
                interatomic_w2 = float(
                    interatomic_wasserstein_1d(
                        samples_jax[:n_sink],
                        sinkhorn_target[:n_sink],
                        n_particles=n_particles,
                        n_spatial_dim=args.n_spatial_dim,
                    )
                )
            else:
                interatomic_w2 = float("nan")
        else:
            forward_kl = float("nan")
            reverse_kl = float("nan")
            wasserstein = float("nan")
            sinkhorn = float("nan")
            ess = float("nan")
            energy_w2 = float("nan")
            interatomic_w2 = float("nan")
        if fixed_target or example_target:
            optimal_p, target_mean = 1.0, float("nan")
        else:
            optimal_p, target_mean = optimal_p_from_target_mean(
                current_lambda,
                args.tau,
                args.safe_q,
                target_params=target_params,
            )

        hist["target/p"].append(float(p))
        hist["target/lambda"].append(float(current_lambda))
        hist["target/sample_mean"].append(sample_mean)
        hist["target/forward_kl"].append(float(forward_kl))
        hist["target/reverse_kl"].append(float(reverse_kl))
        hist["target/wasserstein"].append(float(wasserstein))
        hist["target/sinkhorn"].append(float(sinkhorn))
        hist["target/ess"].append(float(ess))
        hist["target/energy_w2"].append(energy_w2)
        hist["target/interatomic_w2"].append(interatomic_w2)
        hist["target/target_mean"].append(float(target_mean))
        hist["target/optimal_p"].append(float(optimal_p))
        hist["model/num_components"].append(int(state.model_state.gmm_state.num_components))

        should_update_p = (not fixed_target) and (not example_target) and args.p_update_freq > 0 and ((step + 1) % args.p_update_freq == 0)
        hist["target/p_updated"].append(float(should_update_p))
        hist["target/p_jumped"].append(0.0)
        hist["target/p_base"].append(float(jax.nn.sigmoid(args.tau * (sample_mean - args.safe_q))))
        hist["target/p_ema"].append(float(p))

        if should_update_p:
            key, k_update = jax.random.split(key)
            p, base_p, ema_p, jumped = update_p_with_ema_and_jump(
                prev_p=p,
                sample_mean_g=sample_mean,
                tau=args.tau,
                q=args.safe_q,
                ema_alpha=args.p_ema_alpha,
                jump_prob=args.p_jump_prob,
                key=k_update,
            )
            hist["target/p"][-1] = float(p)
            hist["target/p_jumped"][-1] = float(jumped)
            hist["target/p_base"][-1] = float(base_p)
            hist["target/p_ema"][-1] = float(ema_p)

        if step in snapshot_iters:
            key, k_snapshot = jax.random.split(key)
            snapshot_samples = np.asarray(sample_model(state, k_snapshot, args.dim01_sample_size))
            snapshots.append(
                {
                    "iter": int(step),
                    "p": float(p),
                    "samples": snapshot_samples,
                }
            )

    final_samples = np.asarray(sample_model(state, key, args.dim01_sample_size))
    hist_arrays = {key: np.asarray(value, dtype=np.float64) for key, value in hist.items()}
    hist_arrays = _prepend_history_point(hist_arrays, initial_hist_values)
    return {
        "hist": hist_arrays,
        "final_samples": final_samples,
        "snapshots": snapshots,
        "function_evals": _prepend_scalar(
            np.arange(1, gmmvi_steps + 1, dtype=np.float64) * float(args.gmmvi_num_envs),
            0.0,
        ),
        "policy_update_steps": _prepend_scalar(
            _policy_update_step_values(gmmvi_steps, args.p_update_freq),
            0.0,
        ),
    }


def gbs_training(args: argparse.Namespace, loss_mode: str) -> dict[str, np.ndarray]:
    target_params = build_compare_target_params(args, args.dim)
    target_beta = _args_target_beta(args)
    gbs_outer_updates = compute_outer_iterations(
        args.gbs_function_evaluations, args.gbs_buffer_size
    )
    snapshot_iters = (
        _build_snapshot_iters(gbs_outer_updates, args.gif_num_frames)
        if args.save_dim01_gif
        else []
    )
    artifact_dir = Path(getattr(args, "run_output_dir", args.output_dir)) / "gbs_artifacts" / f"compare_{loss_mode}"
    fixed_target = _is_fixed_target(args.target_version)
    example_target = _is_target_example(args.target_version)
    if fixed_target:
        if args.dim != 2:
            raise ValueError(f"{args.target_version} is only defined for dim=2, got dim={args.dim}")
        target_logprob_fixed, low, high, prior_loc, clip_prior, process_center = get_fixed_target_setup(
            args.target_version, target_beta
        )
        proc = VP(
            diff_coeff_sq_min=0.01,
            diff_coeff_sq_max=10.0,
            scale_diff_coeff=args.gbs_scale_diff,
            terminal_t=1.0,
            generative=False,
            sign=-1.0,
        )
        low = jnp.asarray(low)
        high = jnp.asarray(high)
        if args.use_tanh_bijection:
            to_box = lambda z: tanh_box_bijector(z, low=low, high=high)
            logabsdet_fn = lambda z: tanh_box_logabsdet(z, low=low, high=high)
            latent_prior_loc = jnp.zeros(args.dim, dtype=jnp.float32)
            process_center = jnp.zeros(args.dim, dtype=jnp.float32)
            clip_prior_without_tanh = False
        else:
            to_box = lambda z: z
            logabsdet_fn = lambda z: jnp.zeros((z.shape[0],), dtype=z.dtype)
            latent_prior_loc = jnp.asarray(prior_loc)
            process_center = jnp.asarray(process_center)
            clip_prior_without_tanh = clip_prior
        use_tanh_bijection = args.use_tanh_bijection
        target_logprob_box_fn = lambda x, lam, policy_p: target_logprob_fixed(x)
        target_loggrad_latent_fn = build_latent_target_loggrad_fn(
            to_box=to_box,
            logabsdet_fn=logabsdet_fn,
            target_logprob_box_fn=target_logprob_box_fn,
        )
        sample_mean_fn = lambda x, policy_p: jnp.mean(x)
        compute_metrics_fn = lambda x, lam, key, policy_p: (
            float("nan"),
            float("nan"),
            float("nan"),
        )
        sample_reference_fn = lambda key, lam, shape, policy_p: sample_target_reference(
            key, low, high, target_logprob_fixed, shape
        )
        energy_w2_fn = lambda x, ref, lam, policy_p: energy_wasserstein_against_target(
            x, ref, target_logprob_fixed
        )
        optimal_p_fn = lambda lam, tau, q: (1.0, float("nan"))
        update_p_fn = lambda prev_p, sample_mean_g, tau, q, ema_alpha, jump_prob, key: (
            prev_p,
            prev_p,
            prev_p,
            False,
        )
    elif example_target:
        example = load_target_example(args.target_version, args.dim, args.n_spatial_dim)
        low, high = target_example_bounds(args.target_version, args.dim)
        proc = VP(
            diff_coeff_sq_min=0.01,
            diff_coeff_sq_max=10.0,
            scale_diff_coeff=args.gbs_scale_diff,
            terminal_t=1.0,
            generative=False,
            sign=-1.0,
        )
        to_box = lambda z: z
        latent_prior_loc = jnp.zeros(args.dim, dtype=jnp.float32)
        process_center = jnp.zeros(args.dim, dtype=jnp.float32)
        clip_prior_without_tanh = False
        use_tanh_bijection = False
        logabsdet_fn = lambda z: jnp.zeros((z.shape[0],), dtype=z.dtype)
        target_logprob_box_fn = lambda x, lam, policy_p: example.log_prob(x)
        target_loggrad_latent_fn = build_latent_target_loggrad_fn(
            to_box=to_box,
            logabsdet_fn=logabsdet_fn,
            target_logprob_box_fn=target_logprob_box_fn,
        )
        sample_mean_fn = lambda x, policy_p: jnp.mean(x)
        compute_metrics_fn = lambda x, lam, key, policy_p: (
            float("nan"),
            float("nan"),
            float("nan"),
        )
        sample_reference_fn = (
            (lambda key, lam, shape, policy_p: sample_target_example_reference(example, key, shape[0]))
            if target_example_has_reference(example)
            else None
        )
        energy_w2_fn = lambda x, ref, lam, policy_p: energy_wasserstein_against_target(
            x, ref, example.log_prob
        )
        optimal_p_fn = lambda lam, tau, q: (1.0, float("nan"))
        update_p_fn = lambda prev_p, sample_mean_g, tau, q, ema_alpha, jump_prob, key: (
            prev_p,
            prev_p,
            prev_p,
            False,
        )
    else:
        target_utils = load_target_utils(args.target_version)
        low = jnp.zeros(args.dim, dtype=jnp.float32)
        high = jnp.ones(args.dim, dtype=jnp.float32)
        proc = VP(
            diff_coeff_sq_min=0.01,
            diff_coeff_sq_max=10.0,
            scale_diff_coeff=args.gbs_scale_diff,
            terminal_t=1.0,
            generative=False,
            sign=-1.0,
        )
        to_box = lambda z: target_utils.tanh_box_bijector(z, low=low, high=high)
        target_logprob_box_fn = lambda x, lam, policy_p: target_utils.target_logprob(
            x, lam, target_params=target_params, policy_p=policy_p
        )
        logabsdet_fn = lambda z: target_utils.tanh_box_logabsdet(z, low=low, high=high)
        target_loggrad_latent_fn = build_latent_target_loggrad_fn(
            to_box=to_box,
            logabsdet_fn=logabsdet_fn,
            target_logprob_box_fn=target_logprob_box_fn,
        )
        if args.target_version == "target_A":
            sample_mean_fn = lambda x, policy_p: jnp.mean(x)
        else:
            sample_mean_fn = lambda x, policy_p: jnp.mean(
                target_energy_values(x, target_params, policy_p=policy_p)
            )
        compute_metrics_fn = lambda x, lam, key, policy_p: target_utils.compute_target_metrics(
            x,
            lam,
            target_params=target_params,
            num_bins=args.metric_num_bins,
            key=key,
            policy_p=policy_p,
        )
        sample_reference_fn = lambda key, lam, shape, policy_p: target_utils.sample_truncated_exponential(
            key, lam, shape, target_params=target_params, policy_p=policy_p
        )
        energy_w2_fn = lambda x, ref, lam, policy_p: energy_wasserstein_1d(
            x, ref, lam, target_params=target_params, policy_p=policy_p
        )
        optimal_p_fn = lambda lam, tau, q: target_utils.optimal_p_from_target_mean(
            lam, tau, q, target_params=target_params
        )
        latent_prior_loc = jnp.zeros(args.dim, dtype=jnp.float32)
        process_center = jnp.zeros(args.dim, dtype=jnp.float32)
        clip_prior_without_tanh = False
        use_tanh_bijection = True
        update_p_fn = target_utils.update_p_with_ema_and_jump

    init_key = jax.random.PRNGKey(args.seed)
    init_key, k_p0 = jax.random.split(init_key)
    if args.initial_p is None:
        initial_p = float(jax.random.uniform(k_p0, minval=0.0, maxval=1.0))
    else:
        initial_p = float(np.clip(args.initial_p, 0.0, 1.0))
    initial_lambda = target_beta * initial_p
    init_key, k_prior = jax.random.split(init_key)
    initial_eval_sample_size = int(args.metric_eval_samples)
    prior_latent = (
        jax.random.normal(k_prior, shape=(initial_eval_sample_size, args.dim), dtype=jnp.float32) * args.gbs_init_std
        + latent_prior_loc[None, :]
    )
    if clip_prior_without_tanh:
        prior_latent = jnp.clip(prior_latent, low, high)
    initial_xT = to_box(prior_latent)
    initial_metric_xT = initial_xT[: min(int(args.metric_eval_samples), int(initial_xT.shape[0]))]
    initial_sample_mean = float(sample_mean_fn(initial_metric_xT, initial_p))
    init_key, k_metric = jax.random.split(init_key)
    initial_forward_kl, initial_reverse_kl, initial_wasserstein = compute_metrics_fn(
        initial_metric_xT, initial_lambda, k_metric, initial_p
    )
    initial_ess = effective_sample_size_from_log_weights(
        target_logprob_box_fn(initial_metric_xT, initial_lambda, initial_p)
    )
    init_key, k_sink = jax.random.split(init_key)
    initial_sinkhorn_target = None if sample_reference_fn is None else sample_reference_fn(
        k_sink, initial_lambda, initial_metric_xT.shape, initial_p
    )
    initial_n_sink = min(int(args.sinkhorn_num_samples), int(initial_metric_xT.shape[0]))
    if initial_sinkhorn_target is None:
        initial_sinkhorn = float("nan")
        initial_energy_w2 = float("nan")
        initial_interatomic_w2 = float("nan")
    else:
        initial_sinkhorn = float(
            sinkhorn_distance(initial_metric_xT[:initial_n_sink], initial_sinkhorn_target[:initial_n_sink])
        )
        initial_energy_w2 = float(
            energy_w2_fn(
                initial_metric_xT[:initial_n_sink],
                initial_sinkhorn_target[:initial_n_sink],
                initial_lambda,
                initial_p,
            )
        )
        if args.dim % args.n_spatial_dim == 0:
            initial_interatomic_w2 = float(
                interatomic_wasserstein_1d(
                    initial_metric_xT[:initial_n_sink],
                    initial_sinkhorn_target[:initial_n_sink],
                    n_particles=args.n_particles,
                    n_spatial_dim=args.n_spatial_dim,
                )
            )
        else:
            initial_interatomic_w2 = float("nan")
    initial_optimal_p, initial_target_mean = optimal_p_fn(initial_lambda, args.tau, args.safe_q)
    initial_hist_values = {
        "target/p": float(initial_p),
        "target/lambda": float(initial_lambda),
        "target/sample_mean": float(initial_sample_mean),
        "target/forward_kl": float(initial_forward_kl),
        "target/reverse_kl": float(initial_reverse_kl),
        "target/wasserstein": float(initial_wasserstein),
        "target/sinkhorn": float(initial_sinkhorn),
        "target/ess": float(initial_ess),
        "target/energy_w2": float(initial_energy_w2),
        "target/interatomic_w2": float(initial_interatomic_w2),
        "target/target_mean": float(initial_target_mean),
        "target/optimal_p": float(initial_optimal_p),
        "target/p_updated": 0.0,
        "target/p_jumped": 0.0,
        "target/p_base": float(jax.nn.sigmoid(args.tau * (initial_sample_mean - args.safe_q))),
        "target/p_ema": float(initial_p),
    }
    for key in gbs_history_keys(loss_mode):
        initial_hist_values[key] = float("nan")

    result = run_gbs(
        low=low,
        high=high,
        dim=args.dim,
        function_evaluations=args.gbs_function_evaluations,
        buffer_size=args.gbs_buffer_size,
        num_steps=args.gbs_num_steps,
        lr=args.gbs_lr,
        init_std=args.gbs_init_std,
        seed=args.seed,
        beta=target_beta,
        tau=args.tau,
        q=args.safe_q,
        initial_p=args.initial_p,
        p_update_freq=args.p_update_freq,
        p_ema_alpha=args.p_ema_alpha,
        p_jump_prob=args.p_jump_prob,
        loss_mode=loss_mode,
        sinkhorn_num_samples=args.sinkhorn_num_samples,
        n_particles=args.n_particles,
        n_spatial_dim=args.n_spatial_dim,
        save_dir=artifact_dir,
        gif_path=None,
        snap_iters=snapshot_iters,
        model_type=args.gbs_model_type,
        model_num_layers=args.gbs_model_num_layers,
        model_num_hid=args.gbs_model_num_hid,
        gbs_scale_diff=args.gbs_scale_diff,
        final_sample_size=args.dim01_sample_size,
        max_rnd=1e8,
        trust_region_bound=args.gbs_trust_region_bound,
        trust_region_lambda_max=args.gbs_trust_region_lambda_max,
        trust_region_lambda_grid_size=args.gbs_trust_region_lambda_grid_size,
        minibatch_size=args.gbs_minibatch_size,
        minibatch_steps=args.gbs_minibatch_steps,
        return_snapshots=args.save_dim01_gif,
        snapshot_sample_size=args.dim01_sample_size,
        metric_eval_samples=args.metric_eval_samples,
        max_metric_eval_points=args.max_eval_points,
        process=proc,
        latent_prior_loc=latent_prior_loc,
        process_center=process_center,
        clip_prior_without_tanh=clip_prior_without_tanh,
        use_tanh_bijection=use_tanh_bijection,
        logabsdet_fn=logabsdet_fn,
        to_box=to_box,
        target_logprob_box_fn=target_logprob_box_fn,
        sample_mean_fn=sample_mean_fn,
        compute_metrics_fn=compute_metrics_fn,
        sample_reference_fn=sample_reference_fn,
        energy_w2_fn=energy_w2_fn,
        optimal_p_fn=optimal_p_fn,
        update_p_fn=update_p_fn,
        target_loggrad_latent_fn=target_loggrad_latent_fn,
        use_lgv=args.gbs_use_lgv,
        use_wandb=args.use_wandb,
        wandb_log_every=args.wandb_log_every,
        wandb_prefix=f"gbs/{loss_mode}",
        wandb_plot_ntraj=args.wandb_plot_ntraj,
    )
    if args.save_dim01_gif:
        _, _, hist, final_samples, snapshots = result
    else:
        _, _, hist, final_samples = result
        snapshots = []
    hist_arrays = {key: np.asarray(value, dtype=np.float64) for key, value in hist.items()}
    hist_arrays = _prepend_history_point(hist_arrays, initial_hist_values)
    return {
        "hist": hist_arrays,
        "final_samples": np.asarray(final_samples),
        "snapshots": snapshots,
        "function_evals": _prepend_scalar(
            np.arange(1, gbs_outer_updates + 1, dtype=np.float64) * float(args.gbs_buffer_size),
            0.0,
        ),
        "policy_update_steps": _prepend_scalar(
            _policy_update_step_values(gbs_outer_updates, args.p_update_freq),
            0.0,
        ),
    }


def save_unified_plot(
    algorithm_results: dict[str, dict[str, object]],
    uniform_baseline: dict[str, np.ndarray],
    output_path: Path,
    hide_initial_point: bool,
    title: str,
    x_axis: str,
    plot_uniform_baseline: bool,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    line_styles = {
        "algorithm": dict(linestyle="-", linewidth=2.4, marker=None, zorder=3.0, alpha=0.95),
        "Uniform baseline": dict(linestyle="-.", linewidth=3.0, marker=None, zorder=2.0, alpha=0.95),
    }

    metric_specs = [
        ("target/p", "Learned policy p", False),
        ("target/sinkhorn", "Sinkhorn Distance", hide_initial_point),
        # ("target/interatomic_w2", r"\mathcal{W}_2$", hide_initial_point),
        ("target/energy_w2", r"$E(\cdot)\,\mathcal{W}_2$", hide_initial_point),
    ]

    for ax, (metric_key, metric_title, mask_first) in zip(axes, metric_specs):
        plotted_series = []
        for algo_name, result in algorithm_results.items():
            spec = ALGORITHM_SPECS[algo_name]
            hist = result["hist"]
            x_values = _plot_x_values(result, x_axis)
            plotted_series.append(hist[metric_key])
            _plot_curve(
                ax,
                x_values,
                hist[metric_key],
                spec["label"],
                spec["color"],
                mask_first,
                **line_styles["algorithm"],
            )
        if plot_uniform_baseline and metric_key == "target/sinkhorn":
            plotted_series.append(uniform_baseline["uniform_baseline/sinkhorn"])
            _plot_curve(
                ax,
                _plot_x_values(next(iter(algorithm_results.values())), x_axis),
                uniform_baseline["uniform_baseline/sinkhorn"],
                "Uniform baseline",
                "black",
                mask_first,
                **line_styles["Uniform baseline"],
            )
        if plot_uniform_baseline and metric_key == "target/energy_w2":
            plotted_series.append(uniform_baseline["uniform_baseline/energy_w2"])
            _plot_curve(
                ax,
                _plot_x_values(next(iter(algorithm_results.values())), x_axis),
                uniform_baseline["uniform_baseline/energy_w2"],
                "Uniform baseline",
                "black",
                mask_first,
                **line_styles["Uniform baseline"],
            )
        _set_robust_ylim(ax, plotted_series)
        # ax.set_title(metric_title)
        ax.set_xlabel(_plot_x_label(x_axis), fontsize=20)
        ax.tick_params(axis="both", labelsize=15)
        ax.legend(framealpha=0.95)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Learned Policy Probability", fontsize=20)
    axes[1].set_ylabel("Sinkhorn Distance", fontsize=20)
    # axes[2].set_ylabel(r"$\mathcal{W}_2$", fontsize=20)
    axes[2].set_ylabel(r"$E(\cdot)\,\mathcal{W}_2$", fontsize=20)
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    # fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.as_posix(), dpi=160)
    plt.close(fig)


def save_lambda_plot(
    algorithm_results: dict[str, dict[str, object]],
    output_path: Path,
    x_axis: str,
) -> bool:
    plotted = False
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    for algo_name, result in algorithm_results.items():
        hist = result["hist"]
        if "train/tr_lv_lambda" not in hist:
            continue
        lam = np.asarray(hist["train/tr_lv_lambda"], dtype=np.float64)
        x_values = _plot_x_values(result, x_axis)
        finite = np.isfinite(lam)
        if not np.any(finite):
            continue
        spec = ALGORITHM_SPECS[algo_name]
        ax.plot(
            x_values[finite],
            lam[finite],
            label=spec["label"],
            color=spec["color"],
            linewidth=2.2,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel(_plot_x_label(x_axis), fontsize=14)
    ax.set_ylabel("trust-region lambda", fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.95)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.as_posix(), dpi=160)
    plt.close(fig)
    return True


def _loss_plot_bounds(values_list: list[np.ndarray]):
    finite_values = [
        np.asarray(values, dtype=np.float64)[np.isfinite(values)]
        for values in values_list
    ]
    finite_values = [values for values in finite_values if values.size > 0]
    if not finite_values:
        return None
    values = np.concatenate(finite_values)
    if values.size < 4:
        y_min = float(np.min(values))
        y_max = float(np.max(values))
    else:
        q_low, q_high = np.nanpercentile(values, [5.0, 95.0])
        core = values[(values >= q_low) & (values <= q_high)]
        if core.size == 0:
            core = values
        y_min = float(np.min(core))
        y_max = float(np.max(core))
    span = y_max - y_min
    scale = max(abs(y_min), abs(y_max), 1e-8)
    pad = max(0.08 * span, 0.02 * scale, 1e-12)
    lower = y_min - pad
    upper = y_max + pad
    if np.all(values >= 0.0):
        lower = max(0.0, lower)
    if upper <= lower:
        upper = lower + max(0.04 * scale, 1e-12)
    clip_lower = bool(np.min(values) < lower)
    clip_upper = bool(np.max(values) > upper)
    return lower, upper, clip_lower, clip_upper


def _draw_loss_break_marks(ax, *, top: bool, bottom: bool) -> None:
    d = 0.012
    kwargs = dict(transform=ax.transAxes, color="black", clip_on=False, linewidth=1.2)
    if top:
        ax.plot((-d, +d), (1.0 - d, 1.0 + d), **kwargs)
        ax.plot((0.025 - d, 0.025 + d), (1.0 - d, 1.0 + d), **kwargs)
    if bottom:
        ax.plot((-d, +d), (-d, +d), **kwargs)
        ax.plot((0.025 - d, 0.025 + d), (-d, +d), **kwargs)


def save_loss_plot(
    algorithm_results: dict[str, dict[str, object]],
    output_path: Path,
    x_axis: str,
) -> bool:
    plotted = False
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ylabel = "loss"
    loss_entries: list[tuple[str, np.ndarray, np.ndarray, dict[str, str]]] = []
    for algo_name, result in algorithm_results.items():
        hist = result["hist"]
        if "train/tr_lv_var" in hist:
            loss = np.asarray(hist["train/tr_lv_var"], dtype=np.float64)
            ylabel = "loss"
        elif "train/neg_elbo_var" in hist:
            loss = np.asarray(hist["train/neg_elbo_var"], dtype=np.float64)
            ylabel = "loss"
        elif "train/neg_elbo_mean" in hist:
            loss = np.asarray(hist["train/neg_elbo_mean"], dtype=np.float64)
            ylabel = "loss"
        else:
            continue
        x_values = _plot_x_values(result, x_axis)
        loss_entries.append((algo_name, x_values, loss, ALGORITHM_SPECS[algo_name]))

    clip_info = _loss_plot_bounds([entry[2] for entry in loss_entries])
    if clip_info is not None:
        y_lower, y_upper, has_lower_clip, has_upper_clip = clip_info
        ax.set_ylim(y_lower, y_upper)
        _draw_loss_break_marks(ax, top=has_upper_clip, bottom=has_lower_clip)

    for algo_name, x_values, loss, spec in loss_entries:
        finite = np.isfinite(loss)
        nan_mask = np.isnan(loss)
        if np.any(finite):
            y_plot = loss.copy()
            clipped_upper = np.zeros_like(finite, dtype=bool)
            clipped_lower = np.zeros_like(finite, dtype=bool)
            if clip_info is not None:
                y_plot = np.clip(y_plot, y_lower, y_upper)
                clipped_upper = finite & (loss > y_upper)
                clipped_lower = finite & (loss < y_lower)
            ax.plot(
                x_values[finite],
                y_plot[finite],
                label=spec["label"],
                color=spec["color"],
                linewidth=2.2,
            )
            if np.any(clipped_upper):
                ax.scatter(
                    x_values[clipped_upper],
                    np.full(np.count_nonzero(clipped_upper), y_upper),
                    color=spec["color"],
                    marker="^",
                    s=48,
                    edgecolors="black",
                    linewidths=0.4,
                    label=f"{spec['label']} clipped high",
                    zorder=5,
                    clip_on=False,
                )
            if np.any(clipped_lower):
                ax.scatter(
                    x_values[clipped_lower],
                    np.full(np.count_nonzero(clipped_lower), y_lower),
                    color=spec["color"],
                    marker="v",
                    s=48,
                    edgecolors="black",
                    linewidths=0.4,
                    label=f"{spec['label']} clipped low",
                    zorder=5,
                    clip_on=False,
                )
            plotted = True
        if np.any(nan_mask):
            ax.scatter(
                x_values[nan_mask],
                np.full(np.count_nonzero(nan_mask), 0.04),
                color="red",
                marker="x",
                s=42,
                linewidths=1.4,
                transform=ax.get_xaxis_transform(),
                label=f"{spec['label']} NaN flag",
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel(_plot_x_label(x_axis), fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if not np.any(np.isfinite(ax.get_ylim())) or ax.get_ylim()[0] == ax.get_ylim()[1]:
        ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.95)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.as_posix(), dpi=160)
    plt.close(fig)
    return True


def save_dim01_plot(
    algorithm_samples: dict[str, np.ndarray],
    target_params,
    algorithm_lams: dict[str, float],
    output_path: Path,
    target_version: str,
    algorithm_ps: dict[str, float],
    n_spatial_dim: int = 1,
    n_grid: int = 180,
    show_samples: bool = True,
) -> None:
    if not algorithm_samples:
        return
    if any(samples.shape[1] < 2 for samples in algorithm_samples.values()):
        return
    sample_dim = next(iter(algorithm_samples.values())).shape[1]

    if _is_fixed_target(target_version):
        target_logprob = get_fixed_target_setup(target_version, algorithm_lams[next(iter(algorithm_lams))])[0]
        low, high = get_fixed_target_setup(target_version, algorithm_lams[next(iter(algorithm_lams))])[1:3]
    elif _is_target_example(target_version):
        target_example = load_target_example(target_version, sample_dim, n_spatial_dim)
        target_logprob = target_example.log_prob
        low, high = target_example_bounds(target_version, sample_dim)
    else:
        target_logprob = load_target_utils(target_version).target_logprob
        low = jnp.zeros(2)
        high = jnp.ones(2)

    if target_params is not None and hasattr(target_params, "phi"):
        params2 = type(target_params)(
            c=target_params.c,
            a=target_params.a[:2],
            k=target_params.k[:2],
            phi=target_params.phi[:2],
        )
    elif target_params is not None:
        params2 = type(target_params)(
            c=target_params.c,
            a=target_params.a[:2],
            k=target_params.k[:2],
            mu0=target_params.mu0[:2],
            b=target_params.b[:2],
        )
    else:
        params2 = None
    x = jnp.linspace(float(low[0]), float(high[0]), n_grid)
    y = jnp.linspace(float(low[1]), float(high[1]), n_grid)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    grid = jnp.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)

    algo_names = list(algorithm_samples.keys())
    fig, axes = plt.subplots(1, len(algo_names), figsize=(5.0 * len(algo_names), 4.5))
    if len(algo_names) == 1:
        axes = [axes]
    contour = None
    for ax, algo_name in zip(axes, algo_names):
        samples = algorithm_samples[algo_name]
        lam = algorithm_lams[algo_name]
        policy_p = algorithm_ps[algo_name]
        title = ALGORITHM_SPECS[algo_name]["label"]
        if _is_fixed_target(target_version):
            logp = np.asarray(target_logprob(grid)).reshape(n_grid, n_grid)
        elif _is_target_example(target_version):
            logp = np.asarray(_target_example_dim01_logprob(target_example, grid, sample_dim)).reshape(n_grid, n_grid)
        else:
            logp = np.asarray(
                target_logprob(grid, lam, target_params=params2, policy_p=policy_p)
            ).reshape(n_grid, n_grid)
        logp = logp - np.max(logp)
        density = np.exp(logp)
        contour = ax.contourf(
            np.asarray(X),
            np.asarray(Y),
            density,
            levels=20,
            cmap="viridis",
        )
        if show_samples:
            ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.20, c="r", marker="x")
        ax.set_xlim(float(low[0]), float(high[0]))
        ax.set_ylim(float(low[1]), float(high[1]))
        ax.set_xlabel("dim 0")
        ax.set_ylabel("dim 1")
        ax.set_title(f"{title} | lambda={lam:.3f} | p={policy_p:.3f}")
    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="5%", pad=0.08)
    fig.colorbar(contour, cax=cax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.as_posix(), dpi=160)
    plt.close(fig)


def save_dim01_gif(
    algorithm_snapshots: dict[str, list[dict[str, object]]],
    target_params,
    beta: float,
    output_path: Path,
    target_version: str,
    n_spatial_dim: int = 1,
    n_grid: int = 180,
    fps: int = 4,
) -> bool:
    if imageio is None:
        raise ImportError(
            "imageio is required for --save_dim01_gif. Install it or disable GIF saving."
        )
    if not algorithm_snapshots:
        return False
    algo_names = [name for name, snapshots in algorithm_snapshots.items() if snapshots]
    if not algo_names:
        return False
    sample_dim = np.asarray(next(iter(algorithm_snapshots.values()))[0]["samples"]).shape[1]
    if _is_fixed_target(target_version):
        fixed_setup = get_fixed_target_setup(target_version, beta)
        target_logprob = fixed_setup[0]
        low, high = fixed_setup[1], fixed_setup[2]
    elif _is_target_example(target_version):
        target_example = load_target_example(target_version, sample_dim, n_spatial_dim)
        target_logprob = target_example.log_prob
        low, high = target_example_bounds(
            target_version, sample_dim
        )
    else:
        target_logprob = load_target_utils(target_version).target_logprob
        low = jnp.zeros(2)
        high = jnp.ones(2)
    if target_params is not None and hasattr(target_params, "phi"):
        params2 = type(target_params)(
            c=target_params.c,
            a=target_params.a[:2],
            k=target_params.k[:2],
            phi=target_params.phi[:2],
        )
    elif target_params is not None:
        params2 = type(target_params)(
            c=target_params.c,
            a=target_params.a[:2],
            k=target_params.k[:2],
            mu0=target_params.mu0[:2],
            b=target_params.b[:2],
        )
    else:
        params2 = None
    x = jnp.linspace(float(low[0]), float(high[0]), n_grid)
    y = jnp.linspace(float(low[1]), float(high[1]), n_grid)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    grid = jnp.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)

    num_frames = min(len(algorithm_snapshots[name]) for name in algo_names)
    rendered = []
    for idx in range(num_frames):
        fig, axes = plt.subplots(1, len(algo_names), figsize=(5.0 * len(algo_names), 4.5))
        if len(algo_names) == 1:
            axes = [axes]
        contour = None
        for ax, algo_name in zip(axes, algo_names):
            frame = algorithm_snapshots[algo_name][idx]
            title = ALGORITHM_SPECS[algo_name]["label"]
            samples = np.asarray(frame["samples"])
            p = float(frame["p"])
            lam = beta * p
            if _is_fixed_target(target_version):
                logp = np.asarray(target_logprob(grid)).reshape(n_grid, n_grid)
            elif _is_target_example(target_version):
                logp = np.asarray(_target_example_dim01_logprob(target_example, grid, sample_dim)).reshape(n_grid, n_grid)
            else:
                logp = np.asarray(
                    target_logprob(grid, lam, target_params=params2, policy_p=p)
                ).reshape(n_grid, n_grid)
            logp = logp - np.max(logp)
            density = np.exp(logp)
            contour = ax.contourf(np.asarray(X), np.asarray(Y), density, levels=20, cmap="viridis")
            ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.20, c="r", marker="x")
            ax.set_xlim(float(low[0]), float(high[0]))
            ax.set_ylim(float(low[1]), float(high[1]))
            ax.set_xlabel("dim 0")
            ax.set_ylabel("dim 1")
            ax.set_title(f"{title} | iter={int(frame['iter'])} | lambda={lam:.3f} | p={p:.3f}")
        divider = make_axes_locatable(axes[-1])
        cax = divider.append_axes("right", size="5%", pad=0.08)
        fig.colorbar(contour, cax=cax)
        fig.tight_layout()
        fig.canvas.draw()
        rendered.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path.as_posix(), rendered, fps=fps)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GBS and GMMVI target experiments and compare learnable p, Sinkhorn, and Energy W2."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument(
        "--target-version",
        choices=["target1", "target2", "target3", "target_A", "target_B", "target_C", *target_example_names()],
        default="target_B",
        help="Target selector: target1/2/3, target_A/B/C, or a supported target_examples name.",
    )
    parser.add_argument("--beta", type=float, default=-200.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--safe_q", type=float, default=1.0)
    parser.add_argument("--initial_p", type=float, default=0.9)
    parser.add_argument("--p_update_freq", type=int, default=10)
    parser.add_argument("--p_ema_alpha", type=float, default=0.97)
    parser.add_argument("--p_jump_prob", type=float, default=0.0)
    parser.add_argument("--metric_num_bins", type=int, default=128)
    parser.add_argument(
        "--metric_eval_samples",
        type=int,
        default=4096,
        help="Shared number of samples used for metric estimates and policy-update statistics.",
    )
    parser.add_argument("--sinkhorn_num_samples", type=int, default=4096)
    parser.add_argument("--n_particles", type=int, default=None)
    parser.add_argument("--n_spatial_dim", type=int, default=1)

    parser.add_argument(
        "--gbs_function_evaluations",
        type=int,
        default=2**24,
    )
    parser.add_argument(
        "--gbs_buffer_size",
        "--gbs_batch_size",
        dest="gbs_buffer_size",
        type=int,
        default=2**14,
    )
    parser.add_argument("--gbs_num_steps", type=int, default=50)
    parser.add_argument("--gbs_lr", type=float, default=5e-4)
    parser.add_argument("--gbs_init_std", type=float, default=0.5)
    parser.add_argument(
        "--gbs_loss_mode",
        "--loss_mode",
        dest="gbs_loss_mode",
        choices=[
            "trust_region_lv",
            "tr_lv",
            "tr_dds_lv",
            "lv",
            "time_reversal_lv",
            "dds",
            "dds_lv",
            "dds_exp_lv",
            "dis",
            "dis_lv",
        ],
        default="tr_dds_lv",
    )
    parser.add_argument("--gbs_model_type", choices=["pisgrad", "potential"], default="pisgrad")
    parser.add_argument("--gbs_model_num_layers", type=int, default=3)
    parser.add_argument("--gbs_model_num_hid", type=int, default=256)
    parser.add_argument("--gbs_scale_diff", type=float, default=2.)
    parser.add_argument("--gbs_use_lgv", action="store_true")
    parser.add_argument("--no_gbs_use_lgv", dest="gbs_use_lgv", action="store_false")
    parser.set_defaults(gbs_use_lgv=False)
    parser.add_argument("--gbs_trust_region_bound", type=float, default=0.1)
    parser.add_argument("--gbs_trust_region_lambda_max", type=float, default=50.0)
    parser.add_argument("--gbs_trust_region_lambda_grid_size", type=int, default=128)
    parser.add_argument("--gbs_minibatch_size", type=int, default=2**11)
    parser.add_argument(
        "--gbs_minibatch_steps",
        "--gbs_buffer_updates",
        dest="gbs_minibatch_steps",
        type=int,
        default=8,
    )
    parser.add_argument("--use_tanh_bijection", action="store_true")
    parser.add_argument("--no_use_tanh_bijection", dest="use_tanh_bijection", action="store_false")
    parser.set_defaults(use_tanh_bijection=True)

    parser.add_argument("--gmmvi_num_envs", type=int, default=2**14)
    parser.add_argument(
        "--gmmvi_function_evaluations",
        dest="gmmvi_function_evaluations",
        type=int,
        default=2**24,
    )
    parser.add_argument("--gmmvi_batch_size", type=int, default=2**14)
    parser.add_argument("--gmmvi_eval_samples", type=int, default=4096)
    parser.add_argument("--gmmvi_prior_scale", type=float, default=0.5)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=list(ALGORITHM_SPECS.keys()),
        default=["gmmvi", "dis", "dis_lv", "dds", "dds_lv", "tr_dds_lv"],
        help="Algorithms to compare.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("compare_GBS_GMMVI"),
        help="Base directory for experiment outputs. Runs are saved under target_dim_beta/seed_<seed>/.",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Experiment log path. Defaults to a timestamped experiment_*.log inside the seed-specific run directory.",
    )
    parser.add_argument(
        "--x_axis",
        "--x-axis",
        type=_normalize_x_axis_mode,
        choices=("function_evaluations", "policy_update_steps"),
        default="function_evaluations",
        help=(
            "X-axis for comparison, loss, and lambda plots. "
            "Use function_evaluations/function_evals or policy_update_steps/policy_updates."
        ),
    )
    parser.add_argument("--hide_initial_point", action="store_true")
    parser.add_argument("--plot_uniform_baseline", action="store_true")
    parser.add_argument("--no_plot_uniform_baseline", dest="plot_uniform_baseline", action="store_false")
    parser.set_defaults(plot_uniform_baseline=False)
    parser.add_argument("--max_eval_points", type=int, default=50)
    parser.add_argument("--save_dim01_gif", action="store_true")
    parser.add_argument("--no_save_dim01_gif", dest="save_dim01_gif", action="store_false")
    parser.set_defaults(save_dim01_gif=True)
    parser.add_argument("--gif_num_frames", type=int, default=100)
    parser.add_argument("--gif_fps", type=int, default=4)
    parser.add_argument(
        "--dim01_sample_size",
        "--gif_sample_size",
        dest="dim01_sample_size",
        type=int,
        default=2**11,
        help="Shared sample count for final dim0/dim1 scatter plots and dim0/dim1 GIF frames.",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.set_defaults(use_wandb=True)
    parser.add_argument("--wandb_project", type=str, default="compare_gbs_gmmvi")
    parser.add_argument("--wandb_entity", type=str, default="tjrcjf410-seoul-national-university")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_log_every", type=int, default=10)
    parser.add_argument("--wandb_plot_ntraj", type=int, default=50)
    add_target_cli_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metric_eval_samples <= 0:
        raise ValueError(f"metric_eval_samples must be positive, got {args.metric_eval_samples}")
    if args.dim01_sample_size <= 0:
        raise ValueError(f"dim01_sample_size must be positive, got {args.dim01_sample_size}")
    args.experiment_beta = float(args.beta)
    args.beta_times_dim = _target_beta_for_dim(args.experiment_beta, args.dim)
    target_beta = _args_target_beta(args)
    has_gbs_algorithm = any(ALGORITHM_SPECS[name]["kind"] == "gbs" for name in args.algorithms)
    if has_gbs_algorithm and args.metric_eval_samples > args.gbs_buffer_size:
        raise ValueError(
            "metric_eval_samples must be <= gbs_buffer_size when comparing GBS algorithms, "
            f"got {args.metric_eval_samples=} and {args.gbs_buffer_size=}"
        )
    _, resolved_family = _resolve_target_config(args.target_version)
    gbs_outer_updates = compute_outer_iterations(
        args.gbs_function_evaluations, args.gbs_buffer_size
    )
    gmmvi_steps = compute_gmmvi_iterations(
        args.gmmvi_function_evaluations, args.gmmvi_num_envs
    )
    n_particles = _safe_n_particles(args.dim, args.n_particles, args.n_spatial_dim)
    args.n_particles = n_particles
    resolved_target_label = _target_label(args.target_version)
    target_dir_name = resolved_target_label + f"_dim{args.dim}_beta{args.experiment_beta}"
    target_output_dir = args.output_dir / target_dir_name
    run_output_dir = target_output_dir / f"seed_{args.seed}"
    target_output_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "target_family", "target_B") != "target_B" and _target_label(args.target_version) != "target_C":
        raise ValueError("--target-family is only needed for unified target_C / version 3 in this script.")
    args.target_family = resolved_family
    args.target_output_dir = target_output_dir
    args.run_output_dir = run_output_dir
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
        )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (
        args.log_file
        if args.log_file is not None
        else run_output_dir / f"experiment_seed{args.seed}_{timestamp}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", buffering=1)
    atexit.register(log_handle.close)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(sys.stdout, log_handle)
    sys.stderr = TeeStream(sys.stderr, log_handle)
    try:
        print(f"Saving experiment log to: {log_path}")
        print(f"Experiment arguments: {vars(args)}")
        print(f"Seed: {args.seed}")
        print(f"Target output root: {target_output_dir}")
        print(f"Seed run output dir: {run_output_dir}")
        target_params = build_compare_target_params(args, args.dim)
        if target_params is not None:
            print(f"Target effective modes: {target_effective_num_modes(target_params)}")
        algorithm_results: dict[str, dict[str, object]] = {}
        for algo_name in args.algorithms:
            spec = ALGORITHM_SPECS[algo_name]
            if spec["kind"] == "gmmvi":
                algorithm_results[algo_name] = gmmvi_training(args)
            else:
                algorithm_results[algo_name] = gbs_training(args, spec["loss_mode"])

        first_hist = algorithm_results[args.algorithms[0]]["hist"]
        uniform_baseline = compute_uniform_metric_baseline(
            first_hist["target/p"],
            beta=target_beta,
            dim=args.dim,
            sinkhorn_num_samples=min(args.sinkhorn_num_samples, args.metric_eval_samples),
            target_params=target_params,
            target_version=args.target_version,
            seed=args.seed + 202,
            n_spatial_dim=args.n_spatial_dim,
            eval_mask=np.isfinite(first_hist["target/sinkhorn"]),
        )
        raw_metrics_dir = run_output_dir / "raw_metrics"
        save_raw_metric_data(algorithm_results, uniform_baseline, raw_metrics_dir, seed=args.seed)

        output_path = run_output_dir / "comparison.png"
        title = (
            f"Algorithm comparison | dim={args.dim}, "
            f"gbs_updates={gbs_outer_updates}, gmmvi_steps={gmmvi_steps}, "
            f"experiment_beta={args.experiment_beta:g}, beta_times_dim={target_beta:g}, tau={args.tau:g}"
        )
        save_unified_plot(
            algorithm_results,
            uniform_baseline,
            output_path=output_path,
            hide_initial_point=args.hide_initial_point,
            title=title,
            x_axis=args.x_axis,
            plot_uniform_baseline=args.plot_uniform_baseline,
        )
        loss_output_path = run_output_dir / "loss.png"
        loss_saved = save_loss_plot(
            algorithm_results,
            output_path=loss_output_path,
            x_axis=args.x_axis,
        )
        lambda_output_path = run_output_dir / "lambda.png"
        lambda_saved = save_lambda_plot(
            algorithm_results,
            output_path=lambda_output_path,
            x_axis=args.x_axis,
        )
        if args.dim >= 2:
            dim01_output_path = run_output_dir / "comparison_dim01.png"
            dim01_clean_output_path = run_output_dir / "comparison_dim01_clean.png"
            dim01_samples = {
                name: _limited_samples(result["final_samples"], args.dim01_sample_size)
                for name, result in algorithm_results.items()
            }
            dim01_lams = {
                name: float(target_beta * result["hist"]["target/p"][-1])
                for name, result in algorithm_results.items()
            }
            dim01_policy_ps = {
                name: float(result["hist"]["target/p"][-1])
                for name, result in algorithm_results.items()
            }
            save_dim01_plot(
                dim01_samples,
                target_params,
                dim01_lams,
                output_path=dim01_output_path,
                target_version=args.target_version,
                algorithm_ps=dim01_policy_ps,
                n_spatial_dim=args.n_spatial_dim,
            )
            save_dim01_plot(
                dim01_samples,
                target_params,
                dim01_lams,
                output_path=dim01_clean_output_path,
                target_version=args.target_version,
                algorithm_ps=dim01_policy_ps,
                n_spatial_dim=args.n_spatial_dim,
                show_samples=False,
            )
            print(f"Saved dim0/dim1 comparison plot to: {dim01_output_path}")
            print(f"Saved clean dim0/dim1 heatmap to: {dim01_clean_output_path}")
            if args.save_dim01_gif:
                dim01_gif_path = run_output_dir / "comparison_dim01.gif"
                gif_saved = save_dim01_gif(
                    {name: result.get("snapshots", []) for name, result in algorithm_results.items()},
                    target_params,
                    beta=target_beta,
                    output_path=dim01_gif_path,
                    target_version=args.target_version,
                    n_spatial_dim=args.n_spatial_dim,
                    fps=args.gif_fps,
                )
                if gif_saved:
                    print(f"Saved dim0/dim1 GIF to: {dim01_gif_path}")
                else:
                    print("Skipped dim0/dim1 GIF: missing snapshots for one or more algorithms.")

        print(f"Saved unified comparison plot to: {output_path}")
        if loss_saved:
            print(f"Saved loss plot to: {loss_output_path}")
        if lambda_saved:
            print(f"Saved lambda plot to: {lambda_output_path}")
        print(f"Saved raw metric data to: {raw_metrics_dir}")
        print(f"Target label: {resolved_target_label}")
        print(f"Target family: {args.target_family}")
        print(f"Experiment beta: {args.experiment_beta:g}")
        print(f"Dimension-scaled beta: {target_beta:g}")
        print(f"Metric evaluation samples: {args.metric_eval_samples}")
        print(f"dim0/dim1 plot samples: {args.dim01_sample_size}")
        actual_use_tanh = bool(args.use_tanh_bijection) and (not _is_target_example(args.target_version))
        print(f"GBS uses tanh bijection: {actual_use_tanh}")
        print(f"Algorithms: {', '.join(ALGORITHM_SPECS[name]['label'] for name in args.algorithms)}")
        for algo_name, result in algorithm_results.items():
            hist = result["hist"]
            label = ALGORITHM_SPECS[algo_name]["label"]
            print(f"{label} initial Wasserstein: {float(hist['target/wasserstein'][0]):.6f}")
            print(f"{label} initial Sinkhorn: {float(hist['target/sinkhorn'][0]):.6f}")
            print(f"{label} final p: {_last_finite(hist['target/p']):.6f}")
            print(f"{label} final Sinkhorn: {_last_finite(hist['target/sinkhorn']):.6f}")
            print(f"{label} final Energy W2: {_last_finite(hist['target/energy_w2']):.6f}")
        print(f"Uniform-baseline initial Sinkhorn: {float(uniform_baseline['uniform_baseline/sinkhorn'][0]):.6f}")
        print(f"Uniform-baseline initial Energy W2: {float(uniform_baseline['uniform_baseline/energy_w2'][0]):.6f}")
        print(f"Uniform-baseline Sinkhorn: {_last_finite(uniform_baseline['uniform_baseline/sinkhorn']):.6f}")
        print(f"Uniform-baseline Energy W2: {_last_finite(uniform_baseline['uniform_baseline/energy_w2']):.6f}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if args.use_wandb:
            wandb.finish()
        log_handle.close()
if __name__ == "__main__":
    main()
