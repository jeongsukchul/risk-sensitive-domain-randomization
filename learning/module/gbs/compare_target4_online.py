from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import functools
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import trange

from learning.module.gbs.gbs_test_toy import build_run_tag as build_gbs_run_tag
from learning.module.gbs.gmmvi_test_toy import build_run_tag as build_gmmvi_run_tag
from learning.module.gbs.sinkhorn_metrics import (
    energy_wasserstein_1d,
    effective_sample_size_from_log_weights,
    interatomic_wasserstein_1d,
    sinkhorn_distance,
)
from learning.module.gbs.target4_family import (
    Target4HarmonicParams,
    add_target4_cli_args,
    build_target4_params_from_args,
    get_target4_harmonic_params,
    make_default_target4_harmonic_params,
    target4_energy_values,
)
from learning.module.gmmvi.network import create_gmm_network_and_state
from learning.module.gmmvi.network import GMMTrainingState


def _resolve_target4_config(version: str) -> tuple[str, str, str]:
    if version == "3":
        return "2", "target4_3", "versioned"
    if version == "uniform":
        return "2", "target4", "uniform"
    return version, "target4", "versioned"


def load_target4_utils(version: str):
    utils_version, _, _ = _resolve_target4_config(version)
    module_name = f"learning.module.gbs.target4_{utils_version}_notebook_utils"
    return importlib.import_module(module_name)


def build_compare_target_params(args: argparse.Namespace, dim: int):
    _, _, target_mode = _resolve_target4_config(args.target4_version)
    if target_mode != "uniform":
        return build_target4_params_from_args(args, dim)

    default = make_default_target4_harmonic_params(dim)
    return Target4HarmonicParams(
        c=default.c,
        a=jnp.zeros_like(default.a),
        k=default.k,
        phi=jnp.zeros_like(default.phi),
    )


def build_gmmvi_fns_dynamic(gmm_network, num_envs: int, target_params, target4_logprob_fn):
    del num_envs

    def _target_scalar_logprob(sample: jax.Array, lam: jax.Array, policy_p: jax.Array) -> jax.Array:
        return target4_logprob_fn(sample[None, :], lam, target_params=target_params, policy_p=policy_p).reshape(())

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


def _safe_n_particles(dim: int, n_particles: int | None, n_spatial_dim: int) -> int:
    if n_particles is not None:
        return n_particles
    if dim % n_spatial_dim != 0:
        raise ValueError(f"dim={dim} must be divisible by n_spatial_dim={n_spatial_dim}")
    return dim // n_spatial_dim


def _plot_curve(
    ax,
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
    x = np.arange(curve.size)
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


def compute_uniform_baseline_curves(
    p_values: np.ndarray,
    *,
    beta: float,
    dim: int,
    sinkhorn_num_samples: int,
    target_params,
    target4_version: str,
    seed: int,
    eval_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    target4_utils = load_target4_utils(target4_version)
    sample_truncated_exponential = target4_utils.sample_truncated_exponential

    sinkhorn_vals = []
    energy_w2_vals = []
    key = jax.random.PRNGKey(seed)
    n_samples = max(1, int(sinkhorn_num_samples))

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
def run_gmmvi_target4_online(args: argparse.Namespace) -> dict[str, np.ndarray]:
    target4_utils = load_target4_utils(args.target4_version)
    compute_target4_metrics = target4_utils.compute_target4_metrics
    optimal_p_from_target_mean = target4_utils.optimal_p_from_target_mean
    sample_truncated_exponential = target4_utils.sample_truncated_exponential
    should_compute_interatomic_w2 = target4_utils.should_compute_interatomic_w2
    target4_logprob = target4_utils.target4_logprob
    update_p_with_ema_and_jump = target4_utils.update_p_with_ema_and_jump
    build_eval_iters = target4_utils.build_eval_iters

    dim = args.dim
    low = jnp.zeros(dim)
    high = jnp.ones(dim)
    n_particles = _safe_n_particles(dim, args.n_particles, args.n_spatial_dim)

    key = jax.random.PRNGKey(args.seed + 1)
    key, k_init, k_p0 = jax.random.split(key, 3)
    state, gmm_network = create_gmm_network_and_state(
        dim,
        args.gmmvi_num_envs,
        args.gmmvi_batch_size,
        k_init,
        prior_scale=args.gmmvi_prior_scale,
        bound_info=(low, high),
    )
    target_params = build_compare_target_params(args, dim)
    gather_samples, train_iter, sample_model, model_log_density = build_gmmvi_fns_dynamic(
        gmm_network, args.gmmvi_num_envs, target_params, target4_logprob
    )

    if args.initial_p is None:
        p = float(jax.random.uniform(k_p0, minval=0.0, maxval=1.0))
    else:
        p = float(np.clip(args.initial_p, 0.0, 1.0))

    hist: dict[str, list[float]] = {
        "target4/p": [],
        "target4/lambda": [],
        "target4/sample_mean": [],
        "target4/forward_kl": [],
        "target4/reverse_kl": [],
        "target4/wasserstein": [],
        "target4/sinkhorn": [],
        "target4/ess": [],
        "target4/energy_w2": [],
        "target4/interatomic_w2": [],
        "target4/target_mean": [],
        "target4/optimal_p": [],
        "target4/p_updated": [],
        "target4/p_jumped": [],
        "target4/p_base": [],
        "target4/p_ema": [],
        "model/num_components": [],
    }
    metric_eval_iters = build_eval_iters(args.iters, args.max_eval_points)
    snapshot_iters = set(_build_snapshot_iters(args.iters, args.gif_num_frames) if args.save_dim01_gif else [])
    snapshots: list[dict[str, object]] = []

    current_lambda = args.beta * p
    for _ in range(max(args.gmmvi_batch_size // args.gmmvi_num_envs, 1)):
        key, subkey = jax.random.split(key)
        state = gather_samples(state, subkey, jnp.asarray(current_lambda), jnp.asarray(p, dtype=jnp.float32))

    for step in trange(args.iters, desc="GMMVI", leave=False):
        current_lambda = args.beta * p

        key, subkey = jax.random.split(key)
        state = train_iter(state, subkey, jnp.asarray(current_lambda), jnp.asarray(p, dtype=jnp.float32))

        key, k_eval, k_metric = jax.random.split(key, 3)
        samples = np.asarray(sample_model(state, k_eval, args.gmmvi_eval_samples))
        _ = model_log_density(state, jnp.asarray(samples))

        sample_mean = float(
            np.mean(np.asarray(target4_energy_values(jnp.asarray(samples), target_params, policy_p=p)))
        )
        if step in metric_eval_iters:
            forward_kl, reverse_kl, wasserstein = compute_target4_metrics(
                samples,
                current_lambda,
                target_params=target_params,
                num_bins=args.metric_num_bins,
                key=k_metric,
                policy_p=p,
            )
            key, k_sink = jax.random.split(key)
            samples_jax = jnp.asarray(samples)
            sinkhorn_target = sample_truncated_exponential(
                k_sink,
                current_lambda,
                samples.shape,
                target_params=target_params,
                policy_p=p,
            )
            n_sink = min(args.sinkhorn_num_samples, samples.shape[0])
            sinkhorn = sinkhorn_distance(samples_jax[:n_sink], sinkhorn_target[:n_sink])
            ess = effective_sample_size_from_log_weights(
                target4_logprob(samples_jax, current_lambda, target_params=target_params, policy_p=p)
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
        optimal_p, target_mean = optimal_p_from_target_mean(
            current_lambda,
            args.tau,
            args.safe_q,
            target_params=target_params,
        )

        hist["target4/p"].append(float(p))
        hist["target4/lambda"].append(float(current_lambda))
        hist["target4/sample_mean"].append(sample_mean)
        hist["target4/forward_kl"].append(float(forward_kl))
        hist["target4/reverse_kl"].append(float(reverse_kl))
        hist["target4/wasserstein"].append(float(wasserstein))
        hist["target4/sinkhorn"].append(float(sinkhorn))
        hist["target4/ess"].append(float(ess))
        hist["target4/energy_w2"].append(energy_w2)
        hist["target4/interatomic_w2"].append(interatomic_w2)
        hist["target4/target_mean"].append(float(target_mean))
        hist["target4/optimal_p"].append(float(optimal_p))
        hist["model/num_components"].append(int(state.model_state.gmm_state.num_components))

        should_update_p = args.p_update_freq > 0 and ((step + 1) % args.p_update_freq == 0)
        hist["target4/p_updated"].append(float(should_update_p))
        hist["target4/p_jumped"].append(0.0)
        hist["target4/p_base"].append(float(jax.nn.sigmoid(args.tau * (sample_mean - args.safe_q))))
        hist["target4/p_ema"].append(float(p))

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
            hist["target4/p"][-1] = float(p)
            hist["target4/p_jumped"][-1] = float(jumped)
            hist["target4/p_base"][-1] = float(base_p)
            hist["target4/p_ema"][-1] = float(ema_p)

        if step in snapshot_iters:
            key, k_snapshot = jax.random.split(key)
            snapshot_samples = np.asarray(sample_model(state, k_snapshot, args.gif_sample_size))
            snapshots.append(
                {
                    "iter": int(step),
                    "p": float(p),
                    "samples": snapshot_samples,
                }
            )

    final_samples = np.asarray(sample_model(state, key, 2**12))
    return {
        "hist": {key: np.asarray(value, dtype=np.float64) for key, value in hist.items()},
        "final_samples": final_samples,
        "snapshots": snapshots,
    }


def save_unified_plot(
    gbs_hist: dict[str, np.ndarray],
    gmmvi_hist: dict[str, np.ndarray],
    uniform_baseline: dict[str, np.ndarray],
    output_path: Path,
    hide_initial_point: bool,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"DIS-LV": "tab:blue", "GMMVI": "tab:orange"}
    line_styles = {
        "GMMVI": dict(linestyle="-", linewidth=3.0, marker=None, zorder=4.0, alpha=0.95),
        "DIS-LV": dict(linestyle="--", linewidth=2.6, marker=None, zorder=3.0, alpha=0.9),
        "Uniform baseline": dict(
            linestyle="-.",
            linewidth=3.0,
            marker=None,
            zorder=2.0,
            alpha=0.95,
        ),
    }

    metric_specs = [
        ("target4/p", "Learned policy p", False),
        ("target4/sinkhorn", "Sinkhorn Distance", hide_initial_point),
        # ("target4/interatomic_w2", r"\mathcal{W}_2$", hide_initial_point),
        ("target4/energy_w2", r"$E(\cdot)\,\mathcal{W}_2$", hide_initial_point),
    ]

    for ax, (metric_key, metric_title, mask_first) in zip(axes, metric_specs):
        plotted_series = [gmmvi_hist[metric_key], gbs_hist[metric_key]]
        _plot_curve(ax, gmmvi_hist[metric_key], "GMMVI", colors["GMMVI"], mask_first, **line_styles["GMMVI"])
        _plot_curve(ax, gbs_hist[metric_key], "DIS-LV", colors["DIS-LV"], mask_first, **line_styles["DIS-LV"])
        if metric_key == "target4/sinkhorn":
            plotted_series.append(uniform_baseline["uniform_baseline/sinkhorn"])
            _plot_curve(
                ax,
                uniform_baseline["uniform_baseline/sinkhorn"],
                "Uniform baseline",
                "black",
                mask_first,
                **line_styles["Uniform baseline"],
            )
        if metric_key == "target4/energy_w2":
            plotted_series.append(uniform_baseline["uniform_baseline/energy_w2"])
            _plot_curve(
                ax,
                uniform_baseline["uniform_baseline/energy_w2"],
                "Uniform baseline",
                "black",
                mask_first,
                **line_styles["Uniform baseline"],
            )
        _set_robust_ylim(ax, plotted_series)
        # ax.set_title(metric_title)
        ax.set_xlabel("iteration", fontsize=20)
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


def save_dim01_plot(
    gbs_samples: np.ndarray,
    gmmvi_samples: np.ndarray,
    target_params,
    lam_gbs: float,
    lam_gmmvi: float,
    output_path: Path,
    target4_version: str,
    policy_p_gbs: float,
    policy_p_gmmvi: float,
    n_grid: int = 180,
) -> None:
    if gbs_samples.shape[1] < 2 or gmmvi_samples.shape[1] < 2:
        return

    target4_logprob = load_target4_utils(target4_version).target4_logprob

    if hasattr(target_params, "phi"):
        params2 = type(target_params)(
            c=target_params.c,
            a=target_params.a[:2],
            k=target_params.k[:2],
            phi=target_params.phi[:2],
        )
    else:
        params2 = type(target_params)(
            c=target_params.c,
            a=target_params.a[:2],
            k=target_params.k[:2],
            mu0=target_params.mu0[:2],
            b=target_params.b[:2],
        )
    x = jnp.linspace(0.0, 1.0, n_grid)
    y = jnp.linspace(0.0, 1.0, n_grid)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    grid = jnp.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    panels = [
        (axes[0], gbs_samples, lam_gbs, "GBS"),
        (axes[1], gmmvi_samples, lam_gmmvi, "GMMVI"),
    ]
    contour = None
    for ax, samples, lam, title in panels:
        policy_p = policy_p_gbs if title == "GBS" else policy_p_gmmvi
        logp = np.asarray(
            target4_logprob(grid, lam, target_params=params2, policy_p=policy_p)
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
        ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.20, c="r", marker="x")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("dim 0")
        ax.set_ylabel("dim 1")
        ax.set_title(f"{title} | lambda={lam:.3f} | p={policy_p:.3f}")
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.08)
    fig.colorbar(contour, cax=cax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.as_posix(), dpi=160)
    plt.close(fig)
def save_dim01_gif(
    gbs_snapshots: list[dict[str, object]],
    gmmvi_snapshots: list[dict[str, object]],
    target_params,
    beta: float,
    output_path: Path,
    target4_version: str,
    n_grid: int = 180,
    fps: int = 4,
) -> bool:
    if not gbs_snapshots or not gmmvi_snapshots:
        return False
    target4_logprob = load_target4_utils(target4_version).target4_logprob
    if hasattr(target_params, "phi"):
        params2 = type(target_params)(
            c=target_params.c,
            a=target_params.a[:2],
            k=target_params.k[:2],
            phi=target_params.phi[:2],
        )
    else:
        params2 = type(target_params)(
            c=target_params.c,
            a=target_params.a[:2],
            k=target_params.k[:2],
            mu0=target_params.mu0[:2],
            b=target_params.b[:2],
        )
    x = jnp.linspace(0.0, 1.0, n_grid)
    y = jnp.linspace(0.0, 1.0, n_grid)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    grid = jnp.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)

    num_frames = min(len(gbs_snapshots), len(gmmvi_snapshots))
    rendered = []
    for idx in range(num_frames):
        gbs_frame = gbs_snapshots[idx]
        gmmvi_frame = gmmvi_snapshots[idx]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        panels = [
            ("GBS", axes[0], gbs_frame),
            ("GMMVI", axes[1], gmmvi_frame),
        ]
        contour = None
        for title, ax, frame in panels:
            samples = np.asarray(frame["samples"])
            p = float(frame["p"])
            lam = beta * p
            logp = np.asarray(
                target4_logprob(grid, lam, target_params=params2, policy_p=p)
            ).reshape(n_grid, n_grid)
            logp = logp - np.max(logp)
            density = np.exp(logp)
            contour = ax.contourf(np.asarray(X), np.asarray(Y), density, levels=20, cmap="viridis")
            ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.20, c="r", marker="x")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("dim 0")
            ax.set_ylabel("dim 1")
            ax.set_title(f"{title} | iter={int(frame['iter'])} | lambda={lam:.3f} | p={p:.3f}")
        divider = make_axes_locatable(axes[1])
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
        description="Run GBS and GMMVI target4 experiments online and compare learnable p, Sinkhorn, and Energy W2."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument(
        "--target4-version",
        choices=["1", "2", "3", "uniform"],
        default="2",
        help="Unified target selector: 1=target4_1, 2=target4_2, 3=target4_3, uniform=uniform on [0,1]^d.",
    )
    parser.add_argument("--beta", type=float, default=-200.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--safe_q", type=float, default=1.0)
    parser.add_argument("--initial_p", type=float, default=0.9)
    parser.add_argument("--p_update_freq", type=int, default=10)
    parser.add_argument("--p_ema_alpha", type=float, default=0.99)
    parser.add_argument("--p_jump_prob", type=float, default=0.0)
    parser.add_argument("--metric_num_bins", type=int, default=128)
    parser.add_argument("--sinkhorn_num_samples", type=int, default=1024)
    parser.add_argument("--n_particles", type=int, default=None)
    parser.add_argument("--n_spatial_dim", type=int, default=1)

    parser.add_argument("--gbs_batch_size", type=int, default=1024)
    parser.add_argument("--gbs_num_steps", type=int, default=100)
    parser.add_argument("--gbs_lr", type=float, default=1e-3)
    parser.add_argument("--gbs_init_std", type=float, default=0.5)
    parser.add_argument("--gbs_loss_mode", choices=["tr_lv", "tr_lv_subtraj", "dis"], default="tr_lv")
    parser.add_argument("--gbs_model_type", choices=["pisgrad", "potential"], default="pisgrad")
    parser.add_argument("--gbs_model_num_layers", type=int, default=2)
    parser.add_argument("--gbs_model_num_hid", type=int, default=64)
    parser.add_argument("--use_tanh_bijection", action="store_true")
    parser.add_argument("--no_use_tanh_bijection", dest="use_tanh_bijection", action="store_false")
    parser.set_defaults(use_tanh_bijection=True)

    parser.add_argument("--gmmvi_num_envs", type=int, default=1024)
    parser.add_argument("--gmmvi_batch_size", type=int, default=4096)
    parser.add_argument("--gmmvi_eval_samples", type=int, default=4096)
    parser.add_argument("--gmmvi_prior_scale", type=float, default=0.5)

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("compare_GBS_GMMVI"),
    )
    parser.add_argument("--hide_initial_point", action="store_true")
    parser.add_argument("--max_eval_points", type=int, default=50)
    parser.add_argument("--save_dim01_gif", default=True, action="store_true")
    parser.add_argument("--gif_num_frames", type=int, default=24)
    parser.add_argument("--gif_fps", type=int, default=4)
    parser.add_argument("--gif_sample_size", type=int, default=2**14)
    add_target4_cli_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, resolved_family, target_mode = _resolve_target4_config(args.target4_version)
    if getattr(args, "target4_family", "target4") != "target4" and args.target4_version != "3":
        raise ValueError("--target4-family is only needed for unified version 3 in this script.")
    args.target4_family = resolved_family
    target4_utils = load_target4_utils(args.target4_version)
    run_gbs_toy_target4 = target4_utils.run_gbs_toy_target4
    n_particles = _safe_n_particles(args.dim, args.n_particles, args.n_spatial_dim)
    args.n_particles = n_particles
    target_dir_name = "target4_uniform" if target_mode == "uniform" else f"target4_v{args.target4_version}"
    target_output_dir = args.output_dir / target_dir_name
    gbs_artifact_dir = target_output_dir / "online_compare_gbs_artifacts"

    target_params = build_compare_target_params(args, args.dim)
    snapshot_iters = _build_snapshot_iters(args.iters, args.gif_num_frames) if args.save_dim01_gif else []
    gbs_hist_raw = run_gbs_toy_target4(
        low=jnp.zeros(args.dim),
        high=jnp.ones(args.dim),
        dim=args.dim,
        T=args.iters,
        batch_size=args.gbs_batch_size,
        num_steps=args.gbs_num_steps,
        lr=args.gbs_lr,
        init_std=args.gbs_init_std,
        seed=args.seed,
        beta=args.beta,
        tau=args.tau,
        q=args.safe_q,
        initial_p=args.initial_p,
        p_update_freq=args.p_update_freq,
        p_ema_alpha=args.p_ema_alpha,
        p_jump_prob=args.p_jump_prob,
        loss_mode=args.gbs_loss_mode,
        metric_num_bins=args.metric_num_bins,
        sinkhorn_num_samples=args.sinkhorn_num_samples,
        n_particles=args.n_particles,
        n_spatial_dim=args.n_spatial_dim,
        save_dir=gbs_artifact_dir,
        use_tanh_bijection=args.use_tanh_bijection,
        model_type=args.gbs_model_type,
        model_num_layers=args.gbs_model_num_layers,
        model_num_hid=args.gbs_model_num_hid,
        final_sample_size=2**12,
        target_params=target_params,
        snap_iters=snapshot_iters,
        return_snapshots=args.save_dim01_gif,
        snapshot_sample_size=args.gif_sample_size,
        max_metric_eval_points=args.max_eval_points,
    )
    if args.save_dim01_gif:
        _, _, gbs_hist, gbs_final_samples, gbs_snapshots = gbs_hist_raw
    else:
        _, _, gbs_hist, gbs_final_samples = gbs_hist_raw
        gbs_snapshots = []
    gbs_hist_np = {key: np.asarray(value, dtype=np.float64) for key, value in gbs_hist.items()}

    gmmvi_result = run_gmmvi_target4_online(args)
    gmmvi_hist_np = gmmvi_result["hist"]
    gmmvi_final_samples = gmmvi_result["final_samples"]
    gmmvi_snapshots = gmmvi_result.get("snapshots", [])
    uniform_baseline = compute_uniform_baseline_curves(
        gmmvi_hist_np["target4/p"],
        beta=args.beta,
        dim=args.dim,
        sinkhorn_num_samples=args.sinkhorn_num_samples,
        target_params=target_params,
        target4_version=args.target4_version,
        seed=args.seed + 202,
        eval_mask=np.isfinite(gmmvi_hist_np["target4/sinkhorn"]),
    )

    gbs_tag_args = argparse.Namespace(
        seed=args.seed,
        dim=args.dim,
        iters=args.iters,
        batch_size=args.gbs_batch_size,
        num_steps=args.gbs_num_steps,
        lr=args.gbs_lr,
        init_std=args.gbs_init_std,
        beta=args.beta,
        tau=args.tau,
        p_update_freq=args.p_update_freq,
        p_ema_alpha=args.p_ema_alpha,
        p_jump_prob=args.p_jump_prob,
        loss_mode=args.gbs_loss_mode,
        use_tanh_bijection=args.use_tanh_bijection,
        model_type=args.gbs_model_type,
    )
    gmmvi_tag_args = argparse.Namespace(
        seed=args.seed + 1,
        dim=args.dim,
        iters=args.iters,
        num_envs=args.gmmvi_num_envs,
        batch_size=args.gmmvi_batch_size,
        n_eval_samples=args.gmmvi_eval_samples,
        prior_scale=args.gmmvi_prior_scale,
        beta=args.beta,
        tau=args.tau,
        p_update_freq=args.p_update_freq,
        p_ema_alpha=args.p_ema_alpha,
        p_jump_prob=args.p_jump_prob,
    )
    gbs_tag = build_gbs_run_tag(gbs_tag_args)
    gmmvi_tag = build_gmmvi_run_tag(gmmvi_tag_args)

    output_path = target_output_dir / (
        f"{gbs_tag}_target4v{args.target4_version}.png"#__gmmvi_{gmmvi_tag}.png"
    )
    title = (
        f"Target4 online comparison | dim={args.dim}, iters={args.iters}, "
        f"beta={args.beta:g}, tau={args.tau:g}"
    )
    save_unified_plot(
        gbs_hist_np,
        gmmvi_hist_np,
        uniform_baseline,
        output_path=output_path,
        hide_initial_point=args.hide_initial_point,
        title=title,
    )
    if args.dim >= 2:
        dim01_output_path = target_output_dir / f"{gbs_tag}_target4v{args.target4_version}_dim01.png"
        save_dim01_plot(
            gbs_final_samples,
            gmmvi_final_samples,
            target_params,
            lam_gbs=float(args.beta * gbs_hist_np["target4/p"][-1]),
            lam_gmmvi=float(args.beta * gmmvi_hist_np["target4/p"][-1]),
            output_path=dim01_output_path,
            target4_version=args.target4_version,
            policy_p_gbs=float(gbs_hist_np["target4/p"][-1]),
            policy_p_gmmvi=float(gmmvi_hist_np["target4/p"][-1]),
        )
        print(f"Saved dim0/dim1 comparison plot to: {dim01_output_path}")
        if args.save_dim01_gif:
            dim01_gif_path = target_output_dir / f"{gbs_tag}_target4v{args.target4_version}_dim01.gif"
            gif_saved = save_dim01_gif(
                gbs_snapshots,
                gmmvi_snapshots,
                target_params,
                beta=args.beta,
                output_path=dim01_gif_path,
                target4_version=args.target4_version,
                fps=args.gif_fps,
            )
            if gif_saved:
                print(f"Saved dim0/dim1 GIF to: {dim01_gif_path}")
            else:
                print(
                    "Skipped dim0/dim1 GIF: missing snapshots "
                    f"(GBS={len(gbs_snapshots)}, GMMVI={len(gmmvi_snapshots)})"
                )

    print(f"Saved unified comparison plot to: {output_path}")
    print(f"Target4 version: {args.target4_version}")
    print(f"Target4 family: {args.target4_family}")
    print(f"Target mode: {target_mode}")
    print(f"GBS uses tanh bijection: {args.use_tanh_bijection}")
    print(f"GBS final p: {_last_finite(gbs_hist_np['target4/p']):.6f}")
    print(f"GMMVI final p: {_last_finite(gmmvi_hist_np['target4/p']):.6f}")
    print(f"GBS final Sinkhorn: {_last_finite(gbs_hist_np['target4/sinkhorn']):.6f}")
    print(f"GMMVI final Sinkhorn: {_last_finite(gmmvi_hist_np['target4/sinkhorn']):.6f}")
    print(f"GBS final Energy W2: {_last_finite(gbs_hist_np['target4/energy_w2']):.6f}")
    print(f"GMMVI final Energy W2: {_last_finite(gmmvi_hist_np['target4/energy_w2']):.6f}")
    print(f"Uniform-baseline Sinkhorn: {_last_finite(uniform_baseline['uniform_baseline/sinkhorn']):.6f}")
    print(f"Uniform-baseline Energy W2: {_last_finite(uniform_baseline['uniform_baseline/energy_w2']):.6f}")


if __name__ == "__main__":
    main()
