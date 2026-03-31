from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from learning.module.gbs.gbs_test_toy import build_run_tag as build_gbs_run_tag
from learning.module.gbs.gmmvi_test_toy import build_run_tag as build_gmmvi_run_tag
from learning.module.gbs.gmmvi_test_toy import build_gmmvi_fns
from learning.module.gbs.sinkhorn_metrics import (
    energy_wasserstein_1d,
    effective_sample_size_from_log_weights,
    interatomic_wasserstein_1d,
    sinkhorn_distance,
)
from learning.module.gbs.target4_notebook_utils import (
    compute_target4_metrics,
    optimal_p_from_target_mean,
    run_gbs_toy_target4,
    sample_truncated_exponential,
    target4_logprob,
    update_p_with_ema_and_jump,
)
from learning.module.gmmvi.network import create_gmm_network_and_state


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


def _plot_curve(ax, values: np.ndarray, label: str, color: str, hide_initial_point: bool) -> None:
    curve = _hide_initial(values) if hide_initial_point else values
    ax.plot(np.arange(curve.size), curve, label=label, color=color, linewidth=2.0)


def run_gmmvi_target4_online(args: argparse.Namespace) -> dict[str, np.ndarray]:
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
    gather_samples, train_iter, sample_model, model_log_density = build_gmmvi_fns(
        gmm_network, args.gmmvi_num_envs
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

    current_lambda = args.beta * p / dim
    for _ in range(max(args.gmmvi_batch_size // args.gmmvi_num_envs, 1)):
        key, subkey = jax.random.split(key)
        state = gather_samples(state, subkey, jnp.asarray(current_lambda))

    for step in trange(args.iters, desc="GMMVI", leave=False):
        current_lambda = args.beta * p / dim

        key, subkey = jax.random.split(key)
        state = train_iter(state, subkey, jnp.asarray(current_lambda))

        key, k_eval, k_metric = jax.random.split(key, 3)
        samples = np.asarray(sample_model(state, k_eval, args.gmmvi_eval_samples))
        _ = model_log_density(state, jnp.asarray(samples))

        sample_mean = float(np.mean(samples))
        forward_kl, reverse_kl, wasserstein = compute_target4_metrics(
            samples,
            current_lambda,
            num_bins=args.metric_num_bins,
            key=k_metric,
        )
        key, k_sink = jax.random.split(key)
        samples_jax = jnp.asarray(samples)
        sinkhorn_target = sample_truncated_exponential(k_sink, current_lambda, samples.shape)
        n_sink = min(args.sinkhorn_num_samples, samples.shape[0])
        sinkhorn = sinkhorn_distance(samples_jax[:n_sink], sinkhorn_target[:n_sink])
        ess = effective_sample_size_from_log_weights(target4_logprob(samples_jax, current_lambda))
        energy_w2 = float(
            energy_wasserstein_1d(
                samples_jax[:n_sink],
                sinkhorn_target[:n_sink],
                current_lambda,
            )
        )
        interatomic_w2 = float(
            interatomic_wasserstein_1d(
                samples_jax[:n_sink],
                sinkhorn_target[:n_sink],
                n_particles=n_particles,
                n_spatial_dim=args.n_spatial_dim,
            )
        )
        optimal_p, target_mean = optimal_p_from_target_mean(current_lambda, args.tau)

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
        hist["target4/p_base"].append(float(jax.nn.sigmoid((sample_mean - 1.0) / args.tau)))
        hist["target4/p_ema"].append(float(p))

        if should_update_p:
            key, k_update = jax.random.split(key)
            p, base_p, ema_p, jumped = update_p_with_ema_and_jump(
                prev_p=p,
                sample_mean=sample_mean,
                tau=args.tau,
                ema_alpha=args.p_ema_alpha,
                jump_prob=args.p_jump_prob,
                key=k_update,
            )
            hist["target4/p_jumped"][-1] = float(jumped)
            hist["target4/p_base"][-1] = float(base_p)
            hist["target4/p_ema"][-1] = float(ema_p)

    return {key: np.asarray(value, dtype=np.float64) for key, value in hist.items()}


def save_unified_plot(
    gbs_hist: dict[str, np.ndarray],
    gmmvi_hist: dict[str, np.ndarray],
    output_path: Path,
    hide_initial_point: bool,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    colors = {"DIS-LV": "tab:blue", "GMMVI": "tab:orange"}

    metric_specs = [
        ("target4/p", "Learned policy p", False),
        ("target4/sinkhorn", "Sinkhorn Distance", hide_initial_point),
        ("target4/interatomic_w2", r"\mathcal{W}_2$", hide_initial_point),
        ("target4/energy_w2", r"$E(\cdot)\,\mathcal{W}_2$", hide_initial_point),
    ]

    for ax, (metric_key, metric_title, mask_first) in zip(axes, metric_specs):
        _plot_curve(ax, gmmvi_hist[metric_key], "GMMVI", colors["GMMVI"], mask_first)
        _plot_curve(ax, gbs_hist[metric_key], "DIS-LV", colors["DIS-LV"], mask_first)
        # ax.set_title(metric_title)
        ax.set_xlabel("iteration", fontsize=20)
        ax.tick_params(axis="both", labelsize=15)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Learned Policy Probability", fontsize=20)
    axes[1].set_ylabel("Sinkhorn Distance", fontsize=20)
    axes[2].set_ylabel(r"$\mathcal{W}_2$", fontsize=20)
    axes[3].set_ylabel(r"$E(\cdot)\,\mathcal{W}_2$", fontsize=20)
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    # fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.as_posix(), dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GBS and GMMVI target4 experiments online and compare learnable p, Sinkhorn, and Energy W2."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--beta", type=float, default=-200.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--initial_p", type=float, default=0.9)
    parser.add_argument("--p_update_freq", type=int, default=1)
    parser.add_argument("--p_ema_alpha", type=float, default=0.99)
    parser.add_argument("--p_jump_prob", type=float, default=0.0)
    parser.add_argument("--metric_num_bins", type=int, default=128)
    parser.add_argument("--sinkhorn_num_samples", type=int, default=1024)
    parser.add_argument("--n_particles", type=int, default=None)
    parser.add_argument("--n_spatial_dim", type=int, default=1)

    parser.add_argument("--gbs_batch_size", type=int, default=128)
    parser.add_argument("--gbs_num_steps", type=int, default=100)
    parser.add_argument("--gbs_lr", type=float, default=1e-3)
    parser.add_argument("--gbs_init_std", type=float, default=0.5)
    parser.add_argument("--gbs_loss_mode", choices=["tr_lv", "tr_lv_subtraj", "dis"], default="tr_lv")
    parser.add_argument("--gbs_model_type", choices=["pisgrad", "potential"], default="pisgrad")
    parser.add_argument("--gbs_model_num_layers", type=int, default=2)
    parser.add_argument("--gbs_model_num_hid", type=int, default=64)

    parser.add_argument("--gmmvi_num_envs", type=int, default=1024)
    parser.add_argument("--gmmvi_batch_size", type=int, default=1024)
    parser.add_argument("--gmmvi_eval_samples", type=int, default=4096)
    parser.add_argument("--gmmvi_prior_scale", type=float, default=0.5)

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/target4_online_comparison"),
    )
    parser.add_argument("--hide_initial_point", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_particles = _safe_n_particles(args.dim, args.n_particles, args.n_spatial_dim)
    args.n_particles = n_particles
    gbs_artifact_dir = args.output_dir / "online_compare_gbs_artifacts"

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
        model_type=args.gbs_model_type,
        model_num_layers=args.gbs_model_num_layers,
        model_num_hid=args.gbs_model_num_hid,
        final_sample_size=2**12,
    )
    _, _, gbs_hist, _ = gbs_hist_raw
    gbs_hist_np = {key: np.asarray(value, dtype=np.float64) for key, value in gbs_hist.items()}

    gmmvi_hist_np = run_gmmvi_target4_online(args)

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

    output_path = args.output_dir / (
        f"target4_online_compare_p_sinkhorn_energy__gbs_{gbs_tag}.png"#__gmmvi_{gmmvi_tag}.png"
    )
    title = (
        f"Target4 online comparison | dim={args.dim}, iters={args.iters}, "
        f"beta={args.beta:g}, tau={args.tau:g}"
    )
    save_unified_plot(
        gbs_hist_np,
        gmmvi_hist_np,
        output_path=output_path,
        hide_initial_point=args.hide_initial_point,
        title=title,
    )

    print(f"Saved unified comparison plot to: {output_path}")
    print(f"GBS final p: {gbs_hist_np['target4/p'][-1]:.6f}")
    print(f"GMMVI final p: {gmmvi_hist_np['target4/p'][-1]:.6f}")
    print(f"GBS final Sinkhorn: {gbs_hist_np['target4/sinkhorn'][-1]:.6f}")
    print(f"GMMVI final Sinkhorn: {gmmvi_hist_np['target4/sinkhorn'][-1]:.6f}")
    print(f"GBS final Energy W2: {gbs_hist_np['target4/energy_w2'][-1]:.6f}")
    print(f"GMMVI final Energy W2: {gmmvi_hist_np['target4/energy_w2'][-1]:.6f}")


if __name__ == "__main__":
    main()
