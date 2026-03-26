from __future__ import annotations

import argparse
import os
from pathlib import Path

from learning.module.gbs.gbs_loss import rnd_no_target

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm

from learning.module.gbs.target4_notebook_utils import (
    run_gbs_toy_target4,
    target4_logprob,
)


def plot_target4_contour(ax, lam: float, n: int = 200, gamma: float = 0.45) -> None:
    x, y = jnp.meshgrid(
        jnp.linspace(0.0, 1.0, n),
        jnp.linspace(0.0, 1.0, n),
        indexing="xy",
    )
    grid = jnp.stack([x.reshape(-1), y.reshape(-1)], axis=-1)
    z = jnp.exp(jnp.clip(target4_logprob(grid, lam), a_min=-1000.0)).reshape(n, n)
    contour = ax.contourf(
        np.asarray(x),
        np.asarray(y),
        np.asarray(z),
        levels=12,
        cmap="viridis",
        norm=PowerNorm(gamma),
    )
    plt.colorbar(contour, ax=ax)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")


def save_metric_plot(hist: dict[str, list[float]], output_path: Path) -> None:
    iters = np.arange(len(hist["target4/p"]))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].plot(iters, hist["target4/p"], label="p", color="tab:blue")
    axes[0].plot(iters, hist["target4/p_base"], label="base p", color="tab:pink")
    axes[0].plot(iters, hist["target4/p_ema"], label="ema p", color="tab:cyan")
    axes[0].plot(iters, hist["target4/lambda"], label="lambda", color="tab:orange")
    axes[0].plot(iters, hist["target4/sample_mean"], label="sample mean", color="tab:green")
    axes[0].set_title("Target4 dynamics")
    axes[0].set_xlabel("iteration")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        iters,
        hist["target4/forward_kl"],
        label="forward KL = KL(target || empirical)",
        color="tab:red",
    )
    axes[1].plot(
        iters,
        hist["target4/reverse_kl"],
        label="reverse KL = KL(empirical || target)",
        color="tab:purple",
    )
    # axes[1].plot(iters, hist["target4/wasserstein"], label="Wasserstein", color="tab:brown")

    axes[1].set_title("Target4 distances")
    axes[1].set_xlabel("iteration")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    # axes[2].plot(iters, hist["target4/ess"], label="ESS", color="tab:olive")
    axes[2].plot(iters, hist["target4/sinkhorn"], label="Sinkhorn", color="tab:cyan")
    # axes[2].plot(iters, hist["target4/p_updated"], label="p updated", color="tab:gray")
    # axes[2].plot(iters, hist["target4/p_jumped"], label="p jumped", color="tab:red")
    axes[2].set_title("Target4 Sinkhorn")
    axes[2].set_xlabel("iteration")
    axes[2].grid(alpha=0.3)
    axes[2].legend()
    axes[3].plot(iters, hist["target4/energy_w2"], label="Energy W2", color="tab:brown")
    axes[3].set_title("Target4 Energy W2")
    axes[3].set_xlabel("iteration")
    axes[3].grid(alpha=0.3)
    axes[3].legend()
    fig.tight_layout()
    fig.savefig(output_path.as_posix(), dpi=150)
    plt.close(fig)


def save_final_sample_plot(samples: np.ndarray, lam: float, output_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_target4_contour(ax, lam=lam)
    ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.2, c="r")
    ax.set_title(f"Final samples vs target4 (lambda={lam:.4f})")
    fig.tight_layout()
    fig.savefig(output_path.as_posix(), dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GBS toy experiment with dynamic target4.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=2, help="Dimension d in lambda = beta * p / d.")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init_std", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=6.0)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--initial_p", type=float, default=None)
    parser.add_argument(
        "--p_update_freq",
        type=int,
        default=1,
        help="Update p every k iterations. Use 0 to keep p fixed for the whole run.",
    )
    parser.add_argument(
        "--p_ema_alpha",
        type=float,
        default=0.9,
        help="EMA coefficient on previous p during scheduled updates.",
    )
    parser.add_argument(
        "--p_jump_prob",
        type=float,
        default=0.1,
        help="Probability that a scheduled p update jumps to Uniform[0, 1].",
    )
    parser.add_argument("--metric_num_bins", type=int, default=128)
    parser.add_argument("--sinkhorn_num_samples", type=int, default=2000)
    parser.add_argument("--n_particles", type=int, default=None)
    parser.add_argument("--n_spatial_dim", type=int, default=1)
    parser.add_argument("--loss_mode", choices=["tr_lv", "dis"], default="tr_lv")
    parser.add_argument("--model_type", choices=["pisgrad", "potential"], default="pisgrad")
    parser.add_argument("--model_num_layers", type=int, default=2)
    parser.add_argument("--model_num_hid", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    low = jnp.zeros(args.dim)
    high = jnp.ones(args.dim)

    fwd_state, bwd_state, hist, xT_final = run_gbs_toy_target4(
        low=low,
        high=high,
        dim=args.dim,
        T=args.iters,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        init_std=args.init_std,
        seed=args.seed,
        beta=args.beta,
        tau=args.tau,
        initial_p=args.initial_p,
        p_update_freq=args.p_update_freq,
        p_ema_alpha=args.p_ema_alpha,
        p_jump_prob=args.p_jump_prob,
        loss_mode=args.loss_mode,
        metric_num_bins=args.metric_num_bins,
        sinkhorn_num_samples=args.sinkhorn_num_samples,
        n_particles=args.n_particles,
        n_spatial_dim=args.n_spatial_dim,
        save_dir=save_dir,
        model_type=args.model_type,
        model_num_layers=args.model_num_layers,
        model_num_hid=args.model_num_hid,
        final_sample_size=2**14
    )

    hist_np = {k: np.asarray(v, dtype=np.float64) for k, v in hist.items()}
    np.savez((save_dir / "target4_history.npz").as_posix(), **hist_np)

    final_p = float(hist["target4/p"][-1])
    final_lambda = float(hist["target4/lambda"][-1])
    final_forward_kl = float(hist["target4/forward_kl"][-1])
    final_reverse_kl = float(hist["target4/reverse_kl"][-1])
    final_wasserstein = float(hist["target4/wasserstein"][-1])
    final_sinkhorn = float(hist["target4/sinkhorn"][-1])
    final_ess = float(hist["target4/ess"][-1])
    final_energy_w2 = float(hist["target4/energy_w2"][-1])

    save_metric_plot(hist, save_dir / "target4_metrics.png")
    if args.dim >= 2:
        save_final_sample_plot(np.asarray(xT_final), final_lambda, save_dir / "target4_final_samples.png")

    print(f"Saved outputs to: {save_dir}")
    print(f"dimension d: {args.dim}")
    print(f"model type: {args.model_type}")
    print(f"p update frequency: {args.p_update_freq} (0 means fixed p)")
    print(f"p ema alpha: {args.p_ema_alpha:.3f}")
    print(f"p jump probability: {args.p_jump_prob:.3f}")
    print(f"final p: {final_p:.6f}")
    print(f"final lambda = beta * p / d: {final_lambda:.6f}")
    print(f"final forward KL: {final_forward_kl:.6f}")
    print(f"final reverse KL: {final_reverse_kl:.6f}")
    print(f"final Wasserstein: {final_wasserstein:.6f}")
    print(f"final Sinkhorn: {final_sinkhorn:.6f}")
    print(f"final ESS: {final_ess:.6f}")
    print(f"final Energy W2: {final_energy_w2:.6f}")

    if args.show:
        metrics = plt.imread((save_dir / "target4_metrics.png").as_posix())
        plt.figure(figsize=(12, 4))
        plt.imshow(metrics)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
