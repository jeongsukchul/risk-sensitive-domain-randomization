from __future__ import annotations

import argparse
import functools
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from tqdm import trange

from learning.module.gmmvi.network import GMMTrainingState, create_gmm_network_and_state
from learning.module.gbs.target4_notebook_utils import (
    compute_target4_metrics,
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(iters, hist["target4/p"], label="p", color="tab:blue")
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
    axes[1].plot(iters, hist["target4/wasserstein"], label="Wasserstein", color="tab:brown")
    axes[1].set_title("Target4 distances")
    axes[1].set_xlabel("iteration")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path.as_posix(), dpi=150)
    plt.close(fig)


def save_final_sample_plot(samples: np.ndarray, lam: float, output_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_target4_contour(ax, lam=lam)
    ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.2, c="r")
    ax.set_title(f"GMMVI final samples vs target4 (lambda={lam:.4f})")
    fig.tight_layout()
    fig.savefig(output_path.as_posix(), dpi=150)
    plt.close(fig)


def _target4_scalar_logprob(sample: jax.Array, lam: jax.Array) -> jax.Array:
    return target4_logprob(sample[None, :], lam).reshape(())


def build_gmmvi_fns(gmm_network, num_envs: int):
    @jax.jit
    def gather_samples(train_state: GMMTrainingState, key: chex.Array, lam: jax.Array):
        target_value_and_grad = jax.value_and_grad(_target4_scalar_logprob, argnums=0)
        key, subkey = jax.random.split(key)
        new_samples, mapping = gmm_network.sample_selector.select_samples(train_state.model_state, subkey)
        new_target_lnpdfs, new_target_grads = jax.vmap(
            lambda sample: target_value_and_grad(sample, lam)
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
    def train_iter(train_state: GMMTrainingState, key: chex.Array, lam: jax.Array):
        target_value_and_grad = jax.value_and_grad(_target4_scalar_logprob, argnums=0)
        key, subkey = jax.random.split(key)
        new_samples, mapping = gmm_network.sample_selector.select_samples(train_state.model_state, subkey)
        new_target_lnpdfs, new_target_grads = jax.vmap(
            lambda sample: target_value_and_grad(sample, lam)
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
    def sample_model(train_state: GMMTrainingState, key: chex.Array, n_samples: int):
        return gmm_network.model.sample(train_state.model_state.gmm_state, key, n_samples)[0]

    @jax.jit
    def model_log_density(train_state: GMMTrainingState, samples: jax.Array):
        return jax.vmap(
            functools.partial(gmm_network.model.log_density, gmm_state=train_state.model_state.gmm_state)
        )(sample=samples)

    return gather_samples, train_iter, sample_model, model_log_density


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GMMVI toy experiment with dynamic target4.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=2, help="Dimension d in lambda = beta * p / d.")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_eval_samples", type=int, default=4096)
    parser.add_argument("--prior_scale", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--initial_p", type=float, default=None)
    parser.add_argument(
        "--p_update_freq",
        type=int,
        default=1,
        help="Update p every k iterations. Use 0 to keep p fixed for the whole run.",
    )
    parser.add_argument("--metric_num_bins", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    low = jnp.zeros(args.dim)
    high = jnp.ones(args.dim)

    key = jax.random.PRNGKey(args.seed)
    key, k_init, k_p0 = jax.random.split(key, 3)
    state, gmm_network = create_gmm_network_and_state(
        args.dim,
        args.num_envs,
        args.batch_size,
        k_init,
        prior_scale=args.prior_scale,
        bound_info=(low, high),
    )
    gather_samples, train_iter, sample_model, model_log_density = build_gmmvi_fns(
        gmm_network, args.num_envs
    )

    if args.initial_p is None:
        p = float(jax.random.uniform(k_p0, minval=0.0, maxval=1.0))
    else:
        p = float(np.clip(args.initial_p, 0.0, 1.0))
    hist = {
        "target4/p": [],
        "target4/lambda": [],
        "target4/sample_mean": [],
        "target4/forward_kl": [],
        "target4/reverse_kl": [],
        "target4/wasserstein": [],
        "target4/p_updated": [],
        "model/num_components": [],
    }

    current_lambda = args.beta * p / args.dim
    for _ in range(max(args.batch_size // args.num_envs, 1)):
        key, subkey = jax.random.split(key)
        state = gather_samples(state, subkey, jnp.asarray(current_lambda))

    for step in trange(args.iters):
        current_lambda = args.beta * p / args.dim

        key, subkey = jax.random.split(key)
        state = train_iter(state, subkey, jnp.asarray(current_lambda))

        key, k_eval, k_metric = jax.random.split(key, 3)
        samples = np.asarray(sample_model(state, k_eval, args.n_eval_samples))
        _ = model_log_density(state, jnp.asarray(samples))

        sample_mean = float(np.mean(samples))
        forward_kl, reverse_kl, wasserstein = compute_target4_metrics(
            samples, current_lambda, num_bins=args.metric_num_bins, key=k_metric
        )

        hist["target4/p"].append(float(p))
        hist["target4/lambda"].append(float(current_lambda))
        hist["target4/sample_mean"].append(sample_mean)
        hist["target4/forward_kl"].append(forward_kl)
        hist["target4/reverse_kl"].append(reverse_kl)
        hist["target4/wasserstein"].append(wasserstein)
        hist["model/num_components"].append(int(state.model_state.gmm_state.num_components))
        should_update_p = args.p_update_freq > 0 and ((step + 1) % args.p_update_freq == 0)
        hist["target4/p_updated"].append(float(should_update_p))
        if should_update_p:
            p = float(jax.nn.sigmoid((sample_mean - 1.0) / args.tau))

    key, k_final = jax.random.split(key)
    x_final = np.asarray(sample_model(state, k_final, 2**14))
    np.save((save_dir / "gmmvi_samples.npy").as_posix(), x_final)
    np.savez(
        (save_dir / "gmmvi_target4_history.npz").as_posix(),
        **{k: np.asarray(v, dtype=np.float64) for k, v in hist.items()},
    )

    final_p = float(hist["target4/p"][-1])
    final_lambda = float(hist["target4/lambda"][-1])
    final_forward_kl = float(hist["target4/forward_kl"][-1])
    final_reverse_kl = float(hist["target4/reverse_kl"][-1])
    final_wasserstein = float(hist["target4/wasserstein"][-1])

    save_metric_plot(hist, save_dir / "gmmvi_target4_metrics.png")
    if args.dim >= 2:
        save_final_sample_plot(x_final, final_lambda, save_dir / "gmmvi_target4_final_samples.png")

    print(f"Saved outputs to: {save_dir}")
    print(f"dimension d: {args.dim}")
    print(f"p update frequency: {args.p_update_freq} (0 means fixed p)")
    print(f"final p: {final_p:.6f}")
    print(f"final lambda = beta * p / d: {final_lambda:.6f}")
    print(f"final forward KL: {final_forward_kl:.6f}")
    print(f"final reverse KL: {final_reverse_kl:.6f}")
    print(f"final Wasserstein: {final_wasserstein:.6f}")
    print(f"final num components: {hist['model/num_components'][-1]}")

    if args.show:
        metrics = plt.imread((save_dir / "gmmvi_target4_metrics.png").as_posix())
        plt.figure(figsize=(12, 4))
        plt.imshow(metrics)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
