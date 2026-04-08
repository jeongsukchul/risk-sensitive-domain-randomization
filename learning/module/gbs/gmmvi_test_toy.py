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
    optimal_p_from_target_mean,
    sample_truncated_exponential,
    target4_logprob,
    update_p_with_ema_and_jump,
)
from learning.module.gbs.target4_family import (
    add_target4_cli_args,
    build_target4_params_from_args,
    get_target4_harmonic_params,
    target4_energy_values,
)
from learning.module.gbs.sinkhorn_metrics import (
    energy_wasserstein_1d,
    effective_sample_size_from_log_weights,
    interatomic_wasserstein_1d,
    sinkhorn_distance,
)


def plot_target4_contour(
    ax,
    lam: float,
    *,
    target_params=None,
    n: int = 200,
    gamma: float = 0.45,
) -> None:
    x, y = jnp.meshgrid(
        jnp.linspace(0.0, 1.0, n),
        jnp.linspace(0.0, 1.0, n),
        indexing="xy",
    )
    grid = jnp.stack([x.reshape(-1), y.reshape(-1)], axis=-1)
    if target_params is None:
        params = get_target4_harmonic_params(2)
    else:
        params = get_target4_harmonic_params(
            2,
            type(target_params)(
                c=target_params.c,
                a=target_params.a[:2],
                k=target_params.k[:2],
                phi=target_params.phi[:2],
            ),
        )
    z = jnp.exp(
        jnp.clip(target4_logprob(grid, lam, target_params=params), a_min=-1000.0)
    ).reshape(n, n)
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


def format_run_hparams(args: argparse.Namespace) -> str:
    return (
        f"seed={args.seed}, dim={args.dim}, iters={args.iters}, envs={args.num_envs}, "
        f"batch={args.batch_size}, "
        f"beta={args.beta:g}, tau={args.tau:g}, p_freq={args.p_update_freq}, "
        f"ema={args.p_ema_alpha:g}"
    )


def build_run_tag(args: argparse.Namespace) -> str:
    def sanitize(value: object) -> str:
        return str(value).replace("-", "m").replace(".", "p")

    parts = [
        f"seed{sanitize(args.seed)}",
        f"d{sanitize(args.dim)}",
        f"T{sanitize(args.iters)}",
        f"envs{sanitize(args.num_envs)}",
        f"bs{sanitize(args.batch_size)}",
        f"eval{sanitize(args.n_eval_samples)}",
        f"prior{sanitize(args.prior_scale)}",
        f"beta{sanitize(args.beta)}",
        f"tau{sanitize(args.tau)}",
        f"pf{sanitize(args.p_update_freq)}",
        f"ema{sanitize(args.p_ema_alpha)}",
        f"jump{sanitize(args.p_jump_prob)}",
    ]
    return "_".join(parts)


def save_metric_plot(hist: dict[str, list[float]], output_path: Path) -> None:
    def hide_initial_point(values: list[float]) -> np.ndarray:
        masked = np.asarray(values, dtype=np.float64).copy()
        if masked.size:
            masked[0] = np.nan
        return masked

    iters = np.arange(len(hist["target4/p"]))
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].plot(iters, hist["target4/p"], label="p", color="tab:blue")
    axes[0].plot(iters, hist["target4/p_base"], label="base p", color="tab:pink")
    axes[0].plot(iters, hist["target4/p_ema"], label="ema p", color="tab:cyan")
    # axes[0].plot(iters, hist["target4/lambda"], label="lambda", color="tab:orange")
    axes[0].plot(iters, hist["target4/sample_mean"], label="sample mean", color="tab:green")
    axes[0].set_title("Target4 dynamics")
    axes[0].set_xlabel("iteration")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        iters,
        hide_initial_point(hist["target4/forward_kl"]),
        label="forward KL = KL(target || empirical)",
        color="tab:red",
    )
    axes[1].plot(
        iters,
        hide_initial_point(hist["target4/reverse_kl"]),
        label="reverse KL = KL(empirical || target)",
        color="tab:purple",
    )
    # axes[1].plot(iters, hist["target4/wasserstein"], label="Wasserstein", color="tab:brown")
    axes[1].set_title("Target4 distances")
    axes[1].set_xlabel("iteration")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        iters,
        hide_initial_point(hist["target4/sinkhorn"]),
        label="Sinkhorn",
        color="tab:cyan",
    )
    # axes[2].plot(iters, hist["target4/p_updated"], label="p updated", color="tab:gray")
    # axes[2].plot(iters, hist["target4/p_jumped"], label="p jumped", color="tab:red")
    axes[2].set_title("Target4 Sinkhorn")
    axes[2].set_xlabel("iteration")
    axes[2].grid(alpha=0.3)
    axes[2].legend()
    axes[3].plot(
        iters,
        hide_initial_point(hist["target4/energy_w2"]),
        label="Energy W2",
        color="tab:brown",
    )
    axes[3].set_title("Target4 Energy W2")
    axes[3].set_xlabel("iteration")
    axes[3].grid(alpha=0.3)
    axes[3].legend()
    axes[4].plot(
        iters,
        hist["target4/optimal_p"],
        label="optimal p",
        color="tab:blue",
    )
    axes[4].plot(
        iters,
        hist["target4/target_mean"],
        label="target mean",
        color="tab:green",
    )
    axes[4].plot(iters, hist["target4/p"], label="sampler p", color="tab:orange")
    axes[4].set_title("Target Boltzmann p")
    axes[4].set_xlabel("iteration")
    axes[4].grid(alpha=0.3)
    axes[4].legend()

    fig.tight_layout()
    fig.savefig(output_path.as_posix(), dpi=150)
    plt.close(fig)


def save_final_sample_plot(
    samples: np.ndarray,
    lam: float,
    output_path: Path,
    *,
    target_params=None,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_target4_contour(ax, lam=lam, target_params=target_params)
    ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.2, c="r")
    ax.set_title(f"GMMVI final samples vs target4 (lambda={lam:.4f})")
    fig.tight_layout()
    fig.savefig(output_path.as_posix(), dpi=150)
    plt.close(fig)


def _target4_scalar_logprob(sample: jax.Array, lam: jax.Array, target_params) -> jax.Array:
    return target4_logprob(sample[None, :], lam, target_params=target_params).reshape(())


def build_gmmvi_fns(gmm_network, num_envs: int, target_params):
    @jax.jit
    def gather_samples(train_state: GMMTrainingState, key: chex.Array, lam: jax.Array):
        target_value_and_grad = jax.value_and_grad(_target4_scalar_logprob, argnums=0)
        key, subkey = jax.random.split(key)
        new_samples, mapping = gmm_network.sample_selector.select_samples(train_state.model_state, subkey)
        new_target_lnpdfs, new_target_grads = jax.vmap(
            lambda sample: target_value_and_grad(sample, lam, target_params)
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
            lambda sample: target_value_and_grad(sample, lam, target_params)
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
    parser.add_argument("--dim", type=int, default=2, help="Dimension d of the target family.")
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_eval_samples", type=int, default=4096)
    parser.add_argument("--prior_scale", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=1.)
    parser.add_argument("--safe_q", type=float, default=.6)
    parser.add_argument("--initial_p", type=float, default=0.8)
    parser.add_argument(
        "--p_update_freq",
        type=int,
        default=1,
        help="Update p every k iterations. Use 0 to keep p fixed for the whole run.",
    )
    parser.add_argument(
        "--p_ema_alpha",
        type=float,
        default=0.99,
        help="EMA coefficient on previous p during scheduled updates.",
    )
    parser.add_argument(
        "--p_jump_prob",
        type=float,
        default=0.0,
        help="Probability that a scheduled p update jumps to Uniform[0, 1].",
    )
    parser.add_argument("--metric_num_bins", type=int, default=128)
    parser.add_argument("--sinkhorn_num_samples", type=int, default=2000)
    parser.add_argument("--n_particles", type=int, default=None)
    parser.add_argument("--n_spatial_dim", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--show", action="store_true")
    add_target4_cli_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_tag = build_run_tag(args)

    low = jnp.zeros(args.dim)
    high = jnp.ones(args.dim)
    if args.n_particles is None:
        if args.dim % args.n_spatial_dim != 0:
            raise ValueError(f"dim={args.dim} must be divisible by n_spatial_dim={args.n_spatial_dim}")
        args.n_particles = args.dim // args.n_spatial_dim
    if args.n_particles * args.n_spatial_dim != args.dim:
        raise ValueError(
            f"n_particles * n_spatial_dim must equal dim, got {args.n_particles} * {args.n_spatial_dim} != {args.dim}"
        )

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
    target_params = build_target4_params_from_args(args, args.dim)
    gather_samples, train_iter, sample_model, model_log_density = build_gmmvi_fns(
        gmm_network, args.num_envs, target_params
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

    current_lambda = args.beta * p
    for _ in range(max(args.batch_size // args.num_envs, 1)):
        key, subkey = jax.random.split(key)
        state = gather_samples(state, subkey, jnp.asarray(current_lambda))

    for step in trange(args.iters):
        current_lambda = args.beta * p

        key, subkey = jax.random.split(key)
        state = train_iter(state, subkey, jnp.asarray(current_lambda))

        key, k_eval, k_metric = jax.random.split(key, 3)
        samples = np.asarray(sample_model(state, k_eval, args.n_eval_samples))
        _ = model_log_density(state, jnp.asarray(samples))

        sample_mean = float(
            np.mean(np.asarray(target4_energy_values(jnp.asarray(samples), target_params)))
        )
        forward_kl, reverse_kl, wasserstein = compute_target4_metrics(
            samples,
            current_lambda,
            target_params=target_params,
            num_bins=args.metric_num_bins,
            key=k_metric,
        )
        key, k_sink = jax.random.split(key)
        samples_jax = jnp.asarray(samples)
        sinkhorn_target = sample_truncated_exponential(
            k_sink,
            current_lambda,
            samples.shape,
            target_params=target_params,
        )
        n_sink = min(args.sinkhorn_num_samples, samples.shape[0])
        sinkhorn = sinkhorn_distance(samples_jax[:n_sink], sinkhorn_target[:n_sink])
        ess = effective_sample_size_from_log_weights(
            target4_logprob(samples_jax, current_lambda, target_params=target_params)
        )
        energy_w2 = float(
            energy_wasserstein_1d(
                samples_jax[:n_sink],
                sinkhorn_target[:n_sink],
                current_lambda,
                target_params=target_params,
            )
        )
        interatomic_w2 = float(
            interatomic_wasserstein_1d(
                samples_jax[:n_sink],
                sinkhorn_target[:n_sink],
                n_particles=args.n_particles,
                n_spatial_dim=args.n_spatial_dim,
            )
        )
        optimal_p, target_mean = optimal_p_from_target_mean(
            current_lambda,
            args.tau,
            args.safe_q,
            target_params=target_params,
        )

        hist["target4/p"].append(float(p))
        hist["target4/lambda"].append(float(current_lambda))
        hist["target4/sample_mean"].append(sample_mean)
        hist["target4/forward_kl"].append(forward_kl)
        hist["target4/reverse_kl"].append(reverse_kl)
        hist["target4/wasserstein"].append(wasserstein)
        hist["target4/sinkhorn"].append(sinkhorn)
        hist["target4/ess"].append(ess)
        hist["target4/energy_w2"].append(energy_w2)
        hist["target4/interatomic_w2"].append(interatomic_w2)
        hist["target4/target_mean"].append(target_mean)
        hist["target4/optimal_p"].append(optimal_p)
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

    key, k_final = jax.random.split(key)
    x_final = np.asarray(sample_model(state, k_final, 2**14))
    samples_path = save_dir / f"gmmvi_samples_{run_tag}.npy"
    history_path = save_dir / f"gmmvi_target4_history_{run_tag}.npz"
    metrics_path = save_dir / f"gmmvi_target4_metrics_{run_tag}.png"
    final_samples_path = save_dir / f"gmmvi_target4_final_samples_{run_tag}.png"
    np.save(samples_path.as_posix(), x_final)
    np.savez(
        history_path.as_posix(),
        **{k: np.asarray(v, dtype=np.float64) for k, v in hist.items()},
    )

    final_p = float(p)
    final_lambda = float(args.beta * final_p)
    final_forward_kl = float(hist["target4/forward_kl"][-1])
    final_reverse_kl = float(hist["target4/reverse_kl"][-1])
    final_wasserstein = float(hist["target4/wasserstein"][-1])
    final_sinkhorn = float(hist["target4/sinkhorn"][-1])
    final_ess = float(hist["target4/ess"][-1])
    final_energy_w2 = float(hist["target4/energy_w2"][-1])
    final_optimal_p = float(hist["target4/optimal_p"][-1])

    save_metric_plot(hist, metrics_path)
    if args.dim >= 2:
        save_final_sample_plot(
            x_final,
            final_lambda,
            final_samples_path,
            target_params=target_params,
        )

    print(f"Saved outputs to: {save_dir}")
    print(f"dimension d: {args.dim}")
    print(f"target4 c: {float(target_params.c):.6f}")
    print(f"target4 a: {np.asarray(target_params.a)}")
    print(f"target4 k: {np.asarray(target_params.k)}")
    print(f"target4 phi: {np.asarray(target_params.phi)}")
    print(f"safe q: {args.safe_q:.6f}")
    print(f"p update frequency: {args.p_update_freq} (0 means fixed p)")
    print(f"p ema alpha: {args.p_ema_alpha:.3f}")
    print(f"p jump probability: {args.p_jump_prob:.3f}")
    print(f"final p: {final_p:.6f}")
    print(f"final lambda = beta * p: {final_lambda:.6f}")
    print(f"final forward KL: {final_forward_kl:.6f}")
    print(f"final reverse KL: {final_reverse_kl:.6f}")
    print(f"final Wasserstein: {final_wasserstein:.6f}")
    print(f"final Sinkhorn: {final_sinkhorn:.6f}")
    print(f"final ESS: {final_ess:.6f}")
    print(f"final Energy W2: {final_energy_w2:.6f}")
    print(f"final target optimal p: {final_optimal_p:.6f}")
    print(f"final num components: {hist['model/num_components'][-1]}")

    if args.show:
        metrics = plt.imread(metrics_path.as_posix())
        plt.figure(figsize=(12, 4))
        plt.imshow(metrics)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
