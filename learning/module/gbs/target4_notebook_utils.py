import jax
import jax.numpy as jnp
import numpy as np
import distrax
import optax
import imageio
import matplotlib.pyplot as plt
from pathlib import Path
from flax.training import train_state
from tqdm import trange

from learning.module.gbs.gbs_loss import (
    VP,
    rnd_no_target,
    rnd_time_reversal_lv_no_target,
    lv_loss_from_values,
    lv_loss_from_rnd,
)
from learning.module.gbs.gbs_trainer import make_gbs_model


def tanh_box_bijector(z, low, high):
    half = 0.5 * (high - low)
    mid = 0.5 * (high + low)
    return mid + half * jnp.tanh(z)


def tanh_box_logabsdet(z, low, high):
    z = jnp.atleast_2d(z)
    half = 0.5 * (high - low)
    jac_diag = half * (1.0 - jnp.tanh(z) ** 2)
    return jnp.sum(jnp.log(jnp.clip(jac_diag, 1e-12)), axis=-1)


def truncated_exponential_logprob_1d(x, lam):
    x = jnp.asarray(x)
    lam = jnp.asarray(lam)
    safe = jnp.abs(lam) > 1e-6
    log_norm = jnp.where(
        safe,
        jnp.log(jnp.abs(lam)) - jnp.log(jnp.abs(jnp.expm1(lam))),
        0.0,
    )
    logp = jnp.where(safe, log_norm + lam * x, 0.0)
    return jnp.where((x >= 0.0) & (x <= 1.0), logp, -jnp.inf)


def target4_logprob(z, lam):
    z = jnp.atleast_2d(z)
    return jnp.sum(truncated_exponential_logprob_1d(z, lam), axis=-1).squeeze()


def update_p_from_samples(x, tau):
    sample_mean = jnp.mean(x)
    p = jax.nn.sigmoid((sample_mean - 1.0) / tau)
    return p, sample_mean


def truncated_exponential_cdf(x, lam):
    x = np.asarray(x, dtype=np.float64)
    lam = float(lam)
    if abs(lam) <= 1e-6:
        return x
    return np.expm1(lam * x) / np.expm1(lam)


def sample_truncated_exponential(key, lam, shape):
    u = jax.random.uniform(key, shape=shape)
    lam = jnp.asarray(lam)
    safe = jnp.abs(lam) > 1e-6
    return jnp.where(safe, jnp.log1p(u * jnp.expm1(lam)) / lam, u)


def compute_target4_metrics(samples, lam, num_bins=128, eps=1e-8, key=None):
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    q_hist, _ = np.histogram(np.clip(samples, 0.0, 1.0), bins=edges, density=False)
    q_probs = q_hist.astype(np.float64)
    q_probs = q_probs / max(q_probs.sum(), 1.0)

    p_probs = truncated_exponential_cdf(edges[1:], lam) - truncated_exponential_cdf(edges[:-1], lam)
    p_probs = np.maximum(p_probs, eps)
    p_probs = p_probs / p_probs.sum()
    q_probs = np.maximum(q_probs, eps)
    q_probs = q_probs / q_probs.sum()

    forward_kl = float(np.sum(p_probs * (np.log(p_probs) - np.log(q_probs))))
    reverse_kl = float(np.sum(q_probs * (np.log(q_probs) - np.log(p_probs))))

    if key is None:
        ref = np.linspace(0.0, 1.0, samples.size, endpoint=False) + 0.5 / max(samples.size, 1)
    else:
        ref = np.asarray(sample_truncated_exponential(key, lam, (samples.size,)))
    wasserstein = float(
        np.mean(np.abs(np.sort(np.clip(samples, 0.0, 1.0)) - np.sort(np.clip(ref, 0.0, 1.0))))
    )
    return forward_kl, reverse_kl, wasserstein


def run_gbs_toy_target4(
    low=jnp.array([0.0, 0.0]),
    high=jnp.array([1.0, 1.0]),
    dim=2,
    T=1000,
    batch_size=512,
    num_steps=25,
    lr=1e-3,
    init_std=0.5,
    gif_path=None,
    snap_iters=None,
    seed=0,
    beta=1.0,
    tau=0.1,
    initial_p=None,
    p_update_freq=1,
    loss_mode="tr_lv",
    metric_num_bins=128,
    save_dir=".",
    model_type="pisgrad",
    model_num_layers=2,
    model_num_hid=64,
    final_sample_size=2**14,
):
    if snap_iters is None:
        snap_iters = []
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    proc = VP(
        diff_coeff_sq_min=0.1,
        diff_coeff_sq_max=10.0,
        scale_diff_coeff=1.0,
        terminal_t=1.0,
        generative=False,
        sign=-1.0,
        include_base_drift=True,
    )

    key = jax.random.PRNGKey(seed)
    key, k_p0 = jax.random.split(key)
    if initial_p is None:
        p = float(jax.random.uniform(k_p0, minval=0.0, maxval=1.0))
    else:
        p = float(np.clip(initial_p, 0.0, 1.0))

    low = jnp.asarray(low)
    high = jnp.asarray(high)
    if low.shape[0] != dim or high.shape[0] != dim:
        raise ValueError(f"low/high must match dim={dim}, got {low.shape} and {high.shape}")

    # Train in unconstrained latent space z and map to the box [low, high] via tanh.
    # This keeps the prior sampler and prior_log_prob consistent while respecting the
    # bounded target support.
    def to_box(z):
        return tanh_box_bijector(z, low=low, high=high)

    def target4_logprob_latent(z, lam):
        x_box = to_box(z)
        return target4_logprob(x_box, lam) + tanh_box_logabsdet(z, low=low, high=high)

    latent_prior_loc = jnp.zeros(dim, dtype=jnp.float32)
    process_center = jnp.zeros(dim, dtype=jnp.float32)

    prior = distrax.MultivariateNormalDiag(
        loc=latent_prior_loc,
        scale_diag=jnp.ones(dim) * init_std,
    )
    prior_sampler = lambda k: jnp.squeeze(prior.sample(seed=k, sample_shape=(1,)))
    prior_log_prob = prior.log_prob

    model_cfg = dict(
        model_type=model_type,
        dim=dim,
        num_layers=model_num_layers,
        num_hid=model_num_hid,
    )
    fwd_model = make_gbs_model(**model_cfg)
    bwd_model = make_gbs_model(**model_cfg)

    key, k1, k2 = jax.random.split(key, 3)
    fwd_params = fwd_model.init(
        k1, jnp.ones([batch_size, dim]), jnp.ones([batch_size, 1]), jnp.ones([batch_size, dim])
    )
    bwd_params = bwd_model.init(
        k2, jnp.ones([batch_size, dim]), jnp.ones([batch_size, 1]), jnp.ones([batch_size, dim])
    )

    opt = optax.chain(optax.zero_nans(), optax.clip(1.0), optax.adam(lr))
    fwd_state = train_state.TrainState.create(apply_fn=fwd_model.apply, params=fwd_params, tx=opt)
    bwd_state = train_state.TrainState.create(apply_fn=bwd_model.apply, params=bwd_params, tx=opt)

    if loss_mode == "tr_lv":
        rnd_jit = jax.jit(rnd_time_reversal_lv_no_target, static_argnums=(3, 4, 5, 6, 7))

        def loss_wrapped(key, model_state, fwd_params, bwd_params, lam):
            x0, xT, rnd_running = rnd_time_reversal_lv_no_target(
                key,
                model_state,
                fwd_params,
                batch_size,
                prior_sampler,
                num_steps,
                proc,
                True,
                process_center=process_center,
            )
            target_lp_vals = jnp.asarray(target4_logprob_latent(xT, lam)).reshape(-1)
            rnd_total = prior_log_prob(x0) + rnd_running - target_lp_vals
            loss, aux, _ = lv_loss_from_rnd(rnd_total, xT=xT)
            return loss, aux

        loss_grad = jax.jit(jax.grad(loss_wrapped, (2, 3), has_aux=True))
        hist = {k: [] for k in ["train/rnd_mean", "train/rnd_var", "train/xT_mean_norm"]}
    elif loss_mode == "dis":
        rnd_jit = jax.jit(rnd_no_target, static_argnums=(4, 5, 6, 7, 8))

        def loss_wrapped(key, model_state, fwd_params, bwd_params, lam):
            x0, xT, log_ratio = rnd_no_target(
                key,
                model_state,
                fwd_params,
                bwd_params,
                batch_size,
                prior_sampler,
                num_steps,
                proc,
                True,
                process_center=process_center,
            )
            target_lp_vals = jnp.asarray(target4_logprob_latent(xT, lam)).reshape(-1)
            loss, aux = lv_loss_from_values(x0, xT, log_ratio, prior_log_prob, target_lp_vals)
            return loss, aux

        loss_grad = jax.jit(jax.grad(loss_wrapped, (2, 3), has_aux=True))
        hist = {k: [] for k in [
            "train/neg_elbo_mean",
            "train/neg_elbo_var",
            "train/running_mean",
            "train/terminal_mean",
            "train/xT_mean_norm",
        ]}
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")

    hist.update(
        {
            "target4/p": [],
            "target4/lambda": [],
            "target4/sample_mean": [],
            "target4/forward_kl": [],
            "target4/reverse_kl": [],
            "target4/wasserstein": [],
            "target4/p_updated": [],
        }
    )

    frames = []

    for t in trange(T):
        current_lambda = beta * p / dim
        key, k_step = jax.random.split(key)
        model_state = (fwd_state, bwd_state)

        if loss_mode == "tr_lv":
            x0, xT_latent, _rnd_running = rnd_jit(
                k_step, model_state, fwd_state.params,
                batch_size, prior_sampler, num_steps, proc, True,
                process_center=process_center,
            )
        else:
            x0, xT_latent, _log_ratio = rnd_jit(
                k_step, model_state, fwd_state.params, bwd_state.params,
                batch_size, prior_sampler, num_steps, proc, True,
                process_center=process_center,
            )
        xT = to_box(xT_latent)

        (fwd_grads, bwd_grads), aux = loss_grad(
            k_step, model_state, fwd_state.params, bwd_state.params, jnp.asarray(current_lambda)
        )
        fwd_state = fwd_state.apply_gradients(grads=fwd_grads)
        bwd_state = bwd_state.apply_gradients(grads=bwd_grads)

        for k in list(aux.keys()):
            if k in hist:
                hist[k].append(float(aux[k]))

        sample_mean = float(jnp.mean(xT))
        key, k_metric = jax.random.split(key)
        forward_kl, reverse_kl, wasserstein = compute_target4_metrics(
            xT, current_lambda, num_bins=metric_num_bins, key=k_metric
        )

        hist["target4/p"].append(float(p))
        hist["target4/lambda"].append(float(current_lambda))
        hist["target4/sample_mean"].append(sample_mean)
        hist["target4/forward_kl"].append(forward_kl)
        hist["target4/reverse_kl"].append(reverse_kl)
        hist["target4/wasserstein"].append(wasserstein)
        should_update_p = p_update_freq > 0 and ((t + 1) % p_update_freq == 0)
        hist["target4/p_updated"].append(float(should_update_p))
        if should_update_p:
            p = float(jax.nn.sigmoid((sample_mean - 1.0) / tau))

        if gif_path and (t in snap_iters):
            frames.append(np.asarray(xT))

    if gif_path and frames:
        rendered = []
        for idx, pts in enumerate(frames):
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.25, c="r")
            ax.set_xlim(float(low[0]), float(high[0]))
            ax.set_ylim(float(low[1]), float(high[1]))
            ax.set_aspect("equal")
            ax.set_title(f"GBS target4 snapshot {idx}")
            fig.tight_layout()
            fig.canvas.draw()
            rendered.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            plt.close(fig)
        imageio.mimsave(gif_path, rendered, fps=4)

    key, k_final = jax.random.split(key)
    B = 2**14
    if loss_mode == "tr_lv":
        _, xT_final_latent, _ = rnd_time_reversal_lv_no_target(
            k_final, (fwd_state, bwd_state), fwd_state.params,
            final_sample_size, prior_sampler, num_steps, proc, True, process_center=process_center
        )
    else:
        _, xT_final_latent, _ = rnd_no_target(
            k_final, (fwd_state, bwd_state), fwd_state.params, bwd_state.params,
            final_sample_size, prior_sampler, num_steps, proc, True, process_center=process_center
        )
    xT_final = to_box(xT_final_latent)
    np.save((save_dir / "gbs_samples.npy").as_posix(), np.array(xT_final))

    return fwd_state, bwd_state, hist, np.asarray(xT_final)
