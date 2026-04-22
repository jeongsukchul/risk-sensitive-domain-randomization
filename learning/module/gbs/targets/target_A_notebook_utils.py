import jax
import jax.numpy as jnp
import numpy as np

from learning.module.gbs.sinkhorn_metrics import emd2_1d_uniform

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


def target_logprob(z, lam, target_params=None, policy_p=None):
    del target_params, policy_p
    z = jnp.atleast_2d(z)
    return jnp.sum(truncated_exponential_logprob_1d(z, lam), axis=-1).squeeze()


def update_p_from_samples(x, tau):
    sample_mean = jnp.mean(x)
    p = jax.nn.sigmoid((sample_mean - 1.0) / tau)
    return p, sample_mean


def update_p_with_ema_and_jump(
    prev_p,
    sample_mean=None,
    tau=0.1,
    ema_alpha=0.9,
    jump_prob=0.0,
    key=None,
    sample_mean_g=None,
    q=1.0,
):
    if sample_mean is None:
        sample_mean = sample_mean_g
    base_p = float(jax.nn.sigmoid(tau * (sample_mean - q)))
    ema_p = float(np.clip(ema_alpha * prev_p + (1.0 - ema_alpha) * base_p, 0.0, 1.0))
    key_jump, key_uniform = jax.random.split(key)
    jumped = bool(jax.random.bernoulli(key_jump, p=jump_prob))
    if jumped:
        new_p = float(jax.random.uniform(key_uniform, minval=0.0, maxval=1.0))
    else:
        new_p = ema_p
    return new_p, base_p, ema_p, jumped


def truncated_exponential_cdf(x, lam):
    x = np.asarray(x, dtype=np.float64)
    lam = float(lam)
    if abs(lam) <= 1e-6:
        return x
    return np.expm1(lam * x) / np.expm1(lam)


def truncated_exponential_mean(lam):
    lam = np.asarray(lam, dtype=np.float64)
    out = np.empty_like(lam, dtype=np.float64)
    small = np.abs(lam) <= 1e-6
    out[small] = 0.5
    ls = lam[~small]
    out[~small] = np.exp(ls) / np.expm1(ls) - 1.0 / ls
    if out.ndim == 0:
        return float(out)
    return out


def optimal_p_from_target_mean(lam, tau, q=1.0, target_params=None):
    del target_params
    target_mean = truncated_exponential_mean(lam)
    optimal_p = 1.0 / (1.0 + np.exp(-tau * (target_mean - q)))
    return float(optimal_p), float(target_mean)


def sample_truncated_exponential(key, lam, shape, target_params=None, policy_p=None):
    del target_params, policy_p
    u = jax.random.uniform(key, shape=shape)
    lam = jnp.asarray(lam)
    safe = jnp.abs(lam) > 1e-6
    return jnp.where(safe, jnp.log1p(u * jnp.expm1(lam)) / lam, u)


def compute_target_metrics(samples, lam, target_params=None, num_bins=128, eps=1e-8, key=None, policy_p=None):
    del target_params, policy_p
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 2:
        samples = samples.reshape(-1)
    else:
        samples = samples.reshape(-1)
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


def target_A_energy_wasserstein_1d(samples, ref_samples, lam):
    samples = jnp.asarray(samples, dtype=jnp.float32)
    ref_samples = jnp.asarray(ref_samples, dtype=jnp.float32)
    lam = jnp.asarray(lam, dtype=jnp.float32)
    gen_energy = lam * samples.reshape(-1)
    ref_energy = lam * ref_samples.reshape(-1)
    return emd2_1d_uniform(gen_energy, ref_energy)


def run_gbs_toy_target(
    low=(0.0, 0.0),
    high=(1.0, 1.0),
    dim=2,
    T=None,
    function_evaluations=None,
    buffer_size=50000,
    batch_size=None,
    num_steps=50,
    lr=5e-4,
    init_std=0.5,
    gif_path=None,
    snap_iters=None,
    seed=0,
    beta=1.0,
    tau=0.1,
    q=None,
    initial_p=None,
    p_update_freq=1,
    p_ema_alpha=0.9,
    p_jump_prob=0.0,
    loss_mode="tr_lv",
    metric_num_bins=128,
    sinkhorn_num_samples=256,
    n_particles=None,
    n_spatial_dim=1,
    save_dir=".",
    use_tanh_bijection=True,
    model_type="pisgrad",
    model_num_layers=6,
    model_num_hid=256,
    final_sample_size=2**14,
    max_rnd=1e8,
    trust_region_bound=0.1,
    trust_region_lambda_max=50.0,
    trust_region_lambda_grid_size=128,
    minibatch_size=2000,
    minibatch_steps=400,
    buffer_updates=None,
    target_params=None,
    return_snapshots: bool = False,
    snapshot_sample_size: int | None = None,
    max_metric_eval_points: int | None = None,
):
    del q, target_params
    if batch_size is not None:
        buffer_size = batch_size
    if buffer_updates is not None:
        minibatch_steps = buffer_updates
    if function_evaluations is None:
        function_evaluations = 50_000_000 if T is None else int(T) * int(buffer_size)
    if snap_iters is None:
        snap_iters = []
    if not use_tanh_bijection:
        raise ValueError("target_A currently requires use_tanh_bijection=True")

    low = jnp.asarray(low)
    high = jnp.asarray(high)
    proc = VP(
        diff_coeff_sq_min=0.1,
        diff_coeff_sq_max=10.0,
        scale_diff_coeff=1.0,
        terminal_t=1.0,
        generative=False,
        sign=-1.0,
        include_base_drift=True,
    )
    to_box = lambda z: tanh_box_bijector(z, low=low, high=high)
    target_logprob_latent = lambda z, lam, policy_p: (
        target_logprob(to_box(z), lam, target_params=None, policy_p=policy_p)
        + tanh_box_logabsdet(z, low=low, high=high)
    )

    return run_gbs(
        low=low,
        high=high,
        dim=dim,
        function_evaluations=int(function_evaluations),
        buffer_size=int(buffer_size),
        num_steps=num_steps,
        lr=lr,
        init_std=init_std,
        seed=seed,
        beta=beta / dim,
        tau=tau,
        q=1.0,
        initial_p=initial_p,
        p_update_freq=p_update_freq,
        p_ema_alpha=p_ema_alpha,
        p_jump_prob=p_jump_prob,
        loss_mode=loss_mode,
        sinkhorn_num_samples=sinkhorn_num_samples,
        n_particles=n_particles,
        n_spatial_dim=n_spatial_dim,
        save_dir=save_dir,
        gif_path=gif_path,
        snap_iters=snap_iters,
        model_type=model_type,
        model_num_layers=model_num_layers,
        model_num_hid=model_num_hid,
        final_sample_size=final_sample_size,
        max_rnd=max_rnd,
        trust_region_bound=trust_region_bound,
        trust_region_lambda_max=trust_region_lambda_max,
        trust_region_lambda_grid_size=trust_region_lambda_grid_size,
        minibatch_size=minibatch_size,
        minibatch_steps=minibatch_steps,
        return_snapshots=return_snapshots,
        snapshot_sample_size=snapshot_sample_size,
        max_metric_eval_points=max_metric_eval_points,
        process=proc,
        latent_prior_loc=jnp.zeros(dim, dtype=jnp.float32),
        process_center=jnp.zeros(dim, dtype=jnp.float32),
        clip_prior_without_tanh=False,
        to_box=to_box,
        target_logprob_latent_fn=target_logprob_latent,
        target_logprob_box_fn=lambda x, lam, policy_p: target_logprob(
            x, lam, target_params=None, policy_p=policy_p
        ),
        sample_mean_fn=lambda x, policy_p: jnp.mean(x),
        compute_metrics_fn=lambda x, lam, key, policy_p: compute_target_metrics(
            x, lam, target_params=None, num_bins=metric_num_bins, key=key, policy_p=policy_p
        ),
        sample_reference_fn=lambda key, lam, shape, policy_p: sample_truncated_exponential(
            key, lam, shape, target_params=None, policy_p=policy_p
        ),
        energy_w2_fn=lambda x, ref, lam, policy_p: target_A_energy_wasserstein_1d(
            x, ref, lam
        ),
        optimal_p_fn=lambda lam, tau, q: optimal_p_from_target_mean(
            lam, tau, q, target_params=None
        ),
        update_p_fn=update_p_with_ema_and_jump,
    )
