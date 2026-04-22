import jax
import jax.numpy as jnp
import numpy as np
from learning.module.gbs.targets.target_family import (
    TargetHarmonicParams,
    TargetShiftedHarmonicParams,
    get_target_harmonic_params,
    sample_target_product,
    target_energy_values,
    target_expected_energy,
    target_log_normalizer,
)


def tanh_box_bijector(z, low, high):
    half = 0.5 * (high - low)
    mid = 0.5 * (high + low)
    return mid + half * jnp.tanh(z)


def tanh_box_logabsdet(z, low, high):
    z = jnp.atleast_2d(z)
    half = 0.5 * (high - low)
    jac_diag = half * (1.0 - jnp.tanh(z) ** 2)
    return jnp.sum(jnp.log(jnp.clip(jac_diag, 1e-12)), axis=-1)


def target_logprob(
    z,
    lam,
    target_params: TargetHarmonicParams | TargetShiftedHarmonicParams | None = None,
    policy_p=None,
):
    z = jnp.atleast_2d(z)
    params = get_target_harmonic_params(z.shape[-1], target_params, policy_p=policy_p)
    log_unnorm = jnp.asarray(lam, dtype=jnp.float32) * target_energy_values(z, params, policy_p=policy_p)
    return (log_unnorm - target_log_normalizer(lam, params)).squeeze()


def update_p_from_samples(
    x,
    tau,
    q,
    target_params: TargetHarmonicParams | TargetShiftedHarmonicParams | None = None,
    policy_p=None,
):
    x = jnp.atleast_2d(jnp.asarray(x, dtype=jnp.float32))
    params = get_target_harmonic_params(x.shape[-1], target_params, policy_p=policy_p)
    sample_mean_g = jnp.mean(target_energy_values(x, params, policy_p=policy_p))
    p = jax.nn.sigmoid(tau * (sample_mean_g - q))
    return p, sample_mean_g


def update_p_with_ema_and_jump(prev_p, sample_mean_g, tau, q, ema_alpha, jump_prob, key):
    base_p = float(jax.nn.sigmoid(tau * (sample_mean_g - q)))
    ema_p = float(np.clip(ema_alpha * prev_p + (1.0 - ema_alpha) * base_p, 0.0, 1.0))
    key_jump, key_uniform = jax.random.split(key)
    jumped = bool(jax.random.bernoulli(key_jump, p=jump_prob))
    if jumped:
        new_p = float(jax.random.uniform(key_uniform, minval=0.0, maxval=1.0))
    else:
        new_p = ema_p
    return new_p, base_p, ema_p, jumped


def optimal_p_from_target_mean(
    lam,
    tau,
    q,
    target_params: TargetHarmonicParams | TargetShiftedHarmonicParams | None = None,
):
    params = get_target_harmonic_params(len(target_params.a) if target_params is not None else 1, target_params)
    target_mean = float(target_expected_energy(lam, params))
    optimal_p = 1.0 / (1.0 + np.exp(-tau * (target_mean - q)))
    return float(optimal_p), float(target_mean)


def sample_truncated_exponential(
    key,
    lam,
    shape,
    target_params: TargetHarmonicParams | TargetShiftedHarmonicParams | None = None,
    policy_p=None,
):
    if len(shape) != 2:
        raise ValueError(f"shape must be rank-2, got {shape}")
    params = get_target_harmonic_params(shape[1], target_params, policy_p=policy_p)
    return sample_target_product(key, lam, shape, params, policy_p=policy_p)


def compute_target_metrics(
    samples,
    lam,
    target_params: TargetHarmonicParams | TargetShiftedHarmonicParams | None = None,
    num_bins=128,
    eps=1e-8,
    key=None,
    policy_p=None,
):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(f"samples must be rank-2, got {samples.shape}")
    dim = samples.shape[1]
    params = get_target_harmonic_params(dim, target_params, policy_p=policy_p)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    grid = np.arange(num_bins, dtype=np.float64)
    mids = (grid + 0.5) / num_bins
    comp = np.asarray(
        target_energy_values(jnp.asarray(mids[:, None].repeat(dim, axis=1)), params, policy_p=policy_p)
    )
    del comp
    forward_terms = []
    reverse_terms = []
    for i in range(dim):
        q_hist, _ = np.histogram(np.clip(samples[:, i], 0.0, 1.0), bins=edges, density=False)
        q_probs = q_hist.astype(np.float64)
        q_probs = q_probs / max(q_probs.sum(), 1.0)

        x_mid = jnp.asarray(mids[:, None], dtype=jnp.float32)
        single_params = TargetHarmonicParams(
            c=jnp.asarray(0.0, dtype=jnp.float32),
            a=params.a[i : i + 1],
            k=params.k[i : i + 1],
            phi=params.phi[i : i + 1],
        )
        logw = np.asarray(
            target_logprob(
                x_mid, jnp.asarray(lam, dtype=jnp.float32), target_params=single_params, policy_p=policy_p
            )
        )
        p_probs = np.exp(logw - scipy_special_logsumexp_np(logw))
        p_probs = np.maximum(p_probs, eps)
        p_probs = p_probs / p_probs.sum()
        q_probs = np.maximum(q_probs, eps)
        q_probs = q_probs / q_probs.sum()

        forward_terms.append(np.sum(p_probs * (np.log(p_probs) - np.log(q_probs))))
        reverse_terms.append(np.sum(q_probs * (np.log(q_probs) - np.log(p_probs))))

    forward_kl = float(np.mean(forward_terms))
    reverse_kl = float(np.mean(reverse_terms))

    if key is None:
        ref = np.tile(
            (np.linspace(0.0, 1.0, samples.shape[0], endpoint=False) + 0.5 / max(samples.shape[0], 1))[:, None],
            (1, dim),
        )
    else:
        ref = np.asarray(sample_truncated_exponential(key, lam, samples.shape, params, policy_p=policy_p))
    wasserstein = float(
        np.mean(
            np.abs(
                np.sort(np.clip(samples, 0.0, 1.0), axis=0)
                - np.sort(np.clip(ref, 0.0, 1.0), axis=0)
            )
        )
    )
    return forward_kl, reverse_kl, wasserstein


def scipy_special_logsumexp_np(values: np.ndarray) -> float:
    vmax = float(np.max(values))
    return vmax + float(np.log(np.sum(np.exp(values - vmax))))

