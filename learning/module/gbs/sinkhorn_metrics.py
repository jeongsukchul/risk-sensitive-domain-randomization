from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.solvers import linear

from learning.module.gbs.targets.target_family import (
    TargetHarmonicParams,
    TargetShiftedHarmonicParams,
    get_target_harmonic_params,
    target_energy_values as harmonic_target_energy_values,
)


@functools.partial(jax.jit, static_argnames=("epsilon",))
def _sinkhorn_distance_jitted(
    x: jax.Array,
    y: jax.Array,
    w_x: jax.Array,
    w_y: jax.Array,
    *,
    epsilon: float = 1e-3,
) -> jax.Array:
    geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
    out = linear.solve(geom, a=w_x, b=w_y)
    return out.primal_cost


def should_compute_interatomic_w2(n_particles: int, max_pairs: int = 4096) -> bool:
    if n_particles <= 1:
        return False
    return (n_particles * (n_particles - 1)) // 2 <= max_pairs


def sinkhorn_distance(
    x,
    y,
    p: int = 2,
    w_x=None,
    w_y=None,
    eps: float = 1e-3,
    max_iters: int = 100,
    stop_thresh: float = 1e-5,
):
    """Approximate Sinkhorn transport cost using `ott-jax`.

    Notes:
    - OTT's `PointCloud` geometry uses the squared Euclidean ground cost by default.
    - `p`, `max_iters`, and `stop_thresh` are kept for backward-compatible call sites;
      `p` must remain `2` for this OTT-backed implementation.
    """
    del max_iters, stop_thresh
    if p != 2:
        raise ValueError("OTT-backed sinkhorn_distance currently supports only p=2.")
    if eps <= 0:
        raise ValueError("eps must be > 0")

    x = jnp.asarray(x, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"x and y must be rank-2 arrays, got {x.shape} and {y.shape}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"x and y must have the same feature dimension, got {x.shape} and {y.shape}")

    if w_x is None:
        w_x = jnp.ones(x.shape[0], dtype=jnp.float32) / max(x.shape[0], 1)
    else:
        w_x = jnp.asarray(w_x, dtype=jnp.float32).reshape(-1)
    if w_y is None:
        w_y = jnp.ones(y.shape[0], dtype=jnp.float32) / max(y.shape[0], 1)
        w_y = w_y * (jnp.sum(w_x) / jnp.maximum(jnp.sum(w_y), 1e-12))
    else:
        w_y = jnp.asarray(w_y, dtype=jnp.float32).reshape(-1)

    if w_x.shape[0] != x.shape[0] or w_y.shape[0] != y.shape[0]:
        raise ValueError("Weight shapes must match the number of points")
    if float(jnp.abs(jnp.sum(w_x) - jnp.sum(w_y))) > 1e-5:
        raise ValueError("w_x and w_y must sum to the same value")

    return float(_sinkhorn_distance_jitted(x, y, w_x, w_y, epsilon=eps))


@jax.jit
def _effective_sample_size_from_log_weights_jitted(log_weights: jax.Array) -> jax.Array:
    stable = log_weights - jnp.max(log_weights)
    weights = jnp.exp(stable)
    denom = jnp.sum(weights ** 2)
    return jnp.where(denom > 0.0, (jnp.sum(weights) ** 2) / denom, 0.0)


def effective_sample_size_from_log_weights(log_weights) -> float:
    log_w = jnp.asarray(log_weights, dtype=jnp.float32).reshape(-1)
    if log_w.size == 0:
        return 0.0
    return float(_effective_sample_size_from_log_weights_jitted(log_w))


@functools.partial(jax.jit, static_argnames=("n_particles", "n_spatial_dim"))
def interatomic_dist(samples: jax.Array, n_particles: int, n_spatial_dim: int) -> jax.Array:
    samples = jnp.asarray(samples, dtype=jnp.float32)
    reshaped = samples.reshape(samples.shape[0], n_particles, n_spatial_dim)
    diffs = reshaped[:, :, None, :] - reshaped[:, None, :, :]
    dists = jnp.sqrt(jnp.sum(diffs ** 2, axis=-1))
    triu_i, triu_j = jnp.triu_indices(n_particles, k=1)
    return dists[:, triu_i, triu_j]


@jax.jit
def emd2_1d_uniform(x: jax.Array, y: jax.Array) -> jax.Array:
    x = jnp.sort(jnp.asarray(x, dtype=jnp.float32).reshape(-1))
    y = jnp.sort(jnp.asarray(y, dtype=jnp.float32).reshape(-1))
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"emd2_1d_uniform expects equal-sized inputs, got {x.shape[0]} and {y.shape[0]}")
    return jnp.mean((x - y) ** 2)


@jax.jit
def target_energy_values(
    samples: jax.Array,
    lam: jax.Array,
    target_params: TargetHarmonicParams | TargetShiftedHarmonicParams | None = None,
    policy_p: jax.Array | float | None = None,
) -> jax.Array:
    samples = jnp.asarray(samples, dtype=jnp.float32)
    lam = jnp.asarray(lam, dtype=jnp.float32)
    if policy_p is None:
        params = get
    else:
        params = get_target_harmonic_params(samples.shape[-1], target_params, policy_p=policy_p)
    return lam * harmonic_target_energy_values(samples, params, policy_p=policy_p)


@functools.partial(jax.jit, static_argnames=("n_particles", "n_spatial_dim"))
def interatomic_wasserstein_1d(
    samples: jax.Array,
    ref_samples: jax.Array,
    n_particles: int,
    n_spatial_dim: int,
) -> jax.Array:
    gen_dist = interatomic_dist(samples, n_particles=n_particles, n_spatial_dim=n_spatial_dim)
    ref_dist = interatomic_dist(ref_samples, n_particles=n_particles, n_spatial_dim=n_spatial_dim)
    return emd2_1d_uniform(gen_dist, ref_dist)


@jax.jit
def energy_wasserstein_1d(
    samples: jax.Array,
    ref_samples: jax.Array,
    lam: jax.Array,
    target_params: TargetHarmonicParams | TargetShiftedHarmonicParams | None = None,
    policy_p: jax.Array | float | None = None,
) -> jax.Array:
    gen_energy = target_energy_values(samples, lam, target_params=target_params, policy_p=policy_p)
    ref_energy = target_energy_values(ref_samples, lam, target_params=target_params, policy_p=policy_p)
    return emd2_1d_uniform(gen_energy, ref_energy)
