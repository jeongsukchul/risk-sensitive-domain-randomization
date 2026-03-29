from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


@functools.partial(jax.jit, static_argnames=("p", "max_iters"))
def _sinkhorn_distance_jitted(
    x: jax.Array,
    y: jax.Array,
    w_x: jax.Array,
    w_y: jax.Array,
    *,
    p: int = 2,
    eps: float = 1e-3,
    max_iters: int = 100,
    stop_thresh: float = 1e-5,
) -> jax.Array:
    if p == 1:
        dist = jnp.sum(jnp.abs(x[:, None, :] - y[None, :, :]), axis=-1)
    else:
        dist = jnp.sum(jnp.abs(x[:, None, :] - y[None, :, :]) ** p, axis=-1) ** (1.0 / p)

    log_a = jnp.log(jnp.clip(w_x, 1e-12))
    log_b = jnp.log(jnp.clip(w_y, 1e-12))
    u0 = jnp.zeros_like(w_x)
    v0 = eps * log_b

    def cond_fn(state):
        i, u, v, prev_u, prev_v = state
        max_err_u = jnp.max(jnp.abs(u - prev_u))
        max_err_v = jnp.max(jnp.abs(v - prev_v))
        return jnp.logical_and(i < max_iters, jnp.logical_or(max_err_u >= stop_thresh, max_err_v >= stop_thresh))

    def body_fn(state):
        i, u, v, _, _ = state
        prev_u = u
        prev_v = v
        u = eps * (log_a - logsumexp((-dist + v[None, :]) / eps, axis=1))
        v = eps * (log_b - logsumexp((-dist + u[:, None]) / eps, axis=0))
        return i + 1, u, v, prev_u, prev_v

    init_state = (0, u0, v0, jnp.full_like(u0, jnp.inf), jnp.full_like(v0, jnp.inf))
    _, u, v, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)

    transport = jnp.exp(jnp.clip((-dist + u[:, None] + v[None, :]) / eps, a_max=50.0))
    return jnp.sum(transport * dist)


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
    """Approximate p-Wasserstein distance with JAX using the same defaults as scalable-pytorch-sinkhorn."""
    if not isinstance(p, int) or p <= 0:
        raise ValueError(f"p must be a positive integer, got {p}")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0")

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

    return float(
        _sinkhorn_distance_jitted(
            x,
            y,
            w_x,
            w_y,
            p=p,
            eps=eps,
            max_iters=max_iters,
            stop_thresh=stop_thresh,
        )
    )


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
def target4_energy_values(samples: jax.Array, lam: jax.Array) -> jax.Array:
    samples = jnp.asarray(samples, dtype=jnp.float32)
    lam = jnp.asarray(lam, dtype=jnp.float32)
    return lam * jnp.sum(samples, axis=-1)


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
) -> jax.Array:
    gen_energy = target4_energy_values(samples, lam)
    ref_energy = target4_energy_values(ref_samples, lam)
    return emd2_1d_uniform(gen_energy, ref_energy)
