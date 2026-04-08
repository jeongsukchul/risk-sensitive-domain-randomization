from __future__ import annotations

import argparse
import functools
from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import i0e, i1e


class Target4HarmonicParams(NamedTuple):
    c: jax.Array
    a: jax.Array
    k: jax.Array
    phi: jax.Array


def make_default_target4_harmonic_params(
    dim: int,
    *,
    amplitude_budget: float = 0.25,
) -> Target4HarmonicParams:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if amplitude_budget <= 0.0 or amplitude_budget > 0.5:
        raise ValueError(
            f"amplitude_budget must lie in (0, 0.5], got {amplitude_budget}"
        )

    idx = np.arange(dim, dtype=np.float32)
    weights = np.ones(dim, dtype=np.float32) / dim
    a = amplitude_budget * weights
    if dim == 1:
        k = np.asarray([4.0], dtype=np.float32)
        k = np.asarray([1.0], dtype=np.float32)
    else:
        k = np.linspace(4.0, 6.0, dim, dtype=np.float32)
        k = np.linspace(1.0, 2.0, dim, dtype=np.float32)
    phi = np.zeros(dim, dtype=np.float32)

    return Target4HarmonicParams(
        c=jnp.asarray(0.75, dtype=jnp.float32),
        a=jnp.asarray(a, dtype=jnp.float32),
        k=jnp.asarray(k, dtype=jnp.float32),
        phi=jnp.asarray(phi, dtype=jnp.float32),
    )


def _validate_target4_params(
    params: Target4HarmonicParams,
    dim: int,
) -> Target4HarmonicParams:
    for name in ("a", "k", "phi"):
        value = jnp.asarray(getattr(params, name), dtype=jnp.float32).reshape(-1)
        if value.shape[0] != dim:
            raise ValueError(f"{name} must have length {dim}, got {value.shape[0]}")

    total_amplitude = float(jnp.sum(jnp.abs(jnp.asarray(params.a, dtype=jnp.float32))))
    c = float(jnp.asarray(params.c, dtype=jnp.float32))
    if total_amplitude > min(c, 1.0 - c) + 1e-6:
        raise ValueError(
            "Target4 cosine amplitudes must satisfy "
            f"sum_i |a_i| <= min(c, 1-c), got {total_amplitude:.6f} with c={c:.6f}"
        )

    return Target4HarmonicParams(
        c=jnp.asarray(params.c, dtype=jnp.float32),
        a=jnp.asarray(params.a, dtype=jnp.float32).reshape(dim),
        k=jnp.asarray(params.k, dtype=jnp.float32).reshape(dim),
        phi=jnp.asarray(params.phi, dtype=jnp.float32).reshape(dim),
    )


def get_target4_harmonic_params(
    dim: int,
    params: Target4HarmonicParams | None = None,
) -> Target4HarmonicParams:
    if params is None:
        return _validate_target4_params(make_default_target4_harmonic_params(dim), dim)
    return Target4HarmonicParams(
        c=jnp.asarray(params.c, dtype=jnp.float32),
        a=jnp.asarray(params.a, dtype=jnp.float32).reshape(dim),
        k=jnp.asarray(params.k, dtype=jnp.float32).reshape(dim),
        phi=jnp.asarray(params.phi, dtype=jnp.float32).reshape(dim),
    )


def target4_component_values(
    samples: jax.Array,
    params: Target4HarmonicParams,
) -> jax.Array:
    x = jnp.asarray(samples, dtype=jnp.float32)
    return params.a * jnp.cos((2.0 * jnp.pi * params.k * x) + params.phi)


def target4_energy_values(
    samples: jax.Array,
    params: Target4HarmonicParams,
) -> jax.Array:
    x = jnp.asarray(samples, dtype=jnp.float32)
    return params.c + jnp.sum(target4_component_values(x, params), axis=-1)


def target4_log_normalizer(
    lam: jax.Array,
    params: Target4HarmonicParams,
) -> jax.Array:
    lam = jnp.asarray(lam, dtype=jnp.float32)
    scaled = lam * params.a
    return (lam * params.c) + jnp.sum(jnp.abs(scaled) + jnp.log(i0e(scaled)))


def target4_expected_energy(
    lam: jax.Array,
    params: Target4HarmonicParams,
) -> jax.Array:
    lam = jnp.asarray(lam, dtype=jnp.float32)
    scaled = lam * params.a
    return params.c + jnp.sum(params.a * (i1e(scaled) / i0e(scaled)))


def target4_log_partition(
    lam: jax.Array,
    params: Target4HarmonicParams,
) -> jax.Array:
    return target4_log_normalizer(lam, params)


def target4_unsafe_action_value(
    lam: jax.Array,
    params: Target4HarmonicParams,
) -> jax.Array:
    return target4_expected_energy(lam, params)


def _sample_single_target4_dim(
    theta_key: jax.Array,
    copy_key: jax.Array,
    lam: jax.Array,
    a_i: jax.Array,
    k_i: jax.Array,
    phi_i: jax.Array,
    n_samples: int,
) -> jax.Array:
    scaled = lam * a_i
    loc = jnp.where(scaled >= 0.0, phi_i, phi_i + jnp.pi)
    concentration = jnp.abs(scaled)
    theta = distrax.VonMises(loc=loc, concentration=concentration).sample(
        seed=theta_key,
        sample_shape=(n_samples,),
    )
    k_int = jnp.maximum(1, jnp.asarray(jnp.round(k_i), dtype=jnp.int32))
    copy_idx = jax.random.randint(
        copy_key,
        shape=(n_samples,),
        minval=0,
        maxval=k_int,
    )
    x = (theta / (2.0 * jnp.pi) + copy_idx.astype(jnp.float32)) / k_i
    return jnp.mod(x, 1.0)


@functools.partial(jax.jit, static_argnames=("n_samples",))
def _sample_target4_product_jitted(
    key: jax.Array,
    lam: jax.Array,
    params: Target4HarmonicParams,
    n_samples: int,
) -> jax.Array:
    dim = params.a.shape[0]
    key_theta, key_copy = jax.random.split(key)
    theta_keys = jax.random.split(key_theta, dim)
    copy_keys = jax.random.split(key_copy, dim)
    samples_by_dim = jax.vmap(
        _sample_single_target4_dim,
        in_axes=(0, 0, None, 0, 0, 0, None),
        out_axes=0,
    )(
        theta_keys,
        copy_keys,
        jnp.asarray(lam, dtype=jnp.float32),
        params.a,
        params.k,
        params.phi,
        n_samples,
    )
    return jnp.swapaxes(samples_by_dim, 0, 1)


def sample_target4_product(
    key: jax.Array,
    lam: jax.Array,
    shape: tuple[int, int],
    params: Target4HarmonicParams,
) -> jax.Array:
    if len(shape) != 2:
        raise ValueError(f"shape must be rank-2, got {shape}")
    n_samples, dim = shape
    if dim != params.a.shape[0]:
        raise ValueError(f"shape dim {dim} does not match params dim {params.a.shape[0]}")
    return _sample_target4_product_jitted(
        key,
        jnp.asarray(lam, dtype=jnp.float32),
        params,
        n_samples,
    )


def add_target4_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target4-c",
        type=float,
        default=0.75,
        help="Constant offset c in g(x)=c+sum_i a_i cos(2pi k_i x_i + phi_i).",
    )
    parser.add_argument(
        "--target4-a",
        type=str,
        default=None,
        help="Comma-separated amplitudes a_i. Must have length dim.",
    )
    parser.add_argument(
        "--target4-k",
        type=str,
        default=None,
        help="Comma-separated frequencies k_i. Must have length dim.",
    )
    parser.add_argument(
        "--target4-phi",
        type=str,
        default=None,
        help="Comma-separated phases phi_i in radians. Must have length dim.",
    )
    parser.add_argument(
        "--target4-amplitude-budget",
        type=float,
        default=0.25,
        help="Used only when --target4-a is omitted.",
    )


def _parse_target4_vector(raw: str | None, dim: int, name: str) -> np.ndarray | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if len(values) != dim:
        raise ValueError(f"{name} must have exactly {dim} comma-separated values, got {len(values)}")
    return np.asarray([float(v) for v in values], dtype=np.float32)


def build_target4_params_from_args(
    args: argparse.Namespace,
    dim: int,
) -> Target4HarmonicParams:
    default = make_default_target4_harmonic_params(
        dim,
        amplitude_budget=float(getattr(args, "target4_amplitude_budget", 0.45)),
    )
    c = getattr(args, "target4_c", None)
    a = _parse_target4_vector(getattr(args, "target4_a", None), dim, "target4-a")
    k = _parse_target4_vector(getattr(args, "target4_k", None), dim, "target4-k")
    phi = _parse_target4_vector(getattr(args, "target4_phi", None), dim, "target4-phi")
    params = Target4HarmonicParams(
        c=default.c if c is None else jnp.asarray(c, dtype=jnp.float32),
        a=default.a if a is None else jnp.asarray(a, dtype=jnp.float32),
        k=default.k if k is None else jnp.asarray(k, dtype=jnp.float32),
        phi=default.phi if phi is None else jnp.asarray(phi, dtype=jnp.float32),
    )
    return _validate_target4_params(params, dim)
