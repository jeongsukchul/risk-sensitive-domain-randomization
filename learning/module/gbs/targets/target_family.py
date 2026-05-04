from __future__ import annotations

import argparse
import functools
import math
from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import i0e, i1e


class TargetHarmonicParams(NamedTuple):
    c: jax.Array
    a: jax.Array
    k: jax.Array
    phi: jax.Array


class TargetShiftedHarmonicParams(NamedTuple):
    c: jax.Array
    a: jax.Array
    k: jax.Array
    mu0: jax.Array
    b: jax.Array


def target1_logprob(x: jax.Array) -> jax.Array:
    x = jnp.atleast_2d(x)

    mean1 = jnp.array([1.0, 0.4], dtype=jnp.float32)
    cov1 = 0.3 * jnp.array([[1.0, 0.3], [0.3, 1.0]], dtype=jnp.float32)
    mean2 = jnp.array([-1.0, -0.4], dtype=jnp.float32)
    cov2 = 0.1 * jnp.array([[1.0, -0.3], [-0.3, 1.0]], dtype=jnp.float32)

    d1 = distrax.MultivariateNormalFullCovariance(mean1, cov1)
    d2 = distrax.MultivariateNormalFullCovariance(mean2, cov2)

    logw = jnp.log(jnp.array([0.4, 0.6], dtype=jnp.float32))
    lp = jnp.stack([d1.log_prob(x), d2.log_prob(x)], axis=-1)
    return jax.nn.logsumexp(lp + logw, axis=-1)


def target2_logprob(z: jax.Array, beta: float = -1.0) -> jax.Array:
    z = jnp.atleast_2d(z)
    z1, z2 = z[:, 0:1], z[:, 1:2]

    r = jnp.hypot(z1, z2)
    logexp1 = -0.5 * jnp.square((z1 - 2.0) / 0.8)
    logexp2 = -0.5 * jnp.square((z1 + 2.0) / 0.8)
    log_mix = jax.nn.logsumexp(
        jnp.concatenate([logexp1, logexp2], axis=-1),
        axis=-1,
        keepdims=True,
    )

    u = 0.5 * jnp.square((r - 4.0) / 0.4) - log_mix
    return (beta * u).squeeze(-1)


def target3_logprob(z: jax.Array, beta: float = -1.0) -> jax.Array:
    z = jnp.atleast_2d(z)
    x_in, y_in = z[:, 0:1], z[:, 1:2]
    m = 3
    r0 = 0.65
    sr = 0.12
    x = 2.0 * (x_in - 0.5)
    y = 2.0 * (y_in - 0.5)
    r = jnp.hypot(x, y)
    theta = jnp.arctan2(y, x)
    ring = jnp.exp(-0.5 * ((r - r0) / sr) ** 2)
    petals = jnp.cos(m * theta)
    u = jnp.tanh(1.6 * (ring * petals))
    return (-beta * u).squeeze(-1)


def get_fixed_target_setup(
    target_name: str,
    beta: float,
) -> tuple[
    callable,
    jax.Array,
    jax.Array,
    jax.Array,
    bool,
    jax.Array,
]:
    target_name = str(target_name)
    if target_name == "target1":
        low = jnp.array([-4.0, -4.0], dtype=jnp.float32)
        high = jnp.array([4.0, 4.0], dtype=jnp.float32)
        prior_loc = jnp.array([0.0, 0.0], dtype=jnp.float32)
        process_center = jnp.array([0.0, 0.0], dtype=jnp.float32)
        return target1_logprob, low, high, prior_loc, False, process_center
    if target_name == "target2":
        low = jnp.array([-6.0, -6.0], dtype=jnp.float32)
        high = jnp.array([6.0, 6.0], dtype=jnp.float32)
        prior_loc = jnp.array([0.0, 0.0], dtype=jnp.float32)
        process_center = jnp.array([0.0, 0.0], dtype=jnp.float32)
        return (
            lambda x: target2_logprob(x, beta=beta),
            low,
            high,
            prior_loc,
            False,
            process_center,
        )
    if target_name == "target3":
        low = jnp.array([0.0, 0.0], dtype=jnp.float32)
        high = jnp.array([1.0, 1.0], dtype=jnp.float32)
        prior_loc = jnp.array([0.5, 0.5], dtype=jnp.float32)
        process_center = jnp.array([0.5, 0.5], dtype=jnp.float32)
        return (
            lambda x: target3_logprob(x, beta=beta),
            low,
            high,
            prior_loc,
            False,
            process_center,
        )
    raise ValueError(f"Unknown fixed target: {target_name}")


def make_default_target_harmonic_params(
    dim: int,
    *,
    amplitude_budget: float = 0.25,
    max_modes: int | None = None,
) -> TargetHarmonicParams:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if amplitude_budget <= 0.0 or amplitude_budget > 0.5:
        raise ValueError(
            f"amplitude_budget must lie in (0, 0.5], got {amplitude_budget}"
        )
    if max_modes is not None and max_modes < 1:
        raise ValueError(f"max_modes must be positive, got {max_modes}")

    a = np.zeros(dim, dtype=np.float32)
    required_active_dim = min(dim, 2)
    required_modes = 2 ** required_active_dim
    if max_modes is not None and max_modes < required_modes:
        raise ValueError(
            f"max_modes={max_modes} is too small: target defaults require "
            f"dimensions 0 and 1 to be active when available, needing at least {required_modes} modes."
        )
    max_active_dim = dim if max_modes is None else int(math.floor(math.log(max_modes, 2.0)))
    active_dim = min(dim, 10, max_active_dim)
    if active_dim > 0:
        a[:active_dim] = amplitude_budget / active_dim

    # Keep defaults nontrivial in high dimension while respecting the requested
    # product-mode cap.
    k = np.ones(dim, dtype=np.float32)
    k[:active_dim] = 2.0

    phi = np.zeros(dim, dtype=np.float32)

    return TargetHarmonicParams(
        c=jnp.asarray(0.75, dtype=jnp.float32),
        a=jnp.asarray(a, dtype=jnp.float32),
        k=jnp.asarray(k, dtype=jnp.float32),
        phi=jnp.asarray(phi, dtype=jnp.float32),
    )


def make_default_target_shifted_harmonic_params(
    dim: int,
    *,
    amplitude_budget: float = 0.25,
    max_modes: int | None = None,
) -> TargetShiftedHarmonicParams:
    base = make_default_target_harmonic_params(
        dim,
        amplitude_budget=amplitude_budget,
        max_modes=max_modes,
    )
    mu0 = np.linspace(.5, 1.0, dim, dtype=np.float32) if dim > 1 else np.asarray([0.25], dtype=np.float32)
    b = np.linspace(-2, 2, dim, dtype=np.float32) if dim > 1 else np.asarray([0.25], dtype=np.float32)
    return TargetShiftedHarmonicParams(
        c=base.c,
        a=base.a,
        k=base.k,
        mu0=jnp.asarray(mu0, dtype=jnp.float32),
        b=jnp.asarray(b, dtype=jnp.float32),
    )


def _validate_target_params(
    params: TargetHarmonicParams,
    dim: int,
    max_modes: int | None = None,
) -> TargetHarmonicParams:
    for name in ("a", "k", "phi"):
        value = jnp.asarray(getattr(params, name), dtype=jnp.float32).reshape(-1)
        if value.shape[0] != dim:
            raise ValueError(f"{name} must have length {dim}, got {value.shape[0]}")

    total_amplitude = float(jnp.sum(jnp.abs(jnp.asarray(params.a, dtype=jnp.float32))))
    c = float(jnp.asarray(params.c, dtype=jnp.float32))
    if total_amplitude > min(c, 1.0 - c) + 1e-6:
        raise ValueError(
            "Target cosine amplitudes must satisfy "
            f"sum_i |a_i| <= min(c, 1-c), got {total_amplitude:.6f} with c={c:.6f}"
        )

    validated = TargetHarmonicParams(
        c=jnp.asarray(params.c, dtype=jnp.float32),
        a=jnp.asarray(params.a, dtype=jnp.float32).reshape(dim),
        k=jnp.asarray(params.k, dtype=jnp.float32).reshape(dim),
        phi=jnp.asarray(params.phi, dtype=jnp.float32).reshape(dim),
    )
    if max_modes is not None:
        mode_count = target_effective_num_modes(validated)
        if mode_count > max_modes:
            raise ValueError(
                f"Target effective modes must be <= {max_modes}, got {mode_count}. "
                "Reduce nonzero amplitudes or lower frequencies."
            )
    return validated


def _validate_target_shifted_params(
    params: TargetShiftedHarmonicParams,
    dim: int,
    max_modes: int | None = None,
) -> TargetShiftedHarmonicParams:
    for name in ("a", "k", "mu0", "b"):
        value = jnp.asarray(getattr(params, name), dtype=jnp.float32).reshape(-1)
        if value.shape[0] != dim:
            raise ValueError(f"{name} must have length {dim}, got {value.shape[0]}")

    total_amplitude = float(jnp.sum(jnp.abs(jnp.asarray(params.a, dtype=jnp.float32))))
    c = float(jnp.asarray(params.c, dtype=jnp.float32))
    if total_amplitude > min(c, 1.0 - c) + 1e-6:
        raise ValueError(
            "Target cosine amplitudes must satisfy "
            f"sum_i |a_i| <= min(c, 1-c), got {total_amplitude:.6f} with c={c:.6f}"
        )

    validated = TargetShiftedHarmonicParams(
        c=jnp.asarray(params.c, dtype=jnp.float32),
        a=jnp.asarray(params.a, dtype=jnp.float32).reshape(dim),
        k=jnp.asarray(params.k, dtype=jnp.float32).reshape(dim),
        mu0=jnp.mod(jnp.asarray(params.mu0, dtype=jnp.float32).reshape(dim), 1.0),
        b=jnp.asarray(params.b, dtype=jnp.float32).reshape(dim),
    )
    if max_modes is not None:
        mode_count = target_effective_num_modes(validated)
        if mode_count > max_modes:
            raise ValueError(
                f"Target effective modes must be <= {max_modes}, got {mode_count}. "
                "Reduce nonzero amplitudes or lower frequencies."
            )
    return validated


def target_effective_num_modes(
    params: TargetHarmonicParams | TargetShiftedHarmonicParams,
    *,
    amplitude_tol: float = 1e-8,
) -> int:
    """Estimate product target modes from active cosine frequencies."""
    a = np.asarray(jax.device_get(jnp.asarray(params.a, dtype=jnp.float32))).reshape(-1)
    k = np.asarray(jax.device_get(jnp.asarray(params.k, dtype=jnp.float32))).reshape(-1)
    active = np.abs(a) > amplitude_tol
    if not np.any(active):
        return 1
    per_dim_modes = np.maximum(1, np.rint(np.abs(k[active])).astype(np.int64))
    return int(np.prod(per_dim_modes, dtype=np.int64))


def get_target_harmonic_params(
    dim: int,
    params: TargetHarmonicParams | None = None,
    policy_p: jax.Array | float | None = None,
) -> TargetHarmonicParams:
    if params is None:
        # Use a tracer-safe default path here because this function is also called
        # inside jitted metric code.
        default = make_default_target_harmonic_params(dim)
        return TargetHarmonicParams(
            c=jnp.asarray(default.c, dtype=jnp.float32),
            a=jnp.asarray(default.a, dtype=jnp.float32).reshape(dim),
            k=jnp.asarray(default.k, dtype=jnp.float32).reshape(dim),
            phi=jnp.asarray(default.phi, dtype=jnp.float32).reshape(dim),
        )
    if isinstance(params, TargetShiftedHarmonicParams):
        # Use a tracer-safe conversion path here because this function is called
        # inside jitted training code for target_C.
        shifted = TargetShiftedHarmonicParams(
            c=jnp.asarray(params.c, dtype=jnp.float32),
            a=jnp.asarray(params.a, dtype=jnp.float32).reshape(dim),
            k=jnp.asarray(params.k, dtype=jnp.float32).reshape(dim),
            mu0=jnp.mod(jnp.asarray(params.mu0, dtype=jnp.float32).reshape(dim), 1.0),
            b=jnp.asarray(params.b, dtype=jnp.float32).reshape(dim),
        )
        p = jnp.asarray(0.0 if policy_p is None else policy_p, dtype=jnp.float32)
        mu = jnp.mod(shifted.mu0 + shifted.b * p, 1.0)
        phi = -2.0 * jnp.pi * shifted.k * mu
        return TargetHarmonicParams(
            c=shifted.c,
            a=shifted.a,
            k=shifted.k,
            phi=phi,
        )
    return TargetHarmonicParams(
        c=jnp.asarray(params.c, dtype=jnp.float32),
        a=jnp.asarray(params.a, dtype=jnp.float32).reshape(dim),
        k=jnp.asarray(params.k, dtype=jnp.float32).reshape(dim),
        phi=jnp.asarray(params.phi, dtype=jnp.float32).reshape(dim),
    )


def get_target_shifted_harmonic_params(
    dim: int,
    params: TargetShiftedHarmonicParams | None = None,
) -> TargetShiftedHarmonicParams:
    if params is None:
        return _validate_target_shifted_params(make_default_target_shifted_harmonic_params(dim), dim)
    return _validate_target_shifted_params(params, dim)


def target_component_values(
    samples: jax.Array,
    params: TargetHarmonicParams | TargetShiftedHarmonicParams,
    policy_p: jax.Array | float | None = None,
) -> jax.Array:
    x = jnp.asarray(samples, dtype=jnp.float32)
    resolved = get_target_harmonic_params(x.shape[-1], params, policy_p=policy_p)
    return resolved.a * jnp.cos((2.0 * jnp.pi * resolved.k * x) + resolved.phi)


def target_energy_values(
    samples: jax.Array,
    params: TargetHarmonicParams | TargetShiftedHarmonicParams,
    policy_p: jax.Array | float | None = None,
) -> jax.Array:
    x = jnp.asarray(samples, dtype=jnp.float32)
    resolved = get_target_harmonic_params(x.shape[-1], params, policy_p=policy_p)
    return resolved.c + jnp.sum(target_component_values(x, resolved), axis=-1)


def target_log_normalizer(
    lam: jax.Array,
    params: TargetHarmonicParams | TargetShiftedHarmonicParams,
) -> jax.Array:
    lam = jnp.asarray(lam, dtype=jnp.float32)
    if isinstance(params, TargetShiftedHarmonicParams):
        params = get_target_shifted_harmonic_params(params.a.shape[0], params)
    scaled = lam * params.a
    return (lam * params.c) + jnp.sum(jnp.abs(scaled) + jnp.log(i0e(scaled)))


def target_expected_energy(
    lam: jax.Array,
    params: TargetHarmonicParams | TargetShiftedHarmonicParams,
) -> jax.Array:
    lam = jnp.asarray(lam, dtype=jnp.float32)
    if isinstance(params, TargetShiftedHarmonicParams):
        params = get_target_shifted_harmonic_params(params.a.shape[0], params)
    scaled = lam * params.a
    return params.c + jnp.sum(params.a * (i1e(scaled) / i0e(scaled)))


def target_log_partition(
    lam: jax.Array,
    params: TargetHarmonicParams,
) -> jax.Array:
    return target_log_normalizer(lam, params)


def target_unsafe_action_value(
    lam: jax.Array,
    params: TargetHarmonicParams,
) -> jax.Array:
    return target_expected_energy(lam, params)


def _sample_single_target_dim(
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
def _sample_target_product_jitted(
    key: jax.Array,
    lam: jax.Array,
    params: TargetHarmonicParams,
    n_samples: int,
) -> jax.Array:
    dim = params.a.shape[0]
    key_theta, key_copy = jax.random.split(key)
    theta_keys = jax.random.split(key_theta, dim)
    copy_keys = jax.random.split(key_copy, dim)
    samples_by_dim = jax.vmap(
        _sample_single_target_dim,
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


def sample_target_product(
    key: jax.Array,
    lam: jax.Array,
    shape: tuple[int, int],
    params: TargetHarmonicParams | TargetShiftedHarmonicParams,
    policy_p: jax.Array | float | None = None,
) -> jax.Array:
    if len(shape) != 2:
        raise ValueError(f"shape must be rank-2, got {shape}")
    n_samples, dim = shape
    resolved = get_target_harmonic_params(dim, params, policy_p=policy_p)
    if dim != resolved.a.shape[0]:
        raise ValueError(f"shape dim {dim} does not match params dim {resolved.a.shape[0]}")
    return _sample_target_product_jitted(
        key,
        jnp.asarray(lam, dtype=jnp.float32),
        resolved,
        n_samples,
    )


def add_target_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target-family",
        choices=["target_A", "target_B", "target_C"],
        default="target_B",
        help="Target family variant. target_C uses policy-dependent moving centers.",
    )
    parser.add_argument(
        "--target-c",
        type=float,
        default=0.75,
        help="Constant offset c in g(x)=c+sum_i a_i cos(2pi k_i x_i + phi_i).",
    )
    parser.add_argument(
        "--target-a",
        type=str,
        default=None,
        help="Comma-separated amplitudes a_i. Must have length dim.",
    )
    parser.add_argument(
        "--target-k",
        type=str,
        default=None,
        help="Comma-separated frequencies k_i. Must have length dim.",
    )
    parser.add_argument(
        "--target-phi",
        type=str,
        default=None,
        help="Comma-separated phases phi_i in radians. Must have length dim.",
    )
    parser.add_argument(
        "--target-amplitude-budget",
        type=float,
        default=0.25,
        help="Used only when --target-a is omitted.",
    )
    parser.add_argument(
        "--target-max-modes",
        "--target-max-mode",
        dest="target_max_modes",
        type=int,
        default=30,
        help="Maximum effective product modes for target_A/B/C parameterization.",
    )
    parser.add_argument("--target-C-c", dest="target_C_c", type=float, default=None)
    parser.add_argument("--target-C-a", dest="target_C_a", type=str, default=None)
    parser.add_argument("--target-C-k", dest="target_C_k", type=str, default=None)
    parser.add_argument("--target-C-mu0", dest="target_C_mu0", type=str, default=None)
    parser.add_argument("--target-C-b", dest="target_C_b", type=str, default=None)
    parser.add_argument("--target-C-amplitude-budget", dest="target_C_amplitude_budget", type=float, default=0.25)


def _parse_target_vector(raw: str | None, dim: int, name: str) -> np.ndarray | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if len(values) != dim:
        raise ValueError(f"{name} must have exactly {dim} comma-separated values, got {len(values)}")
    return np.asarray([float(v) for v in values], dtype=np.float32)


def build_target_params_from_args(
    args: argparse.Namespace,
    dim: int,
) -> TargetHarmonicParams | TargetShiftedHarmonicParams:
    family = getattr(args, "target_family", "target_B")
    max_modes = getattr(args, "target_max_modes", None)
    if family == "target_C":
        default = make_default_target_shifted_harmonic_params(
            dim,
            amplitude_budget=float(getattr(args, "target_C_amplitude_budget", 0.25)),
            max_modes=max_modes,
        )
        c = getattr(args, "target_C_c", None)
        a = _parse_target_vector(getattr(args, "target_C_a", None), dim, "target-C-a")
        k = _parse_target_vector(getattr(args, "target_C_k", None), dim, "target-C-k")
        mu0 = _parse_target_vector(getattr(args, "target_C_mu0", None), dim, "target-C-mu0")
        b = _parse_target_vector(getattr(args, "target_C_b", None), dim, "target-C-b")
        params = TargetShiftedHarmonicParams(
            c=default.c if c is None else jnp.asarray(c, dtype=jnp.float32),
            a=default.a if a is None else jnp.asarray(a, dtype=jnp.float32),
            k=default.k if k is None else jnp.asarray(k, dtype=jnp.float32),
            mu0=default.mu0 if mu0 is None else jnp.asarray(mu0, dtype=jnp.float32),
            b=default.b if b is None else jnp.asarray(b, dtype=jnp.float32),
        )
        return _validate_target_shifted_params(params, dim, max_modes=max_modes)

    default = make_default_target_harmonic_params(
        dim,
        amplitude_budget=float(getattr(args, "target_amplitude_budget", 0.45)),
        max_modes=max_modes,
    )
    c = getattr(args, "target_c", None)
    a = _parse_target_vector(getattr(args, "target_a", None), dim, "target-a")
    k = _parse_target_vector(getattr(args, "target_k", None), dim, "target-k")
    phi = _parse_target_vector(getattr(args, "target_phi", None), dim, "target-phi")
    params = TargetHarmonicParams(
        c=default.c if c is None else jnp.asarray(c, dtype=jnp.float32),
        a=default.a if a is None else jnp.asarray(a, dtype=jnp.float32),
        k=default.k if k is None else jnp.asarray(k, dtype=jnp.float32),
        phi=default.phi if phi is None else jnp.asarray(phi, dtype=jnp.float32),
    )
    return _validate_target_params(params, dim, max_modes=max_modes)
