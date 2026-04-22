# gbs_loss.py
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpyro.distributions as npdist


# -------------------------
# Gaussian kernel utilities
# -------------------------
def sample_kernel(key, mean, scale):
    eps = jax.random.normal(key, shape=mean.shape)
    return mean + scale * eps


def log_prob_kernel(x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)



@dataclass(frozen=True)
class VP:
    """Variance Preserving (VP) SDE coefficients (JAX version).

    Matches the coefficient schedule you shared:
      diff_coeff_sq_min/max, scale_diff_coeff, terminal_t.

      x <- x + drift_coeff_t(t) * x * dt
    in addition to the learned control term (diff^2 * u * dt).
    """

    diff_coeff_sq_min: float = 0.01
    diff_coeff_sq_max: float = 10.0
    scale_diff_coeff: float = 1.0
    schedule_type : str = "cosine" # "cosine"
    terminal_t: float = 1.0

    # When True, reverse the schedule direction (kept for parity with your torch code).
    generative: bool = False 
    # Sign of the linear drift coefficient (torch code uses `self.sign` from OU base).
    sign: float = -1.0

    def _diff_coeff_sq_t(self, t: jax.Array) -> jax.Array:
        if self.schedule_type == "linear":
            frac = jnp.clip(t / self.terminal_t, 0.0, 1.0)
        elif self.schedule_type == "cosine":
            frac = jnp.cos( t* jnp.pi/(2* self.terminal_t)) 
        else:
            raise ValueError(f"Not implemented schedule type for VP {self.schedule_type}")
        a = jnp.asarray(self.diff_coeff_sq_min, dtype=jnp.float32)
        b = jnp.asarray(self.diff_coeff_sq_max, dtype=jnp.float32)
        if self.generative:
            # max -> min
            return b + frac * (a - b)
        # min -> max
        return a + frac * (b - a)

    def drift_coeff_t(self, t: jax.Array) -> jax.Array:
        return 0.5 * jnp.asarray(self.sign, dtype=jnp.float32) * self._diff_coeff_sq_t(t)

    def diff_coeff_t(self, t: jax.Array) -> jax.Array:
        return jnp.asarray(self.scale_diff_coeff, dtype=jnp.float32) * jnp.sqrt(
            self._diff_coeff_sq_t(t)
        )


@dataclass(frozen=True)
class Langevin:
    """Langevin-style diffusion (JAX version).

    This class encodes the *uncontrolled* SDE:
      dX = diff_coeff dW

    The GBS sampler adds a learned control in the drift (via the networks):
      dX = diff_coeff * u(t, X) dt + diff_coeff dW

    Notes:
    - This matches the Torch `ControlledSDE` pattern where drift += diff * ctrl.
    - There is no base (linear) drift term.
    """

    diff_coeff: float = 1.0
    terminal_t: float = 1.0

    include_base_drift: bool = False

    def drift_coeff_t(self, t: jax.Array) -> jax.Array:
        return jnp.asarray(0.0, dtype=jnp.float32)

    def diff_coeff_t(self, t: jax.Array) -> jax.Array:
        del t
        return jnp.asarray(self.diff_coeff, dtype=jnp.float32)


def reference_ctrl(
    process: VP | Langevin,
    x: jax.Array,
    t: jax.Array,
    process_center: jax.Array | None = None,
) -> jax.Array:
    """Fixed DDS reference control r.

    For VP, use the paper's Gaussian-reference form
      r(x,t) = -sigma(t) * (x - c) / nu^2.
    For Langevin, fall back to the uncontrolled reference r = 0.
    """
    center = (
        jnp.asarray(0.0, dtype=jnp.float32)
        if process_center is None
        else jnp.asarray(process_center, dtype=jnp.float32)
    )
    if isinstance(process, VP):
        diff = process.diff_coeff_t(t)
        nu2 = jnp.asarray(process.scale_diff_coeff, dtype=jnp.float32) **2
        return -(diff / nu2) * (x - center)
    return jnp.zeros_like(x)


def terminal_uncontrolled_log_prob(
    process: VP | Langevin,
    x: jax.Array,
    process_center: jax.Array | None = None,
) -> jax.Array:
    """Terminal uncontrolled-process density log P_T(x).

    In the combined.pdf TR-LV / DDS setting, the terminal cost is
      g(x) = log P_T(x) - log rho_target(x),
    where P_T is the terminal marginal of the *uncontrolled* process.

    For our VP parameterization
      dX_t = -a(t) (X_t - c) dt + scale_diff_coeff * sqrt(2 a(t)) dW_t,
    the uncontrolled process has the stationary Gaussian marginal
      P_T = N(c, scale_diff_coeff^2 I),
    which is also the terminal density used in the DDS reference-process form.
    """
    center = (
        jnp.asarray(0.0, dtype=jnp.float32)
        if process_center is None
        else jnp.asarray(process_center, dtype=jnp.float32)
    )
    if isinstance(process, VP):
        loc = jnp.broadcast_to(center, x.shape)
        scale = jnp.full_like(x, jnp.asarray(process.scale_diff_coeff, dtype=x.dtype))
        dist = npdist.Independent(npdist.Normal(loc=loc, scale=scale), 1)
        return dist.log_prob(x)
    return jnp.zeros(x.shape[0], dtype=x.dtype)


def target_lgv_term(
    x: jax.Array,
    target_loggrad_fn=None,
    use_lgv: bool = False,
) -> jax.Array:
    """Return the target log-gradient term, or zeros when disabled."""
    if target_loggrad_fn is None or not use_lgv:
        return jnp.zeros_like(x)
    return jnp.asarray(target_loggrad_fn(x), dtype=x.dtype)

# -------------------------
# Pure GBS per-sample rollout
# -------------------------
def rnd_no_target(
    key,
    model_state,
    fwd_params,
    bwd_params,
    batch_size,
    prior_sampler,
    num_steps,
    process,
    use_reference_ctrl: bool = False,
    stop_grad=True,
    process_center: jax.Array | None = None,
    use_ito: bool = True,
    target_loggrad_fn=None,
    use_lgv: bool = False,
    integrator_type: str = "euler",   # "euler" or "exp"
    alpha: float = 1.0,               # only used for exp
):
    del bwd_params
    fwd_state = model_state[0] if isinstance(model_state, tuple) else model_state
    dt = jnp.asarray(process.terminal_t, dtype=jnp.float32) / jnp.asarray(
        num_steps, dtype=jnp.float32
    )
    center = (
        jnp.asarray(0.0, dtype=jnp.float32)
        if process_center is None
        else jnp.asarray(process_center, dtype=jnp.float32)
    )

    def per_sample(seed):
        key, k0 = jax.random.split(seed)
        x0 = prior_sampler(k0)
        x = x0
        rnd = jnp.asarray(0.0, dtype=jnp.float32)

        def step_fn(carry, step_i):
            x, rnd, key = carry
            step = step_i.astype(jnp.float32)
            x_in = jax.lax.stop_gradient(x) if stop_grad else x
            t = step * dt

            lgv_term = target_lgv_term(x_in, target_loggrad_fn, use_lgv)
            u_fwd = fwd_state.apply_fn(
                fwd_params, x_in, t * jnp.ones((1,)), lgv_term
            )
            ref = (
                reference_ctrl(process, x_in, t, process_center)
                if use_reference_ctrl
                else jnp.zeros_like(x_in)
            )

            u_ctrl = u_fwd
            sde_ctrl = jax.lax.stop_gradient(u_ctrl) if stop_grad else u_ctrl
            delta_ctrl = u_ctrl - ref

            key, k_eps = jax.random.split(key)
            noise = jax.random.normal(k_eps, shape=x.shape)

            if integrator_type == "euler":
                diff = process.diff_coeff_t(t)
                base_drift = (
                    process.drift_coeff_t(t) * (x_in - center)
                    if getattr(process, "include_base_drift", False)
                    else 0.0
                )

                db = noise * jnp.sqrt(dt)
                running = jnp.sum(
                    delta_ctrl * (sde_ctrl - 0.5 * (u_ctrl + ref))
                ) * dt
                ito = jnp.sum(delta_ctrl * db)

                x_next = x + (base_drift + diff * sde_ctrl) * dt + diff * db
                rnd_next = rnd + running + (ito if use_ito else 0.0)

            elif integrator_type == "exp":
                if not isinstance(process, VP):
                    raise ValueError("integrator_type='exp' currently only supports VP")

                sigma = jnp.asarray(process.scale_diff_coeff, dtype=jnp.float32)
                beta_k = jnp.clip(jnp.asarray(alpha, dtype=jnp.float32) * jnp.sqrt(dt), 0.0, 1.0)
                alpha_k = jnp.sqrt(1.0 - beta_k**2)

                running = beta_k**2 * sigma**2 * jnp.sum(
                    delta_ctrl * (sde_ctrl - 0.5 * (u_ctrl + ref))
                )
                ito = jnp.sum(sigma * delta_ctrl * noise * beta_k)

                x_next = x * alpha_k + beta_k**2 * sigma**2 * sde_ctrl + sigma * beta_k * noise
                rnd_next = rnd + running + (ito if use_ito else 0.0)

            else:
                raise ValueError(f"Unknown integrator_type: {integrator_type}")

            return (x_next, rnd_next, key), None

        (xT, rnd_running, _), _ = jax.lax.scan(
            step_fn, (x, rnd, key), jnp.arange(num_steps)
        )
        return x0, xT, -rnd_running

    seeds = jax.random.split(key, batch_size)
    x0, xT, log_ratio = jax.vmap(per_sample)(seeds)
    return x0, xT, log_ratio

def simul_forward_sde_for_buffer(
    key: jax.Array,
    model_state,
    fwd_params,
    batch_size: int,
    prior_sampler,
    num_steps: int,
    process: VP | Langevin,
    process_center: jax.Array | None = None,
    target_loggrad_fn=None,
    use_lgv: bool = False,
    integrator_type: str = "euler",   # "euler" or "exp"
    alpha: float = 1.0,               # only used for "exp"
):
    """Samples a batch of trajectories under the current control for TR-LV.

    Returns latent states and Brownian/noise increments so trust-region losses
    can be computed from a fixed rollout buffer.
    """
    fwd_state, _ = model_state
    dt = jnp.asarray(process.terminal_t, dtype=jnp.float32) / jnp.asarray(
        num_steps, dtype=jnp.float32
    )
    center = (
        jnp.asarray(0.0, dtype=jnp.float32)
        if process_center is None
        else jnp.asarray(process_center, dtype=jnp.float32)
    )

    def per_sample(seed):
        key, k0 = jax.random.split(seed)
        x0 = prior_sampler(k0)
        x = x0

        def step_fn(carry, step_i):
            x, key = carry
            step = step_i.astype(jnp.float32)
            t = step * dt

            x_in = jax.lax.stop_gradient(x)
            lgv_term = target_lgv_term(x_in, target_loggrad_fn, use_lgv)
            u = fwd_state.apply_fn(
                fwd_params, x_in, t * jnp.ones((1,)), lgv_term
            )
            sde_ctrl = jax.lax.stop_gradient(u)

            key, k_eps = jax.random.split(key)
            noise = jax.random.normal(k_eps, shape=x.shape)

            if integrator_type == "euler":
                diff = process.diff_coeff_t(t)
                base_drift = (
                    process.drift_coeff_t(t) * (x_in - center)
                    if getattr(process, "include_base_drift", False)
                    else 0.0
                )
                db = noise * jnp.sqrt(dt)
                x_next = x + (base_drift + diff * sde_ctrl) * dt + diff * db
                incr = db

            elif integrator_type == "exp":
                if not isinstance(process, VP):
                    raise ValueError("integrator_type='exp' currently only supports VP")

                sigma = jnp.asarray(process.scale_diff_coeff, dtype=jnp.float32)
                beta_k = jnp.clip(
                    jnp.asarray(alpha, dtype=jnp.float32) * jnp.sqrt(dt), 0.0, 1.0
                )
                alpha_k = jnp.sqrt(1.0 - beta_k**2)

                x_next = (
                    x * alpha_k
                    + beta_k**2 * sigma**2 * sde_ctrl
                    + sigma * beta_k * noise
                )
                # store the stochastic increment actually used by the exp step
                incr = sigma * beta_k * noise

            else:
                raise ValueError(f"Unknown integrator_type: {integrator_type}")

            return (x_next, key), (x_next, incr)

        (xT, _), (xs, incrs) = jax.lax.scan(
            step_fn, (x, key), jnp.arange(num_steps)
        )
        path = jnp.concatenate([x0[None, :], xs], axis=0)
        return path, incrs

    seeds = jax.random.split(key, batch_size)
    paths, incrs = jax.vmap(per_sample)(seeds)
    return paths, incrs

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def solve_trust_region_lambda_grid_golden(
    logw: jax.Array,
    trust_region_bound: float,
    lambda_max: float = 50.0,
    grid_size: int = 129,
    maxiter: int = 60,
) -> jax.Array:
    eps = jnp.asarray(trust_region_bound, dtype=jnp.float32)
    n = jnp.asarray(logw.shape[0], dtype=jnp.float32)

    def neg_dual(lam):
        alpha = 1.0 / (1.0 + lam)
        logz = jsp.special.logsumexp(-alpha * logw) - jnp.log(n)
        dual = -(1.0 + lam) * logz - lam * eps
        return -dual

    grid = jnp.linspace(0.0, lambda_max, grid_size, dtype=jnp.float32)
    vals = jax.vmap(neg_dual)(grid)
    idx = jnp.argmin(vals)

    left_idx = jnp.maximum(idx - 1, 0)
    right_idx = jnp.minimum(idx + 1, grid_size - 1)

    a0 = grid[left_idx]
    b0 = grid[right_idx]

    phi = (1.0 + jnp.sqrt(jnp.array(5.0, dtype=jnp.float32))) / 2.0
    invphi = 1.0 / phi

    c0 = b0 - (b0 - a0) * invphi
    d0 = a0 + (b0 - a0) * invphi
    fc0 = neg_dual(c0)
    fd0 = neg_dual(d0)

    def body_fn(i, state):
        a, b, c, d, fc, fd = state
        go_left = fc < fd

        new_a = jnp.where(go_left, a, c)
        new_b = jnp.where(go_left, d, b)

        new_c = jnp.where(go_left, new_b - (new_b - new_a) * invphi, d)
        new_d = jnp.where(go_left, c, new_a + (new_b - new_a) * invphi)

        new_fc = jnp.where(go_left, neg_dual(new_c), fd)
        new_fd = jnp.where(go_left, fc, neg_dual(new_d))
        return (new_a, new_b, new_c, new_d, new_fc, new_fd)

    a, b, c, d, fc, fd = jax.lax.fori_loop(
        0, maxiter, body_fn, (a0, b0, c0, d0, fc0, fd0)
    )

    candidates = jnp.array([grid[idx], 0.5 * (a + b), 0.0, lambda_max], dtype=jnp.float32)
    cand_vals = jax.vmap(neg_dual)(candidates)
    return candidates[jnp.argmin(cand_vals)]
def solve_trust_region_lambda_from_logw(
    logw: jax.Array,
    trust_region_bound: float,
    lambda_max: float = 50.0,
    lambda_grid_size: int = 128,
) -> jax.Array:
    eps = jnp.asarray(trust_region_bound, dtype=jnp.float32)
    upper = jnp.asarray(lambda_max, dtype=jnp.float32)
    n = jnp.asarray(logw.shape[0], dtype=jnp.float32)
    maxiter = max(int(lambda_grid_size), 1)

    def neg_dual(lam):
        alpha = 1.0 / (1.0 + lam)
        logz = jsp.special.logsumexp(-alpha * logw) - jnp.log(n)
        dual = -(1.0 + lam) * logz - lam * eps
        return -dual

    g = jax.grad(neg_dual)

    lo = jnp.array(0.0, dtype=jnp.float32)
    hi = upper
    g_lo = g(lo)
    g_hi = g(hi)

    # If derivative does not change sign, optimum is at a boundary.
    def boundary_case():
        f0 = neg_dual(lo)
        f1 = neg_dual(hi)
        return jnp.where(f0 <= f1, lo, hi)

    def interior_case():
        def body_fn(_, state):
            lo, hi, g_lo = state
            mid = 0.5 * (lo + hi)
            g_mid = g(mid)
            same_sign = jnp.sign(g_mid) == jnp.sign(g_lo)
            lo = jnp.where(same_sign, mid, lo)
            hi = jnp.where(same_sign, hi, mid)
            g_lo = jnp.where(same_sign, g_mid, g_lo)
            return (lo, hi, g_lo)

        lo2, hi2 = jax.lax.fori_loop(0, maxiter, body_fn, (lo, hi, g_lo))
        lam = 0.5 * (lo2 + hi2)

        # endpoint safeguard
        f0 = neg_dual(lo)
        fm = neg_dual(lam)
        f1 = neg_dual(hi)
        lam = jnp.where(f0 <= fm, lo, lam)
        best_f = jnp.minimum(f0, fm)
        lam = jnp.where(f1 <= best_f, hi, lam)
        return lam

    has_bracket = (g_lo * g_hi) <= 0.0
    return jax.lax.cond(has_bracket, lambda: interior_case(), lambda: boundary_case())

def trust_region_log_ratio_from_brownian(
    *,
    model_state,
    behavior_params,
    candidate_params,
    paths: jax.Array,
    dbs: jax.Array,
    num_steps: int,
    process: VP | Langevin,
    process_center: jax.Array | None = None,
    target_loggrad_fn=None,
    use_lgv: bool = False,
):
    """trust-region correction using stored Brownian increments."""
    fwd_state, _ = model_state
    dt = jnp.asarray(process.terminal_t, dtype=jnp.float32) / jnp.asarray(
        num_steps, dtype=jnp.float32
    )
    center = (
        jnp.asarray(0.0, dtype=jnp.float32)
        if process_center is None
        else jnp.asarray(process_center, dtype=jnp.float32)
    )

    xs = paths[:, :-1, :]
    x_next = paths[:, 1:, :]
    steps = jnp.arange(num_steps, dtype=jnp.float32)
    times = jnp.broadcast_to(
        steps[None, :, None] * dt,
        (xs.shape[0], xs.shape[1], 1),
    )

    flat_times = times.reshape(-1, 1)
    step_times = steps * dt
    flat_xs = xs.reshape(-1, xs.shape[-1])
    flat_lgv = target_lgv_term(flat_xs, target_loggrad_fn, use_lgv)
    u_behavior = fwd_state.apply_fn(
        behavior_params,
        flat_xs,
        flat_times,
        flat_lgv,
    ).reshape(xs.shape)
    u_candidate = fwd_state.apply_fn(
        candidate_params,
        flat_xs,
        flat_times,
        flat_lgv,
    ).reshape(xs.shape)
    del x_next, center
    delta = u_behavior - u_candidate
    return jnp.sum(0.5 * jnp.sum(delta**2, axis=-1) * dt + jnp.sum(delta * dbs, axis=-1), axis=1)


def dds_logw_from_buffer(
    *,
    model_state,
    behavior_params,
    paths: jax.Array,
    dbs: jax.Array,
    num_steps: int,
    process: VP | Langevin,
    target_lp_vals: jax.Array,
    process_center: jax.Array | None = None,
    target_loggrad_fn=None,
    use_lgv: bool = False,
):
    fwd_state, _ = model_state
    dt = jnp.asarray(process.terminal_t, dtype=jnp.float32) / jnp.asarray(
        num_steps, dtype=jnp.float32
    )
    xs = paths[:, :-1, :]
    xT = paths[:, -1, :]
    steps = jnp.arange(num_steps, dtype=jnp.float32)
    times = jnp.broadcast_to(steps[None, :, None] * dt, (xs.shape[0], xs.shape[1], 1))
    flat_xs = xs.reshape(-1, xs.shape[-1])
    flat_times = times.reshape(-1, 1)
    flat_lgv = target_lgv_term(flat_xs, target_loggrad_fn, use_lgv)
    u_behavior = fwd_state.apply_fn(
        behavior_params,
        flat_xs,
        flat_times,
        flat_lgv,
    ).reshape(xs.shape)
    step_times = steps * dt
    ref = jax.vmap(lambda t, x: reference_ctrl(process, x, t, process_center), in_axes=(0, 0))
    ref = jax.vmap(ref, in_axes=(None, 0))(step_times, xs)
    delta = u_behavior - ref
    rnd_running = jnp.sum(
        0.5 * jnp.sum(delta**2, axis=-1) * dt + jnp.sum(delta * dbs, axis=-1),
        axis=1,
    )
    terminal_lp = terminal_uncontrolled_log_prob(process, xT, process_center)
    return target_lp_vals - terminal_lp - rnd_running


def tr_lv_loss_from_buffer(
    *,
    model_state,
    behavior_params,
    candidate_params,
    paths: jax.Array,
    dbs: jax.Array,
    logw_behavior: jax.Array,
    num_steps: int,
    process: VP | Langevin,
    max_rnd: float | None = None,
    process_center: jax.Array | None = None,
    fixed_lambda: jax.Array | float | None = None,
    target_lp_vals: jax.Array | None = None,
    target_loggrad_fn=None,
    use_lgv: bool = False,
):
    """Trust-region log-variance objective on a fixed trajectory batch."""
    x0 = paths[:, 0, :]
    xT = paths[:, -1, :]
    lam = jnp.asarray(fixed_lambda, dtype=jnp.float32)
    alpha = 1.0 / (1.0 + lam)

    # For the lambda==0 ablation, make TR-DDS-LV collapse to the buffered DDS-LV
    # objective instead of the trust-region correction form.
    def dds_lv_case():
        candidate_logw = dds_logw_from_buffer(
            model_state=model_state,
            behavior_params=candidate_params,
            paths=paths,
            dbs=dbs,
            num_steps=num_steps,
            process=process,
            target_lp_vals=target_lp_vals,
            process_center=process_center,
            target_loggrad_fn=target_loggrad_fn,
            use_lgv=use_lgv,
        )
        objective = -candidate_logw
        mask = jnp.isfinite(objective)
        if max_rnd is not None:
            # Match lv_loss_from_values exactly for the lambda==0 ablation.
            mask = mask & (objective < max_rnd)
        objective_masked = jnp.where(mask, objective, jnp.nan)
        aux = {
            "train/tr_lv_mean": jnp.nanmean(objective_masked),
            "train/tr_lv_var": jnp.nanvar(objective_masked),
            "train/tr_lv_lambda": lam,
            "train/tr_lv_alpha": alpha,
            "train/logw_mean": jnp.nanmean(jnp.where(mask, candidate_logw, jnp.nan)),
            "train/logw_var": jnp.nanvar(jnp.where(mask, candidate_logw, jnp.nan)),
            "train/xT_mean_norm": jnp.mean(jnp.linalg.norm(xT, axis=-1)),
            "train/n_filtered": jnp.sum(~mask),
        }
        return jnp.nanvar(objective_masked), aux, objective

    def tr_case():
        log_ratio = trust_region_log_ratio_from_brownian(
            model_state=model_state,
            behavior_params=behavior_params,
            candidate_params=candidate_params,
            paths=paths,
            dbs=dbs,
            num_steps=num_steps,
            process=process,
            process_center=process_center,
            target_loggrad_fn=target_loggrad_fn,
            use_lgv=use_lgv,
        )
        objective = alpha * logw_behavior + log_ratio
        mask = jnp.isfinite(objective)
        if max_rnd is not None:
            mask = mask & (jnp.abs(objective) < max_rnd)
        objective_masked = jnp.where(mask, objective, jnp.nan)
        aux = {
            "train/tr_lv_mean": jnp.nanmean(objective_masked),
            "train/tr_lv_var": jnp.nanvar(objective_masked),
            "train/tr_lv_lambda": lam,
            "train/tr_lv_alpha": alpha,
            "train/logw_mean": jnp.nanmean(jnp.where(mask, logw_behavior, jnp.nan)),
            "train/logw_var": jnp.nanvar(jnp.where(mask, logw_behavior, jnp.nan)),
            "train/xT_mean_norm": jnp.mean(jnp.linalg.norm(xT, axis=-1)),
            "train/n_filtered": jnp.sum(~mask),
        }
        return jnp.nanvar(objective_masked), aux, objective

    use_dds_lv = jnp.equal(lam, 0.0)
    return jax.lax.cond(use_dds_lv, dds_lv_case, tr_case)


def lv_loss_from_rnd(
    rnd: jax.Array,
    xT: jax.Array | None = None,
    max_rnd: float | None = None,
) -> tuple[jax.Array, dict, jax.Array]:
    """Log-variance loss from a scalar rnd per sample with optional filtering."""
    mask = jnp.isfinite(rnd)
    if max_rnd is not None:
        mask = mask & (rnd < max_rnd)
    rnd_masked = jnp.where(mask, rnd, jnp.nan)
    loss = jnp.nanvar(rnd_masked)
    aux = {
        "train/rnd_mean": jnp.nanmean(rnd_masked),
        "train/rnd_var": jnp.nanvar(rnd_masked),
        "train/n_filtered": jnp.sum(~mask),
    }
    if xT is not None:
        aux["train/xT_mean_norm"] = jnp.mean(jnp.linalg.norm(xT, axis=-1))
    return loss, aux, rnd


def re_loss_from_rnd(
    rnd: jax.Array,
    xT: jax.Array | None = None,
    max_rnd: float | None = None,
) -> tuple[jax.Array, dict, jax.Array]:
    """Reverse-KL / relative-entropy style loss from a per-sample estimator."""
    mask = jnp.isfinite(rnd)
    if max_rnd is not None:
        mask = mask & (rnd < max_rnd)
    rnd_masked = jnp.where(mask, rnd, jnp.nan)
    loss = jnp.nanmean(rnd_masked)
    aux = {
        "train/rnd_mean": jnp.nanmean(rnd_masked),
        "train/rnd_var": jnp.nanvar(rnd_masked),
        "train/n_filtered": jnp.sum(~mask),
    }
    if xT is not None:
        aux["train/xT_mean_norm"] = jnp.mean(jnp.linalg.norm(xT, axis=-1))
    return loss, aux, rnd


def lv_loss_from_values(
    x0,                 # [B,D]
    xT,                 # [B,D] (not strictly needed for loss, but useful for logging)
    log_ratio,          # [B]
    prior_log_prob,     # callable: prior_log_prob(x0)->[B] or scalar per item
    target_lp_vals,     # [B]  <-- numeric values already computed!
    max_rnd: float | None = None,
):
    running_cost = -log_ratio                         # [B]
    terminal_cost = prior_log_prob(x0) - target_lp_vals  # [B]
    neg_elbo = running_cost + terminal_cost           # [B]
    mask = jnp.isfinite(neg_elbo)
    if max_rnd is not None:
        mask = mask & (neg_elbo < max_rnd)
    neg_elbo_masked = jnp.where(mask, neg_elbo, jnp.nan)
    loss = jnp.nanvar(neg_elbo_masked)
    aux = {
        "train/neg_elbo_mean": jnp.nanmean(neg_elbo_masked),
        "train/neg_elbo_var": jnp.nanvar(neg_elbo_masked),
        "train/running_mean": jnp.nanmean(jnp.where(mask, running_cost, jnp.nan)),
        "train/terminal_mean": jnp.nanmean(jnp.where(mask, terminal_cost, jnp.nan)),
        "train/xT_mean_norm": jnp.mean(jnp.linalg.norm(xT, axis=-1)),
        "train/n_filtered": jnp.sum(~mask),
    }
    return loss, aux, neg_elbo


def re_loss_from_values(
    x0,
    xT,
    log_ratio,
    prior_log_prob,
    target_lp_vals,
    max_rnd: float | None = None,
):
    running_cost = -log_ratio
    terminal_cost = prior_log_prob(x0) - target_lp_vals
    neg_elbo = running_cost + terminal_cost
    mask = jnp.isfinite(neg_elbo)
    if max_rnd is not None:
        mask = mask & (neg_elbo < max_rnd)
    neg_elbo_masked = jnp.where(mask, neg_elbo, jnp.nan)
    loss = jnp.nanmean(neg_elbo_masked)
    aux = {
        "train/neg_elbo_mean": jnp.nanmean(neg_elbo_masked),
        "train/neg_elbo_var": jnp.nanvar(neg_elbo_masked),
        "train/running_mean": jnp.nanmean(jnp.where(mask, running_cost, jnp.nan)),
        "train/terminal_mean": jnp.nanmean(jnp.where(mask, terminal_cost, jnp.nan)),
        "train/xT_mean_norm": jnp.mean(jnp.linalg.norm(xT, axis=-1)),
        "train/n_filtered": jnp.sum(~mask),
    }
    return loss, aux, neg_elbo


def dds_lv_loss_from_values(
    x0,
    xT,
    log_ratio,
    process: VP | Langevin,
    target_lp_vals,
    process_center: jax.Array | None = None,
    max_rnd: float | None = None,
):
    del x0
    running_cost = -log_ratio
    terminal_cost = terminal_uncontrolled_log_prob(process, xT, process_center) - target_lp_vals
    neg_elbo = running_cost + terminal_cost
    mask = jnp.isfinite(neg_elbo)
    if max_rnd is not None:
        mask = mask & (neg_elbo < max_rnd)
    neg_elbo_masked = jnp.where(mask, neg_elbo, jnp.nan)
    loss = jnp.nanvar(neg_elbo_masked)
    aux = {
        "train/neg_elbo_mean": jnp.nanmean(neg_elbo_masked),
        "train/neg_elbo_var": jnp.nanvar(neg_elbo_masked),
        "train/running_mean": jnp.nanmean(jnp.where(mask, running_cost, jnp.nan)),
        "train/terminal_mean": jnp.nanmean(jnp.where(mask, terminal_cost, jnp.nan)),
        "train/xT_mean_norm": jnp.mean(jnp.linalg.norm(xT, axis=-1)),
        "train/n_filtered": jnp.sum(~mask),
    }
    return loss, aux, neg_elbo


def dds_re_loss_from_values(
    x0,
    xT,
    log_ratio,
    process: VP | Langevin,
    target_lp_vals,
    process_center: jax.Array | None = None,
    max_rnd: float | None = None,
):
    del x0
    running_cost = -log_ratio
    terminal_cost = terminal_uncontrolled_log_prob(process, xT, process_center) - target_lp_vals
    neg_elbo = running_cost + terminal_cost
    mask = jnp.isfinite(neg_elbo)
    if max_rnd is not None:
        mask = mask & (neg_elbo < max_rnd)
    neg_elbo_masked = jnp.where(mask, neg_elbo, jnp.nan)
    loss = jnp.nanmean(neg_elbo_masked)
    aux = {
        "train/neg_elbo_mean": jnp.nanmean(neg_elbo_masked),
        "train/neg_elbo_var": jnp.nanvar(neg_elbo_masked),
        "train/running_mean": jnp.nanmean(jnp.where(mask, running_cost, jnp.nan)),
        "train/terminal_mean": jnp.nanmean(jnp.where(mask, terminal_cost, jnp.nan)),
        "train/xT_mean_norm": jnp.mean(jnp.linalg.norm(xT, axis=-1)),
        "train/n_filtered": jnp.sum(~mask),
    }
    return loss, aux, neg_elbo
