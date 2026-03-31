# gbs_loss.py
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
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


def _apply_sde_ctrl_perturbation(
    key: jax.Array,
    generative_ctrl: jax.Array,
    drift: jax.Array,
    diff: jax.Array,
    sde_ctrl_noise: float | None = None,
    sde_ctrl_dropout: float | None = None,
) -> jax.Array:
    """Mirror `sde_sampler`'s optional LV control perturbations."""
    noise = jnp.asarray(
        0.0 if sde_ctrl_noise is None else sde_ctrl_noise, dtype=generative_ctrl.dtype
    )
    dropout = jnp.asarray(
        0.0 if sde_ctrl_dropout is None else sde_ctrl_dropout, dtype=generative_ctrl.dtype
    )

    key, kn, kd = jax.random.split(key, 3)
    sde_ctrl = generative_ctrl

    # Additive control noise when enabled.
    sde_ctrl = jax.lax.cond(
        noise > 0.0,
        lambda ctrl: ctrl + noise * jax.random.normal(kn, shape=ctrl.shape),
        lambda ctrl: ctrl,
        sde_ctrl,
    )

    # Mirror torch behavior only when dropout > 0:
    # mask = rand > p ; sde_ctrl[mask] = -(drift / diff)
    def _apply_dropout(ctrl):
        mask = jax.random.uniform(kd, shape=ctrl.shape) > dropout
        return jnp.where(mask, -drift / diff, ctrl)

    sde_ctrl = jax.lax.cond(dropout > 0.0, _apply_dropout, lambda ctrl: ctrl, sde_ctrl)
    return sde_ctrl


@dataclass(frozen=True)
class VP:
    """Variance Preserving (VP) SDE coefficients (JAX version).

    Matches the coefficient schedule you shared:
      diff_coeff_sq_min/max, scale_diff_coeff, terminal_t.

    If `include_base_drift=True`, the sampler uses the base linear drift term:
      x <- x + drift_coeff_t(t) * x * dt
    in addition to the learned control term (diff^2 * u * dt).
    """

    diff_coeff_sq_min: float = 0.1
    diff_coeff_sq_max: float = 10.0
    scale_diff_coeff: float = 1.0
    terminal_t: float = 1.0

    # When True, reverse the schedule direction (kept for parity with your torch code).
    generative: bool = False

    # Sign of the linear drift coefficient (torch code uses `self.sign` from OU base).
    sign: float = -1.0

    # Whether to include the base linear drift term in the discrete kernel.
    include_base_drift: bool = True

    def _diff_coeff_sq_t(self, t: jax.Array) -> jax.Array:
        frac = jnp.clip(t / self.terminal_t, 0.0, 1.0)
        a = jnp.asarray(self.diff_coeff_sq_min, dtype=jnp.float32)
        b = jnp.asarray(self.diff_coeff_sq_max, dtype=jnp.float32)
        if self.generative:
            # max -> min
            return b + frac * (a - b)
        # min -> max
        return a + frac * (b - a)

    def drift_coeff_t(self, t: jax.Array) -> jax.Array:
        return jnp.asarray(self.sign, dtype=jnp.float32) * 0.5 * self._diff_coeff_sq_t(t)

    def diff_coeff_t(self, t: jax.Array) -> jax.Array:
        return jnp.asarray(self.scale_diff_coeff, dtype=jnp.float32) * jnp.sqrt(
            self._diff_coeff_sq_t(t)
        )

    def int_drift_coeff_t(self, s: jax.Array, t: jax.Array) -> jax.Array:
        dt = t - s
        return (
            jnp.asarray(self.sign, dtype=jnp.float32)
            * 0.25
            * (self._diff_coeff_sq_t(t) + self._diff_coeff_sq_t(s))
            * dt
        )

    def int_diff_coeff_sq_t(self, s: jax.Array, t: jax.Array) -> jax.Array:
        dt = t - s
        scale_sq = jnp.asarray(self.scale_diff_coeff, dtype=jnp.float32) ** 2
        return 0.5 * scale_sq * (self._diff_coeff_sq_t(t) + self._diff_coeff_sq_t(s)) * dt

    def marginal_params(
        self,
        t: jax.Array,
        x_init: jax.Array,
        var_init: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        s = jnp.asarray(0.0, dtype=jnp.float32)
        int_drift_coeff = self.int_drift_coeff_t(s, t)
        loc = jnp.exp(int_drift_coeff)
        scale_sq = jnp.asarray(self.scale_diff_coeff, dtype=jnp.float32) ** 2
        var = (1.0 - jnp.exp(2.0 * int_drift_coeff)) * scale_sq
        if var_init is not None:
            var = var + (loc**2) * var_init
        return loc * x_init, var


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

# -------------------------
# Pure GBS per-sample rollout
# -------------------------
def rnd_no_target(
    key,
    model_state,          # (fwd_state, bwd_state)
    fwd_params,
    bwd_params,
    batch_size,
    prior_sampler,        # callable: prior_sampler(key)->x0 [D]
    num_steps,
    noise_schedule,       # callable: sigma(step)->scalar OR VP instance
    stop_grad=True,       # for LV training, usually True
    process_center: jax.Array | None = None,
):
    fwd_state, bwd_state = model_state
    use_process = hasattr(noise_schedule, "diff_coeff_t") and hasattr(
        noise_schedule, "drift_coeff_t"
    )
    if use_process:
        dt = jnp.asarray(noise_schedule.terminal_t, dtype=jnp.float32) / jnp.asarray(num_steps, dtype=jnp.float32)
    else:
        dt = 1.0 / num_steps

    center = jnp.asarray(0.0, dtype=jnp.float32) if process_center is None else jnp.asarray(process_center, dtype=jnp.float32)

    def zero_lgv(x):
        return jnp.zeros_like(x)

    def per_sample(seed):
        key, k0 = jax.random.split(seed)
        x0 = prior_sampler(k0)   # [D]
        x = x0
        log_w = 0.0

        def step_fn(carry, step_i):
            x, log_w, key = carry
            step = step_i.astype(jnp.float32)

            x_in = jax.lax.stop_gradient(x) if stop_grad else x
            if use_process:
                t = step * dt
                t_next = (step + 1.0) * dt
                diff_t = noise_schedule.diff_coeff_t(t)
                diff_next = noise_schedule.diff_coeff_t(t_next)
                drift_coeff_t = noise_schedule.drift_coeff_t(t)
                drift_coeff_next = noise_schedule.drift_coeff_t(t_next)

                scale = diff_t * jnp.sqrt(dt)
                base_drift = (
                    drift_coeff_t * (x_in - center)
                    if getattr(noise_schedule, "include_base_drift", False)
                    else 0.0
                )

                time_fwd = t
                time_bwd = t_next
            else:
                sigma_t = noise_schedule(step)
                sigma2_t = sigma_t**2
                scale = sigma_t * jnp.sqrt(2.0 * dt)
                drift_term = 0.0
                time_fwd = step
                time_bwd = step + 1.0

            u_fwd = fwd_state.apply_fn(
                fwd_params, x_in, time_fwd * jnp.ones((1,)), zero_lgv(x_in)
            )
            if use_process:
                # Controlled SDE discretization: mean = x + (base_drift + diff * u) dt
                fwd_mean = x_in + (base_drift + diff_t * u_fwd) * dt
            else:
                # Legacy: keep previous OU-style scaling.
                fwd_mean = x_in + (drift_term + sigma2_t * u_fwd) * dt

            key, k1 = jax.random.split(key)
            x_new = sample_kernel(k1, fwd_mean, scale)

            x_new_in = jax.lax.stop_gradient(x_new) if stop_grad else x_new

            u_bwd = bwd_state.apply_fn(
                bwd_params,
                x_new_in,
                time_bwd * jnp.ones((1,)),
                zero_lgv(x_new_in),
            )
            if use_process:
                base_drift_bwd = (
                    drift_coeff_next * (x_new_in - center)
                    if getattr(noise_schedule, "include_base_drift", False)
                    else 0.0
                )
                bwd_mean = x_new_in + (base_drift_bwd + diff_next * u_bwd) * dt
            else:
                # Legacy: keep previous OU-style scaling.
                drift_term_bwd = 0.0
                bwd_mean = x_new_in + (drift_term_bwd + sigma2_t * u_bwd) * dt

            fwd_lp = log_prob_kernel(x_new, fwd_mean, scale)
            bwd_lp = log_prob_kernel(x, bwd_mean, scale)

            log_w = log_w + (bwd_lp - fwd_lp)
            return (x_new, log_w, key), None

        (xT, log_ratio, _), _ = jax.lax.scan(step_fn, (x, log_w, key), jnp.arange(num_steps))
        return x0, xT, log_ratio

    seeds = jax.random.split(key, batch_size)
    x0, xT, log_ratio = jax.vmap(per_sample)(seeds)
    return x0, xT, log_ratio


def rnd_time_reversal_lv_no_target(
    key: jax.Array,
    model_state,  # (fwd_state, bwd_state) but only fwd_state is used
    fwd_params,
    batch_size: int,
    prior_sampler,  # callable: prior_sampler(key)->x0 [D]
    num_steps: int,
    process: VP | Langevin,
    stop_grad: bool = True,
    sde_ctrl_noise: float | None = None,
    sde_ctrl_dropout: float | None = None,
    process_center: jax.Array | None = None,
):
    """Time-reversal-style LV rollout (GBS-compatible building block).

    This mirrors `sde_sampler.losses.oc.TimeReversalLoss.simulate` for `method=lv`
    in the special case:
      - no inference control
      - diagonal diffusion independent of x
      - Euler-Maruyama discretization on a uniform grid

    Returns:
      x0: [B,D]
      xT: [B,D]
      rnd_running: [B]  (running cost + Ito integral; does NOT include terminal cost)
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

    def zero_lgv(x):
        return jnp.zeros_like(x)

    def per_sample(seed):
        key, k0 = jax.random.split(seed)
        x0 = prior_sampler(k0)  # [D]
        x = x0
        rnd = jnp.asarray(0.0, dtype=jnp.float32)

        def step_fn(carry, step_i):
            x, rnd, key = carry
            step = step_i.astype(jnp.float32)
            t = step * dt

            x_in = jax.lax.stop_gradient(x) if stop_grad else x
            diff = process.diff_coeff_t(t)
            base_drift = (
                process.drift_coeff_t(t) * (x_in - center)
                if getattr(process, "include_base_drift", False)
                else 0.0
            )

            u = fwd_state.apply_fn(
                fwd_params, x_in, t * jnp.ones((1,)), zero_lgv(x_in)
            )  # [D]
            generative_ctrl = u
            sde_ctrl = jax.lax.stop_gradient(generative_ctrl) if stop_grad else generative_ctrl
            sde_ctrl = _apply_sde_ctrl_perturbation(
                key=key,
                generative_ctrl=sde_ctrl,
                drift=base_drift,
                diff=diff,
                sde_ctrl_noise=sde_ctrl_noise,
                sde_ctrl_dropout=sde_ctrl_dropout,
            )

            # LV "change_sde_ctrl" running cost:
            #   u · (sde_ctrl - 0.5 u) dt
            running = jnp.sum(generative_ctrl * (sde_ctrl - 0.5 * generative_ctrl)) * dt

            key, k_eps = jax.random.split(key)
            eps = jax.random.normal(k_eps, shape=x.shape)
            db = eps * jnp.sqrt(dt)
            x_next = x + (base_drift + diff * sde_ctrl) * dt + diff * db

            # Ito integral term (included for LV in sde_sampler)
            ito = jnp.sum(u * db)
            rnd_next = rnd + running + ito

            return (x_next, rnd_next, key), None

        (xT, rnd_running, _), _ = jax.lax.scan(
            step_fn, (x, rnd, key), jnp.arange(num_steps)
        )
        return x0, xT, rnd_running

    seeds = jax.random.split(key, batch_size)
    x0, xT, rnd_running = jax.vmap(per_sample)(seeds)
    return x0, xT, rnd_running


def _sample_subtraj_indices(
    key: jax.Array,
    num_steps: int,
    subtraj_prob: float,
    fix_terminal: bool,
    subtraj_steps: int | None,
) -> tuple[jax.Array, jax.Array]:
    key_use, key_init, key_end = jax.random.split(jax.random.fold_in(key, 17), 3)
    use_subtraj = jax.random.uniform(key_use) < jnp.asarray(subtraj_prob, dtype=jnp.float32)
    idx_init = jax.random.randint(key_init, (), 0, num_steps)

    if fix_terminal:
        idx_end = jnp.asarray(num_steps, dtype=jnp.int32)
    elif subtraj_steps is not None:
        idx_end = jnp.minimum(
            idx_init + jnp.asarray(subtraj_steps, dtype=jnp.int32),
            jnp.asarray(num_steps, dtype=jnp.int32),
        )
    else:
        idx_end = jax.random.randint(key_end, (), idx_init + 1, num_steps + 1)

    idx_init = jnp.where(use_subtraj, idx_init, 0)
    idx_end = jnp.where(use_subtraj, idx_end, num_steps)
    return idx_init, idx_end


def sample_uniform_domain(
    key: jax.Array,
    batch_size: int,
    domain_low: jax.Array,
    domain_high: jax.Array,
) -> jax.Array:
    domain_low = jnp.asarray(domain_low)
    domain_high = jnp.asarray(domain_high)
    u = jax.random.uniform(
        key,
        shape=(batch_size, domain_low.shape[0]),
        minval=0.0,
        maxval=1.0,
        dtype=domain_low.dtype,
    )
    return domain_low + (domain_high - domain_low) * u


def box_to_tanh_latent(
    x: jax.Array,
    low: jax.Array,
    high: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    scaled = 2.0 * (x - low) / (high - low) - 1.0
    scaled = jnp.clip(scaled, -1.0 + eps, 1.0 - eps)
    return jnp.arctanh(scaled)


def _potential_log_prob(
    apply_fn,
    params,
    x: jax.Array,
    t: jax.Array,
) -> jax.Array:
    time_array = jnp.ones((x.shape[0], 1), dtype=x.dtype) * t
    zeros = jnp.zeros_like(x)
    return apply_fn(
        params,
        x,
        time_array,
        zeros,
        return_potential=True,
    )


def _lv_rollout_segment(
    key: jax.Array,
    model_state,
    fwd_params,
    x_init: jax.Array,
    step_start: jax.Array,
    step_end: jax.Array,
    num_steps: int,
    process: VP | Langevin,
    stop_grad: bool = True,
    sde_ctrl_noise: float | None = None,
    sde_ctrl_dropout: float | None = None,
    process_center: jax.Array | None = None,
):
    fwd_state, _ = model_state
    dt = jnp.asarray(process.terminal_t, dtype=jnp.float32) / jnp.asarray(
        num_steps, dtype=jnp.float32
    )
    center = (
        jnp.asarray(0.0, dtype=jnp.float32)
        if process_center is None
        else jnp.asarray(process_center, dtype=jnp.float32)
    )

    def step_fn(carry, step_i):
        x, rnd, key = carry
        step = step_i.astype(jnp.float32)
        t = step * dt

        x_in = jax.lax.stop_gradient(x) if stop_grad else x
        diff = process.diff_coeff_t(t)
        base_drift = (
            process.drift_coeff_t(t) * (x_in - center)
            if getattr(process, "include_base_drift", False)
            else 0.0
        )

        u = fwd_state.apply_fn(
            fwd_params,
            x_in,
            t * jnp.ones((x.shape[0], 1), dtype=x.dtype),
            jnp.zeros_like(x_in),
        )
        generative_ctrl = u
        sde_ctrl = jax.lax.stop_gradient(generative_ctrl) if stop_grad else generative_ctrl
        sde_ctrl = _apply_sde_ctrl_perturbation(
            key=key,
            generative_ctrl=sde_ctrl,
            drift=base_drift,
            diff=diff,
            sde_ctrl_noise=sde_ctrl_noise,
            sde_ctrl_dropout=sde_ctrl_dropout,
        )

        running = (
            jnp.sum(generative_ctrl * (sde_ctrl - 0.5 * generative_ctrl), axis=-1) * dt
        )

        key, k_eps = jax.random.split(key)
        eps = jax.random.normal(k_eps, shape=x.shape)
        db = eps * jnp.sqrt(dt)
        x_next = x + (base_drift + diff * sde_ctrl) * dt + diff * db

        ito = jnp.sum(u * db, axis=-1)
        rnd_next = rnd + running + ito

        active = (step_i >= step_start) & (step_i < step_end)
        x_out = jnp.where(active, x_next, x)
        rnd_out = jnp.where(active, rnd_next, rnd)

        return (x_out, rnd_out, key), None

    (x_end, rnd_running, key_end), _ = jax.lax.scan(
        step_fn,
        (x_init, jnp.zeros((x_init.shape[0],), dtype=x_init.dtype), key),
        jnp.arange(num_steps, dtype=jnp.int32),
    )
    return x_end, rnd_running, key_end


def rnd_time_reversal_lv_subtraj_no_target(
    key: jax.Array,
    model_state,
    fwd_params,
    batch_size: int,
    prior_sampler,
    num_steps: int,
    process: VP | Langevin,
    stop_grad: bool = True,
    fix_terminal: bool = False,
    subtraj_steps: int | None = None,
    domain_low: jax.Array | None = None,
    domain_high: jax.Array | None = None,
    use_tanh_bijection: bool = False,
    sde_ctrl_noise: float | None = None,
    sde_ctrl_dropout: float | None = None,
    process_center: jax.Array | None = None,
):
    """Random subtrajectory LV rollout.

    The segment starts from a uniformly sampled point in the current domain,
    mirroring the torch `SubtrajBridge` logic.
    """
    del prior_sampler
    if domain_low is None or domain_high is None:
        raise ValueError("Subtrajectory rollout requires domain_low and domain_high.")
    idx_init, idx_end = _sample_subtraj_indices(
        key=key,
        num_steps=num_steps,
        subtraj_prob=1.0,
        fix_terminal=fix_terminal,
        subtraj_steps=subtraj_steps,
    )
    key_start, key_rollout = jax.random.split(key)
    x_start_box = sample_uniform_domain(key_start, batch_size, domain_low, domain_high)
    x_start = jax.lax.cond(
        use_tanh_bijection,
        lambda x: box_to_tanh_latent(x, domain_low, domain_high),
        lambda x: x,
        x_start_box,
    )
    x_end, rnd_running, _ = _lv_rollout_segment(
        key=key_rollout,
        model_state=model_state,
        fwd_params=fwd_params,
        x_init=x_start,
        step_start=idx_init,
        step_end=idx_end,
        num_steps=num_steps,
        process=process,
        stop_grad=stop_grad,
        sde_ctrl_noise=sde_ctrl_noise,
        sde_ctrl_dropout=sde_ctrl_dropout,
        process_center=process_center,
    )
    return x_start, x_end, rnd_running, idx_init, idx_end


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


def lv_subtraj_loss(
    *,
    fwd_state,
    fwd_params,
    x_start: jax.Array,
    x_end: jax.Array,
    rnd_running: jax.Array,
    idx_init: jax.Array,
    idx_end: jax.Array,
    num_steps: int,
    prior_log_prob,
    target_lp_vals: jax.Array,
    max_rnd: float | None = None,
):
    dt = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)
    t_init = idx_init.astype(jnp.float32) * dt
    t_end = idx_end.astype(jnp.float32) * dt

    init_lp = _potential_log_prob(fwd_state.apply_fn, fwd_params, x_start, t_init)
    init_lp = jax.lax.stop_gradient(init_lp)
    end_lp = _potential_log_prob(fwd_state.apply_fn, fwd_params, x_end, t_end)

    initial_term = jnp.where(idx_init == 0, prior_log_prob(x_start), init_lp)
    terminal_term = jnp.where(idx_end == num_steps, target_lp_vals, end_lp)
    rnd_total = initial_term + rnd_running - terminal_term

    loss, aux, rnd = lv_loss_from_rnd(rnd_total, xT=x_end, max_rnd=max_rnd)
    seg_scale = (idx_end - idx_init).astype(jnp.float32) / jnp.asarray(
        num_steps, dtype=jnp.float32
    )
    loss = loss * seg_scale
    aux.update(
        {
            "train/subtraj_scale": seg_scale,
            "train/subtraj_idx_init": idx_init.astype(jnp.float32),
            "train/subtraj_idx_end": idx_end.astype(jnp.float32),
        }
    )
    return loss, aux, rnd


def add_subtraj_aux_defaults(
    aux: dict,
    *,
    idx_init: jax.Array | float = 0.0,
    idx_end: jax.Array | float = 0.0,
    scale: jax.Array | float = 0.0,
) -> dict:
    aux = dict(aux)
    aux.setdefault("train/subtraj_scale", jnp.asarray(scale, dtype=jnp.float32))
    aux.setdefault("train/subtraj_idx_init", jnp.asarray(idx_init, dtype=jnp.float32))
    aux.setdefault("train/subtraj_idx_end", jnp.asarray(idx_end, dtype=jnp.float32))
    return aux

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
