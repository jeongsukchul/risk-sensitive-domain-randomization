"""DORAEMON beta sampler and constrained distribution updates."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp


class DoraemonState(NamedTuple):
    x_opt: jnp.ndarray  # unconstrained params (2*ndim,)
    train_until_done: jnp.bool_  # scalar
    it: jnp.int32  # scalar


def _sigmoid_bounded(x, lb, ub):
    x = jnp.clip(x, -60.0, 60.0)  # avoid exp overflow
    return (ub - lb) / (1.0 + jnp.exp(-x)) + lb


def _inv_sigmoid_bounded(y, lb, ub):
    y = jnp.clip(y, lb + 1e-12, ub - 1e-12)
    return -jnp.log((ub - lb) / (y - lb) - 1.0)


def make_initial_state(ndim, init_beta_param, min_bound, max_bound):
    flat = jnp.ones((2 * ndim,), dtype=jnp.float32) * init_beta_param
    x_opt0 = _inv_sigmoid_bounded(flat, min_bound, max_bound)
    return DoraemonState(
        x_opt=x_opt0, train_until_done=jnp.bool_(False), it=jnp.int32(0)
    )


def _unpack_beta(x_opt, ndim, min_bound, max_bound):
    flat = _sigmoid_bounded(x_opt, min_bound, max_bound)
    a = flat[:ndim]
    b = flat[ndim:]
    return a, b


def sample_beta_on_box(key, a, b, low, high, n):
    dim = a.shape[0]
    keys = jax.random.split(key, dim * 2)
    g1 = jnp.stack(
        [jax.random.gamma(keys[i], a[i], shape=(n,)) for i in range(dim)], axis=-1
    )
    g2 = jnp.stack(
        [jax.random.gamma(keys[i + dim], b[i], shape=(n,)) for i in range(dim)], axis=-1
    )
    z = g1 / (g1 + g2 + 1e-12)
    return low + (high - low) * z


def _log_prob_beta_on_box(a, b, low, high, x):
    z = (x - low) / (high - low)
    z = jnp.clip(z, 1e-7, 1.0 - 1e-7)

    logB = jsp.special.gammaln(a) + jsp.special.gammaln(b) - jsp.special.gammaln(a + b)
    lp = (a - 1.0) * jnp.log(z) + (b - 1.0) * jnp.log1p(-z) - logB
    lp = lp - jnp.log(high - low)  # scaling Jacobian
    return jnp.sum(lp, axis=-1)  # (...,)


def _kl_beta_beta(a, b, c, d):
    # KL(Beta(a,b) || Beta(c,d)) summed over dims
    logB_ab = (
        jsp.special.gammaln(a) + jsp.special.gammaln(b) - jsp.special.gammaln(a + b)
    )
    logB_cd = (
        jsp.special.gammaln(c) + jsp.special.gammaln(d) - jsp.special.gammaln(c + d)
    )
    psi = jsp.special.digamma
    kl_dim = (
        (logB_cd - logB_ab)
        + (a - c) * psi(a)
        + (b - d) * psi(b)
        + (c + d - a - b) * psi(a + b)
    )
    return jnp.sum(kl_dim)


def _backtracking_search(x, direction, accept, *, max_ls_steps, ls_shrink):
    """Find the first bounded candidate that satisfies the supplied constraints."""

    def ls(carry):
        alpha, xbest, found, k = carry
        xcand = jnp.clip(x + alpha * direction, -60.0, 60.0)
        ok = accept(xcand)
        xbest = jnp.where(ok, xcand, xbest)
        found = found | ok
        alpha = jnp.where(ok, alpha, alpha * ls_shrink)
        return alpha, xbest, found, k + 1

    def cond(carry):
        _, _, found, k = carry
        return (k < max_ls_steps) & (~found)

    _, xnew, _, _ = jax.lax.while_loop(cond, ls, (1.0, x, False, 0))
    return xnew


def make_doraemon_update_fn(
    *,
    low,
    high,
    success_threshold: float,
    success_rate_condition: float,
    kl_upper_bound: float,
    min_bound: float,
    max_bound: float,
    train_until_performance_lb: bool = True,
    hard_performance_constraint: bool = True,
    n_steps_main: int = 25,
    n_steps_restore: int = 30,
    step_size_main: float = 0.25,  # gradient step magnitude (normalized)
    step_size_restore: float = 0.25,  # ascent step magnitude (normalized)
    max_ls_steps: int = 20,
    ls_shrink: float = 0.5,
):
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    ndim = low.shape[0]

    success_threshold = float(success_threshold)
    success_rate_condition = float(success_rate_condition)
    kl_upper_bound = float(kl_upper_bound)

    def update(state: DoraemonState, contexts: jnp.ndarray, returns: jnp.ndarray):
        """
        Pure-JAX update. Compatible with lax.scan (no SciPy).
        contexts: (N, ndim)
        returns:  (N,)
        """

        x0 = state.x_opt

        # Center distribution is the current distribution at start of update.
        a0, b0 = _unpack_beta(x0, ndim, min_bound, max_bound)
        lp0 = _log_prob_beta_on_box(a0, b0, low, high, contexts)

        succ = (returns >= success_threshold).astype(jnp.float32)

        def kl_to_center(x_opt):
            a1, b1 = _unpack_beta(x_opt, ndim, min_bound, max_bound)
            return _kl_beta_beta(a0, b0, a1, b1)  # KL(center || proposed)

        def perf(x_opt):
            a1, b1 = _unpack_beta(x_opt, ndim, min_bound, max_bound)
            lp1 = _log_prob_beta_on_box(a1, b1, low, high, contexts)
            logw = jnp.clip(lp1 - lp0, -50.0, 50.0)
            w = jnp.exp(logw)
            w = jnp.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
            return jnp.mean(w * succ)

        def obj(x_opt):
            # KL(Beta(a,b) || Uniform) == KL(Beta(a,b) || Beta(1,1))
            a, b = _unpack_beta(x_opt, ndim, min_bound, max_bound)
            one = jnp.ones_like(a)
            return _kl_beta_beta(a, b, one, one)

        perf0 = perf(x0)

        # Optionally wait for the initial policy success lower bound.
        def maybe_skip(_):
            return state  # no update

        def do_update(_):
            # -------- feasibility restoration (maximize perf under KL bound) if needed --------
            def restore_step(x):
                g = jax.grad(perf)(x)
                g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                gnorm = jnp.linalg.norm(g) + 1e-12
                direction = (step_size_restore * g) / gnorm

                perf_x = perf(x)

                def accept(xcand):
                    ok_tr = kl_to_center(xcand) <= kl_upper_bound
                    ok_inc = perf(xcand) >= perf_x
                    return ok_tr & ok_inc

                return _backtracking_search(
                    x, direction, accept, max_ls_steps=max_ls_steps, ls_shrink=ls_shrink
                )

            def run_restore(x_init):
                return jax.lax.fori_loop(
                    0, n_steps_restore, lambda i, x: restore_step(x), x_init
                )

            x_restored = jax.lax.cond(
                (hard_performance_constraint & (perf0 < success_rate_condition)),
                lambda _: run_restore(x0),
                lambda _: x0,
                operand=None,
            )

            # -------- main objective minimization under constraints --------
            def main_step(x):
                f = obj(x)
                g = jax.grad(obj)(x)
                g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                gnorm = jnp.linalg.norm(g) + 1e-12
                direction = -(step_size_main * g) / gnorm

                def accept(xcand):
                    ok_tr = kl_to_center(xcand) <= kl_upper_bound
                    ok_perf = (
                        (perf(xcand) >= success_rate_condition)
                        if hard_performance_constraint
                        else True
                    )
                    ok_dec = obj(xcand) <= f
                    return ok_tr & ok_perf & ok_dec

                return _backtracking_search(
                    x, direction, accept, max_ls_steps=max_ls_steps, ls_shrink=ls_shrink
                )

            def run_main(x_init):
                return jax.lax.fori_loop(
                    0, n_steps_main, lambda i, x: main_step(x), x_init
                )

            x_new = run_main(x_restored)
            return DoraemonState(
                x_opt=x_new, train_until_done=jnp.bool_(True), it=state.it + 1
            )

        new_state = jax.lax.cond(
            (
                train_until_performance_lb
                & (~state.train_until_done)
                & (perf0 < success_rate_condition)
            ),
            maybe_skip,
            do_update,
            operand=None,
        )

        return new_state

    return update
