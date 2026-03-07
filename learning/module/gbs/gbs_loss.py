# gbs_loss.py
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


# -------------------------
# Pure GBS per-sample rollout
# -------------------------
import jax
import jax.numpy as jnp
import numpyro.distributions as npdist

def sample_kernel(key, mean, scale):
    eps = jax.random.normal(key, shape=mean.shape)
    return mean + scale * eps

def log_prob_kernel(x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)

def rnd_no_target(
    key,
    model_state,          # (fwd_state, bwd_state)
    fwd_params,
    bwd_params,
    batch_size,
    prior_sampler,        # callable: prior_sampler(key)->x0 [D]
    num_steps,
    noise_schedule,       # callable: sigma(step)->scalar
    stop_grad=True,       # for LV training, usually True
):
    fwd_state, bwd_state = model_state
    dt = 1.0 / num_steps

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
            sigma_t = noise_schedule(step)
            scale = sigma_t * jnp.sqrt(2.0 * dt)

            u_fwd = fwd_state.apply_fn(
                fwd_params, x_in, step * jnp.ones((1,)), zero_lgv(x_in)
            )
            fwd_mean = x_in + (sigma_t**2) * u_fwd * dt

            key, k1 = jax.random.split(key)
            x_new = sample_kernel(k1, fwd_mean, scale)

            x_new_in = jax.lax.stop_gradient(x_new) if stop_grad else x_new

            u_bwd = bwd_state.apply_fn(
                bwd_params, x_new_in, (step + 1.0) * jnp.ones((1,)), zero_lgv(x_new_in)
            )
            bwd_mean = x_new_in + (sigma_t**2) * u_bwd * dt

            fwd_lp = log_prob_kernel(x_new, fwd_mean, scale)
            bwd_lp = log_prob_kernel(x, bwd_mean, scale)

            log_w = log_w + (bwd_lp - fwd_lp)
            return (x_new, log_w, key), None

        (xT, log_ratio, _), _ = jax.lax.scan(step_fn, (x, log_w, key), jnp.arange(num_steps))
        return x0, xT, log_ratio

    seeds = jax.random.split(key, batch_size)
    x0, xT, log_ratio = jax.vmap(per_sample)(seeds)
    return x0, xT, log_ratio

def lv_loss_from_values(
    x0,                 # [B,D]
    xT,                 # [B,D] (not strictly needed for loss, but useful for logging)
    log_ratio,          # [B]
    prior_log_prob,     # callable: prior_log_prob(x0)->[B] or scalar per item
    target_lp_vals,     # [B]  <-- numeric values already computed!
):
    running_cost = -log_ratio                         # [B]
    terminal_cost = prior_log_prob(x0) - target_lp_vals  # [B]
    neg_elbo = running_cost + terminal_cost           # [B]
    loss = jnp.var(neg_elbo)
    aux = {
        "train/neg_elbo_mean": jnp.mean(neg_elbo),
        "train/neg_elbo_var": jnp.var(neg_elbo),
        "train/running_mean": jnp.mean(running_cost),
        "train/terminal_mean": jnp.mean(terminal_cost),
        "train/xT_mean_norm": jnp.mean(jnp.linalg.norm(xT, axis=-1)),
    }
    return loss, aux, neg_elbo