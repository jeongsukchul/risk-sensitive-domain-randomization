"""
Code for the General Bridge Sampler (GBS).
Fur further details see: https://arxiv.org/abs/2307.01198
"""

from functools import partial
from time import time

import distrax
import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training import train_state
from flax import linen as nn

# IMPORTANT: gbs_loss must provide a sampler that does NOT need target logprob inside.
# I assume you expose: rnd_no_target(...) -> (x0, xT, log_ratio)
from .gbs_loss import rnd_no_target


class PISGRADNet(nn.Module):
    dim: int

    num_layers: int = 2
    num_hid: int = 64
    outer_clip: float = 1e4
    inner_clip: float = 1e2

    weight_init: float = 1e-8
    bias_init: float = 0.0

    def setup(self):
        self.timestep_phase = self.param(
            "timestep_phase", nn.initializers.zeros_init(), (1, self.num_hid)
        )
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_hid)[None]

        self.time_coder_state = nn.Sequential(
            [nn.Dense(self.num_hid), nn.gelu, nn.Dense(self.num_hid)]
        )

        self.time_coder_grad = nn.Sequential(
            [nn.Dense(self.num_hid)]
            + [
                nn.Sequential([nn.gelu, nn.Dense(self.num_hid)])
                for _ in range(self.num_layers)
            ]
            + [
                nn.Dense(
                    self.dim,
                    kernel_init=nn.initializers.constant(self.weight_init),
                    bias_init=nn.initializers.constant(self.bias_init),
                )
            ]
        )

        self.state_time_net = nn.Sequential(
            [nn.Sequential([nn.Dense(self.num_hid), nn.gelu]) for _ in range(self.num_layers)]
            + [
                nn.Dense(
                    self.dim,
                    kernel_init=nn.initializers.constant(1e-8),
                    bias_init=nn.initializers.zeros_init(),
                )
            ]
        )

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin((self.timestep_coeff * timesteps) + self.timestep_phase)
        cos_embed_cond = jnp.cos((self.timestep_coeff * timesteps) + self.timestep_phase)
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(self, input_array, time_array, lgv_term):
        time_array_emb = self.get_fourier_features(time_array)
        if len(input_array.shape) == 1:
            time_array_emb = time_array_emb[0]

        t_net1 = self.time_coder_state(time_array_emb)
        t_net2 = self.time_coder_grad(time_array_emb)

        extended_input = jnp.concatenate((input_array, t_net1), axis=-1)
        out_state = self.state_time_net(extended_input)
        out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)

        lgv_term = jnp.clip(lgv_term, -self.inner_clip, self.inner_clip)
        out_state_p_grad = out_state + t_net2 * lgv_term
        return out_state_p_grad


# -------------------------
# LV loss from precomputed target log-prob VALUES
# -------------------------
def lv_loss_from_values(x0, xT, log_ratio, prior_log_prob, target_lp_vals):
    """
    x0: [B,D]
    xT: [B,D]
    log_ratio: [B]  (or [B,])
    prior_log_prob: callable: prior.log_prob(x0) -> [B]
    target_lp_vals: [B] numeric values (already computed outside JAX)
    """
    running_cost = -log_ratio                      # [B]
    terminal_cost = prior_log_prob(x0) - target_lp_vals  # [B]
    neg_elbo = running_cost + terminal_cost        # [B]
    loss = jnp.var(neg_elbo)
    aux = {
        "train/neg_elbo_mean": jnp.mean(neg_elbo),
        "train/neg_elbo_var": jnp.var(neg_elbo),
        "train/running_mean": jnp.mean(running_cost),
        "train/terminal_mean": jnp.mean(terminal_cost),
        "train/xT_mean_norm": jnp.mean(jnp.linalg.norm(xT, axis=-1)),
    }
    return loss, aux


def gbs_trainer(cfg, target, target_log_prob):
    """
    target: used for sampling/eval only if you want; not required for training loss now.
    target_log_prob: function that takes final_state and returns NUMERIC logprob values.
                     IMPORTANT: this is NOT traced by JAX; we call it outside jit.
    """
    key_gen = jax.random.PRNGKey(cfg.seed)
    dim = target.dim
    alg_cfg = cfg.algorithm

    # Prior
    prior = distrax.MultivariateNormalDiag(jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std)
    prior_sampler = lambda key: jnp.squeeze(prior.sample(seed=key, sample_shape=(1,)))  # [D]
    prior_log_prob = prior.log_prob  # JAX-traceable

    # Models
    fwd_model = PISGRADNet(**alg_cfg.model)
    key, key_gen = jax.random.split(key_gen)
    fwd_params = fwd_model.init(
        key,
        jnp.ones([alg_cfg.batch_size, dim]),
        jnp.ones([alg_cfg.batch_size, 1]),
        jnp.ones([alg_cfg.batch_size, dim]),
    )
    bwd_model = PISGRADNet(**alg_cfg.model)
    key, key_gen = jax.random.split(key_gen)
    bwd_params = bwd_model.init(
        key,
        jnp.ones([alg_cfg.batch_size, dim]),
        jnp.ones([alg_cfg.batch_size, 1]),
        jnp.ones([alg_cfg.batch_size, dim]),
    )

    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip(alg_cfg.grad_clip),
        optax.adam(learning_rate=alg_cfg.step_size),
    )
    fwd_state = train_state.TrainState.create(apply_fn=fwd_model.apply, params=fwd_params, tx=optimizer)
    bwd_state = train_state.TrainState.create(apply_fn=bwd_model.apply, params=bwd_params, tx=optimizer)

    # 1) JIT sampler without target
    rnd_jit = jax.jit(
        rnd_no_target,
        static_argnums=(4, 5, 6, 7),  # batch_size, prior_sampler, num_steps, noise_schedule
    )

    # 2) JIT grad of LV loss, where target_lp_vals is a normal array input
    def loss_wrapped(key, model_state, fwd_params, bwd_params, batch_size, prior_sampler, num_steps, noise_schedule, target_lp_vals):
        x0, xT, log_ratio = rnd_no_target(
            key, model_state, fwd_params, bwd_params,
            batch_size, prior_sampler, num_steps, noise_schedule,
            stop_grad=True
        )
        loss, aux = lv_loss_from_values(x0, xT, log_ratio, prior_log_prob, target_lp_vals)
        return loss, aux

    loss_grad = jax.jit(
        jax.grad(loss_wrapped, (2, 3), has_aux=True),
        static_argnums=(4, 5, 6, 7),  # keep callables static
    )

    timer = 0.0
    for step in range(alg_cfg.iters):
        key, key_gen = jax.random.split(key_gen)
        iter_time = time()

        model_state = (fwd_state, bwd_state)

        # ---- Phase A: sample xT (JAX) ----
        # We need xT to compute target log-prob values OUTSIDE JAX.
        x0, xT, log_ratio = rnd_jit(
            key,
            model_state,
            fwd_state.params,
            bwd_state.params,
            alg_cfg.batch_size,
            prior_sampler,
            alg_cfg.num_steps,
            alg_cfg.noise_schedule,
            True,  # stop_grad=True
        )

        # ---- Phase B: compute target log-prob values outside JAX ----
        # You said you can do: target_log_prob(final_state) once final_state is given.
        # Ensure it returns shape [B], float32/float64.
        target_lp_vals = target_log_prob(xT)  # MUST return a JAX array or something convertible to jnp.array
        target_lp_vals = jnp.asarray(target_lp_vals).reshape(-1)

        # ---- Phase C: compute grads (JAX), using target_lp_vals as input data ----
        (fwd_grads, bwd_grads), aux = loss_grad(
            key,
            model_state,
            fwd_state.params,
            bwd_state.params,
            alg_cfg.batch_size,
            prior_sampler,
            alg_cfg.num_steps,
            alg_cfg.noise_schedule,
            target_lp_vals,
        )

        timer += time() - iter_time

        fwd_state = fwd_state.apply_gradients(grads=fwd_grads)
        bwd_state = bwd_state.apply_gradients(grads=bwd_grads)

        # Optional: simple logging without full eval framework
        if cfg.use_wandb and (step % getattr(cfg, "log_every", 100) == 0):
            log_dict = {"stats/step": step, "stats/wallclock": timer}
            log_dict.update({k: float(v) for k, v in aux.items()})
            wandb.log(log_dict)