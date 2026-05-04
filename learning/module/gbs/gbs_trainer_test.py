"""
Code for the General Bridge Sampler (GBS).
Fur further details see: https://arxiv.org/abs/2307.01198
"""

from functools import partial
from pathlib import Path
from time import time
from typing import Optional, Sequence

import distrax
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import matplotlib.pyplot as plt
from flax.training import train_state
from flax import linen as nn
from matplotlib.ticker import MaxNLocator
from tqdm import trange
import logging
import plotly.graph_objects as go
from learning.module.gbs.sinkhorn_metrics import (
    effective_sample_size_from_log_weights,
    interatomic_wasserstein_1d,
    sinkhorn_distance,
)

# IMPORTANT: gbs_loss must provide a sampler that does NOT need target logprob inside.
# I assume you expose: rnd_no_target(...) -> (x0, xT, log_ratio)
from .gbs_loss_test import (
    dds_logw_from_buffer,
    dds_lv_loss_from_values,
    dds_re_loss_from_values,
    VP,
    rnd_no_target,
    simul_forward_sde_for_buffer,
    lv_loss_from_values,
    re_loss_from_values,
    solve_trust_region_lambda_from_logw,
    solve_trust_region_lambda_grid_golden,
    tr_lv_loss_from_buffer,
)


def _normalize_loss_mode(loss_mode: str) -> str:
    normalized = loss_mode.lower().replace("-", "_")
    aliases = {
        "lv": "dds_lv",
        "time_reversal_lv": "dds_lv",
        "dds_euler_lv": "dds_lv",
        "euler_dds_lv": "dds_lv",
        "trust_region_lv": "tr_dds_lv",
        "tr_lv": "tr_dds_lv",
        "trust_region_dds_lv": "tr_dds_lv",
    }
    return aliases.get(normalized, normalized)


def _aux_scalar(value):
    arr = jnp.asarray(value)
    if arr.ndim > 0:
        arr = jnp.nanmean(arr)
    return float(jax.device_get(arr))
def _to_python_scalar(x):
    if isinstance(x, (float, int)):
        return x
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return float(x)


def _last_hist_value(hist: dict[str, list], key: str):
    if key not in hist or len(hist[key]) == 0:
        return None
    value = hist[key][-1]
    try:
        value = _to_python_scalar(value)
    except Exception:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _select_loss_key_for_logging(loss_mode: str, hist: dict[str, list]) -> str | None:
    loss_mode = _normalize_loss_mode(loss_mode)
    if loss_mode == "tr_dds_lv":
        candidates = ["train/tr_lv_var", "train/tr_lv_mean"]
    else:
        candidates = ["train/neg_elbo_var", "train/neg_elbo_mean"]
    for key in candidates:
        if key in hist:
            return key
    return None


def plot_evolution_plotly(
    ts,
    xs,
    dim: int = 0,
    ntraj: int = 50,
    domain=None,
    decimals: int = 6,
):
    """
    ts: [T]
    xs: [T, N, D]
    domain: optional array-like of shape [D, 2]
    """
    ts = np.asarray(ts)
    xs = np.asarray(xs)

    fig = go.Figure()

    if domain is not None:
        domain = np.asarray(domain)
        fig.update_layout(yaxis_range=domain[dim].tolist())

    trajs = xs[:, :, dim].T  # [N, T]

    mask = np.isfinite(trajs).all(axis=1)
    discard = int(mask.size - mask.sum())
    if discard > 0:
        logging.warning("Filtering %d trajectories with non-finite values.", discard)

    if discard < mask.size:
        trajs = trajs[mask][:ntraj]

        final_vals = trajs[:, -1]
        denom = 1e-8 + final_vals.max() - final_vals.min()
        hues = 100.0 * (final_vals - final_vals.min()) / denom

        x = np.round(ts, decimals=decimals)
        for traj, hue in zip(trajs, hues):
            hue_degrees = 3.6 * float(np.round(hue, decimals))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.round(traj, decimals=decimals),
                    mode="lines",
                    line={
                        "color": f"hsl({hue_degrees:.6f},100%,50%)",
                        "width": 0.4,
                    },
                    showlegend=False,
                )
            )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="t",
        yaxis_title=f"x[{dim}]",
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig

def gbs_history_keys(loss_mode: str) -> list[str]:
    loss_mode = _normalize_loss_mode(loss_mode)
    if loss_mode in ("dis", "dis_lv", "dds", "dds_lv"):
        return [
            "train/neg_elbo_mean",
            "train/neg_elbo_var",
            "train/running_mean",
            "train/terminal_mean",
            "train/xT_mean_norm",
            "train/n_filtered",
        ]
    if loss_mode == "tr_dds_lv":
        return [
            "train/tr_lv_mean",
            "train/tr_lv_var",
            "train/tr_lv_lambda",
            "train/tr_lv_alpha",
            "train/logw_mean",
            "train/logw_var",
            "train/xT_mean_norm",
            "train/n_filtered",
        ]
    raise ValueError(f"Unknown loss_mode: {loss_mode}")


def make_gbs_sampler_jit(
    gbs_loss_mode: str,
    batch_size,
    prior_sampler,
    num_steps,
    process,
    stop_grad,
    gbs_center,
    target_loggrad_latent_fn=None,
    use_lgv: bool = False,
    integrator_type : str = "euler",
):
    gbs_loss_mode = _normalize_loss_mode(gbs_loss_mode)
    use_reference_ctrl = gbs_loss_mode not in ("dis", "dis_lv")
    @jax.jit
    def rnd_wrapped(
        key,
        model_state,
        fwd_params,
        bwd_params,
        current_lambda,
        current_policy_p,
    ):
        target_loggrad_fn = None
        if target_loggrad_latent_fn is not None:
            target_loggrad_fn = lambda x: target_loggrad_latent_fn(
                x, current_lambda, current_policy_p
            )

        return rnd_no_target(
            key,
            model_state,
            fwd_params,
            bwd_params,
            batch_size,
            prior_sampler,
            num_steps,
            process,
            use_reference_ctrl,
            stop_grad,
            gbs_center,
            use_ito= True if "lv" in gbs_loss_mode else False,
            target_loggrad_fn=target_loggrad_fn,
            use_lgv=use_lgv,
            integrator_type=integrator_type,
        )

    return rnd_wrapped


def make_gbs_train_step_jit(
    gbs_batch_size: int,
    gbs_prior_sampler,
    gbs_num_steps: int,
    gbs_process,
    gbs_loss_mode: str,
    gbs_center,
    gbs_use_tanh_bijection: bool,
    gbs_logabsdet,
    gbs_prior,
    gbs_max_rnd: float,
    gbs_trust_region_bound: float,
    gbs_trust_region_lambda_max: float,
    gbs_trust_region_lambda_grid_size: int,
    gbs_minibatch_size: int = 2000,
    gbs_minibatch_steps: int = 400,
    gbs_target_loggrad_latent_fn=None,
    gbs_use_lgv: bool = False,
    integrator_type: str = "euler",
):
    gbs_loss_mode = _normalize_loss_mode(gbs_loss_mode)

    def buffer_loss_wrapped(
        model_state,
        fwd_params,
        bwd_params,
        paths,
        dbs,
        logw_behavior,
        target_lp_vals_mb,
        fixed_lambda,
        current_lambda,
        current_policy_p,
    ):
        del bwd_params
        target_loggrad_fn = None
        if gbs_target_loggrad_latent_fn is not None:
            target_loggrad_fn = lambda x: gbs_target_loggrad_latent_fn(
                x, current_lambda, current_policy_p
            )
        loss, aux, _ = tr_lv_loss_from_buffer(
            model_state=model_state,
            behavior_params=jax.lax.stop_gradient(fwd_params),
            candidate_params=fwd_params,
            paths=paths,
            dbs=dbs,
            logw_behavior=logw_behavior,
            num_steps=gbs_num_steps,
            process=gbs_process,
            max_rnd=gbs_max_rnd,
            process_center=gbs_center,
            fixed_lambda=fixed_lambda,
            target_lp_vals=target_lp_vals_mb,
            target_loggrad_fn=target_loggrad_fn,
            use_lgv=gbs_use_lgv,
        )
        return loss, aux

    buffer_loss_grad = jax.jit(jax.grad(buffer_loss_wrapped, 1, has_aux=True))

    @jax.jit
    def gbs_train_step_jit(
        key,
        fwd_state,
        bwd_state,
        target_lnpdf,
        current_lambda,
        current_policy_p,
    ):
        target_lp_vals = target_lnpdf
        model_state = (fwd_state, bwd_state)

        if gbs_loss_mode == "tr_dds_lv":
            paths, dbs = simul_forward_sde_for_buffer(
                key,
                model_state,
                fwd_state.params,
                gbs_batch_size,
                gbs_prior_sampler,
                gbs_num_steps,
                gbs_process,
                gbs_center,
                target_loggrad_fn=(
                    None
                    if gbs_target_loggrad_latent_fn is None
                    else lambda x: gbs_target_loggrad_latent_fn(
                        x, current_lambda, current_policy_p
                    )
                ),
                use_lgv=gbs_use_lgv,
                integrator_type=integrator_type,
            )
            xT = paths[:, -1, :]
            if gbs_use_tanh_bijection:
                target_lp_vals = target_lp_vals + gbs_logabsdet(xT)
            logw_behavior = dds_logw_from_buffer(
                model_state=model_state,
                behavior_params=fwd_state.params,
                paths=paths,
                dbs=dbs,
                num_steps=gbs_num_steps,
                process=gbs_process,
                target_lp_vals=target_lp_vals,
                process_center=gbs_center,
                target_loggrad_fn=(
                    None
                    if gbs_target_loggrad_latent_fn is None
                    else lambda x: gbs_target_loggrad_latent_fn(
                        x, current_lambda, current_policy_p
                    )
                ),
                use_lgv=gbs_use_lgv,
            )
            fixed_lambda = solve_trust_region_lambda_grid_golden(
                logw_behavior,
                trust_region_bound=gbs_trust_region_bound,
                lambda_max=gbs_trust_region_lambda_max,
                grid_size=gbs_trust_region_lambda_grid_size,
            )
            mb_size = min(int(gbs_minibatch_size), int(gbs_batch_size))
            aux_keys = tuple(gbs_history_keys(gbs_loss_mode))
            init_aux = {k: jnp.asarray(0.0, dtype=jnp.float32) for k in aux_keys}

            def body_fn(_, carry):
                key_inner, fwd_state_inner, aux_acc = carry
                key_inner, k_idx = jax.random.split(key_inner)
                idx = jax.random.randint(k_idx, (mb_size,), 0, gbs_batch_size)
                grads, aux = buffer_loss_grad(
                    (fwd_state_inner, bwd_state),
                    fwd_state_inner.params,
                    bwd_state.params,
                    paths[idx],
                    dbs[idx],
                    logw_behavior[idx],
                    target_lp_vals[idx],
                    fixed_lambda,
                    current_lambda,
                    current_policy_p,
                )
                fwd_state_inner = fwd_state_inner.apply_gradients(grads=grads)
                aux_acc = {k: aux_acc[k] + jnp.asarray(aux[k], dtype=jnp.float32) for k in aux_keys}
                return key_inner, fwd_state_inner, aux_acc

            key_out, new_fwd_state, aux_sum = jax.lax.fori_loop(
                0,
                int(gbs_minibatch_steps),
                body_fn,
                (key, fwd_state, init_aux),
            )
            del key_out
            aux_mean = {
                k: aux_sum[k] / jnp.asarray(float(gbs_minibatch_steps), dtype=jnp.float32)
                for k in aux_keys
            }
            return new_fwd_state, bwd_state, aux_mean
        else:
            def loss_from_params(fwd_params):
                local_target_lp = target_lp_vals
                use_reference_ctrl = gbs_loss_mode in ("dds", "dds_lv")
                target_loggrad_fn = None
                if gbs_target_loggrad_latent_fn is not None:
                    target_loggrad_fn = lambda x: gbs_target_loggrad_latent_fn(
                        x, current_lambda, current_policy_p
                    )

                x0, xT, log_ratio = rnd_no_target(
                    key,
                    (fwd_state, bwd_state),
                    fwd_params,
                    None,
                    gbs_batch_size,
                    gbs_prior_sampler,
                    gbs_num_steps,
                    gbs_process,
                    use_reference_ctrl,
                    True,
                    gbs_center,
                    use_ito= True if "lv" in gbs_loss_mode else False,
                    target_loggrad_fn=target_loggrad_fn,
                    use_lgv=gbs_use_lgv,
                    integrator_type=integrator_type,
                )
                if gbs_use_tanh_bijection:
                    local_target_lp = local_target_lp + gbs_logabsdet(xT)
                if gbs_loss_mode == "dis":
                    loss, aux, _ = re_loss_from_values(
                        x0, xT, log_ratio, gbs_prior.log_prob, local_target_lp, max_rnd=gbs_max_rnd
                    )
                elif gbs_loss_mode == "dis_lv":
                    loss, aux, _ = lv_loss_from_values(
                        x0, xT, log_ratio, gbs_prior.log_prob, local_target_lp, max_rnd=gbs_max_rnd
                    )
                elif gbs_loss_mode == "dds":
                    loss, aux, _ = dds_re_loss_from_values(
                        x0, xT, log_ratio, gbs_process, local_target_lp, process_center=gbs_center, max_rnd=gbs_max_rnd
                    )
                else:
                    loss, aux, _ = dds_lv_loss_from_values(
                        x0, xT, log_ratio, gbs_process, local_target_lp, process_center=gbs_center, max_rnd=gbs_max_rnd
                    )
                return loss, aux

            grads, aux = jax.grad(loss_from_params, has_aux=True)(fwd_state.params)
            new_fwd_state = fwd_state.apply_gradients(grads=grads)
            return new_fwd_state, bwd_state, aux

    return gbs_train_step_jit


class PISGRADNet(nn.Module):
    dim: int

    num_layers: int = 6
    num_hid: int = 256
    outer_clip: float = 1e4
    inner_clip: float = 1e2
    gbs_scale_diff : float = 1.
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

    def __call__(self, input_array, time_array, lgv_term, return_potential: bool = False):
        if return_potential:
            raise ValueError("PISGRADNet does not expose a scalar potential.")
        time_array_emb = self.get_fourier_features(time_array)
        if len(input_array.shape) == 1:
            time_array_emb = time_array_emb[0]

        t_net1 = self.time_coder_state(time_array_emb)
        t_net2 = self.time_coder_grad(time_array_emb)

        extended_input = jnp.concatenate((input_array, t_net1), axis=-1)
        out_state = self.state_time_net(extended_input)
        out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)

        lgv_term = jnp.clip(lgv_term, -self.inner_clip, self.inner_clip)
        out_state_p_grad = out_state + t_net2 * (lgv_term + input_array / self.gbs_scale_diff)
        return out_state_p_grad


class PotentialPISGRADNet(nn.Module):
    """Potential-based control model for GBS.

    This model parameterizes a scalar potential phi(t, x) and returns its
    gradient with respect to x. It keeps the same call signature as
    ``PISGRADNet`` so it can be swapped into the existing GBS trainer without
    changing the rollout code.

    Note:
    - ``lgv_term`` is accepted for API compatibility but intentionally unused.
    - The sampler already applies the process diffusion coefficient outside the
      model, so this class returns only grad_x phi(t, x).
    """

    dim: int

    num_layers: int = 6
    num_hid: int = 256
    outer_clip: float = 1e4

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

        self.potential_net = nn.Sequential(
            [nn.Sequential([nn.Dense(self.num_hid), nn.gelu]) for _ in range(self.num_layers)]
            + [
                nn.Dense(
                    1,
                    kernel_init=nn.initializers.constant(self.weight_init),
                    bias_init=nn.initializers.constant(self.bias_init),
                )
            ]
        )

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin((self.timestep_coeff * timesteps) + self.timestep_phase)
        cos_embed_cond = jnp.cos((self.timestep_coeff * timesteps) + self.timestep_phase)
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def _potential_single(self, x_single, t_emb_single):
        extended_input = jnp.concatenate((x_single, t_emb_single), axis=-1)
        out = self.potential_net(extended_input)
        return jnp.squeeze(out, axis=-1)

    def potential(self, input_array, time_array):
        time_array_emb = self.get_fourier_features(time_array)
        t_net = self.time_coder_state(time_array_emb)

        if len(input_array.shape) == 1:
            if time_array_emb.ndim > 1:
                t_net = t_net[0]
            return self._potential_single(input_array, t_net)

        pot_fn = jax.vmap(self._potential_single, in_axes=(0, 0))
        return pot_fn(input_array, t_net)

    def __call__(self, input_array, time_array, lgv_term, return_potential: bool = False):
        del lgv_term
        potential = self.potential(input_array, time_array)
        if return_potential:
            return jnp.clip(potential, -self.outer_clip, self.outer_clip)

        if len(input_array.shape) == 1:
            grad_fn = jax.grad(self._potential_single, argnums=0)
            time_array_emb = self.get_fourier_features(time_array)
            t_net = self.time_coder_state(time_array_emb)
            if time_array_emb.ndim > 1:
                t_net = t_net[0]
            out = grad_fn(input_array, t_net)
        else:
            time_array_emb = self.get_fourier_features(time_array)
            t_net = self.time_coder_state(time_array_emb)
            grad_fn = jax.vmap(jax.grad(self._potential_single, argnums=0), in_axes=(0, 0))
            out = grad_fn(input_array, t_net)

        return jnp.clip(out, -self.outer_clip, self.outer_clip)


def make_gbs_model(model_type: str = "pisgrad", **model_kwargs):
    model_type = model_type.lower()
    if model_type == "pisgrad":
        return PISGRADNet(**model_kwargs)
    if model_type == "potential":
        return PotentialPISGRADNet(**model_kwargs)
    raise ValueError(f"Unknown GBS model_type: {model_type}")


def plot_sample_density_2d(
    samples,
    low,
    high,
    bins: int = 80,
    levels: int = 20,
    max_scatter_points: int = 300,
    dims: Optional[Sequence[int]] = None,
    title: str = "GBS sample density",
):
    """Sample-density view with pairwise marginal support for dim>2."""
    s = np.asarray(samples)
    low = np.asarray(low)
    high = np.asarray(high)
    if s.ndim != 2:
        raise ValueError(f"samples must be [N,D], got {s.shape}")
    d = s.shape[1]
    if low.shape[0] != d or high.shape[0] != d:
        raise ValueError(
            f"low/high shape mismatch: got low={low.shape}, high={high.shape}, D={d}"
        )

    if dims is None:
        dims = tuple(range(d))
    else:
        dims = tuple(dims)
    if len(dims) < 2:
        raise ValueError("Need at least 2 dimensions for pairwise plotting.")

    rng = np.random.default_rng(0)
    n_plot = min(max_scatter_points, s.shape[0])
    idx = rng.choice(s.shape[0], size=n_plot, replace=False)
    s_plot = s[idx]

    # 2D shortcut keeps backward compatibility with existing callers.
    if len(dims) == 2:
        i, j = dims[0], dims[1]
        density, xedges, yedges = np.histogram2d(
            s[:, i],
            s[:, j],
            bins=bins,
            range=[[low[i], high[i]], [low[j], high[j]]],
            density=True,
        )
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(xc, yc, indexing="xy")

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ctf = ax.contourf(X, Y, density.T, levels=levels, cmap="viridis")
        fig.colorbar(ctf, ax=ax, label="estimated density")
        h_scatter = ax.scatter(
            s_plot[:, i], s_plot[:, j], c="r", alpha=0.35, marker="x", label="samples"
        )
        ax.set_xlim(float(low[i]), float(high[i]))
        ax.set_ylim(float(low[j]), float(high[j]))
        ax.set_xlabel(f"dim {i}")
        ax.set_ylabel(f"dim {j}")
        ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
        fig.subplots_adjust(right=0.80)
        ax.legend(
            handles=[h_scatter],
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            borderaxespad=0.0,
        )
        return fig, ax

    # dim>2: pairwise marginal-style grid (lower triangle + diagonal histograms).
    k = len(dims)
    fig, axes = plt.subplots(k, k, figsize=(3.5 * k, 3.5 * k), squeeze=False)
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    mappable = None
    scatter_handle = None

    for row, dim_i in enumerate(dims):
        for col, dim_j in enumerate(dims):
            ax = axes[row, col]

            if row < col:
                ax.axis("off")
                continue

            if row == col:
                ax.hist(s[:, dim_i], bins=30, density=True, alpha=0.8)
                ax.set_xlim(float(low[dim_i]), float(high[dim_i]))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
                if row == k - 1:
                    ax.set_xlabel(f"dim {dim_i}")
                else:
                    ax.set_xticklabels([])
                ax.set_yticklabels([])
                continue

            density, xedges, yedges = np.histogram2d(
                s[:, dim_j],
                s[:, dim_i],
                bins=bins,
                range=[[low[dim_j], high[dim_j]], [low[dim_i], high[dim_i]]],
                density=True,
            )
            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])
            X, Y = np.meshgrid(xc, yc, indexing="xy")
            ctf = ax.contourf(X, Y, density.T, levels=levels, cmap="viridis")
            mappable = ctf
            scatter_handle = ax.scatter(
                s_plot[:, dim_j],
                s_plot[:, dim_i],
                c="r",
                alpha=0.25,
                s=8,
                marker="x",
                label="samples",
            )
            ax.set_xlim(float(low[dim_j]), float(high[dim_j]))
            ax.set_ylim(float(low[dim_i]), float(high[dim_i]))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            if row == k - 1:
                ax.set_xlabel(f"dim {dim_j}")
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(f"dim {dim_i}")
            else:
                ax.set_yticklabels([])

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("estimated density")
    if scatter_handle is not None:
        fig.legend(
            handles=[scatter_handle],
            loc="center left",
            bbox_to_anchor=(0.92, 0.5),
            frameon=True,
            borderaxespad=0.0,
        )

    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 0.98])
    return fig, axes




def run_gbs(
    *,
    low,
    high,
    dim: int,
    function_evaluations: int,
    buffer_size: int,
    num_steps: int,
    lr: float,
    init_std: float,
    seed: int,
    beta: float,
    tau: float,
    q: float,
    initial_p: float | None,
    p_update_freq: int,
    p_ema_alpha: float,
    p_jump_prob: float,
    loss_mode: str,
    sinkhorn_num_samples: int,
    n_particles: int | None,
    n_spatial_dim: int,
    save_dir,
    gif_path,
    snap_iters,
    model_type: str,
    model_num_layers: int,
    model_num_hid: int,
    gbs_scale_diff : float,
    final_sample_size: int,
    max_rnd: float,
    trust_region_bound: float,
    trust_region_lambda_max: float,
    trust_region_lambda_grid_size: int,
    minibatch_size: int,
    minibatch_steps: int,
    return_snapshots: bool,
    snapshot_sample_size: int | None,
    max_metric_eval_points: int | None,
    process,
    latent_prior_loc,
    process_center,
    clip_prior_without_tanh: bool,
    use_tanh_bijection: bool,
    logabsdet_fn,
    to_box,
    target_logprob_box_fn,
    sample_mean_fn,
    compute_metrics_fn,
    sample_reference_fn,
    energy_w2_fn,
    optimal_p_fn,
    update_p_fn,
    target_loggrad_latent_fn=None,
    use_lgv: bool = False,
    # ---- new wandb args ----
    use_wandb: bool = False,
    wandb_log_every: int = 1,
    wandb_prefix: str = "gbs",
    wandb_plot_ntraj: int = 50,
    metric_eval_samples: int | None = None,
):
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    process_center = jnp.asarray(process_center, dtype=jnp.float32)
    latent_prior_loc = jnp.asarray(latent_prior_loc, dtype=jnp.float32)
    outer_iterations = int(function_evaluations) // int(buffer_size)
    if snap_iters is None:
        snap_iters = []
    if snapshot_sample_size is None:
        snapshot_sample_size = final_sample_size
    if metric_eval_samples is None:
        metric_eval_samples = sinkhorn_num_samples
    if int(metric_eval_samples) <= 0:
        raise ValueError(f"metric_eval_samples must be positive, got {metric_eval_samples}")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metric_eval_iters = set()
    if max_metric_eval_points is None or outer_iterations <= max_metric_eval_points:
        metric_eval_iters = set(range(outer_iterations))
    else:
        metric_eval_iters = set(
            np.unique(np.linspace(0, outer_iterations - 1, max_metric_eval_points).astype(int)).tolist()
        )

    key = jax.random.PRNGKey(seed)
    key, k_p0 = jax.random.split(key)
    if initial_p is None:
        p = float(jax.random.uniform(k_p0, minval=0.0, maxval=1.0))
    else:
        p = float(np.clip(initial_p, 0.0, 1.0))

    if n_particles is None:
        if dim % n_spatial_dim != 0:
            raise ValueError(f"dim={dim} must be divisible by n_spatial_dim={n_spatial_dim}")
        n_particles = dim // n_spatial_dim
    if n_particles * n_spatial_dim != dim:
        raise ValueError(
            f"n_particles * n_spatial_dim must equal dim, got {n_particles} * {n_spatial_dim} != {dim}"
        )

    prior = distrax.MultivariateNormalDiag(
        loc=latent_prior_loc,
        scale_diag=jnp.ones(dim, dtype=jnp.float32) * init_std,
    )
    if clip_prior_without_tanh:
        prior_sampler = lambda k: jnp.clip(
            jnp.squeeze(prior.sample(seed=k, sample_shape=(1,))), low, high
        )
    else:
        prior_sampler = lambda k: jnp.squeeze(prior.sample(seed=k, sample_shape=(1,)))

    model_cfg = dict(
        model_type=model_type,
        dim=dim,
        num_layers=model_num_layers,
        num_hid=model_num_hid,
        gbs_scale_diff = gbs_scale_diff,
    )
    fwd_model = make_gbs_model(**model_cfg)
    bwd_model = make_gbs_model(**model_cfg)
    key, k1, k2 = jax.random.split(key, 3)
    fwd_params = fwd_model.init(
        k1, jnp.ones([buffer_size, dim]), jnp.ones([buffer_size, 1]), jnp.ones([buffer_size, dim])
    )
    bwd_params = bwd_model.init(
        k2, jnp.ones([buffer_size, dim]), jnp.ones([buffer_size, 1]), jnp.ones([buffer_size, dim])
    )
    opt = optax.chain(optax.zero_nans(), optax.clip(1.0), optax.adam(lr))
    fwd_state = train_state.TrainState.create(apply_fn=fwd_model.apply, params=fwd_params, tx=opt)
    bwd_state = train_state.TrainState.create(apply_fn=bwd_model.apply, params=bwd_params, tx=opt)

    loss_mode = _normalize_loss_mode(loss_mode)
    integrator_type = "euler" if "dis" in loss_mode else "exp"
    sampler_jit = make_gbs_sampler_jit(
        loss_mode,
        buffer_size,
        prior_sampler,
        num_steps,
        process,
        True,
        process_center,
        target_loggrad_latent_fn=target_loggrad_latent_fn,
        use_lgv=use_lgv,
        integrator_type=integrator_type,
    )
    snapshot_sampler_jit = sampler_jit
    if int(snapshot_sample_size) != int(buffer_size):
        snapshot_sampler_jit = make_gbs_sampler_jit(
            loss_mode,
            int(snapshot_sample_size),
            prior_sampler,
            num_steps,
            process,
            True,
            process_center,
            target_loggrad_latent_fn=target_loggrad_latent_fn,
            use_lgv=use_lgv,
            integrator_type=integrator_type,
        )
    final_sampler_jit = snapshot_sampler_jit
    if int(final_sample_size) != int(snapshot_sample_size):
        final_sampler_jit = make_gbs_sampler_jit(
            loss_mode,
            int(final_sample_size),
            prior_sampler,
            num_steps,
            process,
            True,
            process_center,
            target_loggrad_latent_fn=target_loggrad_latent_fn,
            use_lgv=use_lgv,
            integrator_type=integrator_type,
        )
    train_step_jit = make_gbs_train_step_jit(
        gbs_batch_size=buffer_size,
        gbs_prior_sampler=prior_sampler,
        gbs_num_steps=num_steps,
        gbs_process=process,
        gbs_loss_mode=loss_mode,
        gbs_center=process_center,
        gbs_use_tanh_bijection=use_tanh_bijection,
        gbs_logabsdet=logabsdet_fn,
        gbs_prior=prior,
        gbs_max_rnd=max_rnd,
        gbs_trust_region_bound=trust_region_bound,
        gbs_trust_region_lambda_max=trust_region_lambda_max,
        gbs_trust_region_lambda_grid_size=trust_region_lambda_grid_size,
        gbs_minibatch_size=minibatch_size,
        gbs_minibatch_steps=minibatch_steps,
        gbs_target_loggrad_latent_fn=target_loggrad_latent_fn,
        gbs_use_lgv=use_lgv,
        integrator_type=integrator_type,
    )

    hist = {k: [] for k in gbs_history_keys(loss_mode)}
    hist.update(
        {
            "target/p": [],
            "target/lambda": [],
            "target/sample_mean": [],
            "target/forward_kl": [],
            "target/reverse_kl": [],
            "target/wasserstein": [],
            "target/sinkhorn": [],
            "target/ess": [],
            "target/energy_w2": [],
            "target/interatomic_w2": [],
            "target/target_mean": [],
            "target/optimal_p": [],
            "target/p_updated": [],
            "target/p_jumped": [],
            "target/p_base": [],
            "target/p_ema": [],
        }
    )

    frames = []
    snapshot_records = []
    
    for t in trange(outer_iterations):
        current_lambda = beta * p
        current_policy_p = p
        key, k_step = jax.random.split(key)
        _, xT_latent, _ = sampler_jit(
            k_step,
            (fwd_state, bwd_state),
            fwd_state.params,
            bwd_state.params,
            current_lambda,
            current_policy_p,
        )
        if use_wandb and wandb.run is not None and wandb_log_every > 0 and (t % wandb_log_every == 0):
            key, k_plot = jax.random.split(key)
            paths_for_plot, _ = simul_forward_sde_for_buffer(
                k_plot,
                (fwd_state, bwd_state),
                fwd_state.params,
                min(int(wandb_plot_ntraj), int(buffer_size)),
                prior_sampler,
                num_steps,
                process,
                process_center,
                target_loggrad_fn=(
                    None
                    if target_loggrad_latent_fn is None
                    else lambda x: target_loggrad_latent_fn(
                        x, current_lambda, current_policy_p
                    )
                ),
                use_lgv=use_lgv,
                integrator_type=integrator_type,
            )
            # simul_forward_sde_for_buffer returns [N, T, D] or [N, T+1, D]
            # convert to [T, N, D] for plotting
            traj_xs = np.asarray(to_box(paths_for_plot)).transpose(1, 0, 2)
            traj_ts = np.linspace(0.0, 1.0, traj_xs.shape[0], dtype=np.float32)
        xT = to_box(xT_latent)
        target_lp_vals = jnp.asarray(
            target_logprob_box_fn(xT, current_lambda, current_policy_p)
        ).reshape(-1)
        fwd_state, bwd_state, aux = train_step_jit(
            k_step,
            fwd_state,
            bwd_state,
            target_lp_vals,
            current_lambda,
            current_policy_p,
        )
        aux_mean = {k: _aux_scalar(aux[k]) for k in gbs_history_keys(loss_mode)}
        for k in gbs_history_keys(loss_mode):
            hist[k].append(aux_mean[k])

        n_metric = min(int(metric_eval_samples), int(xT.shape[0]))
        metric_xT = xT[:n_metric]
        sample_mean_g = float(sample_mean_fn(metric_xT, current_policy_p))
        key, k_metric, k_update = jax.random.split(key, 3)
        if t in metric_eval_iters:
            forward_kl, reverse_kl, wasserstein = compute_metrics_fn(
                metric_xT, current_lambda, k_metric, current_policy_p
            )
            ess = effective_sample_size_from_log_weights(
                target_logprob_box_fn(metric_xT, current_lambda, current_policy_p)
            )
            key, k_sink = jax.random.split(key)
            sinkhorn_target = None if sample_reference_fn is None else sample_reference_fn(
                k_sink, current_lambda, metric_xT.shape, current_policy_p
            )
            n_sink = min(int(sinkhorn_num_samples), int(metric_xT.shape[0]))
            if sinkhorn_target is None:
                sinkhorn = float("nan")
                energy_w2 = float("nan")
                interatomic_w2 = float("nan")
            else:
                sinkhorn = sinkhorn_distance(metric_xT[:n_sink], sinkhorn_target[:n_sink])
                energy_w2 = float(
                    energy_w2_fn(metric_xT[:n_sink], sinkhorn_target[:n_sink], current_lambda, current_policy_p)
                )
                if dim % n_spatial_dim == 0:
                    interatomic_w2 = float(
                        interatomic_wasserstein_1d(
                            metric_xT[:n_sink],
                            sinkhorn_target[:n_sink],
                            n_particles=n_particles,
                            n_spatial_dim=n_spatial_dim,
                        )
                    )
                else:
                    interatomic_w2 = float("nan")
        else:
            forward_kl = reverse_kl = wasserstein = sinkhorn = ess = energy_w2 = interatomic_w2 = float("nan")
        optimal_p, target_mean = optimal_p_fn(current_lambda, tau, q)

        hist["target/p"].append(float(p))
        hist["target/lambda"].append(float(current_lambda))
        hist["target/sample_mean"].append(sample_mean_g)
        hist["target/forward_kl"].append(forward_kl)
        hist["target/reverse_kl"].append(reverse_kl)
        hist["target/wasserstein"].append(wasserstein)
        hist["target/sinkhorn"].append(sinkhorn)
        hist["target/ess"].append(ess)
        hist["target/energy_w2"].append(energy_w2)
        hist["target/interatomic_w2"].append(interatomic_w2)
        hist["target/target_mean"].append(target_mean)
        hist["target/optimal_p"].append(optimal_p)
        should_update_p = p_update_freq > 0 and ((t + 1) % p_update_freq == 0)
        hist["target/p_updated"].append(float(should_update_p))
        hist["target/p_jumped"].append(0.0)
        hist["target/p_base"].append(float(jax.nn.sigmoid(tau * (sample_mean_g - q))))
        hist["target/p_ema"].append(float(p))
        if should_update_p:
            p, base_p, ema_p, jumped = update_p_fn(
                prev_p=p,
                sample_mean_g=sample_mean_g,
                tau=tau,
                q=q,
                ema_alpha=p_ema_alpha,
                jump_prob=p_jump_prob,
                key=k_update,
            )
            hist["target/p"][-1] = float(p)
            hist["target/p_jumped"][-1] = float(jumped)
            hist["target/p_base"][-1] = float(base_p)
            hist["target/p_ema"][-1] = float(ema_p)

        if gif_path and (t in snap_iters):
            frame_pts = np.asarray(xT)
            frames.append(frame_pts)

            if use_wandb and wandb.run is not None:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.scatter(frame_pts[:, 0], frame_pts[:, 1], s=3, alpha=0.25, c="r")
                ax.set_xlim(float(low[0]), float(high[0]))
                ax.set_ylim(float(low[1]), float(high[1]))
                ax.set_aspect("equal")
                ax.set_title(f"GBS target snapshot {t}")
                fig.tight_layout()

                wandb.log({f"{wandb_prefix}/frame": wandb.Image(fig)}, step=t)
                plt.close(fig)
        if return_snapshots and (t in snap_iters):
            key, k_snap = jax.random.split(key)
            _, xT_snap_latent, _ = snapshot_sampler_jit(
                k_snap,
                (fwd_state, bwd_state),
                fwd_state.params,
                bwd_state.params,
                current_lambda,
                current_policy_p,
            )
            snapshot_records.append(
                {"iter": int(t), "p": float(p), "samples": np.asarray(to_box(xT_snap_latent))}
            )

        # ------------------------------------------------------------------
        # Per-iteration W&B logging
        # ------------------------------------------------------------------
        if use_wandb and (wandb_log_every > 0) and ((t % wandb_log_every) == 0):
            payload = {
                f"{wandb_prefix}/iter": int(t),
                f"{wandb_prefix}/function_evals": float((t + 1) * buffer_size),
                f"{wandb_prefix}/target/p": _last_hist_value(hist, "target/p"),
                f"{wandb_prefix}/target/sinkhorn": _last_hist_value(hist, "target/sinkhorn"),
                f"{wandb_prefix}/target/energy_w2": _last_hist_value(hist, "target/energy_w2"),
            }

            if "train/tr_lv_lambda" in hist:
                payload[f"{wandb_prefix}/train/tr_lv_lambda"] = _last_hist_value(
                    hist, "train/tr_lv_lambda"
                )

            loss_key = _select_loss_key_for_logging(loss_mode, hist)
            if loss_key is not None:
                payload[f"{wandb_prefix}/{loss_key}"] = _last_hist_value(hist, loss_key)

            payload = {k: v for k, v in payload.items() if v is not None}
            wandb.log(payload, step=t)

            if traj_ts is not None and traj_xs is not None:
                domain = np.stack([np.asarray(low), np.asarray(high)], axis=1)
                evolution_payload = {}
                for dim_idx in range(int(traj_xs.shape[-1])):
                    evolution_fig = plot_evolution_plotly(
                                ts=traj_ts,
                                xs=traj_xs,
                                dim=dim_idx,
                                ntraj=wandb_plot_ntraj,
                                domain=domain,
                        )
                    evolution_payload[f"{wandb_prefix}/evolution/dim_{dim_idx}"] = wandb.Plotly(evolution_fig)
                wandb.log(evolution_payload, step=t)

    if gif_path and frames:
        rendered = []
        for idx, pts in enumerate(frames):
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.25, c="r")
            ax.set_xlim(float(low[0]), float(high[0]))
            ax.set_ylim(float(low[1]), float(high[1]))
            ax.set_aspect("equal")
            ax.set_title(f"GBS target snapshot {idx}")
            fig.tight_layout()
            fig.canvas.draw()
            rendered.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            plt.close(fig)
        imageio.mimsave(gif_path, rendered, fps=4)

    key, k_final = jax.random.split(key)
    _, xT_final_latent, _ = final_sampler_jit(
        k_final,
        (fwd_state, bwd_state),
        fwd_state.params,
        bwd_state.params,
        current_lambda,
        current_policy_p,
    )
    xT_final = to_box(xT_final_latent)
    np.save((save_dir / f"gbs_samples_{loss_mode}.npy").as_posix(), np.array(xT_final))
    result = (fwd_state, bwd_state, hist, np.asarray(xT_final))
    if return_snapshots:
        return result + (snapshot_records,)
    return result
