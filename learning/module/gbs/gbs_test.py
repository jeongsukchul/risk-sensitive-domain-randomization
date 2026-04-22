from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

import distrax
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib import pyplot as plt

from learning.module.gbs.gbs_loss import VP
from learning.module.gbs.gbs_trainer import (
    run_gbs,
)
from learning.module.gbs.target_examples_bridge import (
    is_target_example,
    load_target_example,
    sample_target_example_reference,
    target_example_bounds,
    target_example_names,
)
from learning.module.gbs.sinkhorn_metrics import sinkhorn_distance
from learning.module.gbs.targets.target_family import (
    add_target_cli_args,
    build_target_params_from_args,
    get_fixed_target_setup,
    target_energy_values,
)


def tanh_box_bijector(z: jax.Array, low: jax.Array, high: jax.Array) -> jax.Array:
  """Map unconstrained z to box [low, high] elementwise via tanh."""
  half = 0.5 * (high - low)
  mid = 0.5 * (high + low)
  return mid + half * jnp.tanh(z)


def tanh_box_logabsdet(z: jax.Array, low: jax.Array, high: jax.Array) -> jax.Array:
  """Log|det d x / d z| for x = tanh_box_bijector(z)."""
  z = jnp.atleast_2d(z)
  half = 0.5 * (high - low)
  jac_diag = half * (1.0 - jnp.tanh(z) ** 2)
  return jnp.sum(jnp.log(jnp.clip(jac_diag, 1e-12)), axis=-1)


def plot_target_contour(
    ax: plt.Axes,
    low: jax.Array,
    high: jax.Array,
    logprob_fn,
    n: int = 200,
    levels: int = 10,
    norm_gamma: float = 0.45,
    title: str = "target",
):
  x, y = jnp.meshgrid(
      jnp.linspace(low[0], high[0], n),
      jnp.linspace(low[1], high[1], n),
      indexing="xy",
  )
  grid = jnp.stack([x.reshape(-1), y.reshape(-1)], axis=-1)
  lp = logprob_fn(grid)
  z = jnp.exp(jnp.clip(lp, a_min=-1000.0)).reshape(n, n)
  ctf = ax.contourf(
      np.array(x),
      np.array(y),
      np.array(z),
      levels=levels,
      cmap="viridis",
      norm=PowerNorm(norm_gamma),
  )
  ax.set_title(title)
  ax.set_xlim(float(low[0]), float(high[0]))
  ax.set_ylim(float(low[1]), float(high[1]))
  ax.set_aspect("equal")
  return ctf


def _sanitize_tag(value: object) -> str:
  return str(value).replace("-", "m").replace(".", "p")


def compute_outer_iterations(function_evaluations: int, buffer_size: int) -> int:
  if buffer_size <= 0:
    raise ValueError(f"buffer_size must be positive, got {buffer_size}")
  if function_evaluations <= 0:
    raise ValueError(
        f"function_evaluations must be positive, got {function_evaluations}"
    )
  if function_evaluations % buffer_size != 0:
    raise ValueError(
        "function_evaluations must be divisible by buffer_size so outer buffered "
        f"updates are exact, got {function_evaluations=} and {buffer_size=}"
    )
  return function_evaluations // buffer_size


def sample_target_reference(
    key: jax.Array,
    low: jax.Array,
    high: jax.Array,
    logprob_fn,
    sample_shape: tuple[int, int],
    grid_size: int = 128,
) -> jax.Array:
  """Approximate target samples by categorical resampling from a dense 2D grid."""
  if len(sample_shape) != 2 or sample_shape[1] != 2:
    raise ValueError(f"Expected sample_shape [N,2], got {sample_shape}")

  x, y = jnp.meshgrid(
      jnp.linspace(low[0], high[0], grid_size),
      jnp.linspace(low[1], high[1], grid_size),
      indexing="xy",
  )
  grid = jnp.stack([x.reshape(-1), y.reshape(-1)], axis=-1)
  logp = jnp.asarray(logprob_fn(grid)).reshape(-1)
  idx = jax.random.categorical(key, logp, shape=(sample_shape[0],))
  return grid[idx]


def energy_wasserstein_against_target(
    samples: jax.Array,
    ref_samples: jax.Array,
    logprob_fn,
) -> float:
  sample_energy = -jnp.asarray(logprob_fn(samples)).reshape(-1)
  ref_energy = -jnp.asarray(logprob_fn(ref_samples)).reshape(-1)
  sample_energy = jnp.sort(sample_energy)
  ref_energy = jnp.sort(ref_energy)
  n = min(sample_energy.shape[0], ref_energy.shape[0])
  return float(jnp.mean((sample_energy[:n] - ref_energy[:n]) ** 2))


def build_latent_target_loggrad_fn(
    *,
    to_box,
    logabsdet_fn,
    target_logprob_box_fn,
):
  def _single_latent_logprob(z: jax.Array, lam: jax.Array, policy_p: jax.Array) -> jax.Array:
    z_batch = z[None, :]
    target_lp = jnp.asarray(target_logprob_box_fn(to_box(z_batch), lam, policy_p)).reshape(())
    logabsdet = jnp.asarray(logabsdet_fn(z_batch)).reshape(())
    return target_lp + logabsdet

  single_grad = jax.grad(_single_latent_logprob, argnums=0)

  @jax.jit
  def target_loggrad(z: jax.Array, lam: jax.Array, policy_p: jax.Array) -> jax.Array:
    z_batch = jnp.atleast_2d(z)
    grads = jax.vmap(single_grad, in_axes=(0, None, None))(z_batch, lam, policy_p)
    if z.ndim == 1:
      return grads[0]
    return grads

  return target_loggrad


def save_metric_plot(
    metric_hist: dict[str, list[float]],
    output_path: Path,
    *,
    target: str,
    beta: float,
) -> None:
  iters = np.arange(len(metric_hist["sinkhorn"]))
  has_lambda = "tr_lambda" in metric_hist
  ncols = 3 if has_lambda else 2
  fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
  if ncols == 2:
    axes = np.asarray(axes)

  axes[0].plot(iters, np.asarray(metric_hist["sinkhorn"], dtype=np.float64), color="tab:blue")
  axes[0].set_title(f"Sinkhorn vs target ({target}, beta={beta:g})")
  axes[0].set_xlabel("iteration")
  axes[0].grid(alpha=0.3)

  axes[1].plot(
      iters,
      np.asarray(metric_hist["energy_w2"], dtype=np.float64),
      color="tab:brown",
  )
  axes[1].set_title(f"Energy W2 vs target ({target}, beta={beta:g})")
  axes[1].set_xlabel("iteration")
  axes[1].grid(alpha=0.3)

  if has_lambda:
    axes[2].plot(
        iters,
        np.asarray(metric_hist["tr_lambda"], dtype=np.float64),
        color="tab:red",
    )
    axes[2].set_title(f"TR lambda ({target}, beta={beta:g})")
    axes[2].set_xlabel("iteration")
    axes[2].grid(alpha=0.3)

  fig.tight_layout()
  fig.savefig(output_path.as_posix(), dpi=150)
  plt.close(fig)


def _loss_plot_bounds(values_list: list[np.ndarray]):
  finite_values = [
      np.asarray(values, dtype=np.float64)[np.isfinite(values)]
      for values in values_list
  ]
  finite_values = [values for values in finite_values if values.size > 0]
  if not finite_values:
    return None
  values = np.concatenate(finite_values)
  if values.size < 4:
    y_min = float(np.min(values))
    y_max = float(np.max(values))
  else:
    q_low, q_high = np.nanpercentile(values, [5.0, 95.0])
    core = values[(values >= q_low) & (values <= q_high)]
    if core.size == 0:
      core = values
    y_min = float(np.min(core))
    y_max = float(np.max(core))
  span = y_max - y_min
  scale = max(abs(y_min), abs(y_max), 1e-8)
  pad = max(0.08 * span, 0.02 * scale, 1e-12)
  lower = y_min - pad
  upper = y_max + pad
  if np.all(values >= 0.0):
    lower = max(0.0, lower)
  if upper <= lower:
    upper = lower + max(0.04 * scale, 1e-12)
  clip_lower = bool(np.min(values) < lower)
  clip_upper = bool(np.max(values) > upper)
  return lower, upper, clip_lower, clip_upper


def _draw_loss_break_marks(ax, *, top: bool, bottom: bool) -> None:
  d = 0.012
  kwargs = dict(transform=ax.transAxes, color="black", clip_on=False, linewidth=1.2)
  if top:
    ax.plot((-d, +d), (1.0 - d, 1.0 + d), **kwargs)
    ax.plot((0.025 - d, 0.025 + d), (1.0 - d, 1.0 + d), **kwargs)
  if bottom:
    ax.plot((-d, +d), (-d, +d), **kwargs)
    ax.plot((0.025 - d, 0.025 + d), (-d, +d), **kwargs)


def save_loss_plot(
    hist: dict[str, np.ndarray | list[float]],
    output_path: Path,
    *,
    target: str,
    loss_mode: str,
) -> bool:
  loss_key = None
  loss_label = None
  if "train/tr_lv_var" in hist:
    loss_key = "train/tr_lv_var"
    loss_label = "TR-DDS-LV loss"
  elif "train/neg_elbo_var" in hist:
    loss_key = "train/neg_elbo_var"
    loss_label = "LV loss"
  elif "train/neg_elbo_mean" in hist:
    loss_key = "train/neg_elbo_mean"
    loss_label = "RE loss"
  if loss_key is None:
    return False

  values = np.asarray(hist[loss_key], dtype=np.float64)
  finite = np.isfinite(values)
  nan_mask = np.isnan(values)
  if not np.any(finite) and not np.any(nan_mask):
    return False

  fig, ax = plt.subplots(1, 1, figsize=(6, 4))
  x = np.arange(values.shape[0], dtype=np.float64)
  clip_info = _loss_plot_bounds([values])
  clipped_upper = np.zeros_like(finite, dtype=bool)
  clipped_lower = np.zeros_like(finite, dtype=bool)
  if clip_info is not None:
    y_lower, y_upper, has_lower_clip, has_upper_clip = clip_info
    ax.set_ylim(y_lower, y_upper)
    clipped_upper = finite & (values > y_upper)
    clipped_lower = finite & (values < y_lower)
    _draw_loss_break_marks(ax, top=has_upper_clip, bottom=has_lower_clip)
  if np.any(finite):
    y_plot = values.copy()
    if clip_info is not None:
      y_plot = np.clip(y_plot, y_lower, y_upper)
    ax.plot(x[finite], y_plot[finite], color="tab:purple", linewidth=2.0, label=loss_label)
    if np.any(clipped_upper):
      ax.scatter(
          x[clipped_upper],
          np.full(np.sum(clipped_upper), y_upper, dtype=np.float64),
          marker="^",
          color="tab:red",
          s=42,
          label="clipped high",
          zorder=5,
          clip_on=False,
      )
    if np.any(clipped_lower):
      ax.scatter(
          x[clipped_lower],
          np.full(np.sum(clipped_lower), y_lower, dtype=np.float64),
          marker="v",
          color="tab:red",
          s=42,
          label="clipped low",
          zorder=5,
          clip_on=False,
      )
  else:
    ax.set_ylim(0.0, 1.0)

  if np.any(nan_mask):
    ax.scatter(
        x[nan_mask],
        np.full(np.sum(nan_mask), 0.04, dtype=np.float64),
        transform=ax.get_xaxis_transform(),
        marker="x",
        color="red",
        s=36,
        linewidths=1.2,
        label="NaN flag",
        zorder=4,
    )
  ax.set_title(f"Loss ({target}, {loss_mode})")
  ax.set_xlabel("iteration")
  ax.set_ylabel(loss_label)
  ax.grid(alpha=0.3)
  ax.legend(framealpha=0.95)
  fig.tight_layout()
  fig.savefig(output_path.as_posix(), dpi=150)
  plt.close(fig)
  return True


def _normalize_loss_mode(loss_mode: str) -> str:
  normalized = loss_mode.lower().replace("-", "_")
  aliases = {
      "lv": "dds_lv",
      "time_reversal_lv": "dds_lv",
      "trust_region_lv": "tr_dds_lv",
      "tr_lv": "tr_dds_lv",
      "relative_entropy": "dds",
      "reverse_kl": "dds",
  }
  return aliases.get(normalized, normalized)


@dataclass(frozen=True)
class RunCfg:
  seed: int = 0
  function_evaluations: int = 50_000_000
  buffer_size: int = 50000
  num_steps: int = 50
  lr: float = 5e-4
  init_std: float = 1.0
  clip_grad: float = 1.0
  sigma_const: float = 1.0
  max_rnd: float = 1e8
  gbs_scale_diff: float = 20.0
  loss_mode: str = "tr_dds_lv"
  use_lgv: bool = False
  model_num_layers: int = 6
  model_num_hid: int = 256
  trust_region_bound: float = 0.1
  trust_region_lambda_max: float = 50.0
  trust_region_lambda_grid_size: int = 128
  minibatch_size: int = 2000
  minibatch_steps: int = 400
  sinkhorn_num_samples: int = 2048
  final_sample_size: int = 4096

  beta: float = -1.
  gamma: float = 0.45

  save_dir: str = "."
  save_gif: bool = True
  gif_name: str = "GBS_training.gif"
  snap_every: int = 50

  process: str = "vp"
  dim: int = 2
  target: str = "target3"
  use_tanh_bijection: bool = True


def _load_target_utils(target_name: str):
  module_map = {
      "target_A": "learning.module.gbs.targets.target_A_notebook_utils",
      "target_B": "learning.module.gbs.targets.target_B_notebook_utils",
      "target_C": "learning.module.gbs.targets.target_C_notebook_utils",
  }
  if target_name not in module_map:
    raise ValueError(f"Unknown target utils module for: {target_name}")
  return importlib.import_module(module_map[target_name])


def _write_run_config(output_path: Path, config: dict) -> None:
  output_path.write_text(json.dumps(config, indent=2, sort_keys=True))


def _run_dynamic_target(cfg: RunCfg, args: argparse.Namespace) -> None:
  save_dir = Path(cfg.save_dir)
  save_dir.mkdir(parents=True, exist_ok=True)
  target_params = build_target_params_from_args(args, cfg.dim)
  target_utils = _load_target_utils(cfg.target)
  low = jnp.zeros(cfg.dim, dtype=jnp.float32)
  high = jnp.ones(cfg.dim, dtype=jnp.float32)
  proc = VP(
      diff_coeff_sq_min=0.1,
      diff_coeff_sq_max=10.0,
      scale_diff_coeff=1.0,
      terminal_t=1.0,
      generative=False,
      sign=-1.0,
  )
  if not cfg.use_tanh_bijection:
    raise ValueError(f"{cfg.target} currently requires use_tanh_bijection=True")
  to_box = lambda z: target_utils.tanh_box_bijector(z, low=low, high=high)
  logabsdet_fn = lambda z: target_utils.tanh_box_logabsdet(z, low=low, high=high)
  target_logprob_box_fn = lambda x, lam, policy_p: target_utils.target_logprob(
      x, lam, target_params=target_params, policy_p=policy_p
  )
  target_loggrad_latent_fn = build_latent_target_loggrad_fn(
      to_box=to_box,
      logabsdet_fn=logabsdet_fn,
      target_logprob_box_fn=target_logprob_box_fn,
  )
  if cfg.target == "target_A":
    sample_mean_fn = lambda x, policy_p: jnp.mean(x)
    energy_w2_fn = lambda x, ref, lam, policy_p: target_utils.target_A_energy_wasserstein_1d(
        x, ref, lam
    )
  else:
    sample_mean_fn = lambda x, policy_p: jnp.mean(
        target_energy_values(x, target_params, policy_p=policy_p)
    )
    energy_w2_fn = lambda x, ref, lam, policy_p: energy_wasserstein_against_target(
        x, ref, lambda y: target_utils.target_logprob(y, lam, target_params=target_params, policy_p=policy_p)
    )
  _, _, hist, x_final = run_gbs(
      low=low,
      high=high,
      dim=cfg.dim,
      function_evaluations=cfg.function_evaluations,
      buffer_size=cfg.buffer_size,
      num_steps=cfg.num_steps,
      lr=cfg.lr,
      init_std=cfg.init_std,
      seed=cfg.seed,
      beta=cfg.beta,
      tau=float(args.tau),
      q=float(args.safe_q),
      initial_p=None if args.initial_p is None else float(args.initial_p),
      p_update_freq=int(args.p_update_freq),
      p_ema_alpha=float(args.p_ema_alpha),
      p_jump_prob=float(args.p_jump_prob),
      loss_mode=cfg.loss_mode,
      sinkhorn_num_samples=int(cfg.sinkhorn_num_samples),
      n_particles=int(args.n_particles) if args.n_particles is not None else None,
      n_spatial_dim=int(args.n_spatial_dim),
      save_dir=cfg.save_dir,
      gif_path=(save_dir / cfg.gif_name).as_posix() if cfg.save_gif else None,
      snap_iters=list(range(0, compute_outer_iterations(cfg.function_evaluations, cfg.buffer_size), max(cfg.snap_every, 1))),
      model_type=str(args.model_type),
      model_num_layers=cfg.model_num_layers,
      model_num_hid=cfg.model_num_hid,
      gbs_scale_diff=cfg.gbs_scale_diff,
      final_sample_size=cfg.final_sample_size,
      max_rnd=cfg.max_rnd,
      trust_region_bound=cfg.trust_region_bound,
      trust_region_lambda_max=cfg.trust_region_lambda_max,
      trust_region_lambda_grid_size=cfg.trust_region_lambda_grid_size,
      minibatch_size=cfg.minibatch_size,
      minibatch_steps=cfg.minibatch_steps,
      return_snapshots=False,
      snapshot_sample_size=None,
      max_metric_eval_points=None,
      process=proc,
      latent_prior_loc=jnp.zeros(cfg.dim, dtype=jnp.float32),
      process_center=jnp.zeros(cfg.dim, dtype=jnp.float32),
      clip_prior_without_tanh=False,
      use_tanh_bijection=True,
      logabsdet_fn=logabsdet_fn,
      to_box=to_box,
      target_logprob_box_fn=target_logprob_box_fn,
      sample_mean_fn=sample_mean_fn,
      compute_metrics_fn=lambda x, lam, key, policy_p: target_utils.compute_target_metrics(
          x,
          lam,
          target_params=target_params,
          num_bins=int(args.metric_num_bins),
          key=key,
          policy_p=policy_p,
      ),
      sample_reference_fn=lambda key, lam, shape, policy_p: target_utils.sample_truncated_exponential(
          key, lam, shape, target_params=target_params, policy_p=policy_p
      ),
      energy_w2_fn=energy_w2_fn,
      optimal_p_fn=lambda lam, tau, q: target_utils.optimal_p_from_target_mean(
          lam, tau, q, target_params=target_params
      ),
      update_p_fn=target_utils.update_p_with_ema_and_jump,
      target_loggrad_latent_fn=target_loggrad_latent_fn,
      use_lgv=cfg.use_lgv,
  )
  history_np = {k: np.asarray(v) for k, v in hist.items()}
  history_path = save_dir / "history.npz"
  np.savez(history_path.as_posix(), **history_np)
  samples_path = save_dir / "samples.npy"
  np.save(samples_path.as_posix(), np.asarray(x_final))
  metrics_plot_path = save_dir / "metrics.png"
  metric_hist = {
      "sinkhorn": list(np.asarray(hist["target/sinkhorn"], dtype=np.float64)),
      "energy_w2": list(np.asarray(hist["target/energy_w2"], dtype=np.float64)),
  }
  if "train/tr_lv_lambda" in hist:
      metric_hist["tr_lambda"] = list(np.asarray(hist["train/tr_lv_lambda"], dtype=np.float64))
  save_metric_plot(metric_hist, metrics_plot_path, target=cfg.target, beta=cfg.beta)
  loss_plot_path = save_dir / "loss.png"
  loss_saved = save_loss_plot(hist, loss_plot_path, target=cfg.target, loss_mode=cfg.loss_mode)
  loss_plot_path = save_dir / "loss.png"
  loss_saved = save_loss_plot(hist, loss_plot_path, target=cfg.target, loss_mode=cfg.loss_mode)
  loss_plot_path = save_dir / "loss.png"
  loss_saved = save_loss_plot(hist, loss_plot_path, target=cfg.target, loss_mode=cfg.loss_mode)
  _write_run_config(
      save_dir / "run_config.json",
      {
          "runner": "gbs_test",
          "target": cfg.target,
          "seed": cfg.seed,
          "dim": cfg.dim,
          "function_evaluations": cfg.function_evaluations,
          "buffer_size": cfg.buffer_size,
          "num_steps": cfg.num_steps,
          "lr": cfg.lr,
          "init_std": cfg.init_std,
          "beta": cfg.beta,
          "loss_mode": cfg.loss_mode,
          "use_lgv": cfg.use_lgv,
          "tau": float(args.tau),
          "safe_q": float(args.safe_q),
          "initial_p": args.initial_p,
          "p_update_freq": int(args.p_update_freq),
          "p_ema_alpha": float(args.p_ema_alpha),
          "p_jump_prob": float(args.p_jump_prob),
          "metric_num_bins": int(args.metric_num_bins),
          "sinkhorn_num_samples": int(cfg.sinkhorn_num_samples),
          "use_tanh_bijection": bool(cfg.use_tanh_bijection),
          "model_type": str(args.model_type),
          "model_num_layers": cfg.model_num_layers,
          "model_num_hid": cfg.model_num_hid,
      },
  )
  print(f"Target: {cfg.target}")
  print(f"Saved history to: {history_path}")
  print(f"Saved samples to: {samples_path}")
  print(f"Saved metrics plot to: {metrics_plot_path}")
  if loss_saved:
      print(f"Saved loss plot to: {loss_plot_path}")


def _run_fixed_target(cfg: RunCfg, args: argparse.Namespace) -> None:
  save_dir = Path(cfg.save_dir)
  save_dir.mkdir(parents=True, exist_ok=True)
  outer_iterations = compute_outer_iterations(cfg.function_evaluations, cfg.buffer_size)
  gif_name = cfg.gif_name
  if gif_name == "GBS_training.gif":
    gif_name = "training.gif"

  dim = 2
  logprob_fn_x, low, high, prior_loc, clip_prior, process_center = get_fixed_target_setup(
      cfg.target, cfg.beta
  )

  proc = VP(
      diff_coeff_sq_min=0.01,
      diff_coeff_sq_max=10.0,
      scale_diff_coeff=1.0,
      terminal_t=1.0,
      generative=False,
      sign=-1.0,
  )

  if cfg.use_tanh_bijection:
    to_box = lambda z: tanh_box_bijector(z, low=low, high=high)
    logabsdet_fn = lambda z: tanh_box_logabsdet(z, low=low, high=high)
    latent_prior_loc = jnp.zeros(dim, dtype=jnp.float32)
    process_center = jnp.zeros(dim, dtype=jnp.float32)
    clip_prior_without_tanh = False
  else:
    to_box = lambda z: z
    logabsdet_fn = lambda z: jnp.zeros((z.shape[0],), dtype=z.dtype)
    latent_prior_loc = prior_loc
    clip_prior_without_tanh = clip_prior
  target_logprob_box_fn = lambda x, lam, policy_p: logprob_fn_x(x)
  target_loggrad_latent_fn = build_latent_target_loggrad_fn(
      to_box=to_box,
      logabsdet_fn=logabsdet_fn,
      target_logprob_box_fn=target_logprob_box_fn,
  )

  result = run_gbs(
      low=low,
      high=high,
      dim=dim,
      function_evaluations=cfg.function_evaluations,
      buffer_size=cfg.buffer_size,
      num_steps=cfg.num_steps,
      lr=cfg.lr,
      init_std=cfg.init_std,
      seed=cfg.seed,
      beta=cfg.beta,
      tau=float(args.tau),
      q=float(args.safe_q),
      initial_p=1.0,
      p_update_freq=0,
      p_ema_alpha=float(args.p_ema_alpha),
      p_jump_prob=float(args.p_jump_prob),
      loss_mode=cfg.loss_mode,
      sinkhorn_num_samples=int(cfg.sinkhorn_num_samples),
      n_particles=int(args.n_particles) if args.n_particles is not None else None,
      n_spatial_dim=int(args.n_spatial_dim),
      save_dir=cfg.save_dir,
      gif_path=(save_dir / gif_name).as_posix() if cfg.save_gif else None,
      snap_iters=list(range(0, outer_iterations, max(cfg.snap_every, 1))),
      model_type=str(args.model_type),
      model_num_layers=cfg.model_num_layers,
      model_num_hid=cfg.model_num_hid,
      gbs_scale_diff=cfg.gbs_scale_diff,
      final_sample_size=cfg.final_sample_size,
      max_rnd=cfg.max_rnd,
      trust_region_bound=cfg.trust_region_bound,
      trust_region_lambda_max=cfg.trust_region_lambda_max,
      trust_region_lambda_grid_size=cfg.trust_region_lambda_grid_size,
      minibatch_size=cfg.minibatch_size,
      minibatch_steps=cfg.minibatch_steps,
      return_snapshots=False,
      snapshot_sample_size=None,
      max_metric_eval_points=None,
      process=proc,
      latent_prior_loc=latent_prior_loc,
      process_center=process_center,
      clip_prior_without_tanh=clip_prior_without_tanh,
      use_tanh_bijection=cfg.use_tanh_bijection,
      logabsdet_fn=logabsdet_fn,
      to_box=to_box,
      target_logprob_box_fn=target_logprob_box_fn,
      sample_mean_fn=lambda x, policy_p: jnp.mean(x),
      compute_metrics_fn=lambda x, lam, key, policy_p: (
          float("nan"),
          float("nan"),
          float("nan"),
      ),
      sample_reference_fn=lambda key, lam, shape, policy_p: sample_target_reference(
          key, low, high, logprob_fn_x, shape
      ),
      energy_w2_fn=lambda x, ref, lam, policy_p: energy_wasserstein_against_target(
          x, ref, logprob_fn_x
      ),
      optimal_p_fn=lambda lam, tau, q: (1.0, float("nan")),
      update_p_fn=lambda prev_p, sample_mean_g, tau, q, ema_alpha, jump_prob, key: (
          prev_p,
          prev_p,
          prev_p,
          False,
      ),
      target_loggrad_latent_fn=target_loggrad_latent_fn,
      use_lgv=cfg.use_lgv,
  )
  _, _, hist, x_final = result

  history_np = {k: np.asarray(v) for k, v in hist.items()}
  history_path = save_dir / "history.npz"
  np.savez(history_path.as_posix(), **history_np)

  fig, ax = plt.subplots(1, 1, figsize=(5, 5))
  ctf = plot_target_contour(
      ax,
      low,
      high,
      logprob_fn_x,
      n=200,
      levels=10,
      norm_gamma=cfg.gamma,
      title=f"Target density + final xT ({cfg.target}, beta={cfg.beta:g})",
  )
  fig.colorbar(ctf, ax=ax)
  pts = np.array(x_final)
  ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.25, c="r")
  fig.tight_layout()
  final_plot_path = save_dir / "final.png"
  fig.savefig(final_plot_path.as_posix(), dpi=150)
  plt.close(fig)

  metrics_plot_path = save_dir / "metrics.png"
  metric_hist = {
      "sinkhorn": list(np.asarray(hist["target/sinkhorn"], dtype=np.float64)),
      "energy_w2": list(np.asarray(hist["target/energy_w2"], dtype=np.float64)),
  }
  if "train/tr_lv_lambda" in hist:
      metric_hist["tr_lambda"] = list(np.asarray(hist["train/tr_lv_lambda"], dtype=np.float64))
  save_metric_plot(metric_hist, metrics_plot_path, target=cfg.target, beta=cfg.beta)
  loss_plot_path = save_dir / "loss.png"
  loss_saved = save_loss_plot(hist, loss_plot_path, target=cfg.target, loss_mode=cfg.loss_mode)

  _write_run_config(
      save_dir / "run_config.json",
      {
          "runner": "gbs_test",
          "target": cfg.target,
          "seed": cfg.seed,
          "dim": dim,
          "function_evaluations": cfg.function_evaluations,
          "buffer_size": cfg.buffer_size,
          "num_steps": cfg.num_steps,
          "lr": cfg.lr,
          "init_std": cfg.init_std,
          "beta": cfg.beta,
          "loss_mode": cfg.loss_mode,
          "process": cfg.process,
          "use_lgv": cfg.use_lgv,
          "use_tanh_bijection": bool(cfg.use_tanh_bijection),
          "trust_region_bound": cfg.trust_region_bound,
          "trust_region_lambda_max": cfg.trust_region_lambda_max,
          "trust_region_lambda_grid_size": cfg.trust_region_lambda_grid_size,
      },
  )
  print(f"Target: {cfg.target}")
  print(f"Saved history to: {history_path}")
  print(f"Saved samples to: {save_dir / 'samples.npy'}")
  print(f"Saved final plot to: {final_plot_path}")
  print(f"Saved metrics plot to: {metrics_plot_path}")
  if loss_saved:
      print(f"Saved loss plot to: {loss_plot_path}")


def _run_target_example(cfg: RunCfg, args: argparse.Namespace) -> None:
  save_dir = Path(cfg.save_dir)
  save_dir.mkdir(parents=True, exist_ok=True)
  outer_iterations = compute_outer_iterations(cfg.function_evaluations, cfg.buffer_size)
  gif_name = cfg.gif_name
  if gif_name == "GBS_training.gif":
    gif_name = "training.gif"

  target_example = load_target_example(cfg.target, cfg.dim, int(args.n_spatial_dim))
  low, high = target_example_bounds(cfg.target, cfg.dim)

  proc = VP(
      diff_coeff_sq_min=0.01,
      diff_coeff_sq_max=10.0,
      scale_diff_coeff=1.0,
      terminal_t=1.0,
      generative=False,
      sign=-1.0,
  )

  result = run_gbs(
      low=low,
      high=high,
      dim=cfg.dim,
      function_evaluations=cfg.function_evaluations,
      buffer_size=cfg.buffer_size,
      num_steps=cfg.num_steps,
      lr=cfg.lr,
      init_std=cfg.init_std,
      seed=cfg.seed,
      beta=cfg.beta,
      tau=float(args.tau),
      q=float(args.safe_q),
      initial_p=1.0,
      p_update_freq=0,
      p_ema_alpha=float(args.p_ema_alpha),
      p_jump_prob=float(args.p_jump_prob),
      loss_mode=cfg.loss_mode,
      sinkhorn_num_samples=int(cfg.sinkhorn_num_samples),
      n_particles=int(args.n_particles) if args.n_particles is not None else None,
      n_spatial_dim=int(args.n_spatial_dim),
      save_dir=cfg.save_dir,
      gif_path=(save_dir / gif_name).as_posix() if cfg.save_gif else None,
      snap_iters=list(range(0, outer_iterations, max(cfg.snap_every, 1))),
      model_type=str(args.model_type),
      model_num_layers=cfg.model_num_layers,
      model_num_hid=cfg.model_num_hid,
      gbs_scale_diff=cfg.gbs_scale_diff,
      final_sample_size=cfg.final_sample_size,
      max_rnd=cfg.max_rnd,
      trust_region_bound=cfg.trust_region_bound,
      trust_region_lambda_max=cfg.trust_region_lambda_max,
      trust_region_lambda_grid_size=cfg.trust_region_lambda_grid_size,
      minibatch_size=cfg.minibatch_size,
      minibatch_steps=cfg.minibatch_steps,
      return_snapshots=False,
      snapshot_sample_size=None,
      max_metric_eval_points=None,
      process=proc,
      latent_prior_loc=jnp.zeros(cfg.dim, dtype=jnp.float32),
      process_center=jnp.zeros(cfg.dim, dtype=jnp.float32),
      clip_prior_without_tanh=False,
      use_tanh_bijection=False,
      logabsdet_fn=lambda z: jnp.zeros((z.shape[0],), dtype=z.dtype),
      to_box=lambda z: z,
      target_logprob_box_fn=lambda x, lam, policy_p: target_example.log_prob(x),
      sample_mean_fn=lambda x, policy_p: jnp.mean(x),
      compute_metrics_fn=lambda x, lam, key, policy_p: (
          float("nan"),
          float("nan"),
          float("nan"),
      ),
      sample_reference_fn=(
          (lambda key, lam, shape, policy_p: sample_target_example_reference(target_example, key, shape[0]))
          if getattr(target_example, "can_sample", False) or hasattr(target_example, "test_set")
          else None
      ),
      energy_w2_fn=lambda x, ref, lam, policy_p: energy_wasserstein_against_target(
          x, ref, target_example.log_prob
      ),
      optimal_p_fn=lambda lam, tau, q: (1.0, float("nan")),
      update_p_fn=lambda prev_p, sample_mean_g, tau, q, ema_alpha, jump_prob, key: (
          prev_p,
          prev_p,
          prev_p,
          False,
      ),
      target_loggrad_latent_fn=build_latent_target_loggrad_fn(
          to_box=lambda z: z,
          logabsdet_fn=lambda z: jnp.zeros((z.shape[0],), dtype=z.dtype),
          target_logprob_box_fn=lambda x, lam, policy_p: target_example.log_prob(x),
      ),
      use_lgv=cfg.use_lgv,
  )
  _, _, hist, x_final = result

  history_np = {k: np.asarray(v) for k, v in hist.items()}
  history_path = save_dir / "history.npz"
  np.savez(history_path.as_posix(), **history_np)

  if cfg.dim == 2:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ctf = plot_target_contour(
        ax,
        low,
        high,
        target_example.log_prob,
        n=200,
        levels=10,
        norm_gamma=cfg.gamma,
        title=f"Target density + final xT ({cfg.target})",
    )
    fig.colorbar(ctf, ax=ax)
    pts = np.array(x_final)
    ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.25, c="r")
    fig.tight_layout()
    final_plot_path = save_dir / "final.png"
    fig.savefig(final_plot_path.as_posix(), dpi=150)
    plt.close(fig)
  else:
    final_plot_path = None

  metrics_plot_path = save_dir / "metrics.png"
  metric_hist = {
      "sinkhorn": list(np.asarray(hist["target/sinkhorn"], dtype=np.float64)),
      "energy_w2": list(np.asarray(hist["target/energy_w2"], dtype=np.float64)),
  }
  if "train/tr_lv_lambda" in hist:
      metric_hist["tr_lambda"] = list(np.asarray(hist["train/tr_lv_lambda"], dtype=np.float64))
  save_metric_plot(metric_hist, metrics_plot_path, target=cfg.target, beta=cfg.beta)
  loss_plot_path = save_dir / "loss.png"
  loss_saved = save_loss_plot(hist, loss_plot_path, target=cfg.target, loss_mode=cfg.loss_mode)

  _write_run_config(
      save_dir / "run_config.json",
      {
          "runner": "gbs_test",
          "target": cfg.target,
          "seed": cfg.seed,
          "dim": cfg.dim,
          "function_evaluations": cfg.function_evaluations,
          "buffer_size": cfg.buffer_size,
          "num_steps": cfg.num_steps,
          "lr": cfg.lr,
          "init_std": cfg.init_std,
          "beta": cfg.beta,
          "loss_mode": cfg.loss_mode,
          "process": cfg.process,
          "use_lgv": cfg.use_lgv,
          "use_tanh_bijection": False,
      },
  )
  print(f"Target: {cfg.target}")
  print(f"Saved history to: {history_path}")
  print(f"Saved samples to: {save_dir / 'samples.npy'}")
  if final_plot_path is not None:
    print(f"Saved final plot to: {final_plot_path}")
  print(f"Saved metrics plot to: {metrics_plot_path}")
  if loss_saved:
    print(f"Saved loss plot to: {loss_plot_path}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
      "--function_evaluations",
      "--iters",
      dest="function_evaluations",
      type=int,
      default=20_000_000,
  )
  parser.add_argument(
      "--buffer_size",
      "--batch_size",
      dest="buffer_size",
      type=int,
      default=20000,
  )
  parser.add_argument("--num_steps", type=int, default=50)
  parser.add_argument("--lr", type=float, default=5e-4)
  parser.add_argument("--init_std", type=float, default=1.0)
  parser.add_argument("--clip_grad", type=float, default=1.0)
  parser.add_argument("--sigma_const", type=float, default=1.0)
  parser.add_argument("--max_rnd", type=float, default=1e8)
  parser.add_argument("--gbs_scale_diff", type=float, default=20.0)
  parser.add_argument(
      "--loss_mode",
      choices=[
          "trust_region_lv",
          "tr_lv",
          "tr_dds_lv",
          "lv",
          "time_reversal_lv",
          "dds",
          "dds_lv",
          "dds_exp_lv",
          "dis",
          "dis_lv",
      ],
      default="tr_dds_lv",
  )
  parser.add_argument("--use_lgv", action="store_true")
  parser.add_argument("--no_use_lgv", dest="use_lgv", action="store_false")
  parser.set_defaults(use_lgv=False)
  parser.add_argument("--model_num_layers", type=int, default=6)
  parser.add_argument("--model_num_hid", type=int, default=256)
  parser.add_argument("--trust_region_bound", type=float, default=0.1)
  parser.add_argument("--trust_region_lambda_max", type=float, default=50.0)
  parser.add_argument("--trust_region_lambda_grid_size", type=int, default=128)
  parser.add_argument("--minibatch_size", type=int, default=2000)
  parser.add_argument(
      "--minibatch_steps",
      "--buffer_updates",
      dest="minibatch_steps",
      type=int,
      default=400,
  )
  parser.add_argument("--sinkhorn_num_samples", type=int, default=2048)
  parser.add_argument("--final_sample_size", type=int, default=4096)
  parser.add_argument("--beta", type=float, default=-1.0)
  parser.add_argument("--gamma", type=float, default=0.45)
  parser.add_argument("--save_dir", type=str, default=".")
  parser.add_argument("--save_gif", action="store_true")
  parser.add_argument("--no_save_gif", dest="save_gif", action="store_false")
  parser.set_defaults(save_gif=True)
  parser.add_argument("--gif_name", type=str, default="GBS_training.gif")
  parser.add_argument("--snap_every", type=int, default=50)
  parser.add_argument("--process", choices=["vp"], default="vp")
  parser.add_argument("--dim", type=int, default=2)
  parser.add_argument(
      "--target",
      choices=["target1", "target2", "target3", "target_A", "target_B", "target_C", *target_example_names()],
      default="target3",
  )
  parser.add_argument("--use_tanh_bijection", action="store_true")
  parser.add_argument("--no_use_tanh_bijection", dest="use_tanh_bijection", action="store_false")
  parser.set_defaults(use_tanh_bijection=True)
  parser.add_argument("--tau", type=float, default=1.0)
  parser.add_argument("--safe_q", type=float, default=0.6)
  parser.add_argument("--initial_p", type=float, default=0.8)
  parser.add_argument("--p_update_freq", type=int, default=1)
  parser.add_argument("--p_ema_alpha", type=float, default=0.9)
  parser.add_argument("--p_jump_prob", type=float, default=0.0)
  parser.add_argument("--metric_num_bins", type=int, default=128)
  parser.add_argument("--n_particles", type=int, default=None)
  parser.add_argument("--n_spatial_dim", type=int, default=1)
  parser.add_argument("--model_type", choices=["pisgrad", "potential"], default="pisgrad")
  add_target_cli_args(parser)
  args = parser.parse_args()
  cfg_keys = {field.name for field in fields(RunCfg)}
  cfg = RunCfg(**{k: v for k, v in vars(args).items() if k in cfg_keys})
  if cfg.target in ("target_A", "target_B", "target_C"):
    _run_dynamic_target(cfg, args)
    return
  if is_target_example(cfg.target):
    _run_target_example(cfg, args)
    return
  _run_fixed_target(cfg, args)


if __name__ == "__main__":
  main()
