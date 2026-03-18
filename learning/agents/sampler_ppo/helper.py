import os
from typing import Any, Callable, MutableMapping, Optional, Sequence, Tuple

from absl import logging
import flax
import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy


def plot_pairwise_sample_density(
    samples: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    bins: int = 60,
    title: str = "Pairwise sample density",
) -> Tuple[plt.Figure, np.ndarray]:
  samples = np.asarray(samples)
  low = np.asarray(low)
  high = np.asarray(high)

  if samples.ndim != 2:
    raise ValueError(f"`samples` must be 2D, got shape {samples.shape}.")

  dim = samples.shape[1]
  fig, axes = plt.subplots(dim, dim, figsize=(3.5 * dim, 3.5 * dim))
  if dim == 1:
    axes = np.asarray([[axes]])

  mappable = None
  for row in range(dim):
    for col in range(dim):
      ax = axes[row, col]
      if row == col:
        ax.hist(
            samples[:, col],
            bins=bins,
            range=(float(low[col]), float(high[col])),
            color="tab:blue",
            alpha=0.85,
        )
        ax.set_xlim(float(low[col]), float(high[col]))
      else:
        hist, xedges, yedges = np.histogram2d(
            samples[:, col],
            samples[:, row],
            bins=bins,
            range=[
                [float(low[col]), float(high[col])],
                [float(low[row]), float(high[row])],
            ],
        )
        mappable = ax.pcolormesh(
            xedges, yedges, hist.T, shading="auto", cmap="viridis"
        )
        ax.set_xlim(float(low[col]), float(high[col]))
        ax.set_ylim(float(low[row]), float(high[row]))

      if row == dim - 1:
        ax.set_xlabel(f"dim {col}")
      else:
        ax.set_xticklabels([])

      if col == 0:
        ax.set_ylabel(f"dim {row}")
      else:
        ax.set_yticklabels([])

  if mappable is not None:
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label("sample density")
  fig.suptitle(title)
  return fig, axes


def finalize_figure(fig: plt.Figure) -> np.ndarray:
  fig.tight_layout()
  fig.canvas.draw()
  return np.asarray(fig.canvas.buffer_rgba())[..., :3]


def maybe_log_wandb_image(
    *,
    use_wandb: bool,
    wandb_module: Any,
    key: str,
    fig: plt.Figure,
    step: int,
) -> None:
  if not use_wandb:
    return
  wandb_module.log({key: wandb_module.Image(fig)}, step=int(step))


def maybe_log_resized_wandb_image(
    *,
    use_wandb: bool,
    wandb_module: Any,
    key: str,
    fig: plt.Figure,
    step: int,
    size: Tuple[float, float] = (12, 8),
    dpi: int = 100,
) -> None:
  if not use_wandb:
    return
  fig.set_size_inches(*size, forward=True)
  fig.set_dpi(dpi)
  wandb_module.log({key: wandb_module.Image(fig)}, step=int(step))


def create_eval_heatmap_figure(
    x: np.ndarray,
    y: np.ndarray,
    reward_1d: np.ndarray,
    current_step: int,
) -> plt.Figure:
  fig = plt.figure()
  reward_2d = np.asarray(reward_1d).reshape(np.asarray(x).shape)
  contour = plt.contourf(x, y, reward_2d, levels=20, cmap="viridis")
  fig.colorbar(contour)
  fig.suptitle(f"Evaluation on Each Params [Step={int(current_step)}]")
  return fig


def create_target_heatmap_figure(
    x: np.ndarray,
    y: np.ndarray,
    target_lnpdfs: np.ndarray,
    current_step: int,
) -> plt.Figure:
  fig = plt.figure()
  contour = plt.contourf(x, y, target_lnpdfs, levels=20, cmap="viridis")
  fig.colorbar(contour)
  fig.suptitle(
      f"target log prob on current occupancy [step={int(current_step)}]"
  )
  return fig


def update_metrics_with_reward_distribution(
    metrics: MutableMapping[str, Any], rewards: np.ndarray
) -> MutableMapping[str, Any]:
  rewards = np.asarray(rewards).reshape(-1)
  if rewards.size == 0:
    return metrics

  k20 = max(int(rewards.shape[0] * 0.2), 1)
  k10 = max(int(rewards.shape[0] * 0.1), 1)
  sorted_rewards = np.sort(rewards)
  metrics["eval/episode_reward_mean"] = np.mean(rewards)
  metrics["eval/episode_reward_p12"] = np.percentile(rewards, 12.5)
  metrics["eval/episode_reward_p25"] = np.percentile(rewards, 25)
  metrics["eval/episode_reward_p75"] = np.percentile(rewards, 75)
  metrics["eval/episode_reward_std"] = np.std(rewards)
  metrics["eval/episode_reward_min"] = np.min(rewards)
  metrics["eval/episode_reward_max"] = np.max(rewards)
  metrics["eval/episode_reward_iqm"] = scipy.stats.trim_mean(
      rewards, proportiontocut=0.25, axis=None
  )
  metrics["eval/episode_reward_CVaR20"] = np.mean(sorted_rewards[:k20])
  metrics["eval/episode_reward_CVaR10"] = np.mean(sorted_rewards[:k10])
  return metrics


def compute_percentile_dynamics_params(
    dynamics_params_eval: np.ndarray,
    rewards_eval: np.ndarray,
    percentile_levels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
  dyn = np.asarray(dynamics_params_eval)
  rew = np.asarray(rewards_eval).reshape(-1)
  percentile_levels = np.asarray(percentile_levels)

  if dyn.ndim == 1:
    dyn = dyn[:, None]

  n = min(dyn.shape[0], rew.shape[0])
  dyn = dyn[:n]
  rew = rew[:n]
  order = np.argsort(rew)
  sorted_dyn = dyn[order]
  sorted_rew = rew[order]
  idx = np.rint((percentile_levels / 100.0) * max(n - 1, 0)).astype(int)
  idx = np.clip(idx, 0, max(n - 1, 0))
  return sorted_dyn[idx], sorted_rew[idx]


def select_sampler_state(
    *,
    sampler_choice: str,
    training_state: Any,
    unpmap_fn: Callable[[Any], Any],
) -> Optional[Any]:
  if sampler_choice == "AutoDR":
    return unpmap_fn(training_state.autodr_state)
  if sampler_choice == "DORAEMON":
    return unpmap_fn(training_state.doraemon_state)
  if sampler_choice == "GBS":
    return unpmap_fn(training_state.flow_state)
  if sampler_choice == "GMM":
    return unpmap_fn(training_state.gmm_training_state)
  if "FLOW" in sampler_choice:
    return None
  return None


def save_sampler_state(
    *,
    save_path: str,
    sampler_choice: str,
    training_state: Any,
    unpmap_fn: Callable[[Any], Any],
    log_prefix: str = "Saved",
) -> bool:
  state_to_save = select_sampler_state(
      sampler_choice=sampler_choice,
      training_state=training_state,
      unpmap_fn=unpmap_fn,
  )
  if state_to_save is None:
    return False

  with open(save_path, "wb") as f:
    f.write(flax.serialization.to_bytes(state_to_save))
  logging.info("%s %s state to %s", log_prefix, sampler_choice, save_path)
  return True


def save_frames_as_gif(
    *,
    frames: Sequence[np.ndarray],
    save_dir: str,
    filename: str,
    fps: int = 4,
) -> None:
  if not frames:
    return
  imageio.mimsave(os.path.join(save_dir, filename), frames, fps=fps)
