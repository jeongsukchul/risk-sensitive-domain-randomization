# Modified for the RSDR distribution. See THIRD_PARTY_NOTICES.md.

# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Host-side plots and animations for sampler evaluation."""

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np


def plot_reward_heatmap(samples, rewards, low, high, bins=80, title="Reward heatmap"):
    """
    samples: (N,2)  [x,y]
    rewards: (N,)
    low/high: (2,) bounds
    """
    s = np.asarray(samples)
    r = np.asarray(rewards).reshape(-1)
    assert s.shape[1] == 2 and s.shape[0] == r.shape[0], (s.shape, r.shape)

    x = s[:, 0]
    y = s[:, 1]

    # Sum of rewards per bin
    sum_r, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[low[0], high[0]], [low[1], high[1]]],
        weights=r,
    )
    # Count per bin
    cnt, _, _ = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[low[0], high[0]], [low[1], high[1]]],
        weights=np.ones_like(r),
    )

    mean_r = sum_r / np.maximum(cnt, 1.0)
    mean_r[cnt == 0] = np.nan  # show empty bins as blank

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(xedges, yedges, mean_r.T, shading="auto")  # note .T
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("mean reward")

    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    # optional: overlay a few points for sanity
    idx = np.random.default_rng(0).choice(
        s.shape[0], size=min(300, s.shape[0]), replace=False
    )
    ax.scatter(x[idx], y[idx], s=10, alpha=0.4, marker="x", label="samples")

    # legend outside
    fig.subplots_adjust(right=0.78)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    return fig, ax


def plot_pairwise_sample_density(
    samples: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    title: str,
    bins: int = 50,
):
    samples = np.asarray(samples)
    low = np.asarray(low)
    high = np.asarray(high)
    dim = samples.shape[1]
    fig, axes = plt.subplots(dim, dim, figsize=(3.5 * dim, 3.5 * dim), squeeze=False)
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    mappable = None

    for row in range(dim):
        for col in range(dim):
            ax = axes[row, col]
            if row < col:
                ax.axis("off")
                continue
            if row == col:
                ax.hist(samples[:, col], bins=30, density=True, alpha=0.85)
                ax.set_xlim(float(low[col]), float(high[col]))
                ax.yaxis.set_ticks([])
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


def save_training_animations(
    save_dir,
    sampler_choice,
    *,
    beta,
    gamma,
    success_threshold,
    success_rate_condition,
    evaluation_frames,
    occupancy_frames,
    autodr_frames,
    doraemon_frames,
    flow_frames,
    gbs_frames,
    gmm_frames,
):
    """Write available 2D diagnostics using the existing artifact filenames."""
    if sampler_choice in ("NODR", "UDR", "EPOpt"):
        animations = {"Evaluation Heatmap": evaluation_frames}
        suffix = ""
    else:
        if sampler_choice == "AutoDR":
            suffix = f" [threshold={success_threshold}]"
            sampler_title, sampler_frames = "Auto DR Heatmap", autodr_frames
        elif sampler_choice == "DORAEMON":
            suffix = (
                f" [threshold={success_threshold}_condition={success_rate_condition}]"
            )
            sampler_title, sampler_frames = "Doraemon Heatmap", doraemon_frames
        elif "FLOW" in sampler_choice:
            suffix = f" [beta={beta}_gamma={gamma}]"
            sampler_title, sampler_frames = "Flow Heatmap", flow_frames
        elif sampler_choice == "GBS":
            suffix = f" [beta={beta}]"
            sampler_title, sampler_frames = "GBS Heatmap", gbs_frames
        elif sampler_choice == "GMM":
            suffix = f" [beta={beta}]"
            sampler_title, sampler_frames = "GMM Log Prob Heatmap", gmm_frames
        else:
            raise ValueError(f"Unknown sampler {sampler_choice!r}.")
        animations = {
            "Evaluation Heatmap": evaluation_frames,
            "target log prob with current occupancy": occupancy_frames,
            sampler_title: sampler_frames,
        }
    for title, frames in animations.items():
        if frames:
            imageio.mimsave(Path(save_dir) / f"{title}{suffix}.gif", frames, fps=4)


def capture_figure(fig, frames, *, key, step, use_wandb, resize_for_wandb=False):
    """Append an owned RGB frame, optionally log it, and release the figure."""
    import wandb

    try:
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if use_wandb:
            if resize_for_wandb:
                fig.set_size_inches(12, 8, forward=True)
                fig.set_dpi(100)
            wandb.log({key: wandb.Image(fig)}, step=step)
    finally:
        plt.close(fig)
