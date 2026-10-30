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


"""Evaluation grids, sampler-matched return estimates, and empirical artifacts."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb

from learning.agents.sampler_ppo.empirical_metrics import (
    compare_empirical_target_and_sampler,
)
from learning.module.wrapper.evaluator import generate_adv_unroll


def make_2d_dynamics_grid(num_points: int, low, high):
    """Return exactly num_points equal-area cell centers in a 2D box."""
    if num_points <= 0:
        raise ValueError("num_points must be positive.")
    # Exact equal-area, cell-centered grid: dim_x * dim_y == num_points.
    dim_x = int(np.floor(np.sqrt(num_points)))
    while dim_x > 1 and num_points % dim_x != 0:
        dim_x -= 1
    if num_points % dim_x != 0:
        dim_x = 1
    dim_y = num_points // dim_x
    if dim_x > dim_y:
        dim_x, dim_y = dim_y, dim_x

    x_values = (
        low[0]
        + (jnp.arange(dim_x, dtype=jnp.float32) + 0.5) * (high[0] - low[0]) / dim_x
    )
    y_values = (
        low[1]
        + (jnp.arange(dim_y, dtype=jnp.float32) + 0.5) * (high[1] - low[1]) / dim_y
    )
    x, y = jnp.meshgrid(x_values, y_values)
    grid = jnp.c_[x.ravel(), y.ravel()]
    return x, y, grid


def compute_percentile_dynamics_params(
    dynamics_params_eval: np.ndarray,
    rewards_eval: np.ndarray,
    percentile_levels,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select observed dynamics settings at the requested return percentiles."""
    dyn = np.asarray(dynamics_params_eval)
    rew = np.asarray(rewards_eval).reshape(-1)
    if dyn.ndim == 1:
        dyn = dyn[:, None]
    n = min(dyn.shape[0], rew.shape[0])
    if n == 0:
        raise ValueError("Percentile selection requires nonempty evaluation data.")
    dyn = dyn[:n]
    rew = rew[:n]
    order = np.argsort(rew)
    sorted_dyn = dyn[order]
    sorted_rew = rew[order]
    idx = np.rint((percentile_levels / 100.0) * max(n - 1, 0)).astype(int)
    idx = np.clip(idx, 0, max(n - 1, 0))
    return sorted_dyn[idx], sorted_rew[idx]


def make_sampler_return_estimator(
    make_policy,
    eval_env,
    *,
    num_eval_envs,
    empirical_num_rollouts,
    empirical_num_unrolls,
    unroll_length,
    reward_scale_for_sampler,
    empirical_clip_negative_rewards,
):
    """Build a JIT estimator with the same reset and rollout protocol as training."""

    @jax.jit
    def sampler_matched_empirical_returns(
        policy_params,
        dynamics_params,
        key,
    ):
        """Estimates the sampler reward signal from independent fresh resets."""
        policy = make_policy(policy_params)
        rollout_keys = jax.random.split(key, empirical_num_rollouts)

        def evaluate_one_rollout(unused_carry, rollout_key):
            reset_key, trajectory_key = jax.random.split(rollout_key)
            reset_keys = jax.random.split(reset_key, num_eval_envs)
            rollout_state = eval_env.reset(reset_keys)

            def collect_unroll(carry, unused):
                current_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                next_state, rollout_data = generate_adv_unroll(
                    eval_env,
                    current_state,
                    dynamics_params,
                    policy,
                    current_key,
                    unroll_length,
                )
                rollout_rewards = rollout_data.reward
                if empirical_clip_negative_rewards:
                    rollout_rewards = jnp.maximum(rollout_rewards, 0.0)
                reward_sum = jnp.sum(rollout_rewards, axis=0)
                return (next_state, next_key), reward_sum

            (_, _), reward_sums = jax.lax.scan(
                collect_unroll,
                (rollout_state, trajectory_key),
                (),
                length=empirical_num_unrolls,
            )
            mean_reward = jnp.sum(reward_sums, axis=0) / (
                empirical_num_unrolls * unroll_length
            )
            return unused_carry, mean_reward * reward_scale_for_sampler

        _, per_rollout_returns = jax.lax.scan(
            evaluate_one_rollout,
            None,
            rollout_keys,
        )
        return jnp.mean(per_rollout_returns, axis=0), per_rollout_returns

    return sampler_matched_empirical_returns


@dataclass(frozen=True)
class EmpiricalGridEvaluator:
    """Record finite-grid sampler diagnostics without depending on trainer state."""

    log_density_fn: Callable
    sampler_choice: str
    reward_scale_for_sampler: float
    kl_radius: float
    fixed_radius: bool
    dual_update_mode: str
    empirical_num_rollouts: int
    unroll_length: int
    empirical_num_unrolls: int
    empirical_clip_negative_rewards: bool
    save_dir: Path
    eval_grid_size_2d: int
    use_wandb: bool

    def add_metrics(
        self,
        metrics,
        gmm_state,
        dynamics_params_grid,
        reward_per_grid_cell,
        episode_length_per_grid_cell,
        current_beta,
        x_grid,
        y_grid,
        step,
        estimated_returns_override=None,
        target_noise_metrics=None,
        per_rollout_returns=None,
    ):
        if self.sampler_choice != "GMM":
            return metrics

        if estimated_returns_override is None:
            estimated_returns = (
                jnp.asarray(reward_per_grid_cell)
                / jnp.maximum(jnp.asarray(episode_length_per_grid_cell), 1.0)
                * self.reward_scale_for_sampler
            )
        else:
            estimated_returns = jnp.asarray(estimated_returns_override)
        sampler_log_density = jax.vmap(
            self.log_density_fn,
            in_axes=(None, 0),
        )(gmm_state, dynamics_params_grid)
        (
            comparison_metrics,
            log_target_mass,
            log_sampler_mass,
            target_center,
        ) = compare_empirical_target_and_sampler(
            estimated_returns,
            current_beta,
            sampler_log_density,
        )

        metrics.update(
            {
                f"eval/empirical_{name}": value
                for name, value in comparison_metrics.items()
            }
        )
        metrics.update(
            {
                "eval/empirical_beta": jnp.asarray(current_beta),
                "eval/empirical_target_center_c": target_center,
                "eval/empirical_grid_size": jnp.asarray(dynamics_params_grid.shape[0]),
                "eval/empirical_weighted_target_reverse_kl_to_uniform": comparison_metrics[
                    "target_reverse_kl_to_uniform"
                ],
                "eval/empirical_num_rollouts": jnp.asarray(self.empirical_num_rollouts),
                "eval/empirical_sampler_unroll_length": jnp.asarray(self.unroll_length),
                "eval/empirical_sampler_num_unrolls": jnp.asarray(
                    self.empirical_num_unrolls
                ),
                "eval/empirical_sampler_total_transitions": jnp.asarray(
                    self.empirical_num_unrolls * self.unroll_length
                ),
                "eval/empirical_clips_negative_rewards": jnp.asarray(
                    self.empirical_clip_negative_rewards, dtype=jnp.float32
                ),
                "eval/empirical_matches_training_reward_clip": jnp.asarray(
                    self.empirical_clip_negative_rewards, dtype=jnp.float32
                ),
                "eval/empirical_uses_stochastic_policy": jnp.asarray(1.0),
            }
        )
        if target_noise_metrics is not None:
            metrics.update(
                {
                    f"eval/empirical_{name}": value
                    for name, value in target_noise_metrics.items()
                }
            )
        target_radius_residual = (
            comparison_metrics["target_reverse_kl_to_uniform"] - self.kl_radius
        )
        sampler_radius_residual = (
            comparison_metrics["sampler_reverse_kl_to_uniform"] - self.kl_radius
        )
        metrics.update(
            {
                "eval/empirical_kl_radius": jnp.asarray(self.kl_radius),
                "eval/empirical_target_kl_radius_residual": target_radius_residual,
                "eval/empirical_sampler_kl_radius_residual": sampler_radius_residual,
                "eval/empirical_target_kl_radius_violation": jnp.maximum(
                    target_radius_residual, 0.0
                ),
                "eval/empirical_sampler_kl_radius_violation": jnp.maximum(
                    sampler_radius_residual, 0.0
                ),
            }
        )
        if self.fixed_radius:
            metrics.update(
                {
                    "eval/empirical_dual_update_mode_beta": jnp.asarray(
                        self.dual_update_mode == "beta", dtype=jnp.float32
                    ),
                }
            )

        target_mass = jnp.exp(log_target_mass)
        sampler_mass = jnp.exp(log_sampler_mass)
        output_stem = f"empirical_target_sampler_{int(step)}"
        np.savez_compressed(
            os.path.join(self.save_dir, f"{output_stem}.npz"),
            dynamics_params=np.asarray(dynamics_params_grid),
            estimated_returns=np.asarray(estimated_returns),
            per_rollout_estimated_returns=(
                np.asarray(per_rollout_returns)
                if per_rollout_returns is not None
                else np.empty((0, dynamics_params_grid.shape[0]))
            ),
            target_log_mass=np.asarray(log_target_mass),
            sampler_log_mass=np.asarray(log_sampler_mass),
            target_mass=np.asarray(target_mass),
            sampler_mass=np.asarray(sampler_mass),
            sampler_log_density=np.asarray(sampler_log_density),
            beta=np.asarray(current_beta),
            target_center_c=np.asarray(target_center),
            target_logsumexp_centered=np.asarray(
                comparison_metrics["target_logsumexp_centered"]
            ),
            target_log_normalizer=np.asarray(
                comparison_metrics["target_log_normalizer"]
            ),
            target_reverse_kl_to_uniform=np.asarray(
                comparison_metrics["target_reverse_kl_to_uniform"]
            ),
            sampler_reverse_kl_to_uniform=np.asarray(
                comparison_metrics["sampler_reverse_kl_to_uniform"]
            ),
            dual_update_mode=np.asarray(
                self.dual_update_mode if self.fixed_radius else "fixed_beta"
            ),
            kl_radius=np.asarray(self.kl_radius if self.fixed_radius else np.nan),
            target_kl_radius_residual=np.asarray(
                target_radius_residual if self.fixed_radius else np.nan
            ),
            sampler_kl_radius_residual=np.asarray(
                sampler_radius_residual if self.fixed_radius else np.nan
            ),
            target_kl_radius_violation=np.asarray(
                jnp.maximum(target_radius_residual, 0.0)
                if self.fixed_radius
                else np.nan
            ),
            sampler_kl_radius_violation=np.asarray(
                jnp.maximum(sampler_radius_residual, 0.0)
                if self.fixed_radius
                else np.nan
            ),
        )

        target_2d = np.asarray(target_mass).reshape(x_grid.shape)
        sampler_2d = np.asarray(sampler_mass).reshape(x_grid.shape)
        difference_2d = sampler_2d - target_2d
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        plot_specs = (
            (target_2d, "Empirical target mass", "viridis"),
            (sampler_2d, "GMM sampler mass", "viridis"),
            (difference_2d, "Sampler - target mass", "coolwarm"),
        )
        for ax, (values, title, cmap) in zip(axes, plot_specs):
            image = ax.pcolormesh(x_grid, y_grid, values, shading="auto", cmap=cmap)
            ax.set_xlabel("dynamics dim 0")
            ax.set_ylabel("dynamics dim 1")
            ax.set_title(title)
            fig.colorbar(image, ax=ax)
        radius_label = (
            f", radius={float(self.kl_radius):.6g}, mode={self.dual_update_mode}"
            if self.fixed_radius
            else ", mode=fixed_beta"
        )
        fig.suptitle(
            f"{int(self.eval_grid_size_2d)}x{int(self.eval_grid_size_2d)} "
            f"empirical target comparison "
            f"[beta={float(current_beta):.6g}, "
            f"target KL(U)="
            f"{float(comparison_metrics['target_reverse_kl_to_uniform']):.6g}, "
            f"sampler KL(U)="
            f"{float(comparison_metrics['sampler_reverse_kl_to_uniform']):.6g}"
            f"{radius_label}, step={int(step)}]"
        )
        figure_path = os.path.join(self.save_dir, f"{output_stem}.png")
        fig.savefig(figure_path, dpi=150)
        if self.use_wandb:
            wandb.log(
                {"Empirical Target vs GMM": wandb.Image(fig)},
                step=int(step),
            )
        plt.close(fig)
        return metrics
