"""Computational-cost accounting for sampler PPO training."""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Callable, Dict, Tuple


@dataclasses.dataclass(frozen=True)
class TrainingCostCounts:
    """Cumulative algorithmic work completed by a training run."""

    env_transitions: int
    policy_updates: int
    policy_optimizer_steps: int
    sampler_updates: int


def env_transitions_per_training_step(
    *,
    sampler_choice: str,
    batch_size: int,
    num_minibatches: int,
    num_envs: int,
    unroll_length: int,
    action_repeat: int,
    epopt_epsilon: float,
) -> int:
    """Returns actual collected transitions, including EPOpt oversampling."""
    num_unrolls = batch_size * num_minibatches // num_envs
    if sampler_choice == "EPOpt":
        if not 0.0 < epopt_epsilon <= 1.0:
            raise ValueError("EPOpt epsilon must be in (0, 1].")
        num_unrolls = int(batch_size * num_minibatches / epopt_epsilon // num_envs)
    return num_envs * num_unrolls * unroll_length * action_repeat


def sampler_updates_after_policy_batches(
    *,
    sampler_choice: str,
    policy_batches: int,
    sampler_update_freq: int,
    n_sampler_iters: int,
) -> int:
    """Counts sampler optimizer updates triggered before each batch increment."""
    if sampler_choice in ("NODR", "UDR", "EPOpt") or policy_batches <= 0:
        return 0
    if sampler_update_freq <= 0:
        raise ValueError("sampler_update_freq must be positive.")
    update_events = (policy_batches - 1) // sampler_update_freq + 1
    iterations_per_event = (
        n_sampler_iters if "FLOW" in sampler_choice or sampler_choice == "GBS" else 1
    )
    return update_events * iterations_per_event


def training_cost_counts(
    *,
    policy_batches: int,
    sampler_choice: str,
    env_transitions_per_batch: int,
    num_updates_per_batch: int,
    num_minibatches: int,
    sampler_update_freq: int,
    n_sampler_iters: int,
) -> TrainingCostCounts:
    """Returns cumulative transition and optimizer-update counts."""
    return TrainingCostCounts(
        env_transitions=policy_batches * env_transitions_per_batch,
        policy_updates=policy_batches,
        policy_optimizer_steps=(
            policy_batches * num_updates_per_batch * num_minibatches
        ),
        sampler_updates=sampler_updates_after_policy_batches(
            sampler_choice=sampler_choice,
            policy_batches=policy_batches,
            sampler_update_freq=sampler_update_freq,
            n_sampler_iters=n_sampler_iters,
        ),
    )


class TrainingPhaseTimer:
    """Aggregates ordered device-callback timestamps for rollout/update phases.

    This timer is intended only for explicitly instrumented, single-device cost
    runs. Device callbacks introduce synchronization and therefore should not be
    enabled for ordinary throughput experiments.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self._clock = clock
        self._lock = threading.Lock()
        self._phase_starts: Dict[int, Tuple[float, float | None]] = {}
        self._rollout_seconds = 0.0
        self._update_seconds = 0.0

    def record(self, phase: str, update_step) -> None:
        """Records one of rollout_start, rollout_end, or update_end."""
        step = int(update_step)
        now = self._clock()
        with self._lock:
            if phase == "rollout_start":
                self._phase_starts[step] = (now, None)
            elif phase == "rollout_end":
                rollout_start, _ = self._phase_starts[step]
                self._rollout_seconds += now - rollout_start
                self._phase_starts[step] = (rollout_start, now)
            elif phase == "update_end":
                _, update_start = self._phase_starts.pop(step)
                if update_start is None:
                    raise RuntimeError("update_end received before rollout_end.")
                self._update_seconds += now - update_start
            else:
                raise ValueError(f"Unknown training phase: {phase}")

    def totals(self) -> Tuple[float, float]:
        with self._lock:
            return self._rollout_seconds, self._update_seconds
