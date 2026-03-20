from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np
import yaml


@dataclass(frozen=True)
class DomainRandomizationSpec:
    task: str
    param_names: tuple[str, ...]
    groups: tuple[str, ...]
    full_low: jnp.ndarray
    full_high: jnp.ndarray
    dynamics_mask: jnp.ndarray
    reset_mask: jnp.ndarray
    learnable_mask: jnp.ndarray
    learnable_low: jnp.ndarray
    learnable_high: jnp.ndarray

    @property
    def full_dim(self) -> int:
        return int(self.full_low.shape[0])

    @property
    def learnable_dim(self) -> int:
        return int(self.learnable_low.shape[0])


_DR_CONFIG_DIR = Path(__file__).resolve().parent / "dr_configs"


def _load_yaml(task: str) -> Optional[dict]:
    path = _DR_CONFIG_DIR / f"{task}.yaml"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dr_spec(
    task: str,
    include_reset_params: bool = True,
    enable_dynamics_learning: bool = True,
    enable_reset_learning: bool = True,
) -> Optional[DomainRandomizationSpec]:
    raw = _load_yaml(task)
    if raw is None:
        return None
    parameters = raw.get("parameters", [])
    kept = []
    for entry in parameters:
        group = entry["group"]
        if group == "reset" and not include_reset_params:
            continue
        kept.append(entry)

    if not kept:
        raise ValueError(f"No DR parameters remain after filtering task '{task}'.")
    param_names = tuple(entry["name"] for entry in kept)
    groups = tuple(entry["group"] for entry in kept)
    full_low = jnp.asarray([float(entry["low"]) for entry in kept], dtype=jnp.float32)
    full_high = jnp.asarray([float(entry["high"]) for entry in kept], dtype=jnp.float32)

    dynamics_mask_np = np.asarray([group == "dynamics" for group in groups], dtype=bool)
    reset_mask_np = np.asarray([group == "reset" for group in groups], dtype=bool)
    default_learnable = np.asarray(
        [bool(entry.get("learnable", True)) for entry in kept],
        dtype=bool,
    )
    group_enabled = np.asarray(
        [
            (group == "dynamics" and enable_dynamics_learning)
            or (group == "reset" and enable_reset_learning)
            for group in groups
        ],
        dtype=bool,
    )
    learnable_mask_np = np.logical_and(default_learnable, group_enabled)
    if not np.any(learnable_mask_np):
        raise ValueError(
            f"Task '{task}' has no learnable DR parameters under the current flags."
        )

    learnable_mask = jnp.asarray(learnable_mask_np)
    return DomainRandomizationSpec(
        task=task,
        param_names=param_names,
        groups=groups,
        full_low=full_low,
        full_high=full_high,
        dynamics_mask=jnp.asarray(dynamics_mask_np),
        reset_mask=jnp.asarray(reset_mask_np),
        learnable_mask=learnable_mask,
        learnable_low=full_low[learnable_mask],
        learnable_high=full_high[learnable_mask],
    )


def get_structural_dr_bounds(
    task: str,
    include_reset_params: bool = True,
) -> Optional[tuple[jnp.ndarray, jnp.ndarray]]:
    spec = build_dr_spec(
        task,
        include_reset_params=include_reset_params,
        enable_dynamics_learning=True,
        enable_reset_learning=True,
    )
    if spec is None:
        return None
    return spec.full_low, spec.full_high
