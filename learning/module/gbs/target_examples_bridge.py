from __future__ import annotations

from dataclasses import dataclass, field
import importlib

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class TargetExampleSpec:
    module_name: str
    class_name: str
    lower: float
    upper: float
    extra_kwargs: dict[str, object] = field(default_factory=dict)


TARGET_EXAMPLE_SPECS: dict[str, TargetExampleSpec] = {
    "funnel": TargetExampleSpec(
        module_name="learning.module.target_examples.funnel",
        class_name="Funnel",
        lower=-12.0,
        upper=12.0,
        extra_kwargs={"sample_bounds": (-30.0, 30.0)},
    ),
    "gmm40": TargetExampleSpec(
        module_name="learning.module.target_examples.gmm40",
        class_name="GMM40",
        lower=-60.0,
        upper=60.0,
    ),
    "gmm1d": TargetExampleSpec(
        module_name="learning.module.target_examples.gmm1d",
        class_name="GMM1D",
        lower=-5.0,
        upper=5.0,
    ),
    "rings": TargetExampleSpec(
        module_name="learning.module.target_examples.rings",
        class_name="ConcentricRings",
        lower=-20.0,
        upper=20.0,
    ),
    "student_t_mixture": TargetExampleSpec(
        module_name="learning.module.target_examples.student_t_mixture",
        class_name="StudentTMixtureModel",
        lower=-15.0,
        upper=15.0,
        extra_kwargs={"num_components": 4},
    ),
    "many_well": TargetExampleSpec(
        module_name="learning.module.target_examples.many_well",
        class_name="ManyWellEnergy",
        lower=-3.0,
        upper=3.0,
    ),
    "lennard_jones": TargetExampleSpec(
        module_name="learning.module.target_examples.lennard_jones",
        class_name="LennardJones",
        lower=-3.0,
        upper=3.0,
    ),
}


def target_example_names() -> tuple[str, ...]:
    return tuple(TARGET_EXAMPLE_SPECS.keys())


def is_target_example(name: str) -> bool:
    return str(name) in TARGET_EXAMPLE_SPECS


def load_target_example(name: str, dim: int, n_spatial_dim: int = 1):
    example_name = str(name)
    if example_name not in TARGET_EXAMPLE_SPECS:
        raise ValueError(f"Unsupported target example: {example_name}")
    spec = TARGET_EXAMPLE_SPECS[example_name]
    module = importlib.import_module(spec.module_name)
    cls = getattr(module, spec.class_name)
    kwargs = dict(spec.extra_kwargs)
    if example_name == "lennard_jones":
        if n_spatial_dim <= 0:
            raise ValueError(f"n_spatial_dim must be positive for {example_name}, got {n_spatial_dim}")
        if dim % n_spatial_dim != 0:
            raise ValueError(
                f"Total dim {dim} must be divisible by n_spatial_dim={n_spatial_dim} for {example_name}"
            )
        kwargs.setdefault("dim", dim // n_spatial_dim)
        kwargs.setdefault("spatial_dim", n_spatial_dim)
    else:
        kwargs.setdefault("dim", dim)
    return cls(**kwargs)


def target_example_bounds(name: str, dim: int):
    example_name = str(name)
    if example_name not in TARGET_EXAMPLE_SPECS:
        raise ValueError(f"Unsupported target example: {example_name}")
    spec = TARGET_EXAMPLE_SPECS[example_name]
    low = jnp.full((dim,), spec.lower, dtype=jnp.float32)
    high = jnp.full((dim,), spec.upper, dtype=jnp.float32)
    return low, high


def sample_target_example_reference(target_example, key, n_samples: int):
    if getattr(target_example, "can_sample", False):
        return target_example.sample(key, (n_samples,))
    if hasattr(target_example, "test_set"):
        test_set = jnp.asarray(target_example.test_set)
        idx = jax.random.randint(key, (n_samples,), 0, test_set.shape[0])
        return test_set[idx]
    raise ValueError(
        f"Target example {type(target_example).__name__} does not provide a sampler or test_set."
    )


def target_example_has_reference(target_example) -> bool:
    return bool(getattr(target_example, "can_sample", False) or hasattr(target_example, "test_set"))
