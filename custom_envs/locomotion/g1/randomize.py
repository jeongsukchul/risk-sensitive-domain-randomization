"""Domain randomization for G1 locomotion."""

import jax
from mujoco import mjx

from custom_envs.locomotion import randomization_utils


def domain_randomize(
    model: mjx.Model,
    dr_range: tuple[jax.Array, jax.Array],
    rng: jax.Array = None,
    params: jax.Array = None,
):
  return randomization_utils.domain_randomize(
      model, dr_range, rng=rng, params=params, floor_geom_id=0, torso_body_id=16
  )


def domain_randomize_eval(
    model: mjx.Model,
    dr_range: tuple[jax.Array, jax.Array],
    rng: jax.Array = None,
    params: jax.Array = None,
):
  return domain_randomize(model, dr_range, rng=rng, params=params)
