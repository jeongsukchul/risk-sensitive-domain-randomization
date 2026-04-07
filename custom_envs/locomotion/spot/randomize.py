"""Domain randomization for Spot locomotion environments."""

import jax
from mujoco import mjx

from custom_envs.locomotion import randomization_utils

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1


def domain_randomize(model: mjx.Model, dr_range, rng: jax.Array, params=None):
  return randomization_utils.domain_randomize_batched(
      model,
      dr_range,
      floor_geom_id=FLOOR_GEOM_ID,
      torso_body_id=TORSO_BODY_ID,
      rng=rng,
      params=params,
  )


def domain_randomize_eval(
    model: mjx.Model, dr_range, rng: jax.Array = None, params=None
):
  return randomization_utils.domain_randomize_single(
      model,
      dr_range,
      floor_geom_id=FLOOR_GEOM_ID,
      torso_body_id=TORSO_BODY_ID,
      rng=rng,
      params=params,
  )
