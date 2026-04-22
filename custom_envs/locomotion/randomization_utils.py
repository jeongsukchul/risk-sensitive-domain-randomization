"""Domain randomization helpers for local locomotion training wrappers."""

import functools
from typing import Optional

import jax
import jax.numpy as jp
from mujoco import mjx


def make_default_dr_range(
    model: mjx.Model,
    *,
    floor_friction_range: tuple[float, float] = (0.4, 1.5),
    dof_friction_range: tuple[float, float] = (0.9, 1.1),
    armature_range: tuple[float, float] = (1.0, 1.05),
    torso_ipos_offset_range: tuple[float, float] = (-0.05, 0.05),
    body_mass_range: tuple[float, float] = (0.5, 1.5),
    torso_mass_offset_range: tuple[float, float] = (-1.5, 1.5),
    qpos_offset_range: tuple[float, float] = (-0.05, 0.05),
) -> tuple[jax.Array, jax.Array]:
  """Builds a parameter range matching current Playground locomotion DR fields."""
  n_dofs = model.nv - 6
  low = jp.array(
      [floor_friction_range[0]]
      + [dof_friction_range[0]] * n_dofs
      + [armature_range[0]] * n_dofs
      + [torso_ipos_offset_range[0]] * 3
      + [body_mass_range[0]] * model.nbody
      + [torso_mass_offset_range[0]]
      + [qpos_offset_range[0]] * n_dofs
  )
  high = jp.array(
      [floor_friction_range[1]]
      + [dof_friction_range[1]] * n_dofs
      + [armature_range[1]] * n_dofs
      + [torso_ipos_offset_range[1]] * 3
      + [body_mass_range[1]] * model.nbody
      + [torso_mass_offset_range[1]]
      + [qpos_offset_range[1]] * n_dofs
  )
  return low, high


def make_default_nominal_params(model: mjx.Model) -> jax.Array:
  """Builds neutral parameters matching make_default_dr_range."""
  n_dofs = model.nv - 6
  return jp.array(
      [1.0]
      + [1.0] * n_dofs
      + [1.0] * n_dofs
      + [0.0] * 3
      + [1.0] * model.nbody
      + [0.0]
      + [0.0] * n_dofs
  )


def _apply_randomization(
    model: mjx.Model,
    params: jax.Array,
    *,
    floor_geom_id: int,
    torso_body_id: int,
):
  idx = 0
  n_dofs = model.nv - 6

  geom_friction = model.geom_friction.at[floor_geom_id, 0].set(params[idx])
  idx += 1

  dof_frictionloss = model.dof_frictionloss.at[6:].set(
      model.dof_frictionloss[6:] * params[idx : idx + n_dofs]
  )
  idx += n_dofs

  dof_armature = model.dof_armature.at[6:].set(
      model.dof_armature[6:] * params[idx : idx + n_dofs]
  )
  idx += n_dofs

  body_ipos = model.body_ipos.at[torso_body_id].set(
      model.body_ipos[torso_body_id] + params[idx : idx + 3]
  )
  idx += 3

  body_mass = model.body_mass.at[:].set(
      model.body_mass * params[idx : idx + model.nbody]
  )
  idx += model.nbody

  body_mass = body_mass.at[torso_body_id].set(
      body_mass[torso_body_id] + params[idx]
  )
  idx += 1

  qpos0 = model.qpos0.at[7:].set(model.qpos0[7:] + params[idx : idx + n_dofs])
  idx += n_dofs

  assert idx == params.shape[0]
  return geom_friction, dof_frictionloss, dof_armature, body_ipos, body_mass, qpos0


def _finalize(model: mjx.Model, randomized):
  geom_friction, dof_frictionloss, dof_armature, body_ipos, body_mass, qpos0 = randomized
  model = model.tree_replace({
      "geom_friction": geom_friction,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "body_ipos": body_ipos,
      "body_mass": body_mass,
      "qpos0": qpos0,
  })

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_ipos": 0,
      "body_mass": 0,
      "qpos0": 0,
  })
  return model, in_axes


def domain_randomize(
    model: mjx.Model,
    dr_range: tuple[jax.Array, jax.Array],
    rng: Optional[jax.Array] = None,
    params: Optional[jax.Array] = None,
    *,
    floor_geom_id: int = 0,
    torso_body_id: int = 1,
):
  """Applies locomotion DR from either rng samples or explicit params."""
  dr_low, dr_high = dr_range

  if rng is not None and params is None:
    dist = functools.partial(
        jax.random.uniform,
        shape=(len(dr_low),),
        minval=dr_low,
        maxval=dr_high,
    )
    randomized = jax.vmap(
        lambda key: _apply_randomization(
            model,
            dist(key),
            floor_geom_id=floor_geom_id,
            torso_body_id=torso_body_id,
        )
    )(rng)
  elif params is not None and rng is None:
    if params.ndim == 1:
      randomized = _apply_randomization(
          model,
          params,
          floor_geom_id=floor_geom_id,
          torso_body_id=torso_body_id,
      )
    else:
      randomized = jax.vmap(
          lambda p: _apply_randomization(
              model,
              p,
              floor_geom_id=floor_geom_id,
              torso_body_id=torso_body_id,
          )
      )(params)
  else:
    raise ValueError("Exactly one of rng or params must be provided.")

  return _finalize(model, randomized)


def domain_randomize_eval(
    model: mjx.Model,
    dr_range: tuple[jax.Array, jax.Array],
    rng: Optional[jax.Array] = None,
    params: Optional[jax.Array] = None,
    *,
    floor_geom_id: int = 0,
    torso_body_id: int = 1,
):
  return domain_randomize(
      model,
      dr_range,
      rng=rng,
      params=params,
      floor_geom_id=floor_geom_id,
      torso_body_id=torso_body_id,
  )
