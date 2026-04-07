"""Shared domain randomization utilities for locomotion envs."""

import functools

import jax
import jax.numpy as jp
from mujoco import mjx


def make_default_dr_range(
    model: mjx.Model,
    *,
    floor_friction_range: tuple[float, float] = (0.4, 1.0),
    dof_friction_range: tuple[float, float] = (0.9, 1.1),
    armature_range: tuple[float, float] = (1.0, 1.05),
    body_mass_range: tuple[float, float] = (0.9, 1.1),
    torso_mass_offset_range: tuple[float, float] = (-1.0, 1.0),
    qpos_offset_range: tuple[float, float] = (-0.05, 0.05),
    dof_damping_range: tuple[float, float] = (0.8, 1.2),
    actuator_gain_range: tuple[float, float] = (0.9, 1.1),
) -> tuple[jax.Array, jax.Array]:
  n_dofs = model.nv - 6
  low = jp.array(
      [floor_friction_range[0]]
      + [dof_friction_range[0]] * n_dofs
      + [armature_range[0]] * n_dofs
      + [body_mass_range[0]] * model.nbody
      + [torso_mass_offset_range[0]]
      + [qpos_offset_range[0]] * n_dofs
      + [dof_damping_range[0]] * n_dofs
      + [actuator_gain_range[0]] * model.nu
  )
  high = jp.array(
      [floor_friction_range[1]]
      + [dof_friction_range[1]] * n_dofs
      + [armature_range[1]] * n_dofs
      + [body_mass_range[1]] * model.nbody
      + [torso_mass_offset_range[1]]
      + [qpos_offset_range[1]] * n_dofs
      + [dof_damping_range[1]] * n_dofs
      + [actuator_gain_range[1]] * model.nu
  )
  return low, high


def make_default_nominal_params(model: mjx.Model, *, floor_geom_id: int = 0):
  n_dofs = model.nv - 6
  return jp.concatenate([
      jp.array([model.geom_friction[floor_geom_id, 0]]),
      jp.ones(n_dofs),
      jp.ones(n_dofs),
      jp.ones(model.nbody),
      jp.zeros(1),
      jp.zeros(n_dofs),
      jp.ones(n_dofs),
      jp.ones(model.nu),
  ])


def _apply_legged_randomization(
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

  body_mass = model.body_mass.at[:].set(
      model.body_mass * params[idx : idx + model.nbody]
  )
  idx += model.nbody

  body_mass = body_mass.at[torso_body_id].set(body_mass[torso_body_id] + params[idx])
  idx += 1

  qpos0 = model.qpos0.at[7:].set(model.qpos0[7:] + params[idx : idx + n_dofs])
  idx += n_dofs

  dof_damping = model.dof_damping.at[6:].set(
      model.dof_damping[6:] * params[idx : idx + n_dofs]
  )
  idx += n_dofs

  kp_val = model.actuator_gainprm[:, 0] * params[idx : idx + model.nu]
  actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp_val)
  actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp_val)
  idx += model.nu

  assert idx == params.shape[0]
  return (
      geom_friction,
      dof_frictionloss,
      dof_armature,
      body_mass,
      qpos0,
      dof_damping,
      actuator_gainprm,
      actuator_biasprm,
  )


def _finalize_randomization(model: mjx.Model, randomized):
  (
      geom_friction,
      dof_frictionloss,
      dof_armature,
      body_mass,
      qpos0,
      dof_damping,
      actuator_gainprm,
      actuator_biasprm,
  ) = randomized

  model = model.tree_replace({
      "geom_friction": geom_friction,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "body_mass": body_mass,
      "qpos0": qpos0,
      "dof_damping": dof_damping,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
  })
  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "qpos0": 0,
      "dof_damping": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
  })
  return model, in_axes


def domain_randomize_batched(
    model: mjx.Model,
    dr_range,
    *,
    floor_geom_id: int,
    torso_body_id: int,
    rng: jax.Array = None,
    params: jax.Array = None,
):
  dr_low, dr_high = dr_range

  if rng is not None and params is None:
    dist = functools.partial(
        jax.random.uniform,
        shape=(len(dr_low),),
        minval=dr_low,
        maxval=dr_high,
    )
    randomized = jax.vmap(
        lambda rng_i: _apply_legged_randomization(
            model,
            dist(rng_i),
            floor_geom_id=floor_geom_id,
            torso_body_id=torso_body_id,
        )
    )(rng)
  elif params is not None and rng is None:
    if params.ndim == 1:
      randomized = _apply_legged_randomization(
          model,
          params,
          floor_geom_id=floor_geom_id,
          torso_body_id=torso_body_id,
      )
    else:
      randomized = jax.vmap(
          lambda p: _apply_legged_randomization(
              model,
              p,
              floor_geom_id=floor_geom_id,
              torso_body_id=torso_body_id,
          )
      )(params)
  else:
    raise ValueError("Exactly one of rng or params must be provided.")

  return _finalize_randomization(model, randomized)


def domain_randomize_single(
    model: mjx.Model,
    dr_range,
    *,
    floor_geom_id: int,
    torso_body_id: int,
    rng: jax.Array = None,
    params=None,
):
  dr_low, dr_high = dr_range
  if rng is not None and params is None:
    params = jax.random.uniform(
        rng, shape=(len(dr_low),), minval=dr_low, maxval=dr_high
    )
  elif params is None:
    raise ValueError("Exactly one of rng or params must be provided.")

  randomized = _apply_legged_randomization(
      model,
      params,
      floor_geom_id=floor_geom_id,
      torso_body_id=torso_body_id,
  )
  return _finalize_randomization(model, randomized)
