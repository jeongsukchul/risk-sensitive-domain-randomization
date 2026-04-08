"""Domain randomization for T1."""

import functools

import jax
import jax.numpy as jp
from mujoco import mjx
import numpy as np

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1
ANKLE_JOINT_IDS = np.array([15, 16, 21, 22])


def _apply_randomization(model: mjx.Model, params: jax.Array):
  idx = 0
  n_dofs = model.nv - 6

  geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(params[idx])
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

  body_mass = body_mass.at[TORSO_BODY_ID].set(body_mass[TORSO_BODY_ID] + params[idx])
  idx += 1

  qpos0 = model.qpos0.at[7:].set(model.qpos0[7:] + params[idx : idx + n_dofs])
  idx += n_dofs

  kp_val = model.actuator_gainprm[:, 0] * params[idx : idx + model.nu]
  actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp_val)
  actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp_val)
  idx += model.nu

  damping_scales = params[idx : idx + n_dofs]
  dof_damping = model.dof_damping.at[6:].set(model.dof_damping[6:] * damping_scales)
  dof_damping = dof_damping.at[6 + ANKLE_JOINT_IDS].set(
      model.dof_damping[6 + ANKLE_JOINT_IDS]
      * jp.clip(damping_scales[ANKLE_JOINT_IDS], 0.5, 2.0)
  )
  idx += n_dofs

  assert idx == params.shape[0]

  randomized_model = model.tree_replace({
      "geom_friction": geom_friction,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "body_mass": body_mass,
      "qpos0": qpos0,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
      "dof_damping": dof_damping,
  })
  in_axes = jax.tree_util.tree_map(lambda x: None, randomized_model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "qpos0": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
      "dof_damping": 0,
  })
  return randomized_model, in_axes


def domain_randomize(
    model: mjx.Model, dr_range, params=None, rng: jax.Array = None
):
  dr_low, dr_high = dr_range
  if rng is not None and params is None:
    dist = functools.partial(
        jax.random.uniform,
        shape=(len(dr_low),),
        minval=dr_low,
        maxval=dr_high,
    )
    params = jax.vmap(dist)(rng)
  elif params is None:
    raise ValueError("Exactly one of rng or params must be provided.")

  if params.ndim == 1:
    return _apply_randomization(model, params)
  model_v = jax.vmap(lambda p: _apply_randomization(model, p)[0])(params)
  _, in_axes = _apply_randomization(model, params[0])
  return model_v, in_axes


def domain_randomize_eval(
    model: mjx.Model, dr_range, params=None, rng: jax.Array = None
):
  if params is None and rng is None:
    n_dofs = model.nv - 6
    params = jp.concatenate([
        jp.array([model.geom_friction[FLOOR_GEOM_ID, 0]]),
        jp.ones(n_dofs),
        jp.ones(n_dofs),
        jp.ones(model.nbody),
        jp.zeros(1),
        jp.zeros(n_dofs),
        jp.ones(model.nu),
        jp.ones(n_dofs),
    ])
  if rng is not None and params is None:
    dr_low, dr_high = dr_range
    params = jax.random.uniform(rng, shape=(len(dr_low),), minval=dr_low, maxval=dr_high)
  return _apply_randomization(model, params)
