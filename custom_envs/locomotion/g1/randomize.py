"""Domain randomization for G1."""

import functools

import jax
import jax.numpy as jp
from mujoco import mjx

TORSO_BODY_ID = 16


def _apply_randomization(model: mjx.Model, params: jax.Array):
  idx = 0
  n_dofs = model.nv - 6

  pair_friction = model.pair_friction.at[0:2, 0:2].set(params[idx])
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

  assert idx == params.shape[0]

  randomized_model = model.tree_replace({
      "pair_friction": pair_friction,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "body_mass": body_mass,
      "qpos0": qpos0,
  })
  in_axes = jax.tree_util.tree_map(lambda x: None, randomized_model)
  in_axes = in_axes.tree_replace({
      "pair_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "qpos0": 0,
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
  in_axes = jax.tree_util.tree_map(lambda x: None, _apply_randomization(model, params[0])[0])
  in_axes = in_axes.tree_replace({
      "pair_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "qpos0": 0,
  })
  return model_v, in_axes


def domain_randomize_eval(
    model: mjx.Model, dr_range, params=None, rng: jax.Array = None
):
  if params is None and rng is None:
    n_dofs = model.nv - 6
    params = jp.concatenate([
        jp.array([model.pair_friction[0, 0]]),
        jp.ones(n_dofs),
        jp.ones(n_dofs),
        jp.ones(model.nbody),
        jp.zeros(1),
        jp.zeros(n_dofs),
    ])
  if rng is not None and params is None:
    dr_low, dr_high = dr_range
    params = jax.random.uniform(rng, shape=(len(dr_low),), minval=dr_low, maxval=dr_high)
  return _apply_randomization(model, params)
