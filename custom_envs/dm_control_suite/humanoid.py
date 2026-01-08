# Copyright 2025 DeepMind Technologies Limited
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
# ==============================================================================
"""Humanoid environment."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import functools
from custom_envs import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "humanoid.xml"
# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.4

# Horizontal speeds above which move reward is 1.
WALK_SPEED = 1.0
RUN_SPEED = 10.0


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.025,
      sim_dt=0.005,  # 0.0025 in DM Control
      episode_length=1000,
      action_repeat=1,
      vision=False,
      impl="jax",
      nconmax=200_000,
      njmax=350,
  )


class Humanoid(mjx_env.MjxEnv):
  """Humanoid environment."""

  def __init__(
      self,
      move_speed: float,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    if self._config.vision:
      raise NotImplementedError(
          f"Vision not implemented for {self.__class__.__name__}."
      )

    self._move_speed = move_speed
    if self._move_speed == 0.0:
      self._stand_or_move_reward = self._stand_reward
    else:
      self._stand_or_move_reward = self._move_reward

    self._xml_path = _XML_PATH.as_posix()
    self._model_assets = common.get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        _XML_PATH.read_text(), self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._head_body_id = self.mj_model.body("head").id
    self._torso_body_id = self.mj_model.body("torso").id

    extremities_ids = []
    for side in ("left_", "right_"):
      for limb in ("hand", "foot"):
        extremities_ids.append(self.mj_model.body(side + limb).id)
    self._extremities_ids = jp.array(extremities_ids)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # TODO(kevin): Add non-penetrating joint randomization.

    data = mjx_env.make_data(
        self.mj_model,
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    metrics = {
        "reward/standing": jp.zeros(()),
        "reward/upright": jp.zeros(()),
        "reward/stand": jp.zeros(()),
        "reward/small_control": jp.zeros(()),
        "reward/move": jp.zeros(()),
    }
    info = {"rng": rng}

    reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
    reward = self._get_reward(data, action, state.info, state.metrics)  # pylint: disable=redefined-outer-name
    obs = self._get_obs(data, state.info)
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    del info  # Unused.
    state =  jp.concatenate([
        self._joint_angles(data),
        self._head_height(data).reshape(1),
        self._extremities(data).ravel(),
        self._torso_vertical_orientation(data),
        self._center_of_mass_velocity(data),
        data.qvel,
    ])
    privileged_state = jp.concatenate([
      state,
      self.mjx_model.geom_friction[FLOOR_GEOM_ID, 0:1],
      self.mjx_model.body_mass[TORSO_BODY_ID][..., None],
      # self.mjx_model.dof_frictionloss[6:],
      # self.mjx_model.dof_armature[6:],
      # self.mjx_model.body_mass[1:],
      # self.mjx_model.qpos0[7:],
    ])
    # return state
    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del info  # Unused.

    standing = reward.tolerance(
        self._head_height(data),
        bounds=(_STAND_HEIGHT, float("inf")),
        margin=_STAND_HEIGHT / 4,
    )
    metrics["reward/standing"] = standing

    upright = reward.tolerance(
        self._torso_upright(data),
        bounds=(0.9, float("inf")),
        sigmoid="linear",
        margin=1.9,
        value_at_margin=0,
    )
    metrics["reward/upright"] = upright

    stand_reward = standing * upright
    metrics["reward/stand"] = stand_reward

    small_control = reward.tolerance(
        action, margin=1, value_at_margin=0, sigmoid="quadratic"
    ).mean()
    small_control = (4 + small_control) / 5
    metrics["reward/small_control"] = small_control

    move_reward = self._stand_or_move_reward(data)
    metrics["reward/move"] = move_reward

    return stand_reward * move_reward * small_control

  def _stand_reward(self, data: mjx.Data) -> jax.Array:
    horizontal_velocity = self._center_of_mass_velocity(data)[:2]
    dont_move = reward.tolerance(horizontal_velocity, margin=2).mean()
    return dont_move

  def _move_reward(self, data: mjx.Data) -> jax.Array:
    move = reward.tolerance(
        jp.linalg.norm(self._center_of_mass_velocity(data)[:2]),
        bounds=(self._move_speed, float("inf")),
        margin=self._move_speed,
        value_at_margin=0,
        sigmoid="linear",
    )
    move = (5 * move + 1) / 6
    return move

  def _joint_angles(self, data: mjx.Data) -> jax.Array:
    """Returns the state without global orientation or position."""
    return data.qpos[7:]

  def _torso_vertical_orientation(self, data: mjx.Data) -> jax.Array:
    """Returns the z-projection of the torso orientation matrix."""
    return data.xmat[self._torso_body_id, 2]

  def _center_of_mass_velocity(self, data: mjx.Data) -> jax.Array:
    """Returns the velocity of the center of mass in global coordinates."""
    return mjx_env.get_sensor_data(self.mj_model, data, "torso_subtreelinvel")

  def _center_of_mass_position(self, data: mjx.Data) -> jax.Array:
    """Returns the position of the center of mass in global coordinates."""
    return data.subtree_com[self._torso_body_id]

  def _head_height(self, data: mjx.Data) -> jax.Array:
    """Returns the height of the torso."""
    return data.xpos[self._head_body_id, -1]

  def _torso_upright(self, data: mjx.Data) -> jax.Array:
    """Returns projection from z-axes of torso to the z-axes of world."""
    return data.xmat[self._torso_body_id, 2, 2]

  def _extremities(self, data: mjx.Data) -> jax.Array:
    """Returns end effector positions in the egocentric frame."""
    torso_frame = data.xmat[self._torso_body_id]
    torso_pos = data.xpos[self._torso_body_id]
    torso_to_limb = data.xpos[self._extremities_ids] - torso_pos
    return torso_to_limb @ torso_frame

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
  @property
  def nominal_params(self) -> jp.ndarray:
    return jp.concatenate([jp.ones(43), jp.zeros(1), jp.ones(15), jp.zeros(21)])
  @property
  def dr_range(self) -> dict:
    #total 80d
    low = jp.array(
        [0.4] +                              #floor_friction_min 
        [0.8] *(self.mjx_model.nv-6) +       #dof friciton(21d)
        [1.] * (self.mjx_model.nv-6) +       #dof armature(21d)
        [-1.] +                              #torso mass(1d)
        [0.8] * (self.mjx_model.nbody-2) +   #link masses(15d)
        [-.05] * (self.mjx_model.nv-6)       #qpos0(21d)
      )
    high = jp.array(
        [4.0] +                              #floor_friction_min 
        [1.2] *(self.mjx_model.nv-6) +       #dof friciton(21d)
        [1.2] * (self.mjx_model.nv-6) +     #dof armature(21d)
        [1.0] +                              #torso mass(1d)
        [1.2] * (self.mjx_model.nbody-2) +   #link masses(15d)
        [.05] * (self.mjx_model.nv-6)        #qpos0(21d)
      )
    return low, high
  
FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1
def domain_randomize(model: mjx.Model, dr_range, params=None, rng:jax.Array=None):
  if rng is not None:
    dr_low, dr_high = dr_range
    dist = functools.partial(jax.random.uniform, shape=(len(dr_low)), minval=dr_low, maxval=dr_high)
  @jax.vmap
  def shift_dynamics(params):
    idx = 0
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(params[idx])
    idx += 1
    dof_frictionloss = model.dof_frictionloss.at[6:].set(params[idx:idx+21])
    idx += 21
    dof_armature  = model.dof_armature.at[6:].set(params[idx:idx+21])
    idx += 21
    body_mass = model.body_mass.at[1].set(params[idx])
    body_mass = model.body_mass.at[2:].set(params[idx+1: idx+16])
    idx += 16
    qpos0 = model.qpos0.at[7:].set(params[idx:idx+21])
    idx +=21
    assert idx == len(params)
    return (
      geom_friction,
      dof_frictionloss,
      dof_armature,
      body_mass,
      qpos0
    )
  @jax.vmap
  def rand_dynamics(rng):
    rng_params = dist(rng)
    idx = 0
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(rng_params[idx])
    idx += 1
    dof_frictionloss = model.dof_frictionloss.at[6:].set(rng_params[idx:idx+21])
    idx += 21
    dof_armature  = model.dof_armature.at[6:].set(rng_params[idx:idx+21])
    idx += 21
    body_mass = model.body_mass.at[1].set(model.body_mass[1] + rng_params[idx])
    body_mass = model.body_mass.at[2:].set(rng_params[idx+1: idx+16])
    idx += 16
    qpos0 = model.qpos0.at[7:].set(rng_params[idx:idx+21])
    idx +=21
    assert idx == len(rng_params)
    return (
      geom_friction,
      dof_frictionloss,
      dof_armature,
      body_mass,
      qpos0
    )
  if rng is None and params is not None:

    (geom_friction, 
     dof_frictionloss,
     dof_armature,
     body_mass, 
     qpos0,
     ) = shift_dynamics(params)
  elif rng is not None and params is None:
    (geom_friction, 
     dof_frictionloss,
     dof_armature,
     body_mass, 
     qpos0,
     ) = rand_dynamics(rng)
  else:
    raise ValueError("rng and params wrong!")
  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
  })
  model = model.tree_replace({
      "geom_friction": geom_friction,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "body_mass": body_mass,
      "qpos0" : qpos0,
  })

  return model, in_axes
def domain_randomize_eval(model: mjx.Model, dr_range, params=None, rng:jax.Array=None):
  if rng is not None:
    dr_low, dr_high = dr_range
    dist = functools.partial(jax.random.uniform, shape=(len(dr_low)), minval=dr_low, maxval=dr_high)

  def shift_dynamics(params):
    idx = 0
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(params[idx])
    idx += 1
    dof_frictionloss = model.dof_frictionloss.at[6:].set(params[idx:idx+21])
    idx += 21
    dof_armature  = model.dof_armature.at[6:].set(params[idx:idx+21])
    idx += 21
    body_mass = model.body_mass.at[1].set(params[idx])
    body_mass = model.body_mass.at[2:].set(params[idx+1: idx+16])
    idx += 16
    qpos0 = model.qpos0.at[7:].set(params[idx:idx+21])
    idx +=21
    assert idx == len(params)
    return (
      geom_friction,
      dof_frictionloss,
      dof_armature,
      body_mass,
      qpos0
    )
  def rand_dynamics(rng):
    rng_params = dist(rng)
    idx = 0
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(rng_params[idx])
    idx += 1
    dof_frictionloss = model.dof_frictionloss.at[6].set(rng_params[idx])
    idx+=1
    dof_frictionloss = model.dof_frictionloss.at[6:].set(rng_params[idx:idx+21])
    idx += 21
    dof_armature  = model.dof_armature.at[6:].set(rng_params[idx:idx+21])
    idx += 21
    body_mass = model.body_mass.at[1].set(rng_params[idx])
    body_mass = model.body_mass.at[2:].set(rng_params[idx+1: idx+16])
    idx += 16
    qpos0 = model.qpos0.at[7:].set(rng_params[idx:idx+21])
    idx +=21
    assert idx == len(rng_params)
    return (
      geom_friction,
      dof_frictionloss,
      dof_armature,
      body_mass,
      qpos0
    )
  if rng is None and params is not None:

    (geom_friction, 
     dof_frictionloss,
     dof_armature,
     body_mass, 
     qpos0,
     ) = shift_dynamics(params)
  elif rng is not None and params is None:
    (geom_friction, 
     dof_frictionloss,
     dof_armature,
     body_mass, 
     qpos0,
     ) = rand_dynamics(rng)
  else:
    raise ValueError("rng and params wrong!")
  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
  })
  model = model.tree_replace({
      "geom_friction": geom_friction,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "body_mass": body_mass,
      "qpos0" : qpos0,
  })
  return model, in_axes