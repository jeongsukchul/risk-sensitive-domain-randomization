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
"""Fish environment."""

import functools
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from custom_envs import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "fish.xml"
_JOINTS = [
    "tail1",
    "tail_twist",
    "tail2",
    "finright_roll",
    "finright_pitch",
    "finleft_roll",
    "finleft_pitch",
]

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 2
def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.04,
      sim_dt=0.002,
      episode_length=1000,
      action_repeat=1,
      vision=False,
      impl="jax",
      nconmax=0,
      njmax=25,
  )


class Swim(mjx_env.MjxEnv):
  """Fish environment."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    if self._config.vision:
      raise NotImplementedError(
          f"Vision not implemented for {self.__class__.__name__}."
      )

    self._xml_path = _XML_PATH.as_posix()
    self._model_assets = common.get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        _XML_PATH.read_text(), self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._torso_body_id = self._mj_model.body("torso").id
    self._joints_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id] for j in _JOINTS
    ])
    self._target_geom_id = self._mj_model.geom("target").id
    self._target_body_id = self._mj_model.body("target").id
    self._mouth_geom_id = self._mj_model.geom("mouth").id
    self._target_mocap_body_id = self.mj_model.body("target").mocapid[0]

    self._radii = self._mj_model.geom_size[
        [self._mouth_geom_id, self._target_geom_id], 0
    ].sum()

    self._min_xyz = jp.array([-0.4, -0.4, 0.1])
    self._max_xyz = jp.array([0.4, 0.4, 0.3])

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

    qpos = jp.zeros(self.mjx_model.nq)

    quat = jax.random.normal(rng1, (4,))
    quat = quat / jp.linalg.norm(quat)
    qpos = qpos.at[3:7].set(quat)

    qpos = qpos.at[self._joints_qposadr].set(
        jax.random.uniform(
            rng2, (len(self._joints_qposadr),), minval=-0.2, maxval=0.2
        )
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # Randomize target position.
    xyz = jax.random.uniform(
        rng3, (3,), minval=self._min_xyz, maxval=self._max_xyz
    )
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._target_mocap_body_id].set(xyz)
    )

    metrics = {
        "reward/in_target": jp.zeros(()),
        "reward/upright": jp.zeros(()),
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
    upright = data.xmat[self._torso_body_id, 2, 2]
    joint_angles = data.qpos[self._joints_qposadr]
    mouth_to_target_global = (
        data.geom_xpos[self._target_geom_id]
        - data.geom_xpos[self._mouth_geom_id]
    )
    mouth_to_target_local = (
        mouth_to_target_global @ data.geom_xmat[self._mouth_geom_id]
    )
    state= jp.concatenate([
        upright.reshape(1),
        joint_angles,
        mouth_to_target_local,
        data.qvel,
    ])
    privileged_state = jp.concatenate([
      state,
      self.mjx_model.geom_friction[FLOOR_GEOM_ID, 0:1],
      self.mjx_model.dof_frictionloss[3:],
      self.mjx_model.body_ipos[TORSO_BODY_ID],
      self.mjx_model.body_mass[1:],
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
    del action, info  # Unused.

    mouth_to_target_global = (
        data.geom_xpos[self._target_geom_id]
        - data.geom_xpos[self._mouth_geom_id]
    )
    mouth_to_target_local = (
        mouth_to_target_global @ data.geom_xmat[self._mouth_geom_id]
    )

    in_target = reward.tolerance(
        jp.linalg.norm(mouth_to_target_local),
        bounds=(0, self._radii),
        margin=2 * self._radii,
    )
    metrics["reward/in_target"] = in_target

    upright = data.xmat[self._torso_body_id, 2, 2]
    is_upright = 0.5 * (upright + 1)
    metrics["reward/upright"] = is_upright

    return (7 * in_target + is_upright) / 8

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
    return jp.array([1., .11,])
  @property
  def dr_range(self) -> dict:

    low = jp.array(
        [0.9] +                             #floor_friction_min 
        [0.109] 
        #[0.1] 
      )
        # [0.] * (self.mjx_model.nv - 3) +   # dof_friction_min
        # [-0.3] * 3 +                          #com_offset_min
        # [0.1] * (self.mjx_model.nbody - 1)) #body_mass_min
    high = jp.array(
        [1.1] +                             #floor_friction_max
        [.11]
        # [3.0] 
      )
        # [1.] * (self.mjx_model.nv - 3) +   #dof_friction_max
        # [0.3] * 3 +                          #com_offset_max
        # [15.0] * (self.mjx_model.nbody - 1)) #body_mass_max
    return low, high


def domain_randomize(model: mjx.Model, dr_range, params=None, rng:jax.Array=None):
  if rng is not None:
    dr_low, dr_high = dr_range
    dist = functools.partial(jax.random.uniform, shape=(len(dr_low)), minval=dr_low, maxval=dr_high)

  @jax.vmap
  def shift_dynamics(params):
    idx = 0
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(params[idx])
    idx += 1
    body_mass = model.body_mass.at[TORSO_BODY_ID].set(params[idx])
    idx+=1
    # body_mass = model.body_mass.at[BTHIGH_BODY_ID].set(params[idx])
    # idx+=1
    # geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(params[idx])
    # idx+=1
    # dof_frictionloss = model.dof_frictionloss.at[3:].set(params[idx:idx+ model.nv-3])
    # idx += model.nv-3
    # offset = jp.array([params[idx], params[idx+1], params[idx+2]])
    # idx += 3
    # body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
    #     model.body_ipos[TORSO_BODY_ID] + offset
    #   )
    # body_mass = jp.ones((model.nbody,))
    # body_mass =model.body_mass.at[1:].set(model.body_mass[1:] * params[idx:idx + model.nbody-1])
    # for i in range(1, model.nbody):
    #   body_mass = model.body_mass.at[i].set(model.body_mass[i] * params[idx])
    #   idx+=1
    # idx += model.nbody-1
    assert idx == len(params)
    return (
      geom_friction,
      # body_ipos,
      body_mass,
      # dof_frictionloss,
    )
  @jax.vmap
  def rand_dynamics(rng):
    # floor friction
    rng_params = dist(rng)
    idx = 0
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(rng_params[idx])
    idx += 1
    body_mass = model.body_mass.at[TORSO_BODY_ID].set(rng_params[idx])
    idx+=1
    # body_mass = model.body_mass.at[BTHIGH_BODY_ID].set(rng_params[idx])
    # idx+=1
    # idx = 0
    # geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(rng_params[idx])
    # idx += 1
    # dof_frictionloss = model.dof_frictionloss.at[3:].set(model.dof_frictionloss[3:] *rng_params[idx:idx+ model.nv-3])
    # idx += model.nv-3
    # offset = jp.array([rng_params[idx], rng_params[idx+1], rng_params[idx+2]])
    # idx += 3
    # body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
    #     model.body_ipos[TORSO_BODY_ID] + offset
    #   )
    # body_mass = jp.ones((model.nbody,))
    # body_mass =model.body_mass.at[1:].set(model.body_mass[1:] * rng_params[idx:idx + model.nbody-1])
    # # for i in range(1, model.nbody):
    # #   body_mass = model.body_mass.at[i].set(model.body_mass[i] * params[idx])
    # #   idx+=1
    # idx += model.nbody-1
    assert idx == len(rng_params)
    return (
      geom_friction,
      # body_ipos,
      body_mass,
      # dof_frictionloss,
    )
  
  if rng is None and params is not None:

    (geom_friction, 
    #  body_ipos, 
     body_mass, 
    #  dof_frictionloss
    )= shift_dynamics(params)
  elif rng is not None and params is None:
    # params = jax.random.uniform(key=jax.random.PRNGKey(0), shape=(rng.shape[0], len(dr_low)), minval=dr_low, maxval=dr_high)
    # rng = jax.random.split(jax.random.PRNGKey(0), rng.shape[0])
    # print("params", params)

    (
      geom_friction,
      # body_ipos,
      body_mass,
      # dof_frictionloss,
    ) = rand_dynamics(rng)
  else:
    raise ValueError("rng and params wrong!")
  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      # "body_ipos": 0,
      "body_mass": 0,
      # "dof_frictionloss": 0,
  })
  model = model.tree_replace({
      "geom_friction": geom_friction,
      # "body_ipos": body_ipos,
      "body_mass": body_mass,
      # "dof_frictionloss": dof_frictionloss,
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
    body_mass = model.body_mass.at[TORSO_BODY_ID].set(params[idx])
    idx+=1
    # body_mass = model.body_mass.at[BTHIGH_BODY_ID].set(params[idx])
    # idx+=1
    # idx = 0
    # geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(params[idx])
    # idx += 1
    # dof_frictionloss = model.dof_frictionloss.at[3:].set(params[idx:idx+ model.nv-3])
    # idx += model.nv-3
    # offset = jp.array([params[idx], params[idx+1], params[idx+2]])
    # idx += 3
    # body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
    #     model.body_ipos[TORSO_BODY_ID] + offset
    #   )
    # body_mass = jp.ones((model.nbody,))
    # body_mass =body_mass.at[0].set(model.body_mass[0])
    # for i in range(1, model.nbody):
    #   body_mass = body_mass.at[i].set(model.body_mass[i] * params[idx])
    #   idx+=1
    assert idx == len(params)
    return (
      geom_friction,
      # body_ipos,
      body_mass,
      # dof_frictionloss,
    )
  def rand_dynamics(rng):
    # floor friction
    rng_params = dist(rng)
    idx = 0
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(rng_params[idx])
    idx += 1
    body_mass = model.body_mass.at[TORSO_BODY_ID].set(rng_params[idx])
    idx+=1
    # body_mass = model.body_mass.at[BTHIGH_BODY_ID].set(rng_params[idx])
    # idx+=1
    # idx=0
    # geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
    #   rng_params[idx]
    # )
    # idx+=1
    # # static friction
    # dof_frictionloss = jp.zeros((model.nv-3,))
    # for i in range(model.nv-3):
    #   dof_frictionloss = model.dof_frictionloss.at[3+i].set(rng_params[idx])
    #   idx+=1
    # # com pos offset
    # dpos = jp.zeros((3,))
    # for i in range(3):
    #   dpos = dpos.at[idx].set(rng_params[idx])
    #   idx+=1
    # body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
    #     model.body_ipos[TORSO_BODY_ID] + dpos
    # )
    # # link mass 
    # body_mass = jp.ones((model.nbody,))
    # body_mass =body_mass.at[0].set(model.body_mass[0])
    # for i in range(1, model.nbody):
    #   body_mass = body_mass.at[i].set(model.body_mass[i] * rng_params[idx])
    #   idx+=1
    assert idx == len(dr_low)
    return (
      geom_friction,
      # body_ipos,
      body_mass,
      # dof_frictionloss,
    )
  
  if rng is None and params is not None:

    (geom_friction, 
     #body_ipos, 
     body_mass, 
     #dof_frictionloss
     ) = shift_dynamics(params)
  elif rng is not None and params is None:
    # params = jax.random.uniform(key=jax.random.PRNGKey(0), shape=(rng.shape[0], len(dr_low)), minval=dr_low, maxval=dr_high)
    # rng = jax.random.split(jax.random.PRNGKey(0), rng.shape[0])
    # print("params", params)
    (
      geom_friction,
      # body_ipos,
      body_mass,
      # dof_frictionloss,
    ) = rand_dynamics(rng)
  else:
    raise ValueError("rng and params wrong!")
  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      # "body_ipos": 0,
      "body_mass": 0,
      # "dof_frictionloss": 0,
  })
  model = model.tree_replace({
      "geom_friction": geom_friction,
      # "body_ipos": body_ipos,
      "body_mass": body_mass,
      # "dof_frictionloss": dof_frictionloss,
  })

  return model, in_axes