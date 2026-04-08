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
"""Quadruped environment."""

import functools
from itertools import product
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from custom_envs import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "quadruped.xml"
_REF_MJ_MODEL = mujoco.MjModel.from_xml_string(
    _XML_PATH.read_text(), common.get_assets()
)

WALK_SPEED = 0.5
RUN_SPEED = 2.0
_STAND_HEIGHT = 0.55

_FLOOR_GEOM_ID = 0
_TORSO_BODY_ID = 1
_BODY_PARAM_DIM = _REF_MJ_MODEL.nbody - 1


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=1000,
      action_repeat=1,
      action_scale=0.5,
      vision=False,
      impl="jax",
      nconmax=150_000,
      njmax=250,
      history_len=1,
  )


_RESET_JOINT_POS_NOISE = 0.05
_RESET_JOINT_VEL_NOISE = 0.1
_DEFAULT_JOINT_POSE = jp.array(
    [
        0.0,
        0.55,
        -0.95,
        0.45,
        0.0,
        0.55,
        -0.95,
        0.45,
        0.0,
        0.55,
        -0.95,
        0.45,
        0.0,
        0.55,
        -0.95,
        0.45,
    ]
)


class Quadruped(mjx_env.MjxEnv):
  """Quadruped locomotion task from the DM Control style suite."""

  def __init__(
      self,
      desired_speed: float,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    if self._config.vision:
      raise NotImplementedError(
          f"Vision not implemented for {self.__class__.__name__}."
      )

    self._desired_speed = desired_speed
    self._xml_path = _XML_PATH.as_posix()
    self._model_assets = common.get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        _XML_PATH.read_text(), self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._torso_body_id = self.mj_model.body("torso").id
    self._default_pose = _DEFAULT_JOINT_POSE
    self._joint_lowers = jp.array(self._mj_model.jnt_range[1:, 0])
    self._joint_uppers = jp.array(self._mj_model.jnt_range[1:, 1])
    self._joint_obs_size = self.mjx_model.nq - 7
    self._joint_vel_size = self.mjx_model.nv - 6
    self._force_torque_names = [
        f"{kind}_toe_{pos}_{side}"
        for kind, pos, side in product(
            ("force", "torque"), ("front", "back"), ("left", "right")
        )
    ]
    self._toe_pos_sensor_names = [
        f"toe_{pos}_{side}_pos"
        for pos, side in product(("front", "back"), ("left", "right"))
    ]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, qpos_rng, qvel_rng = jax.random.split(rng, 3)

    qpos = jp.array(self.mj_model.qpos0)
    qpos = qpos.at[7:].set(self._default_pose)
    joint_noise = jax.random.uniform(
        qpos_rng,
        (self.mjx_model.nq - 7,),
        minval=-_RESET_JOINT_POS_NOISE,
        maxval=_RESET_JOINT_POS_NOISE,
    )
    qpos = qpos.at[7:].add(joint_noise)
    qpos = qpos.at[7:].set(jp.clip(qpos[7:], self._joint_lowers, self._joint_uppers))

    qvel = jax.random.uniform(
        qvel_rng,
        (self.mjx_model.nv,),
        minval=-_RESET_JOINT_VEL_NOISE,
        maxval=_RESET_JOINT_VEL_NOISE,
    )
    qvel = qvel.at[:6].set(0.0)
    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)
    # Keep the crouched default pose, but avoid backend-specific contact-count
    # fields during reset so this works across MJX implementations.

    metrics = {
        "reward/standing": jp.zeros(()),
        "reward/upright": jp.zeros(()),
        "reward/stand": jp.zeros(()),
        "reward/small_control": jp.zeros(()),
        "reward/move": jp.zeros(()),
    }
    info = {
        "rng": rng,
        "qpos_history": jp.tile(qpos[7:], self._config.history_len),
        "qvel_history": jp.tile(qvel[6:], self._config.history_len),
    }

    reward_value, done = jp.zeros(2)
    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, reward_value, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    lower = self._mj_model.actuator_ctrlrange[:, 0]
    upper = self._mj_model.actuator_ctrlrange[:, 1]
    action = self._default_pose + action * self._config.action_scale
    action = jp.clip(action, lower, upper)
    data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
    reward_value = self._get_reward(data, action, state.info, state.metrics)
    obs = self._get_obs(data, state.info)
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    return mjx_env.State(
        data, obs, reward_value, done, state.metrics, state.info
    )

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
    qpos_history = jp.roll(info["qpos_history"], self._joint_obs_size).at[
        : self._joint_obs_size
    ].set(data.qpos[7:])
    qvel_history = jp.roll(info["qvel_history"], self._joint_vel_size).at[
        : self._joint_vel_size
    ].set(data.qvel[6:])
    info["qpos_history"] = qpos_history
    info["qvel_history"] = qvel_history
    toe_pos = self._toe_positions(data)
    state = jp.concatenate(
        [
            qpos_history,
            qvel_history,
            toe_pos,
            self._torso_velocity(data),
            self._torso_upright(data).reshape(1),
            self._imu(data),
            self._force_torque(data),
        ]
    )
    privileged_state = jp.concatenate(
        [
            state,
            self.mjx_model.geom_friction[_FLOOR_GEOM_ID, 0:1],
            self.mjx_model.body_mass[1:],
            self.mjx_model.body_ipos[1:, 0],
            data.qpos[7:],
            data.qvel[6:],
        ]
    )
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
    del info
    torso_height = data.xpos[self._torso_body_id, -1]
    standing = reward.tolerance(
        torso_height,
        bounds=(_STAND_HEIGHT, float("inf")),
        margin=_STAND_HEIGHT / 2,
    )
    metrics["reward/standing"] = standing

    upright = reward.tolerance(
        self._torso_upright(data),
        bounds=(0.8, float("inf")),
        sigmoid="linear",
        margin=1.8,
        value_at_margin=0.0,
    )
    metrics["reward/upright"] = upright

    stand_reward = standing * upright
    metrics["reward/stand"] = stand_reward

    small_control = reward.tolerance(
        action, margin=1.0, value_at_margin=0.0, sigmoid="quadratic"
    ).mean()
    small_control = (4.0 + small_control) / 5.0
    metrics["reward/small_control"] = small_control

    move_reward = reward.tolerance(
        self._torso_velocity(data)[0],
        bounds=(self._desired_speed, float("inf")),
        sigmoid="linear",
        margin=self._desired_speed,
        value_at_margin=0.0,
    )
    metrics["reward/move"] = move_reward
    return stand_reward * move_reward * small_control

  def _egocentric_state(self, data: mjx.Data) -> jax.Array:
    return jp.concatenate([data.qpos[7:], data.qvel[6:], data.act])

  def _torso_upright(self, data: mjx.Data) -> jax.Array:
    return data.xmat[self._torso_body_id, 2, 2]

  def _torso_velocity(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "velocimeter")

  def _imu(self, data: mjx.Data) -> jax.Array:
    gyro = mjx_env.get_sensor_data(self.mj_model, data, "imu_gyro")
    accel = mjx_env.get_sensor_data(self.mj_model, data, "imu_accel")
    return jp.concatenate([gyro, accel])

  def _force_torque(self, data: mjx.Data) -> jax.Array:
    return jp.concatenate(
        [
            mjx_env.get_sensor_data(self.mj_model, data, name)
            for name in self._force_torque_names
        ]
    )

  def _toe_positions(self, data: mjx.Data) -> jax.Array:
    return jp.concatenate(
        [
            mjx_env.get_sensor_data(self.mj_model, data, name)
            for name in self._toe_pos_sensor_names
        ]
    )

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
    return jp.concatenate([
        self.mjx_model.geom_friction[_FLOOR_GEOM_ID, 0:1],
        self.mjx_model.body_mass[1:],
        jp.zeros(_BODY_PARAM_DIM),
    ])

  @property
  def dr_range(self) -> tuple[jp.ndarray, jp.ndarray]:
    low = jp.concatenate([
        jp.array([0.5]),
        self.mjx_model.body_mass[1:] * 0.8,
        jp.full((_BODY_PARAM_DIM,), -5e-2),
    ])
    high = jp.concatenate([
        jp.array([10.]),
        self.mjx_model.body_mass[1:] * 2.,
        jp.full((_BODY_PARAM_DIM,), 5e-2),
    ])
    return low, high

  @property
  def dr_label(self) -> tuple[str, ...]:
    return (
        "floor friction scale",
        *(f"body mass {i}" for i in range(_BODY_PARAM_DIM)),
        *(f"body ipos x offset {i}" for i in range(_BODY_PARAM_DIM)),
    )


def _apply_domain_params(model: mjx.Model, params: jax.Array):
  idx = 0
  floor_friction = params[idx]
  idx += 1
  body_mass_params = params[idx : idx + _BODY_PARAM_DIM]
  idx += _BODY_PARAM_DIM
  body_ipos_x_params = params[idx : idx + _BODY_PARAM_DIM]
  idx += _BODY_PARAM_DIM
  assert idx == params.shape[0]

  geom_friction = model.geom_friction.at[_FLOOR_GEOM_ID, 0].set(floor_friction)
  geom_friction = jp.clip(geom_friction, a_min=1e-3)

  body_mass = model.body_mass.at[1:].set(body_mass_params)
  body_ipos = model.body_ipos.at[1:, 0].set(
      model.body_ipos[1:, 0] + body_ipos_x_params
  )

  return geom_friction, body_mass, body_ipos


def domain_randomize(model: mjx.Model, dr_range, params=None, rng: jax.Array = None):
  if rng is not None:
    dr_low, dr_high = dr_range
    dist = functools.partial(
        jax.random.uniform,
        shape=(len(dr_low),),
        minval=dr_low,
        maxval=dr_high,
    )

  @jax.vmap
  def shift_dynamics(param_vec):
    return _apply_domain_params(model, param_vec)

  @jax.vmap
  def rand_dynamics(rng_i):
    return _apply_domain_params(model, dist(rng_i))

  if rng is None and params is not None:
    (
        geom_friction,
        body_mass,
        body_ipos,
    ) = shift_dynamics(params)
  elif rng is not None and params is None:
    (
        geom_friction,
        body_mass,
        body_ipos,
    ) = rand_dynamics(rng)
  else:
    raise ValueError("rng and params wrong!")

  in_axes = jax.tree_util.tree_map(lambda _: None, model)
  in_axes = in_axes.tree_replace(
      {
          "geom_friction": 0,
          "body_mass": 0,
          "body_ipos": 0,
      }
  )
  model = model.tree_replace(
      {
          "geom_friction": geom_friction,
          "body_mass": body_mass,
          "body_ipos": body_ipos,
      }
  )
  return model, in_axes


def domain_randomize_eval(
    model: mjx.Model, dr_range, params=None, rng: jax.Array = None
):
  if rng is not None:
    dr_low, dr_high = dr_range
    dist = functools.partial(
        jax.random.uniform,
        shape=(len(dr_low),),
        minval=dr_low,
        maxval=dr_high,
    )

  def shift_dynamics(param_vec):
    return _apply_domain_params(model, param_vec)

  def rand_dynamics(rng_i):
    return _apply_domain_params(model, dist(rng_i))

  if rng is None and params is not None:
    (
        geom_friction,
        body_mass,
        body_ipos,
    ) = shift_dynamics(params)
  elif rng is not None and params is None:
    (
        geom_friction,
        body_mass,
        body_ipos,
    ) = rand_dynamics(rng)
  else:
    raise ValueError("rng and params wrong!")

  in_axes = jax.tree_util.tree_map(lambda _: None, model)
  model = model.tree_replace(
      {
          "geom_friction": geom_friction,
          "body_mass": body_mass,
          "body_ipos": body_ipos,
      }
  )
  return model, in_axes
