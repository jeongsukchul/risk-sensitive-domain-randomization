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

_XML_PATH = mjx_env.ROOT_PATH.parent / "quadruped" / "quadruped.xml"

WALK_SPEED = 0.5
RUN_SPEED = 5.0

_FLOOR_GEOM_ID = 0
_TORSO_BODY_ID = 1
_TOE_GEOM_IDS = jp.array([7, 11, 15, 19])
_JOINT_DOF_IDS = jp.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
_FRICTION_DIM = 3
_DAMPING_DIM = 16

_ACTUATOR_NAME_TO_ID = {
    "yaw_front_left": 0,
    "lift_front_left": 1,
    "extend_front_left": 2,
    "yaw_front_right": 3,
    "lift_front_right": 4,
    "extend_front_right": 5,
    "yaw_back_right": 6,
    "lift_back_right": 7,
    "extend_back_right": 8,
    "yaw_back_left": 9,
    "lift_back_left": 10,
    "extend_back_left": 11,
}


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=1000,
      action_repeat=1,
      vision=False,
      impl="jax",
      nconmax=150_000,
      njmax=250,
  )


def _quat_from_yaw(yaw: jax.Array) -> jax.Array:
  half = 0.5 * yaw
  return jp.array([jp.cos(half), 0.0, 0.0, jp.sin(half)])


def _find_non_contacting_height(
    mjx_model: mjx.Model,
    data: mjx.Data,
    orientation: jax.Array,
    x_pos: float = 0.0,
    y_pos: float = 0.0,
) -> mjx.Data:
  def body_fn(state):
    z_pos, num_contacts, num_attempts, current_data = state
    qpos = current_data.qpos.at[:3].set(jp.array([x_pos, y_pos, z_pos]))
    qpos = qpos.at[3:7].set(orientation)
    next_data = current_data.replace(qpos=qpos)
    next_data = mjx.forward(mjx_model, next_data)
    return (z_pos + 0.01, next_data.ncon, num_attempts + 1, next_data)

  initial_state = (0.0, 1, 0, data)
  *_, num_attempts, next_data = jax.lax.while_loop(
      lambda state: (state[1] > 0) & (state[2] <= 10_000),
      body_fn,
      initial_state,
  )
  return jax.tree_util.tree_map(
      lambda new, old: jp.where(num_attempts < 10_000, new, old), next_data, data
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
    self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._torso_body_id = self.mj_model.body("torso").id
    self._force_torque_names = [
        f"{kind}_toe_{pos}_{side}"
        for kind, pos, side in product(
            ("force", "torque"), ("front", "back"), ("left", "right")
        )
    ]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, yaw_rng = jax.random.split(rng)

    qpos = jp.array(self.mj_model.qpos0)
    # qpos = qpos.at[3:7].set(
    #     _quat_from_yaw(
    #         jax.random.uniform(yaw_rng, (), minval=-jp.pi, maxval=jp.pi)
    #     )
    # )
    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)
    # data = _find_non_contacting_height(self.mjx_model, data, qpos[3:7])

    metrics = {
        "reward/upright": jp.zeros(()),
        "reward/move": jp.zeros(()),
    }
    info = {"rng": rng}

    reward_value, done = jp.zeros(2)
    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, reward_value, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    lower = self._mj_model.actuator_ctrlrange[:, 0]
    upper = self._mj_model.actuator_ctrlrange[:, 1]
    action = (action + 1.0) / 2.0 * (upper - lower) + lower
    data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
    reward_value = self._get_reward(data, action, state.info, state.metrics)
    obs = self._get_obs(data, state.info)
    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    return mjx_env.State(
        data, obs, reward_value, done, state.metrics, state.info
    )

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
    del info
    state = jp.concatenate(
        [
            self._egocentric_state(data),
            self._torso_velocity(data),
            self._torso_upright(data).reshape(1),
            self._imu(data),
            self._force_torque(data),
        ]
    )
    privileged_state = jp.concatenate(
        [
            state,
            jp.array(
                [
                    self.mjx_model.geom_friction[_FLOOR_GEOM_ID, 0],
                    self.mjx_model.body_mass[_TORSO_BODY_ID],
                    jp.mean(self.mjx_model.dof_damping[_JOINT_DOF_IDS]),
                    self.mjx_model.actuator_gear[_ACTUATOR_NAME_TO_ID["lift_front_left"], 0],
                    self.mjx_model.actuator_gear[_ACTUATOR_NAME_TO_ID["yaw_front_left"], 0],
                    self.mjx_model.actuator_gear[_ACTUATOR_NAME_TO_ID["extend_front_left"], 0],
                ]
            ),
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
    del info, action
    move_reward = reward.tolerance(
        self._torso_velocity(data)[0],
        bounds=(self._desired_speed, float("inf")),
        sigmoid="linear",
        margin=self._desired_speed,
        value_at_margin=0.5,
    )
    upright_reward = reward.tolerance(
        self._torso_upright(data),
        bounds=(1.0, float("inf")),
        sigmoid="linear",
        margin=2.0,
        value_at_margin=0.0,
    )
    metrics["reward/move"] = move_reward
    metrics["reward/upright"] = upright_reward
    return move_reward * upright_reward

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
    return jp.ones(1 + _FRICTION_DIM + _DAMPING_DIM + 3)

  @property
  def dr_range(self) -> tuple[jp.ndarray, jp.ndarray]:
    low = jp.concatenate([
        jp.array([0.3]),
        jp.full((_FRICTION_DIM,), 0.5),
        jp.full((_DAMPING_DIM,), 0.9),
        jp.full((3,), 0.9),
    ])
    high = jp.concatenate([
        jp.array([10.]),
        jp.full((_FRICTION_DIM,), 3.),
        jp.full((_DAMPING_DIM,), 1.1),
        jp.full((3,), 1.1),
    ])
    return low, high

  @property
  def dr_label(self) -> tuple[str, ...]:
    return (
        "torso mass scale",
        "friction slide scale",
        "friction torsion scale",
        "friction rolling scale",
        *(f"joint damping scale {i}" for i in range(_DAMPING_DIM)),
        "lift gear scale",
        "yaw gear scale",
        "extend gear scale",
    )


def _apply_domain_params(model: mjx.Model, params: jax.Array):
  idx = 0
  torso_scale = params[idx]
  idx += 1
  friction_scale = params[idx : idx + _FRICTION_DIM]
  idx += _FRICTION_DIM
  damping_scale = params[idx : idx + _DAMPING_DIM]
  idx += _DAMPING_DIM
  gear_lift, gear_yaw, gear_extend = params[idx : idx + 3]
  idx += 3
  assert idx == params.shape[0]

  geom_friction = model.geom_friction
  geom_friction = geom_friction.at[_FLOOR_GEOM_ID].multiply(friction_scale)
  geom_friction = geom_friction.at[_TOE_GEOM_IDS].multiply(friction_scale)
  geom_friction = jp.clip(geom_friction, a_min=1e-3)

  body_mass = model.body_mass.at[_TORSO_BODY_ID].multiply(torso_scale)
  body_inertia = model.body_inertia.at[_TORSO_BODY_ID].multiply(torso_scale**3)

  dof_damping = model.dof_damping
  dof_damping = dof_damping.at[_JOINT_DOF_IDS].multiply(damping_scale)
  dof_damping = jp.clip(dof_damping, a_min=1e-4)

  actuator_gear = model.actuator_gear
  for name, actuator_id in _ACTUATOR_NAME_TO_ID.items():
    if "yaw" in name:
      scale = gear_yaw
    elif "lift" in name:
      scale = gear_lift
    else:
      scale = gear_extend
    actuator_gear = actuator_gear.at[actuator_id, 0].multiply(scale)

  return geom_friction, body_mass, body_inertia, dof_damping, actuator_gear


def domain_randomize(model: mjx.Model, dr_range, params=None, rng: jax.Array = None):
  if rng is not None:
    dr_low, dr_high = dr_range
    dist = functools.partial(
        jax.random.uniform,
        shape=(dr_low.shape[0],),
        minval=dr_low,
        maxval=dr_high,
    )

    @jax.vmap
    def rand_dynamics(rng_i):
      return _apply_domain_params(model, dist(rng_i))

    (geom_friction, body_mass, body_inertia, dof_damping, actuator_gear) = rand_dynamics(rng)
  elif params is not None:
    geom_friction, body_mass, body_inertia, dof_damping, actuator_gear = _apply_domain_params(model, params)
  else:
    raise ValueError("Provide exactly one of `params` or `rng`.")

  in_axes = jax.tree_util.tree_map(lambda _: None, model)
  in_axes = in_axes.tree_replace(
      {
          "geom_friction": 0,
          "body_mass": 0,
          "body_inertia": 0,
          "dof_damping": 0,
          "actuator_gear": 0,
      }
  )
  model = model.tree_replace(
      {
          "geom_friction": geom_friction,
          "body_mass": body_mass,
          "body_inertia": body_inertia,
          "dof_damping": dof_damping,
          "actuator_gear": actuator_gear,
      }
  )
  return model, in_axes


def domain_randomize_eval(
    model: mjx.Model, dr_range, params=None, rng: jax.Array = None
):
  if rng is not None:
    return domain_randomize(model, dr_range, params=None, rng=rng)
  if params is None:
    raise ValueError("Provide `params` or `rng`.")

  geom_friction, body_mass, body_inertia, dof_damping, actuator_gear = _apply_domain_params(model, params)
  in_axes = jax.tree_util.tree_map(lambda _: None, model)
  model = model.tree_replace(
      {
          "geom_friction": geom_friction,
          "body_mass": body_mass,
          "body_inertia": body_inertia,
          "dof_damping": dof_damping,
          "actuator_gear": actuator_gear,
      }
  )
  return model, in_axes
