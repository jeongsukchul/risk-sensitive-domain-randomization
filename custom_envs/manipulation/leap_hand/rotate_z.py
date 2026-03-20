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
"""Rotate-z with leap hand."""

import functools
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from jax.scipy import special as jsp_special
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from custom_envs import mjx_env
from custom_envs.manipulation.leap_hand import base as leap_hand_base
from custom_envs.manipulation.leap_hand import leap_hand_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.01,
      action_scale=0.6,
      action_repeat=1,
      episode_length=500,
      early_termination=True,
      history_len=1,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              angvel=1.0,
              linvel=0.0,
              pose=0.0,
              torques=0.0,
              energy=0.0,
              termination=-100.0,
              action_rate=0.0,
          ),
      ),
      reset_randomization_in_domain_randomization=True,
      impl='jax',
      naconmax=30 * 8192,
      njmax=128,
  )


_MODEL_PARAM_SIZE = 1 + 1 + 3 + 16 + 16 + 17 + 16 + 16
_RESET_PARAM_SIZE = consts.NQ + 3 + 3
_UNIT_INTERVAL_EPS = 1e-6


def _unit_uniform_to_normal(u: jax.Array) -> jax.Array:
  u = jp.clip(u, _UNIT_INTERVAL_EPS, 1.0 - _UNIT_INTERVAL_EPS)
  return jp.sqrt(2.0) * jsp_special.erfinv(2.0 * u - 1.0)


def _uniform_quat_from_unit_interval(u: jax.Array) -> jax.Array:
  u = jp.clip(u, _UNIT_INTERVAL_EPS, 1.0 - _UNIT_INTERVAL_EPS)
  return jp.array([
      jp.sqrt(1 - u[0]) * jp.sin(2 * jp.pi * u[1]),
      jp.sqrt(1 - u[0]) * jp.cos(2 * jp.pi * u[1]),
      jp.sqrt(u[0]) * jp.sin(2 * jp.pi * u[2]),
      jp.sqrt(u[0]) * jp.cos(2 * jp.pi * u[2]),
  ])


class CubeRotateZAxis(leap_hand_base.LeapHandEnv):
  """Rotate a cube around the z-axis as fast as possible wihout dropping it."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.CUBE_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
    self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
    self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._cube_geom_id = self._mj_model.geom("cube").id

    home_key = self._mj_model.keyframe("home")
    self._init_q = jp.array(home_key.qpos)
    self._default_pose = self._init_q[self._hand_qids]
    self._lowers, self._uppers = self.mj_model.actuator_ctrlrange.T

  def reset(
      self, rng: jax.Array, params: Optional[jax.Array] = None
  ) -> mjx_env.State:
    use_reset_randomization_params = (
        params is not None
        and self._config.reset_randomization_in_domain_randomization
    )

    if not use_reset_randomization_params:
      rng, pos_rng, vel_rng = jax.random.split(rng, 3)
      q_hand = jp.clip(
          self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ,)),
          self._lowers,
          self._uppers,
      )
      v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))

      rng, p_rng, quat_rng = jax.random.split(rng, 3)
      start_pos = jp.array([0.1, 0.0, 0.05]) + jax.random.uniform(
          p_rng, (3,), minval=-0.01, maxval=0.01
      )
      start_quat = leap_hand_base.uniform_quat(quat_rng)
    else:
      reset_params = params[_MODEL_PARAM_SIZE : _MODEL_PARAM_SIZE + _RESET_PARAM_SIZE]
      idx = 0
      q_hand = jp.clip(
          self._default_pose
          + 0.1
          * _unit_uniform_to_normal(reset_params[idx : idx + consts.NQ]),
          self._lowers,
          self._uppers,
      )
      idx += consts.NQ
      v_hand = jp.zeros(consts.NV)

      start_pos = jp.array([0.1, 0.0, 0.05]) + reset_params[idx : idx + 3]
      idx += 3
      start_quat = _uniform_quat_from_unit_interval(reset_params[idx : idx + 3])

    q_cube = jp.array([*start_pos, *start_quat])
    v_cube = jp.zeros(6)

    qpos = jp.concatenate([q_hand, q_cube])
    qvel = jp.concatenate([v_hand, v_cube])
    data = mjx_env.make_data(
        self._mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=q_hand,
        mocap_pos=jp.array([-100.0, -100.0, -100.0]),  # Hide goal for task.
        impl=self._mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": data.ctrl,
        "last_cube_angvel": jp.zeros(3),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs_history = jp.zeros(self._config.history_len * 32)
    obs = self._get_obs(data, info, obs_history)
    reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    obs = self._get_obs(data, state.info, state.obs["state"])
    done = self._get_termination(data)

    rewards = self._get_reward(data, action, state.info, state.metrics, done)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt  # pylint: disable=redefined-outer-name

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["last_cube_angvel"] = self.get_cube_angvel(data)
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    return self.get_cube_position(data)[2] < -0.05

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], obs_history: jax.Array
  ) -> Dict[str, jax.Array]:
    joint_angles = data.qpos[self._hand_qids]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    state = jp.concatenate([
        noisy_joint_angles,
        info["last_act"],
    ])
    obs_history = jp.roll(obs_history, state.size)
    obs_history = obs_history.at[: state.size].set(state)

    cube_pos = self.get_cube_position(data)
    palm_pos = self.get_palm_position(data)
    cube_pos_error = palm_pos - cube_pos
    cube_quat = self.get_cube_orientation(data)
    cube_angvel = self.get_cube_angvel(data)
    cube_linvel = self.get_cube_linvel(data)
    fingertip_positions = self.get_fingertip_positions(data)
    joint_torques = data.actuator_force

    privileged_state = jp.concatenate([
        state,
        joint_angles,
        data.qvel[self._hand_dqids],
        joint_torques,
        fingertip_positions,
        cube_pos_error,
        cube_quat,
        cube_angvel,
        cube_linvel,
    ])

    return {
        "state": obs_history,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics
    cube_pos = self.get_cube_position(data)
    palm_pos = self.get_palm_position(data)
    cube_pos_error = palm_pos - cube_pos
    cube_angvel = self.get_cube_angvel(data)
    cube_linvel = self.get_cube_linvel(data)
    return {
        "angvel": self._reward_angvel(cube_angvel, cube_pos_error),
        "linvel": self._cost_linvel(cube_linvel),
        "termination": done,
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "pose": self._cost_pose(data.qpos[self._hand_qids]),
        "torques": self._cost_torques(data.actuator_force),
        "energy": self._cost_energy(
            data.qvel[self._hand_dqids], data.actuator_force
        ),
    }

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_linvel(self, cube_linvel: jax.Array) -> jax.Array:
    return jp.linalg.norm(cube_linvel, ord=1, axis=-1)

  def _reward_angvel(
      self, cube_angvel: jax.Array, cube_pos_error: jax.Array
  ) -> jax.Array:
    del cube_pos_error
    return cube_angvel @ jp.array([0.0, 0.0, 1.0])

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act
    return jp.sum(jp.square(act - last_act))

  def _cost_pose(self, joint_angles: jax.Array) -> jax.Array:
    return jp.sum(jp.square(joint_angles - self._default_pose))

  @property
  def nominal_params(self):
    model_params = jp.concatenate([
        jp.ones(1),
        jp.zeros(1),
        jp.zeros(3),
        jp.zeros(16),
        jp.ones(16 + 16 + 17 + 16 + 16),
    ])
    if not self._config.reset_randomization_in_domain_randomization:
      return model_params

    reset_params = jp.concatenate([
        jp.full((consts.NQ,), 0.5),
        jp.zeros(3),
        jp.full((3,), 0.5),
    ])
    return jp.concatenate([model_params, reset_params])

  @property
  def reset_param_size(self):
    if not self._config.reset_randomization_in_domain_randomization:
      return 0
    return _RESET_PARAM_SIZE

  @property
  def dr_range(self):
    low = [jp.array([0.5])]
    high = [jp.array([1.0])]

    low.append(jp.array([0.8]))
    high.append(jp.array([1.2]))

    low.append(jp.full((3,), -0.005))
    high.append(jp.full((3,), 0.005))

    low.append(jp.full((16,), -0.05))
    high.append(jp.full((16,), 0.05))

    low.append(jp.full((16,), 0.5))
    high.append(jp.full((16,), 2.0))

    low.append(jp.full((16,), 1.0))
    high.append(jp.full((16,), 1.05))

    low.append(jp.full((17,), 0.9))
    high.append(jp.full((17,), 1.1))

    low.append(jp.full((16,), 0.8))
    high.append(jp.full((16,), 1.2))

    low.append(jp.full((16,), 0.8))
    high.append(jp.full((16,), 1.2))

    if not self._config.reset_randomization_in_domain_randomization:
      return jp.concatenate(low), jp.concatenate(high)

    low.append(jp.full((consts.NQ,), _UNIT_INTERVAL_EPS))
    high.append(jp.full((consts.NQ,), 1.0 - _UNIT_INTERVAL_EPS))

    low.append(jp.full((3,), -0.01))
    high.append(jp.full((3,), 0.01))

    low.append(jp.full((3,), _UNIT_INTERVAL_EPS))
    high.append(jp.full((3,), 1.0 - _UNIT_INTERVAL_EPS))

    return jp.concatenate(low), jp.concatenate(high)


def _apply_domain_randomization(model: mjx.Model, params: jax.Array):
  mj_model = CubeRotateZAxis().mj_model
  cube_body_id = mj_model.body("cube").id
  hand_qids = mjx_env.get_qpos_ids(mj_model, consts.JOINT_NAMES)
  hand_body_names = [
      "palm",
      "if_bs",
      "if_px",
      "if_md",
      "if_ds",
      "mf_bs",
      "mf_px",
      "mf_md",
      "mf_ds",
      "rf_bs",
      "rf_px",
      "rf_md",
      "rf_ds",
      "th_mp",
      "th_bs",
      "th_px",
      "th_ds",
  ]
  hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])
  fingertip_geoms = ["th_tip", "if_tip", "mf_tip", "rf_tip"]
  fingertip_geom_ids = np.array([mj_model.geom(g).id for g in fingertip_geoms])

  idx = 0
  geom_friction = model.geom_friction.at[fingertip_geom_ids, 0].set(params[idx])
  idx += 1

  body_mass = model.body_mass
  body_inertia = model.body_inertia.at[cube_body_id].set(
      model.body_inertia[cube_body_id] * params[idx]
  )
  idx += 1

  body_ipos = model.body_ipos.at[cube_body_id].set(
      model.body_ipos[cube_body_id] + params[idx : idx + 3]
  )
  idx += 3

  qpos0 = model.qpos0.at[hand_qids].set(
      model.qpos0[hand_qids] + params[idx : idx + 16]
  )
  idx += 16

  dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(
      model.dof_frictionloss[hand_qids] * params[idx : idx + 16]
  )
  idx += 16

  dof_armature = model.dof_armature.at[hand_qids].set(
      model.dof_armature[hand_qids] * params[idx : idx + 16]
  )
  idx += 16

  body_mass = body_mass.at[hand_body_ids].set(
      model.body_mass[hand_body_ids] * params[idx : idx + 17]
  )
  idx += 17

  new_kp = model.actuator_gainprm[:, 0] * params[idx : idx + 16]
  actuator_gainprm = model.actuator_gainprm.at[:, 0].set(new_kp)
  actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-new_kp)
  idx += 16

  dof_damping = model.dof_damping.at[hand_qids].set(
      model.dof_damping[hand_qids] * params[idx : idx + 16]
  )

  return (
      geom_friction,
      body_mass,
      body_inertia,
      body_ipos,
      qpos0,
      dof_frictionloss,
      dof_armature,
      dof_damping,
      actuator_gainprm,
      actuator_biasprm,
  )


def _finalize_domain_randomization(
    model: mjx.Model,
    geom_friction,
    body_mass,
    body_inertia,
    body_ipos,
    qpos0,
    dof_frictionloss,
    dof_armature,
    dof_damping,
    actuator_gainprm,
    actuator_biasprm,
):
  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_mass": 0,
      "body_inertia": 0,
      "body_ipos": 0,
      "qpos0": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "dof_damping": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
  })

  model = model.tree_replace({
      "geom_friction": geom_friction,
      "body_mass": body_mass,
      "body_inertia": body_inertia,
      "body_ipos": body_ipos,
      "qpos0": qpos0,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "dof_damping": dof_damping,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
  })

  return model, in_axes


def domain_randomize(model: mjx.Model, dr_range, params=None, rng: jax.Array = None):
  if rng is not None:
    dr_low, dr_high = dr_range
    dr_low = dr_low[:_MODEL_PARAM_SIZE]
    dr_high = dr_high[:_MODEL_PARAM_SIZE]
    dist = functools.partial(
        jax.random.uniform, shape=(len(dr_low),), minval=dr_low, maxval=dr_high
    )

    @jax.vmap
    def rand_dynamics(rng):
      return _apply_domain_randomization(model, dist(rng))

    randomized = rand_dynamics(rng)
  elif params is not None:
    params = params[..., :_MODEL_PARAM_SIZE]
    if params.ndim == 1:
      randomized = _apply_domain_randomization(model, params)
    else:
      randomized = jax.vmap(lambda p: _apply_domain_randomization(model, p))(params)
  else:
    raise ValueError("rng and params wrong!")

  return _finalize_domain_randomization(model, *randomized)


def domain_randomize_eval(
    model: mjx.Model, dr_range, params=None, rng: jax.Array = None
):
  return domain_randomize(model, dr_range, params=params, rng=rng)
