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
"""Bring a box to a target and orientation."""

import functools
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from custom_envs import mjx_env
from learning.dr_config import get_structural_dr_bounds
from mujoco_playground._src.manipulation.franka_emika_panda import panda
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member
import numpy as np


_MODEL_PARAM_SIZE = 29
_RESET_PARAM_SIZE = 2 + 3 + 3 + 1


def default_config() -> config_dict.ConfigDict:
  """Returns the default config for bring_to_target tasks."""
  config = config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=150,
      action_repeat=1,
      action_scale=0.04,
      dynamics_randomization_in_domain_randomization=True,
      reset_randomization_in_domain_randomization=True,
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=4.0 / 10,
              # Box goes to the target mocap.
              box_target=8.0 / 10,
              # Do not collide the gripper with the floor.
              no_floor_collision=0.25 /10,
              # Arm stays close to target pose.
              robot_target_qpos=0.3 / 10,
          )
      ),
      impl='jax',
      naconmax=24 * 2048,
      naccdmax=24 * 2048,
      njmax=128,
  )
  return config


class PandaPickCube(panda.PandaBase):
  """Bring a box to a target."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      sample_orientation: bool = False,
  ):
    xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "franka_emika_panda"
        / "xmls"
        / "mjx_single_cube.xml"
    )
    super().__init__(
        xml_path,
        config,
        config_overrides,
    )
    self._post_init(obj_name="box", keyframe="home")
    self._sample_orientation = sample_orientation

    # Contact sensor IDs.
    self._floor_hand_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
    ]

  def reset(
      self, rng: jax.Array, params: Optional[jax.Array] = None
  ) -> State:
    use_reset_randomization_params = (
        params is not None
        and self._config.reset_randomization_in_domain_randomization
    )

    if not use_reset_randomization_params:
      rng, rng_box, rng_target = jax.random.split(rng, 3)

      box_pos = (
          jax.random.uniform(
              rng_box,
              (3,),
              minval=jp.array([-0.2, -0.2, 0.0]),
              maxval=jp.array([0.2, 0.2, 0.0]),
          )
          + self._init_obj_pos
      )

      target_pos = (
          jax.random.uniform(
              rng_target,
              (3,),
              minval=jp.array([-0.2, -0.2, 0.2]),
              maxval=jp.array([0.2, 0.2, 0.4]),
          )
          + self._init_obj_pos
      )

      target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
      if self._sample_orientation:
        rng, rng_axis, rng_theta = jax.random.split(rng, 3)
        perturb_axis = jax.random.uniform(rng_axis, (3,), minval=-1, maxval=1)
        perturb_axis = perturb_axis / math.norm(perturb_axis)
        perturb_theta = jax.random.uniform(rng_theta, maxval=np.deg2rad(45))
        target_quat = math.axis_angle_to_quat(perturb_axis, perturb_theta)
    else:
      reset_params = params[_MODEL_PARAM_SIZE : _MODEL_PARAM_SIZE + _RESET_PARAM_SIZE]
      idx = 0

      box_pos = jp.array([
          self._init_obj_pos[0] + reset_params[idx],
          self._init_obj_pos[1] + reset_params[idx + 1],
          self._init_obj_pos[2],
      ])
      idx += 2

      target_pos = self._init_obj_pos + reset_params[idx : idx + 3]
      idx += 3

      target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
      if self._sample_orientation:
        perturb_axis = reset_params[idx : idx + 3]
        axis_norm = jp.maximum(math.norm(perturb_axis), 1e-6)
        perturb_axis = perturb_axis / axis_norm
        idx += 3
        perturb_theta = reset_params[idx]
        target_quat = math.axis_angle_to_quat(perturb_axis, perturb_theta)

    # initialize data
    init_q = (
        jp.array(self._init_q)
        .at[self._obj_qposadr : self._obj_qposadr + 3]
        .set(box_pos)
    )
    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
        impl=self._mjx_model.impl.value,
        naconmax=self._config.naconmax,
        naccdmax=self._config.naccdmax,
        njmax=self._config.njmax,
    )

    # set target mocap position
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),
    )

    # initialize env state and info
    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    }
    info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0}
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    state = State(data, obs, reward, done, metrics, info)
    return state

  @property
  def nominal_params(self):
    model_params = jp.ones(_MODEL_PARAM_SIZE)

    reset_params = jp.concatenate([
        jp.zeros(2),
        jp.array([0.0, 0.0, 0.3]),
        jp.array([1.0, 0.0, 0.0]),
        jp.array([0.0]),
    ])
    if not self._config.reset_randomization_in_domain_randomization:
      return model_params
    return jp.concatenate([model_params, reset_params])

  @property
  def reset_param_size(self):
    if not self._config.reset_randomization_in_domain_randomization:
      return 0
    return _RESET_PARAM_SIZE

  @property
  def dr_range(self):
    task_name = (
        "PandaPickCubeOrientation" if self._sample_orientation else "PandaPickCube"
    )
    bounds = get_structural_dr_bounds(
        task_name,
        include_reset_params=self._config.reset_randomization_in_domain_randomization,
    )
    if bounds is None:
      raise ValueError(f"Missing DR YAML config for {task_name}.")
    return tuple(jp.asarray(x) for x in bounds)

  def step(self, state: State, action: jax.Array) -> State:
    delta = action * self._action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)

    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

    raw_rewards = self._get_reward(data, state.info)
    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
    box_pos = data.xpos[self._obj_body]
    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)

    state.metrics.update(
        **raw_rewards, out_of_bounds=out_of_bounds.astype(float)
    )

    obs = self._get_obs(data, state.info)
    state = State(data, obs, reward, done, state.metrics, state.info)

    return state

  def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
    target_pos = info["target_pos"]
    box_pos = data.xpos[self._obj_body]
    gripper_pos = data.site_xpos[self._gripper_site]
    pos_err = jp.linalg.norm(target_pos - box_pos)
    box_mat = data.xmat[self._obj_body]
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    rot_err = jp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])

    box_target = 1 - jp.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))
    gripper_box = 1 - jp.tanh(5 * jp.linalg.norm(box_pos - gripper_pos))
    robot_target_qpos = 1 - jp.tanh(
        jp.linalg.norm(
            data.qpos[self._robot_arm_qposadr]
            - self._init_q[self._robot_arm_qposadr]
        )
    )

    # Check for collisions with the floor
    hand_floor_collision = [
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._floor_hand_found_sensor
    ]
    floor_collision = sum(hand_floor_collision) > 0
    no_floor_collision = (1 - floor_collision).astype(float)

    info["reached_box"] = 1.0 * jp.maximum(
        info["reached_box"],
        (jp.linalg.norm(box_pos - gripper_pos) < 0.012),
    )

    rewards = {
        "gripper_box": gripper_box,
        "box_target": box_target * info["reached_box"],
        "no_floor_collision": no_floor_collision,
        "robot_target_qpos": robot_target_qpos,
    }
    return rewards

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    state = jp.concatenate([
        data.qpos,
        data.qvel,
        gripper_pos,
        gripper_mat[3:],
        data.xmat[self._obj_body].ravel()[3:],
        data.xpos[self._obj_body] - data.site_xpos[self._gripper_site],
        info["target_pos"] - data.xpos[self._obj_body],
        target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])
    privileged_state = jp.concatenate([
        state,
        data.qfrc_bias,
        data.actuator_force,
        self.mjx_model.geom_friction[self._left_finger_geom, 0:1],
        self.mjx_model.body_mass,
        self.mjx_model.actuator_gainprm[:, 0],
        self.mjx_model.dof_damping[:9],
        self.mjx_model.dof_armature[:9],
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }


class PandaPickCubeOrientation(PandaPickCube):
  """Bring a box to a target and orientation."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides, sample_orientation=True)


def _apply_domain_randomization(model: mjx.Model, params: jax.Array):
  env = PandaPickCubeOrientation()
  cube_body = env._obj_body
  arm_qids = jp.arange(7)
  joint_qids = jp.arange(9)
  link_ids = jp.arange(11) + 1
  left_finger_geom = env._left_finger_geom
  right_finger_geom = env._right_finger_geom

  idx = 0
  geom_friction = model.geom_friction.at[left_finger_geom, 0].set(params[idx])
  geom_friction = geom_friction.at[right_finger_geom, 0].set(params[idx])
  idx += 1

  body_mass = model.body_mass.at[cube_body].set(
      model.body_mass[cube_body] * params[idx]
  )
  idx += 1
  body_mass = body_mass.at[link_ids].set(
      model.body_mass[link_ids] * params[idx : idx + 11]
  )
  idx += 11

  dof_armature = model.dof_armature
  dof_damping = model.dof_damping.at[joint_qids].set(
      model.dof_damping[joint_qids] * params[idx : idx + 9]
  )
  idx += 9

  kp_val = model.actuator_gainprm[arm_qids, 0] * params[idx : idx + 7]
  actuator_gainprm = model.actuator_gainprm.at[arm_qids, 0].set(kp_val)
  actuator_biasprm = model.actuator_biasprm.at[arm_qids, 1].set(-kp_val)
  idx += 7

  assert idx == _MODEL_PARAM_SIZE

  return (
      geom_friction,
      body_mass,
      dof_armature,
      dof_damping,
      actuator_gainprm,
      actuator_biasprm,
  )


def _finalize_domain_randomization(
    model: mjx.Model,
    geom_friction,
    body_mass,
    dof_armature,
    dof_damping,
    actuator_gainprm,
    actuator_biasprm,
):
  model = model.tree_replace({
      'geom_friction': geom_friction,
      'body_mass': body_mass,
      'dof_armature': dof_armature,
      'dof_damping': dof_damping,
      'actuator_gainprm': actuator_gainprm,
      'actuator_biasprm': actuator_biasprm,
  })

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      'geom_friction': 0,
      'body_mass': 0,
      'dof_armature': 0,
      'dof_damping': 0,
      'actuator_gainprm': 0,
      'actuator_biasprm': 0,
  })

  return model, in_axes


def domain_randomize(
    model: mjx.Model, dr_range, params: jax.Array = None, rng: jax.Array = None
):
  dr_low, dr_high = dr_range
  model_low = dr_low[:_MODEL_PARAM_SIZE]
  model_high = dr_high[:_MODEL_PARAM_SIZE]

  if rng is not None:
    dist = functools.partial(
        jax.random.uniform,
        shape=(len(model_low),),
        minval=model_low,
        maxval=model_high,
    )

    @jax.vmap
    def rand_dynamics(rng):
      return _apply_domain_randomization(model, dist(rng))

    randomized = rand_dynamics(rng)
  elif params is not None:
    params = params[..., :_MODEL_PARAM_SIZE]

    @jax.vmap
    def shift_dynamics(p):
      return _apply_domain_randomization(model, p)

    randomized = (
        _apply_domain_randomization(model, params)
        if params.ndim == 1
        else shift_dynamics(params)
    )
  else:
    raise ValueError("rng and params wrong!")

  return _finalize_domain_randomization(model, *randomized)


def domain_randomize_eval(
    model: mjx.Model, dr_range, params: jax.Array = None, rng: jax.Array = None
):
  dr_low, dr_high = dr_range
  model_low = dr_low[:_MODEL_PARAM_SIZE]
  model_high = dr_high[:_MODEL_PARAM_SIZE]

  if rng is not None:
    dist = functools.partial(
        jax.random.uniform,
        shape=(len(model_low),),
        minval=model_low,
        maxval=model_high,
    )

    def rand_dynamics(rng):
      return _apply_domain_randomization(model, dist(rng))

    randomized = rand_dynamics(rng)
  elif params is not None:
    params = params[:_MODEL_PARAM_SIZE]
    randomized = _apply_domain_randomization(model, params)
  else:
    raise ValueError("rng and params wrong!")

  return _finalize_domain_randomization(model, *randomized)
