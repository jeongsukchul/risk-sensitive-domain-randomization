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
"""Stack a red cube onto a green cube with the Franka Panda."""

import functools
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math

from custom_envs import mjx_env
from learning.dr_config import get_structural_dr_bounds
from custom_envs.manipulation.franka_emika_panda import panda
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member


_MODEL_PARAM_SIZE = 30
_RESET_PARAM_SIZE = 6


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=300,
      action_repeat=1,
      action_scale=0.04,
      dynamics_randomization_in_domain_randomization=True,
      reset_randomization_in_domain_randomization=True,
      reward_config=config_dict.create(
          scales=config_dict.create(
              gripper_box=2.0,
              box_goal=4.0,
              release=1.0,
              static=1.0,
              no_floor_collision=0.1,
              robot_target_qpos=0.1,
              success=2.0,
          )
      ),
      impl="jax",
      nconmax=24 * 4096,
      njmax=128,
  )


class PandaStackCube(panda.PandaBase):
  """Stack the red cube on top of the green cube."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "franka_emika_panda"
        / "xmls"
        / "mjx_stack_cube.xml"
    )
    super().__init__(xml_path, config, config_overrides)
    self._post_init(obj_name="box", keyframe="home")

    self._support_body = self._mj_model.body("cubeB").id
    self._support_qposadr = self._mj_model.jnt_qposadr[
        self._mj_model.body("cubeB").jntadr[0]
    ]
    self._box_dofadr = self._mj_model.jnt_dofadr[
        self._mj_model.body("box").jntadr[0]
    ]
    self._support_dofadr = self._mj_model.jnt_dofadr[
        self._mj_model.body("cubeB").jntadr[0]
    ]
    self._cube_half_size = 0.02
    self._release_distance = 0.04
    self._release_open_sum = 0.06
    self._static_lin_thresh = 1e-2
    self._static_ang_thresh = 0.5
    self._support_init_pos = jp.array(
        self._init_q[self._support_qposadr : self._support_qposadr + 3],
        dtype=jp.float32,
    )

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
      rng, rng_anchor, rng_radius, rng_angle, rng_yaw_a, rng_yaw_b = (
          jax.random.split(rng, 6)
      )

      anchor = jax.random.uniform(
          rng_anchor,
          (2,),
          minval=jp.array([-0.02, -0.04]),
          maxval=jp.array([0.02, 0.04]),
      )
      radius = jax.random.uniform(rng_radius, (), minval=0.07, maxval=0.11)
      angle = jax.random.uniform(rng_angle, (), minval=-jp.pi, maxval=jp.pi)
      offset = radius * jp.array([jp.cos(angle), jp.sin(angle)])

      box_xy = self._init_obj_pos[:2] + anchor + 0.5 * offset
      support_xy = self._support_init_pos[:2] + anchor - 0.5 * offset
      box_xy = jp.clip(box_xy, jp.array([0.35, -0.20]), jp.array([0.65, 0.20]))
      support_xy = jp.clip(
          support_xy, jp.array([0.35, -0.20]), jp.array([0.65, 0.20])
      )

      box_pos = jp.array([box_xy[0], box_xy[1], self._cube_half_size])
      support_pos = jp.array([support_xy[0], support_xy[1], self._cube_half_size])

      box_quat = math.axis_angle_to_quat(
          jp.array([0.0, 0.0, 1.0]),
          jax.random.uniform(rng_yaw_a, (), minval=-jp.pi, maxval=jp.pi),
      )
      support_quat = math.axis_angle_to_quat(
          jp.array([0.0, 0.0, 1.0]),
          jax.random.uniform(rng_yaw_b, (), minval=-jp.pi, maxval=jp.pi),
      )
    else:
      reset_params = params[_MODEL_PARAM_SIZE : _MODEL_PARAM_SIZE + _RESET_PARAM_SIZE]
      box_pos = jp.array([
          self._init_obj_pos[0] + reset_params[0],
          self._init_obj_pos[1] + reset_params[1],
          self._cube_half_size,
      ])
      support_pos = jp.array([
          self._support_init_pos[0] + reset_params[2],
          self._support_init_pos[1] + reset_params[3],
          self._cube_half_size,
      ])
      box_quat = math.axis_angle_to_quat(
          jp.array([0.0, 0.0, 1.0]), reset_params[4]
      )
      support_quat = math.axis_angle_to_quat(
          jp.array([0.0, 0.0, 1.0]), reset_params[5]
      )

    init_q = jp.array(self._init_q)
    init_q = init_q.at[self._obj_qposadr : self._obj_qposadr + 3].set(box_pos)
    init_q = init_q.at[self._obj_qposadr + 3 : self._obj_qposadr + 7].set(box_quat)
    init_q = init_q.at[self._support_qposadr : self._support_qposadr + 3].set(
        support_pos
    )
    init_q = init_q.at[
        self._support_qposadr + 3 : self._support_qposadr + 7
    ].set(support_quat)

    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self._mjx_model, data)
    data = self._update_target_marker(data)

    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        "is_box_grasped": jp.array(0.0, dtype=float),
        "is_box_on_support": jp.array(0.0, dtype=float),
        "is_box_static": jp.array(0.0, dtype=float),
        "success": jp.array(0.0, dtype=float),
        **{k: jp.array(0.0, dtype=float) for k in self._config.reward_config.scales},
    }
    info = {"rng": rng}
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return State(data, obs, reward, done, metrics, info)

  @property
  def nominal_params(self):
    model_params = jp.ones(_MODEL_PARAM_SIZE)
    reset_params = jp.zeros(_RESET_PARAM_SIZE)
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
    bounds = get_structural_dr_bounds(
        "PandaStackCube",
        include_reset_params=self._config.reset_randomization_in_domain_randomization,
    )
    if bounds is None:
      raise ValueError("Missing DR YAML config for PandaStackCube.")
    return tuple(jp.asarray(x) for x in bounds)

  def step(self, state: State, action: jax.Array) -> State:
    delta = action * self._action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)

    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
    data = self._update_target_marker(data)

    raw_rewards = self._get_reward_terms(data)
    reward = jp.clip(
        sum(
            raw_rewards[k] * self._config.reward_config.scales[k]
            for k in self._config.reward_config.scales
        ),
        -1e4,
        1e4,
    )

    box_pos = data.xpos[self._obj_body]
    support_pos = data.xpos[self._support_body]
    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= jp.any(jp.abs(support_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    out_of_bounds |= support_pos[2] < 0.0
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)

    metrics = dict(state.metrics)
    metrics.update(
        **raw_rewards,
        out_of_bounds=out_of_bounds.astype(float),
    )

    obs = self._get_obs(data, state.info)
    return State(data, obs, reward, done, metrics, state.info)

  def _get_reward_terms(self, data: mjx.Data) -> Dict[str, jax.Array]:
    box_pos = data.xpos[self._obj_body]
    support_pos = data.xpos[self._support_body]
    gripper_pos = data.site_xpos[self._gripper_site]
    goal_pos = self._goal_pos(support_pos)

    box_to_gripper = jp.linalg.norm(box_pos - gripper_pos)
    box_to_goal = jp.linalg.norm(box_pos - goal_pos)
    gripper_box = 1 - jp.tanh(5 * box_to_gripper)
    box_goal = 1 - jp.tanh(5 * box_to_goal)

    hand_floor_collision = [
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._floor_hand_found_sensor
    ]
    no_floor_collision = (1 - (sum(hand_floor_collision) > 0)).astype(float)

    robot_target_qpos = 1 - jp.tanh(
        jp.linalg.norm(
            data.qpos[self._robot_arm_qposadr]
            - self._init_q[self._robot_arm_qposadr]
        )
    )

    is_box_on_support = self._is_box_on_support(box_pos, support_pos).astype(float)
    is_box_static = self._is_box_static(data).astype(float)
    is_box_grasped = self._is_box_grasped(data, box_pos, gripper_pos).astype(float)
    release = is_box_on_support * (1.0 - is_box_grasped)
    static = is_box_on_support * is_box_static
    success = (is_box_on_support * is_box_static * (1.0 - is_box_grasped)).astype(
        float
    )

    return {
        "gripper_box": gripper_box,
        "box_goal": box_goal * jp.maximum(is_box_grasped, is_box_on_support),
        "release": release,
        "static": static,
        "no_floor_collision": no_floor_collision,
        "robot_target_qpos": robot_target_qpos,
        "success": success,
        "is_box_grasped": is_box_grasped,
        "is_box_on_support": is_box_on_support,
        "is_box_static": is_box_static,
    }

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
    del info
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    box_pos = data.xpos[self._obj_body]
    support_pos = data.xpos[self._support_body]
    goal_pos = self._goal_pos(support_pos)

    state = jp.concatenate([
        data.qpos,
        data.qvel,
        gripper_pos,
        gripper_mat[3:],
        data.xmat[self._obj_body].ravel()[3:],
        data.xmat[self._support_body].ravel()[3:],
        box_pos - gripper_pos,
        support_pos - gripper_pos,
        goal_pos - box_pos,
        support_pos - box_pos,
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])
    privileged_state = jp.concatenate([
        state,
        data.qfrc_bias,
        data.actuator_force,
        self.mjx_model.geom_friction[self._left_finger_geom, 0:1],
        self.mjx_model.body_mass[self._obj_body : self._obj_body + 1],
        self.mjx_model.body_mass[self._support_body : self._support_body + 1],
        self.mjx_model.actuator_gainprm[:, 0],
        self.mjx_model.dof_damping[:9],
        self.mjx_model.dof_armature[:9],
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _goal_pos(self, support_pos: jax.Array) -> jax.Array:
    return support_pos + jp.array([0.0, 0.0, 2 * self._cube_half_size])

  def _update_target_marker(self, data: mjx.Data) -> mjx.Data:
    goal_pos = self._goal_pos(data.xpos[self._support_body])
    return data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(goal_pos),
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(
            jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        ),
    )

  def _is_box_on_support(
      self, box_pos: jax.Array, support_pos: jax.Array
  ) -> jax.Array:
    offset = box_pos - support_pos
    xy_ok = jp.linalg.norm(offset[:2]) <= jp.sqrt(2.0) * self._cube_half_size + 0.005
    z_ok = jp.abs(offset[2] - 2 * self._cube_half_size) <= 0.005
    return xy_ok & z_ok

  def _is_box_static(self, data: mjx.Data) -> jax.Array:
    box_qvel = data.qvel[self._box_dofadr : self._box_dofadr + 6]
    lin_vel = jp.linalg.norm(box_qvel[:3])
    ang_vel = jp.linalg.norm(box_qvel[3:])
    return (lin_vel <= self._static_lin_thresh) & (
        ang_vel <= self._static_ang_thresh
    )

  def _is_box_grasped(
      self, data: mjx.Data, box_pos: jax.Array, gripper_pos: jax.Array
  ) -> jax.Array:
    finger_open_sum = jp.sum(data.qpos[self._robot_qposadr[-2:]])
    gripper_far = jp.linalg.norm(box_pos - gripper_pos) > self._release_distance
    gripper_open = finger_open_sum > self._release_open_sum
    return ~(gripper_far & gripper_open)


def _apply_domain_randomization(model: mjx.Model, params: jax.Array):
  env = PandaStackCube()
  box_body = env._obj_body
  support_body = env._support_body
  arm_qids = jp.arange(7)
  joint_qids = jp.arange(9)
  link_ids = jp.arange(11) + 1
  left_finger_geom = env._left_finger_geom
  right_finger_geom = env._right_finger_geom

  idx = 0
  geom_friction = model.geom_friction.at[left_finger_geom, 0].set(params[idx])
  geom_friction = geom_friction.at[right_finger_geom, 0].set(params[idx])
  idx += 1

  body_mass = model.body_mass.at[box_body].set(model.body_mass[box_body] * params[idx])
  idx += 1
  body_mass = body_mass.at[support_body].set(
      model.body_mass[support_body] * params[idx]
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
      "geom_friction": geom_friction,
      "body_mass": body_mass,
      "dof_armature": dof_armature,
      "dof_damping": dof_damping,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
  })

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_mass": 0,
      "dof_armature": 0,
      "dof_damping": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
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
