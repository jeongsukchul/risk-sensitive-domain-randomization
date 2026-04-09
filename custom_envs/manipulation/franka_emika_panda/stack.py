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
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from custom_envs import mjx_env
from custom_envs.manipulation.franka_emika_panda import panda
from custom_envs.manipulation.franka_emika_panda import panda_kinematics
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member

_MODEL_PARAM_SIZE = 30
_CUBE_HALF_SIZE = 0.02
_RELEASE_DISTANCE = 0.04
_RELEASE_OPEN_SUM = 0.06
_STATIC_LIN_THRESH = 1e-2
_STATIC_ANG_THRESH = 0.5
_RETREAT_Z_OFFSET = 0.12
_RETREAT_Z_TOL = 0.01
_RESET_RADIUS_RANGE = (0.43, 0.51)
_RESET_STACK_ANGLE = -2.0 * jp.pi / 4.0
_RESET_SUPPORT_ANGLE_RANGE = (
    11.0 * jp.pi / 36.0,
    13.0 * jp.pi / 36.0,
)

_XML_PATH = (
    mjx_env.ROOT_PATH
    / "manipulation"
    / "franka_emika_panda"
    / "xmls"
    / "mjx_stack_cube.xml"
)
_REF_MODEL = mujoco.MjModel.from_xml_string(
    _XML_PATH.read_text(), assets=panda.get_assets()
)
_LEFT_FINGER_GEOM = _REF_MODEL.geom("left_finger_pad").id
_RIGHT_FINGER_GEOM = _REF_MODEL.geom("right_finger_pad").id
_BOX_BODY = _REF_MODEL.body("box").id
_SUPPORT_BODY = _REF_MODEL.body("cubeB").id
_LINK_IDS = np.arange(11) + 1
_ARM_QIDS = np.arange(7)
_JOINT_QIDS = np.arange(9)


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=300,
      action_repeat=1,
      action_scale=0.04,
      reward_config=config_dict.create(
          scales=config_dict.create(
              gripper_box=2.0/5,
              box_goal=4.0/5,
              release=1.0/5,
              static=1.0/5,
              no_floor_collision=0.5/5,
              robot_target_qpos=0.1/5,
              success=2.0/5,
          )
      ),
      impl="jax",
      nconmax=48 * 2048,
      njmax=192,
  )


class PandaStackCube(panda.PandaBase):
  """Stack the red cube on top of the green cube."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(_XML_PATH, config, config_overrides)
    self._post_init(obj_name="box", keyframe="home")

    self._support_body = self._mj_model.body("cubeB").id
    self._robot_base_body = self._mj_model.body("link0").id
    self._support_qposadr = self._mj_model.jnt_qposadr[
        self._mj_model.body("cubeB").jntadr[0]
    ]
    self._box_dofadr = self._mj_model.jnt_dofadr[
        self._mj_model.body("box").jntadr[0]
    ]
    self._support_init_pos = jp.array(
        self._init_q[self._support_qposadr : self._support_qposadr + 3],
        dtype=jp.float32,
    )
    base_quat = jp.array(
        self._mj_model.body_quat[self._robot_base_body], dtype=jp.float32
    )
    self._robot_base_yaw = jp.arctan2(
        2.0 * (base_quat[0] * base_quat[3] + base_quat[1] * base_quat[2]),
        1.0 - 2.0 * (base_quat[2] ** 2 + base_quat[3] ** 2),
    )
    self._floor_hand_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
    ]

  def reset(self, rng: jax.Array) -> State:
    rng, rng_radius, rng_support_angle, rng_yaw_a, rng_yaw_b = jax.random.split(
        rng, 5
    )

    radius = jax.random.uniform(
        rng_radius,
        (),
        minval=_RESET_RADIUS_RANGE[0],
        maxval=_RESET_RADIUS_RANGE[1],
    )
    support_angle = self._robot_base_yaw + jax.random.uniform(
        rng_support_angle,
        (),
        minval=_RESET_SUPPORT_ANGLE_RANGE[0],
        maxval=_RESET_SUPPORT_ANGLE_RANGE[1],
    )
    box_angle = support_angle + _RESET_STACK_ANGLE

    goal_xy = radius * jp.array([jp.cos(support_angle), jp.sin(support_angle)])
    base_xy = radius * jp.array([jp.cos(box_angle), jp.sin(box_angle)])

    box_pos = jp.array([goal_xy[0], goal_xy[1], _CUBE_HALF_SIZE])
    support_pos = jp.array([base_xy[0], base_xy[1], _CUBE_HALF_SIZE])
    box_quat = math.axis_angle_to_quat(
        jp.array([0.0, 0.0, 1.0]),
        jax.random.uniform(rng_yaw_a, (), minval=-jp.pi, maxval=jp.pi),
    )
    support_quat = math.axis_angle_to_quat(
        jp.array([0.0, 0.0, 1.0]),
        jax.random.uniform(rng_yaw_b, (), minval=-jp.pi, maxval=jp.pi),
    )

    init_q = jp.array(self._init_q)
    init_q = init_q.at[self._obj_qposadr : self._obj_qposadr + 3].set(box_pos)
    init_q = init_q.at[self._obj_qposadr + 3 : self._obj_qposadr + 7].set(
        box_quat
    )
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
    data = self._update_target_marker(data)

    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        "is_box_grasped": jp.array(0.0, dtype=float),
        "is_box_on_support": jp.array(0.0, dtype=float),
        "is_box_static": jp.array(0.0, dtype=float),
        "success": jp.array(0.0, dtype=float),
        **{k: jp.array(0.0, dtype=float) for k in self._config.reward_config.scales},
    }
    info = {
        "rng": rng,
        "return_phase": jp.array(0, dtype=jp.int32),
        "retreat_z": jp.array(0.0, dtype=jp.float32),
    }
    obs = self._get_obs(data, info)
    reward_value, done = jp.zeros(2)
    return State(data, obs, reward_value, done, metrics, info)

  def step(self, state: State, action: jax.Array) -> State:
    delta = action * self._action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)
    ctrl = self._apply_return_ctrl(
        state.data,
        ctrl,
        state.info["return_phase"],
        state.info["retreat_z"],
    )

    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
    data = self._update_target_marker(data)

    raw_rewards = self._get_reward_terms(data)
    data = self._release_gripper_if_stacked(data, raw_rewards["is_box_on_support"])
    raw_rewards = self._get_reward_terms(data)
    reward_value = jp.clip(
        sum(
            raw_rewards[key] * self._config.reward_config.scales[key]
            for key in self._config.reward_config.scales
        ),
        -1e4,
        1e4,
    )
    success = raw_rewards["success"]
    box_pos = data.xpos[self._obj_body]
    support_pos = data.xpos[self._support_body]
    return_phase = jp.where(
        (state.info["return_phase"] == 0) & (success > 0),
        jp.array(1, dtype=jp.int32),
        state.info["return_phase"],
    )
    retreat_z = jp.where(
        (state.info["return_phase"] == 0) & (success > 0),
        jp.maximum(
            data.site_xpos[self._gripper_site][2] + _RETREAT_Z_OFFSET,
            self._goal_pos(support_pos)[2] + _RETREAT_Z_OFFSET,
        ),
        state.info["retreat_z"],
    )
    return_phase = self._advance_return_phase(data, return_phase, retreat_z)
    data = data.replace(
        ctrl=self._apply_return_ctrl(data, data.ctrl, return_phase, retreat_z)
    )

    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= jp.any(jp.abs(support_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    out_of_bounds |= support_pos[2] < 0.0
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)

    state.metrics.update(**raw_rewards, out_of_bounds=out_of_bounds.astype(float))
    obs = self._get_obs(data, state.info)
    state.info["return_phase"] = return_phase
    state.info["retreat_z"] = retreat_z
    return State(data, obs, reward_value, done, state.metrics, state.info)

  def _release_gripper_if_stacked(
      self, data: mjx.Data, is_box_on_support: jax.Array
  ) -> mjx.Data:
    # Force the gripper command open once the cube is placed on the support.
    release_ctrl = data.ctrl.at[-1].set(
        jp.where(is_box_on_support > 0, self._uppers[-1], data.ctrl[-1])
    )
    return data.replace(ctrl=release_ctrl)

  def _apply_return_ctrl(
      self,
      data: mjx.Data,
      ctrl: jax.Array,
      return_phase: jax.Array,
      retreat_z: jax.Array,
  ) -> jax.Array:
    retreat_ctrl = self._get_retreat_ctrl(data, retreat_z)
    ctrl = jp.where(return_phase == 1, retreat_ctrl, ctrl)
    ctrl = jp.where(return_phase == 2, self._init_ctrl, ctrl)
    return ctrl

  def _advance_return_phase(
      self, data: mjx.Data, return_phase: jax.Array, retreat_z: jax.Array
  ) -> jax.Array:
    gripper_z = data.site_xpos[self._gripper_site][2]
    finger_open_sum = jp.sum(data.qpos[self._robot_qposadr[-2:]])
    lifted_clear = gripper_z >= (retreat_z - _RETREAT_Z_TOL)
    gripper_open = finger_open_sum > _RELEASE_OPEN_SUM
    return jp.where(
        (return_phase == 1) & lifted_clear & gripper_open,
        jp.array(2, dtype=jp.int32),
        return_phase,
    )

  def _get_retreat_ctrl(self, data: mjx.Data, retreat_z: jax.Array) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_rot = data.site_xmat[self._gripper_site]
    target_pos = gripper_pos.at[2].set(jp.clip(retreat_z, 0.1, 0.5))
    target_tf = jp.eye(4, dtype=gripper_pos.dtype)
    target_tf = target_tf.at[:3, :3].set(gripper_rot)
    target_tf = target_tf.at[:3, 3].set(target_pos)

    arm_ctrl = panda_kinematics.compute_franka_ik(
        target_tf, data.ctrl[6], data.ctrl[:7]
    )
    arm_ctrl = jp.where(jp.any(jp.isnan(arm_ctrl)), data.ctrl[:7], arm_ctrl)

    retreat_ctrl = data.ctrl.at[:7].set(arm_ctrl)
    retreat_ctrl = retreat_ctrl.at[7].set(self._uppers[7])
    return retreat_ctrl

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

    is_box_on_support = self._is_cube_stacked(box_pos, support_pos).astype(float)
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

  def _goal_pos(self, base_pos: jax.Array) -> jax.Array:
    return base_pos + jp.array([0.0, 0.0, 2 * _CUBE_HALF_SIZE])

  def _update_target_marker(self, data: mjx.Data) -> mjx.Data:
    goal_pos = self._goal_pos(data.xpos[self._support_body])
    return data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(goal_pos),
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(
            jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        ),
    )

  def _is_cube_stacked(
      self, upper_pos: jax.Array, lower_pos: jax.Array
  ) -> jax.Array:
    offset = upper_pos - lower_pos
    xy_ok = jp.linalg.norm(offset[:2]) <= jp.sqrt(2.0) * _CUBE_HALF_SIZE + 0.005
    z_ok = jp.abs(offset[2] - 2 * _CUBE_HALF_SIZE) <= 0.005
    return xy_ok & z_ok

  def _is_box_static(self, data: mjx.Data) -> jax.Array:
    box_qvel = data.qvel[self._box_dofadr : self._box_dofadr + 6]
    lin_vel = jp.linalg.norm(box_qvel[:3])
    ang_vel = jp.linalg.norm(box_qvel[3:])
    return (lin_vel <= _STATIC_LIN_THRESH) & (ang_vel <= _STATIC_ANG_THRESH)

  def _is_box_grasped(
      self, data: mjx.Data, box_pos: jax.Array, gripper_pos: jax.Array
  ) -> jax.Array:
    finger_open_sum = jp.sum(data.qpos[self._robot_qposadr[-2:]])
    gripper_far = jp.linalg.norm(box_pos - gripper_pos) > _RELEASE_DISTANCE
    gripper_open = finger_open_sum > _RELEASE_OPEN_SUM
    return ~(gripper_far & gripper_open)

  @property
  def nominal_params(self) -> jax.Array:
    return jp.ones(_MODEL_PARAM_SIZE)

  @property
  def dr_range(self) -> tuple[jax.Array, jax.Array]:
    low = []
    high = []
    low.append(jp.array([0.3]))
    high.append(jp.array([10.0]))
    low.append(jp.array([0.1]))
    high.append(jp.array([10.0]))
    low.append(jp.array([0.1]))
    high.append(jp.array([10.0]))
    low.append(jp.full((11,), 0.7))
    high.append(jp.full((11,), 1.3))
    low.append(jp.full((9,), 0.8))
    high.append(jp.full((9,), 1.2))
    low.append(jp.full((7,), 0.9))
    high.append(jp.full((7,), 1.1))
    return jp.concatenate(low), jp.concatenate(high)


def _apply_domain_randomization(model: mjx.Model, params: jax.Array):
  idx = 0
  geom_friction = model.geom_friction.at[_LEFT_FINGER_GEOM, 0].set(params[idx])
  geom_friction = geom_friction.at[_RIGHT_FINGER_GEOM, 0].set(params[idx])
  idx += 1

  body_mass = model.body_mass.at[_BOX_BODY].set(
      model.body_mass[_BOX_BODY] * params[idx]
  )
  idx += 1
  body_mass = body_mass.at[_SUPPORT_BODY].set(
      model.body_mass[_SUPPORT_BODY] * params[idx]
  )
  idx += 1
  body_mass = body_mass.at[_LINK_IDS].set(
      model.body_mass[_LINK_IDS] * params[idx : idx + 11]
  )
  idx += 11

  dof_armature = model.dof_armature
  dof_damping = model.dof_damping.at[_JOINT_QIDS].set(
      model.dof_damping[_JOINT_QIDS] * params[idx : idx + 9]
  )
  idx += 9

  kp_val = model.actuator_gainprm[_ARM_QIDS, 0] * params[idx : idx + 7]
  actuator_gainprm = model.actuator_gainprm.at[_ARM_QIDS, 0].set(kp_val)
  actuator_biasprm = model.actuator_biasprm.at[_ARM_QIDS, 1].set(-kp_val)
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


def _finalize_randomization(
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
    model: mjx.Model, dr_range: tuple, params: jax.Array = None, rng: jax.Array = None
):
  dr_low, dr_high = dr_range

  if rng is not None:
    dist = functools.partial(
        jax.random.uniform,
        shape=(len(dr_low),),
        minval=dr_low,
        maxval=dr_high,
    )

    @jax.vmap
    def rand_dynamics(rng_i):
      return _apply_domain_randomization(model, dist(rng_i))

    randomized = rand_dynamics(rng)
  elif params is not None:
    if params.ndim == 1:
      randomized = _apply_domain_randomization(model, params)
    else:
      randomized = jax.vmap(
          lambda p: _apply_domain_randomization(model, p)
      )(params)
  else:
    raise ValueError("rng and params wrong!")

  return _finalize_randomization(model, *randomized)


def domain_randomize_eval(
    model: mjx.Model, dr_range: tuple, params: jax.Array = None, rng: jax.Array = None
):
  dr_low, dr_high = dr_range

  if rng is not None:
    dist = functools.partial(
        jax.random.uniform,
        shape=(len(dr_low),),
        minval=dr_low,
        maxval=dr_high,
    )
    randomized = _apply_domain_randomization(model, dist(rng))
  elif params is not None:
    randomized = _apply_domain_randomization(model, params)
  else:
    raise ValueError("rng and params wrong!")

  return _finalize_randomization(model, *randomized)
