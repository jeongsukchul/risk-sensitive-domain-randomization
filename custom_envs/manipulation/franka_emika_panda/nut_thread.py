"""Robosuite-inspired Panda nut threading / assembly task."""

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
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member

_MODEL_PARAM_SIZE = 29
_NUT_HALF_HEIGHT = 0.024
_SUCCESS_XY = 0.0375
_SUCCESS_Z = 0.036
_RELEASE_DISTANCE = 0.15
_RELEASE_OPEN_SUM = 0.055

_XML_PATH = (
    mjx_env.ROOT_PATH
    / "manipulation"
    / "franka_emika_panda"
    / "xmls"
    / "mjx_nut_thread.xml"
)
_REF_MODEL = mujoco.MjModel.from_xml_string(
    _XML_PATH.read_text(), assets=panda.get_assets()
)
_LEFT_FINGER_GEOM = _REF_MODEL.geom("left_finger_pad").id
_RIGHT_FINGER_GEOM = _REF_MODEL.geom("right_finger_pad").id
_NUT_BODY = _REF_MODEL.body("nut").id
_LINK_IDS = np.arange(11) + 1
_ARM_QIDS = np.arange(7)
_JOINT_QIDS = np.arange(9)


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=350,
      action_repeat=1,
      action_scale=0.04,
      reward_config=config_dict.create(
          scales=config_dict.create(
              gripper_nut=1.5,
              nut_lift=1.0,
              nut_align=2.0,
              nut_thread=4.0,
              release=1.0,
              no_floor_collision=0.1,
              robot_target_qpos=0.1,
              success=2.0,
          )
      ),
      impl="jax",
      nconmax=96 * 2048,
      njmax=400,
  )


class PandaNutThread(panda.PandaBase):
  """Single Panda nut-on-peg task inspired by robosuite NutAssembly."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(_XML_PATH, config, config_overrides)
    self._post_init(obj_name="nut", keyframe="home")

    self._nut_center_site = self._mj_model.site("nut_center").id
    self._nut_handle_site = self._mj_model.site("nut_handle_site").id
    self._peg_site = self._mj_model.site("peg_site").id
    self._peg_top_site = self._mj_model.site("peg_top_site").id
    self._peg_hover_site = self._mj_model.site("peg_hover_site").id
    self._peg_body = self._mj_model.body("peg").id
    self._peg_qposadr = self._mj_model.jnt_qposadr[
        self._mj_model.joint("peg_x").id
    ]
    self._nut_dofadr = self._mj_model.jnt_dofadr[
        self._mj_model.body("nut").jntadr[0]
    ]
    self._floor_hand_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
    ]
    self._peg_base_xy = jp.array(self._mj_model.body("peg").pos[:2], dtype=jp.float32)
    self._goal_z = jp.array(self._mj_model.site("peg_site").pos[2], dtype=jp.float32)
    self._hover_z = jp.array(
        self._mj_model.site("peg_hover_site").pos[2], dtype=jp.float32
    )

  def reset(self, rng: jax.Array) -> State:
    rng, rng_peg_xy, rng_yaw = jax.random.split(rng, 3)

    nut_pos = jp.array(self._init_obj_pos)
    peg_xy = jax.random.uniform(
        rng_peg_xy,
        (2,),
        minval=jp.array([0.54, -0.18]),
        maxval=jp.array([0.64, -0.06]),
    )
    peg_offset = peg_xy - self._peg_base_xy
    nut_quat = math.axis_angle_to_quat(
        jp.array([0.0, 0.0, 1.0]),
        jax.random.uniform(rng_yaw, (), minval=-jp.pi, maxval=jp.pi),
    )

    init_q = jp.array(self._init_q)
    init_q = init_q.at[self._obj_qposadr : self._obj_qposadr + 3].set(nut_pos)
    init_q = init_q.at[self._obj_qposadr + 3 : self._obj_qposadr + 7].set(nut_quat)
    init_q = init_q.at[self._peg_qposadr : self._peg_qposadr + 2].set(peg_offset)

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
        "is_nut_grasped": jp.array(0.0, dtype=float),
        "success": jp.array(0.0, dtype=float),
        **{k: jp.array(0.0, dtype=float) for k in self._config.reward_config.scales},
    }
    info = {"rng": rng}
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return State(data, obs, reward, done, metrics, info)

  def step(self, state: State, action: jax.Array) -> State:
    delta = action * self._action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)

    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
    data = self._update_target_marker(data)

    raw_rewards = self._get_reward_terms(data)
    reward = jp.clip(
        sum(
            raw_rewards[key] * self._config.reward_config.scales[key]
            for key in self._config.reward_config.scales
        ),
        -1e4,
        1e4,
    )

    nut_pos = data.xpos[self._obj_body]
    out_of_bounds = jp.any(jp.abs(nut_pos) > 1.0)
    out_of_bounds |= nut_pos[2] < -0.02
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)

    state.metrics.update(**raw_rewards, out_of_bounds=out_of_bounds.astype(float))
    obs = self._get_obs(data, state.info)
    return State(data, obs, reward, done, state.metrics, state.info)

  def _get_reward_terms(self, data: mjx.Data) -> Dict[str, jax.Array]:
    nut_pos = data.site_xpos[self._nut_center_site]
    nut_handle = data.site_xpos[self._nut_handle_site]
    peg_pos = data.site_xpos[self._peg_site]
    peg_hover = data.site_xpos[self._peg_hover_site]
    gripper_pos = data.site_xpos[self._gripper_site]

    handle_dist = jp.linalg.norm(nut_handle - gripper_pos)
    xy_err = jp.linalg.norm((nut_pos - peg_pos)[:2])
    thread_z_err = jp.abs(nut_pos[2] - peg_pos[2])
    hover_err = jp.linalg.norm(nut_pos - peg_hover)

    gripper_nut = 1.0 - jp.tanh(8.0 * handle_dist)
    nut_lift = 1.0 - jp.tanh(12.0 * jp.abs(nut_pos[2] - peg_hover[2]))
    nut_align = 1.0 - jp.tanh(10.0 * (1.5 * xy_err + 0.25 * hover_err))
    nut_thread = 1.0 - jp.tanh(18.0 * (2.5 * xy_err + thread_z_err))

    is_nut_grasped = self._is_nut_grasped(data, nut_pos, gripper_pos).astype(float)
    thread_ready = jp.maximum(is_nut_grasped, (xy_err < 2.0 * _SUCCESS_XY).astype(float))
    release = nut_thread * (1.0 - is_nut_grasped)
    success = (
        (xy_err < _SUCCESS_XY)
        & (thread_z_err < _SUCCESS_Z)
        & (~self._is_nut_grasped(data, nut_pos, gripper_pos))
    ).astype(float)

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

    return {
        "gripper_nut": gripper_nut,
        "nut_lift": nut_lift * is_nut_grasped,
        "nut_align": nut_align * jp.maximum(is_nut_grasped, nut_lift > 0.25),
        "nut_thread": nut_thread * thread_ready,
        "release": release,
        "no_floor_collision": no_floor_collision,
        "robot_target_qpos": robot_target_qpos,
        "success": success,
        "is_nut_grasped": is_nut_grasped,
    }

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
    del info
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    nut_pos = data.site_xpos[self._nut_center_site]
    nut_handle = data.site_xpos[self._nut_handle_site]
    peg_pos = data.site_xpos[self._peg_site]
    peg_top = data.site_xpos[self._peg_top_site]

    state = jp.concatenate([
        data.qpos,
        data.qvel,
        gripper_pos,
        gripper_mat[3:],
        data.xmat[self._obj_body].ravel()[3:],
        nut_pos - gripper_pos,
        nut_handle - gripper_pos,
        peg_pos - nut_pos,
        peg_top - nut_pos,
        peg_pos - gripper_pos,
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])
    privileged_state = jp.concatenate([
        state,
        data.qfrc_bias,
        data.actuator_force,
        self.mjx_model.geom_friction[self._left_finger_geom, 0:1],
        self.mjx_model.body_mass[self._obj_body : self._obj_body + 1],
        self.mjx_model.actuator_gainprm[:, 0],
        self.mjx_model.dof_damping[:9],
        self.mjx_model.dof_armature[:9],
    ])
    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _update_target_marker(self, data: mjx.Data) -> mjx.Data:
    goal_pos = data.site_xpos[self._peg_site]
    return data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(goal_pos),
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(
            jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        ),
    )

  def _is_nut_grasped(
      self, data: mjx.Data, nut_pos: jax.Array, gripper_pos: jax.Array
  ) -> jax.Array:
    finger_open_sum = jp.sum(data.qpos[self._robot_qposadr[-2:]])
    gripper_far = jp.linalg.norm(nut_pos - gripper_pos) > _RELEASE_DISTANCE
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
    high.append(jp.array([3.0]))
    low.append(jp.array([0.25]))
    high.append(jp.array([4.0]))
    low.append(jp.full((11,), 0.8))
    high.append(jp.full((11,), 1.2))
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

  body_mass = model.body_mass.at[_NUT_BODY].set(model.body_mass[_NUT_BODY] * params[idx])
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
