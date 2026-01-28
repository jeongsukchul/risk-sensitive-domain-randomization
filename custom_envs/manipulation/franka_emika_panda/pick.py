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

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from custom_envs import mjx_env
from custom_envs.manipulation.franka_emika_panda import panda
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member
import numpy as np


def default_config() -> config_dict.ConfigDict:
  """Returns the default config for bring_to_target tasks."""
  config = config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=300,
      action_repeat=1,
      action_scale=0.04,
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=.4, #4.0,
              # Box goes to the target mocap.
              box_target=.8, #8.0,
              # Do not collide the gripper with the floor.
              no_floor_collision=.025, #0.25,
              # Arm stays close to target pose.
              robot_target_qpos=0.03, #0.3,
          )
      ),
      impl='jax',
      nconmax=24 * 2048,
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

  def reset(self, rng: jax.Array) -> State:
    rng, rng_box, rng_target = jax.random.split(rng, 3)

    # intialize box position
    box_pos = (
        jax.random.uniform(
            rng_box,
            (3,),
            minval=jp.array([-0.2, -0.2, 0.0]),
            maxval=jp.array([0.2, 0.2, 0.0]),
        )
        + self._init_obj_pos
    )

    # initialize target position
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
      # sample a random direction
      rng, rng_axis, rng_theta = jax.random.split(rng, 3)
      perturb_axis = jax.random.uniform(rng_axis, (3,), minval=-1, maxval=1)
      perturb_axis = perturb_axis / math.norm(perturb_axis)
      perturb_theta = jax.random.uniform(rng_theta, maxval=np.deg2rad(45))
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
        nconmax=self._config.nconmax,
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

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    obs = jp.concatenate([
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
    # --- Privileged state for Critic ---
        # Note: self._left_finger_geom and other IDs should be defined in PandaBase
    privileged_state = jp.concatenate([
        obs,
        data.qfrc_bias,               # Coriolis, centrifugal, gravity
        data.actuator_force,          # Actual applied forces
        # Latent physical properties (the "Ground Truth" for the Critic)
        self.mjx_model.geom_friction[self._left_finger_geom, 0:1],
        self.mjx_model.body_mass[:],
        self.mjx_model.actuator_gainprm[:, 0], 
        self.mjx_model.dof_damping[:9],
        self.mjx_model.dof_armature[:9],
    ])

    return {
        "state": obs,
        "privileged_state": privileged_state,
    }
    return obs
  @property
  def nominal_params(self) -> jax.Array:
        return jp.ones(29) #jp.ones(38)

  @property
  def dr_range(self) -> tuple[jax.Array, jax.Array]:
        """Defines the lower and upper bounds for randomization (38 dimensions)."""
        low, high = [], []
        # 1. Gripper Friction (1 param)
        low.append(jp.array([0.3])); high.append(jp.array([3.0]))
        # 2. Cube Mass Scale (1 param)
        low.append(jp.array([0.1])); high.append(jp.array([9.0]))
        # 3. Franka Link Mass Scale (11 params)
        low.append(jp.full((11,), 0.5)); high.append(jp.full((11,), 1.5))
        # 4. Robot Armature Scale (9 params)
        # low.append(jp.full((9,), 0.8)); high.append(jp.full((9,), 1.2))
        # 5. Joint Damping Scale (9 params)
        low.append(jp.full((9,), 0.8)); high.append(jp.full((9,), 1.2))
        # 6. Actuator Gain (KP) Scale (7 params)
        low.append(jp.full((7,), 0.9)); high.append(jp.full((7,), 1.1))

        return jp.concatenate(low), jp.concatenate(high)

class PandaPickCubeOrientation(PandaPickCube):
  """Bring a box to a target and orientation."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides, sample_orientation=True)
LEFT_FINGER_GEOM = 73
RIGHT_FINGER_GEOM = 80
import functools

def domain_randomize(model: mjx.Model, dr_range: tuple, params: jax.Array = None, rng: jax.Array = None):
    """Applies randomization to the MJX Model."""
    dr_low, dr_high = dr_range
    
    from mujoco import mj_id2name, mj_name2id
    import functools
    # Identify relevant IDs
    cube_body = model.body_mass.shape[0] - 2 # Or specific name
    arm_qids = jp.arange(7) # First 7 joints of Panda
    joint_qids = jp.arange(9) # First 9 joints of Panda
    link_ids = jp.arange(11)+1
    if rng is not None:
        dist = functools.partial(jax.random.uniform, shape=(len(dr_low)), minval=dr_low, maxval=dr_high)
        params = jax.random.uniform(rng, (len(dr_low),), minval=dr_low, maxval=dr_high)

    @jax.vmap
    def shift_dynamics(p):
        idx = 0
        # 1. Friction
        friction = model.geom_friction.at[LEFT_FINGER_GEOM, 0].set(p[idx])
        friction = model.geom_friction.at[RIGHT_FINGER_GEOM, 0].set(p[idx]); idx += 1
        # 2. Mass
        mass = model.body_mass.at[cube_body].set(model.body_mass[cube_body] * p[idx]); idx += 1
        mass = mass.at[:11].set(model.body_mass[link_ids] * p[idx:idx+11]); idx += 11
        # 3. Armature
        # armature = model.dof_armature.at[joint_qids].set(model.dof_armature[joint_qids] * p[idx:idx+9]); idx += 9
        armature=model.dof_armature
        # 4. Damping
        damping = model.dof_damping.at[joint_qids].set(model.dof_damping[joint_qids] * p[idx:idx+9]); idx += 9
        # 5. Gain/Bias (KP)
        kp_val = model.actuator_gainprm[arm_qids, 0] * p[idx:idx+7]
        gain = model.actuator_gainprm.at[arm_qids, 0].set(kp_val)
        bias = model.actuator_biasprm.at[arm_qids, 1].set(-kp_val); idx += 7
        assert idx ==len(dr_low)
        return friction, mass, armature, damping, gain, bias
    @jax.vmap
    def rand_dynamics(rng):
        p = dist(rng)
        idx = 0
        # 1. Friction
        friction = model.geom_friction.at[LEFT_FINGER_GEOM, 0].set(p[idx])
        friction = model.geom_friction.at[RIGHT_FINGER_GEOM, 0].set(p[idx]); idx += 1
        # 2. Mass
        mass = model.body_mass.at[cube_body].set(model.body_mass[cube_body] * p[idx]); idx += 1
        mass = mass.at[:11].set(model.body_mass[link_ids] * p[idx:idx+11]); idx += 11
        # 3. Armature
        # armature = model.dof_armature.at[joint_qids].set(model.dof_armature[joint_qids] * p[idx:idx+9]); idx += 9
        armature=model.dof_armature
        # 4. Damping
        damping = model.dof_damping.at[joint_qids].set(model.dof_damping[joint_qids] * p[idx:idx+9]); idx += 9
        # 5. Gain/Bias (KP)
        kp_val = model.actuator_gainprm[arm_qids, 0] * p[idx:idx+7]
        gain = model.actuator_gainprm.at[arm_qids, 0].set(kp_val)
        bias = model.actuator_biasprm.at[arm_qids, 1].set(-kp_val); idx += 7
        assert idx ==len(dr_low)
        return friction, mass, armature, damping, gain, bias

    friction, mass, armature, damping, gain, bias = rand_dynamics(rng) if rng is not None else shift_dynamics(params) 

    # Replace model fields with randomized arrays
    model = model.tree_replace({
        'geom_friction': friction,
        'body_mass': mass,
        'dof_armature': armature,
        'dof_damping': damping,
        'actuator_gainprm': gain,
        'actuator_biasprm': bias,
    })
    
    # Generate in_axes for JAX transformation
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({k: 0 for k in ['geom_friction', 'body_mass', 'dof_armature', 'dof_damping', 'actuator_gainprm', 'actuator_biasprm']})

    return model, in_axes

def domain_randomize_eval(model: mjx.Model, dr_range: tuple, params: jax.Array = None, rng: jax.Array = None):
    """Applies randomization to the MJX Model."""
    dr_low, dr_high = dr_range
    
    from mujoco import mj_id2name, mj_name2id
    # Identify relevant IDs
    cube_body = model.body_mass.shape[0] - 2 # Or specific name
    arm_qids = jp.arange(7) # First 7 joints of Panda
    joint_qids = jp.arange(9) # First 9 joints of Panda
    link_ids = jp.arange(11)+1
    if rng is not None:
        dist = functools.partial(jax.random.uniform, shape=(len(dr_low)), minval=dr_low, maxval=dr_high)
        # params = jax.random.uniform(rng, (len(dr_low),), minval=dr_low, maxval=dr_high)

    def shift_dynamics(p):
        idx = 0
        # 1. Friction
        friction = model.geom_friction.at[LEFT_FINGER_GEOM, 0].set(p[idx])
        friction = model.geom_friction.at[RIGHT_FINGER_GEOM, 0].set(p[idx]); idx += 1
        # 2. Mass
        mass = model.body_mass.at[cube_body].set(model.body_mass[cube_body] * p[idx]); idx += 1
        mass = mass.at[:11].set(model.body_mass[link_ids] * p[idx:idx+11]); idx += 11
        # 3. Armature
        # armature = model.dof_armature.at[joint_qids].set(model.dof_armature[joint_qids] * p[idx:idx+9]); idx += 9
        armature=model.dof_armature
        # 4. Damping
        damping = model.dof_damping.at[joint_qids].set(model.dof_damping[joint_qids] * p[idx:idx+9]); idx += 9
        # 5. Gain/Bias (KP)
        kp_val = model.actuator_gainprm[arm_qids, 0] * p[idx:idx+7]
        gain = model.actuator_gainprm.at[arm_qids, 0].set(kp_val)
        bias = model.actuator_biasprm.at[arm_qids, 1].set(-kp_val); idx += 7
        assert idx ==len(dr_low)
        return friction, mass, armature, damping, gain, bias
    def rand_dynamics(rng):
        p = dist(rng)
        idx = 0
        # 1. Friction
        friction = model.geom_friction.at[LEFT_FINGER_GEOM, 0].set(p[idx])
        friction = model.geom_friction.at[RIGHT_FINGER_GEOM, 0].set(p[idx]); idx += 1
        # 2. Mass
        mass = model.body_mass.at[cube_body].set(model.body_mass[cube_body] * p[idx]); idx += 1
        mass = mass.at[:11].set(model.body_mass[link_ids] * p[idx:idx+11]); idx += 11
        # 3. Armature
        # armature = model.dof_armature.at[joint_qids].set(model.dof_armature[joint_qids] * p[idx:idx+9]); idx += 9
        armature=model.dof_armature
        # 4. Damping
        damping = model.dof_damping.at[joint_qids].set(model.dof_damping[joint_qids] * p[idx:idx+9]); idx += 9
        # 5. Gain/Bias (KP)
        kp_val = model.actuator_gainprm[arm_qids, 0] * p[idx:idx+7]
        gain = model.actuator_gainprm.at[arm_qids, 0].set(kp_val)
        bias = model.actuator_biasprm.at[arm_qids, 1].set(-kp_val); idx += 7
        assert idx ==len(dr_low)
        return friction, mass, armature, damping, gain, bias

    friction, mass, armature, damping, gain, bias = rand_dynamics(rng) if rng is not None else shift_dynamics(params) 

    # Replace model fields with randomized arrays
    model = model.tree_replace({
        'geom_friction': friction,
        'body_mass': mass,
        'dof_armature': armature,
        'dof_damping': damping,
        'actuator_gainprm': gain,
        'actuator_biasprm': bias,
    })
    
    # Generate in_axes for JAX transformation
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({k: 0 for k in ['geom_friction', 'body_mass', 'dof_armature', 'dof_damping', 'actuator_gainprm', 'actuator_biasprm']})

    return model, in_axes