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
"""Peg insertion task for ALOHA."""

from typing import Any, Dict, Optional, Union

import jax
from jax import numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import functools

from custom_envs import mjx_env
from mujoco_playground._src import reward as reward_util
from custom_envs.manipulation.aloha import aloha_constants as consts
from custom_envs.manipulation.aloha import base as aloha_base


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.0025,
      sim_dt=0.0025,
      episode_length=1000,
      action_repeat=2,
      action_scale=0.005,
      reward_config=config_dict.create(
          scales=config_dict.create(
              left_reward=1,
              right_reward=1,
              left_target_qpos=0.3,
              right_target_qpos=0.3,
              no_table_collision=0.3,
              socket_z_up=0.5,
              peg_z_up=0.5,
              socket_entrance_reward=4,
              peg_end2_reward=4,
              peg_insertion_reward=8,
          )
      ),
      impl="jax",
      nconmax=24 * 1024,
      njmax=256,
  )


class SinglePegInsertion(aloha_base.AlohaEnv):
  """Single peg insertion task for ALOHA."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=(consts.XML_PATH / "mjx_single_peg_insertion.xml").as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self):
    self._post_init_aloha(keyframe="home")
    self._socket_entrance_site = self._mj_model.site("socket_entrance").id
    self._socket_rear_site = self._mj_model.site("socket_rear").id
    self._peg_end2_site = self._mj_model.site("peg_end2").id
    self._socket_body = self._mj_model.body("socket").id
    self._peg_body = self._mj_model.body("peg").id

    self._socket_qadr = self._mj_model.jnt_qposadr[
        self._mj_model.body_jntadr[self._socket_body]
    ]
    self._peg_qadr = self._mj_model.jnt_qposadr[
        self._mj_model.body_jntadr[self._peg_body]
    ]

    # Lift goal: both in the air.
    self._socket_entrance_goal_pos = jp.array([-0.05, 0, 0.15])
    self._peg_end2_goal_pos = jp.array([0.05, 0, 0.15])

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng_peg, rng_socket = jax.random.split(rng, 3)

    peg_xy = jax.random.uniform(rng_peg, (2,), minval=-0.1, maxval=0.1)
    socket_xy = jax.random.uniform(rng_socket, (2,), minval=-0.1, maxval=0.1)
    init_q = self._init_q.at[self._peg_qadr : self._peg_qadr + 2].add(peg_xy)
    init_q = init_q.at[self._socket_qadr : self._socket_qadr + 2].add(socket_xy)

    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )

    info = {"rng": rng}
    obs = self._get_obs(data)
    reward, done = jp.zeros(2)
    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        "peg_end2_dist_to_line": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    }

    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    delta = action * self._config.action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)

    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

    socket_entrance_pos = data.site_xpos[self._socket_entrance_site]
    socket_rear_pos = data.site_xpos[self._socket_rear_site]
    peg_end2_pos = data.site_xpos[self._peg_end2_site]
    # Insertion reward: if peg end2 is aligned with hole entrance, then reward
    # distance from peg end to socket interior.
    socket_ab = socket_entrance_pos - socket_rear_pos
    socket_t = jp.dot(peg_end2_pos - socket_rear_pos, socket_ab)
    socket_t /= jp.dot(socket_ab, socket_ab) + 1e-6
    nearest_pt = data.site_xpos[self._socket_rear_site] + socket_t * socket_ab
    peg_end2_dist_to_line = jp.linalg.norm(peg_end2_pos - nearest_pt)

    out_of_bounds = jp.any(jp.abs(data.xpos[self._socket_body]) > 1.0)
    out_of_bounds |= jp.any(jp.abs(data.xpos[self._peg_body]) > 1.0)

    raw_rewards = self._get_reward(
        data, use_peg_insertion_reward=(peg_end2_dist_to_line < 0.005)
    )
    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    reward = sum(rewards.values()) / sum(
        self._config.reward_config.scales.values()
    )

    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    reward_finite = jp.isfinite(reward)
    done = done | (~reward_finite)
    reward = jp.where(reward_finite, reward, 0.0)
    done = done.astype(float)
    state.metrics.update(
        **rewards,
        peg_end2_dist_to_line=peg_end2_dist_to_line,
        out_of_bounds=out_of_bounds.astype(float),
    )
    obs = self._get_obs(data)
    return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

  def _get_obs(self, data: mjx.Data) -> dict[str, jax.Array]:
    left_gripper_pos = data.site_xpos[self._left_gripper_site]
    socket_pos = data.xpos[self._socket_body]
    right_gripper_pos = data.site_xpos[self._right_gripper_site]
    peg_pos = data.xpos[self._peg_body]
    socket_entrance_pos = data.site_xpos[self._socket_entrance_site]
    peg_end2_pos = data.site_xpos[self._peg_end2_site]
    socket_z = data.xmat[self._socket_body].ravel()[6:]
    peg_z = data.xmat[self._peg_body].ravel()[6:]

    state = jp.concatenate([
        data.qpos,
        data.qvel,
        left_gripper_pos,
        socket_pos,
        right_gripper_pos,
        peg_pos,
        socket_entrance_pos,
        peg_end2_pos,
        socket_z,
        peg_z,
    ])

    privileged_state = jp.concatenate([
        state,
        data.qfrc_bias,
        data.actuator_force,
        jp.mean(self.mjx_model.geom_friction[:, 0:1], axis=0),
        self.mjx_model.body_mass[:],
        self.mjx_model.actuator_gainprm[:, 0],
        self.mjx_model.dof_damping[:16],
        self.mjx_model.dof_armature[:16],
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  @property
  def nominal_params(self) -> jax.Array:
    return jp.ones(5)

  @property
  def dr_range(self) -> tuple[jax.Array, jax.Array]:
    low = jp.array([
        .9,  # geom friction (mu)
        .9,  # object mass scale (socket & peg)
        0.9,  # robot mass scale
        0.9,  # joint damping scale
        0.9,  # actuator gain scale
    ])
    high = jp.array([
        1.1,
        1.1,
        1.1,
        1.1,
        1.1,
    ])
    return low, high

  def _get_reward(
      self, data: mjx.Data, use_peg_insertion_reward: bool
  ) -> Dict[str, jax.Array]:
    left_socket_dist = jp.linalg.norm(
        data.xpos[self._socket_body] - data.site_xpos[self._left_gripper_site]
    )
    left_reward = reward_util.tolerance(
        left_socket_dist, (0, 0.001), margin=0.3, sigmoid="linear"
    )
    right_peg_dist = jp.linalg.norm(
        data.xpos[self._peg_body] - data.site_xpos[self._right_gripper_site]
    )
    right_reward = reward_util.tolerance(
        right_peg_dist, (0, 0.001), margin=0.3, sigmoid="linear"
    )

    robot_qpos_diff = data.qpos[self._arm_qadr] - self._init_q[self._arm_qadr]
    left_pose = jp.linalg.norm(robot_qpos_diff[:6])
    left_pose = reward_util.tolerance(left_pose, (0, 0.01), margin=2.0)
    right_pose = jp.linalg.norm(robot_qpos_diff[6:])
    right_pose = reward_util.tolerance(right_pose, (0, 0.01), margin=2.0)

    socket_dist = jp.linalg.norm(
        self._socket_entrance_goal_pos - data.xpos[self._socket_body]
    )
    socket_lift = reward_util.tolerance(
        socket_dist, (0, 0.01), margin=0.15, sigmoid="linear"
    )

    peg_dist = jp.linalg.norm(
        self._peg_end2_goal_pos - data.xpos[self._peg_body]
    )
    peg_lift = reward_util.tolerance(
        peg_dist, (0, 0.01), margin=0.15, sigmoid="linear"
    )

    table_collision = self.hand_table_collision(data)

    socket_orientation = jp.dot(
        data.xmat[self._socket_body][2], jp.array([0.0, 0.0, 1.0])
    )
    socket_orientation = reward_util.tolerance(
        socket_orientation, (0.99, 1.0), margin=0.03, sigmoid="linear"
    )
    peg_orientation = jp.dot(
        data.xmat[self._peg_body][2], jp.array([0.0, 0.0, 1.0])
    )
    peg_orientation = reward_util.tolerance(
        peg_orientation, (0.99, 1.0), margin=0.03, sigmoid="linear"
    )

    peg_insertion_dist = jp.linalg.norm(
        data.site_xpos[self._peg_end2_site]
        - data.site_xpos[self._socket_rear_site]
    )
    peg_insertion_reward = (
        reward_util.tolerance(
            peg_insertion_dist, (0, 0.001), margin=0.1, sigmoid="linear"
        )
        * use_peg_insertion_reward
    )

    return {
        "left_reward": left_reward,
        "right_reward": right_reward,
        "left_target_qpos": left_pose * left_reward * right_reward,
        "right_target_qpos": right_pose * left_reward * right_reward,
        "no_table_collision": 1 - table_collision,
        "socket_entrance_reward": socket_lift,
        "peg_end2_reward": peg_lift,
        "socket_z_up": socket_orientation * socket_lift,
        "peg_z_up": peg_orientation * peg_lift,
        "peg_insertion_reward": peg_insertion_reward,
    }


def domain_randomize(
    model: mjx.Model,
    dr_range: tuple[jax.Array, jax.Array],
    params: jax.Array = None,
    rng: jax.Array = None,
):
  """Applies domain randomization to AlohaSinglePegInsertion MJX model.

  Supports both single params (shape [D]) and batched params (shape [B, D]).
  """
  dr_low, dr_high = dr_range
  socket_body = model.body_mass.shape[0] - 2
  peg_body = model.body_mass.shape[0] - 1
  obj_dofs = 12  # two free joints

  def _shift(p):
    idx = 0
    friction_mu = p[idx]
    idx += 1
    obj_mass_scale = p[idx]
    idx += 1
    robot_mass_scale = p[idx]
    idx += 1
    damping_scale = p[idx]
    idx += 1
    gain_scale = p[idx]
    idx += 1
    assert idx == len(dr_low)

    geom_friction = model.geom_friction.at[:, 0].set(friction_mu)
    body_mass = model.body_mass
    body_mass = body_mass.at[1:socket_body].set(body_mass[1:socket_body] * robot_mass_scale)
    body_mass = body_mass.at[socket_body].set(body_mass[socket_body] * obj_mass_scale)
    body_mass = body_mass.at[peg_body].set(body_mass[peg_body] * obj_mass_scale)

    dof_damping = model.dof_damping.at[: model.nv - obj_dofs].set(
        model.dof_damping[: model.nv - obj_dofs] * damping_scale
    )
    dof_armature = model.dof_armature

    kp_val = model.actuator_gainprm[:, 0] * gain_scale
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp_val)
    actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp_val)

    return geom_friction, body_mass, dof_damping, dof_armature, actuator_gainprm, actuator_biasprm

  if rng is not None:
    if rng.ndim == 1:
      p = jax.random.uniform(rng, (len(dr_low),), minval=dr_low, maxval=dr_high)
      geom_friction, body_mass, dof_damping, dof_armature, actuator_gainprm, actuator_biasprm = _shift(p)
    else:
      dist = functools.partial(
          jax.random.uniform, shape=(len(dr_low),), minval=dr_low, maxval=dr_high
      )
      geom_friction, body_mass, dof_damping, dof_armature, actuator_gainprm, actuator_biasprm = jax.vmap(
          lambda key: _shift(dist(key))
      )(rng)
  else:
    if params.ndim == 1:
      geom_friction, body_mass, dof_damping, dof_armature, actuator_gainprm, actuator_biasprm = _shift(params)
    else:
      geom_friction, body_mass, dof_damping, dof_armature, actuator_gainprm, actuator_biasprm = jax.vmap(_shift)(params)

  model = model.tree_replace({
      "geom_friction": geom_friction,
      "body_mass": body_mass,
      "dof_damping": dof_damping,
      "dof_armature": dof_armature,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
  })

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  if (params is not None and getattr(params, "ndim", 0) == 2) or (
      rng is not None and getattr(rng, "ndim", 0) == 2
  ):
    in_axes = in_axes.tree_replace({
        "geom_friction": 0,
        "body_mass": 0,
        "dof_damping": 0,
        "dof_armature": 0,
        "actuator_gainprm": 0,
        "actuator_biasprm": 0,
    })

  return model, in_axes


def domain_randomize_eval(
    model: mjx.Model,
    dr_range: tuple[jax.Array, jax.Array],
    params: jax.Array = None,
    rng: jax.Array = None,
):
  return domain_randomize(model=model, dr_range=dr_range, params=params, rng=rng)
