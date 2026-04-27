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
"""Acrobot environment."""

import functools
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from custom_envs import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "acrobot.xml"


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.01,
      sim_dt=0.01,
      episode_length=1000,
      action_repeat=1,
      vision=False,
      impl="warp",
      nconmax=0,
      njmax=0,
  )
FLOOR_GEOM_ID = 0
UPPER_ARM_BODY_ID = 1
LOWER_ARM_BODY_ID = 2

class Balance(mjx_env.MjxEnv):
    """Acrobot environment."""

    def __init__(
        self,
        sparse: bool,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)
        if self._config.vision:
            raise NotImplementedError(
                f"Vision not implemented for {self.__class__.__name__}."
            )

        self._margin = 0.0 if sparse else 1.0

        self._xml_path = _XML_PATH.as_posix()
        self._model_assets = common.get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(
            _XML_PATH.read_text(), self._model_assets
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:
        self._upper_arm_body_id = self._mj_model.body("upper_arm").id
        self._lower_arm_body_id = self._mj_model.body("lower_arm").id
        self._tip_site_id = self._mj_model.site("tip").id
        self._target_site_id = self._mj_model.site("target").id
        self._target_radius = self._mj_model.site_size[self._target_site_id, 0]

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng1 = jax.random.split(rng, 2)

        qpos = jax.random.uniform(
            rng1, (self.mjx_model.nq,), minval=-jp.pi/8, maxval=jp.pi/8,
        )
        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            impl=self.mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        metrics = {
            "distance": jp.zeros(()),
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

    def _get_obs(self, data: mjx.Data, unused_info: dict[str, Any]) -> dict[str, jax.Array]:
        state = jp.concatenate([
            self._orientations(data),
            data.qvel,
        ])

        privileged_state = jp.concatenate([
            state,
            self.mjx_model.body_mass[UPPER_ARM_BODY_ID:LOWER_ARM_BODY_ID + 1],
        ])

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
        del action, info, metrics  # Unused.
        return reward.tolerance(
            self._to_target(data),
            bounds=(0.0, self._target_radius),
            margin=self._margin,
        )

    def _horizontal(self, data: mjx.Data) -> jax.Array:
        """Returns horizontal (x) component of body frame z-axes."""
        return data.xmat[[self._upper_arm_body_id, self._lower_arm_body_id], 0, 2]

    def _vertical(self, data: mjx.Data) -> jax.Array:
        """Returns vertical (z) component of body frame z-axes."""
        return data.xmat[[self._upper_arm_body_id, self._lower_arm_body_id], 2, 2]

    def _to_target(self, data: mjx.Data) -> jax.Array:
        """Returns the distance from the tip to the target."""
        target_pos = data.site_xpos[self._target_site_id]
        tip_pos = data.site_xpos[self._tip_site_id]
        return jp.linalg.norm(target_pos - tip_pos)

    def _orientations(self, data: mjx.Data) -> jax.Array:
        """Returns the sines and cosines of the pole angles."""
        return jp.concatenate([self._horizontal(data), self._vertical(data)])

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
        return jp.array([1.0, 1.0, 1.0])

    @property
    def dr_range(self):
        low = jp.array([
            0.78,   # upper arm mass
            0.5,   # lower arm mass
        ])
        high = jp.array([
            3.,  # upper arm mass
            1.2,  # lower arm mass
        ])  
        return low, high

    @property
    def dr_range_wide(self):
        low = jp.array([
            0.05,
            0.05,
        ])
        high = jp.array([
            30.0,
            30.0,
        ])
        return low, high

    @property
    def dr_label(self):
        return ("upper arm mass", "lower arm mass")

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
  def shift_dynamics(params):
    idx = 0


    body_mass = model.body_mass.at[UPPER_ARM_BODY_ID].set(params[idx])
    idx += 1
    body_mass = body_mass.at[LOWER_ARM_BODY_ID].set(params[idx])
    idx += 1

    assert idx == len(params)
    return body_mass

  @jax.vmap
  def rand_dynamics(rng):
    rng_params = dist(rng)
    idx = 0


    body_mass = model.body_mass.at[UPPER_ARM_BODY_ID].set(rng_params[idx])
    idx += 1
    body_mass = body_mass.at[LOWER_ARM_BODY_ID].set(rng_params[idx])
    idx += 1

    assert idx == len(rng_params)
    return body_mass

  if rng is None and params is not None:
    body_mass = shift_dynamics(params)
  elif rng is not None and params is None:
    body_mass = rand_dynamics(rng)
  else:
    raise ValueError("rng and params wrong!")

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "body_mass": 0,
  })

  model = model.tree_replace({
      "body_mass": body_mass,
  })

  return model, in_axes


def domain_randomize_eval(model: mjx.Model, dr_range, params=None, rng: jax.Array = None):
    if rng is not None:
        dr_low, dr_high = dr_range
        dist = functools.partial(
            jax.random.uniform,
            shape=(len(dr_low),),
            minval=dr_low,
            maxval=dr_high,
        )

    def shift_dynamics(params):
        idx = 0


        body_mass = model.body_mass.at[UPPER_ARM_BODY_ID].set(params[idx])
        idx += 1
        body_mass = body_mass.at[LOWER_ARM_BODY_ID].set(params[idx])
        idx += 1

        assert idx == len(params)
        return body_mass

    def rand_dynamics(rng):
        rng_params = dist(rng)
        idx = 0


        body_mass = model.body_mass.at[UPPER_ARM_BODY_ID].set(rng_params[idx])
        idx += 1
        body_mass = body_mass.at[LOWER_ARM_BODY_ID].set(rng_params[idx])
        idx += 1

        assert idx == len(rng_params)
        return body_mass

    if rng is None and params is not None:
        body_mass = shift_dynamics(params)
    elif rng is not None and params is None:
        body_mass = rand_dynamics(rng)
    else:
        raise ValueError("rng and params wrong!")

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "body_mass": 0,
    })

    model = model.tree_replace({
        "body_mass": body_mass,
    })

    return model, in_axes