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
"""Base classes for T1."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from custom_envs import mjx_env
from custom_envs.locomotion.t1 import t1_constants as consts


def get_assets() -> Dict[str, bytes]:
  assets = {}
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "assets")
  path = mjx_env.MENAGERIE_PATH / "booster_t1"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  return assets


class T1Env(mjx_env.MjxEnv):
  """Base class for T1 environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)

    self._model_assets = get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mj_model.opt.ccd_iterations = 10

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path

  # Sensor readings.

  def get_gravity(self, data: mjx.Data) -> jax.Array:
    """Return the gravity vector in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GRAVITY_SENSOR}"
    )

  def get_global_linvel(self, data: mjx.Data) -> jax.Array:
    """Return the linear velocity of the robot in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GLOBAL_LINVEL_SENSOR}"
    )

  def get_global_angvel(self, data: mjx.Data) -> jax.Array:
    """Return the angular velocity of the robot in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GLOBAL_ANGVEL_SENSOR}"
    )

  def get_local_linvel(self, data: mjx.Data) -> jax.Array:
    """Return the linear velocity of the robot in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.LOCAL_LINVEL_SENSOR}"
    )

  def get_accelerometer(self, data: mjx.Data) -> jax.Array:
    """Return the accelerometer readings in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.ACCELEROMETER_SENSOR}"
    )

  def get_gyro(self, data: mjx.Data) -> jax.Array:
    """Return the gyroscope readings in the local frame."""
    return mjx_env.get_sensor_data(self.mj_model, data, f"{consts.GYRO_SENSOR}")

  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

  @property
  def dr_range(self):
    n_dofs = self._mjx_model.nv - 6
    return (
        jp.concatenate([
            jp.array([0.2]),
            jp.full((n_dofs,), 0.9),
            jp.full((n_dofs,), 1.0),
            jp.full((self._mjx_model.nbody,), 0.98),
            jp.array([-1.0]),
            jp.full((n_dofs,), -0.05),
            jp.full((self._mjx_model.nu,), 0.9),
            jp.full((n_dofs,), 0.9),
        ]),
        jp.concatenate([
            jp.array([0.6]),
            jp.full((n_dofs,), 1.1),
            jp.full((n_dofs,), 1.05),
            jp.full((self._mjx_model.nbody,), 1.02),
            jp.array([1.0]),
            jp.full((n_dofs,), 0.05),
            jp.full((self._mjx_model.nu,), 1.1),
            jp.full((n_dofs,), 1.1),
        ]),
    )

  @property
  def nominal_params(self):
    n_dofs = self._mjx_model.nv - 6
    return jp.concatenate([
        jp.array([self._mjx_model.geom_friction[0, 0]]),
        jp.ones(n_dofs),
        jp.ones(n_dofs),
        jp.ones(self._mjx_model.nbody),
        jp.zeros(1),
        jp.zeros(n_dofs),
        jp.ones(self._mjx_model.nu),
        jp.ones(n_dofs),
    ])
