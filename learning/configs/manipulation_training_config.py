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
"""RL config for DM Control Suite."""

from ml_collections import config_dict

# from mujoco_playground._src import dm_control_suite, locomotion
from custom_envs import dm_control_suite, manipulation, locomotion
from module.termination_fn import get_termination_fn

def manipulation_ppo_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax PPO config for the given environment."""
  env_config = manipulation.get_default_config(env_name)

  rl_config = config_dict.create(
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=env_config.action_repeat,
      reward_scaling=1.0,
      network_factory=config_dict.create(
          policy_hidden_layer_sizes=(32, 32, 32, 32),
          value_hidden_layer_sizes=(256, 256, 256, 256, 256),
          policy_obs_key="state",
          value_obs_key="state",
      ),
  )
  if env_name == "AlohaHandOver":
    rl_config.num_timesteps = 100_000_000
    rl_config.num_evals = int(rl_config.num_timesteps / 4_000_000)
    rl_config.unroll_length = 15
    rl_config.num_minibatches = 32
    rl_config.num_updates_per_batch = 8
    rl_config.discounting = 0.97
    rl_config.learning_rate = 1e-3
    rl_config.entropy_cost = 2e-2
    rl_config.num_envs = 2048
    rl_config.num_eval_envs = 128
    rl_config.batch_size = 512
    rl_config.max_grad_norm = 1.0
    rl_config.network_factory.policy_hidden_layer_sizes = (256, 256, 256)
  elif env_name == "AlohaSinglePegInsertion":
    rl_config.num_timesteps = 150_000_000
    rl_config.num_evals = 10
    rl_config.unroll_length = 40
    rl_config.num_minibatches = 32
    rl_config.num_updates_per_batch = 8
    rl_config.discounting = 0.97
    rl_config.learning_rate = 3e-4
    rl_config.entropy_cost = 1e-2
    rl_config.num_envs = 1024
    rl_config.batch_size = 512
    rl_config.network_factory.policy_hidden_layer_sizes = (256, 256, 256, 256)
  elif env_name == "PandaOpenCabinet":
    rl_config.num_timesteps = 40_000_000
    rl_config.num_evals = 4
    rl_config.unroll_length = 10
    rl_config.num_minibatches = 32
    rl_config.num_updates_per_batch = 8
    rl_config.discounting = 0.97
    rl_config.learning_rate = 1e-3
    rl_config.entropy_cost = 2e-2
    rl_config.num_envs = 2048
    rl_config.batch_size = 512
    rl_config.reward_scaling= 10.
    rl_config.num_resets_per_eval = 1
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(32, 32, 32, 32),
        value_hidden_layer_sizes=(256, 256, 256, 256, 256),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
  elif env_name == "PandaPickCubeCartesian":
    rl_config.num_timesteps = 5_000_000
    rl_config.num_evals = 5
    rl_config.unroll_length = 10
    rl_config.num_minibatches = 8
    rl_config.num_updates_per_batch = 8
    rl_config.discounting = 0.97
    rl_config.learning_rate = 5.0e-4
    rl_config.entropy_cost = 7.5e-3
    rl_config.num_envs = 1024
    rl_config.batch_size = 256
    rl_config.reward_scaling = 0.1
    rl_config.network_factory.policy_hidden_layer_sizes = (256, 256)
    rl_config.num_resets_per_eval = 1
    rl_config.max_grad_norm = 1.0
  elif env_name.startswith("PandaPickCube"):
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 4
    rl_config.unroll_length = 10
    rl_config.num_minibatches = 32
    rl_config.num_updates_per_batch = 8
    rl_config.discounting = 0.97
    rl_config.learning_rate = 1e-3
    rl_config.entropy_cost = 2e-2
    rl_config.num_envs = 2048
    rl_config.reward_scaling = 10.
    rl_config.batch_size = 512
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(32, 32, 32, 32),
        value_hidden_layer_sizes=(256, 256, 256, 256, 256),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
  elif env_name == "PandaRobotiqPushCube":
    rl_config.num_timesteps = 1_800_000_000
    rl_config.num_evals = 10
    rl_config.unroll_length = 100
    rl_config.num_minibatches = 32
    rl_config.num_updates_per_batch = 8
    rl_config.discounting = 0.994
    rl_config.learning_rate = 6e-4
    rl_config.entropy_cost = 1e-2
    rl_config.num_envs = 8192
    rl_config.batch_size = 512
    rl_config.num_resets_per_eval = 1
    rl_config.num_eval_envs = 32
    rl_config.network_factory.policy_hidden_layer_sizes = (64, 64, 64, 64)
  elif env_name == "LeapCubeRotateZAxis":
    rl_config.num_timesteps = 100_000_000
    rl_config.num_evals = 10
    rl_config.num_minibatches = 32
    rl_config.unroll_length = 40
    rl_config.num_updates_per_batch = 4
    rl_config.discounting = 0.97
    rl_config.learning_rate = 3e-4
    rl_config.entropy_cost = 1e-2
    rl_config.num_envs = 8192
    rl_config.batch_size = 256
    rl_config.num_resets_per_eval = 1
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
  elif env_name == "LeapCubeReorient":
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 20
    rl_config.num_minibatches = 32
    rl_config.unroll_length = 40
    rl_config.num_updates_per_batch = 4
    rl_config.discounting = 0.99
    rl_config.learning_rate = 3e-4
    rl_config.entropy_cost = 1e-2
    rl_config.num_envs = 8192
    rl_config.batch_size = 256
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
    rl_config.num_resets_per_eval = 1
  else:
    raise ValueError(f"Unsupported env: {env_name}")

  return rl_config

def manipulation_td3_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax SAC config for the given environment."""
  
  env_config = manipulation.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=20_000_000,
      num_evals=10,
      reward_scaling=10.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      discounting=0.97,
      learning_rate=1e-3,
      num_envs=128,
      batch_size=8196,
      grad_updates_per_step=8,
      max_replay_size=1048576 * 4,
      min_replay_size=8192, #8192,
      std_min=0.01,
      std_max=0.4,
      policy_noise=0.2, 
      noise_clip=0.5,
      network_factory=config_dict.create(
          q_network_layer_norm=False,
          hidden_layer_sizes=(256, 256, 256, 256, 256),
          policy_obs_key="privileged_state",
          value_obs_key="state",
      ),
  )
  if env_name =="LeapCubeReorient":
    rl_config.action_repeat=3

  return rl_config


def manipulation_tdmpc_config(env_name: str) -> config_dict.ConfigDict:
  """Returns tuned Brax SAC config for the given environment."""
  env_config = dm_control_suite.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=5_000_000,
      batch_size = 256,
      num_envs= 128,
      num_eval_envs=128,
      reward_coef = 0.1,
      value_coef = 0.1,
      consistency_coef = 20,
      entropy_coef = 1e-4,
      rho = 0.5,
      enc_lr_scale = 0.3,
      grad_clip_norm = 20,
      # discount_denom = 5,
      # discount_min = 0.95,
      # discount_max = 0.995,
      max_replay_size=1048576 * 8,
      min_replay_size=8192,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      discounting=0.99,
      learning_rate=1e-3,
      grad_updates_per_step=8,
      tau = 0.01,
      latent_size=512,
      mppi_iterations =6,
      num_samples = 512,
      num_elites = 64,
      policy_prior_samples =24,
      horizon = 3,
      min_plan_std = 0.05,
      max_plan_std = 2,
      temperature = 0.5,
      # actor_mode = "residual",
      # log_std_min = -10,
      # log_std_max = 2,
      # prior_coef = 1.0,
      # scale_threshold = 2.0,
      # awac_lambda = 0.3333,
      # exp_adv_min = 0.1,
      # exp_adv_max = 10.0,
      # num_channels = 32,
      # task_dim = 96,
      network_factory=config_dict.create(
        num_bins=101,
        n_critics=5,
        symlog_min=-10,
        symlog_max=10,
        simnorm_dim = 8,
      ),
  )

  return rl_config
