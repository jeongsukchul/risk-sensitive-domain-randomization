# Copyright 2025 The Brax Authors.
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

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import logger as metric_logger
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from matplotlib.ticker import MaxNLocator
from learning.agents.sampler_ppo.scheduling import GMMScheduler, SchedulerState
import scipy
from agents.sampler_ppo import losses as samplerppo_losses
from agents.sampler_ppo import networks as samplerppo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
from learning.agents.sampler_ppo.distributions import ADRState, DoraemonState, _log_prob_beta_on_box, _unpack_beta, get_adr_sample, plot_adr_density_2d, plot_beta_density_2d, sample_beta_on_box
from learning.helper import make_dir
from learning.module.gmmvi.network import GMMTrainingState
from learning.module.bijx.utils import render_flow_pdf_2d_subplots
from learning.module.target_examples.funnel import Funnel
from learning.module.wrapper.adv_wrapper import wrap_for_adv_training
from learning.module.wrapper.dr_wrapper import wrap_for_dr_training
import learning.module.gmmvi.utils as gmmvi_utils
import numpy as np
import optax
import wandb
import matplotlib.pyplot as plt
from learning.module.wrapper.wrapper import Wrapper, wrap_for_brax_training
from learning.module.wrapper.evaluator import AdvEvaluator, Evaluator, generate_adv_unroll, generate_unroll
from flax import nnx
import imageio
import os
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""

  optimizer_state: optax.OptState
  params: samplerppo_losses.SAMPLERPPONetworkParams
  gmm_training_state :GMMTrainingState
  flow_opt_state : optax.OptState
  flow_state : Any
  autodr_state: ADRState
  doraemon_state: DoraemonState
  normalizer_params: running_statistics.RunningStatisticsState
  env_steps: types.UInt64
  update_steps: jnp.int32
  scheduler_state : SchedulerState

def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
  # brax user code is sometimes ambiguous about weak_type.  in order to
  # avoid extra jit recompilations we strip all weak types from user input
  def f(leaf):
    leaf = jnp.asarray(leaf)
    return leaf.astype(leaf.dtype)

  return jax.tree_util.tree_map(f, tree)


def _validate_madrona_args(
    madrona_backend: bool,
    num_envs: int,
    num_eval_envs: int,
    action_repeat: int,
    eval_env: Optional[envs.Env] = None,
):
  """Validates arguments for Madrona-MJX."""
  if madrona_backend:
    if eval_env:
      raise ValueError("Madrona-MJX doesn't support multiple env instances")
    if num_eval_envs != num_envs:
      raise ValueError('Madrona-MJX requires a fixed batch size')
    if action_repeat != 1:
      raise ValueError(
          "Implement action_repeat using PipelineEnv's _n_frames to avoid"
          ' unnecessary rendering!'
      )
def _random_translate_pixels(
    obs: Mapping[str, jax.Array], key: PRNGKey
) -> Mapping[str, jax.Array]:
  """Apply random translations to B x T x ... pixel observations.

  The same shift is applied across the unroll_length (T) dimension.

  Args:
    obs: a dictionary of observations
    key: a PRNGKey

  Returns:
    A dictionary of observations with translated pixels
  """
  @jax.vmap
  def rt_all_views(
      ub_obs: Mapping[str, jax.Array], key: PRNGKey
  ) -> Mapping[str, jax.Array]:
    # Expects dictionary of unbatched observations.
    def rt_view(
        img: jax.Array, padding: int, key: PRNGKey
    ) -> jax.Array:  # TxHxWxC
      # Randomly translates a set of pixel inputs.
      # Adapted from
      # https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/drq/augmentations.py
      crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
      zero = jnp.zeros((1,), dtype=jnp.int32)
      crop_from = jnp.concatenate([zero, crop_from, zero])
      padded_img = jnp.pad(
          img,
          ((0, 0), (padding, padding), (padding, padding), (0, 0)),
          mode='edge',
      )
      return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

    out = {}
    for k_view, v_view in ub_obs.items():
      if k_view.startswith('pixels/'):
        key, key_shift = jax.random.split(key)
        out[k_view] = rt_view(v_view, 4, key_shift)
    return {**ub_obs, **out}

  bdim = next(iter(obs.items()), None)[1].shape[0]
  keys = jax.random.split(key, bdim)
  obs = rt_all_views(obs, keys)
  return obs

def plot_reward_heatmap(samples, rewards, low, high, bins=80, title="Reward heatmap"):
    """
    samples: (N,2)  [x,y]
    rewards: (N,)
    low/high: (2,) bounds
    """
    s = np.asarray(samples)
    r = np.asarray(rewards).reshape(-1)
    assert s.shape[1] == 2 and s.shape[0] == r.shape[0], (s.shape, r.shape)

    x = s[:, 0]
    y = s[:, 1]

    # Sum of rewards per bin
    sum_r, xedges, yedges = np.histogram2d(
        x, y,
        bins=bins,
        range=[[low[0], high[0]], [low[1], high[1]]],
        weights=r,
    )
    # Count per bin
    cnt, _, _ = np.histogram2d(
        x, y,
        bins=bins,
        range=[[low[0], high[0]], [low[1], high[1]]],
        weights=np.ones_like(r),
    )

    mean_r = sum_r / np.maximum(cnt, 1.0)
    mean_r[cnt == 0] = np.nan  # show empty bins as blank

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(xedges, yedges, mean_r.T, shading="auto")  # note .T
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("mean reward")

    ax.set_xlim(low[0], high[0]); ax.set_ylim(low[1], high[1])
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)

    # optional: overlay a few points for sanity
    idx = np.random.default_rng(0).choice(s.shape[0], size=min(300, s.shape[0]), replace=False)
    ax.scatter(x[idx], y[idx], s=10, alpha=0.4, marker="x", label="samples")

    # legend outside
    fig.subplots_adjust(right=0.78)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    return fig, ax
def _remove_pixels(
    obs: Union[jnp.ndarray, Mapping[str, jax.Array]],
) -> Union[jnp.ndarray, Mapping[str, jax.Array]]:
  """Removes pixel observations from the observation dict."""
  if not isinstance(obs, Mapping):
    return obs
  return {k: v for k, v in obs.items() if not k.startswith('pixels/')}


def train(
    environment: envs.Env,
    num_timesteps: int,
    max_devices_per_host: Optional[int] = None,
    # high-level control flow
    madrona_backend: bool = False,
    augment_pixels: bool = False,
    # environment wrapper
    num_envs:int  = 1024,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    # ppo params
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    normalize_advantage: bool = True,
    network_factory: types.NetworkFactory[
        samplerppo_networks.SAMPLERPPONetworks
    ] = samplerppo_networks.make_samplerppo_networks,
    seed: int = 0,
    # eval
    num_evals: int = 1,
    eval_env: Optional[envs.Env] = None,
    num_eval_envs: int = 1024,
    # training metrics
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    # callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    run_evals: bool = True,
    dr_train_ratio = 1.0,
    use_wandb= False,
    sampler_update_freq=10,
    gamma = 0.5,
    beta = 1.0,
    sampler_choice : str = "NODR", #(NODR, UDR, AutoDR, DORAEMON, FLOW_NS, FLOW_REALNVP, GMM)
    n_sampler_iters : int = 1.,
    min_bound :float  = .8,
    max_bound :float = 110.,
    success_threshold : float = .6, 
    success_rate_condition : float = 0.5,
    work_dir: str= None,
    use_scheduling : bool = False,
    scheduler_lr: float = 0.2,
    scheduler_window_size: int = 20,
    epsilon: float = 0.1,
    start_beta : float =10.,
    end_beta : float = -40.,
    scheduler_mode : str = "linear",
):
  """
  Returns:
    Tuple of (make_policy function, network params, metrics)
  """
  num_eval_envs=4096
  assert batch_size * num_minibatches % num_envs == 0
  _validate_madrona_args(
      madrona_backend, num_envs, num_eval_envs, action_repeat, eval_env
  )
  xt = time.time()
  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d',
      jax.device_count(),
      process_count,
      process_id,
      local_device_count,
      local_devices_to_use,
  )
  device_count = local_devices_to_use * process_count
  print("sgd steps per env step",  batch_size * num_minibatches // num_envs)
  # The number of environment steps executed for every training step.
  env_step_per_training_step = (
      batch_size * unroll_length * num_minibatches * action_repeat
  )
  num_evals_after_init = max(num_evals - 1, 1)
  # The number of training_step calls per training_epoch call.
  # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
  #                                 num_resets_per_eval))

  num_training_steps_per_epoch = np.ceil(
      num_timesteps
      / (
          num_evals_after_init
          * env_step_per_training_step
          * max(num_resets_per_eval, 1)
      )
  ).astype(int)
  num_training_steps = num_training_steps_per_epoch * num_evals_after_init
  print("num_training steps", num_training_steps)
  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  del key
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, key_env, eval_key = jax.random.split(local_key, 3)
  # key_networks should be global, so that networks are initialized the same
  # way for different processes.
  key_policy, key_value, gmm_key = jax.random.split(global_key, 3)
  del global_key

  assert num_envs % device_count == 0
  import copy
  env = copy.deepcopy(environment)
  nominal_dynamics_params= env.nominal_params
  print("num timesteps", num_timesteps)
  print("num_evals", num_evals)
  print("num_trainign_Steps_per_epoch", num_training_steps_per_epoch)
  print("nominal params", nominal_dynamics_params)
  print("dr range", env.dr_range)
  print("sampler update freq", sampler_update_freq)
  save_dir = make_dir(work_dir / "results" / sampler_choice)
  if hasattr(env,'dr_range') :
    low, high = env.dr_range
    dr_mid = (low  + high) / 2.
    dr_scale = dr_train_ratio * (high - low) / 2.
    dr_range_low, dr_range_high = dr_mid - dr_train_ratio*dr_scale, dr_mid + dr_train_ratio*dr_scale
    training_dr_range = (dr_range_low,  dr_range_high)
  else:
    training_dr_range = None
  training_randomization_fn = None

  training_randomization_fn = functools.partial(
        randomization_fn,
        dr_range=training_dr_range,
    )
  # if sampler_choice == "NODR":
  #   env = wrap_for_brax_training(
  #     env,
  #     episode_length=episode_length,
  #     action_repeat=action_repeat,
  #   )
  # else:
  env = wrap_for_adv_training(
    env,
    episode_length=episode_length,
    action_repeat=action_repeat,
    randomization_fn=training_randomization_fn,
    param_size = len(dr_range_low),
    dr_range_low=dr_range_low,
    dr_range_high=dr_range_high,
  )

  key_envs = jax.random.split(key_env, num_envs // process_count)
  key_envs = jnp.reshape(
      key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
  )
  env_state= jax.pmap(env.reset)(key_envs)
  # Discard the batch axes over devices and envs.
  obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)
  print("SAMPLERPPO OBS SIZE", obs_shape)
  normalize_fn = lambda x, y: x
  if normalize_observations:
    normalize_fn = running_statistics.normalize
  samplerppo_network, autodr_update_fn, doraemon_update_fn, init_states= \
    network_factory(
      observation_size = obs_shape,
      action_size= env.action_size, 
      dynamics_param_size=len(dr_range_low), 
      batch_size= num_envs//jax.process_count(),
      num_envs = num_envs//jax.process_count(),
      init_key=gmm_key,
      preprocess_observations_fn=normalize_fn,
      bound_info = training_dr_range,
      success_threshold= success_threshold,
      success_rate_condition= success_rate_condition,
      sampler_choice = sampler_choice,
    )
  init_autodr_state, init_doraemon_state, init_flow_state, init_gmm_state = init_states
  scheduler, init_scheduler_state =  GMMScheduler.create(
        arms=jnp.arange(41)-30, 
        lr=scheduler_lr, 
        window_size=scheduler_window_size,
    )
  
  make_policy = samplerppo_networks.make_inference_fn(samplerppo_network)
  flow_model = nnx.merge(samplerppo_network.flow_network, init_flow_state)
  optimizer = optax.adam(learning_rate=learning_rate)
  flow_optimizer = nnx.Optimizer(
      flow_model,
      optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(1e-3, weight_decay=1e-4), # Adds weight decay for regularization
      ),
      wrt=nnx.Param,
  )
  flow_opt_graphdef,  init_flow_opt_state =  nnx.split(flow_optimizer)
  if max_grad_norm is not None:
    # TODO: Move gradient clipping to `training/gradients.py`.
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=learning_rate),
    )

  loss_fn = functools.partial(
      samplerppo_losses.compute_samplerppo_loss,
      samplerppo_network=samplerppo_network,
      entropy_cost=entropy_cost,
      discounting=discounting,
      reward_scaling=reward_scaling,
      gae_lambda=gae_lambda,
      clipping_epsilon=clipping_epsilon,
      normalize_advantage=normalize_advantage,
  )
  flowloss_fn = functools.partial(
    samplerppo_losses.flow_loss,
    dr_range_high=dr_range_high,
    dr_range_low=dr_range_low,
    gamma=gamma,
    entropy_update= False,
  )
  gmm_update_fn = samplerppo_losses.make_gmm_update(samplerppo_network.gmm_network)
  gradient_update_fn = gradients.gradient_update_fn(
      loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
  )
  @jax.jit
  def flow_update_fn(graphdef, state, opt_state, dynamics_params, prev_sample, prev_logp, target_lnpdf, rng):
    model = nnx.merge(graphdef, state)
    optimizer = nnx.merge(flow_opt_graphdef, opt_state)

    (loss, others), grads = nnx.value_and_grad(flowloss_fn, has_aux=True)(\
      model,
      dynamics_params,
      prev_sample,
      prev_logp,
      target_lnpdf,
      rng,
    )
    optimizer.update(grads=grads, model=model)
    _, new_state = nnx.split(model)
    _, new_opt_state = nnx.split(optimizer)
    _, state = nnx.split(model)
    return (loss, new_state, new_opt_state), others
  metrics_aggregator = metric_logger.EpisodeMetricsLogger(
      steps_between_logging=training_metrics_steps
      or env_step_per_training_step,
      progress_fn=progress_fn,
  )

  def minibatch_step(
      carry,
      data,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, key = carry
    key, key_loss = jax.random.split(key, 2)
    (_, metrics), params, optimizer_state = gradient_update_fn(
        params,
        normalizer_params,
        data,
        key_loss,
        optimizer_state=optimizer_state,
    )
    
    return (optimizer_state, params, key), metrics

  def sgd_step(
      carry,
      unused_t,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, key = carry
    key, key_perm, key_grad = jax.random.split(key, 3)

    if augment_pixels:
      key, key_rt = jax.random.split(key)
      r_translate = functools.partial(_random_translate_pixels, key=key_rt)
      data = types.Transition(
          observation=r_translate(data.observation),
          action=data.action,
          reward=data.reward,
          discount=data.discount,
          next_observation=r_translate(data.next_observation),
          extras=data.extras,
      )
    def convert_data(x: jnp.ndarray):
      x = jax.random.permutation(key_perm, x)
      x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
      return x

    shuffled_data = jax.tree_util.tree_map(convert_data, data)
    (optimizer_state, params, _), metrics = jax.lax.scan(
        functools.partial(minibatch_step, normalizer_params=normalizer_params),
        (optimizer_state, params, key_grad),
        shuffled_data,
        length=num_minibatches,
    )
    return (optimizer_state, params, key), metrics
  
  def get_experience(
      training_state,
      state,
      dynamics_params,
      key_generate_unroll,
      unroll_length=10,
      num_episodes=1,
  ):
    policy = make_policy((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))
    def f(carry, unused_t):
      current_state, current_key = carry
      current_key, next_key = jax.random.split(current_key)
      if dynamics_params is None:
        next_state, data = generate_unroll(
            env,
            current_state,
            policy,
            current_key,
            unroll_length,
            extra_fields=('truncation', 'episode_metrics', 'episode_done'),
        )
      else:
        next_state, data = generate_adv_unroll(
            env,
            current_state,
            dynamics_params,
            policy,
            current_key,
            unroll_length,
            extra_fields=('truncation', 'episode_metrics', 'episode_done'),
        )
      return (next_state, next_key), data
    (state, _), data = jax.lax.scan(
        f,
        (state, key_generate_unroll),
        (),
        length=num_episodes
    )
    return state, data
  def training_step(
      carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
  ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
    training_state, state, cumulated_values, key = carry
    key_sgd, key_generate_unroll, param_key, key_update, new_key = jax.random.split(key, 5)
    if sampler_choice=="NODR":
      dynamics_params = jnp.ones((num_envs// jax.process_count(), len(dr_range_low))) * nominal_dynamics_params[None, :]
    elif sampler_choice=="UDR" or sampler_choice=="EPOpt":
        dynamics_params = jax.random.uniform(param_key, shape=(num_envs //jax.process_count(), len(dr_range_low)), minval=dr_range_low, maxval=dr_range_high)
    elif sampler_choice=="AutoDR":
        dynamics_params = get_adr_sample(training_state.autodr_state, 
                                num_envs// jax.process_count(),  param_key)
    elif sampler_choice == "DORAEMON":
        a, b = _unpack_beta(training_state.doraemon_state.x_opt, len(dr_range_low), min_bound, max_bound)
        dynamics_params = sample_beta_on_box(param_key, a, b, dr_range_low, dr_range_high, num_envs //jax.process_count())
    elif "FLOW" in sampler_choice:
      param_key, param_key2= jax.random.split(param_key)
      flow_model = nnx.merge(samplerppo_network.flow_network, training_state.flow_state)
      dynamics_params_sampler, logp = flow_model.sample((num_envs // jax.process_count() // local_devices_to_use,),\
                                                    rng=param_key)
      data = get_experience(training_state, state, dynamics_params_sampler,\
            key_generate_unroll, unroll_length, batch_size * num_minibatches // num_envs,)[1]
      rewards = data.reward
      
      dynamics_params = jax.random.uniform(param_key2, shape=(num_envs //jax.process_count(), len(dr_range_low)), minval=dr_range_low, maxval=dr_range_high)

    elif sampler_choice == "GMM":
        dynamics_params, mapping = samplerppo_network.gmm_network.model.sample(\
          training_state.gmm_training_state.model_state.gmm_state, param_key, num_envs //jax.process_count())
        # dynamics_params_for_training, mapping = samplerppo_network.gmm_network.sample_selector.select_samples(\
        #       training_state.gmm_training_state.model_state, param_key)
    else:
      raise ValueError("No Sampler Available")
    # #for debugging
    # dynamics_params = (dr_range_low + dr_range_high)/2 +\
    #   jax.random.normal(key=param_key, shape=(num_envs//jax.process_count(),len(dr_range_low)))\
    #     * (dr_range_high - dr_range_low)/100
    # dynamics_params = jnp.clip(dynamics_params, dr_range_low, dr_range_high)

    state, data = get_experience(training_state, state, dynamics_params,\
            key_generate_unroll, unroll_length, batch_size * num_minibatches // num_envs if sampler_choice!="EPOpt" else int(batch_size * num_minibatches/epsilon //num_envs) ,)
    obs_for_approx = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), data.observation)
    value_approx = samplerppo_network.value_network.apply(training_state.normalizer_params, training_state.params.value, obs_for_approx)
    print("value approx", value_approx)
    value_approx =value_approx.mean(0) / episode_length

    if "FLOW" not in sampler_choice:
      rewards = data.reward     #(K, L, B)
    # Have leading dimensions (batch_size * num_minibatches, unroll_length)
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
    data = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
    )      # [B * num_minibatches, unroll_length]
    if sampler_choice=="EPOpt":
      print("data size", data)

      def filter_datas(batched_data):
        returns = jnp.sum(batched_data.reward * batched_data.discount, axis=1)  # [B]
        k = batch_size*num_minibatches
        _, indicies = jax.lax.top_k(-returns, k)
        filtered_data = jax.tree_util.tree_map(
            lambda x: x[indicies],
            batched_data,
        )
        return filtered_data
      data = filter_datas(data)
      print("filtered data size", data)
    assert data.discount.shape[1:] == (unroll_length,)
    if log_training_metrics:  # log unroll metrics
      jax.debug.callback(
          metrics_aggregator.update_episode_metrics,
          data.extras['state_extras']['episode_metrics'],
          data.extras['state_extras']['episode_done'],
      )

    # Update normalization params and normalize observations.
    normalizer_params = running_statistics.update(
        training_state.normalizer_params,
        _remove_pixels(data.observation),
        pmap_axis_name=_PMAP_AXIS_NAME,
    )

    (optimizer_state, params, _), metrics = jax.lax.scan(
        functools.partial(
            sgd_step, data=data, normalizer_params=normalizer_params
        ),
        (training_state.optimizer_state, training_state.params, key_sgd),
        (),
        length=num_updates_per_batch,
    )
    # if sampler_choice != "GMM":
    values = rewards.mean(axis=(0,1)) if sampler_choice!="EPOpt" else 0#+ bootstrap_value 
    cumulated_values += values
    # else:
    #   cumulated_values +=value_approx
    # For Debuggin GMM
    # target = Funnel(dim=2, sample_bounds=[-30, 30])
    # target_logprob = jax.vmap(target.log_prob)
    # target_pdf = target_logprob(dynamics_params - (dr_range_low + dr_range_high)/2) # [num_envs]
    # scheduler update
    if use_scheduling:
      scheduler_key, key = jax.random.split(key)
      def update_scheduler(scheduler_state, _cummulated_values):
        sorted_values = jnp.sort(_cummulated_values)
        N = _cummulated_values.shape[0]
        k20 = int(N* .2)
        CVaR20 = sorted_values[:k20].mean()
        feedback = CVaR20 - scheduler_state.prev_cvar
        scheduler_state = scheduler_state.replace(prev_cvar = CVaR20)
        return scheduler.update_dists(scheduler_state, feedback)
      scheduler_state = jax.lax.cond(training_state.update_steps % sampler_update_freq==0, \
                lambda: update_scheduler(training_state.scheduler_state, cumulated_values), \
                 lambda: training_state.scheduler_state )
      scheduler_state, _beta = scheduler.sample(scheduler_state, scheduler_key)
      # linear schedule
      if scheduler_mode=="linear":
        _beta = start_beta - (start_beta - end_beta) * (training_state.update_steps / num_training_steps)
      elif scheduler_mode=="exp":
        k = 3.
        # exponential schedule
        _beta = start_beta - (start_beta - end_beta) * (1 - jnp.exp(-k * training_state.update_steps / num_training_steps))/ (1  - jnp.exp(-training_state.update_steps))
    else:
      scheduler_state = training_state.scheduler_state
      _beta = beta

  
    gmm_training_state = training_state.gmm_training_state
    flow_state = training_state.flow_state
    flow_opt_state= training_state.flow_opt_state
    autodr_state = training_state.autodr_state
    doraemon_state = training_state.doraemon_state
    def update_flow(flow_state, flow_opt_state):
      target_lnpdf = _beta * cumulated_values/sampler_update_freq
      flow_model = nnx.merge(samplerppo_network.flow_network, flow_state)
      prev_sample, prev_logq = flow_model.sample(
        (10000,), rng=param_key)
      def body(carry, _):
        fs, fos, prev_sample_carry, prev_logp_carry = carry
        (loss, fs_n, fos_n), (metric, current_sample, current_logq) = flow_update_fn(
            samplerppo_network.flow_network, fs, fos,
            dynamics_params_sampler,
            prev_sample_carry,
            prev_logp_carry,
            target_lnpdf,
            key_update,
        )
        return (fs_n, fos_n, current_sample, current_logq), None
      (flow_state_new, flow_opt_state_new, _, _), _  = jax.lax.scan(body, (flow_state, flow_opt_state, prev_sample, prev_logq), (), length=n_sampler_iters)
      return (flow_state_new, flow_opt_state_new, 1.)
    def update_adr(autodr_state, returns):
      return (autodr_update_fn(autodr_state, dynamics_params, returns, key_update), 1.)
    def update_doraemon(doraemon_state, returns):
      return (doraemon_update_fn(doraemon_state, dynamics_params, returns), 1.)

    def update_gmm(gts):
      target_lnpdf = _beta * cumulated_values/sampler_update_freq
      new_sample_db_state = samplerppo_network.gmm_network.sample_selector.save_samples(gmm_training_state.model_state, \
                    gts.sample_db_state, dynamics_params, target_lnpdf, \
                      jnp.zeros_like(dynamics_params), mapping)
      new_gmm_training_state = gmm_training_state._replace(sample_db_state=new_sample_db_state)
      new_gmm_training_state = gmm_update_fn(new_gmm_training_state, key_update)

      return new_gmm_training_state, 1.
    if sampler_choice=="NODR" or sampler_choice=="UDR" or sampler_choice=='EPOpt':
        update_signal = jax.lax.cond(training_state.update_steps % sampler_update_freq==0, \
                lambda: 1., \
                 lambda: 0., )
    elif sampler_choice=='AutoDR':
      autodr_state, update_signal = jax.lax.cond(training_state.update_steps % sampler_update_freq==0, \
            lambda: update_adr(training_state.autodr_state, cumulated_values/sampler_update_freq), \
              lambda: (training_state.autodr_state, 0.) )
    elif sampler_choice=='DORAEMON':
      doraemon_state, update_signal = jax.lax.cond(training_state.update_steps % sampler_update_freq==0, \
            lambda: update_doraemon(training_state.doraemon_state, cumulated_values/sampler_update_freq), \
              lambda: (training_state.doraemon_state, 0.) )
    elif "FLOW" in sampler_choice:
        flow_state, flow_opt_state, update_signal = jax.lax.cond(training_state.update_steps % sampler_update_freq==0, \
                lambda: update_flow(training_state.flow_state, training_state.flow_opt_state), \
                 lambda: (training_state.flow_state, training_state.flow_opt_state, 0.) )
    elif sampler_choice=="GMM":
      gmm_training_state, update_signal = jax.lax.cond(training_state.update_steps % sampler_update_freq==0, \
                  lambda: update_gmm(gmm_training_state), \
                  lambda: (training_state.gmm_training_state, 0.) )
    else:
      raise ValueError("No Sampler!")
    # update_signal=1
    print("beta", _beta)
    metrics.update({
      'target_pdf_min': update_signal* cumulated_values.min(),
      'target_pdf_max': update_signal* cumulated_values.max(),
      'target_pdf_mean': update_signal* cumulated_values.mean(),
      'target_pdf_q25': update_signal* jnp.quantile(cumulated_values, .25),
      'target_pdf_q50': update_signal* jnp.quantile(cumulated_values, .50),
      'target_pdf_q75': update_signal* jnp.quantile(cumulated_values, .75),
      'target_pdf_std': update_signal* cumulated_values.std(),
      'beta': _beta,
    })
    new_training_state = TrainingState(
        optimizer_state=optimizer_state,
        params=params,
        gmm_training_state=gmm_training_state,
        flow_state=flow_state,
        flow_opt_state= flow_opt_state,
        autodr_state=autodr_state,
        doraemon_state=doraemon_state,
        scheduler_state = scheduler_state,
        normalizer_params=normalizer_params,
        env_steps=training_state.env_steps + env_step_per_training_step,
        update_steps = training_state.update_steps + 1,
    )
    cumulated_values = (1-update_signal) * cumulated_values
    return (new_training_state, state, cumulated_values, new_key), metrics

  def training_epoch(
      training_state: TrainingState, state: envs.State, key: PRNGKey
  ) -> Tuple[TrainingState, envs.State, Metrics]:
    (training_state, state, _, _), loss_metrics = jax.lax.scan(
        training_step,
        (training_state, state, jnp.zeros(num_envs//jax.process_count()), key),
        (),
        length=num_training_steps_per_epoch,
    )
    loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
    # for k, v in loss_metrics.items():
    #   if 'target_pdf' in k:
    #     loss_metrics[k] *=sampler_update_freq
    return training_state, state, loss_metrics

  training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      training_state: TrainingState, env_state: envs.State, key: PRNGKey
  ) -> Tuple[TrainingState, envs.State, Metrics]:
    nonlocal training_walltime
    t = time.time()
    training_state, env_state = _strip_weak_type((training_state, env_state))
    result = training_epoch(training_state, env_state, key)
    training_state, env_state, metrics = _strip_weak_type(result)

    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (
        num_training_steps_per_epoch
        * env_step_per_training_step
        * max(num_resets_per_eval, 1)
    ) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()},
    }
    return training_state, env_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade
  def evaluation_on_current_occupancy(
      training_state: TrainingState,
      env_state: envs.State,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, PRNGKey]:
    # shape = np.sqrt(num_envs).astype(int)
    value_apply = samplerppo_network.value_network.apply
    x, y = jnp.meshgrid(jnp.linspace(dr_range_low[0], dr_range_high[0], 32),\
                          jnp.linspace(dr_range_low[1], dr_range_high[1], 32))
    dynamics_params_grid = jnp.c_[x.ravel(), y.ravel()]
    # print("state in occupancy measure", env_state.obs)
    def f(carry, unused):
      key, dynamics_params_grid = carry
      key, new_key = jax.random.split(key)

      _, datas = get_experience(
          training_state,
          env_state,
          dynamics_params_grid,
          key,
          unroll_length,
          batch_size * num_minibatches // num_envs,# * sampler_update_freq,
      )
      # rewards = datas.reward #* reward_scaling
      # terminal_obs = jax.tree_util.tree_map(lambda x: x[:, -1], datas.next_observation)
      first_obs = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), datas.next_observation)
      # first_obs = jax.tree_util.tree_map(lambda x: x[:100], first_obs)
      # bootstrap_value = value_apply(training_state.normalizer_params, training_state.params.value, terminal_obs)
      # values = rewards.mean(axis=(0,1))# + bootstrap_value
      values =  value_apply(training_state.normalizer_params, training_state.params.value, first_obs)
      discounts = jnp.power(discounting, jnp.arange(values.shape[0]))
      # print("values", values)
      # print('discounst', discounts)
      # values= (discounts[...,None]*values).mean(0)
      values= values.mean(0)
      # values =  value_apply(training_state.normalizer_params, training_state.params.value, datas.next_observation)
      # values = values.mean(0)/ episode_length
      # values = jax.nn.log_softmax((values - values.mean())/ values.std()+1e-6)
      # vs, advantages = jax.vmap(gae_fn)(truncation, termination, rewards, baseline, bootstrap_value)
      return (new_key, dynamics_params_grid), values#pdf_values
    return jax.lax.scan(
        f,
        (key, dynamics_params_grid), (), length=10#xs=dynamics_params_grid,
    )[1]
  evaluation_on_current_occupancy = jax.pmap(
      evaluation_on_current_occupancy, axis_name=_PMAP_AXIS_NAME
  )
  # Initialize model params and training state.
  init_params = samplerppo_losses.SAMPLERPPONetworkParams(
      policy=samplerppo_network.policy_network.init(key_policy),
      value=samplerppo_network.value_network.init(key_value),
  )

  obs_shape = jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
  )
  print("SAMPLERPPO OBS SHAPE2", obs_shape)
  training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
      optimizer_state=optimizer.init(init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
      params=init_params,
      gmm_training_state=init_gmm_state,
      flow_state = init_flow_state,
      flow_opt_state = init_flow_opt_state,
      autodr_state = init_autodr_state,
      doraemon_state= init_doraemon_state,
      scheduler_state= init_scheduler_state,
      normalizer_params=running_statistics.init_state(
          _remove_pixels(obs_shape)
      ),
      env_steps=types.UInt64(hi=0, lo=0),
      update_steps=0,
  )

  if num_timesteps == 0:
    return (
        make_policy,
        (
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        ),
        {},
    )

  #pmaped
  training_state = jax.device_put_replicated(
      training_state, jax.local_devices()[:local_devices_to_use]
  )

  eval_env = copy.deepcopy(environment)
  v_randomization_fn=None
  v_randomization_fn = functools.partial(
    randomization_fn,
    dr_range=eval_env.dr_range,
  )
  eval_env = wrap_for_adv_training(
      eval_env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=v_randomization_fn,
      param_size = len(dr_range_low),
      dr_range_low=dr_range_low,
      dr_range_high=dr_range_high,
  )  # pytype: disable=wrong-keyword-args

  evaluator = AdvEvaluator(
      eval_env,
      functools.partial(make_policy, deterministic=True),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key,
  )


  training_metrics = {}
  training_walltime = 0
  current_step = 0
  evaluation_key, local_key = jax.random.split(local_key)
  evaluation_frames = []
  autodr_frames = []
  doraemon_frames = []
  flow_frames = []
  gmm_frames = []
  gmm_training_frames = []
  occupancy_frames = []
  # Run initial eval
  print("-------------------------len parameter--------------------------------------------", len(dr_range_low))
  metrics = {}
  if process_id == 0 and num_evals > 1 and run_evals and len(dr_range_low)>2:
    eval_key, local_key = jax.random.split(local_key)
    dynamics_params_grid = jax.random.uniform(eval_key, shape=(num_eval_envs, len(dr_range_low)), minval=dr_range_low, maxval=dr_range_high)
    metrics, reward_1d, epi_length = evaluator.run_evaluation(
        _unpmap((
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        )),
        dynamics_params=dynamics_params_grid,
        training_metrics=metrics,
        num_eval_seeds=10,
        success_threshold=success_threshold,
    )
    logging.info(metrics)
    progress_fn(0, metrics)
  elif process_id == 0 and num_evals > 1 and run_evals and len(dr_range_low)==2:
    x, y = jnp.meshgrid(jnp.linspace(dr_range_low[0], dr_range_high[0], 64),\
                              jnp.linspace(dr_range_low[1], dr_range_high[1], 64))
    dynamics_params_grid = jnp.c_[x.ravel(), y.ravel()]
    metrics, reward_1d,_ = evaluator.run_evaluation(
        _unpmap((
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        )),
        dynamics_params=dynamics_params_grid,
        training_metrics={},
        num_eval_seeds=10,
        success_threshold=success_threshold,
    )
    eval_fig = plt.figure()
    reward_2d = reward_1d.reshape(x.shape)
    # vmin, vmax = 0, 1000
    # levels = np.linspace(vmin, vmax, 21)  # 21 levels = 20 color intervals
    ctf = plt.contourf(x, y, reward_2d, levels=20, cmap='viridis')
    cbar = eval_fig.colorbar(ctf)
    eval_fig.suptitle(f"Evaluation on Each Params [Step={int(current_step)}]")
    eval_fig.tight_layout()
    eval_fig.canvas.draw()
    evaluation_frames.append(np.asarray(eval_fig.canvas.buffer_rgba())[...,:3])
    if use_wandb:
      wandb.log({
        'eval on each params' : wandb.Image(eval_fig)
      }, step=int(0))
    sample_key, local_key = jax.random.split(local_key,2)
    if sampler_choice=="AutoDR":
      fig, ax = plt.subplots()
      ax = plot_adr_density_2d(
                adr_state=_unpmap(training_state.autodr_state),
                domain_low=low,
                domain_high=high,
                key=sample_key,
                dim_x=0,
                dim_y=1,
                ax=ax,
            )
      fig.tight_layout()
      fig.canvas.draw()
      autodr_frames.append(np.asarray(fig.canvas.buffer_rgba())[...,:3])
      
      if use_wandb:
        wandb.log(
            {f"Sampler Heatmap": wandb.Image(fig)},
            step=int(current_step),
        )
    elif sampler_choice=='DORAEMON':
      fig, ax = plt.subplots()
      # _, reward_1d = evaluator.run_evaluation(
      #   _unpmap((
      #       training_state.normalizer_params,
      #       training_state.params.policy,
      #       training_state.params.value,
      #   )),
      #     dynamics_params=dynamics_params_grid,
      #     training_metrics={},
      #     num_eval_seeds=10,
      # )
      a, b = _unpack_beta(_unpmap(training_state.doraemon_state.x_opt), len(dr_range_low), min_bound, max_bound)
      
      dynamics_params_doraemon = sample_beta_on_box(sample_key, a, b, low, high, num_envs //jax.process_count())
      ax = plot_beta_density_2d(
                _unpmap(training_state.doraemon_state), low, high, len(low), 0.8, 110.,
                title=f"Doraemon Result [step={int(current_step)}]",
                contexts=dynamics_params_doraemon, success_threshold=success_threshold,
                ax=ax,
            )
      fig.tight_layout()
      fig.canvas.draw()
      doraemon_frames.append(np.asarray(fig.canvas.buffer_rgba())[...,:3])

      if use_wandb:
        wandb.log(
            {f"Sampler Heatmap": wandb.Image(fig)},
            step=int(current_step),
        )

    elif "FLOW" in sampler_choice:
      flow_model = nnx.merge(samplerppo_network.flow_network, _unpmap(training_state.flow_state))
      samples, _ = flow_model.sample( 
        (num_envs // jax.process_count() // local_devices_to_use,),
        sample_key,
      )
      fig, ax = render_flow_pdf_2d_subplots(
        log_prob_fn=lambda x : flow_model.log_density(x=x),
        low=dr_range_low,
        high=dr_range_high,
        samples = samples,
        training_step=current_step,
        use_wandb=use_wandb,
        suptitle=f"Flow results with beta={beta} [step={int(current_step)}]"
      )
      fig.tight_layout()
      fig.canvas.draw()
      flow_frames.append(np.asarray(fig.canvas.buffer_rgba())[...,:3])
    elif sampler_choice=="GMM":
      sample_key1, sample_key2, unroll_key,  local_key = jax.random.split(local_key, 4)
      samples, _ = samplerppo_network.gmm_network.sample_selector.select_samples(\
              _unpmap(training_state.gmm_training_state.model_state), sample_key1)
      data_for_gmm = get_experience(_unpmap(training_state), _unpmap(env_state), samples,\
            unroll_key, unroll_length, batch_size * num_minibatches // num_envs)[1]
      rewards = data_for_gmm.reward
      rewards = rewards.mean(axis=(0,1))
      target_lnpdf = beta * rewards
      # training_fig, _ = plot_reward_heatmap(samples, \
      #       target_lnpdf, dr_range_low, dr_range_high, bins=80, \
      #       title = f"Training Heatmap with beta={beta} [step={int(current_step)}]")
      # training_fig.tight_layout()
      # training_fig.canvas.draw()
      # gmm_training_frames.append(np.asarray(training_fig.canvas.buffer_rgba())[...,:3])
      # if use_wandb:
      #   wandb.log(
      #           {"Training Heatmap" :wandb.Image(training_fig)},
      #           step=int(current_step),
      #       )
      eval_samples = samplerppo_network.gmm_network.model.sample(_unpmap(\
        training_state.gmm_training_state.model_state.gmm_state), sample_key2, 2**14)[0]
      
      log_prob_fn = jax.vmap(functools.partial(samplerppo_network.gmm_network.model.log_density,\
                  gmm_state=_unpmap(training_state.gmm_training_state.model_state.gmm_state)))
      model_fig, model_fig_raw = gmmvi_utils.visualise(
        log_prob_fn,
        dr_range_low,
        dr_range_high,
        # samples,
        eval_samples=eval_samples,
        bijector_log_prob=jax.vmap(samplerppo_network.gmm_network.model.bijector_log_prob)
      )
      model_fig.tight_layout()
      model_fig.canvas.draw()
      gmm_frames.append(np.asarray(model_fig.canvas.buffer_rgba())[...,:3])
      if use_wandb:
        wandb.log(
                {"Sampler Heatmap" :wandb.Image(model_fig)},
                step=int(current_step),
            )
    evaluation_key = jax.random.split(evaluation_key, local_devices_to_use)
    rewards = evaluation_on_current_occupancy(
        training_state, env_state, evaluation_key
    )
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), rewards)
    rewards = rewards.mean((0,1)).squeeze()
    x, y = jnp.meshgrid(jnp.linspace(dr_range_low[0], dr_range_high[0], 32),\
                          jnp.linspace(dr_range_low[1], dr_range_high[1], 32))
    target_lnpdfs = beta* rewards
    target_lnpdfs = jnp.reshape(target_lnpdfs, x.shape)
    target_fig = plt.figure()
    
    ctf = plt.contourf(x, y, target_lnpdfs, levels=20, cmap='viridis')
    cbar = target_fig.colorbar(ctf)
    target_fig.suptitle(f"target log prob on current occupancy [step={current_step}]")
    target_fig.tight_layout()
    target_fig.canvas.draw()
    occupancy_frames.append(np.asarray(target_fig.canvas.buffer_rgba())[...,:3])
    if use_wandb:
      wandb.log({
        'target log prob on current occupancy with returns' : wandb.Image(target_fig)
      }, step=0)
    logging.info(metrics)
    progress_fn(0, metrics)
  # Run initial policy_params_fn.
  params = _unpmap((
      training_state.normalizer_params,
      training_state.params.policy,
      training_state.params.value,
  ))
  policy_params_fn(current_step, make_policy, params)
    
    
  for it in range(num_evals_after_init):
    logging.info('starting iteration %s %s', it, time.time() - xt)

    for _ in range(max(num_resets_per_eval, 1)):
      # optimization
      epoch_key, local_key = jax.random.split(local_key)
      epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
      (training_state, env_state, training_metrics) = (
          training_epoch_with_timing(training_state, env_state, epoch_keys)
      )
      current_step = int(_unpmap(training_state.env_steps))

      key_envs = jax.vmap(
          lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
      )(key_envs, key_envs.shape[1])
      # TODO: move extra reset logic to the AutoResetWrapper.
      env_state = jax.pmap(env.reset)(key_envs) if num_resets_per_eval > 0 else env_state

    if process_id != 0:
      continue

    # Process id == 0.
    params = _unpmap((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))

    policy_params_fn(current_step, make_policy, params)

    if num_evals > 0:
      metrics = training_metrics
      
      if run_evals:
        metric_key, local_key = jax.random.split(local_key)
        if sampler_choice=="DORAEMON" or "FLOW" in sampler_choice or sampler_choice=="GMM" or sampler_choice=="AutoDR":
          num_samples_bench = num_eval_envs
          if sampler_choice=="DORAEMON":
            a, b = _unpack_beta(_unpmap(training_state.doraemon_state.x_opt), len(dr_range_low), min_bound, max_bound)
            samples = sample_beta_on_box(metric_key, a, b, low, high, num_samples_bench)
            logq = _log_prob_beta_on_box(a, b, low, high, samples)
          elif "FLOW" in sampler_choice:
            samples, logq = flow_model.sample( (num_samples_bench,), metric_key)
          elif sampler_choice=="GMM":
            samples, logq = samplerppo_network.gmm_network.model.sample(_unpmap(\
              training_state.gmm_training_state.model_state.gmm_state), metric_key, num_samples_bench)
          elif sampler_choice == "AutoDR":
            samples = get_adr_sample(_unpmap(training_state.autodr_state), num_samples_bench, metric_key)  #
            np.save(os.path.join(save_dir, f"current_range_{current_step}.npy"), 
                    np.array([_unpmap(training_state.autodr_state).current_low, _unpmap(training_state.autodr_state).current_high]))
          np.save(os.path.join(save_dir, f"samples_in_sampler_{current_step}.npy"), samples)

          _, reward_sampler, epi_length = evaluator.run_evaluation(
              _unpmap((
                  training_state.normalizer_params,
                  training_state.params.policy,
                  training_state.params.value,
              )),
              dynamics_params=samples,
              training_metrics=metrics,
              num_eval_seeds=10,
              success_threshold=success_threshold,
          )
          _beta = 1 if sampler_choice=="DORAEMON" else beta
          target_lnpdf = _beta* reward_sampler/epi_length
          if sampler_choice!="AutoDR":
            np.save(os.path.join(save_dir, f"logq_in_sampler_{current_step}.npy"), logq)
            log_ratio = target_lnpdf - logq
            valid_mask = jnp.isfinite(log_ratio)
            safe_log_ratio = jnp.where(valid_mask, log_ratio, -1e20)
            valid_count = jnp.sum(valid_mask)
            # Prevent div-by-zero if all are NaN
            safe_denom = jnp.maximum(valid_count, 1.0) 
            elbo = jnp.sum(jnp.where(valid_mask, log_ratio, 0.0)) / safe_denom
            metrics['eval/elbo'] = elbo
            weights = jnp.where(valid_mask, jnp.exp(safe_log_ratio - jnp.max(safe_log_ratio)), 0.)
            metrics['eval/reverse_ess'] = jnp.sum(weights)**2 / jnp.maximum(num_samples_bench * jnp.sum(weights**2), 1e-10)
            metrics['eval/ln_z'] = (jnp.log(weights.sum()) + jnp.max(safe_log_ratio)) - jnp.log(num_samples_bench)
        if len(dr_range_low)>2:
          rewards = []
          for i in range(4): # 16384 envs.
            eval_key, local_key = jax.random.split(local_key)
            dynamics_params_grid = jax.random.uniform(eval_key, shape=(num_eval_envs, len(dr_range_low)), minval=dr_range_low, maxval=dr_range_high)
            metrics, _reward_1d, ep_length = evaluator.run_evaluation(
                _unpmap((
                    training_state.normalizer_params,
                    training_state.params.policy,
                    training_state.params.value,
                )),
                dynamics_params=dynamics_params_grid,
                training_metrics=metrics,
                num_eval_seeds=10,
                success_threshold=success_threshold,
            )
            rewards.append(_reward_1d)
          rewards = jnp.stack(rewards, axis=-1).reshape(-1)
          reward_1d= rewards
          N = rewards.shape[0]
          k20 = int(N* .2)
          k10 = int(N* .1)
          sorted_rewards = np.sort(rewards)
          metrics['eval/episode_reward_mean'] = np.mean(rewards)
          metrics['eval/episode_reward_p12'] = np.percentile(rewards,12.5)
          metrics['eval/episode_reward_p25'] = np.percentile(rewards,25)
          metrics['eval/episode_reward_p75'] = np.percentile(rewards,75)
          metrics['eval/episode_reward_std'] = np.std(rewards)
          metrics['eval/episode_reward_min'] = np.min(rewards)
          metrics['eval/episode_reward_max'] = np.max(rewards)
          metrics['eval/episode_reward_iqm'] = scipy.stats.trim_mean(rewards, proportiontocut=0.25, axis=None)
          metrics['eval/episode_reward_CVaR20'] = np.mean(sorted_rewards[:k20])
          metrics['eval/episode_reward_CVaR10'] = np.mean(sorted_rewards[:k10])
      if run_evals and len(dr_range_low)==2:
        x, y = jnp.meshgrid(jnp.linspace(dr_range_low[0], dr_range_high[0], 64),\
                              jnp.linspace(dr_range_low[1], dr_range_high[1], 64))
        dynamics_params_grid = jnp.c_[x.ravel(), y.ravel()]
        metrics, reward_1d, _ = evaluator.run_evaluation(
            _unpmap((
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            )),
            dynamics_params=dynamics_params_grid,
            training_metrics=metrics,
            num_eval_seeds=10,
            success_threshold=success_threshold,
        )
        print("reward_1d", reward_1d.shape)
        eval_fig = plt.figure()
        reward_2d = reward_1d.reshape(x.shape)
        # vmin, vmax = 0, 1000
        # levels = np.linspace(vmin, vmax, 21)  # 21 levels = 20 color intervals
        ctf = plt.contourf(x, y, reward_2d, levels=20, cmap='viridis')
        cbar = eval_fig.colorbar(ctf)
        eval_fig.suptitle(f"Evaluation on Each Params [Step={int(current_step)}]")
        eval_fig.tight_layout()
        eval_fig.canvas.draw()
        evaluation_frames.append(np.asarray(eval_fig.canvas.buffer_rgba())[...,:3])
        if use_wandb:
          wandb.log({
            'eval on each params' : wandb.Image(eval_fig)
          }, step=int(current_step))
        if sampler_choice=="AutoDR":
          fig, ax = plt.subplots()
          ax = plot_adr_density_2d(
                    adr_state=_unpmap(training_state.autodr_state),
                    domain_low=low,
                    domain_high=high,
                    key=sample_key,
                    dim_x=0,
                    dim_y=1,
                    ax=ax,
                    step=current_step,
                )
          if use_wandb:
            wandb.log(
                {"Sampler Heatmap": wandb.Image(fig)},
                step=int(current_step),
            )
          fig.tight_layout()
          fig.canvas.draw()
          autodr_frames.append(np.asarray(fig.canvas.buffer_rgba())[...,:3])
        elif sampler_choice=='DORAEMON':
          fig, ax = plt.subplots()
          # _, reward_1d = evaluator.run_evaluation(
          #   _unpmap((
          #       training_state.normalizer_params,
          #       training_state.params.policy,
          #       training_state.params.value,
          #   )),
          #     dynamics_params=dynamics_params_grid,
          #     training_metrics={},
          #     num_eval_seeds=10,
          # )
          a, b = _unpack_beta(_unpmap(training_state.doraemon_state.x_opt), len(dr_range_low), min_bound, max_bound)
          
          dynamics_params_doraemon = sample_beta_on_box(sample_key, a, b, low, high, num_envs //jax.process_count())
          ax = plot_beta_density_2d(
                    _unpmap(training_state.doraemon_state), low, high, len(low), 0.8, 110.,
                    title=f"Doraemon Result [step={int(current_step)}]",
                    contexts=dynamics_params_doraemon, success_threshold=success_threshold,
                    ax=ax,
                )
        
          if use_wandb:
            wandb.log(
                {"Sampler Heatmap": wandb.Image(fig)},
                step=int(current_step),
            )
          fig.tight_layout()
          fig.canvas.draw()
          doraemon_frames.append(np.asarray(fig.canvas.buffer_rgba())[...,:3])

        elif "FLOW" in sampler_choice:
          flow_model = nnx.merge(samplerppo_network.flow_network, _unpmap(training_state.flow_state))
          samples, _ = flow_model.sample( 
            (num_envs // jax.process_count() // local_devices_to_use,),
            sample_key,
          )
          fig, ax = render_flow_pdf_2d_subplots(
            log_prob_fn=lambda x : flow_model.log_density(x=x),
            low=dr_range_low,
            high=dr_range_high,
            samples = samples,
            training_step=current_step,
            use_wandb=use_wandb,
            suptitle=f"Flow results with beta={beta} [step={int(current_step)}]"
          )
          fig.tight_layout()
          fig.canvas.draw()
          flow_frames.append(np.asarray(fig.canvas.buffer_rgba())[...,:3])
        elif sampler_choice=="GMM":
          sample_key1, sample_key2, unroll_key,  local_key = jax.random.split(local_key, 4)
          samples, _ = samplerppo_network.gmm_network.sample_selector.select_samples(\
                  _unpmap(training_state.gmm_training_state.model_state), sample_key1)
          data_for_gmm = get_experience(_unpmap(training_state), _unpmap(env_state), samples,\
                unroll_key, unroll_length, batch_size * num_minibatches // num_envs)[1]
          rewards = data_for_gmm.reward
          rewards = rewards.mean(axis=(0,1))
          target_lnpdf = beta * rewards
          # training_fig, _ = plot_reward_heatmap(samples, \
          #       target_lnpdf, dr_range_low, dr_range_high, bins=80,\
          #           title = f"Training Heatmap with beta={beta} [step={int(current_step)}]")
          # training_fig.tight_layout()
          # training_fig.canvas.draw()
          # gmm_training_frames.append(np.asarray(training_fig.canvas.buffer_rgba())[...,:3])
          # if use_wandb:
          #   wandb.log(
          #           {"training_heatmap" :wandb.Image(training_fig)},
          #           step=int(current_step),
          #       )
          eval_samples = samplerppo_network.gmm_network.model.sample(_unpmap(\
            training_state.gmm_training_state.model_state.gmm_state), sample_key2, 2**16)[0]
          
          log_prob_fn = jax.vmap(functools.partial(samplerppo_network.gmm_network.model.log_density,\
                      gmm_state=_unpmap(training_state.gmm_training_state.model_state.gmm_state)))
          model_fig, model_fig_raw = gmmvi_utils.visualise(
            log_prob_fn,
            dr_range_low,
            dr_range_high,
            # samples,
            eval_samples=eval_samples,
            bijector_log_prob=jax.vmap(samplerppo_network.gmm_network.model.bijector_log_prob)
          )
          model_fig.tight_layout()
          model_fig.canvas.draw()
          gmm_frames.append(np.asarray(model_fig.canvas.buffer_rgba())[...,:3])
          if use_wandb:
            wandb.log(
                    {"Sampler Heatmap" :wandb.Image(model_fig)},
                    step=int(current_step),
                )
        np.save(os.path.join(save_dir, f"rewards_{current_step}.npy"), reward_1d)
        
        evaluation_key, local_key = jax.random.split(local_key)
        evaluation_key = jax.random.split(evaluation_key, local_devices_to_use)
        rewards = evaluation_on_current_occupancy(
            training_state, env_state, evaluation_key
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), rewards)
        rewards = rewards.mean((0,1)).squeeze()

        x, y = jnp.meshgrid(jnp.linspace(dr_range_low[0], dr_range_high[0], 32),\
                              jnp.linspace(dr_range_low[1], dr_range_high[1], 32))
        target_lnpdfs = beta * rewards
        target_lnpdfs = jnp.reshape(target_lnpdfs, x.shape)
        target_fig = plt.figure()
        
        ctf = plt.contourf(x, y, target_lnpdfs, levels=20, cmap='viridis')
        cbar = target_fig.colorbar(ctf)
        target_fig.suptitle(f"target log prob on current occupancy [step={current_step}]")
        target_fig.tight_layout()
        target_fig.canvas.draw()
        occupancy_frames.append(np.asarray(target_fig.canvas.buffer_rgba())[...,:3])
        if use_wandb:
          wandb.log({
            'target log prob on current occupancy with returns' : wandb.Image(target_fig)
          }, step=int(current_step))
      logging.info(metrics)
      progress_fn(current_step, metrics)

  total_steps = current_step
  if not total_steps >= num_timesteps:
    raise AssertionError(
        f'Total steps {total_steps} is less than `num_timesteps`='
        f' {num_timesteps}.'
    )
  print("-------------final metrics------------------", metrics)
  if work_dir is not None:
    save_dir = make_dir(work_dir / "results" / sampler_choice)
    np.save(os.path.join(save_dir, f"reward_1d_{current_step}.npy"), reward_1d)
  if len(dr_range_low)==2:
    if sampler_choice == "NODR" or sampler_choice=="UDR" or sampler_choice=="EPOpt":
      imageio.mimsave(os.path.join(save_dir, f"Evaluation Heatmap.gif"), evaluation_frames, fps=4)
    elif sampler_choice =="AutoDR":
      imageio.mimsave(os.path.join(save_dir, f"Evaluation Heatmap [threshold={success_threshold}].gif"), evaluation_frames, fps=4)
      imageio.mimsave(os.path.join(save_dir, f"target log prob with current occupancy [threshold={success_threshold}].gif"), occupancy_frames, fps=4)
      imageio.mimsave(os.path.join(save_dir, f"Auto DR Heatmap [threshold={success_threshold}].gif"), autodr_frames, fps=4)
    elif sampler_choice =="DORAEMON":
      imageio.mimsave(os.path.join(save_dir, f"Evaluation Heatmap [threshold={success_threshold}_condition={success_rate_condition}].gif"), evaluation_frames, fps=4)
      imageio.mimsave(os.path.join(save_dir, f"target log prob with current occupancy [threshold={success_threshold}_condition={success_rate_condition}].gif"), occupancy_frames, fps=4)
      imageio.mimsave(os.path.join(save_dir, f"Doraemon Heatmap [threshold={success_threshold}_condition={success_rate_condition}].gif"), doraemon_frames, fps=4)
    elif "FLOW" in sampler_choice:
      imageio.mimsave(os.path.join(save_dir, f"Evaluation Heatmap [beta={beta}_gamma={gamma}].gif"), evaluation_frames, fps=4)
      imageio.mimsave(os.path.join(save_dir, f"target log prob with current occupancy [beta={beta}_gamma={gamma}].gif"), occupancy_frames, fps=4)
      imageio.mimsave(os.path.join(save_dir, f"Flow Heatmap [beta={beta}_gamma={gamma}].gif"), flow_frames, fps=4)
    elif sampler_choice =="GMM":
      imageio.mimsave(os.path.join(save_dir, f"Evaluation Heatmap [beta={beta}].gif"), evaluation_frames, fps=4)
      imageio.mimsave(os.path.join(save_dir, f"target log prob with current occupancy [beta={beta}].gif"), occupancy_frames, fps=4)
      # imageio.mimsave(os.path.join(save_dir, f"GMM Training Heatmap [beta={beta}].gif"), gmm_training_frames, fps=4)
      imageio.mimsave(os.path.join(save_dir, f"GMM Log Prob Heatmap [beta={beta}].gif"), gmm_frames, fps=4)
  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  params = _unpmap((
      training_state.normalizer_params,
      training_state.params.policy,
      training_state.params.value,
  ))
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()

  return (make_policy, params, metrics)
