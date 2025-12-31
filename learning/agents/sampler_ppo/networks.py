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

"""SAMPLERPPO networks."""

from typing import Sequence, Tuple, Any

from brax.training import distribution
from learning.agents.sampler_ppo.distributions import make_adr_update_fn, make_doraemon_update_fn, make_initial_state
from learning.module.bijx.autoregressive import make_autoregressive_nsf_bijx
from learning.module.bijx.realnvp import make_realnvp_bijx
from learning.module.gmmvi.network import create_gmm_network_and_state
from module import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax
from flax import nnx

@flax.struct.dataclass
class SAMPLERPPONetworks:
  policy_network: networks.FeedForwardNetwork
  value_network: networks.FeedForwardNetwork
  gmm_network: networks.FeedForwardNetwork
  flow_network: Any
  parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(samplerppo_networks: SAMPLERPPONetworks):
  """Creates params and inference function for the SAMPLERPPO agent."""

  def make_policy(
      params: types.Params, deterministic: bool = False
  ) -> types.Policy:
    policy_network = samplerppo_networks.policy_network
    parametric_action_distribution = samplerppo_networks.parametric_action_distribution

    def policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      param_subset = (params[0], params[1])  # normalizer and policy params
      logits = policy_network.apply(*param_subset, observations)
      if deterministic:
        return samplerppo_networks.parametric_action_distribution.mode(logits), {}
      raw_actions = parametric_action_distribution.sample_no_postprocessing(
          logits, key_sample
      )
      log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
      postprocessed_actions = parametric_action_distribution.postprocess(
          raw_actions
      )
      return postprocessed_actions, {
          'log_prob': log_prob,
          'raw_action': raw_actions,
      }

    return policy

  return make_policy


def make_samplerppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    dynamics_param_size : int,
    num_envs :int,
    batch_size : int,
    init_key : jax.random.PRNGKey,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
    bound_info : Tuple = None,
    sampler_choice: str = "NODR",
    success_threshold = 600,
    success_rate_condition = 0.5,
    kl_upper_bound= 0.1,
) -> SAMPLERPPONetworks:
  """Make SAMPLERPPO networks with preprocessor."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size
  )
  policy_network = networks.make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      obs_key=policy_obs_key,
  )
  value_network = networks.make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
      obs_key=value_obs_key,
  )
  print("num envs", num_envs)
  print('batch size', batch_size)
  init_gmmvi_state, gmm_network = create_gmm_network_and_state(dynamics_param_size, \
                                                               num_envs, batch_size, init_key,\
                                                               prior_scale=.5,
                                                                bound_info=bound_info)
  dr_low, dr_high = bound_info
  init_fn, autodr_update_fn = make_adr_update_fn(
    domain_low=dr_low,
    domain_high=dr_high,
    success_threshold=success_threshold,
    expansion_factor=1.1,
  )
  init_autodr_state = init_fn()
  if sampler_choice=="FLOW_NS":
    init_flow_model = make_autoregressive_nsf_bijx(
      ndim=dynamics_param_size,
      bins=16,
      hidden_features=(64, 64),
      n_transforms=3,
      seed=0,
      domain_range=bound_info,
  )

  else:
    init_flow_model = make_realnvp_bijx(
      ndim=dynamics_param_size,
      channels=256,
      n_layers=8,
      seed=0,
      domain_range=bound_info,
  )
  min_bound = 0.8
  max_bound = 110.0
  init_beta_param = 30.0
  init_doraemon_state = make_initial_state(dynamics_param_size, init_beta_param, min_bound, max_bound)
  flow_network, init_flow_state = nnx.split(init_flow_model)
  doraemon_update_fn = make_doraemon_update_fn(
        low=dr_low, high=dr_high,
        success_threshold=success_threshold,
        success_rate_condition=success_rate_condition,
        kl_upper_bound=kl_upper_bound,
        min_bound=min_bound,
        max_bound=max_bound,
        train_until_performance_lb=False,   # just update every iter for toy
        hard_performance_constraint=True,
        n_steps_main=35,
        n_steps_restore=35,
        step_size_main=0.35,
        step_size_restore=0.35,
  )
  return SAMPLERPPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      gmm_network=gmm_network,
      flow_network =flow_network,
      parametric_action_distribution=parametric_action_distribution,
  ), autodr_update_fn, doraemon_update_fn, (init_autodr_state, init_doraemon_state, init_flow_state, init_gmmvi_state)
