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

"""td3 networks."""

from typing import Sequence, Tuple

from brax.training import distribution
from module import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax.numpy as jnp
import jax
@flax.struct.dataclass
class Td3Networks:
  policy_network: networks.FeedForwardNetwork
  q_network: networks.FeedForwardNetwork


def make_inference_fn(td3_networks: Td3Networks):
  """Creates params and inference function for the td3 agent."""

  def make_policy(
      params: types.PolicyParams, deterministic: bool = False, std_min: float = 0.05, std_max: float = 0.8 
  ) -> types.Policy:

    def deterministic_policy(
        observations: types.Observation,
        key: PRNGKey = None,
    ) -> Tuple[types.Action, types.Extra]:
      return td3_networks.policy_network.apply(*params, observations), None

    def stochastic_policy(
        observations: types.Observation,
        noise_scales: jnp.ndarray,
        key: PRNGKey,
    ):  
        act = td3_networks.policy_network.apply(*params, observations)
        noise = jax.random.normal(key, shape=act.shape) * noise_scales[..., None]
        return act + noise, None
    if deterministic:
        return deterministic_policy
    else:
        return stochastic_policy

  return make_policy
def make_td3_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
    distributional_q : bool= False,
    num_atoms: int = 101,
    v_min: float = 0.,
    v_max: float = 0.,
) -> Td3Networks:
    """Make td3 networks."""
    policy_network = networks.make_deterministic_policy_network(
        action_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=policy_network_layer_norm,
        obs_key = policy_obs_key
    )
    if distributional_q:
        q_network=networks.make_distributional_q_network(
        observation_size,
        action_size,
        num_atoms,
        v_max,
        v_min,
        preprocess_observations_fn=preprocess_observations_fn,
        #    hidden_layer_sizes=hidden_layer_sizes,
    )
    else:
        q_network = networks.make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
        obs_key = value_obs_key,
    )
    return Td3Networks(
        policy_network=policy_network,
        q_network=q_network,
    ), jnp.linspace(v_min, v_max, num_atoms)