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

"""TD3 losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

from typing import Any

from brax.training import types
from agents.td3 import networks as td3_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp

Transition = types.Transition


def make_losses(
    td3_network: td3_networks.Td3Networks,
    reward_scaling: float,
    discounting: float,
    distributional_q : bool=False,
):
    """Creates the td3 losses."""

    policy_network = td3_network.policy_network
    q_network = td3_network.q_network
    if distributional_q:
        def critic_loss(
            q_params: Params,
            policy_params: Params,
            normalizer_params: Any,
            target_q_params: Params,
            transitions: Transition,
            noise: jnp.ndarray,
            q_support:jnp.ndarray,            
            key: PRNGKey,
        ) -> jnp.ndarray:
            """ 
            B : batch_size
            A : num_atoms
            N : num_critics(often just use 2)
            """
            batch_size = transitions.action.shape[0]
            q_old_action = q_network.apply(
                normalizer_params, q_params, transitions.observation, transitions.action #[B, A, N]
            )
            next_action = policy_network.apply(
                normalizer_params, policy_params, transitions.next_observation
            )
            next_action = jnp.clip(next_action + noise, -1.0, 1.0)

            target_q =  transitions.reward[..., None] * reward_scaling \
                + transitions.discount[..., None] * discounting * q_support[None, ...] #[B, A,]
            next_q_proj = q_network.apply(normalizer_params, target_q_params, \
                                          transitions.next_observation, next_action, target_q, mode="projection") # [B, A, N]
            next_q_value = jnp.sum(next_q_proj * q_support[None, :, None], axis=1) #[B, N]
            min_idx = next_q_value.argmin(axis=-1)       #[B,]
            target_dist = jnp.swapaxes(next_q_proj, 1, 2)[jnp.arange(batch_size), min_idx] #[B,A]

            q_error = -jnp.sum(target_dist[...,None]*jax.nn.log_softmax(q_old_action, axis=1), axis=(1,2))
            # Better bootstrapping for truncated episodes.
            truncation = transitions.extras['state_extras']['truncation']
            q_error *= 1-truncation
            q_loss = q_error.mean()
            # q_loss = 0.5 * jnp.mean(jnp.square(q_error))
            return q_loss, (q_old_action, next_q_value)
    else:
        def critic_loss(
            q_params: Params,
            policy_params: Params,
            normalizer_params: Any,
            target_q_params: Params,
            transitions: Transition,
            noise: jnp.ndarray,
            key: PRNGKey,
        ) -> jnp.ndarray:
            q_old_action = q_network.apply(
                normalizer_params, q_params, transitions.observation, transitions.action
            )
            next_action = policy_network.apply(
                normalizer_params, policy_params, transitions.next_observation
            )
            next_action = jnp.clip(next_action + noise, -1.0, 1.0)
            next_q = q_network.apply(
                normalizer_params,
                target_q_params,
                transitions.next_observation,
                next_action,
            )
            next_v = jnp.min(next_q, axis=-1) 
            target_q = jax.lax.stop_gradient(
                transitions.reward * reward_scaling
                + transitions.discount * discounting * next_v
            )
            q_error = q_old_action - jnp.expand_dims(target_q, -1)

            # Better bootstrapping for truncated episodes.
            truncation = transitions.extras['state_extras']['truncation']
            q_error *= jnp.expand_dims(1 - truncation, -1)

            q_loss = 0.5 * jnp.mean(jnp.square(q_error))
            return q_loss, (q_old_action, next_v)

    def actor_loss(
        policy_params: Params,
        normalizer_params: Any,
        q_params: Params,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        action = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        q_action = q_network.apply(
            normalizer_params, q_params, transitions.observation, action
        )
        min_q = jnp.min(q_action, axis=-1)
        return -jnp.mean(min_q)

    return critic_loss, actor_loss
