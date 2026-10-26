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


"""GMMVI and normalizing-flow sampler objectives."""

from typing import Any

import jax
import jax.numpy as jnp

from learning.module.gmmvi.network import GMMNetwork, GMMTrainingState


def make_gmm_update(gmm_network: GMMNetwork):
    """Build a MORE update while retaining the sampler state schema."""

    def gmm_update(gmmvi_state, key):
        del key  # Component adaptation is disabled; the callback API retains its key.
        samples, _, sample_dist_densities, target_lnpdfs, target_lnpdf_grads = (
            gmm_network.sample_selector.select_train_datas(gmmvi_state.sample_db_state)
        )
        new_component_stepsizes = gmm_network.component_stepsize_fn(
            gmmvi_state.model_state
        )
        new_model_state = gmm_network.model.update_stepsizes(
            gmmvi_state.model_state, new_component_stepsizes
        )
        expected_hessian_neg, expected_grad_neg = gmm_network.more_ng_estimator(
            new_model_state,
            samples,
            sample_dist_densities,
            target_lnpdfs,
            target_lnpdf_grads,
        )
        new_model_state = gmm_network.component_updater(
            new_model_state,
            expected_hessian_neg,
            expected_grad_neg,
            new_model_state.stepsizes,
        )

        new_model_state = gmm_network.weight_updater(
            new_model_state,
            samples,
            sample_dist_densities,
            target_lnpdfs,
            gmmvi_state.weight_stepsize,
        )
        new_num_updates = gmmvi_state.num_updates + 1

        return GMMTrainingState(
            temperature=gmmvi_state.temperature,
            model_state=new_model_state,
            component_adaptation_state=gmmvi_state.component_adaptation_state,
            num_updates=new_num_updates,
            sample_db_state=gmmvi_state.sample_db_state,
            weight_stepsize=gmmvi_state.weight_stepsize,
        )

    return gmm_update


def flow_loss(
    flow_model,
    dynamics_params: jnp.ndarray,
    prev_sample: jnp.ndarray,
    prev_logp: jnp.ndarray,
    target_lnpdf: Any,
    key: jax.random.PRNGKey,
    dr_range_high,
    dr_range_low,
    gamma,
    entropy_update=True,
):
    """Loss for training the flow network to generate adversarial dynamics parameters."""
    # If dr_low/dr_high are 1-D (shape: [D]) → scalar
    volume = jnp.prod(dr_range_high - dr_range_low)
    data_log_prob = flow_model.log_density(x=dynamics_params)
    data_log_prob = jnp.clip(data_log_prob, -1e6, 1e6)
    current_sample, current_logp = flow_model.sample((10000,), key)
    current_logp_on_prev = flow_model.log_density(x=prev_sample)
    kl_loss = (prev_logp - current_logp_on_prev).mean()
    # Get next action and log prob from policy for adversarial observation
    if entropy_update:
        value_loss = (
            -volume * (data_log_prob * jnp.exp(data_log_prob) * target_lnpdf).mean()
        )
        entropy_loss = volume * (jnp.exp(current_logp) * current_logp).mean()
    else:
        value_loss = (data_log_prob * target_lnpdf).mean()
        entropy_loss = (current_logp).mean()
    loss = value_loss + entropy_loss

    return loss + gamma * kl_loss, (
        {
            "flow_value_loss": value_loss,
            "flow_entropy_loss": entropy_loss,
            "flow_kl_loss": kl_loss,
        },
        current_sample,
        current_logp,
    )
