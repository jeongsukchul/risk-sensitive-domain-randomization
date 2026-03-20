from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
import os
import pickle
from typing import List, NamedTuple, Optional, Sequence
from learning.runtime_env import configure_jax_runtime

configure_jax_runtime()

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import wandb
from brax.training.networks import FeedForwardNetwork
from brax.training import types


@dataclass
class Args:
    # --- Experiment ---
    seed: int = 0
    dataset: str = "halfcheetah-medium-v2"
    algorithm: str = "dynamics"
    eval_interval: int = 10_000
    # --- Logging ---
    log: bool = False
    wandb_project: str = "unifloral"
    wandb_team: str = "flair"
    wandb_group: str = "debug"
    model_path: str = "dynamics_models"
    # --- Generic optimization ---
    lr: float = 0.001
    batch_size: int = 256
    # --- Dynamics training ---
    n_layers: int = 4
    layer_size: int = 200
    num_ensemble: int = 7
    num_elites: int = 5
    num_epochs: int = 400
    logvar_diff_coef: float = 0.01
    weight_decay: float = 2.5e-5
    validation_split: float = 0.2
    precompute_term_stats: bool = False


r"""
     |\  __
     \| /_/
      \|
    ___|_____
    \       /
     \     /
      \___/     Preliminaries
"""


class SingleDynamicsModel(nn.Module):
    obs_dim: int
    hidden_dims: List[int]

    @nn.compact
    def __call__(self, delta_obs_action):
        x = delta_obs_action
        for dim in self.hidden_dims:
            x = nn.relu(nn.Dense(dim)(x))
        obs_reward_stats = nn.Dense(2 * (self.obs_dim + 1))(x)
        return obs_reward_stats


class EnsembleDynamicsModel(nn.Module):
    obs_dim: int
    action_dim: int
    num_ensemble: int
    hidden_dims: List[int]
    max_logvar_init: float = 0.5
    min_logvar_init: float = -10.0

    @nn.compact
    def __call__(self, obs_action):
        # --- Compute ensemble predictions ---
        batched_model = nn.vmap(
            SingleDynamicsModel,
            variable_axes={"params": 0},  # Parameters not shared between critics
            split_rngs={"params": True},  # Different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.num_ensemble,
        )
        ensemble = batched_model(
            obs_dim=self.obs_dim,
            hidden_dims=self.hidden_dims,
            name="ensemble",
        )
        output = ensemble(obs_action)
        pred_mean, logvar = jnp.split(output, 2, axis=-1)

        # --- Soft clamp log-variance ---
        max_logvar = self.param(
            "max_logvar",
            init_fn=lambda key: jnp.full((self.obs_dim + 1,), self.max_logvar_init),
        )
        min_logvar = self.param(
            "min_logvar",
            init_fn=lambda key: jnp.full((self.obs_dim + 1,), self.min_logvar_init),
        )
        logvar = max_logvar - nn.softplus(max_logvar - logvar)
        logvar = min_logvar + nn.softplus(logvar - min_logvar)
        return pred_mean, logvar



def make_dynamics_network(obs_size, action_size, 
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor, 
    postprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor, 
    n_ensemble : int = 7,
    hidden_layer_sizes : Sequence[int] = [256, 256, 256, 256],
    ):
    dynamics_module = EnsembleDynamicsModel(
        obs_dim=obs_size,
        action_dim=action_size,
        num_ensemble=n_ensemble,
        hidden_dims = hidden_layer_sizes,
    )
    def apply(processor_params, dynamics_params, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        return dynamics_module.apply(dynamics_params, jnp.concatenate([obs, actions], axis=-1)), preprocess_observations_fn, postprocess_observations_fn,
    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    return FeedForwardNetwork(
        init= lambda key: dynamics_module.init(key, jnp.concatenate([dummy_obs, dummy_action], axis=-1)), apply=apply)


def create_dataset_iter(rng, inputs, targets, batch_size):
    """Create a batched dataset iterator."""
    perm = jax.random.permutation(rng, inputs.shape[0])
    shuffled_inputs, shuffled_targets = inputs[perm], targets[perm]
    num_batches = inputs.shape[0] // batch_size
    iter_size = num_batches * batch_size
    dataset_iter = jax.tree_util.tree_map(
        lambda x: x[:iter_size].reshape(num_batches, batch_size, *x.shape[1:]),
        (shuffled_inputs, shuffled_targets),
    )
    return dataset_iter


def save_dynamics_model(args, dynamics_model):
    """Save the EnsembleDynamics object."""
    filename = f"ensemble_dynamics_model_{args.dataset}"
    filename += f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, filename), "wb") as f:
        pickle.dump(dynamics_model, f)


def load_dynamics_model(path):
    """Load the EnsembleDynamics object."""
    print(f"Loading dynamics model from {path}")
    with open(path, "rb") as f:
        dynamics_model = pickle.load(f)
    return dynamics_model


def log_info(info):
    """Log metrics to wandb."""
    info = {"dynamics/" + k: v for k, v in info.items()}
    jax.experimental.io_callback(wandb.log, None, info)
