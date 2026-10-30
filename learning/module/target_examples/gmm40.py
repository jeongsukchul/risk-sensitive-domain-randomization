# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions adapted from variational_sampling_methods.
# Copyright (c) 2023 Denis Blessing
# Modified for RSDR; upstream MIT terms are in
# THIRD_PARTY_NOTICES.md.

import itertools
from typing import List, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import distrax
from matplotlib import pyplot as plt
import wandb
import numpy as np
from learning.module.target_examples.base_target import Target
import matplotlib
def plot_contours_2D(log_prob_func,
                     fig,
                     ax: Optional[plt.Axes] = None,
                     bound: float = 3,
                     levels: int = 20):
    """Plot the contours of a 2D log prob function."""
    if ax is None:
        fig, ax = plt.subplots(1)
    n_points = 100
    x_points_dim1 = np.linspace(-bound, bound, n_points)
    x_points_dim2 = np.linspace(-bound, bound, n_points)
    x_points = np.array(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_probs = log_prob_func(x_points)
    log_probs = jnp.clip(log_probs, a_min=-1000, a_max=None)
    x1 = x_points[:, 0].reshape(n_points, n_points)
    x2 = x_points[:, 1].reshape(n_points, n_points)
    z = log_probs.reshape(n_points, n_points)
    z = jnp.exp(z)
    ctf = ax.contourf(x1, x2, z, levels=levels, cmap='viridis')
    # ct = ax.contour(x1, x2, z, levels=levels)
    fig.colorbar(ctf, ax=ax)
    # ax.contourf(x1, x2, np.exp(z), levels = 20, cmap = 'viridis')



def plot_marginal_pair(samples: chex.Array,
                  ax: Optional[plt.Axes] = None,
                  marginal_dims: Tuple[int, int] = (0, 1),
                  bounds: Tuple[float, float] = (-5, 5),
                  alpha: float = 0.5):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    if not ax:
        fig, ax = plt.subplots(1)
    samples = jnp.clip(samples, bounds[0], bounds[1])
    ax.plot(samples[:, marginal_dims[0]], samples[:, marginal_dims[1]], "o", alpha=alpha)


# matplotlib.use('agg')

class GMM40(Target):
    def __init__(
            self,
            dim: int = 2, num_components: int = 40, loc_scaling: float = 40,
            scale_scaling: float = 1.0, seed: int = 0, sample_bounds=None, can_sample=True, log_Z=0
    ) -> None:
        super().__init__(dim, log_Z, can_sample)

        self.seed = seed
        self.n_mixes = num_components

        key = jax.random.PRNGKey(seed)
        logits = jnp.ones(num_components)
        mean = jax.random.uniform(shape=(num_components, dim), key=key, minval=-1.0, maxval=1.0) * loc_scaling
        scale = jnp.ones(shape=(num_components, dim)) * scale_scaling

        mixture_dist = distrax.Categorical(logits=logits)
        components_dist = distrax.Independent(
            distrax.Normal(loc=mean, scale=scale), reinterpreted_batch_ndims=1
        )
        self.distribution = distrax.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            components_distribution=components_dist,
        )

        self._plot_bound = loc_scaling * 1.5

    def log_prob(self, x: chex.Array) -> chex.Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None,]

        log_prob = self.distribution.log_prob(x)
        
        if not batched:
            log_prob = 10* jnp.squeeze(log_prob, axis=0)
        # log_prob = jnp.where(jnp.logical_or(x[:,0] > self._plot_bound, x[:,0] < -self._plot_bound) , jnp.max(log_prob)* jnp.ones_like(log_prob), log_prob).squeeze()
        # log_prob = jnp.where(jnp.logical_or(x[:,1] > self._plot_bound, x[:,1] < -self._plot_bound) , jnp.max(log_prob)* jnp.ones_like(log_prob), log_prob).squeeze()

        return  log_prob

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        return self.distribution.sample(seed=seed, sample_shape=sample_shape)

    def entropy(self, samples: chex.Array = None):
        expanded = jnp.expand_dims(samples, axis=-2)
        # Compute `log_prob` in every component.
        idx = jnp.argmax(self.distribution.components_distribution.log_prob(expanded), 1)
        unique_elements, counts = jnp.unique(idx, return_counts=True)
        mode_dist = counts / samples.shape[0]
        entropy = -jnp.sum(mode_dist * (jnp.log(mode_dist) / jnp.log(self.n_mixes)))
        return entropy

    def visualise(self, samples: chex.Array = None, axes=None, model_log_prob_fn=None, show=False, prefix='') -> dict:
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot()
        # ax2 = fig.add_subplot(312)
        if samples is not None:
            plot_marginal_pair(samples[:, :2], ax, bounds=(-self._plot_bound, self._plot_bound))
            # jnp.save(project_path(f'samples/gmm40_samples'), samples)
        if self.dim == 2:
            # x, y = jnp.meshgrid(jnp.linspace(-self._plot_bound*1.2, self._plot_bound*1.2, 100), jnp.linspace(-self._plot_bound*1.2, self._plot_bound*1.2, 100))
            # grid = jnp.c_[x.ravel(), y.ravel()]
            # pdf_values = jax.vmap(jnp.exp)(self.log_prob(grid))
            # pdf_values = jnp.reshape(pdf_values, x.shape)
            # ctf = plt.contourf(x, y, pdf_values, levels=50, cmap='viridis')
            # cbar = fig.colorbar(ctf)
            plot_contours_2D(self.log_prob, fig, ax, bound=self._plot_bound*1.2, levels=50)
        # if model_log_prob_fn is not None:
        #     ax3 = fig.add_subplot(122)
        #     grid = jnp.c_[x.ravel(), y.ravel()]
        #     pdf_values = jax.vmap(jnp.exp)(model_log_prob_fn(sample=grid))
        #     pdf_values = jnp.reshape(pdf_values, x.shape)
        #     ctf = ax3.contourf(x, y, pdf_values, levels=20, cmap='viridis')
        #     cbar = fig.colorbar(ctf, ax=ax3)
        plt.xticks([])
        plt.yticks([])
        # import os
        # plt.savefig(os.path.join(project_path('./samples/gaussian_mixture40'), f"{prefix}gmm40.pdf"), bbox_inches='tight', pad_inches=0.1)

        wb = {"figures/vis": [wandb.Image(fig)]}
        if show:
            plt.show()

        return wb
        # import tikzplotlib
        # import os
        # tikzplotlib.save(os.path.join(project_path('./figures/'), f"gmm40.tex"))


if __name__ == '__main__':
    gmm = GMM40()
    samples = gmm.sample(jax.random.PRNGKey(0), (2000,))
    gmm.log_prob(samples)
    gmm.entropy(samples)
    # gmm.visualise( show=True)
    gmm.visualise(show=True)
