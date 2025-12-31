import os
from typing import List

import jax.numpy as jnp
import distrax
import chex
import jax.random
import matplotlib.pyplot as plt
import wandb

from learning.module.target_examples.base_target import Target


class Funnel(Target):
    def __init__(self, dim, log_Z=0., can_sample=True, sample_bounds=None) -> None:
        super().__init__(dim, log_Z, can_sample)
        self.data_ndim = dim
        self.dist_dominant = distrax.Normal(jnp.array([0.0]), jnp.array([3.]))
        self.mean_other = jnp.zeros(dim - 1, dtype=float)
        self.cov_eye = jnp.eye(dim - 1).reshape((1, dim - 1, dim - 1))
        self.sample_bounds = sample_bounds

    def log_prob(self, x: chex.Array):
        batched = x.ndim == 2
        if not batched:
            x = x[None,]

        dominant_x = x[:, 0]
        log_density_dominant = self.dist_dominant.log_prob(dominant_x)  # (B, )

        log_sigma = 0.5 * x[:, 0:1]
        sigma2 = jnp.exp(x[:, 0:1])
        neglog_density_other = 0.5 * jnp.log(2 * jnp.pi) + log_sigma + 0.5 * x[:, 1:] ** 2 / sigma2
        log_density_other = jnp.sum(-neglog_density_other, axis=-1)
        low = jnp.array([-10,-5])
        high= jnp.array([5,5])
        
        log_prob = log_density_dominant + log_density_other
        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)
        # max_log_prob = jnp.max(log_prob)
        # log_prob = -10 *jnp.ones_like(log_prob)
        # log_prob = jnp.where(jnp.logical_or(x[:,0] > high[0], x[:,0] < low[0]) , max_log_prob* jnp.ones_like(log_prob), log_prob).squeeze()
        # log_prob = jnp.where(jnp.logical_or(x[:,1] > high[1], x[:,1] < low[1]) , max_log_prob* jnp.ones_like(log_prob), log_prob).squeeze()
        
        return log_prob

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        key1, key2 = jax.random.split(seed)
        dominant_x = self.dist_dominant.sample(seed=key1, sample_shape=sample_shape)  # (B,1)
        x_others = self._dist_other(dominant_x).sample(seed=key2)  # (B, dim-1)
        if self.sample_bounds is not None:
            return jnp.hstack([dominant_x, x_others]).clip(min=self.sample_bounds[0], max=self.sample_bounds[1])
        else:
            return jnp.hstack([dominant_x, x_others])

    def _dist_other(self, dominant_x):
        variance_other = jnp.exp(dominant_x)
        cov_other = variance_other.reshape(-1, 1, 1) * self.cov_eye
        # use covariance matrix, not std
        return distrax.MultivariateNormalFullCovariance(self.mean_other, cov_other)

    def visualise(self, samples: chex.Array = None, axes: List[plt.Axes] = None, model_log_prob_fn=None, show=False, prefix='') -> dict:
        plt.close()
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        # ax2 = fig.add_subplot(122)
        x, y = jnp.meshgrid(jnp.linspace(-12, 7, 100), jnp.linspace(-6, 6, 100))
        grid = jnp.c_[x.ravel(), y.ravel()]
        pdf_values = jax.vmap(jnp.exp)(self.log_prob(grid))
        pdf_values = jnp.reshape(pdf_values, x.shape)
        ctf1 = ax1.contourf(x, y, pdf_values, levels=20, cmap='viridis')
        # ctf2 = ax2.contourf(x, y, 1-pdf_values, levels=20, cmap='viridis')
        fig.colorbar(ctf1, ax=ax1)
        # fig.colorbar(ctf2, ax=ax2)
        if samples is not None:
            idx = jax.random.choice(jax.random.PRNGKey(0), samples.shape[0], (300,))
            sample_x = jnp.clip(samples[idx, 0],-10,5)
            sample_y = jnp.clip(samples[idx, 1],-5,5)
            ax1.scatter(sample_x, sample_y, c='r', alpha=0.5, marker='x')
        if model_log_prob_fn is not None:
            ax3 = fig.add_subplot(122)
            grid = jnp.c_[x.ravel(), y.ravel()]
            pdf_values = jax.vmap(jnp.exp)(model_log_prob_fn(sample=grid))
            pdf_values = jnp.reshape(pdf_values, x.shape)
            ctf = ax3.contourf(x, y, pdf_values, levels=20, cmap='viridis')
            fig.colorbar(ctf, ax=ax3)

        # plt.xlabel('X')
        # plt.ylabel('Y')
        plt.xticks([])
        plt.yticks([])
        # plt.xlim(-10, 5)
        # plt.ylim(-5, 5)

        # plt.savefig(os.path.join(project_path('./samples/funnel/'), f"{prefix}funnel.pdf"), bbox_inches='tight', pad_inches=0.1)

        wb = {"figures/vis": [wandb.Image(fig)]}
        if show:
            plt.show()

        return wb
