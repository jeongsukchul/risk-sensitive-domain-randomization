"""Generic bounded distributions and compatibility imports for DR samplers.

AutoDR and DORAEMON implementations live in their corresponding modules.
The re-exports retain the original import paths used by analysis notebooks.
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from learning.agents.sampler_ppo.adr import (
    ADRState as ADRState,
)
from learning.agents.sampler_ppo.adr import (
    BoundarySamplingDist as BoundarySamplingDist,
)
from learning.agents.sampler_ppo.adr import (
    get_adr_log_prob as get_adr_log_prob,
)
from learning.agents.sampler_ppo.adr import (
    get_adr_sample as get_adr_sample,
)
from learning.agents.sampler_ppo.adr import (
    get_adr_train_dist as get_adr_train_dist,
)
from learning.agents.sampler_ppo.adr import (
    init_adr_state as init_adr_state,
)
from learning.agents.sampler_ppo.adr import (
    make_adr_update_fn as make_adr_update_fn,
)
from learning.agents.sampler_ppo.distribution_plots import (
    plot_adr_density_2d as plot_adr_density_2d,
)
from learning.agents.sampler_ppo.distribution_plots import (
    plot_beta_density_2d as plot_beta_density_2d,
)
from learning.agents.sampler_ppo.doraemon import (
    DoraemonState as DoraemonState,
)
from learning.agents.sampler_ppo.doraemon import (
    _inv_sigmoid_bounded as _inv_sigmoid_bounded,
)
from learning.agents.sampler_ppo.doraemon import (
    _kl_beta_beta as _kl_beta_beta,
)
from learning.agents.sampler_ppo.doraemon import (
    _log_prob_beta_on_box as _log_prob_beta_on_box,
)
from learning.agents.sampler_ppo.doraemon import (
    _sigmoid_bounded as _sigmoid_bounded,
)
from learning.agents.sampler_ppo.doraemon import (
    _unpack_beta as _unpack_beta,
)
from learning.agents.sampler_ppo.doraemon import (
    make_doraemon_update_fn as make_doraemon_update_fn,
)
from learning.agents.sampler_ppo.doraemon import (
    make_initial_state as make_initial_state,
)
from learning.agents.sampler_ppo.doraemon import (
    sample_beta_on_box as sample_beta_on_box,
)


class UniformDist:
    """Uniform distribution over a bounded domain."""

    def __init__(self, low, high):
        """
        Initialize uniform distribution.

        Args:
            low: Lower bounds
            high: Upper bounds
        """
        self.low = jnp.array(low, dtype=jnp.float32)
        self.high = jnp.array(high, dtype=jnp.float32)
        self.ndim = len(low)

    def rsample(self, sample_shape: Tuple[int, ...] = (), key=None):
        """Sample uniformly from the domain."""
        if key is None:
            key = jax.random.PRNGKey(0)

        n_samples = sample_shape[0] if len(sample_shape) > 0 else 1
        samples = (
            jax.random.uniform(key, (n_samples, self.ndim)) * (self.high - self.low)
            + self.low
        )
        return samples

    def log_prob(self, value):
        """Compute log probability under uniform distribution."""
        in_range = jnp.all((value >= self.low) & (value <= self.high), axis=-1)
        volume = jnp.prod(self.high - self.low)
        return jnp.where(in_range, -jnp.log(volume), -jnp.inf)


class BetasDist:
    """Beta distribution(s) over a bounded domain using JAX."""

    def __init__(self, alphas, betas, low, high):
        """
        Initialize beta distribution.

        Args:
            alphas: Alpha parameters for beta distribution (array-like)
            betas: Beta parameters for beta distribution (array-like)
            low: Lower bounds for support
            high: Upper bounds for support
        """
        self.alphas = jnp.array(alphas, dtype=jnp.float32)
        self.betas = jnp.array(betas, dtype=jnp.float32)
        self.low = jnp.array(low, dtype=jnp.float32)
        self.high = jnp.array(high, dtype=jnp.float32)
        self.ndim = len(self.alphas)

    def rsample(self, sample_shape=(), key=None):
        """
        Sample from beta distribution.

        Args:
            sample_shape: Shape of samples (default: scalar)
            key: JAX random key

        Returns:
            Samples transformed to [low, high] range
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        n_samples = sample_shape[0] if len(sample_shape) > 0 else 1

        # Generate beta-distributed samples for each dimension
        keys = jax.random.split(key, self.ndim)
        samples_list = []
        for i in range(self.ndim):
            # Sample from beta distribution using gamma distribution ratio
            key_g1, key_g2 = jax.random.split(keys[i])
            g1 = jax.random.gamma(key_g1, self.alphas[i], shape=(n_samples,))
            g2 = jax.random.gamma(key_g2, self.betas[i], shape=(n_samples,))
            beta_sample = g1 / (g1 + g2)
            samples_list.append(beta_sample)

        samples = jnp.stack(samples_list, axis=-1)

        # Transform samples to the desired range
        output = self.low + (self.high - self.low) * samples

        return output

    def log_prob(self, value):
        """
        Compute log probability of values under this distribution.

        Args:
            value: Array of shape (..., ndim)

        Returns:
            Log probability array of shape (...)
        """
        # Transform value back to [0, 1] range
        transformed_value = (value - self.low) / (self.high - self.low)

        # Clamp to avoid log(0)
        transformed_value = jnp.clip(transformed_value, 1e-7, 1 - 1e-7)

        # Compute log beta function: log B(alpha, beta) = log Gamma(alpha) + log Gamma(beta) - log Gamma(alpha + beta)
        from scipy.special import gammaln

        log_beta_fn = (
            gammaln(self.alphas)
            + gammaln(self.betas)
            - gammaln(self.alphas + self.betas)
        )

        # Compute log probability for each dimension
        log_probs = (
            (self.alphas - 1) * jnp.log(transformed_value)
            + (self.betas - 1) * jnp.log(1 - transformed_value)
            - log_beta_fn
        )

        # Account for the transformation to [low, high]
        log_probs -= jnp.log(self.high - self.low)

        return jnp.sum(log_probs, axis=-1)

    def to_flat(self):
        """Convert distribution parameters to a flat array."""
        return jnp.concatenate([self.alphas, self.betas])

    @classmethod
    def from_flat(cls, flat_params, low, high):
        """Create a BetasDist instance from a flat array of parameters."""
        ndim = len(low)
        alphas = flat_params[:ndim]
        betas = flat_params[ndim:]
        return cls(alphas, betas, low, high)

    def kl_divergence(self, other):
        """
        Compute KL divergence between this BetasDist and another BetasDist or UniformDist.
        """
        from scipy.special import digamma

        kl_div = 0.0
        for i in range(self.ndim):
            if isinstance(other, UniformDist):
                # KL divergence from Beta(alpha, beta) to Uniform[0,1]
                from scipy.special import gammaln

                log_beta_fn = (
                    gammaln(self.alphas[i])
                    + gammaln(self.betas[i])
                    - gammaln(self.alphas[i] + self.betas[i])
                )
                kl_div_i = (
                    log_beta_fn
                    + (1 - self.alphas[i]) * digamma(self.alphas[i])
                    + (1 - self.betas[i]) * digamma(self.betas[i])
                    - (2 - self.alphas[i] - self.betas[i])
                    * digamma(self.alphas[i] + self.betas[i])
                )
                # Adjust for the change in scale
                kl_div_i -= jnp.log(self.high[i] - self.low[i])
                kl_div += float(kl_div_i)

            elif isinstance(other, BetasDist):
                # KL divergence between two Beta distributions
                from scipy.special import gammaln

                log_beta_p = (
                    gammaln(self.alphas[i])
                    + gammaln(self.betas[i])
                    - gammaln(self.alphas[i] + self.betas[i])
                )
                log_beta_q = (
                    gammaln(other.alphas[i])
                    + gammaln(other.betas[i])
                    - gammaln(other.alphas[i] + other.betas[i])
                )
                kl_div_i = (
                    log_beta_q
                    - log_beta_p
                    + (self.alphas[i] - other.alphas[i]) * digamma(self.alphas[i])
                    + (self.betas[i] - other.betas[i]) * digamma(self.betas[i])
                    + (
                        other.alphas[i]
                        + other.betas[i]
                        - self.alphas[i]
                        - self.betas[i]
                    )
                    * digamma(self.alphas[i] + self.betas[i])
                )
                # Adjust for different bounds if necessary
                if not jnp.allclose(self.low[i], other.low[i]) or not jnp.allclose(
                    self.high[i], other.high[i]
                ):
                    kl_div_i += jnp.log(
                        (other.high[i] - other.low[i]) / (self.high[i] - self.low[i])
                    )
                kl_div += float(kl_div_i)
            else:
                raise ValueError(f"Unsupported distribution type: {type(other)}")

        return kl_div
