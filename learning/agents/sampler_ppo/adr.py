"""AutoDR boundary sampling and adaptive domain-range updates."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class BoundarySamplingDist:
    """Distribution that samples from boundaries with a specified probability.

    Sampling rule:
      - with prob (1 - boundary_prob): uniform in the full box [low, high]
      - with prob boundary_prob:
          * pick a random dimension d
          * pick side low[d] or high[d] with prob 0.5
          * set coordinate d to that boundary value, others remain uniform
    """

    def __init__(self, low, high, boundary_prob: float = 0.5):
        self.low = jnp.asarray(low, dtype=jnp.float32)
        self.high = jnp.asarray(high, dtype=jnp.float32)
        self.ndim = int(self.low.shape[0])
        self.boundary_prob = float(boundary_prob)

    def rsample(self, n_samples, key):
        """
        Sample from the distribution (non-JIT wrapper).

        Args:
            sample_shape: (N,) -> N samples, shape will be (N, ndim)
                          If (), returns a single sample with shape (1, ndim).
            key: JAX PRNGKey
        """
        samples = self._rsample_vectorized(
            n_samples=n_samples,
            low=self.low,
            high=self.high,
            ndim=self.ndim,
            boundary_prob=self.boundary_prob,
            key=key,
        )
        return samples

    @staticmethod
    def _rsample_vectorized(
        n_samples: int,
        low: jnp.ndarray,
        high: jnp.ndarray,
        ndim: int,
        boundary_prob: float,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Fully vectorized JAX implementation without Python loops.

        Returns:
            samples: (n_samples, ndim)
        """
        key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 5)

        # Decide which samples will be on the boundary: (n_samples,)
        boundary_mask = jax.random.uniform(subkey1, (n_samples,)) < boundary_prob

        # Uniform interior samples: (n_samples, ndim)
        interior_samples = (
            jax.random.uniform(subkey2, (n_samples, ndim)) * (high - low) + low
        )

        # For boundary samples, choose random dimension and side
        boundary_dims = jax.random.randint(subkey3, (n_samples,), 0, ndim)
        boundary_is_high = jax.random.uniform(subkey4, (n_samples,)) < 0.5

        # Prepare boundary values for each sample and dim:
        #   if boundary_is_high[i] -> use high
        #   else -> use low
        boundary_values = jnp.where(
            boundary_is_high[:, None],  # (n_samples, 1)
            high[None, :],  # (1, ndim)
            low[None, :],  # (1, ndim)
        )  # -> (n_samples, ndim)

        # Build a mask (n_samples, ndim) that is True only at the chosen boundary dim
        dim_ids = jnp.arange(ndim)[None, :]  # (1, ndim)
        dim_mask = dim_ids == boundary_dims[:, None]  # (n_samples, ndim)

        # We only apply boundary Values where *both* boundary_mask and dim_mask are True
        full_mask = (boundary_mask[:, None]) & dim_mask  # (n_samples, ndim)

        samples = jnp.where(full_mask, boundary_values, interior_samples)
        return samples

    def log_prob(self, value):
        """
        Approximate log probability.

        WARNING: The true density of this sampling scheme is not well-defined
        w.r.t. Lebesgue measure because of the Dirac at boundaries. Here we
        *approximate* the distribution as uniform over the box for any in-range
        x. This is fine for diagnostics/plots but NOT mathematically correct
        for exact importance sampling.
        """
        value = jnp.asarray(value, dtype=jnp.float32)
        in_range = jnp.all((value >= self.low) & (value <= self.high), axis=-1)

        volume = jnp.prod(self.high - self.low)
        log_uniform = -jnp.log(volume)

        return jnp.where(in_range, log_uniform, -jnp.inf)


class ADRState(NamedTuple):
    current_low: jnp.ndarray  # shape: (ndim,)
    current_high: jnp.ndarray  # shape: (ndim,)
    it: jnp.int32  # iteration counter (optional)


def init_adr_state(
    domain_low: jnp.ndarray,
    domain_high: jnp.ndarray,
    initial_dr_percentage: float = 0.2,
) -> ADRState:
    """
    Initialize ADRState around the mid-point of [domain_low, domain_high],
    with width = initial_dr_percentage * full_range.
    """
    domain_low = jnp.asarray(domain_low)
    domain_high = jnp.asarray(domain_high)
    mid = 0.5 * (domain_low + domain_high)
    span = domain_high - domain_low

    half_width = 0.5 * initial_dr_percentage * span
    current_low = jnp.clip(mid - half_width, domain_low, domain_high)
    current_high = jnp.clip(mid + half_width, domain_low, domain_high)

    return ADRState(current_low=current_low, current_high=current_high, it=jnp.int32(0))


def get_adr_train_dist(state: ADRState, boundary_prob: float = 0.5):
    return BoundarySamplingDist(state.current_low, state.current_high, boundary_prob)


def get_adr_sample(state: ADRState, num_samples, key):
    return BoundarySamplingDist(
        state.current_low, state.current_high, boundary_prob=0.5
    ).rsample(num_samples, key)


def get_adr_log_prob(state: ADRState, sample):
    return BoundarySamplingDist(
        state.current_low, state.current_high, boundary_prob=0.5
    ).log_prob(sample)


def make_adr_update_fn(
    *,
    domain_low: jnp.ndarray,
    domain_high: jnp.ndarray,
    success_threshold: float,
    expansion_factor: float = 1.1,
    initial_dr_percentage: float = 0.2,
):
    """
    Returns:
      init_fn() -> ADRState
      update_fn(state, contexts, returns, key) -> ADRState
    """
    domain_low = jnp.asarray(domain_low)
    domain_high = jnp.asarray(domain_high)
    ndim = domain_low.shape[0]

    lower_threshold = success_threshold / 2.0
    upper_threshold = success_threshold

    def init_fn() -> ADRState:
        return init_adr_state(domain_low, domain_high, initial_dr_percentage)

    def update_fn(
        state: ADRState,
        dynamics_params: jnp.ndarray,  # (batch_size, ndim)
        returns: jnp.ndarray,  # (batch_size,)
        key: jax.random.PRNGKey,
    ) -> ADRState:
        """
        Pure JAX update: one random dimension per call.
        """
        current_low = state.current_low
        current_high = state.current_high

        # Choose one dimension to adapt
        dim = jax.random.randint(key, shape=(), minval=0, maxval=ndim)

        low_boundary = current_low[dim]
        high_boundary = current_high[dim]

        # tolerance based on span in that dim
        boundary_tolerance = jnp.maximum(
            1e-3,
            0.01 * (high_boundary - low_boundary),
        )

        # parameters in that dimension
        x_dim = dynamics_params[:, dim]
        returns = returns.astype(jnp.float32)

        low_mask = jnp.abs(x_dim - low_boundary) < boundary_tolerance
        high_mask = jnp.abs(x_dim - high_boundary) < boundary_tolerance

        def success_rate(mask):
            count = jnp.sum(mask)
            total_success = jnp.sum(returns * mask)
            return jnp.where(count > 0, total_success / (count + 1e-8), 0.0)

        low_success_rate = success_rate(low_mask)
        high_success_rate = success_rate(high_mask)

        midpoint = 0.5 * (low_boundary + high_boundary)
        dlow = domain_low[dim]
        dhigh = domain_high[dim]

        # --- lower boundary update ---

        # Expand: move lower boundary further down (toward domain_low)
        expand_low = jnp.maximum(
            midpoint - (midpoint - low_boundary) * expansion_factor,
            dlow,
        )

        # Contract: move lower boundary up (toward midpoint)
        contract_low = jnp.minimum(
            midpoint - (midpoint - low_boundary) / expansion_factor,
            midpoint,
        )

        new_low_dim = jnp.where(
            low_success_rate > upper_threshold,
            expand_low,
            jnp.where(
                low_success_rate < lower_threshold,
                contract_low,
                low_boundary,
            ),
        )

        # --- upper boundary update ---

        # Expand: move upper boundary further up (toward domain_high)
        expand_high = jnp.minimum(
            midpoint + (high_boundary - midpoint) * expansion_factor,
            dhigh,
        )

        # Contract: move upper boundary down (toward midpoint)
        contract_high = jnp.maximum(
            midpoint + (high_boundary - midpoint) / expansion_factor,
            0.5 * (low_boundary + high_boundary),
        )

        new_high_dim = jnp.where(
            high_success_rate > upper_threshold,
            expand_high,
            jnp.where(
                high_success_rate < lower_threshold,
                contract_high,
                high_boundary,
            ),
        )

        # Scatter back into vectors
        current_low_new = current_low.at[dim].set(new_low_dim)
        current_high_new = current_high.at[dim].set(new_high_dim)

        return ADRState(
            current_low=current_low_new,
            current_high=current_high_new,
            it=state.it + jnp.int32(1),
        )

    return init_fn, update_fn
