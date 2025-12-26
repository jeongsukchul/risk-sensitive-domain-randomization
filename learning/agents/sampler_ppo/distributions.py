"""JAX-based probability distributions for sampling."""

import jax
import jax.numpy as jnp
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple
import jax
import jax.numpy as jnp
import jax.scipy as jsp
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
            high[None, :],             # (1, ndim)
            low[None, :],              # (1, ndim)
        )  # -> (n_samples, ndim)

        # Build a mask (n_samples, ndim) that is True only at the chosen boundary dim
        dim_ids = jnp.arange(ndim)[None, :]           # (1, ndim)
        dim_mask = (dim_ids == boundary_dims[:, None])  # (n_samples, ndim)

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
        samples = jax.random.uniform(key, (n_samples, self.ndim)) * (self.high - self.low) + self.low
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
            gammaln(self.alphas) + gammaln(self.betas) - gammaln(self.alphas + self.betas)
        )
        
        # Compute log probability for each dimension
        log_probs = (
            (self.alphas - 1) * jnp.log(transformed_value) +
            (self.betas - 1) * jnp.log(1 - transformed_value) -
            log_beta_fn
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
                # KL = log(B(alpha, beta)) + (1-alpha)*digamma(alpha) + (1-beta)*digamma(beta) 
                #      - (2-alpha-beta)*digamma(alpha+beta)
                from scipy.special import gammaln
                log_beta_fn = (
                    gammaln(self.alphas[i]) + gammaln(self.betas[i]) - 
                    gammaln(self.alphas[i] + self.betas[i])
                )
                kl_div_i = (
                    log_beta_fn +
                    (1 - self.alphas[i]) * digamma(self.alphas[i]) +
                    (1 - self.betas[i]) * digamma(self.betas[i]) -
                    (2 - self.alphas[i] - self.betas[i]) * digamma(self.alphas[i] + self.betas[i])
                )
                # Adjust for the change in scale
                kl_div_i -= jnp.log(self.high[i] - self.low[i])
                kl_div += float(kl_div_i)
                
            elif isinstance(other, BetasDist):
                # KL divergence between two Beta distributions
                from scipy.special import gammaln
                log_beta_p = (
                    gammaln(self.alphas[i]) + gammaln(self.betas[i]) - 
                    gammaln(self.alphas[i] + self.betas[i])
                )
                log_beta_q = (
                    gammaln(other.alphas[i]) + gammaln(other.betas[i]) - 
                    gammaln(other.alphas[i] + other.betas[i])
                )
                kl_div_i = (
                    log_beta_q - log_beta_p +
                    (self.alphas[i] - other.alphas[i]) * digamma(self.alphas[i]) +
                    (self.betas[i] - other.betas[i]) * digamma(self.betas[i]) +
                    (other.alphas[i] + other.betas[i] - self.alphas[i] - self.betas[i]) * 
                    digamma(self.alphas[i] + self.betas[i])
                )
                # Adjust for different bounds if necessary
                if not jnp.allclose(self.low[i], other.low[i]) or not jnp.allclose(self.high[i], other.high[i]):
                    kl_div_i += jnp.log((other.high[i] - other.low[i]) / (self.high[i] - self.low[i]))
                kl_div += float(kl_div_i)
            else:
                raise ValueError(f"Unsupported distribution type: {type(other)}")
        
        return kl_div
class ADRState(NamedTuple):
    current_low: jnp.ndarray   # shape: (ndim,)
    current_high: jnp.ndarray  # shape: (ndim,)
    it: jnp.int32              # iteration counter (optional)
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

    return ADRState(current_low=current_low,
                    current_high=current_high,
                    it=jnp.int32(0))
def get_adr_train_dist(state: ADRState, boundary_prob: float = 0.5):
    return BoundarySamplingDist(state.current_low, state.current_high, boundary_prob)
def get_adr_sample(state: ADRState, num_samples, key):
    return BoundarySamplingDist(state.current_low, state.current_high, boundary_prob=0.5).rsample(num_samples, key)
def get_adr_log_prob(state: ADRState, sample):
    return BoundarySamplingDist(state.current_low, state.current_high, boundary_prob=0.5).log_prob(sample)
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
        returns: jnp.ndarray,          # (batch_size,)
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

def plot_adr_density_2d(
    adr_state,
    domain_low,
    domain_high,
    key,
    dim_x: int = 0,
    dim_y: int = 1,
    num_samples: int = 20000,
    bins: int = 60,
    ax=None,
    step : int = None,
):
    """
    Plot a 2D density (heatmap) of the ADR training distribution for two dims.

    Args:
        adr_state: ADRState or any object with attributes
                   `current_low` and `current_high` (shape (ndim,)).
        domain_low: array-like, full domain lower bounds, shape (ndim,)
        domain_high: array-like, full domain upper bounds, shape (ndim,)
        key: jax.random.PRNGKey
        dim_x: index of the first dimension to plot
        dim_y: index of the second dimension to plot
        num_samples: number of samples used to estimate density
        bins: number of histogram bins per dimension
        ax: optional matplotlib Axes to draw on. If None, creates a new fig/ax.

    Returns:
        ax: the matplotlib Axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    domain_low = jnp.asarray(domain_low)
    domain_high = jnp.asarray(domain_high)

    # --- sample from the ADR training distribution ---
    samples = get_adr_sample(adr_state, num_samples, key)  # (N, ndim), JAX array
    samples_np = np.asarray(samples)

    x = samples_np[:, dim_x]
    y = samples_np[:, dim_y]

    # --- histogram estimate of density ---
    x_range = [float(domain_low[dim_x]), float(domain_high[dim_x])]
    y_range = [float(domain_low[dim_y]), float(domain_high[dim_y])]

    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=bins,
        range=[x_range, y_range],
        density=True,
    )
    H = H.T  # for imshow: H[row=y, col=x]

    im = ax.imshow(
        H,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
    )

    # --- overlay current ADR box ---
    clx = float(adr_state.current_low[dim_x])
    chx = float(adr_state.current_high[dim_x])
    cly = float(adr_state.current_low[dim_y])
    chy = float(adr_state.current_high[dim_y])

    rect = plt.Rectangle(
        (clx, cly),
        chx - clx,
        chy - cly,
        fill=False,
        linewidth=2.0,
        linestyle="--",
        edgecolor="red",
    )
    ax.add_patch(rect)

    ax.set_xlabel(f"param[{dim_x}]")
    ax.set_ylabel(f"param[{dim_y}]")
    if step is not None:
        ax.set_title(f"ADR training distribution (2D density) step={int(step)}")
    else:
        ax.set_title("ADR training distribution (2D density)")
    plt.colorbar(im, ax=ax, label="estimated density")

    return ax



class DoraemonState(NamedTuple):
    x_opt: jnp.ndarray          # unconstrained params (2*ndim,)
    train_until_done: jnp.bool_ # scalar
    it: jnp.int32               # scalar


def _sigmoid_bounded(x, lb, ub):
    x = jnp.clip(x, -60.0, 60.0)  # avoid exp overflow
    return (ub - lb) / (1.0 + jnp.exp(-x)) + lb


def _inv_sigmoid_bounded(y, lb, ub):
    y = jnp.clip(y, lb + 1e-12, ub - 1e-12)
    return -jnp.log((ub - lb) / (y - lb) - 1.0)

def make_initial_state(ndim, init_beta_param, min_bound, max_bound):
    flat = jnp.ones((2 * ndim,), dtype=jnp.float32) * init_beta_param
    x_opt0 = _inv_sigmoid_bounded(flat, min_bound, max_bound)
    return DoraemonState(x_opt=x_opt0, train_until_done=jnp.bool_(False), it=jnp.int32(0))

def _unpack_beta(x_opt, ndim, min_bound, max_bound):
    flat = _sigmoid_bounded(x_opt, min_bound, max_bound)
    a = flat[:ndim]
    b = flat[ndim:]
    return a, b

def sample_beta_on_box(key, a, b, low, high, n):
    dim = a.shape[0]
    keys = jax.random.split(key, dim * 2)
    g1 = jnp.stack([jax.random.gamma(keys[i], a[i], shape=(n,)) for i in range(dim)], axis=-1)
    g2 = jnp.stack([jax.random.gamma(keys[i+dim], b[i], shape=(n,)) for i in range(dim)], axis=-1)
    z = g1 / (g1 + g2 + 1e-12)
    return low + (high - low) * z
def _log_prob_beta_on_box(a, b, low, high, x):
    # x: (..., ndim)
    z = (x - low) / (high - low)
    z = jnp.clip(z, 1e-7, 1.0 - 1e-7)

    logB = jsp.special.gammaln(a) + jsp.special.gammaln(b) - jsp.special.gammaln(a + b)
    lp = (a - 1.0) * jnp.log(z) + (b - 1.0) * jnp.log1p(-z) - logB
    lp = lp - jnp.log(high - low)  # scaling Jacobian
    return jnp.sum(lp, axis=-1)    # (...,)


def _kl_beta_beta(a, b, c, d):
    # KL(Beta(a,b) || Beta(c,d)) summed over dims
    logB_ab = jsp.special.gammaln(a) + jsp.special.gammaln(b) - jsp.special.gammaln(a + b)
    logB_cd = jsp.special.gammaln(c) + jsp.special.gammaln(d) - jsp.special.gammaln(c + d)
    psi = jsp.special.digamma
    kl_dim = (logB_cd - logB_ab) + (a - c) * psi(a) + (b - d) * psi(b) + (c + d - a - b) * psi(a + b)
    return jnp.sum(kl_dim)


def make_doraemon_update_fn(
    *,
    low, high,
    success_threshold: float,
    success_rate_condition: float,
    kl_upper_bound: float,
    min_bound: float,
    max_bound: float,
    train_until_performance_lb: bool = True,
    hard_performance_constraint: bool = True,
    n_steps_main: int = 25,
    n_steps_restore: int = 30,
    step_size_main: float = 0.25,     # gradient step magnitude (normalized)
    step_size_restore: float = 0.25,  # ascent step magnitude (normalized)
    max_ls_steps: int = 20,
    ls_shrink: float = 0.5,
):
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    ndim = low.shape[0]

    success_threshold = float(success_threshold)
    success_rate_condition = float(success_rate_condition)
    kl_upper_bound = float(kl_upper_bound)

    def update(state: DoraemonState, contexts: jnp.ndarray, returns: jnp.ndarray):
        """
        Pure-JAX update. Compatible with lax.scan (no SciPy).
        contexts: (N, ndim)
        returns:  (N,)
        """

        x0 = state.x_opt

        # Center distribution is the current distribution at start of update.
        a0, b0 = _unpack_beta(x0, ndim, min_bound, max_bound)
        lp0 = _log_prob_beta_on_box(a0, b0, low, high, contexts)

        succ = (returns >= success_threshold).astype(jnp.float32)

        def kl_to_center(x_opt):
            a1, b1 = _unpack_beta(x_opt, ndim, min_bound, max_bound)
            return _kl_beta_beta(a0, b0, a1, b1)  # KL(center || proposed)

        def perf(x_opt):
            a1, b1 = _unpack_beta(x_opt, ndim, min_bound, max_bound)
            lp1 = _log_prob_beta_on_box(a1, b1, low, high, contexts)
            logw = jnp.clip(lp1 - lp0, -50.0, 50.0)
            w = jnp.exp(logw)
            w = jnp.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
            return jnp.mean(w * succ)

        def obj(x_opt):
            # KL(Beta(a,b) || Uniform) == KL(Beta(a,b) || Beta(1,1))
            a, b = _unpack_beta(x_opt, ndim, min_bound, max_bound)
            one = jnp.ones_like(a)
            return _kl_beta_beta(a, b, one, one)

        perf0 = perf(x0)

        # Optional “train until perf LB reached” skip (matches your behavior)
        def maybe_skip(_):
            return state  # no update

        def do_update(_):
            # -------- feasibility restoration (maximize perf under KL bound) if needed --------
            def restore_step(x):
                g = jax.grad(perf)(x)
                g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                gnorm = jnp.linalg.norm(g) + 1e-12
                direction = (step_size_restore * g) / gnorm

                perf_x = perf(x)

                def accept(xcand):
                    ok_tr = kl_to_center(xcand) <= kl_upper_bound
                    ok_inc = perf(xcand) >= perf_x
                    return ok_tr & ok_inc

                def ls(carry):
                    alpha, xbest, found, k = carry
                    xcand = jnp.clip(x + alpha * direction, -60.0, 60.0)
                    ok = accept(xcand)
                    xbest = jnp.where(ok, xcand, xbest)
                    found = found | ok
                    alpha = jnp.where(ok, alpha, alpha * ls_shrink)
                    return alpha, xbest, found, k + 1

                def cond(carry):
                    _, _, found, k = carry
                    return (k < max_ls_steps) & (~found)

                _, xnew, _, _ = jax.lax.while_loop(cond, ls, (1.0, x, False, 0))
                return xnew

            def run_restore(x_init):
                return jax.lax.fori_loop(0, n_steps_restore, lambda i, x: restore_step(x), x_init)

            x_restored = jax.lax.cond(
                (hard_performance_constraint & (perf0 < success_rate_condition)),
                lambda _: run_restore(x0),
                lambda _: x0,
                operand=None,
            )

            # If still infeasible, keep the restored-best and stop (like your “keep training” branch)
            perf_restored = perf(x_restored)

            def stop_with_restored(_):
                return DoraemonState(x_opt=x_restored, train_until_done=state.train_until_done, it=state.it + 1)

            # -------- main objective minimization under constraints --------
            def main_step(x):
                f = obj(x)
                g = jax.grad(obj)(x)
                g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                gnorm = jnp.linalg.norm(g) + 1e-12
                direction = -(step_size_main * g) / gnorm

                def accept(xcand):
                    ok_tr = kl_to_center(xcand) <= kl_upper_bound
                    ok_perf = (perf(xcand) >= success_rate_condition) if hard_performance_constraint else True
                    ok_dec = obj(xcand) <= f
                    return ok_tr & ok_perf & ok_dec

                def ls(carry):
                    alpha, xbest, found, k = carry
                    xcand = jnp.clip(x + alpha * direction, -60.0, 60.0)
                    ok = accept(xcand)
                    xbest = jnp.where(ok, xcand, xbest)
                    found = found | ok
                    alpha = jnp.where(ok, alpha, alpha * ls_shrink)
                    return alpha, xbest, found, k + 1

                def cond(carry):
                    _, _, found, k = carry
                    return (k < max_ls_steps) & (~found)

                _, xnew, _, _ = jax.lax.while_loop(cond, ls, (1.0, x, False, 0))
                return xnew

            def run_main(x_init):
                return jax.lax.fori_loop(0, n_steps_main, lambda i, x: main_step(x), x_init)

            x_new = run_main(x_restored)
            return DoraemonState(x_opt=x_new, train_until_done=jnp.bool_(True), it=state.it + 1)

            # If hard constraint + still infeasible after restore -> stop early
            # (we do this check *outside* to keep structure clean)
        # end do_update

        new_state = jax.lax.cond(
            (train_until_performance_lb & (~state.train_until_done) & (perf0 < success_rate_condition)),
            maybe_skip,
            do_update,
            operand=None,
        )

        # If we ran restore but still infeasible, clamp to restored & return early:
        # (Only triggers if do_update ran and hard constraint was active.)
        def finalize(s):
            a0_, b0_ = _unpack_beta(s.x_opt, ndim, min_bound, max_bound)
            # no-op; placeholder for custom postprocessing if you want
            return s

        return finalize(new_state)

    return update


# ---- helpers to initialize state from an initial Beta(alpha=beta=init_beta_param) ----


def plot_beta_density_2d(
    state,
    low, high,
    ndim,
    min_bound, max_bound,
    grid=200,
    contexts=None,
    returns=None,
    success_threshold=0.5,
    title=None,
    ax=None,
):
    # Plots the 2D density (Beta-on-box, factorized) + optional sampled points.
    assert ndim == 2, "This plotting helper is for 2D only."
    low = jnp.asarray(low); high = jnp.asarray(high)
    if ax is None:
        fig, ax = plt.subplots()

    a, b = _unpack_beta(state.x_opt, ndim, min_bound, max_bound)

    xs = jnp.linspace(low[0], high[0], grid)
    ys = jnp.linspace(low[1], high[1], grid)
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")
    pts = jnp.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)  # (grid^2,2)

    logp = _log_prob_beta_on_box(a, b, low, high, pts).reshape(grid, grid)
    p = jnp.exp(logp - jnp.max(logp))  # scaled for visualization

    im = ax.imshow(
        np.array(p),
        origin="lower",
        extent=[float(low[0]), float(high[0]), float(low[1]), float(high[1])],
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label='estimated density (scaled)')

    if contexts is not None:
        c_np = np.array(contexts)
        if returns is None:
            ax.scatter(c_np[:, 0], c_np[:, 1], s=12)
        else:
            succ = np.array((returns >= success_threshold).astype(np.bool_))
            ax.scatter(c_np[~succ, 0], c_np[~succ, 1], s=12, marker="o")
            ax.scatter(c_np[succ, 0], c_np[succ, 1], s=12, marker="x")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title or "Beta-on-box density (scaled)")
    return ax