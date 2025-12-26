from typing import Sequence, Union
import jax
import jax.numpy as jnp
from flax import nnx
import bijx
from bijx.bijections.base import Bijection

from learning.module.bijx.utils import BoxAffine


class ARConditioner(nnx.Module):
    """
    Autoregressive conditioner:
      context ∈ R^{..., ndim} -> all_params ∈ R^{..., ndim, param_dim}
    Then we select params for dimension i inside the flow.
    """

    def __init__(
        self,
        ndim: int,
        param_dim: int,
        hidden=(64, 64),
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        super().__init__()
        self.ndim = ndim
        self.param_dim = param_dim

        h1, h2 = hidden
        self.l1 = nnx.Linear(ndim, h1, rngs=rngs)
        self.l2 = nnx.Linear(h1, h2, rngs=rngs)
        self.l3 = nnx.Linear(h2, ndim * param_dim, rngs=rngs)
        #initialization should not too much perturbed from base distribution
        self.l3.kernel.value = jnp.zeros_like(self.l3.kernel.value)
        self.l3.bias.value   = jnp.zeros_like(self.l3.bias.value)

    def __call__(self, context_full):
        """
        context_full: (..., ndim)  (we zero out "future" dims by masking)
        returns: (..., ndim, param_dim)
        """
        h = jax.nn.relu(self.l1(context_full))
        h = jax.nn.relu(self.l2(h))
        h = self.l3(h)  # (..., ndim * param_dim)
        return h.reshape(*h.shape[:-1], self.ndim, self.param_dim)
class AutoregressiveRQS(Bijection):
    """
    Autoregressive RQ spline flow (NSF-style) as a bijx.Bijection.

    - base_bij: scalar MonotoneRQSpline (wrapped with ModuleReconstructor)
    - conditioner: ARConditioner giving params for all dims
    """

    def __init__(
        self,
        ndim: int,
        bins: int = 16,
        hidden=(64, 64),
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        super().__init__()
        self.ndim = ndim

        # Scalar monotone RQ spline template
        base = bijx.MonotoneRQSpline(
            knots=bins,
            event_shape=(),  # scalar
            rngs=rngs,
        )
        self.base_bij = bijx.ModuleReconstructor(base)
        self.param_dim = self.base_bij.params_total_size

        # Autoregressive conditioner MLP
        self.net = ARConditioner(
            ndim=ndim,
            param_dim=self.param_dim,
            hidden=hidden,
            rngs=rngs,
        )

    # ---------- helpers ----------

    def _step_forward(self, z, y, logd, i):
        """
        One AR step in forward direction: z[..., i] -> y[..., i],
        updating logd with -log|∂y_i/∂z_i|.
        """
        # context: previous outputs y_0..y_{i-1}, future dims zeroed
        idxs = jnp.arange(self.ndim)
        mask_prev = (idxs < i).astype(z.dtype)  # (ndim,)
        # broadcast mask to batch
        while mask_prev.ndim < y.ndim:
            mask_prev = mask_prev[jnp.newaxis, ...]
        context_full = y * mask_prev  # (..., ndim)

        # all params for all dims, slice dim i
        all_params = self.net(context_full)          # (..., ndim, param_dim)
        theta_i = all_params[..., i, :]              # (..., param_dim)

        # build scalar bijection for this dim, vmapped over batch
        scalar_bij = self.base_bij.from_params(theta_i, autovmap=True)

        z_i = z[..., i]                              # (...,)
        ld0 = jnp.zeros_like(logd)
        y_i, ld1 = scalar_bij.forward(z_i, ld0)      # ld1 = -log|∂y_i/∂z_i|

        y = y.at[..., i].set(y_i)
        logd = logd + ld1
        return z, y, logd

    def _step_reverse(self, x, z, logd, i):
        """
        One AR step in reverse direction: x[..., i] -> z[..., i],
        updating logd with +log|∂z_i/∂x_i|.
        """
        idxs = jnp.arange(self.ndim)
        mask_prev = (idxs < i).astype(x.dtype)
        while mask_prev.ndim < x.ndim:
            mask_prev = mask_prev[jnp.newaxis, ...]
        context_full = x * mask_prev  # (..., ndim)

        all_params = self.net(context_full)          # (..., ndim, param_dim)
        theta_i = all_params[..., i, :]              # (..., param_dim)
        scalar_bij = self.base_bij.from_params(theta_i, autovmap=True)

        x_i = x[..., i]                              # (...,)
        ld0 = jnp.zeros_like(logd)
        z_i, ld1 = scalar_bij.reverse(x_i, ld0)      # ld1 = +log|∂z_i/∂x_i|

        z = z.at[..., i].set(z_i)
        logd = logd + ld1
        return x, z, logd

    # ---------- bijection API ----------

    def forward(self, x, log_density, **kwargs):
        """
        x: base samples z ~ N(0,I)  (shape: batch + (ndim,))
        log_density: log p_base(z)
        returns: y, log p_base(z) - log|det J(z -> y)|
        """
        z = x
        y = jnp.zeros_like(z)
        logd = log_density

        for i in range(self.ndim):
            z, y, logd = self._step_forward(z, y, logd, i)

        return y, logd

    def reverse(self, y, log_density, **kwargs):
        """
        y: transformed samples x
        log_density: typically zeros when used in log_density evaluation
        returns: z, log p_base(z) = log_density + log|det J^{-1}(y -> z)|
        """
        x = y
        z = jnp.zeros_like(x)
        logd = log_density

        for i in range(self.ndim):
            x, z, logd = self._step_reverse(x, z, logd, i)

        return z, logd


def make_autoregressive_nsf_bijx(
    ndim: int,
    bins: int = 16,
    hidden_features=(64, 64),
    n_transforms: int = 3,
    seed: int = 0,
    domain_range : Union[Sequence[float], None] = None,
):
    """
    Zuko-style NSF-AR in bijx with `n_transforms` AR blocks.

    Rough analogue of:
        MAF(
            features=ndim,
            univariate=MonotonicRQSTransform,
            shapes=[(bins,), (bins,), (bins-1,)],
            hidden_features=hidden,
            transforms=n_transforms,
        )
    Flow = Chain( AR_block_1, ..., AR_block_n_transforms )
    """

    # base distribution ~ N(0, I_ndim)
    # prior = bijx.IndependentNormal(event_shape=(ndim,))
    # prior = bijx.DiagonalNormal(jnp.zeros(ndim), jnp.ones(ndim)/2)
    prior = bijx.IndependentUniform(event_shape=(ndim,))
# 
    # build multiple AR blocks with different RNG seeds
    layers = []
    key = jax.random.PRNGKey(seed)
    for k in range(n_transforms):
        key, k_block = jax.random.split(key)
        rngs_block = nnx.Rngs(params=int(jax.random.randint(k_block, (), 0, 1_000_000)))

        ar_block = AutoregressiveRQS(
            ndim=ndim,
            bins=bins,
            hidden=hidden_features,
            rngs=rngs_block,
        )
        layers.append(ar_block)
    # pre = SigmoidToUnit()
    if domain_range is not None:
        low, high = domain_range
        post = BoxAffine(low=low, high=high)
        flow_bij = bijx.Chain( *layers, post)
    else:
        flow_bij = bijx.Chain( *layers)
    dist = bijx.Transformed(prior, flow_bij)
    return dist

