from typing import Sequence, Union
import jax
import jax.numpy as jnp
import bijx
from flax import nnx
import flax.linen as nn
from functools import partial
from bijx.bijections.base import Bijection, ApplyBijection

from learning.module.bijx.utils import BoxAffine, BoxSigmoid
"""Code builds on https://github.com/google-deepmind/annealed_flow_transport """



class InvertiblePLU(ApplyBijection):
    def __init__(self, ndim: int, rngs: nnx.Rngs = nnx.Rngs(0)):
        super().__init__()

        self.ndim = ndim
        # w_init = jnp.eye(ndim, dtype=jnp.float32)
        w_init = nnx.initializers.orthogonal()(rngs.params(), (ndim, ndim), jnp.float32)
        P, L, U_full = jax.scipy.linalg.lu(w_init)

        s_init = jnp.diag(U_full)
        U_strict = jnp.triu(U_full - jnp.diag(s_init), k=1)

        self.perm = jnp.argmax(P, axis=0).astype(jnp.int32)
        self.inv_perm = jnp.argsort(self.perm).astype(jnp.int32)
        # self.perm = nnx.Variable(perm)
        # self.inv_perm = nnx.Variable(inv_perm)

        self.L_free = nnx.Param(jnp.tril(L, k=-1).astype(jnp.float32))
        self.U_free = nnx.Param(jnp.triu(U_strict, k=1).astype(jnp.float32))
        self.s_free = nnx.Param(s_init.astype(jnp.float32))
        # s = jnp.maximum(jnp.abs(s_init), 1e-6).astype(jnp.float32)
        # sign_s = jnp.sign(s_init).astype(jnp.float32)
        # self.sign_s = jnp.where(sign_s == 0.0, 1.0, sign_s)
        # self.log_s = nnx.Param(jnp.log(abs_s))

    @staticmethod
    def _right_solve_triangular(y, A, *, lower: bool, unit_diagonal: bool = False):
        d = y.shape[-1]
        y2 = y.reshape(-1, d).T  # (d, N)
        sol = jax.scipy.linalg.solve_triangular(
            A.T, y2, lower=(not lower), unit_diagonal=unit_diagonal
        )
        return sol.T.reshape(y.shape)
    
    def apply(self, x, log_density, reverse: bool, **kwargs):
        d = self.ndim

        L = jnp.tril(self.L_free.get_value(), k=-1) + jnp.eye(d, dtype=x.dtype)
        U = jnp.triu(self.U_free.get_value(), k=1)
        s = self.s_free.get_value()
        # raw = jnp.clip(self.log_s.get_value(), -8.0, 8.0).astype(x.dtype)
        # s = self.sign_s.astype(x.dtype) * jnp.exp(raw)
        Ubar = U + jnp.diag(s)

        log_s = jnp.log(jnp.abs(s) + 1e-6) # Add epsilon for safety
        logdet = jnp.sum(log_s).astype(x.dtype)
        def matmul_prec(a, b):
                # a: (..., d), b: (d, d)  -> (..., d)
                return jnp.einsum('...i,ij->...j', a, b, precision=jax.lax.Precision.HIGHEST)
        if not reverse:
            # y = x @ (P L Ubar)
            y = x[..., self.perm]
            y = matmul_prec(y, L) #y @ L
            y = matmul_prec(y, Ubar) #y @ Ubar

            if log_density is None:
                return y, None
            return y, log_density + logdet

        # reverse: x = y @ (Ubar^{-1} L^{-1} P^{-1})
        y = self._right_solve_triangular(x, Ubar, lower=False, unit_diagonal=False)
        y = self._right_solve_triangular(y, L, lower=True, unit_diagonal=True)
        y = y[..., self.inv_perm]

        if log_density is None:
            return y, None
        return y, log_density - logdet

class TinyMLP(nnx.Module):
    """
    Autoregressive conditioner:
      context ∈ R^{..., ndim} -> all_params ∈ R^{..., ndim, param_dim}
    Then we select params for dimension i inside the flow.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 256,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        super().__init__()
        self.n1 = nnx.LayerNorm(num_features=hidden, rngs=rngs)
        self.n2 = nnx.LayerNorm(num_features=hidden, rngs=rngs)
        # kernel: U[-k, k] with k = sqrt(1 / fan_in)
        def kernel_init(key, shape, dtype=jnp.float32):
            fan_in = shape[0]  # Linear kernel shape = (in_features, out_features)
            k = jnp.sqrt(1.0 / fan_in).astype(dtype)
            return jax.random.uniform(key, shape, dtype=dtype, minval=-k, maxval=k)

        # bias: same k, but fan_in is not inferable from bias shape (out_features,)
        def make_bias_init(fan_in: int):
            def bias_init(key, shape, dtype=jnp.float32):
                k = jnp.sqrt(1.0 / fan_in).astype(dtype)
                return jax.random.uniform(key, shape, dtype=dtype, minval=-k, maxval=k)
            return bias_init
        self.l1 = nnx.Linear(
            in_dim, hidden,
            rngs=rngs,
            kernel_init=kernel_init,
            bias_init=make_bias_init(in_dim),
        )
        self.l2 = nnx.Linear(
            hidden, hidden,
            rngs=rngs,
            kernel_init=kernel_init,
            bias_init=make_bias_init(hidden),
        )

        # last layer: start at 0 to avoid perturbing the base distribution
        self.l3 = nnx.Linear(
            hidden, out_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
        )

    def __call__(self, x):
        """
        context_full: (..., ndim)  (we zero out "future" dims by masking)
        returns: (..., ndim, param_dim)
        """
        h = self.n1(jax.nn.leaky_relu(self.l1(x)))
        h = self.n2(jax.nn.leaky_relu(self.l2(h)))
        h = self.l3(h)  # (..., ndim * param_dim)
        return h
class MetaBlock(Bijection):

    def __init__(
        self,
        ndim:int,       
        channels: int = 256,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):

        self.plu = InvertiblePLU(ndim, rngs)
        self.split_idx = ndim // 2
        cond_dim = self.split_idx
        trans_dim = ndim - self.split_idx
        # t-network: inputs are x_cond only (no y)
        self.s_net = TinyMLP(in_dim=cond_dim, out_dim=trans_dim, hidden=channels, rngs=rngs)
        self.t_net = TinyMLP(in_dim=cond_dim, out_dim=trans_dim, hidden=channels, rngs=rngs)


    def forward(self, x, log_density, **kwargs):
        x, log_det = self.plu.forward(x, log_density)  # (B, Dx), (B,)
        x_cond, x_trans = x[..., :self.split_idx], x[..., self.split_idx:]
        s = self.s_net(x_cond)
        t = self.t_net(x_cond)
        x_trans = (x_trans - t) * jnp.exp(-s)
        x = jnp.concatenate((x_cond, x_trans), axis=1)
        log_det = log_det - jnp.sum(s, axis=1)  # (B,)
        return x, log_det

    def reverse(self, z, log_density, **kwargs):
        z_cond, z_trans = jnp.array_split(z, 2, axis=1)
        s = self.s_net(z_cond)
        t = self.t_net(z_cond)
        z_trans = z_trans * jnp.exp(s) + t
        z = jnp.concatenate((z_cond, z_trans), axis=1)
        log_det = log_density + jnp.sum(s, axis=1)  # (B,)
        z, log_det = self.plu.reverse(z, log_det)  # (B, Dx), (B,)
        return z, log_det


def make_realnvp_bijx(
    ndim: int,
    n_layers: int = 8,
    channels: int =256,
    seed: int = 0,
    domain_range: Union[Sequence[float], None] = None,
) -> bijx.Distribution:
    prior = bijx.IndependentNormal(event_shape=(ndim,))
    # prior = bijx.IndependentUniform(event_shape=(ndim,))


    layers = []
    key = jax.random.PRNGKey(seed)
    for _ in range(n_layers):
        key, k_block = jax.random.split(key)
        rngs_block = nnx.Rngs(params=int(jax.random.randint(k_block, (), 0, 1_000_000)))
        layers.append(MetaBlock(
            ndim,
            channels,
            rngs_block,
        ))

    
    if domain_range is not None:
        low, high = domain_range
        # post = BoxAffine(low=low, high=high)
        post = BoxSigmoid(low=low, high=high)
        flow_bij = bijx.Chain(*layers, post)
    else:
        flow_bij = bijx.Chain(*layers)

    return bijx.Transformed(prior, flow_bij)


# --- 3. Run Check ---
if __name__ == "__main__":
    # jax.config.update("jax_enable_x64", True)
    # jax.config.update("jax_default_matmul_precision", "highest")
    D = 6
    print(f"Initializing for D={D}...")
    key  = jax.random.PRNGKey(1)
    rngs = nnx.Rngs(params=int(jax.random.randint(key, (), 0, 1_000_000)))
    bound_info = jnp.zeros(D), jnp.ones(D)
    dist = make_realnvp_bijx(D, n_layers=4, channels=256, seed=42, domain_range=bound_info)
    # prior = bijx.IndependentNormal(event_shape=(D,))
    # dist = bijx.Transformed(prior, InvertiblePLU(D, ))

    key = jax.random.PRNGKey(2)
    print("Sampling...")
    x, logp = dist.sample(batch_shape=(16,), rng=key)
    
    print("Computing log_density...")
    logp2 = dist.log_density(x)

    print("\nSuccess!")
    print("x.shape   :", x.shape)
    print("logp      :", logp)
    print("logp2      :", logp2)
    print("diff      :", jnp.max(jnp.abs(logp - logp2)))