import jax
import jax.numpy as jnp
from flax import nnx
import bijx
from bijx.bijections.base import Bijection
from typing import Sequence, Callable, Optional, Union

# Assuming MaskedMLP is imported correctly
from learning.module.bijx.utils import BoxAffine, BoxSigmoid, MaskedMLP

import jax
import jax.numpy as jnp
from flax import nnx
import bijx
from bijx.bijections.base import Bijection
from typing import Sequence, Callable, Optional, Union

# Import your MaskedMLP (assuming it's fixed as per previous steps)
from learning.module.bijx.utils import MaskedMLP
import bijx
from bijx.bijections.base import Bijection
from learning.module.bijx.utils import BoxAffine # Assuming you still want this available

class ConditionalRQSpline:
    def __init__(
        self, 
        bins: int = 8, 
        bound: float = 3.0, 
        min_bin_width: float = 1e-3, 
        min_bin_height: float = 1e-3, 
        min_derivative: float = 1e-3
    ):
        self.bins = bins
        self.bound = bound
        self.param_dim = 3 * bins - 1 
        
        # Stability constants
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

    @property
    def param_count(self) -> int:
        return self.param_dim

    def _split_and_constrain_params(self, params: jax.Array):
        """
        Splits raw MLP outputs and enforces spline constraints:
        - Widths/Heights: Softmax (sum to 1, positive)
        - Derivatives: Softplus (strictly positive)
        """
        # 1. Split raw logits
        w_logits = params[..., :self.bins]
        h_logits = params[..., self.bins : 2 * self.bins]
        d_logits = params[..., 2 * self.bins :]

        # 2. Constrain Widths (Softmax + Min Width)
        #    Equation: w = softmax(logits) * (1 - N*min) + min
        w = jax.nn.softmax(w_logits, axis=-1)
        w = w * (1 - self.bins * self.min_bin_width) + self.min_bin_width

        # 3. Constrain Heights (Softmax + Min Height)
        h = jax.nn.softmax(h_logits, axis=-1)
        h = h * (1 - self.bins * self.min_bin_height) + self.min_bin_height

        # 4. Constrain Derivatives (Softplus + Min Slope)
        d = jax.nn.softplus(d_logits) + self.min_derivative

        return w, h, d

    def forward(self, x: jax.Array, params: jax.Array):
        # 1. Cast to float64 for calculation stability
        x_64 = x.astype(jnp.float64)
        params_64 = params.astype(jnp.float64)
        
        # 2. Get VALID spline parameters
        w, h, d = self._split_and_constrain_params(params_64)

        # 3. Handle Boundaries (Identity outside [-bound, bound])
        inside_interval = (x_64 >= -self.bound) & (x_64 <= self.bound)
        x_clamped = jnp.clip(x_64, -self.bound, self.bound)

        # 4. Normalize to [0, 1] for bijx
        x_norm = (x_clamped + self.bound) / (2 * self.bound)
        
        # 5. Compute Spline (Forward)
        #    Note: inverse=False
        z_norm, log_det_spline = bijx.rational_quadratic_spline(
            x_norm, w, h, d, inverse=False
        )
        
        # 6. Denormalize
        z_spline = z_norm * (2 * self.bound) - self.bound

        # 7. Apply Boundary Mask
        #    If outside bound, z = x (Identity) and log_det = 0
        z_final = jnp.where(inside_interval, z_spline, x_64)
        log_det_final = jnp.where(inside_interval, log_det_spline, 0.0)

        # 8. Cast back to float32
        return z_final.astype(jnp.float32), log_det_final.astype(jnp.float32)

    def inverse(self, z: jax.Array, params: jax.Array):
        # 1. Cast to float64
        z_64 = z.astype(jnp.float64)
        params_64 = params.astype(jnp.float64)

        # 2. Get VALID spline parameters
        w, h, d = self._split_and_constrain_params(params_64)
        
        # 3. Handle Boundaries
        inside_interval = (z_64 >= -self.bound) & (z_64 <= self.bound)
        z_clamped = jnp.clip(z_64, -self.bound, self.bound)
        
        # 4. Normalize
        z_norm = (z_clamped + self.bound) / (2 * self.bound)
        
        # 5. Compute Spline (Inverse)
        #    Note: inverse=True
        x_norm, log_det_spline = bijx.rational_quadratic_spline(
            z_norm, w, h, d, inverse=True
        )
        
        # 6. Denormalize
        x_spline = x_norm * (2 * self.bound) - self.bound
        
        # 7. Apply Boundary Mask
        x_final = jnp.where(inside_interval, x_spline, z_64)
        log_det_final = jnp.where(inside_interval, log_det_spline, 0.0)
        
        return x_final.astype(jnp.float32), log_det_final.astype(jnp.float32)
# --- Transform ---
class MaskedAutoregressiveTransform(Bijection):
    def __init__(
        self,
        features: int,
        rngs: nnx.Rngs,
        univariate: ConditionalRQSpline,
        passes: Optional[int] = None,
        order: Optional[jax.Array] = None,
        adjacency: Optional[jax.Array] = None,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable = nnx.relu,
    ):
        super().__init__()
        self.features = features
        self.univariate = univariate
        self.total_param_dim = self.univariate.param_count

        # --- Adjacency / Masking Logic ---
        if adjacency is None:
            if passes is None: passes = features
            if order is None: order = jnp.arange(features)
            else: order = jnp.array(order, dtype=int)
            self.passes = min(max(passes, 1), features)
            
            # Simple version of grouping for Made masks
            groups = jnp.floor(order / jnp.ceil(features / self.passes))
            adjacency = groups[:, None] > groups 
        else:
            adjacency = jnp.array(adjacency, dtype=bool)
            # Ensure diagonal is zero (autoregressive property)
            adjacency = adjacency & (~jnp.eye(features, dtype=bool))
        
        # Expand adjacency for the parameter dimension of the spline
        # shape: (features * param_dim, features) if doing dense, 
        # or handled internally by MaskedMLP logic.
        # Assuming MaskedMLP handles the repeating:
        self.hyper = MaskedMLP(
            features=features,
            hidden_features=hidden_features,
            params_per_feature=self.total_param_dim,
            rngs=rngs,
            activation=activation,
        )

    def _compute_params(self, x: jax.Array):
        out = self.hyper(x)
        return out.reshape(out.shape[:-1] + (self.features, self.total_param_dim))

    def forward(self, x: jax.Array, log_density, **kwargs):
        """
        Forward pass (Sampling direction).
        Formula: log_prob(y) = log_prob(x) - log|det J_f|
        """
        params = self._compute_params(x)
        z, log_det_per_dim = self.univariate.forward(x, params)
        
        # [FIX] Changed from + to - 
        # We must SUBTRACT the forward log-determinant.
        return z, log_density - jnp.sum(log_det_per_dim, axis=-1)

    def reverse(self, z: jax.Array, log_density, **kwargs):
        """
        Reverse pass (Density evaluation direction).
        Formula: return (x, delta) such that result = prior - delta
        This implies delta should be +log|det J_f|.
        """
        def body_fn(carry, i):
            x_curr, log_det_inv_acc = carry
            
            # 1. Compute params based on currently known x values (autoregressive)
            all_params = self._compute_params(x_curr)
            params_i = all_params[..., i, :]
            
            # 2. Invert the i-th dimension
            z_i = z[..., i]
            x_i, log_det_inv_i = self.univariate.inverse(z_i, params_i)
            
            # 3. Update x and accumulate INVERSE log det
            x_new = x_curr.at[..., i].set(x_i)
            return (x_new, log_det_inv_acc + log_det_inv_i), None

        init_x = jnp.zeros_like(z)
        init_log = jnp.zeros(z.shape[:-1])

        (x_final, total_inverse_log_det), _ = jax.lax.scan(
            body_fn, (init_x, init_log), jnp.arange(self.features)
        )
        
        return x_final, log_density - total_inverse_log_det

# --- Factory ---
def make_autoregressive_nsf_bijx(
    ndim: int,
    bins: int = 16,
    hidden_features=(64, 64),
    n_transforms: int = 3,
    seed: int = 0,
    domain_range: Union[Sequence[float], None] = None,
):
    prior = bijx.IndependentNormal(event_shape=(ndim,))
    layers = []
    key = jax.random.PRNGKey(seed)
    
    for k in range(n_transforms):
        key, k_block = jax.random.split(key)
        rngs_block = nnx.Rngs(params=int(jax.random.randint(k_block, (), 0, 1_000_000)))
        
        layers.append(MaskedAutoregressiveTransform(
            features=ndim,
            rngs=rngs_block,
            hidden_features=hidden_features,
            univariate=ConditionalRQSpline(bins=bins, bound=4.0)
        ))

    if domain_range is not None:
        low, high = domain_range
        post = BoxSigmoid(low=low, high=high)
        flow_bij = bijx.Chain( *layers, post)
    else:
        flow_bij = bijx.Chain( *layers)
    return bijx.Transformed(prior, flow_bij)

if __name__ == "__main__":
    D = 2
    print(f"Initializing for D={D}...")
    dist = make_autoregressive_nsf_bijx(D, bins=12, n_transforms=3, seed=1) # Start with 1 transform to debug

    # 1. Check Jacobian Triangularity (Autoregressive Property)
    print("Checking Autoregressive Property (Jacobian)...")
    def get_z(x):
        z, _ = dist.bijection.forward(x, 0)
        return z
        
    x_dummy = jnp.ones(D)
    jac = jax.jacfwd(get_z)(x_dummy)
    
    # Check strictly lower triangular (diagonal is allowed, upper must be 0)
    # Actually, for x->z (MAF), the Jacobian is Triangular.
    # Specifically, z_i depends on x_<=i. So dz_i/dx_j is non-zero if j <= i.
    # This means Lower Triangular.
    is_lower_triangular = jnp.allclose(jnp.triu(jac, k=1), 0.0, atol=1e-5)
    
    print(f"Jacobian:\n{jnp.round(jac, 2)}")
    print(f"Is Lower Triangular (No Leakage)? {is_lower_triangular}")
    
    if not is_lower_triangular:
        print("CRITICAL ERROR: Masking is failing. Future features are leaking into past.")
    else:
        # 2. Check Density Consistency
        key = jax.random.PRNGKey(123)
        print("\nSampling...")
        x, logp_sample = dist.sample(batch_shape=(1024,), rng=key)
        
        print("Computing density...")
        logp_density = dist.log_density(x)
        
        diff = jnp.max(jnp.abs(logp_sample - logp_density))
        print(f"Max Diff: {diff:.6e}")

    # Insert this after your Jacobian check
    print("\n--- Diagnostic: Bijector Consistency ---")
    key = jax.random.PRNGKey(55)
    x_in = jax.random.normal(key, (1, D)) # Single sample

    # 1. Forward Pass
    z_fwd, log_det_fwd = dist.bijection.forward(x_in, 0.0)

    # 2. Reverse Pass (using the z we just calculated)
    x_rec, log_density_out = dist.bijection.reverse(z_fwd, 0.0) 
    # Note: reverse returns (log_density_in - total_log_det)
    # So log_det_rev_accumulated = -log_density_out


    # 3. Analysis
    rec_error = jnp.max(jnp.abs(x_in - x_rec))
    det_sum = log_det_fwd + log_density_out  # Should be 0.0 ideally

    print(f"Original x:  {x_in[0, :3]} ...")
    print(f"Reconst x:   {x_rec[0, :3]} ...")
    print(f"Reconstruction Error (x - x_rec): {rec_error:.2e}")
    print(f"Log Det Forward: {log_det_fwd[0]:.4f}")
    print(f"Log Det Reverse: {log_density_out[0]:.4f}")
    print(f"Det Consistency (Sum should be 0): {det_sum[0]:.4f}")

    if jnp.abs(det_sum) > 1e-4:
        if jnp.isclose(log_det_fwd, log_density_out, rtol=1e-2):
            print(">>> DIAGNOSIS: Sign Error. 'reverse' returns Forward Det. Flip the sign in reverse().")
        else:
            print(">>> DIAGNOSIS: Value Mismatch. The Spline inputs (params) differ between passes.")