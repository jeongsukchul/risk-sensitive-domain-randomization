import jax.numpy as jnp
import jax
from matplotlib.colors import LogNorm, PowerNorm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from bijx.bijections.base import Bijection, ApplyBijection
import jax
import jax.numpy as jnp
# Assuming bijx is your library alias

class BoxSigmoid(ApplyBijection):
    """
    Elementwise box-sigmoid bijection:
        x ∈ ℝ^d  ->  y ∈ (low, high)^d
        y = low + (high - low) * sigmoid(x)
    """

    def __init__(self, low, high):
        super().__init__()
        self.low = jnp.asarray(low)
        self.high = jnp.asarray(high)
        # Calculate scale once during init
        self.scale = self.high - self.low

    def apply(self, x, log_density, reverse=False, **kwargs):
        # Ensure correct broadcasting
        low = self.low
        scale = self.scale

        if not reverse:
            # forward: ℝ -> (low, high)
            # x is in (-inf, inf)
            z = jax.nn.sigmoid(x)
            y = low + scale * z

            # Jacobian of y = low + scale * sigmoid(x)
            # J = scale * sigmoid(x) * (1 - sigmoid(x))
            # log|J| = log(scale) + log_sigmoid(x) + log_sigmoid(-x)
            
            # CORRECTED: Use x (input), not y (output)
            log_det = jnp.sum(
                jnp.log(scale) + 
                jax.nn.log_sigmoid(x) + 
                jax.nn.log_sigmoid(-x),
                axis=-1
            )

            log_density = log_density + log_det
            return y, log_density

        else:
            # reverse: (low, high) -> ℝ
            # Input x here is actually y in the domain (low, high)
            y = x 
            
            # z = (y - low) / scale
            z = (y - low) / scale
            
            # Numerical stability clipping
            z = jnp.clip(z, 1e-6, 1.0 - 1e-6)

            # Inverse sigmoid is logit
            # x_pre = log(z) - log(1-z)
            x_pre = jnp.log(z) - jnp.log1p(-z)

            # We subtract log|det J_forward| 
            # We can calculate J using z which we just computed
            log_det = jnp.sum(
                jnp.log(scale) + 
                jnp.log(z) + 
                jnp.log1p(-z),
                axis=-1
            )
            
            log_density = log_density - log_det
            return x_pre, log_density
class SigmoidToUnit(Bijection):
    def forward(self, x, logd, **kwargs):
        z = jax.nn.sigmoid(x)
        # log|det J| = sum log(sigmoid'(x)) = sum [log z + log(1-z)]
        ladj = jnp.sum(jnp.log(z) + jnp.log1p(-z), axis=-1)
        return z, logd - ladj

    def reverse(self, y, logd, **kwargs):
        y = jnp.clip(y, 1e-10, 1.0 - 1e-10)
        x = jnp.log(y) - jnp.log1p(-y)
        ladj = jnp.sum(jnp.log(y) + jnp.log1p(-y), axis=-1)
        return x, logd + ladj
class BoxAffine(Bijection):
    def __init__(self, low, high):
        super().__init__()
        self.low = jnp.asarray(low)
        self.high = jnp.asarray(high)
        self.mid = (self.low + self.high)/2
        self.scale = (self.high - self.low)
    def forward(self, x, logd, **kwargs):
        y = self.low + self.scale * x
        ladj = jnp.sum(jnp.log(jnp.abs(self.scale)), axis=-1)
        return y, logd + ladj

    def reverse(self, y, logd, **kwargs):
        x = (y - self.low) / self.scale
        ladj = jnp.sum(jnp.log(jnp.abs(self.scale)), axis=-1)
        return x, logd - ladj
def render_flow_pdf_2d_subplots(
    log_prob_fn, low, high,
    samples = None, resolution=256,
    normalize=True,
    suptitle="Flow PDF — 2D marginals via MC",
    use_wandb=False, training_step=0, ax=None, gamma=0.45
):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    # ---- constants on device
    low  = jnp.asarray(low,  jnp.float32)
    high = jnp.asarray(high, jnp.float32)
    x = jnp.linspace(low[0], high[0], resolution, dtype=jnp.float32)  # (R,)
    y = jnp.linspace(low[1], high[1], resolution, dtype=jnp.float32)  # (R,)
    XX, YY = jnp.meshgrid(x, y, indexing="xy")               # (R, R)

    grid_flat_i = XX.reshape(-1)  # (R*R,)
    grid_flat_j = YY.reshape(-1)  # (R*R,)

    X = jnp.zeros((resolution*resolution, 2))
    X = X.at[:, 0].set(grid_flat_i)
    X = X.at[:, 1].set(grid_flat_j)
    logp =log_prob_fn(X).reshape(resolution, resolution)  # (R, R)

    # stabilize + exp
    logp_shift = logp - jnp.max(logp)
    pdf = jnp.exp(logp_shift)  # unnormalized marginal on the box

    if normalize:
        dx = (high[0] - low[0]) / (resolution - 1)
        dy = (high[1] - low[1]) / (resolution - 1)
        mass = jnp.sum(pdf) * dx * dy
        pdf = jnp.where(mass > 0, pdf / mass, pdf)
   
    # ---- plotting
    ctf = ax.contourf(x, y, pdf, levels=10, cmap="viridis",  linewidths=0.6, norm=PowerNorm(gamma))
    # ctf=ax.pcolormesh(x, y, pdf, shading="nearest")
    plt.colorbar(ctf, ax=ax, fraction=0.046, pad=0.04)
    if samples is not None:
      idx = jax.random.choice(jax.random.PRNGKey(0), samples.shape[0], (300,))
      sample_x = samples[idx,0]
      sample_y = samples[idx,1]
      # sample_x = jnp.clip(samples[idx, 0],low[0], high[0])
      # sample_y = jnp.clip(samples[idx, 1],low[1], high[1])
      ax.scatter(sample_x, sample_y, c='r', alpha=0.5, marker='x')
      ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
      ax.yaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
    if suptitle:
        plt.suptitle(suptitle, y=0.99, fontsize=10)

    if use_wandb:
        try:
            import wandb
            wandb.log(
                {f"Sampler Heatmap": wandb.Image(fig)},
                step=int(training_step),
            )
        except Exception as e:
            print(f"[render_flow_pdf_2d_subplots] W&B log failed: {e}")
    if fig is not None:
        return fig, ax
    else:
        return ax
