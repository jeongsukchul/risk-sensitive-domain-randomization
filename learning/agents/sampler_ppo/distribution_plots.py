"""Plots for AutoDR and DORAEMON distributions."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from learning.agents.sampler_ppo.adr import get_adr_sample
from learning.agents.sampler_ppo.doraemon import _log_prob_beta_on_box, _unpack_beta


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
    step: int = None,
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
        x,
        y,
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


def plot_beta_density_2d(
    state,
    low,
    high,
    ndim,
    min_bound,
    max_bound,
    grid=200,
    contexts=None,
    returns=None,
    success_threshold=0.5,
    title=None,
    ax=None,
):
    # Plots the 2D density (Beta-on-box, factorized) + optional sampled points.
    assert ndim == 2, "This plotting helper is for 2D only."
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    if ax is None:
        fig, ax = plt.subplots()

    a, b = _unpack_beta(state.x_opt, ndim, min_bound, max_bound)

    xs = jnp.linspace(low[0], high[0], grid)
    ys = jnp.linspace(low[1], high[1], grid)
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")
    pts = jnp.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)  # (grid^2,2)

    logp = _log_prob_beta_on_box(a, b, low, high, pts).reshape(grid, grid)
    p = jnp.exp(logp - jnp.max(logp))  # scaled for visualization

    ax.imshow(
        np.array(p),
        origin="lower",
        extent=[float(low[0]), float(high[0]), float(low[1]), float(high[1])],
        aspect="auto",
    )

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
    ax.set_title(title or "Beta-on-box density")
    return ax
