"""Utilities."""

__all__ = [
    "get_angles",
    "plot_acceleration_field",
    "plot_theta_of_gamma",
]


import functools as ft

import galax.potential as gp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Real
from matplotlib.cm import ScalarMappable

from .custom_types import SzN, SzN2

PI_ON_2 = np.pi / 2


@ft.partial(jax.jit)
def get_angles(acc_xy_unit: SzN2, kappa_hat: SzN2) -> SzN:
    r"""Return angle between the normal and acceleration vectors at a position.

    Calculate the angles between the normal vector at given position along the
    stream and the acceleration at given position along the stream.

    Parameters
    ----------
    acc_xy_unit
        An array representing the planar acceleration at each input position.
        Shape (N, 2).
    kappa_hat
        The unit curvature vector (or named normal vector). Shape (N, 2).

    Returns
    -------
    angles
        An array of angles in radians in the range (-pi, pi), with shape (N,).

    """
    dot_product = jnp.einsum("ij,ij->i", acc_xy_unit, kappa_hat)
    cross_product = jnp.cross(acc_xy_unit, kappa_hat)
    return jnp.atan2(cross_product, dot_product)


PI_ON_2 = np.pi / 2


def plot_theta_of_gamma(
    gamma: Real[Array, "gamma"],
    param: Real[Array, "param"],
    angles: Real[Array, "param gamma"],
    *,
    mle_idx: int | None = None,
    param_label: str = r"$q$",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the angles as a function of gamma."""
    # Create colormap and normalization objects
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=jnp.min(param), vmax=jnp.max(param))

    fig, ax = plt.subplots(dpi=150)

    # Plot the angles for each gamma
    ax.scatter(
        jnp.tile(gamma, angles.shape[0]),
        angles.ravel(),
        c=jnp.repeat(param, len(gamma)),
        cmap=cmap,
        norm=norm,
        s=1,
    )

    # Add the colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(param_label)

    # Exclusion line for pi/2 to pi
    ax.axhspan(PI_ON_2, np.pi, color="gray", alpha=0.5)
    ax.axhline(PI_ON_2, color="k", alpha=0.5)
    ax.text(
        -0.85, PI_ON_2 + 0.2, r"$\theta > \pi/2$", color="k", ha="center", va="center"
    )
    ax.text(0.75, 2.5, "Ruled out", color="k", ha="center", va="center")

    # Exclusion line for -pi to -pi/2
    ax.axhspan(-np.pi, -PI_ON_2, color="gray", alpha=0.5)
    ax.axhline(-PI_ON_2, color="k", alpha=0.5)
    ax.text(
        -0.85, -PI_ON_2 - 0.2, r"$\theta < -\pi/2$", color="k", ha="center", va="center"
    )

    # MLE point
    if mle_idx is not None:
        ax.plot(gamma, angles[mle_idx], c="red", lw=3, label=r"MLE$^*$")
        ax.legend()

    # Plot properties
    ax.set(xlabel=r"$\gamma$", ylabel=r"$\theta$", ylim=(-np.pi, np.pi))
    ax.legend()
    ax.minorticks_on()

    return fig, ax


# =============================================================================


def plot_acceleration_field(
    potential: gp.AbstractPotential,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int = 20,
    ax: plt.Axes | None = None,
    vec_width: float = 0.003,
    vec_scale: float = 30,
) -> plt.Axes:
    """Plot the acceleration field of a potential."""
    if ax is None:
        _, ax = plt.subplots(dpi=150, figsize=(10, 10))

    # Position grid
    X, Y = jnp.meshgrid(
        np.linspace(*xlim, grid_size),
        jnp.linspace(*ylim, grid_size),
    )
    Z = jnp.zeros_like(X)
    pos_grid = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # Acceleration grid
    acc_grid = potential.acceleration(pos_grid, t=0)
    acc_hat_grid = acc_grid / np.linalg.norm(acc_grid, axis=1, keepdims=True)

    ax.quiver(
        X,
        Y,
        acc_hat_grid[:, 0],
        acc_hat_grid[:, 1],
        color=(0.5, 0.56, 0.5),  # gray-green
        width=vec_width,
        scale=vec_scale,
        label=r"$\vec{a}$ (global)",
        alpha=0.5,
    )

    return ax
