"""Utilities."""

__all__ = ["plot_data_spline_tangent_curvature_acceleration", "plot_theta_of_gamma"]


import galax.potential as gp
import interpax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Bool, Int, Real
from matplotlib.cm import ScalarMappable

from .custom_types import SzN
from .likelihood import (
    get_unit_curvature,
    get_unit_tangents,
)


def plot_theta_of_gamma(
    qs: Real[Array, "Q"], gammas: Real[Array, "gamma"], thetas: Real[Array, "Q gamma"]
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the angles as a function of gamma."""

    # Create colormap and normalization objects
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=jnp.min(qs), vmax=jnp.max(qs))

    fig, ax = plt.subplots(dpi=150)

    # Plot the angles for each gamma
    for q, angles in zip(qs, thetas, strict=False):
        ax.scatter(gammas, angles, color=cmap(norm(q)), s=1)

    # Add the colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r"$q$")

    # Add the exclusion lines
    ylim = 3
    ax.axhspan(jnp.pi / 2, ylim, color="gray", alpha=0.5)
    ax.axhline(jnp.pi / 2, color="k", label=r"$\pi/2$", alpha=0.5)

    ax.axhspan(-ylim, -jnp.pi / 2, color="gray", alpha=0.5)
    ax.axhline(-jnp.pi / 2, color="k", label=r"$\pi/2$", alpha=0.5)

    ax.text(-0.8, jnp.pi / 2 + 0.2, "> π/2", color="k", ha="center", va="center")
    ax.text(-0.8, -jnp.pi / 2 - 0.2, "< -π/2", color="k", ha="center", va="center")
    ax.text(0.75, 2.5, "Rule out", color="k", ha="center", va="center")

    # Add labels
    ax.set(xlabel=r"$\gamma$", ylabel=r"$\theta$", xlim=(-1, 1), ylim=(-ylim, ylim))
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
    X, Y = jnp.meshgrid(np.linspace(*xlim, grid_size), jnp.linspace(*ylim, grid_size))
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
        color="gray",
        width=vec_width,
        scale=vec_scale,
        label="Accelerations (global)",
    )

    return ax


def plot_tangents(
    gamma: SzN,
    track: interpax.Interpolator1D,
    subselect: slice | Bool[Array, "N"] | Int[Array, "..."],
    *,
    ax: plt.Axes | None = None,
    vec_width: float = 0.003,
    vec_scale: float = 30,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(dpi=150, figsize=(10, 10))

    points = track(gamma)
    tangents_hat = jax.vmap(get_unit_tangents, in_axes=(0, None))(gamma, track)

    ax.quiver(
        points[subselect, 0],
        points[subselect, 1],
        tangents_hat[subselect, 0],
        tangents_hat[subselect, 1],
        color="blue",
        scale=vec_scale,
        label=r"$\hat{T}$",
        width=vec_width,
    )
    return ax


def plot_curvature(
    gamma: SzN,
    track: interpax.Interpolator1D,
    subselect: slice | Bool[Array, "N"] | Int[Array, "..."],
    *,
    ax: plt.Axes | None = None,
    vec_width: float = 0.003,
    vec_scale: float = 30,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(dpi=150, figsize=(10, 10))

    points = track(gamma)
    curvature_hat = jax.vmap(get_unit_curvature, in_axes=(0, None))(gamma, track)

    ax.quiver(
        points[subselect, 0],
        points[subselect, 1],
        curvature_hat[subselect, 0],
        curvature_hat[subselect, 1],
        color="blue",
        scale=vec_scale,
        label=r"$\hat{\kappa}$",
        width=vec_width,
    )
    return ax


def plot_data_spline_tangent_curvature_acceleration(
    potential: gp.AbstractPotential,
    gamma_eval: Real[Array, "gamma"],
    points: Real[Array, "N 2"],
    spline: interpax.Interpolator1D,
    *,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(dpi=150, figsize=(10, 10))
    else:
        fig = ax.figure

    # Setup
    vec_width = 0.003
    vec_scale = 30
    _gamma = jnp.linspace(gamma_eval.min(), gamma_eval.max(), 500)

    # Acceleration grid
    xylim = 500
    N_acc = 20

    plot_acceleration_field(
        potential,
        xlim=(-xylim, xylim),
        ylim=(-xylim, xylim),
        ax=ax,
        grid_size=N_acc,
        vec_width=vec_width,
        vec_scale=vec_scale,
    )

    # Along the track
    ax.plot(*points.T, "o", label="Original data", zorder=-1)
    ax.plot(*spline(_gamma).T, c="red", ls="-", label="Spline fit curve")
    ax.scatter(*spline.f.T, s=10, c="red", zorder=10)

    sel = slice(None, None, 40)
    plot_tangents(
        gamma_eval, spline, sel, ax=ax, vec_width=vec_width, vec_scale=vec_scale
    )
    plot_curvature(
        gamma_eval, spline, sel, ax=ax, vec_width=vec_width, vec_scale=vec_scale
    )

    # Acceleration along the track
    points_eval = spline(gamma_eval)
    pos = jnp.stack((*points_eval.T, jnp.zeros_like(points_eval[:, 0])), axis=1)
    acc = potential.acceleration(pos, t=0)
    acc_unit = acc / np.linalg.norm(acc, axis=1, keepdims=True)
    acc_xy_unit = acc_unit[:, :2]

    N_curve_point = len(gamma_eval)
    indices = jnp.linspace(0, N_curve_point - 1, N_acc, dtype=int)
    ax.quiver(
        points_eval[indices, 0],
        points_eval[indices, 1],
        acc_xy_unit[indices, 0],
        acc_xy_unit[indices, 1],
        color="green",
        width=vec_width,
        scale=vec_scale,
        label="Accelerations (local)",
    )

    # Plot properties
    ax.set(xlabel="X (kpc)", ylabel="Y (kpc)", xlim=(-250, 250), ylim=(-250, 250))
    ax.set_aspect("equal")
    ax.legend()

    return fig, ax
