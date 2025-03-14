"""Utilities."""

__all__ = ["plot_data_spline_tangent_curvature_acceleration", "plot_theta_of_gamma"]


import galax.potential as gp
import interpax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Real
from matplotlib.cm import ScalarMappable

from .likelihood import get_unit_tangents_and_curvature


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


def plot_data_spline_tangent_curvature_acceleration(
    potential: gp.AbstractPotential,
    gamma_eval: Real[Array, "gamma"],
    points: Real[Array, "N 2"],
    spline: interpax.Interpolator1D,
) -> tuple[plt.Figure, plt.Axes]:
    # Setup
    vec_width = 0.003
    vec_scale = 30
    _gamma = jnp.linspace(gamma_eval.min(), gamma_eval.max(), 500)

    # Points, and derivatives
    points_eval = spline(gamma_eval)
    tangent_hat, kappa_hat = get_unit_tangents_and_curvature(gamma_eval, spline)
    pos = jnp.stack((*points_eval.T, jnp.zeros_like(points_eval[:, 0])), axis=1)

    # Acceleration grid
    xylim = 500
    N_acc = 20
    X_acc_grid, Y_acc_grid = jnp.meshgrid(
        np.linspace(-xylim, xylim, N_acc),
        jnp.linspace(-xylim, xylim, N_acc),
    )
    Z_acc_grid = jnp.zeros_like(X_acc_grid)

    pos_grid = jnp.stack(
        [X_acc_grid.ravel(), Y_acc_grid.ravel(), Z_acc_grid.ravel()], axis=1
    )
    acc_grid = potential.acceleration(pos_grid, t=0)
    acc_grid_unit = acc_grid / np.linalg.norm(acc_grid, axis=1, keepdims=True)
    acc_grid_xy_unit = acc_grid_unit[:, :2]

    acc = potential.acceleration(pos, t=0)
    acc_unit = acc / np.linalg.norm(acc, axis=1, keepdims=True)
    acc_xy_unit = acc_unit[:, :2]

    N_curve_point = len(gamma_eval)
    indices = jnp.linspace(0, N_curve_point - 1, N_acc, dtype=int)

    # Plot
    fig, ax = plt.subplots(dpi=150, figsize=(10, 10))

    ax.quiver(
        X_acc_grid,
        Y_acc_grid,
        acc_grid_xy_unit[:, 0],
        acc_grid_xy_unit[:, 1],
        color="gray",
        width=vec_width,
        scale=vec_scale,
        label="Accelerations (global)",
    )

    ax.plot(*points.T, "o", label="Original data", zorder=-1)
    ax.plot(*spline(_gamma).T, c="red", ls="-", label="Spline fit curve")
    ax.scatter(*spline.f.T, s=10, c="red", zorder=10)

    sel = slice(None, None, 40)

    ax.quiver(
        points_eval[sel, 0],
        points_eval[sel, 1],
        tangent_hat[sel, 0],
        tangent_hat[sel, 1],
        color="blue",
        scale=vec_scale,
        label="Tangent",
        width=vec_width,
    )
    ax.quiver(
        points_eval[sel, 0],
        points_eval[sel, 1],
        kappa_hat[sel, 0],
        kappa_hat[sel, 1],
        color="red",
        scale=vec_scale,
        label="Curvature",
        width=vec_width,
    )

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

    ax.set(xlabel="X (kpc)", ylabel="Y (kpc)", xlim=(-250, 250), ylim=(-250, 250))
    ax.set_aspect("equal")
    ax.legend()

    return fig, ax
