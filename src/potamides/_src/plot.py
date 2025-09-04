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

from .custom_types import SzGamma, SzN, SzN2

PI_ON_2 = np.pi / 2


@ft.partial(jax.jit)
def get_angles(acc_xy_unit: SzN2, kappa_hat: SzN2) -> SzN:
    r"""Return angle between the normal and acceleration vectors at a position.

    Calculate the angles between the normal vector at given position along the
    stream and the acceleration at given position along the stream. This is
    fundamental for analyzing stream dynamics, as the angle between the normal
    vector (perpendicular to the stream) and gravitational acceleration
    determines whether the stream is expanding or contracting.

    Parameters
    ----------
    acc_xy_unit : Array[float, (N, 2)]
        An array representing the planar acceleration at each input position.
        These should be unit vectors (normalized), but the function will
        re-normalize them to ensure unit length.
    kappa_hat : Array[float, (N, 2)]
        The unit curvature vector (or named normal vector) at each position.
        This is perpendicular to the stream direction. Also re-normalized
        to ensure unit length.

    Returns
    -------
    Array[float, (N,)]
        An array of angles in radians in the range (-π, π). Positive angles
        indicate the acceleration points "outward" from the stream, negative
        angles indicate "inward" acceleration.

    Notes
    -----
    The angle is computed using :func:`jax.numpy.atan2` applied to the cross
    product and dot product of the input vectors:

    .. math::
        \theta = \arctan2(\vec{a} \times \hat{\kappa}, \vec{a} \cdot \hat{\kappa})

    where :math:`\vec{a}` is the acceleration and :math:`\hat{\kappa}` is the
    normal vector.

    Examples
    --------
    Basic usage with simple 2D vectors:

    >>> import jax.numpy as jnp
    >>> import potamides as ptd

    >>> # Create acceleration vectors pointing in +x direction
    >>> acc_xy = jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    >>> # Create normal vectors: +y, +x, -y directions
    >>> kappa_hat = jnp.array([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])

    >>> angles = ptd.get_angles(acc_xy, kappa_hat)
    >>> print(f"Angles in radians: {angles}")
    Angles in radians: [ 1.57079633  0.         -1.57079633]

    >>> # Convert to degrees for interpretation
    >>> angles_deg = jnp.degrees(angles)
    >>> print(f"Angles in degrees: {angles_deg}")
    Angles in degrees: [ 90.  0. -90.]

    Physical interpretation for stream dynamics:

    >>> # Simulate a stream with positions along x-axis
    >>> import numpy as np
    >>> positions = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    >>> # Acceleration pointing outward from stream (in +y direction)
    >>> acc_outward = jnp.array([[0.0, 0.1], [0.0, 0.1], [0.0, 0.1]])
    >>> # Normal vectors perpendicular to stream (pointing in +y)
    >>> normals = jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    >>> angles_outward = ptd.get_angles(acc_outward, normals)
    >>> print(f"Outward acceleration angles: {jnp.degrees(angles_outward)}")
    Outward acceleration angles: [0. 0. 0.]

    >>> # Acceleration pointing inward (in -y direction)
    >>> acc_inward = jnp.array([[0.0, -0.1], [0.0, -0.1], [0.0, -0.1]])
    >>> angles_inward = ptd.get_angles(acc_inward, normals)
    >>> print(f"Inward acceleration angles: {jnp.degrees(angles_inward)}")
    Inward acceleration angles: [180. 180. 180.]

    Working with non-unit vectors (function handles normalization):

    >>> # Large magnitude vectors - function normalizes internally
    >>> large_acc = jnp.array([[100.0, 0.0], [50.0, 50.0]])
    >>> large_normals = jnp.array([[0.0, 200.0], [100.0, 0.0]])

    >>> angles_large = ptd.get_angles(large_acc, large_normals)
    >>> print(f"Angles from large vectors: {jnp.degrees(angles_large)}")
    Angles from large vectors: [ 90. -45.]

    """
    # Ensure the input vectors are unit vectors
    acc_xy_unit = acc_xy_unit / jnp.linalg.norm(acc_xy_unit, axis=1, keepdims=True)
    kappa_hat = kappa_hat / jnp.linalg.norm(kappa_hat, axis=1, keepdims=True)

    # Calculate the angle using the dot product and cross product
    dot_product = jnp.einsum("ij,ij->i", acc_xy_unit, kappa_hat)
    cross_product = jnp.cross(acc_xy_unit, kappa_hat)
    return jnp.atan2(cross_product, dot_product)


PI_ON_2 = np.pi / 2


def plot_theta_of_gamma(
    gamma: SzGamma,
    param: Real[Array, "param"],
    angles: Real[Array, "param gamma"],
    *,
    mle_idx: int | None = None,
    param_label: str = r"$q$",
) -> tuple[plt.Figure, plt.Axes]:
    r"""Plot angles θ as a function of stream parameter gamma with parameter colormap.

    Create a scatter plot showing how the angle between acceleration and normal
    vectors varies along the stream (:math:`\gamma`) for different parameter values. This
    visualization is crucial for understanding stream dynamics and identifying
    regions where theoretical constraints are violated.

    Parameters
    ----------
    gamma : Array[float, (G,)]
        Stream parameter values (e.g., parametric distance along the stream).
        Typically in the range [-1, 1] or similar normalized coordinates.
    param : Array[float, (P,)]
        Parameter values being analyzed (e.g., axis ratio q, velocity dispersion).
        Each value corresponds to a different model configuration.
    angles : Array[float, (P, G)]
        Angle values in radians for each parameter and gamma combination.
        Shape (n_params, n_gamma) where angles[i, j] is the angle for
        param[i] at gamma[j].
    mle_idx : int, optional
        Index of the maximum likelihood estimate in the param array.
        If provided, this parameter's angles will be highlighted in red.
    param_label : str, default r"$q$"
        Label for the colorbar showing the parameter values. Supports LaTeX.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object for further customization.

    Notes
    -----
    The plot includes gray exclusion regions for |θ| > π/2, which typically
    represent physically unrealistic configurations where the acceleration
    is pointing "backwards" relative to the stream normal.

    Examples
    --------
    Basic usage with synthetic data:

    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>> import potamides as ptd

    >>> # Create synthetic stream parameter and model parameters
    >>> gamma = jnp.linspace(-1, 1, 50)
    >>> q_values = jnp.array([0.5, 0.7, 0.9, 1.0, 1.2])
    >>>
    >>> # Generate synthetic angle data (varies smoothly with gamma and q)
    >>> angles = jnp.zeros((len(q_values), len(gamma)))
    >>> for i, q in enumerate(q_values):
    ...     # Simulate how angles might vary: more asymmetric for lower q
    ...     angles = angles.at[i].set(0.3 * jnp.sin(2 * jnp.pi * gamma) / q + 0.1 * gamma)
    >>>
    >>> fig, ax = ptd.plot.plot_theta_of_gamma(
    ...     gamma, q_values, angles,
    ...     mle_idx=2,  # Highlight q=0.9 as MLE
    ...     param_label=r"Axis ratio $q$"
    ... )
    >>> ax.set_title("Stream Angle Analysis")  # doctest: +SKIP

    .. plot::
       :context: close-figs

       import numpy as np
       import jax.numpy as jnp
       import potamides as ptd
       import matplotlib.pyplot as plt

       # Create synthetic stream parameter and model parameters
       gamma = jnp.linspace(-1, 1, 50)
       q_values = jnp.array([0.5, 0.7, 0.9, 1.0, 1.2])

       # Generate synthetic angle data (varies smoothly with gamma and q)
       angles = jnp.zeros((len(q_values), len(gamma)))
       for i, q in enumerate(q_values):
           # Simulate how angles might vary: more asymmetric for lower q
           angles = angles.at[i].set(0.3 * jnp.sin(2 * jnp.pi * gamma) / q + 0.1 * gamma)

       fig, ax = ptd.plot.plot_theta_of_gamma(
           gamma, q_values, angles,
           mle_idx=2,  # Highlight q=0.9 as MLE
           param_label=r"Axis ratio $q$"
       )
       ax.set_title("Stream Angle Analysis")
       plt.show()

    Analyzing realistic galactic stream data:

    .. plot::
       :context: close-figs

       # Simulate a more realistic scenario
       gamma_stream = jnp.linspace(-0.8, 0.8, 30)
       halo_masses = jnp.logspace(jnp.log10(1e11), jnp.log10(1e12), 8)

       # Simulate angles that depend on halo mass and position
       angles_realistic = jnp.zeros((len(halo_masses), len(gamma_stream)))
       for i, mass in enumerate(halo_masses):
           # Higher mass -> smaller angles, asymmetric about center
           base_angle = 0.5 / (mass / 1e11)
           angles_realistic = angles_realistic.at[i].set(base_angle * (
               jnp.sin(jnp.pi * gamma_stream) + 0.2 * gamma_stream**2
           ))

       fig, ax = ptd.plot.plot_theta_of_gamma(
           gamma_stream,
           halo_masses,
           angles_realistic,
           mle_idx=4,  # Best-fit mass
           param_label=r"$M_{\mathrm{halo}}$ [$M_{\odot}$]"
       )
       ax.set_title("Halo Mass vs Stream Angles")
       plt.show()

    """
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
    color: str = "#808F80",
) -> plt.Axes:
    """Plot the acceleration field of a gravitational potential.

    Create a vector field plot showing the direction and relative magnitude
    of gravitational acceleration across a 2D spatial grid. This visualization
    helps understand the gravitational environment and how it affects
    stream dynamics.

    Parameters
    ----------
    potential : galax.potential.AbstractPotential
        The gravitational potential object from galax. Can be any potential
        type including LogarithmicPotential, MiyamotoNagaiPotential, or
        composite potentials.
    xlim : tuple[float, float]
        The x-axis limits for the plot in the same units as the potential
        (typically kpc for galactic potentials).
    ylim : tuple[float, float]
        The y-axis limits for the plot in the same units as the potential.
    grid_size : int, default 20
        Number of grid points along each axis. Total number of vectors
        plotted will be grid_size².
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure and axes
        with high DPI and large size.
    vec_width : float, default 0.003
        Width of the quiver arrows. Smaller values create thinner arrows.
    vec_scale : float, default 30
        Scale factor for arrow length. Larger values create shorter arrows.
    color : str, default "#808F80"
        Color for the acceleration vectors. Can be any matplotlib color.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the quiver plot for further customization.

    Notes
    -----
    The function plots unit acceleration vectors (normalized to length 1)
    to show direction only. The magnitude information is lost but this
    makes the field structure clearer. All vectors are computed at z=0.

    Examples
    --------
    Basic logarithmic halo potential:

    >>> import galax.potential as gp
    >>> import unxt as u
    >>> import potamides as ptd

    >>> # Create a logarithmic halo potential
    >>> halo_pot = gp.LMJ09LogarithmicPotential(
    ...     v_c=u.Quantity(250, "km/s"),
    ...     r_s=u.Quantity(16, "kpc"),
    ...     q1=1.0, q2=0.9, q3=0.8, phi=0.0,
    ...     units="galactic"
    ... )
    >>>
    >>> # Plot the acceleration field around the halo center
    >>> ax = ptd.plot.plot_acceleration_field(
    ...     halo_pot,
    ...     xlim=(-20, 20),
    ...     ylim=(-20, 20),
    ...     grid_size=15
    ... )
    >>> ax.set_title("Logarithmic Halo Acceleration Field")  # doctest: +SKIP

    .. plot::
       :context: close-figs

       import galax.potential as gp
       import unxt as u
       import potamides as ptd
       import matplotlib.pyplot as plt

       # Create a logarithmic halo potential
       halo_pot = gp.LMJ09LogarithmicPotential(
           v_c=u.Quantity(250, "km/s"),
           r_s=u.Quantity(16, "kpc"),
           q1=1.0, q2=0.9, q3=0.8, phi=0.0,
           units="galactic"
       )

       # Plot the acceleration field around the halo center
       ax = ptd.plot.plot_acceleration_field(
           halo_pot,
           xlim=(-20, 20),
           ylim=(-20, 20),
           grid_size=15
       )
       ax.set_title("Logarithmic Halo Acceleration Field")
       plt.show()

    Disk potential with custom styling:

    .. plot::
       :context: close-figs

       # Create a Miyamoto-Nagai disk potential
       disk_pot = gp.MiyamotoNagaiPotential(
           m_tot=u.Quantity(1.2e10, "Msun"),
           a=u.Quantity(3, "kpc"),
           b=u.Quantity(0.5, "kpc"),
           units="galactic"
       )

       # Plot with custom appearance
       ax = ptd.plot.plot_acceleration_field(
           disk_pot,
           xlim=(-10, 10),
           ylim=(-8, 8),
           grid_size=25,
           vec_width=0.005,
           vec_scale=20,
           color='blue'
       )
       ax.set_title("Miyamoto-Nagai Disk Field")
       ax.set_xlabel("x [kpc]")
       ax.set_ylabel("y [kpc]")
       plt.show()

    Composite potential with stream overlay:

    .. plot::
       :context: close-figs

       import numpy as np

       # Combine multiple potentials
       composite_pot = gp.CompositePotential(
           halo=gp.LMJ09LogarithmicPotential(
               v_c=u.Quantity(220, "km/s"), r_s=u.Quantity(20, "kpc"),
               q1=1.0, q2=1.0, q3=1.0, phi=0.0,
               units="galactic"
           ),
           disk=gp.MiyamotoNagaiPotential(
               m_tot=u.Quantity(1e10, "Msun"), a=u.Quantity(3, "kpc"), b=u.Quantity(0.3, "kpc"),
               units="galactic"
           )
       )

       # Create figure with stream positions
       fig, ax = plt.subplots(figsize=(10, 8))

       # Plot some stream positions
       stream_x = np.linspace(-5, 5, 20)
       stream_y = 0.1 * stream_x**2  # Parabolic stream
       ax.plot(stream_x, stream_y, 'k-', linewidth=3, label='Stream')

       # Add acceleration field to same plot
       ptd.plot.plot_acceleration_field(
           composite_pot,
           xlim=(-8, 8),
           ylim=(-2, 6),
           grid_size=12,
           ax=ax,
           color='gray',
           vec_width=0.004
       )

       ax.legend()
       ax.set_title("Stream in Composite Potential Field")
       plt.show()

    """
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
        color=color,
        width=vec_width,
        scale=vec_scale,
        label=r"$\vec{a}$ (global)",
        alpha=0.5,
    )

    return ax
