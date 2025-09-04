"""Spline-related tools."""

__all__ = [
    "AbstractTrack",
    "Track",
]

import functools as ft
from dataclasses import dataclass
from typing import Any, Literal, final

import equinox as eqx
import galax.potential as gp
import interpax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
from jaxtyping import Array, Bool, Real

from . import splinelib
from .custom_types import LikeSz0, Sz0, Sz2, SzGamma, SzGammaF, SzN, SzN2

log2pi = jnp.log(2 * jnp.pi)


@dataclass(frozen=True, slots=True, eq=False)
class AbstractTrack:
    r"""ABC for track classes.

    It is strongly recommended to ensure that gamma is proportional to the
    arc-length of the track. A good definition of gamma is to normalize the
    arc-length to the range [-1, 1], such that

    $$ \gamma = \frac{2s}{L} - 1, $$

    where $s$ is the arc-length and $L$ is the total arc-length of the track.

    Parameters
    ----------
    ridge_line : interpax.Interpolator1D[(N, F), method="cubic2"]
        The spline interpolator for the track, parametrized by gamma. It is
        necessary for the spline to be twice-differentiable (cubic2) to compute
        the curvature vectors.

    Raises
    ------
    Exception
        If the spline is not cubic2.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import potamides as ptd
    >>> import matplotlib.pyplot as plt

    >>> # Create a parametric circle with radius 2
    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
    >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
    >>> track = ptd.Track(gamma, xy)

    Basic position evaluation:

    >>> gamma_test = jnp.array([0, jnp.pi / 2, jnp.pi])
    >>> positions = track(gamma_test)
    >>> print("Positions:", positions.round(2))
    Positions: [[ 2.  0.]
                [ 0.  2.]
                [-2.  0.]]

    Spherical coordinates (radius, angle):

    >>> spherical = track.spherical_position(gamma_test)
    >>> print("Spherical (r, theta):", spherical.round(4))
    Spherical (r, theta): [[2.     0.    ]
                           [2.     1.5708]
                           [2.     3.1416]]

    Tangent vectors along the track:

    >>> tangents = track.tangent(gamma_test)
    >>> print("Tangent vectors:", tangents.round(2))
    Tangent vectors: [[ 0.  2.]
                      [-2.  0.]
                      [ 0. -2.]]

    Curvature magnitude (for a circle, should be constant 1/radius):

    >>> kappa_values = track.kappa(gamma_test)
    >>> print("Curvature kappa:", kappa_values.round(4))
    Curvature kappa: [0.5 0.5 0.5]

    Curvature vectors:

    >>> curvature_vecs = track.curvature(gamma_test)
    >>> print("Curvature vectors:", curvature_vecs.round(2))
    Curvature vectors: [[-0.5  0. ]
                        [ 0.  -0.5]
                        [ 0.5  0. ]]

    Principal unit normal vectors (point toward center for circle):

    >>> normals = track.principle_unit_normal(gamma_test)
    >>> print("Unit normals:", normals.round(2))
    Unit normals: [[-1.  0.]
                   [ 0. -1.]
                   [ 1.  0.]]

    Access to track properties:

    >>> print("Number of knots:", len(track.knots))
    Number of knots: 10000
    >>> print("Gamma range:", track.gamma.min().round(2), "to", track.gamma.max().round(2))
    Gamma range: 0.0 to 6.28

    For visualization (requires matplotlib):

    .. plot::
       :include-source:

       import jax.numpy as jnp
       import potamides as ptd
       import matplotlib.pyplot as plt

       # Create a parametric circle with radius 2
       gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
       xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
       track = ptd.Track(gamma, xy)

       # Create the plot
       fig, ax = plt.subplots(figsize=(8, 8))
       gamma_plot = jnp.linspace(0, 2*jnp.pi, 50)
       track.plot_all(gamma_plot, ax=ax)
       ax.set_aspect('equal')
       ax.set_title('Track Example: Circle with Geometry Vectors')
       plt.tight_layout()
       plt.show()

    """

    ridge_line: interpax.Interpolator1D
    "The spline interpolator for the track, parametrized by gamma."

    def __post_init__(self) -> None:
        _ = eqx.error_if(
            self.ridge_line,
            self.ridge_line.method != "cubic2",
            f"Spline must be twice-differentiable (cubic2) to compute curvature vectors, got {self.ridge_line.method}.",
        )

    @property
    def gamma(self) -> SzN:
        """Return the gamma values of the track."""
        return self.ridge_line.x

    @property
    def knots(self) -> Real[Array, "N F"]:
        """Return the knot points along the track."""
        return self.ridge_line.f

    # =====================================================

    # -------------------------------------------
    # Positions

    def positions(self, gamma: SzN) -> SzN2:
        """Return the position at a given gamma.

        Examples
        --------
        Compute the position for specific points on the unit circle:

        >>> import jax.numpy as jnp
        >>> import interpax
        >>> import potamides as ptd

        >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
        >>> xy = 2  * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
        >>> track = ptd.Track(gamma, xy)

        >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
        >>> print(track.positions(gamma).round(2))
        [[ 2.  0.]
         [ 0.  2.]
         [-2.  0.]]

        """
        return self.ridge_line(gamma)

    def __call__(self, gamma: SzN) -> Real[Array, "N 2"]:
        """Return the position at a given gamma."""
        return self.positions(gamma)

    @ft.partial(jnp.vectorize, signature="()->(2)", excluded=(0,))
    @ft.partial(jax.jit)
    def spherical_position(self, gamma: SzN, /) -> SzN2:
        r"""Compute $|\vec{f}(gamma)|$ at `gamma`.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import potamides as ptd

        >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
        >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
        >>> track = ptd.Track(gamma, xy)

        >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
        >>> r = track.spherical_position(gamma)
        >>> print(r.round(4))
        [[2.     0.    ]
         [2.     1.5708]
         [2.     3.1416]]

        """
        return splinelib.spherical_position(self.ridge_line, gamma)

    # -------------------------------------------
    # Tangents

    @ft.partial(jnp.vectorize, signature="()->(2)", excluded=(0,))
    @ft.partial(jax.jit)
    def tangent(self, gamma: Sz0, /) -> Sz2:
        r"""Compute the tangent vector at a given position along the stream.

        The tangent vector is defined as:

        $$ T(\gamma) = \frac{d\vec{x}}{d\gamma} $$

        Parameters
        ----------
        gamma
            The gamma value at which to evaluate the spline.

        Returns
        -------
        Array[real, (*batch, 2)]
            The tangent vector at the specified position.

        Examples
        --------
        Compute the tangent vector for specific points on the unit circle:

        >>> import jax.numpy as jnp
        >>> import interpax
        >>> import potamides as ptd

        >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
        >>> x = 2 * jnp.cos(gamma)
        >>> y = 2 * jnp.sin(gamma)
        >>> track = ptd.Track(gamma, jnp.stack([x, y], axis=-1))

        >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
        >>> tangents = track.tangent(gamma)
        >>> print(tangents.round(2))
        [[ 0.  2.]
         [-2.  0.]
         [ 0. -2.]]

        """
        return splinelib.tangent(self.ridge_line, gamma)

    @ft.partial(jnp.vectorize, signature="()->()", excluded=(0,))
    @ft.partial(jax.jit)
    def state_speed(self, gamma: Sz0, /) -> Sz0:
        r"""Return the speed in gamma of the track at a given position.

        This is the norm of the tangent vector at the given position.

        $$
            \mathbf{v}(\gamma) = \left\| \frac{d\mathbf{x}(\gamma)}{d\gamma}
            \right\|
        $$

        An important note is that this is also equivalent to the derivative of
        the arc-length with respect to gamma.

        On a 2D flat surface (the flat-sky approximation is reasonable for
        observations of extragalactic stellar streams) the differential
        arc-length is given by:

        $$
            s = \int_{\gamma_0}^{\gamma} \sqrt{\left(\frac{dx}{d\gamma}\right)^2
                + \left(\frac{dy}{d\gamma}\right)^2} d\gamma.
        $$

        Thus, the arc-length element is:

        $$
            \frac{ds}{d\gamma} = \sqrt{\left(\frac{dx}{d\gamma}\right)^2
                + \left(\frac{dy}{d\gamma}\right)^2}
        $$

        If $\gamma$ is proportional to the arc-length, which is a very good and
        common choice, then for $\gamma \in [-1, 1] = \frac{2s}{L} - 1$, we have

        $$
            \frac{ds}{d\gamma} = \frac{L}{2},
        $$

        where $L$ is the total arc-length of the stream.

        Since this is a constant, there is no need to compute this function. It
        is sufficient to just use $L/2$. This function is provided for
        completeness.

        Parameters
        ----------
        gamma
            The gamma value at which to evaluate the spline.

        """
        # TODO: confirm that this equals L/2 for gamma \propto s
        return splinelib.speed(self.ridge_line, gamma)

    # -------------------------------------------
    # Arc-length

    @ft.partial(jax.jit, static_argnames=("method", "method_kw"))
    def arc_length(
        self,
        gamma0: LikeSz0 = -1,
        gamma1: LikeSz0 = 1,
        *,
        method: Literal["p2p", "quad", "ode"] = "p2p",
        method_kw: dict[str, Any] | None = None,
    ) -> Sz0:
        r"""Return the arc-length of the track.

        $$
            s(\gamma_0, \gamma_1) = \int_{\gamma_0}^{\gamma_1} \left\|
            \frac{d\mathbf{x}(\gamma)}{d\gamma} \right\| \, d\gamma
        $$

        Computing the arc-length requires computing an integral over the norm of
        the tangent vector. This can be done using many different methods. We
        provide three options, specified by the `method` parameter.

        Parameters
        ----------
        gamma0, gamma1
            The starting / ending gamma value between which to compute the
            arc-length. The default is [-1, 1], which is the full range of gamma
            for the track.

        method
            The method to use for computing the arc-length. Options are "p2p",
            "quad", or "ode". The default is "p2p".

            - "p2p": point-to-point distance. This method computes the distance
                between each pair of points along the track and sums them up.
                Accuracy is limited by the 1e5 points used.
            - "quad": quadrature. This method uses fixed quadrature to compute
                the integral. It is the default method. It also uses 1e5 points.
            - "ode": ODE integration. This method uses ODE integration to
              compute the integral.

        """
        return splinelib.arc_length(
            self.ridge_line, gamma0, gamma1, method=method, method_kw=method_kw
        )

    @property
    def total_arc_length(self) -> Sz0:
        r"""Return the total arc-length of the track.

        $$
            L = s(-1, 1) = \int_{-1}^{1} \left\| \frac{d\mathbf{x}(\gamma)}{d\gamma} \right\| \, d\gamma
        $$

        This is equivalent to `arc_length` with gamma0=-1 and gamma1=1.
        The method used is the default method, which is "quad".

        """
        return self.arc_length(gamma0=self.gamma.min(), gamma1=self.gamma.max())

    # -------------------------------------------
    # Acceleration

    @ft.partial(jnp.vectorize, signature="()->(2)", excluded=(0,))
    @ft.partial(jax.jit)
    def acceleration(self, gamma: Sz0, /) -> Sz2:
        r"""Return the acceleration vector at a given position along the stream.

        The acceleration vector is defined as: $ \frac{d^2\vec{x}}{d\gamma^2} $

        Parameters
        ----------
        gamma
            The gamma value at which to evaluate the acceleration.

        Returns
        -------
        Array[float, (N, 2)]
            The acceleration vector $\vec{a}$ at $\gamma$.

        Examples
        --------
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import potamides as ptd

        >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
        >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
        >>> track = ptd.Track(gamma, xy)

        >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
        >>> acc = track.acceleration(gamma)
        >>> print(acc.round(5))
        [[-2.  0.]
         [ 0. -2.]
         [ 2.  0.]]

        """
        return splinelib.acceleration(self.ridge_line, gamma)

    @ft.partial(jnp.vectorize, signature="()->(2)", excluded=(0,))
    @ft.partial(jax.jit)
    def principle_unit_normal(self, gamma: Sz0, /) -> Sz2:
        r"""Return the unit normal vector at a given position along the stream.

        The unit normal vector is defined as the normalized acceleration vector:

        $$ \hat{N} = \frac{d^2\vec{x}/d\gamma^2}{\left\| d^2\vec{x}/d\gamma^2
        \right\|} $$

        Parameters
        ----------
        gamma
            The gamma value at which to evaluate the normal vector.

        Returns
        -------
        Array[float, (N, 2)]
            The unit normal vector $\hat{N}$ at $\gamma$.

        Examples
        --------
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import potamides as ptd

        >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
        >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
        >>> track = ptd.Track(gamma, xy)

        >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
        >>> Nhat = track.principle_unit_normal(gamma)
        >>> print(Nhat.round(5))
        [[-1.  0.]
         [ 0. -1.]
         [ 1.  0.]]

        """
        return splinelib.principle_unit_normal(self.ridge_line, gamma)

    # -------------------------------------------
    # Curvature

    @ft.partial(jnp.vectorize, signature="()->(2)", excluded=(0,))
    @ft.partial(jax.jit)
    def curvature(self, gamma: Sz0, /) -> Sz0:
        r"""Return the curvature at a given position along the stream.

        This method computes the curvature by taking the ratio of the gamma
        derivative of the unit tangent vector to the derivative of the
        arc-length with respect to gamma. In other words, if

        $$ \frac{d\hat{T}}{d\gamma} = \frac{ds}{d\gamma} \frac{d\hat{T}}{ds}, $$

        and since the curvature vector is defined as

        $$ \frac{d\hat{T}}{ds} = \kappa \hat{N}, $$

        where $ \kappa $ is the curvature and $ \hat{N} $ the unit normal
        vector, then dividing $ \frac{d\hat{T}}{d\gamma} $ by $
        \frac{ds}{d\gamma} $ yields

        $$ \kappa \hat{N} = \frac{d\hat{T}/d\gamma}{ds/d\gamma}. $$

        Here, $\frac{d\hat{T}}{d\gamma}$ (computed by ``dThat_dgamma``)
        describes how the direction of the tangent changes with respect to the
        affine parameter $\gamma$, and $\frac{ds}{d\gamma}$ (obtained from
        state_speed) represents the state speed (i.e. the rate of change of
        arc-length with respect to $\gamma$).

        This formulation assumes that $\gamma$ is chosen to be proportional to
        the arc-length of the track.

        Parameters
        ----------
        gamma
            The gamma value at which to evaluate the curvature.

        Returns
        -------
        Array[float, (N, 2)]
            The curvature vector $\kappa$ at $\gamma$.

        Examples
        --------
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import potamides as ptd

        >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
        >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
        >>> track = ptd.Track(gamma, xy)

        >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
        >>> kappa = track.curvature(gamma)
        >>> print(kappa.round(5))
        [[-0.5  0. ]
         [ 0.  -0.5]
         [ 0.5  0. ]]

        """
        return splinelib.curvature(self.ridge_line, gamma)

    @ft.partial(jnp.vectorize, signature="()->()", excluded=(0,))
    @ft.partial(jax.jit)
    def kappa(self, gamma: Sz0, /) -> Sz0:
        r"""Return the scalar curvature $\kappa(\gamma)$ along the track.

        Parameters
        ----------
        gamma
            The gamma value at which to evaluate the curvature.

        Returns
        -------
        Array[float, (N, 2)]
            The scalar curvature $\kappa$ at $\gamma$.

        Examples
        --------
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import potamides as ptd

        >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
        >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
        >>> track = ptd.Track(gamma, xy)

        >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
        >>> kappa = track.kappa(gamma)
        >>> print(kappa.round(5))
        [0.5 0.5 0.5]

        """
        return splinelib.kappa(self.ridge_line, gamma)

    # =====================================================

    def __eq__(self, other: object) -> Bool[Array, ""]:  # type: ignore[override, unused-ignore]
        """Check if two tracks are equal.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import potamides as ptd

        >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
        >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
        >>> track1 = ptd.Track(gamma, xy)
        >>> track2 = ptd.Track(gamma, xy)
        >>> track1 == track2
        Array(True, dtype=bool)

        >>> track3 = ptd.Track(gamma, xy + 1)
        >>> track1 == track3
        Array(False, dtype=bool)

        """
        if not isinstance(other, AbstractTrack):
            return NotImplemented

        eq_tree = jtu.tree_map(jnp.array_equal, self, other)
        return jnp.array(jtu.tree_reduce(jnp.logical_and, eq_tree, True))

    # =====================================================
    # Plotting methods

    def plot_track(
        self,
        gamma: SzN,
        /,
        *,
        ax: plt.Axes | None = None,
        label: str | None = r"$\vec{x}$($\gamma$)",
        c: str = "red",
        knot_size: int = 10,
    ) -> plt.Axes:
        """Plot the track curve itself with knot points.

        This method visualizes the parametric track curve as a continuous line
        and overlays the knot points used in the spline interpolation.

        Parameters
        ----------
        gamma : Array[float, "N"]
            The gamma values to evaluate and plot the track at.
        ax : plt.Axes, optional
            The matplotlib axes to plot on. If None, creates a new figure.
        label : str, optional
            The label for the track curve in the legend.
        c : str, default "red"
            The color for the track curve and knot points.
        knot_size : int, default 10
            The size of the knot point markers.

        Returns
        -------
        plt.Axes
            The matplotlib axes containing the plot.

        Examples
        --------
        .. plot::
           :include-source:

           import jax.numpy as jnp
           import potamides as ptd
           import matplotlib.pyplot as plt

           # Create a circular track
           gamma = jnp.linspace(0, 2 * jnp.pi, 100)
           xy = 3 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
           track = ptd.Track(gamma, xy)

           # Plot just the track
           fig, ax = plt.subplots(figsize=(8, 8))
           gamma_plot = jnp.linspace(0, 2*jnp.pi, 200)
           track.plot_track(gamma_plot, ax=ax)
           ax.set_aspect('equal')
           ax.set_title('Track Plot: Circle with Knot Points')
           ax.legend()
           plt.tight_layout()
           plt.show()

        """
        if ax is None:
            _, ax = plt.subplots(dpi=150, figsize=(10, 10))

        # Plot track itself
        ax.plot(*self(gamma).T, c=c, ls="-", lw=1, label=label)

        # Add the knot points
        ax.scatter(*self.knots.T, s=knot_size, c=c, zorder=10)

        return ax

    def plot_tangents(
        self,
        gamma: SzN,
        *,
        ax: plt.Axes | None = None,
        vec_width: float = 0.003,
        vec_scale: float = 30,
        color: str = "red",
        label: str | None = r"$\hat{T}$",
    ) -> plt.Axes:
        """Plot the unit tangent vectors along the track.

        This method visualizes the normalized tangent vectors at specified points
        along the track. The tangent vectors show the direction of motion along
        the parametric curve.

        Parameters
        ----------
        gamma : Array[float, "N"]
            The gamma values where tangent vectors will be plotted.
        ax : plt.Axes, optional
            The matplotlib axes to plot on. If None, creates a new figure.
        vec_width : float, default 0.003
            The width of the quiver arrows.
        vec_scale : float, default 30
            The scale factor for arrow lengths (higher = shorter arrows).
        color : str, default "red"
            The color of the tangent vector arrows.
        label : str, optional
            The label for the tangent vectors in the legend.

        Returns
        -------
        plt.Axes
            The matplotlib axes containing the plot.

        Examples
        --------
        .. plot::
           :include-source:

           import jax.numpy as jnp
           import potamides as ptd
           import matplotlib.pyplot as plt

           # Create a circular track
           gamma = jnp.linspace(0, 2 * jnp.pi, 100)
           xy = 3 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
           track = ptd.Track(gamma, xy)

           # Plot track with tangent vectors
           fig, ax = plt.subplots(figsize=(8, 8))
           gamma_plot = jnp.linspace(0, 2*jnp.pi, 200)
           gamma_vectors = jnp.linspace(0, 2*jnp.pi, 12)

           track.plot_track(gamma_plot, ax=ax, c='black', label='Track')
           track.plot_tangents(gamma_vectors, ax=ax, color='red', vec_scale=20)

           ax.set_aspect('equal')
           ax.set_title('Track with Tangent Vectors')
           ax.legend()
           plt.tight_layout()
           plt.show()

        """
        if ax is None:
            _, ax = plt.subplots(dpi=150, figsize=(10, 10))

        points = self(gamma)
        T_hat = self.tangent(gamma)
        T_hat = T_hat / jnp.linalg.norm(T_hat, axis=1, keepdims=True)

        ax.quiver(
            points[:, 0],
            points[:, 1],
            T_hat[:, 0],
            T_hat[:, 1],
            color=color,
            scale=vec_scale,
            label=label,
            width=vec_width,
        )
        return ax

    def plot_curvature(
        self,
        /,
        gamma: SzN,
        *,
        ax: plt.Axes | None = None,
        vec_width: float = 0.003,
        vec_scale: float = 30,
        color: str = "blue",
        label: str | None = r"$\hat{\kappa}$",
    ) -> plt.Axes:
        """Plot the principal unit normal vectors (curvature direction) along the track.

        This method visualizes the principal unit normal vectors at specified points
        along the track. These vectors point in the direction of curvature and are
        perpendicular to the tangent vectors, showing how the track curves.

        Parameters
        ----------
        gamma : Array[float, "N"]
            The gamma values where normal vectors will be plotted.
        ax : plt.Axes, optional
            The matplotlib axes to plot on. If None, creates a new figure.
        vec_width : float, default 0.003
            The width of the quiver arrows.
        vec_scale : float, default 30
            The scale factor for arrow lengths (higher = shorter arrows).
        color : str, default "blue"
            The color of the normal vector arrows.
        label : str, optional
            The label for the normal vectors in the legend.

        Returns
        -------
        plt.Axes
            The matplotlib axes containing the plot.

        Examples
        --------
        .. plot::
           :include-source:

           import jax.numpy as jnp
           import potamides as ptd
           import matplotlib.pyplot as plt

           # Create a circular track
           gamma = jnp.linspace(0, 2 * jnp.pi, 100)
           xy = 3 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
           track = ptd.Track(gamma, xy)

           # Plot track with curvature vectors
           fig, ax = plt.subplots(figsize=(8, 8))
           gamma_plot = jnp.linspace(0, 2*jnp.pi, 200)
           gamma_vectors = jnp.linspace(0, 2*jnp.pi, 12)

           track.plot_track(gamma_plot, ax=ax, c='black', label='Track')
           track.plot_curvature(gamma_vectors, ax=ax, color='blue', vec_scale=20)

           ax.set_aspect('equal')
           ax.set_title('Track with Curvature Vectors (Principal Unit Normals)')
           ax.legend()
           plt.tight_layout()
           plt.show()

        """
        if ax is None:
            _, ax = plt.subplots(dpi=150, figsize=(10, 10))

        points = self(gamma)
        # kappa_vec points in the direction of Nhat
        Nhat = self.principle_unit_normal(gamma)

        ax.quiver(
            points[:, 0],
            points[:, 1],
            Nhat[:, 0],
            Nhat[:, 1],
            color=color,
            scale=vec_scale,
            label=label,
            width=vec_width,
        )
        return ax

    def plot_local_accelerations(
        self,
        potential: gp.AbstractPotential,
        gamma: SzN,
        /,
        t: float = 0,
        *,
        vec_width: float = 0.003,
        vec_scale: float = 30,
        ax: plt.Axes | None = None,
        label: str | None = r"$\vec{a}$ (local)",
        color: str = "green",
    ) -> plt.Axes:
        """Plot the local gravitational acceleration vectors along the track.

        This method visualizes the gravitational acceleration vectors from a given
        potential at specified points along the track. This is useful for understanding
        how the gravitational field affects the motion along the track.

        Parameters
        ----------
        potential : galax.potential.AbstractPotential
            The gravitational potential to evaluate accelerations.
        gamma : Array[float, "N"]
            The gamma values where acceleration vectors will be plotted.
        t : float, default 0
            The time at which to evaluate the potential (for time-dependent potentials).
        vec_width : float, default 0.003
            The width of the quiver arrows.
        vec_scale : float, default 30
            The scale factor for arrow lengths (higher = shorter arrows).
        ax : plt.Axes, optional
            The matplotlib axes to plot on. If None, creates a new figure.
        label : str, optional
            The label for the acceleration vectors in the legend.
        color : str, default "green"
            The color of the acceleration vector arrows.

        Returns
        -------
        plt.Axes
            The matplotlib axes containing the plot.

        Examples
        --------
        .. plot::
           :include-source:

           import jax.numpy as jnp
           import potamides as ptd
           import galax.potential as gp
           import matplotlib.pyplot as plt

           # Create a circular track
           gamma = jnp.linspace(0, 2 * jnp.pi, 100)
           xy = 3 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
           track = ptd.Track(gamma, xy)

           # Create a simple point mass potential at origin
           potential = gp.KeplerPotential(m_tot=1e12, units="galactic")

           # Plot track with local acceleration vectors
           fig, ax = plt.subplots(figsize=(8, 8))
           gamma_plot = jnp.linspace(0, 2*jnp.pi, 200)
           gamma_vectors = jnp.linspace(0, 2*jnp.pi, 12)

           track.plot_track(gamma_plot, ax=ax, c='black', label='Track')
           track.plot_local_accelerations(potential, gamma_vectors, ax=ax,
                                         color='green', vec_scale=10)

           ax.set_aspect('equal')
           ax.set_title('Track with Local Gravitational Acceleration')
           ax.legend()
           plt.tight_layout()
           plt.show()

        """
        if ax is None:
            _, ax = plt.subplots(dpi=150, figsize=(10, 10))

        # Construct evaluation points along the track
        pos = jnp.zeros((len(gamma), 3))
        pos = pos.at[:, :2].set(self(gamma))

        # Compute the acceleration at the evaluation points
        acc = potential.acceleration(pos, t=t)
        acc_unit = acc / jnp.linalg.norm(acc, axis=1, keepdims=True)
        acc_xy_unit = acc_unit[:, :2]

        ax.quiver(
            pos[:, 0],
            pos[:, 1],
            acc_xy_unit[:, 0],
            acc_xy_unit[:, 1],
            color=color,
            width=vec_width,
            scale=vec_scale,
            label=label,
        )
        return ax

    def plot_all(
        self,
        gamma: SzN,
        /,
        potential: gp.AbstractPotential | None = None,
        *,
        ax: plt.Axes | None = None,
        vec_width: float = 0.003,
        vec_scale: float = 30,
        labels: bool = True,
        show_tangents: bool = True,
        show_curvature: bool = True,
        curvature_kwargs: dict[str, Any] | None = None,
        acceleration_kwargs: dict[str, Any] | None = None,
    ) -> plt.Axes:
        r"""Plot the track, tangents, curvature, and local accelerations.

        This method combines all the plotting methods into a single function to
        easily visualize the track, tangents, curvature, and local accelerations
        along the track. This is useful for quickly inspecting the geometry of a
        track.

        Parameters
        ----------
        gamma : Array[float, "N"]
            The gamma values to evaluate the track and geometry at.
        potential : galax.potential.AbstractPotential, optional
            The potential to use for computing local accelerations. If `None`,
            the local acceleration vectors will not be plotted.
        ax : plt.Axes, optional
            The `matplotlib.axes.Axes` object to plot on. If `None`, a new
            figure and axes will be created. Defaults to `None`.
        vec_width : float, default 0.003
            The width of the quiver arrows. Defaults to `0.003`.
        vec_scale : float, default 30
            The scale factor for the quiver arrows. This affects the length of
            the arrows. Defaults to `30`.
        labels : bool, default True
            Whether to show labels. Defaults to `True`.
        show_tangents : bool, default True
            Whether to plot the unit tangent vectors. Defaults to `True`.
        show_curvature : bool, default True
            Whether to plot the unit curvature vectors. Defaults to `True`.
        curvature_kwargs : dict, optional
            Additional keyword arguments to pass to the curvature plotting method.
        acceleration_kwargs : dict, optional
            Additional keyword arguments to pass to the acceleration plotting method.

        Returns
        -------
        plt.Axes
            The matplotlib axes containing the complete plot.

        Examples
        --------
        Basic track visualization with geometry vectors:

        .. plot::
           :include-source:

           import jax.numpy as jnp
           import potamides as ptd
           import matplotlib.pyplot as plt

           # Create a figure-8 track for interesting geometry
           gamma = jnp.linspace(0, 2 * jnp.pi, 200)
           x = 2 * jnp.sin(gamma)
           y = jnp.sin(2 * gamma)
           xy = jnp.stack([x, y], axis=-1)
           track = ptd.Track(gamma, xy)

           # Plot all geometric features
           fig, ax = plt.subplots(figsize=(10, 8))
           gamma_vectors = jnp.linspace(0, 2*jnp.pi, 16)

           track.plot_all(gamma_vectors, ax=ax, vec_scale=15)
           ax.set_aspect('equal')
           ax.set_title('Complete Track Visualization: Figure-8 with Geometry Vectors')
           ax.legend()
           plt.tight_layout()
           plt.show()

        Track with gravitational potential:

        .. plot::
           :include-source:

           import jax.numpy as jnp
           import potamides as ptd
           import galax.potential as gp
           import matplotlib.pyplot as plt

           # Create a circular track
           gamma = jnp.linspace(0, 2 * jnp.pi, 100)
           xy = 5 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
           track = ptd.Track(gamma, xy)

           # Add a gravitational potential
           potential = gp.KeplerPotential(m_tot=1e12, units="galactic")

           # Plot everything including gravitational field
           fig, ax = plt.subplots(figsize=(10, 10))
           gamma_vectors = jnp.linspace(0, 2*jnp.pi, 12)

           track.plot_all(gamma_vectors, potential=potential, ax=ax, vec_scale=8)
           ax.set_aspect('equal')
           ax.set_title('Track with Gravitational Field')
           ax.legend()
           plt.tight_layout()
           plt.show()

        """
        if ax is None:
            _, ax = plt.subplots(dpi=150, figsize=(10, 10))

        # Plot track itself
        self.plot_track(
            jnp.linspace(gamma.min(), gamma.max(), len(gamma) * 10),
            ax=ax,
            label=r"$\vec{x}$($\gamma$)" if labels else None,
        )

        # Geometry along the track
        if show_tangents:
            self.plot_tangents(
                gamma,
                ax=ax,
                vec_width=vec_width,
                vec_scale=vec_scale,
                label=r"$\hat{T}$" if labels else None,
            )
        if show_curvature:
            self.plot_curvature(
                gamma,
                ax=ax,
                vec_width=vec_width,
                vec_scale=vec_scale,
                label=r"$\hat{K}$" if labels else None,
                **(curvature_kwargs or {}),
            )

        # Plot the local acceleration, assuming a potential
        if potential is not None:
            self.plot_local_accelerations(
                potential,
                gamma,
                t=0,
                ax=ax,
                vec_width=vec_width,
                vec_scale=vec_scale,
                label=r"$\vec{a}$ (local)" if labels else None,
                **(acceleration_kwargs or {}),
            )

        return ax


# ============================================================================


@final
@ft.partial(jtu.register_dataclass, data_fields=["ridge_line"], meta_fields=[])
@dataclass(frozen=True, slots=True, eq=False)
class Track(AbstractTrack):
    r"""Concrete implementation of a parametric track using spline interpolation.

    This class represents a smooth parametric curve in 2D space using cubic spline
    interpolation. It provides a concrete implementation of the AbstractTrack
    interface with automatic spline construction from data points or direct
    spline specification.

    The track is parameterized by gamma values and uses cubic2 spline interpolation
    to ensure twice-differentiability, which is required for computing curvature
    vectors and other geometric properties.

    Parameters
    ----------
    gamma : Array[float, "N"], optional
        The parameter values along the track. Must be provided together with `knots`
        if `ridge_line` is not specified.
    knots : Array[float, "N F"], optional
        The position data points corresponding to gamma values, where F is the
        spatial dimension (typically 2 for x,y coordinates). Must be provided
        together with `gamma` if `ridge_line` is not specified.
    ridge_line : interpax.Interpolator1D, optional
        Pre-constructed spline interpolator. If provided, `gamma` and `knots`
        must be None.

    Raises
    ------
    ValueError
        If neither (`gamma`, `knots`) nor `ridge_line` is provided, or if both
        are provided simultaneously.
    ValueError
        If the spline method is not "cubic2" (required for curvature computation).

    Examples
    --------
    Create a circular track from parametric data:

    >>> import jax.numpy as jnp
    >>> import potamides as ptd

    >>> # Generate circle data
    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 100)
    >>> x = 3 * jnp.cos(gamma)
    >>> y = 3 * jnp.sin(gamma)
    >>> knots = jnp.stack([x, y], axis=-1)
    >>> track = ptd.Track(gamma, knots)

    >>> # Evaluate track at specific points
    >>> test_gamma = jnp.array([0, jnp.pi/2, jnp.pi])
    >>> positions = track(test_gamma)
    >>> print("Positions:", positions.round(2))
    Positions: [[ 3.  0.]
                [ 0.  3.]
                [-3.  0.]]

    Create a sinusoidal track:

    >>> gamma = jnp.linspace(-1, 1, 50)
    >>> x = gamma
    >>> y = jnp.sin(3 * jnp.pi * gamma)
    >>> knots = jnp.stack([x, y], axis=-1)
    >>> track = ptd.Track(gamma, knots)

    >>> # Compute geometric properties
    >>> gamma_test = jnp.array([-0.5, 0, 0.5])
    >>> tangents = track.tangent(gamma_test)
    >>> curvatures = track.kappa(gamma_test)
    >>> print("Curvatures:", curvatures.round(3))
    Curvatures: [88.684  0.    88.684]

    Create from an existing spline:

    >>> import interpax
    >>> gamma = jnp.linspace(0, 1, 20)
    >>> y = gamma**2
    >>> knots = jnp.stack([gamma, y], axis=-1)
    >>> spline = interpax.Interpolator1D(gamma, knots, method="cubic2")
    >>> track = ptd.Track(ridge_line=spline)

    Track properties and methods:

    >>> print("Gamma range:", track.gamma.min(), "to", track.gamma.max())
    Gamma range: 0.0 to 1.0
    >>> print("Number of knots:", len(track.knots))
    Number of knots: 20
    >>> arc_length = track.total_arc_length
    >>> print("Total arc length:", arc_length.round(3))
    Total arc length: 1.479

    Visualization example:

    .. plot::
       :include-source:

       import jax.numpy as jnp
       import potamides as ptd
       import matplotlib.pyplot as plt

       # Create a spiral track
       gamma = jnp.linspace(0, 4*jnp.pi, 200)
       r = 1 + 0.3 * gamma
       x = r * jnp.cos(gamma)
       y = r * jnp.sin(gamma)
       knots = jnp.stack([x, y], axis=-1)
       track = ptd.Track(gamma, knots)

       # Plot the track with geometric vectors
       fig, ax = plt.subplots(figsize=(10, 10))
       gamma_plot = jnp.linspace(0, 4*jnp.pi, 400)
       gamma_vectors = jnp.linspace(0, 4*jnp.pi, 20)

       track.plot_all(gamma_vectors, ax=ax, vec_scale=10)
       ax.set_aspect('equal')
       ax.set_title('Spiral Track with Geometric Properties')
       plt.tight_layout()
       plt.show()

    See Also
    --------
    AbstractTrack : Base class defining the track interface
    interpax.Interpolator1D : The underlying spline interpolation class

    Notes
    -----
    The Track class is designed to work seamlessly with JAX transformations
    including jit compilation, automatic differentiation, and vectorization.
    All geometric computations are performed using JAX operations for optimal
    performance.

    The spline interpolation uses the "cubic2" method which ensures the track
    is twice-differentiable everywhere, enabling computation of curvature
    vectors and higher-order geometric properties.

    """

    #: [x,y](gamma) spline. It must be twice-differentiable (cubic2) to compute
    #: curvature vectors.
    ridge_line: interpax.Interpolator1D

    def __init__(
        self,
        gamma: SzGamma | None = None,
        knots: SzGammaF | None = None,
        /,
        *,
        ridge_line: interpax.Interpolator1D | None = None,
    ) -> None:
        # Jax jit uses this branch
        if ridge_line is not None:
            spline = ridge_line
            if gamma is not None or knots is not None:
                msg = "gamma, data must be None when using the ridge_line kwarg."
                raise ValueError(msg)
        elif gamma is None or knots is None:
            msg = "Either ridge_line or both gamma and data must be provided."
            raise ValueError(msg)
        else:
            spline = interpax.Interpolator1D(gamma, knots, method="cubic2")

        object.__setattr__(self, "ridge_line", spline)
        self.__post_init__()

    @classmethod
    def from_spline(cls: "type[Track]", spline: interpax.Interpolator1D, /) -> "Track":
        """Create a Track from an existing spline interpolator.

        Parameters
        ----------
        spline : interpax.Interpolator1D
            An existing spline interpolator that will be used as the ridge_line
            for the track. The spline must use the "cubic2" method to ensure
            twice-differentiability for curvature computations.

        Returns
        -------
        Track
            A new Track instance using the provided spline as its ridge_line.

        Raises
        ------
        ValueError
            If the spline method is not "cubic2", which is required for
            computing curvature vectors and other second-order geometric
            properties.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import interpax
        >>> import potamides as ptd

        >>> gamma = jnp.linspace(-2, 2, 50)
        >>> knots = jnp.stack([gamma, gamma**2], axis=-1)

        >>> spline = interpax.Interpolator1D(gamma, knots, method="cubic2")

        >>> track = ptd.Track.from_spline(spline)

        """
        # TODO: set directly without deconstructing
        if spline.method != "cubic2":
            msg = f"Spline must be cubic2, got {spline.method}."
            raise ValueError(msg)

        return cls(ridge_line=spline)
