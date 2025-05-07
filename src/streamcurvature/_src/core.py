"""Spline-related tools."""

__all__ = [
    "AbstractTrack",
    "Track",
]

import functools as ft
from dataclasses import dataclass, fields
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

    """

    ridge_line: interpax.Interpolator1D

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

    def __call__(self, gamma: SzN) -> Real[Array, "N 2"]:
        """Return the position at a given gamma.

        This is just evaluating the spline at the given gamma values.

        Examples
        --------
        Compute the tangent vector for specific points on the unit circle:

        >>> import jax.numpy as jnp
        >>> import interpax
        >>> import streamcurvature as sc

        >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
        >>> xy = 2  * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
        >>> track = sc.Track(gamma, xy)

        >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
        >>> print(track(gamma).round(2))
        [[ 2.  0.]
         [ 0.  2.]
         [-2.  0.]]

        """
        return self.ridge_line(gamma)

    def positions(self, gamma: SzN) -> SzN2:
        """Compute the position at a given gamma. See `__call__` for details."""
        return self(gamma)

    @ft.partial(jnp.vectorize, signature="()->(2)", excluded=(0,))
    @ft.partial(jax.jit)
    def spherical_position(self, gamma: SzN, /) -> SzN2:
        r"""Compute $|\vec{f}(gamma)|$ at `gamma`.

        Examples
        --------
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import interpax
        >>> import streamcurvature.splinelib as splib

        >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
        >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
        >>> spline = interpax.Interpolator1D(gamma, xy, method="cubic2")

        >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
        >>> r = jax.vmap(splib.spherical_position, (None, 0))(spline, gamma)
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
        >>> import streamcurvature as sc

        >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
        >>> x = 2 * jnp.cos(gamma)
        >>> y = 2 * jnp.sin(gamma)
        >>> track = sc.Track(gamma, jnp.stack([x, y], axis=-1))

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
        >>> import streamcurvature as sc

        >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
        >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
        >>> track = sc.Track(gamma, xy)

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
        >>> import streamcurvature as sc

        >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
        >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
        >>> track = sc.Track(gamma, xy)

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
        >>> import streamcurvature as sc

        >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
        >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
        >>> track = sc.Track(gamma, xy)

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
        >>> import streamcurvature as sc

        >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
        >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
        >>> track = sc.Track(gamma, xy)

        >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
        >>> kappa = track.kappa(gamma)
        >>> print(kappa.round(5))
        [0.5 0.5 0.5]

        """
        return splinelib.kappa(self.ridge_line, gamma)

    # =====================================================

    def __eq__(self, other: object) -> Bool[Array, ""]:
        """Check if two tracks are equal."""
        if not isinstance(other, AbstractTrack):
            return NotImplemented

        all_fields = [
            (
                jnp.all(getattr(self, f.name) == getattr(other, f.name))
                if hasattr(other, f.name)
                else False
            )
            for f in fields(self)
        ]
        return jnp.all(jnp.array(all_fields))

    # =====================================================
    # Plotting methods

    def plot_track(
        self,
        gamma: SzN,
        /,
        *,
        ax: plt.Axes | None = None,
        label: str | None = r"$\vec{x}$($\gamma$)",
    ) -> plt.Axes:
        """Plot the track itself."""
        if ax is None:
            _, ax = plt.subplots(dpi=150, figsize=(10, 10))

        # Plot track itself
        ax.plot(*self(gamma).T, c="red", ls="-", lw=1, label=label)

        # Add the knot points
        ax.scatter(*self.knots.T, s=10, c="red", zorder=10)

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
        """Plot the unit tangent vectors along the track."""
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
    ) -> plt.Axes:
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
            color="green",
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
    ) -> plt.Axes:
        r"""Plot the track, tangents, curvature, and local accelerations.

        This method combines all the plotting methods into a single function to
        easily visualize the track, tangents, curvature, and local accelerations
        along the track. This is useful for quickly inspecting the geometry of a
        track.

        Parameters
        ----------
        gamma
            The gamma values to evaluate the track and geometry at.
        potential
            The potential to use for computing local accelerations. If `None`,
            the local acceleration vectors will not be plotted.

        ax
            The `matplotlib.axes.Axes` object to plot on. If `None`, a new
            figure and axes will be created. Defaults to `None`.
        vec_width
            The width of the quiver arrows. Defaults to `0.003`.
        vec_scale
            The scale factor for the quiver arrows. This affects the length of
            the arrows. Defaults to `30`.
        labels
            Whether to show labels. Defaults to `True`.
        show_tangents
            Whether to plot the unit tangent vectors. Defaults to `True`.
        show_curvature
            Whether to plot the unit curvature vectors. Defaults to `True`.

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
            )

        return ax


# ============================================================================


@final
@ft.partial(jtu.register_dataclass, data_fields=["ridge_line"], meta_fields=[])
@dataclass(frozen=True, slots=True, eq=False)
class Track(AbstractTrack):
    """A track with data and a spline."""

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
        """Create a Track from an existing spline."""
        # TODO: set directly without deconstructing
        if spline.method != "cubic2":
            msg = f"Spline must be cubic2, got {spline.method}."
            raise ValueError(msg)

        return cls(ridge_line=spline)
