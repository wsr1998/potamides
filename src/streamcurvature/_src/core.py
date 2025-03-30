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
from jax.experimental.ode import odeint
from jaxtyping import Array, Real

from .custom_types import LikeSz0, Sz0, Sz2, SzGamma, SzGammaF, SzN
from .splinelib import point_to_point_distance

log2pi = jnp.log(2 * jnp.pi)


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

    def __post_init__(self) -> None:
        self.ridge_line: interpax.Interpolator1D

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

    def __call__(self, gamma: SzN) -> Real[Array, "N F"]:
        """Return the position at a given gamma.

        This is just evaluating the spline at the given gamma values.

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
        >>> print(track(gamma).round(2))
        [[ 2.  0.]
         [ 0.  2.]
         [-2.  0.]]

        """
        return self.ridge_line(gamma)

    def positions(self, gamma: SzN) -> Real[Array, "N F"]:
        """Compute the position at a given gamma. See `__call__` for details."""
        return self(gamma)

    # -------------------------------------------
    # Tangents

    @ft.partial(jax.jit, static_argnames=("forward",))
    @ft.partial(jnp.vectorize, signature="()->(2)", excluded=(0,))
    def tangent(self, gamma: Sz0, /, *, forward: bool = True) -> Sz2:
        r"""Compute the tangent vector at a given position along the stream.

        The tangent vector is defined as:

        $$
            \frac{d\mathbf{r}}{d\gamma} =
                \begin{bmatrix}
                    \frac{dx}{d\gamma} \\ \frac{dy}{d\gamma}
                \end{bmatrix}
        $$

        This function is scalar. To compute the unit tangent vector at multiple
        positions, use `jax.vmap`.

        Parameters
        ----------
        gamma
            The scalar gamma value at which to evaluate the spline.
            This is for `jax.vmap` compatibility.
        spline
            The spline interpolator.
        forward
            If `True`, compute forward tangents; otherwise, compute backward
            tangents. Defaults to `True`.

        Returns
        -------
        Array[real, (2,)]
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
        jac_fn = jax.jacfwd if forward else jax.jacrev
        tangent_fn = jac_fn(self.ridge_line)
        return tangent_fn(gamma)

    @ft.partial(jax.jit, static_argnames=("forward",))
    @ft.partial(jnp.vectorize, signature="()->(2)", excluded=(0,))
    def unit_tangent(self, gamma: Sz0, /, *, forward: bool = True) -> Sz2:
        r"""Compute the unit tangent vector at a given position along the stream.

        The unit tangent vector is defined as:

        $$ \hat{\mathbf{T}} = \mathbf{T} / \|\mathbf{T}\| $$

        This function is scalar. To compute the unit tangent vector at multiple
        positions, use `jax.vmap`.

        Parameters
        ----------
        gamma
            The scalar gamma value at which to evaluate the spline. This is for
            `jax.vmap` compatibility.
        spline
            The spline interpolator.
        forward
            If `True`, compute forward tangents; otherwise, compute backward
            tangents. Defaults to `True`.

        Returns
        -------
        Array[real, (2,)]
            The unit tangent vector at the specified position.

        Examples
        --------
        Compute the unit tangent vector for specific points on the unit circle:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import interpax
        >>> import streamcurvature as sc

        >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_00)
        >>> x = jnp.cos(gamma)
        >>> y = jnp.sin(gamma)
        >>> track = sc.Track(gamma, jnp.stack([x, y], axis=-1))

        >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
        >>> unit_tangents = track.unit_tangent(gamma)
        >>> print(unit_tangents.round(2))
        [[ 0.  1.]
         [-1.  0.]
         [ 0. -1.]]

        """
        tangents = self.tangent(gamma, forward=forward)
        return tangents / jnp.linalg.vector_norm(tangents)

    @ft.partial(jax.jit, static_argnames=("forward",))
    @ft.partial(jnp.vectorize, signature="()->()", excluded=(0,))
    def state_speed(self, gamma: Sz0, /, *, forward: bool = True) -> Sz0:
        """Return the speed in gamma of the track at a given position.

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
        forward
            If `True`, compute using forward-mode differentiation; otherwise,
            compute using backward-mode differentiation. Defaults to `True`.

        """
        # TODO: confirm that this equals L/2 for gamma \propto s
        return jnp.linalg.norm(self.tangent(gamma, forward=forward))

    # -------------------------------------------
    # Arc-length

    @ft.partial(jax.jit, inline=True, static_argnames=("num"))
    def _arc_length_p2p(
        self, gamma0: LikeSz0 = -1, gamma1: LikeSz0 = 1, *, num: int = 100_000
    ) -> Sz0:
        """Compute the arc-length using point-to-point distance.

        Parameters
        ----------
        gamma0, gamma1
            The starting / ending gamma value between which to compute the
            arc-length. The default is [-1, 1], which is the full range of gamma
            for the track.
        num
            The number of points to use for the quadrature. The default is
            100,000.

        """
        gammas = jnp.linspace(gamma0, gamma1, num, dtype=float)
        x = self.ridge_line(gammas)
        d_p2p = point_to_point_distance(x)
        return jnp.sum(d_p2p)

    @ft.partial(jax.jit, inline=True, static_argnames=("num"))
    def _arc_length_quadtrature(
        self, gamma0: LikeSz0 = -1, gamma1: LikeSz0 = 1, *, num: int = 100_000
    ) -> Sz0:
        """Compute the arc-length using fixed quadrature.

        Parameters
        ----------
        gamma0, gamma1
            The starting / ending gamma value between which to compute the
            arc-length. The default is [-1, 1], which is the full range of gamma
            for the track.
        num
            The number of points to use for the quadrature. The default is
            100,000.

        """
        gammas = jnp.linspace(gamma0, gamma1, num, dtype=float)
        speeds = jax.vmap(self.state_speed)(gammas)
        dgamma = (gamma1 - gamma0) / (num - 1)
        return jnp.sum(speeds) * dgamma

    @ft.partial(
        jax.jit, inline=True, static_argnames=("rtol", "atol", "mxstep", "hmax")
    )
    def _arc_length_odeint(
        self,
        gamma0: LikeSz0 = -1,
        gamma1: LikeSz0 = 1,
        *,
        rtol: float = 1.4e-8,
        atol: float = 1.4e-8,
        mxstep: float = jnp.inf,
        hmax: float = jnp.inf,
    ) -> Sz0:
        """Compute the arc-length using ODE integration.

        Parameters
        ----------
        gamma0, gamma1
            The starting / ending gamma value between which to compute the
            arc-length. The default is [-1, 1], which is the full range of gamma
            for the track.

        rtol, atol
            The relative and absolute tolerances for the ODE solver. The default
            is 1.4e-8.
        mxstep, hmax
            The maximum number of steps and maximum step size for the ODE
            solver. The default is inf.

        """

        @ft.partial(jax.jit)
        def ds_dgamma(_: Sz0, gamma: Sz0) -> Sz0:
            return self.state_speed(gamma)

        # Set integration endpoints.
        t = jnp.array([gamma0, gamma1], dtype=float)
        s0 = 0.0  # initial arc length

        # Use odeint to integrate the ODE.
        s = odeint(ds_dgamma, s0, t, rtol=rtol, atol=atol, mxstep=mxstep, hmax=hmax)
        arc_length = s[-1]
        return arc_length

    @ft.partial(jax.jit, static_argnames=("method", "method_kw"))
    def arc_length(
        self,
        gamma0: LikeSz0 = -1,
        gamma1: LikeSz0 = 1,
        *,
        method: Literal["p2p", "quad", "ode"] = "quad",
        method_kw: dict[str, Any] | None = None,
    ) -> Sz0:
        """Return the arc-length of the track.

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
            "quad", or "ode". The default is "quad".

            - "p2p": point-to-point distance. This method computes the distance
                between each pair of points along the track and sums them up.
                Accuracy is limited by the 1e5 points used.
            - "quad": quadrature. This method uses fixed quadrature to compute
                the integral. It is the default method. It also uses 1e5 points.
            - "ode": ODE integration. This method uses ODE integration to
              compute the integral.

        """
        methods = ("p2p", "quad", "ode")
        kw = method_kw if method_kw is not None else {}
        branches = [
            eqx.Partial(self._arc_length_p2p, **kw),
            eqx.Partial(self._arc_length_quadtrature, **kw),
            eqx.Partial(self._arc_length_odeint, **kw),
        ]
        operands = (gamma0, gamma1)
        return jax.lax.switch(methods.index(method), branches, *operands)

    @property
    def total_arc_length(self) -> Sz0:
        """Return the total arc-length of the track.

        $$
            L = s(-1, 1) = \int_{-1}^{1} \left\| \frac{d\mathbf{x}(\gamma)}{d\gamma} \right\| \, d\gamma
        $$

        This is equivalent to `arc_length` with gamma0=-1 and gamma1=1.
        The method used is the default method, which is "quad".

        """
        return self.arc_length(gamma0=-1, gamma1=1)

    # -------------------------------------------
    # Curvature

    @ft.partial(jax.jit, static_argnames=("forward",))
    @ft.partial(jnp.vectorize, signature="()->(2)", excluded=(0,))
    def dThat_dgamma(self, gamma: Sz0, /, *, forward: bool = True) -> Sz2:
        r"""Return the gamma derivative of the unit tangent vector.

        .. note::

            The following applies if $\gamma$ is proportional to the arc-length.

        The derivative of the unit tangent vector with respect to $\gamma$ can
        relate to the curvature vector. If $\gamma$ is defined as the arc-length
        parameter, normalized to the range [-1, 1], then the derivative of the
        unit tangent vector with respect to $\gamma$ is the scaled curvature
        vector.

        The curvature vector is defined as the derivative of the unit tangent
        vector with respect to arc-length $s$:

        $$ \frac{d\hat{T}}{ds} = \kappa \hat{N}, $$

        where $\kappa$ is the curvature (its magnitude) and \hat{N} is the
        principal unit normal vector.

        If $\gamma$ is proportional to the arc-length, we can write

        $$ \gamma = \frac{2s}{L} - 1, $$

        where $L$ is the total arc-length of the stream. Then by the chain rule,
        we have

        $$ \frac{d\hat{T}}{d\gamma} = \frac{ds}{d\gamma} \frac{d\hat{T}}{ds}. $$

        Because $\frac{ds}{d\gamma} = \frac{L}{2},$ and $\frac{d\hat{T}}{ds} =
        \kappa\,\hat{N},$ it follows that

        $$ \frac{d\hat{T}}{d\gamma} = \frac{L}{2} \kappa \hat{N} \propto \kappa
        \hat{N}. $$

        Therefore the derivative of the unit tangent vector with respect to
        gamma is proportional to the curvature vector.

        """
        jac_fn = jax.jacfwd if forward else jax.jacrev
        dThat_dgamma_fn = jac_fn(self.unit_tangent)
        return dThat_dgamma_fn(gamma, forward=forward)

    def curvature(self, gamma: Sz0, /, *, forward: bool = True) -> Sz0:
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
        forward
            If `True`, compute using forward-mode differentiation; otherwise,
            compute using backward-mode differentiation. Defaults to `True`.

        """
        dThat = self.dThat_dgamma(gamma, forward=forward)
        ds = self.state_speed(gamma, forward=forward)
        return dThat / ds

    @ft.partial(jax.jit, static_argnames=("forward",))
    @ft.partial(jnp.vectorize, signature="()->(2)", excluded=(0,))
    def unit_curvature(self, gamma: Sz0, /, *, forward: bool = True) -> Sz2:
        r"""Return the unit curvature vector.

        .. warning::

            This function assumes that the input gamma is proportional to the
            arc-length. If this is not the case, the unit-curvature vector may
            not be accurate.

        See ``Track.dThat_dgamma`` for the relationship between the
        gamma-derivative of the unit-tangent vector and the curvature vector.

        For

        $$ \frac{d\hat{T}}{d\gamma} \propto \kappa \hat{N}, $$

        where $\kappa \hat{N}$ is the curvature vector and $\hat{N}$ is the unit
        normal vector (aka unit curvature vector), it follows that

        $$ \hat{N} = \frac{\kappa \hat{N}}{\|\kappa \hat{N}\|}. $$

        """
        dThat = self.dThat_dgamma(gamma, forward=forward)
        unit_curvature = dThat / jnp.linalg.vector_norm(dThat)
        return unit_curvature

    # =====================================================
    # Plotting methods

    def plot_track(
        self,
        gamma: SzN,
        /,
        *,
        ax: plt.Axes | None = None,
        label: str | None = r"[$x,y$]($\gamma$)",
    ) -> plt.Axes:
        """Plot the track itself."""
        if ax is None:
            _, ax = plt.subplots(dpi=150, figsize=(10, 10))

        # Plot track itself
        ax.plot(*self(gamma).T, c="red", ls="-", label=label)

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
        tangents_hat = self.unit_tangent(gamma)

        ax.quiver(
            points[:, 0],
            points[:, 1],
            tangents_hat[:, 0],
            tangents_hat[:, 1],
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
        curvature_hat = self.unit_curvature(gamma)

        ax.quiver(
            points[:, 0],
            points[:, 1],
            curvature_hat[:, 0],
            curvature_hat[:, 1],
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
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(dpi=150, figsize=(10, 10))

        # Plot track itself
        self.plot_track(
            jnp.linspace(gamma.min(), gamma.max(), len(gamma) * 10),
            ax=ax,
            label=r"[$x,y$]($\gamma$)" if labels else None,
        )

        # Geometry along the track
        self.plot_tangents(
            gamma,
            ax=ax,
            vec_width=vec_width,
            vec_scale=vec_scale,
            label=r"$\hat{T}$" if labels else None,
        )
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
@dataclass(frozen=True, slots=True)
class Track(AbstractTrack):
    """A track with data and a spline."""

    #: [x,y](gamma) spline. It must be twice-differentiable (cubic2) to compute
    #: curvature vectors.
    ridge_line: interpax.Interpolator1D

    def __init__(
        self,
        gamma: SzGamma | None = None,
        data: SzGammaF | None = None,
        /,
        *,
        ridge_line: interpax.Interpolator1D | None = None,
    ) -> None:
        # Jax jit uses this branch
        if ridge_line is not None:
            spline = ridge_line
            if gamma is not None or data is not None:
                msg = "gamma, data must be None when using the ridge_line kwarg."
                raise ValueError(msg)
        elif gamma is None or data is None:
            msg = "Either ridge_line or both gamma and data must be provided."
            raise ValueError(msg)
        else:
            spline = interpax.Interpolator1D(gamma, data, method="cubic2")

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
