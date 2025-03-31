"""Functional interface to `interpax.Interpolator1D`."""

__all__ = [
    "arc_length",
    "arc_length_odeint",
    "arc_length_p2p",
    "arc_length_quadtrature",
    "curvature",
    "dThat_dgamma",
    "position",
    "speed",
    "tangent",
    "unit_curvature",
    "unit_tangent",
]

import functools as ft
from typing import Any, Literal

import interpax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental.ode import odeint
from jaxtyping import Array, Real

from streamcurvature._src.custom_types import LikeSz0, Sz0, Sz2, SzN

from .data import point_to_point_distance


def position(spline: interpax.Interpolator1D, gamma: SzN, /) -> Real[Array, "N F"]:
    r"""Compute $\vec{f}(gamma)$ for `spline` $\vec{f}$ at `gamma`."""
    return spline(gamma)


# ============================================================================
# Tangent


@ft.partial(jax.jit, static_argnames=("forward",))
def tangent(
    spline: interpax.Interpolator1D, gamma: Sz0, /, *, forward: bool = True
) -> Sz2:
    r"""Compute the tangent vector at a given position along the stream.

    The tangent vector is defined as:

    $$
        \frac{d\vec{x}}{d\gamma} =
            \begin{bmatrix}
                \frac{dx}{d\gamma} \\ \frac{dy}{d\gamma}
            \end{bmatrix}
    $$

    This function is scalar. To compute the unit tangent vector at multiple
    positions, use `jax.vmap`.

    Parameters
    ----------
    spline
        The spline interpolator.
    gamma
        The scalar gamma value at which to evaluate the spline.

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
    >>> import streamcurvature.splinelib as splib

    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
    >>> x = 2 * jnp.cos(gamma)
    >>> y = 2 * jnp.sin(gamma)
    >>> spline = interpax.Interpolator1D(gamma, jnp.stack([x, y], axis=-1), kind="cubic2")

    >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
    >>> tangents = splib.tangent(spline, gamma)
    >>> print(tangents.round(2))
    [[ 0.  2.]
     [-2.  0.]
     [ 0. -2.]]

    """
    jac_fn = jax.jacfwd if forward else jax.jacrev
    return jac_fn(spline)(gamma)


@ft.partial(jax.jit, static_argnames=("forward",))
def unit_tangent(
    spline: interpax.Interpolator1D, gamma: Sz0, /, *, forward: bool = True
) -> Sz2:
    r"""Compute the unit tangent vector at a given position along the stream.

    The unit tangent vector is defined as:

    $$ \hat{\mathbf{T}} = \mathbf{T} / \|\mathbf{T}\| $$

    This function is scalar. To compute the unit tangent vector at multiple
    positions, use `jax.vmap`.

    Parameters
    ----------
    spline
        The spline interpolator.
    gamma
        The scalar gamma value at which to evaluate the spline.

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
    >>> import streamcurvature.splinelib as splib

    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
    >>> x = 2 * jnp.cos(gamma)
    >>> y = 2 * jnp.sin(gamma)
    >>> spline = interpax.Interpolator1D(gamma, jnp.stack([x, y], axis=-1), kind="cubic2")

    >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
    >>> unit_tangents = splib.unit_tangent(spline, gamma)
    >>> print(unit_tangents.round(2))
    [[ 0.  1.]
     [-1.  0.]
     [ 0. -1.]]

    """
    T = tangent(spline, gamma, forward=forward)
    return T / jnp.linalg.vector_norm(T)


@ft.partial(jax.jit, static_argnames=("forward",))
def speed(
    spline: interpax.Interpolator1D, gamma: Sz0, /, *, forward: bool = True
) -> Sz0:
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
    spline
        The spline interpolator.
    gamma
        The gamma value at which to evaluate the spline.
    forward
        If `True`, compute using forward-mode differentiation; otherwise,
        compute using backward-mode differentiation. Defaults to `True`.

    """
    # TODO: confirm that this equals L/2 for gamma \propto s
    T = tangent(spline, gamma, forward=forward)
    return jnp.linalg.vector_norm(T)


# ============================================================================
# Arc-length


@ft.partial(jax.jit, static_argnames=("num"))
def arc_length_p2p(
    spline: interpax.Interpolator1D,
    gamma0: LikeSz0 = -1,
    gamma1: LikeSz0 = 1,
    *,
    num: int = 100_000,
) -> Sz0:
    """Compute the arc-length using point-to-point distance.

    Parameters
    ----------
    spline
        The spline interpolator.
    gamma0, gamma1
        The starting / ending gamma value between which to compute the
        arc-length. The default is [-1, 1], which is the full range of gamma
        for the track.
    num
        The number of points to use for the quadrature. The default is
        100,000.

    """
    gammas = jnp.linspace(gamma0, gamma1, num, dtype=float)
    y = position(spline, gammas)
    d_p2p = point_to_point_distance(y)
    return jnp.sum(d_p2p)


@ft.partial(jax.jit, static_argnames=("num"))
def arc_length_quadtrature(
    spline: interpax.Interpolator1D,
    gamma0: LikeSz0 = -1,
    gamma1: LikeSz0 = 1,
    *,
    num: int = 100_000,
) -> Sz0:
    """Compute the arc-length using fixed quadrature.

    Parameters
    ----------
    spline
        The spline interpolator.
    gamma0, gamma1
        The starting / ending gamma value between which to compute the
        arc-length. The default is [-1, 1], which is the full range of gamma
        for the track.
    num
        The number of points to use for the quadrature. The default is
        100,000.

    """
    gammas = jnp.linspace(gamma0, gamma1, num, dtype=float)
    speeds = jax.vmap(speed, in_axes=(None, 0))(spline, gammas)
    dgamma = (gamma1 - gamma0) / (num - 1)
    return jnp.sum(speeds) * dgamma


@ft.partial(jax.jit, static_argnames=("rtol", "atol", "mxstep", "hmax"))
def arc_length_odeint(
    spline: interpax.Interpolator1D,
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
    spline
        The spline interpolator.
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
        return speed(spline, gamma)

    # Set integration endpoints.
    t = jnp.array([gamma0, gamma1], dtype=float)
    s0 = 0.0  # initial arc length

    # Use odeint to integrate the ODE.
    s = odeint(ds_dgamma, s0, t, rtol=rtol, atol=atol, mxstep=mxstep, hmax=hmax)
    arc_length = s[-1]
    return arc_length


@ft.partial(jax.jit, static_argnames=("method", "method_kw"))
def arc_length(
    spline: interpax.Interpolator1D,
    gamma0: LikeSz0 = -1,
    gamma1: LikeSz0 = 1,
    *,
    method: Literal["p2p", "quad", "ode"] = "quad",
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
        jtu.Partial(arc_length_p2p, **kw),
        jtu.Partial(arc_length_quadtrature, **kw),
        jtu.Partial(arc_length_odeint, **kw),
    ]
    operands = (spline, gamma0, gamma1)
    return jax.lax.switch(methods.index(method), branches, *operands)


# ============================================================================
# Curvature


@ft.partial(jax.jit, static_argnames=("forward",))
def dThat_dgamma(
    spline: interpax.Interpolator1D, gamma: Sz0, /, *, forward: bool = True
) -> Sz2:
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
    return jac_fn(unit_tangent, argnums=1)(spline, gamma, forward=forward)


@ft.partial(jax.jit, static_argnames=("forward",))
def curvature(
    spline: interpax.Interpolator1D, gamma: Sz0, /, *, forward: bool = True
) -> Sz0:
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
    spline
        The spline interpolator.
    gamma
        The gamma value at which to evaluate the curvature.
    forward
        If `True`, compute using forward-mode differentiation; otherwise,
        compute using backward-mode differentiation. Defaults to `True`.

    """
    dThat = dThat_dgamma(spline, gamma, forward=forward)
    ds = speed(spline, gamma, forward=forward)
    return dThat / ds


@ft.partial(jax.jit, static_argnames=("forward",))
def unit_curvature(
    spline: interpax.Interpolator1D, gamma: Sz0, /, *, forward: bool = True
) -> Sz2:
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
    kappa = curvature(spline, gamma, forward=forward)
    return kappa / jnp.linalg.vector_norm(kappa)
