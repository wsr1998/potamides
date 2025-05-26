"""Functional interface to `interpax.Interpolator1D`."""

__all__ = [  # noqa: RUF022
    # ---------
    # Positions
    "position",
    "spherical_position",
    # ---------
    "tangent",
    "speed",  # also differential arc-length
    # ---------
    "arc_length",
    "arc_length_odeint",
    "arc_length_p2p",
    "arc_length_quadtrature",
    # ---------
    "acceleration",
    "principle_unit_normal",
    # ---------
    "curvature",
    "kappa",
]

import functools as ft
from typing import Any, Final, Literal

import interpax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental.ode import odeint

import potamides._src.custom_types as ct
from potamides._src.custom_types import LikeSz0, Sz0

from .data import point_to_point_distance


@ft.partial(jax.jit, inline=True)
def position(spline: interpax.Interpolator1D, gamma: ct.SzN, /) -> ct.SzNF:
    r"""Compute $\vec{x}(gamma)$ for `spline` $\vec{x}$ at `gamma`.

    This is the Cartesian position vector at the given parameter values `gamma`.
    The output is an array with shape `(N, F)`, where `N` is the number of input
    `gamma` values and `F` is the number of dimensions of the spline.

    Parameters
    ----------
    spline
        The spline interpolator.
    gamma
        The gamma values at which to evaluate the spline. This can be a scalar
        or an array of shape `(N,)`.

    Returns
    -------
    Array[real, (N, F)]
        The position vector $\vec{x}(\gamma)$ at the specified positions. The
        shape is `(N, F)`, where `N` is the number of input `gamma` values and
        `F` is the number of dimensions of the spline.

    See Also
    --------
    `potamides.Track.position`
        This method auto-vectorizes to support arbitrarily shaped `gamma`
        inputs.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import interpax
    >>> import potamides.splinelib as splib

    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
    >>> xy = jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
    >>> spline = interpax.Interpolator1D(gamma, xy, method="cubic2")

    >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
    >>> pos = jax.vmap(splib.position, (None, 0))(spline, gamma)
    >>> print(pos.round(4))
    [[ 1.  0.]
     [ 0.  1.]
     [-1.  0.]]

    """
    return spline(gamma)


# -------------------------------------------------------------------------
# Spherical coordinates


@ft.partial(jax.jit)
def spherical_position(spline: interpax.Interpolator1D, gamma: ct.Sz0, /) -> ct.SzF:
    r"""Compute $|\vec{f}(gamma)|$ for `spline` $\vec{f}$ at `gamma`.

    This is the spherical coordinate at the given parameter values `gamma`. The
    output is an array with shape `(N, F)`, where `N` is the number of input
    `gamma` values and `F` is the number of dimensions of the spline. The 0th
    feature is the radius, and the remaining features are the angular
    coordinates.

    The radius is defined as the Euclidean norm of the position vector:

    .. math::

        r(\gamma) = \left\| \vec{x}(\gamma) \right\|

    The angular coordinates are computed recursively using:

    .. math::

        \phi_i(\gamma) = \arctan2( R_{i+1}, x_i ), \quad \text{for } i = 0,
        \dots, F-2

    where :math:`R_{i+1} = \sqrt{\sum_{j=i+1}^{F-1} x_j^2}` is the partial radius from
    the $i$-th coordinate to the last coordinate.

    The last angular coordinate is special-cased as it only depends on the last
    two coordinates:

    .. math::

        \phi_{F-1}(\gamma) = \arctan2\left(x_F, x_{F-1}\right)

    For more details, see https://en.wikipedia.org/wiki/N-sphere.

    Parameters
    ----------
    spline
        The spline interpolator.
    gamma
        The scalar gamma value at which to evaluate the spline.

    Returns
    -------
    Array[real, (F,)]
        The spherical coordinates at the specified position. The shape is
        `(F,)`, where `F` is the number of dimensions of the spline. The 0th
        feature is the radius, and the remaining features are the angular
        coordinates. To evaluate the spherical coordinates at multiple
        positions, use `jax.vmap`.

    See Also
    --------
    `potamides.Track.spherical_position`
        This method auto-vectorizes to support arbitrarily shaped `gamma`
        inputs.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import interpax
    >>> import potamides.splinelib as splib

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
    assert gamma.ndim == 0

    # Position vector at gamma.
    x = spline(gamma)  # shape (F,)

    # 1) radius
    r = jnp.linalg.norm(x, axis=-1)

    # 2) angular coordinates up to the second-to-last dimension.
    # Compute partial radii: R[k] = sqrt(x_F^2 + ... + x_{k+1}^2)
    R = jnp.sqrt(jnp.cumsum(jnp.square(x[1:])[::-1])[::-1])
    # angles $\phi_i = atan2(R_{i+1}, x_i)$
    phis = jnp.arctan2(R[:-1], x[:-2])
    phif = jnp.arctan2(x[-1], x[-2])  # last angle is special case

    # Pack the coordinates
    xsph = jnp.empty_like(x)
    xsph = xsph.at[0].set(r)  # set radius
    xsph = xsph.at[1:-1].set(phis)  # set angles
    xsph = xsph.at[-1].set(phif)  # set last angle
    return xsph


# ============================================================================
# Tangent


@ft.partial(jax.jit, inline=True)
def tangent(spline: interpax.Interpolator1D, gamma: ct.Sz0, /) -> ct.SzF:
    r"""Compute the tangent vector at a given position along the stream.

    The tangent vector is defined as:

    .. math::
        T(\gamma) = \frac{d\vec{x}}{d\gamma}

    This function is scalar. To compute the unit tangent vector at multiple
    positions, use `jax.vmap`.

    Parameters
    ----------
    spline
        The spline interpolator.
    gamma
        The scalar gamma value at which to evaluate the spline.
        To evaluate the tangent vector at multiple positions, use `jax.vmap`.

    Returns
    -------
    Array[real, (F,)]
        The tangent vector at the specified position.

    See Also
    --------
    `potamides.Track.tangent`
        This method auto-vectorizes to support arbitrarily shaped `gamma`
        inputs.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import interpax
    >>> import potamides.splinelib as splib

    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
    >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
    >>> spline = interpax.Interpolator1D(gamma, xy, method="cubic2")

    >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
    >>> tangents = jax.vmap(splib.tangent, (None, 0))(spline, gamma)
    >>> print(tangents.round(2))
    [[ 0.  2.]
     [-2.  0.]
     [ 0. -2.]]

    """
    assert gamma.ndim == 0
    return jax.jacfwd(spline)(gamma)


# Private function
@ft.partial(jax.jit, inline=True)
def unit_tangent(spline: interpax.Interpolator1D, gamma: ct.Sz0, /) -> ct.SzF:
    r"""Compute the unit tangent vector at a given position along the stream.

    The unit tangent vector is defined as:

    .. math::

        \hat{\vec{T}} = \vec{T} / \|\vec{T}\|

    This function is scalar. To compute the unit tangent vector at multiple
    positions, use `jax.vmap`.

    Parameters
    ----------
    spline
        The spline interpolator.
    gamma
        The scalar gamma value at which to evaluate the spline. To evaluate the
        unit tangent vector at multiple positions, use `jax.vmap`.

    Returns
    -------
    Array[real, (F,)]
        The unit tangent vector at the specified position.

    See Also
    --------
    `potamides.splinelib.tangent`
        The non-unit tangent vector at the specified position.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import interpax

    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
    >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
    >>> spline = interpax.Interpolator1D(gamma, xy, method="cubic2")

    >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
    >>> T_hat = jax.vmap(unit_tangent, (None, 0))(spline, gamma)
    >>> print(T_hat.round(2))
    [[ 0.  1.]
     [-1.  0.]
     [ 0. -1.]]

    """
    T = tangent(spline, gamma)
    return T / jnp.linalg.vector_norm(T)


@ft.partial(jax.jit, inline=True)
def speed(spline: interpax.Interpolator1D, gamma: ct.Sz0, /) -> ct.SzF:
    r"""Return the speed in gamma of the track at a given position.

    This is the norm of the tangent vector at the given position.

    .. math::

        v(\gamma) = \| \frac{d\vec{x}(\gamma)}{d\gamma} \|
        = \|\vec{T}(\gamma) \|

    An important note is that this is also the differential arc-length!

    .. math::

        s = \int_{\gamma_0}^{\gamma} \|\frac{\vec{x}}{d\gamma'}\| d\gamma'.

    Thus, the arc-length element is:

    .. math::

        \frac{ds}{d\gamma} = \|\frac{\vec{x}}{d\gamma'}\|

    If we are working in 2D in the flat-sky approximation for extragalactic
    streams, then it is recommended for $\gamma$ to be proportional to the
    arc-length with $\gamma \in [-1, 1] = \frac{2s}{L} - 1$, we have

    .. math::

        \frac{ds}{d\gamma} = \frac{L}{2},

    where $L$ is the total arc-length of the stream.

    Parameters
    ----------
    spline
        The spline interpolator.
    gamma
        The scalar gamma value at which to evaluate the spline. To evaluate the
        speed at multiple gammas, use `jax.vmap`.

    Returns
    -------
    Array[real, (F,)]
        The speed at the specified position. The shape is `(F,)`, where `F` is
        the number of dimensions of the spline.

    See Also
    --------
    `potamides.Track.state_speed`
        This method auto-vectorizes to support arbitrarily shaped `gamma`
        inputs.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import interpax
    >>> import potamides.splinelib as splib

    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
    >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
    >>> spline = interpax.Interpolator1D(gamma, xy, method="cubic2")

    >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
    >>> speed = jax.vmap(splib.speed, (None, 0))(spline, gamma)
    >>> print(speed)  # see 2 in xy
    [2. 2. 2.]

    """
    return jnp.linalg.vector_norm(tangent(spline, gamma))


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
    """Compute the arc-length using a point-to-point distance approximation.

    Parameters
    ----------
    spline
        The spline interpolator.
    gamma0, gamma1
        The starting / ending gamma value between which to compute the
        arc-length. The default is [-1, 1], which is the recommended range of
        gamma for the track.
    num
        The integer number of points to use for the quadrature. The default is
        100,000.

    See Also
    --------
    `potamides.Track.arc_length`
        This method auto-vectorizes to support arbitrarily shaped `gamma`
        inputs. It also allows for other methods of computing the arc-length.
    `potamides.splinelib.arc_length_quadtrature`
        This method uses fixed quadrature to compute the integral. It is also
        limited in accuracy by the number of points used.
    `potamides.splinelib.arc_length_odeint`
        This method uses ODE integration to compute the integral.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import interpax
    >>> import potamides.splinelib as splib

    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
    >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
    >>> spline = interpax.Interpolator1D(gamma, xy, method="cubic2")

    >>> s = splib.arc_length_p2p(spline, 0, 2 * jnp.pi) / jnp.pi
    >>> print(s.round(5))
    4.0

    """
    gammas = jnp.linspace(gamma0, gamma1, num, dtype=float)
    y = position(spline, gammas)
    d_p2p = point_to_point_distance(y)
    return jnp.sum(d_p2p)


speed_fn = jax.vmap(speed, in_axes=(None, 0))


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
        arc-length. The default is [-1, 1], which is the full range of gamma for
        the track.
    num
        The number of points to use for the quadrature. The default is 100,000.

    See Also
    --------
    `potamides.Track.arc_length`
        This method auto-vectorizes to support arbitrarily shaped `gamma`
        inputs. It also allows for other methods of computing the arc-length.
    `potamides.splinelib.arc_length_p2p`
        This method computes the distance between each pair of points along the
        track and sums them up. Accuracy is limited by the number of points
        used.
    `potamides.splinelib.arc_length_odeint`
        This method uses ODE integration to compute the integral.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import interpax
    >>> import potamides.splinelib as splib

    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
    >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
    >>> spline = interpax.Interpolator1D(gamma, xy, method="cubic2")

    >>> s = splib.arc_length_quadtrature(spline, 0, 2 * jnp.pi, num=500_000) / jnp.pi
    >>> print(s.round(4))  # harder to make accurate
    4.0

    """
    gammas = jnp.linspace(gamma0, gamma1, num, dtype=float)
    speeds = speed_fn(spline, gammas)
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

    See Also
    --------
    `potamides.Track.arc_length`
        This method auto-vectorizes to support arbitrarily shaped `gamma`
        inputs. It also allows for other methods of computing the arc-length.
    `potamides.splinelib.arc_length_p2p`
        This method computes the distance between each pair of points along the
        track and sums them up. Accuracy is limited by the number of points
        used.
    `potamides.splinelib.arc_length_quadtrature`
        This method uses fixed quadrature to compute the integral. It is also
        limited in accuracy by the number of points used.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import interpax
    >>> import potamides.splinelib as splib

    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
    >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
    >>> spline = interpax.Interpolator1D(gamma, xy, method="cubic2")

    >>> s = splib.arc_length_odeint(spline, 0, 2 * jnp.pi) / jnp.pi
    >>> print(s.round(5))
    4.0

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


_ARC_LENGTH_METHODS: Final = ("p2p", "quad", "ode")


@ft.partial(jax.jit, static_argnames=("method", "method_kw"))
def arc_length(
    spline: interpax.Interpolator1D,
    gamma0: LikeSz0 = -1,
    gamma1: LikeSz0 = 1,
    *,
    method: Literal["p2p", "quad", "ode"] = "p2p",
    method_kw: dict[str, Any] | None = None,
) -> Sz0:
    r"""Return the arc-length of the track.

    .. math::

        s(\gamma_0, \gamma_1) = \int_{\gamma_0}^{\gamma_1} \left\|
        \frac{d\mathbf{x}(\gamma)}{d\gamma} \right\| \, d\gamma


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

    See Also
    --------
    `potamides.Track.arc_length`
        This method auto-vectorizes to support arbitrarily shaped `gamma`
        inputs.
    `potamides.splinelib.arc_length_p2p`
        This method computes the distance between each pair of points along the
        track and sums them up. Accuracy is limited by the number of points
        used. THis can be selected by setting `method="p2p"`.
    `potamides.splinelib.arc_length_quadtrature`
        This method uses fixed quadrature to compute the integral. It is also
        limited in accuracy by the number of points used. This can be selected
        by setting `method="quad"`.
    `potamides.splinelib.arc_length_odeint`
        This method uses ODE integration to compute the integral. This can be
        selected by setting `method="ode"`.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import interpax
    >>> import potamides.splinelib as splib

    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
    >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
    >>> spline = interpax.Interpolator1D(gamma, xy, method="cubic2")

    >>> s = splib.arc_length(spline, 0, 2 * jnp.pi) / jnp.pi
    >>> print(s.round(5))
    4.0

    >>> s = splib.arc_length(spline, 0, 2 * jnp.pi, method="quad") / jnp.pi
    >>> print(s.round(4))
    4.0

    >>> s = splib.arc_length(spline, 0, 2 * jnp.pi, method="ode") / jnp.pi
    >>> print(s.round(5))
    4.0

    """
    kw = method_kw if method_kw is not None else {}
    branches = [
        jtu.Partial(arc_length_p2p, **kw),
        jtu.Partial(arc_length_quadtrature, **kw),
        jtu.Partial(arc_length_odeint, **kw),
    ]
    operands = (spline, gamma0, gamma1)
    return jax.lax.switch(_ARC_LENGTH_METHODS.index(method), branches, *operands)


# ============================================================================
# Acceleration


@ft.partial(jax.jit, inline=True)
def acceleration(spline: interpax.Interpolator1D, gamma: ct.Sz0, /) -> ct.SzF:
    r"""Compute the acceleration vector :math:`\vec{a} = d^2\vec{x}/d\gamma^2`.

    This is the second derivative of the spline position with respect to the
    curve parameter $\gamma$. Equivalently, it's the derivative of the
    tangent vector:

    .. math::

        \vec{a}(\gamma)= \frac{d^2\vec{x}}{d\gamma^2}
        = \frac{d}{d\gamma} (\frac{d\vec{x}}{d\gamma}).

    Parameters
    ----------
    spline : interpax.Interpolator1D
        A twice-differentiable 1D spline for :math:`\vec{x}(\gamma)$`.
    gamma : float
        The scalar parameter value at which to compute the acceleration.

    Returns
    -------
    Array[float, (F,)]
        The acceleration vector :math:`\frac{d^2\vec{x}}{d\gamma^2}` of length F,
        where F is the spatial dimension of the spline.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import interpax
    >>> from potamides.splinelib import acceleration

    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
    >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
    >>> spline = interpax.Interpolator1D(gamma, xy, method="cubic2")

    >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
    >>> acc = jax.vmap(acceleration, in_axes=(None, 0))(spline, gamma)
    >>> print(acc.round(5))
    [[-2.  0.]
     [ 0. -2.]
     [ 2.  0.]]

    """
    return jax.jacfwd(tangent, argnums=1)(spline, gamma)


# ============================================================================
# Curvature


@ft.partial(jax.jit)
def principle_unit_normal(spline: interpax.Interpolator1D, gamma: ct.Sz0, /) -> ct.SzF:
    r"""Return the unit normal vector :math:`\hat{N}(\gamma)` along the spline.

    The unit normal vector is defined as the projection of the acceleration
    vector onto the plane orthogonal to the unit tangent vector, divided by its
    norm:

    .. math::

        \hat{N}(\gamma) = \frac{d\hat{T}/d\gamma}{|d\hat{T}/d\gamma|}.

    where :math:`\hat{T}(\gamma)` is the unit tangent vector at :math:`\gamma` and
    $\vec{a}(\gamma)$ is the acceleration vector at :math:`\gamma`. This function is
    scalar. To compute the unit normal vector at multiple positions, use
    `jax.vmap`.

    Parameters
    ----------
    spline
        The spline interpolator.
    gamma
        The scalar gamma value at which to evaluate the unit normal vector. To
        evaluate the unit normal vector at multiple positions, use `jax.vmap`.

    Returns
    -------
    Array[real, (F,)]
        The unit normal vector at the specified position. The shape is `(F,)`,
        where `F` is the number of dimensions of the spline.

    See Also
    --------
    `potamides.Track.principle_unit_normal`
        This method auto-vectorizes to support arbitrarily shaped `gamma`
        inputs.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import interpax
    >>> import potamides.splinelib as splib

    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
    >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
    >>> spline = interpax.Interpolator1D(gamma, xy, method="cubic2")

    >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
    >>> N_hat = jax.vmap(splib.principle_unit_normal, in_axes=(None, 0))(spline, gamma)
    >>> print(N_hat.round(5))
    [[-1.  0.]
     [ 0. -1.]
     [ 1.  0.]]

    We can reverse the direction of the track by reversing the order of the

    """
    dThat_dgamma = jax.jacfwd(unit_tangent, argnums=1)(spline, gamma)
    return dThat_dgamma / jnp.linalg.vector_norm(dThat_dgamma)


@ft.partial(jax.jit)
def curvature(spline: interpax.Interpolator1D, gamma: ct.Sz0, /) -> ct.SzF:
    r"""Return the curvature vector at a given position along the stream.

    This method computes the curvature by taking the ratio of the gamma
    derivative of the unit tangent vector to the derivative of the
    arc-length with respect to gamma. In other words, if

    .. math::

        \frac{d\hat{T}}{d\gamma} = \frac{ds}{d\gamma} \frac{d\hat{T}}{ds},

    and since the curvature vector is defined as

    .. math::

        \frac{d\hat{T}}{ds} = \kappa \hat{N},

    where :math:`\kappa` is the curvature and :math:`\hat{N}` the unit normal
    vector, then dividing :math:`\frac{d\hat{T}}{d\gamma}` by :math:`
    \frac{ds}{d\gamma}` yields

    .. math::

        \kappa \hat{N} = \frac{d\hat{T}/d\gamma}{ds/d\gamma}.

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

    See Also
    --------
    `potamides.Track.curvature`
        This method auto-vectorizes to support arbitrarily shaped `gamma`
        inputs.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import interpax
    >>> import potamides.splinelib as splib

    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
    >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
    >>> spline = interpax.Interpolator1D(gamma, xy, method="cubic2")

    >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
    >>> kappa_vec = jax.vmap(splib.curvature, (None, 0))(spline, gamma)
    >>> print(kappa_vec.round(5))
    [[-0.5  0. ]
     [ 0.  -0.5]
     [ 0.5  0. ]]

    """
    dThat_dgamma = jax.jacfwd(unit_tangent, argnums=1)(spline, gamma)
    ds_dgamma = speed(spline, gamma)
    return dThat_dgamma / ds_dgamma


@ft.partial(jax.jit)
def kappa(spline: interpax.Interpolator1D, gamma: ct.Sz0, /) -> ct.Sz0:
    r"""Return the curvature magnitude at a given position along the stream.

    Parameters
    ----------
    spline
        The spline interpolator.
    gamma
        The gamma value at which to evaluate the curvature.

    See Also
    --------
    `potamides.Track.kappa`
        This method auto-vectorizes to support arbitrarily shaped `gamma`
        inputs.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import interpax
    >>> import potamides.splinelib as splib

    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
    >>> xy = 2 * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
    >>> spline = interpax.Interpolator1D(gamma, xy, method="cubic2")

    >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
    >>> kappa_val = jax.vmap(splib.kappa, (None, 0))(spline, gamma)
    >>> print(kappa_val.round(5))  # circles have constant curvature
    [0.5 0.5 0.5]

    """
    kappa_vec = curvature(spline, gamma)
    return jnp.linalg.vector_norm(kappa_vec)
