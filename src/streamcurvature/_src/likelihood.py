"""Curvature analysis functions."""

__all__ = [
    "compute_dThat_dgamma",
    "compute_darclength_dgamma",
    "compute_tangent",
    "compute_unit_curvature",
    "compute_unit_tangent",
]

from functools import partial

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Bool

from .custom_types import Sz0, Sz2, SzN, SzN2

log2pi = jnp.log(2 * jnp.pi)


@partial(eqx.filter_jit)
@partial(jnp.vectorize, signature="()->(2)", excluded=(0,))
def compute_tangent(
    spline: interpax.Interpolator1D, gamma_eval: Sz0, /, *, forward: bool = True
) -> SzN2:
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
    gamma_eval
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

    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 600)
    >>> x = 2 * jnp.cos(gamma)
    >>> y = 2 * jnp.sin(gamma)
    >>> spline = interpax.Interpolator1D(gamma, jnp.stack([x, y], axis=-1))

    >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
    >>> tangents = jnp.array([sc.compute_tangent(spline, g) for g in gamma])
    >>> print(tangents)  # Tangents at gamma = 0, pi/2, pi
    [[ 0.  2.]
     [-2.  0.]
     [ 0. -2.]]

    """
    tangent_fn = (jax.jacfwd if forward else jax.jacrev)(spline)
    return tangent_fn(gamma_eval)


@partial(eqx.filter_jit)
@partial(jnp.vectorize, signature="()->(2)", excluded=(0,))
def compute_unit_tangent(
    spline: interpax.Interpolator1D, gamma_eval: Sz0, /, *, forward: bool = True
) -> Sz2:
    r"""Compute the unit tangent vector at a given position along the stream.

    The unit tangent vector is defined as:

    $$ \hat{\mathbf{T}} = \mathbf{T} / \|\mathbf{T}\| $$

    This function is scalar. To compute the unit tangent vector at multiple
    positions, use `jax.vmap`.

    Parameters
    ----------
    gamma_eval
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

    >>> gamma = jnp.linspace(0, 2 * jnp.pi, 600)
    >>> x = jnp.cos(gamma)
    >>> y = jnp.sin(gamma)
    >>> spline = interpax.Interpolator1D(gamma, jnp.stack([x, y], axis=-1))

    >>> gamma = jnp.array([0, jnp.pi / 2, jnp.pi])
    >>> unit_tangents = sc.compute_unit_tangent(spline, gamma)
    >>> print(unit_tangents)  # Unit tangents at gamma = 0, pi/2, pi
    [[ 0.  1.]
     [-1.  0.]
     [ 0. -1.]]

    """
    tangents = compute_tangent(spline, gamma_eval, forward=forward)
    return tangents / jnp.linalg.vector_norm(tangents)


@partial(eqx.filter_jit)
@partial(jnp.vectorize, signature="()->(2)", excluded=(0,))
def compute_dThat_dgamma(
    spline: interpax.Interpolator1D, gamma_eval: Sz0, /, *, forward: bool = True
) -> Sz2:
    r"""Return the derivative of the unit tangent vector with respect to gamma.

    .. note::

        The following applies if $\gamma$ is proportional to the arc-length.

    The derivative of the unit tangent vector with respect to $\gamma$ can
    relate to the curvature vector. If $\gamma$ is defined as the arc-length
    parameter, normalized to the range [-1, 1], then the derivative of the unit
    tangent vector with respect to $\gamma$ is the scaled curvature vector.

    The curvature vector is defined as the derivative of the unit tangent vector
    with respect to arc-length $s$:

    $$ \frac{d\hat{T}}{ds} = \kappa \hat{N}, $$

    where $\kappa$ is the curvature (its magnitude) and \hat{N} is the principal
    unit normal vector.

    If $\gamma$ is proportional to the arc-length, we can write

    $$ \gamma = \frac{2s}{L} - 1, $$

    where $L$ is the total arc-length of the stream. Then by the chain rule, we
    have

    $$ \frac{d\hat{T}}{d\gamma} = \frac{ds}{d\gamma} \frac{d\hat{T}}{ds}. $$

    Because $\frac{ds}{d\gamma} = \frac{L}{2},$ and $\frac{d\hat{T}}{ds} =
    \kappa\,\hat{N},$ it follows that

    $$
        \frac{d\hat{T}}{d\gamma} = \frac{L}{2} \kappa \hat{N} \propto \kappa
        \hat{N}.
    $$

    Therefore the derivative of the unit tangent vector with respect to gamma is
    proportional to the curvature vector.

    """
    dThat_dgamma_fn = (jax.jacfwd if forward else jax.jacrev)(
        compute_unit_tangent, argnums=1
    )
    return dThat_dgamma_fn(spline, gamma_eval, forward=forward)


@partial(eqx.filter_jit)
@partial(jnp.vectorize, signature="()->(2)", excluded=(0,))
def compute_unit_curvature(
    spline: interpax.Interpolator1D, gamma_eval: Sz0, /, *, forward: bool = True
) -> Sz2:
    r"""Return the unit curvature vector at a given position along the stream.

    .. warning::

        This function assumes that the input gamma is proportional to the
        arc-length. If this is not the case, the unit-curvature vector may not
        be accurate.

    See `compute_dThat_dgamma` for the relationship between the gamma-derivative
    of the unit-tangent vector and the curvature vector.

    For

    $$ \frac{d\hat{T}}{d\gamma} \propto \kappa \hat{N}, $$

    where $\kappa \hat{N}$ is the curvature vector and $\hat{N}$ is the unit
    normal vector (aka unit curvature vector), it follows that

    $$ \hat{N} = \frac{\kappa \hat{N}}{\|\kappa \hat{N}\|}. $$

    """
    dThat_dgamma = compute_dThat_dgamma(spline, gamma_eval, forward=forward)
    unit_curvature = dThat_dgamma / jnp.linalg.vector_norm(dThat_dgamma)
    return unit_curvature


@partial(eqx.filter_jit)
@partial(jnp.vectorize, signature="()->(2)", excluded=(0,))
def compute_darclength_dgamma(
    spline: interpax.Interpolator1D, gamma_eval: Sz0, /, *, forward: bool = True
) -> Sz0:
    """Return the derivative of the arc-length with respect to gamma.

    On a 2D flat surface (the flat-sky approximation is reasonable for
    observations of extragalactic stellar streams) the differential arc-length
    is given by:

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

    Since this is a constant, there is no need to compute this function. It is
    sufficient to just use $L/2$. This function is provided for completeness.

    """
    # TODO: confirm that this equals L/2 for gamma \propto s
    return jnp.hypot(compute_tangent(spline, gamma_eval, forward=forward))


# ============================================================================


@partial(jax.jit, static_argnames=("debug",))
def compute_likelihood(
    kappa_hat: SzN2,
    acc_xy_unit: SzN2,
    where_straight: Bool[Array, "N"] | None = None,
    *,
    sigma_theta: float = jnp.deg2rad(10.0),
    debug: bool = False,
) -> SzN:
    """Return the likelihood of the accelerations given the track's curvature.

    Calculates the likelihood based on the angles between the unit curvature
    vector at given positions along the stream and the acceleration at these
    positions. Since the accelerations are computed from a model of a
    gravitational potential, the likelihood is a goodness of fit measure of the
    potential.

    Parameters:
    ----------
    kappa_hat
      An array of shape (N, 2). The unit curvature vector (or named normal
      vector).
    acc_xy_unit
      An array of shape (N, 2) representing the planar acceleration at each
      input position.
    where_straight
      Boolean array indicating indices where the stream is linear (has no
      curvature). If `None`, all points are assumed to be curved.
    sigma_theta
      The standard deviation of the angle between the planar acceleration
      vectors and the unit curvature vectors, given in radians. only used if
      `where_straight` is an array with `True` elements.

    debug:
      Whether to print debug information. Default `False`.

    Returns:
    ----------
    Array[real, (n,)]
      The computed logarithm of the likelihood.

    """

    # Number of evaluation points
    N = len(kappa_hat)

    # Determine which points are curved. If `where_straight` is not provided,
    # assume all points are curved.
    where_curved = (
        jnp.ones(N, dtype=bool) if where_straight is None else ~where_straight
    )
    num_curved = jnp.sum(where_curved)  # Count the number of curved points

    # TODO: figure out thresh_f0 and sigma_theta values.
    # Compute the variance of the angle in radians from the given standard deviation in degrees
    sigma_theta2 = sigma_theta**2
    # f1: fraction of eval points with compatible curvature vectors and planar accelerations
    acc_curv_align = jnp.where(
        where_curved[:, None], acc_xy_unit * kappa_hat, jnp.zeros_like(kappa_hat)
    )
    f1 = jnp.sum(jnp.abs(1 + jnp.sign(jnp.sum(acc_curv_align, axis=1))) / 2) / N
    f2 = (num_curved / N) - f1
    f3 = 1 - f1 - f2

    @partial(jax.jit)
    def on_true() -> tuple[Sz0, SzN, Sz0, Sz0, Sz0]:
        f1_logf1 = lax.select(jnp.isclose(f1, 0.0), jnp.array(0.0), f1 * jnp.log(f1))
        f2_logf2 = lax.select(jnp.isclose(f2, 0.0), jnp.array(0.0), f2 * jnp.log(f2))
        f3_logf3 = lax.select(jnp.isclose(f3, 0.0), jnp.array(0.0), f3 * jnp.log(f3))

        if where_straight is not None:
            acc_linear_align = jnp.where(
                ~where_curved[:, None],
                acc_xy_unit * kappa_hat,
                jnp.zeros_like(kappa_hat),
            )
            # Depending on the convention, the result of theta_T may differ by a sign, but since the square of theta_T is used later in the calculation, this is not a significant issue.
            theta_T = jnp.pi / 2 - jnp.arccos(jnp.sum(acc_linear_align, axis=1))
            ln_normal = -0.5 * (
                log2pi + jnp.log(sigma_theta2) + (theta_T - 0) ** 2 / sigma_theta2
            )
            ln_like = N * (f1_logf1 + f2_logf2 + f3_logf3) + jnp.sum(ln_normal)
        else:
            # Not considering the tangent condition means directly discarding all zero-curvature points.
            ln_normal = jnp.zeros(N, dtype=kappa_hat.dtype)
            ln_like = N * (f1_logf1 + f2_logf2 + 0.0)

        return ln_like, ln_normal, f1_logf1, f2_logf2, f3_logf3

    @partial(jax.jit)
    def on_false() -> tuple[Sz0, SzN, Sz0, Sz0, Sz0]:
        ln_like = -jnp.inf
        ln_normal = jnp.zeros(N, dtype=kappa_hat.dtype)
        f1_logf1, f2_logf2, f3_logf3 = jnp.array([0.0, 0.0, 0.0])
        return (ln_like, ln_normal, f1_logf1, f2_logf2, f3_logf3)

    # TODO: why this exact predicate?
    pred = jnp.logical_and(f3 < 0.5, f1 > f2)
    ln_like, ln_normal, f1_logf1, f2_logf2, f3_logf3 = lax.cond(pred, on_true, on_false)

    if debug:
        jax.debug.print(
            "f1: {f1:.4f}, f2: {f2:.4f}, f3: {f3:.4f}, ln_normal_sum: {sum_ln_normal:.2f}, ln_like: {ln_like:.2f}, f1logf1: {f1_logf1:.3E}, f2logf2: {f2_logf2:.3E}, f3logf3: {f3_logf3:.3E}",
            f1=f1,
            f2=f2,
            f3=f3,
            sum_ln_normal=jnp.sum(ln_normal),
            ln_like=ln_like,
            f1_logf1=f1_logf1,
            f2_logf2=f2_logf2,
            f3_logf3=f3_logf3,
        )

    return ln_like
