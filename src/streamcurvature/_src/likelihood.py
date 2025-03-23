"""Curvature analysis functions."""

__all__ = [
    "compute_likelihood",
]

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Bool

from .custom_types import Sz0, SzN, SzN2

log2pi = jnp.log(2 * jnp.pi)


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
            ln_lik = N * (f1_logf1 + f2_logf2 + f3_logf3) + jnp.sum(ln_normal)
        else:
            # Not considering the tangent condition means directly discarding all zero-curvature points.
            ln_normal = jnp.zeros(N, dtype=kappa_hat.dtype)
            ln_lik = N * (f1_logf1 + f2_logf2 + 0.0)

        return ln_lik, ln_normal, f1_logf1, f2_logf2, f3_logf3

    @partial(jax.jit)
    def on_false() -> tuple[Sz0, SzN, Sz0, Sz0, Sz0]:
        ln_lik = -jnp.inf
        ln_normal = jnp.zeros(N, dtype=kappa_hat.dtype)
        f1_logf1, f2_logf2, f3_logf3 = jnp.array([0.0, 0.0, 0.0])
        return (ln_lik, ln_normal, f1_logf1, f2_logf2, f3_logf3)

    # TODO: why this exact predicate?
    pred = jnp.logical_and(f3 < 0.5, f1 > f2)
    ln_lik, ln_normal, f1_logf1, f2_logf2, f3_logf3 = lax.cond(pred, on_true, on_false)

    if debug:
        jax.debug.print(
            "f1: {f1:.4f}, f2: {f2:.4f}, f3: {f3:.4f}, ln_normal_sum: {sum_ln_normal:.2f}, ln_lik: {ln_lik:.2f}, f1logf1: {f1_logf1:.3E}, f2logf2: {f2_logf2:.3E}, f3logf3: {f3_logf3:.3E}",
            f1=f1,
            f2=f2,
            f3=f3,
            sum_ln_normal=jnp.sum(ln_normal),
            ln_lik=ln_lik,
            f1_logf1=f1_logf1,
            f2_logf2=f2_logf2,
            f3_logf3=f3_logf3,
        )

    return ln_lik
