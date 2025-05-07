"""Curvature analysis functions."""

__all__ = ["combine_ln_likelihoods", "compute_ln_lik_curved", "compute_ln_likelihood"]

import functools as ft
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Int, Real

from .custom_types import BoolSzGamma, Sz0, SzGamma2

log2pi = jnp.log(2 * jnp.pi)


@ft.partial(jax.jit)
def compute_ln_lik_curved(
    ngamma: Sz0, f1_logf1: Sz0, f2_logf2: Sz0, f3_logf3: Sz0
) -> Sz0:
    """Log-Likelihood of the curved part of the stream."""
    return ngamma * (f1_logf1 + f2_logf2 + f3_logf3)


@ft.partial(jax.jit)
def compute_lnlik_good(
    kappa_hat: SzGamma2,
    acc_xy_unit: SzGamma2,
    where_straight: BoolSzGamma,
    f1_logf1: Sz0,
    f2_logf2: Sz0,
    f3_logf3: Sz0,
    sigma_theta: float,
) -> Sz0:
    # Log-likelihood of the curved part of the stream
    lnlik_curved = compute_ln_lik_curved(len(kappa_hat), f1_logf1, f2_logf2, f3_logf3)

    # TODO: it is more efficient to lax cond on where_straight having any True.

    # Log-likelihood of the straight part of the stream
    # If no part is straight then `acc_linear_align` is all zeros
    acc_linear_align = jnp.where(
        where_straight[:, None],
        acc_xy_unit * kappa_hat,
        jnp.zeros_like(kappa_hat),
    )
    # Angle between planar acceleration and stream track (Nibauer et al. 2023,
    # Eq. 15). acc_linear_align = 0 => theta_T = 0
    theta_T = jnp.pi / 2 - jnp.arccos(jnp.sum(acc_linear_align, axis=1))
    # The likelihoods of the straight segment (Nibauer et al. 2023, Eq. 16)
    ln_normal = -0.5 * (
        log2pi + 2 * jnp.log(sigma_theta) + (theta_T - 0) ** 2 / sigma_theta**2
    )
    lnlik_straight = jnp.sum(ln_normal)  # sum to get the total likelihood

    # Return the total log-likelihood
    return lnlik_curved + lnlik_straight


@ft.partial(jax.jit)
def compute_lnlik_bad(*_: Any) -> Sz0:
    """Log-Likelihood when the majority of the curved segments are incompatible."""
    return -jnp.inf


@ft.partial(jax.jit)
def compute_ln_likelihood(
    kappa_hat: SzGamma2,
    acc_xy_unit: SzGamma2,
    where_straight: BoolSzGamma | None = None,
    *,
    sigma_theta: float = jnp.deg2rad(10.0),
) -> Sz0:
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
    # ---------------------------------------------------
    # Compute the 'fractions' f1, f2, f3 (Eq. 18 of Nibauer et al. 2023)

    # - f1: fraction of eval points with compatible curvature vectors and planar
    #   accelerations, where compatible means that theta -- the angle between
    #   the unit curvature vector and the planar acceleration vector -- is less
    #   than pi/2.
    N = len(kappa_hat)  # Number of gamma points
    where_curved = jnp.ones(N, bool) if where_straight is None else ~where_straight
    acc_curv_align: SzGamma2 = jnp.where(
        where_curved[:, None], acc_xy_unit * kappa_hat, jnp.zeros_like(kappa_hat)
    )
    f1 = jnp.sum(jnp.abs(1 + jnp.sign(jnp.sum(acc_curv_align, axis=1))) / 2) / N

    # - f2: fraction of eval points with incompatible curvature vectors and
    #   planar accelerations.
    num_curved = jnp.sum(where_curved)  # number of curved points
    f2 = (num_curved / N) - f1

    # - f3: is the fraction of evaluation points with undefined curvature
    #   vectors. This is fixed for each stream track and therefore doesn't
    #   really matter since the likelihoods are ultimately divided by the
    #   maximum likelihood, so this term will cancel out.
    f3 = 1 - (f1 + f2)

    # We actually need f * log(f).
    f1_logf1 = lax.select(jnp.isclose(f1, 0.0), jnp.array(0.0), f1 * jnp.log(f1))
    f2_logf2 = lax.select(jnp.isclose(f2, 0.0), jnp.array(0.0), f2 * jnp.log(f2))
    f3_logf3 = lax.select(jnp.isclose(f3, 0.0), jnp.array(0.0), f3 * jnp.log(f3))

    # ---------------------------------------------------

    # The likelihood is degenerate with the "f" parameters. To break the degeneracy we require f1 > f2 (Nibauer et al. 2023, Eq. 20).
    mostly_good = f1 > f2
    operands = (
        kappa_hat,
        acc_xy_unit,
        ~where_curved,  # NOTE: the inversion
        f1_logf1,
        f2_logf2,
        f3_logf3,
        sigma_theta,
    )
    ln_lik = lax.cond(mostly_good, compute_lnlik_good, compute_lnlik_bad, *operands)

    return ln_lik


@ft.partial(jnp.vectorize, signature="(n),(n),(n)->()")
@ft.partial(jax.jit)
def combine_ln_likelihoods(
    lnliks: Real[Array, "S"],
    /,
    ngammas: Int[Array, "S"],
    arclengths: Real[Array, "S"],
) -> Sz0:
    """Combine likelihoods from different stream segments.

    Parameters
    ----------
    lnliks
        The log-likelihoods of the stream segments.
    ngammas
        The number of gamma points in each segment.
    arclengths
        The total arclengths of the stream segments.

    """
    # Compute the mean measurement density of the stream segments. This is the
    # ratio of the total number of gamma points to the total arclength.
    mean_gamma_density = jnp.sum(ngammas) / jnp.sum(arclengths)

    # Compute the weights for each segment. This is the ratio of the total
    # measurement density to the measurement density of each segment. For
    # streams with lower measurement density the likelihood is up-weighted and
    # vice versa for streams with higher measurement density.
    gamma_densities = ngammas / arclengths
    weights = mean_gamma_density / gamma_densities

    # Compute the weighted log-likelihoods
    lnliks_weighted = weights * lnliks
    # TODO: does this need to be normalized by the sum of the weights?

    # Return the total log-likelihood
    return jnp.sum(lnliks_weighted)
