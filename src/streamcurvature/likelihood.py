"""Curvature analysis functions."""

__all__: list[str] = []

from functools import partial

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np  # type: ignore[import-not-found]
import numpy.typing as npt  # type: ignore[import-not-found]
from jax import lax
from jaxtyping import Array, Bool, Real

from .custom_types import Sz0, SzN, SzN2

log2pi = jnp.log(2 * jnp.pi)


# ============================================================================


@partial(eqx.filter_jit)
def get_tangents(gamma_eval: Sz0, spline: interpax.Interpolator1D) -> SzN2:
    """Return the tangent vector at a given position along the stream."""
    ddata_dgamma = jax.jacrev(spline)(gamma_eval)
    return ddata_dgamma


@partial(eqx.filter_jit)
def get_unit_tangents(gamma_eval: Sz0, spline: interpax.Interpolator1D) -> SzN2:
    """Return T_hat."""
    tangents = get_tangents(gamma_eval, spline)
    return tangents / jnp.linalg.vector_norm(tangents)


@partial(jax.vmap, in_axes=(0, None))
@partial(eqx.filter_jit)
def get_unit_tangents_and_curvature(
    gamma_eval: Sz0, spline: interpax.Interpolator1D
) -> tuple[SzN2, SzN2]:
    T_hat = get_unit_tangents(gamma_eval, spline)

    dThat_dgamma = jax.jacfwd(get_unit_tangents)(gamma_eval, spline)
    unit_curvature = dThat_dgamma / jnp.linalg.vector_norm(dThat_dgamma)

    return T_hat, unit_curvature


@partial(jax.vmap, in_axes=(0, None))
@partial(eqx.filter_jit)
def get_dl_dgamma(gamma_eval: Sz0, spline: interpax.Interpolator1D) -> Sz0:
    return jnp.hypot(get_tangents(gamma_eval, spline))


# ============================================================================


@partial(jax.jit)
def get_angles(acc_xy_unit: SzN2, kappa_hat: SzN2) -> Real[Array, "N"]:
    r"""Return angle between the normal and acceleration vectors at a position.

    Calculate the angles between the normal vector at given position along the
    stream and the acceleration at given position along the stream.

    Parameters
    ----------
    acc_xy_unit
        An array representing the planar acceleration at each input position.
        Shape (N, 2).
    kappa_hat
        The unit curvature vector (or named normal vector). Shape (N, 2).

    Returns
    -------
    angles
        An array of angles in radians in the range (-pi, pi), with shape (N,).
    """

    dot_product = jnp.einsum("ij,ij->i", acc_xy_unit, kappa_hat)
    cross_product = jnp.cross(acc_xy_unit, kappa_hat)
    return jnp.atan2(cross_product, dot_product)


# ============================================================================


@partial(jax.jit, static_argnames=("tangent_condition", "debug"))
def get_likelihood(
    where_linear: Bool[Array, "N"],
    kappa_hat: SzN2,
    acc_xy_unit: SzN2,
    sigma_theta_deg: float = 10.0,
    *,
    tangent_condition: bool = True,
    debug: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Calculate the likelihood based on the angles between the unit curvature vector at given positions along the stream
    and the acceleration at these positions.

    Parameters:
    ----------
    where_linear
      Boolean array indicating indices where the stream is linear (has no curvature).
    kappa_hat
      An array of shape (N, 2). The unit curvature vector (or named normal vector).
    acc_xy_unit
      An array of shape (N, 2) representing the planar acceleration at each input position.
    sigma_theta_deg
      The standard deviation of the angle between the planar acceleration vectors and the unit curvature vectors, given in degrees.

    tangent_condition
      If `True`, applies a tangent condition that affects the likelihood calculation.

    debug:
      Whether to print debug information. Default `False`.


    Returns:
    ----------
    ln_like : float
      The computed logarithm of the likelihood.
    """
    # 注意thresh_f0和sigma_theta_deg的值是如何算出来的，我还不是很清楚
    #
    N = len(kappa_hat)
    nl = ~where_linear
    N_def = jnp.sum(nl)
    sigma_theta2 = jnp.deg2rad(sigma_theta_deg) ** 2
    # f1: fraction of eval points with compatible curvature vectors and planar accelerations
    acc_curv_align = jnp.where(
        nl[:, None], acc_xy_unit * kappa_hat, jnp.zeros_like(kappa_hat)
    )
    f1 = jnp.sum(jnp.abs(1 + jnp.sign(jnp.sum(acc_curv_align, axis=1))) / 2) / N
    f2 = (N_def / N) - f1
    f3 = 1 - f1 - f2

    def on_true() -> tuple[Sz0, SzN, Sz0, Sz0, Sz0]:
        f1_logf1 = lax.select(jnp.isclose(f1, 0.0), jnp.array(0.0), f1 * jnp.log(f1))
        f2_logf2 = lax.select(jnp.isclose(f2, 0.0), jnp.array(0.0), f2 * jnp.log(f2))
        f3_logf3 = lax.select(jnp.isclose(f3, 0.0), jnp.array(0.0), f3 * jnp.log(f3))

        if tangent_condition:
            acc_linear_align = jnp.where(
                nl[:, None], acc_xy_unit * kappa_hat, jnp.zeros_like(kappa_hat)
            )
            theta_T = (
                jnp.pi / 2 - jnp.arccos(jnp.sum(acc_linear_align, axis=1))
            )  # 不同的约定下，theta_T的结果可能相差一个正负号，但是后面计算中用的theta_T的平方，所以问题不大。
            ln_normal = -0.5 * (
                log2pi + jnp.log(sigma_theta2) + (theta_T - 0) ** 2 / sigma_theta2
            )
            ln_like = N * (f1_logf1 + f2_logf2 + f3_logf3) + jnp.sum(ln_normal)
        else:
            # 这里不考虑tangent condition的意思就是说直接把zero curvature的点全部扔掉
            ln_normal = jnp.zeros(N, dtype=kappa_hat.dtype)
            ln_like = N_def * (f1_logf1 + f2_logf2 + 0.0)

        return ln_like, ln_normal, f1_logf1, f2_logf2, f3_logf3

    def on_false() -> tuple[Sz0, SzN, Sz0, Sz0, Sz0]:
        return (
            -jnp.inf,
            jnp.zeros(N, dtype=kappa_hat.dtype),
            jnp.array(0.0),
            jnp.array(0.0),
            jnp.array(0.0),
        )

    pred = jnp.logical_and(f3 < 0.5, f1 > f2)
    ln_like, ln_normal, f1_logf1, f2_logf2, _ = lax.cond(pred, on_true, on_false)

    if debug:
        jax.debug.print(
            "f1: {f1:.4f}, f2: {f2:.4f}, f3: {f3:.4f}, ln_normal_sum: {sum_ln_normal:.2f}, ln_like: {ln_like:.2f}, f1logf1: {f1_logf1:.3E}, f2logf2: {f2_logf2:.3E}",
            f1=f1,
            f2=f2,
            f3=f3,
            sum_ln_normal=jnp.sum(ln_normal),
            ln_like=ln_like,
            f1_logf1=f1_logf1,
            f2_logf2=f2_logf2,
        )

    return ln_like
