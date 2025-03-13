"""Curvature analysis functions."""

__all__: list[str] = []

from functools import partial
from typing import TypeAlias

import galax.potential as gp  # type: ignore[import-not-found]
import jax
import jax.numpy as jnp
import numpy as np  # type: ignore[import-not-found]
import numpy.typing as npt  # type: ignore[import-not-found]
import unxt as u
from jax import lax
from jaxtyping import Array, Bool, Real
from unxt.quantity import AllowValue
from unxt.unitsystems import galactic

Sz0: TypeAlias = Real[Array, ""]
LikeSz0: TypeAlias = Real[Array, ""] | float | int
LikeQorVSz0: TypeAlias = Real[u.Quantity, ""] | LikeSz0
SzN: TypeAlias = Real[Array, "N"]
SzN2: TypeAlias = Real[Array, "N 2"]
SzN3: TypeAlias = Real[Array, "N 3"]
QuSzN3: TypeAlias = Real[u.AbstractQuantity, "N 3"]
QorVSzN3: TypeAlias = SzN3 | QuSzN3


log2pi = jnp.log(2 * jnp.pi)

# ============================================================================


@partial(jax.jit, inline=True)
def rotation_z(theta_z: Sz0) -> Real[Array, "3 3"]:
    """Rotation about the fixed z-axis by theta_z (counterclockwise)."""
    c, s = jnp.cos(theta_z), jnp.sin(theta_z)
    return jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@partial(jax.jit, inline=True)
def rotation_x(theta_x: Sz0) -> Real[Array, "3 3"]:
    """Rotation about the fixed x-axis by theta_x (counterclockwise)."""
    c, s = jnp.cos(theta_x), jnp.sin(theta_x)
    return jnp.array([[1, 0, 0], [0, c, -s], [0, s, c]])


@partial(jax.jit)
def total_rotation(theta_z: Sz0, theta_x: Sz0) -> Real[Array, "3 3"]:
    """First rotate about z (fixed) by theta_z, then about x (fixed) by theta_x."""
    return rotation_x(theta_x) @ rotation_z(theta_z)


# ============================================================================


@partial(jax.jit, static_argnames=("withdisk",))
def get_acceleration(
    pos: QorVSzN3,  # [kpc]
    rot_z: LikeQorVSz0,
    rot_x: LikeQorVSz0,
    q1: LikeSz0,
    q2: LikeSz0,
    rs_halo: LikeQorVSz0 = 16,  # [kpc]
    vc_halo: LikeQorVSz0 = u.Quantity(250, "km / s").ustrip("kpc/Myr"),
    origin: LikeQorVSz0 = np.array([0.0, 0.0, 0.0]),  # [kpc]
    Mdisk: LikeQorVSz0 = 1.2e10,  # [Msun]
    *,
    withdisk: bool = False,
) -> SzN2:
    """
    Calculate the planar acceleration (x-y plane, ignoring the z-component along the line-of-sight direction) at each given position.

    The gravitational potentials are modeled using two types: a Logarithmic potential for the halo and a Miyamoto-Nagai potential for the disk, if included.

    Parameters
    ----------
    pos
      An array of shape (N, 3) where N is the number of postitons. Each posititon is a 3D coordinate (x, y, z).
    beta, rot_zp
      Rotation angle [radians] around the y-axis and z'-axis, respectively.
    q1, q2
      Halo flattening.
    rs_halo, vc_halo
      Halo scale radius and circular velocity
    origin
      Halo center
    Mdisk
      Disk mass. Only used if `withdisk` is `True`.

    withdisk
      If `True` the graivational potential of the disk is included. If `False` (default) it is not.

    Returns
    -------
    acc_xy_unit
      An array of shape (N, 2) representing the planar (XY) acceleration unit vectors at each input position.

    Examples
    --------
    >>> import numpy as np

    """
    pos = u.ustrip(AllowValue, galactic["length"], pos)  # Q(/Array) -> Array

    halo_base_pot = gp.LMJ09LogarithmicPotential(
        v_c=vc_halo, r_s=rs_halo, q1=q1, q2=q2, q3=1, phi=0, units=galactic
    )

    if withdisk:
        halo_pot = gp.TranslatedPotential(halo_base_pot, translation=origin)

        disk_pot = gp.MiyamotoNagaiPotential(m_tot=Mdisk, a=3, b=0.5, units=galactic)

        # Calculate the position in the disk's reference frame
        R = total_rotation(rot_z, rot_x)
        pos_prime = R @ pos
        # Calculate the acceleration in the disk's frame and convert it back to the halp's frame
        acc_disk_prime = disk_pot.acceleration(pos_prime, t=0)
        acc_disk = R.T @ acc_disk_prime
        acc_halo = halo_pot.acceleration(pos, t=0)

        acc = acc_halo + acc_disk
    else:
        acc = halo_base_pot.acceleration(pos, t=0)

    acc_unit = acc / jnp.linalg.norm(acc, axis=1, keepdims=True)
    acc_xy_unit = acc_unit[:, :2]  # Extract x-y components
    return acc_xy_unit


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
