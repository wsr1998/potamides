"""Curvature analysis functions."""

__all__: list[str] = []

from functools import partial

import galax.potential as gp
import jax
import jax.numpy as jnp
import numpy as np
import unxt as u
from jaxtyping import Array, Real
from unxt.quantity import AllowValue
from unxt.unitsystems import galactic

from .custom_types import LikeQorVSz0, LikeSz0, QorVSzN3, Sz0, SzN2

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
def compute_accelerations(
    pos: QorVSzN3,  # [kpc]
    rot_z: LikeQorVSz0 = 0.0,
    rot_x: LikeQorVSz0 = 0.0,
    q1: LikeSz0 = 1.0,
    q2: LikeSz0 = 1.0,
    q3: LikeSz0 = 1.0,
    phi: LikeSz0 = 0.0,
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
        v_c=vc_halo, r_s=rs_halo, q1=q1, q2=q2, q3=q3, phi=phi, units=galactic
    )
    halo_pot = gp.TranslatedPotential(halo_base_pot, translation=origin)

    if withdisk:
        disk_pot = gp.MiyamotoNagaiPotential(m_tot=Mdisk, a=3, b=0.5, units=galactic)

        # Calculate the position in the disk's reference frame
        R = total_rotation(rot_z, rot_x)
        pos_prime = R @ pos
        # Calculate the acceleration in the disk's frame and convert it back to the halo's frame
        acc_disk_prime = disk_pot.acceleration(pos_prime, t=0)
        acc_disk = R.T @ acc_disk_prime
        acc_halo = halo_pot.acceleration(pos, t=0)

        acc = acc_halo + acc_disk
    else:
        acc = halo_pot.acceleration(pos, t=0)

    acc_unit = acc / jnp.linalg.norm(acc, axis=1, keepdims=True)
    acc_xy_unit = acc_unit[:, :2]  # Extract x-y components
    return acc_xy_unit
