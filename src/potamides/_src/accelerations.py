"""Curvature analysis functions."""

__all__: tuple[str, ...] = ()

import functools as ft

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


@ft.partial(jax.jit, inline=True)
def rotation_z(theta_z: Sz0) -> Real[Array, "3 3"]:
    """Rotation about the fixed z-axis by theta_z (counterclockwise)."""
    c, s = jnp.cos(theta_z), jnp.sin(theta_z)
    return jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@ft.partial(jax.jit, inline=True)
def rotation_x(theta_x: Sz0) -> Real[Array, "3 3"]:
    """Rotation about the fixed x-axis by theta_x (counterclockwise)."""
    c, s = jnp.cos(theta_x), jnp.sin(theta_x)
    return jnp.array([[1, 0, 0], [0, c, -s], [0, s, c]])


@ft.partial(jax.jit)
def total_rotation(theta_z: Sz0, theta_x: Sz0) -> Real[Array, "3 3"]:
    """First rotate about z (fixed) by theta_z, then about x (fixed) by theta_x."""
    return rotation_x(theta_x) @ rotation_z(theta_z)


# ============================================================================


@ft.partial(jax.jit, static_argnames=("withdisk",))
def compute_accelerations(
    pos: QorVSzN3,  # [kpc]
    /,
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
    pos : Array[float, (N, 3)]
        An array of shape (N, 3) where N is the number of positions. Each position
        is a 3D coordinate (x, y, z) in kpc.
    rot_z : float, default 0.0
        Rotation angle [radians] around the z-axis (applied first).
    rot_x : float, default 0.0
        Rotation angle [radians] around the x-axis (applied second).
    q1, q2, q3 : float, default 1.0
        Halo axis ratios for the logarithmic potential. q1 and q2 control
        flattening in the x-y plane, q3 controls flattening along z-axis.
    phi : float, default 0.0
        Orientation angle [radians] of the halo potential.
    rs_halo : float, default 16.0
        Halo scale radius [kpc].
    vc_halo : float, default 250 km/s converted to kpc/Myr
        Halo circular velocity.
    origin : Array[float, (3,)], default [0, 0, 0]
        Halo center coordinates [kpc].
    Mdisk : float, default 1.2e10
        Disk mass [Msun]. Only used if `withdisk` is True.
    withdisk : bool, default False
        If True, include a Miyamoto-Nagai disk potential in addition to the halo.

    Returns
    -------
    Array[float, (N, 2)]
        An array of shape (N, 2) representing the planar (x-y) acceleration
        unit vectors at each input position.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> import unxt as u
    >>> import potamides as ptd

    >>> # Basic usage: compute accelerations at a few positions
    >>> positions = jnp.array([
    ...     [8.0, 0.0, 0.0],    # Solar neighborhood
    ...     [0.0, 8.0, 0.0],    # 90 degrees around
    ...     [4.0, 4.0, 1.0],    # Inner galaxy, off-plane
    ... ])
    >>> acc_xy = ptd.compute_accelerations(positions)
    >>> print(f"Shape: {acc_xy.shape}")
    Shape: (3, 2)
    >>> print(f"All finite: {jnp.all(jnp.isfinite(acc_xy))}")
    All finite: True

    >>> # Using quantities with units
    >>> pos_with_units = u.Quantity([8.0, 0.0, 0.0], "kpc").reshape(1, 3)
    >>> acc_xy = ptd.compute_accelerations(pos_with_units)
    >>> print(f"Single position result shape: {acc_xy.shape}")
    Single position result shape: (1, 2)

    >>> # Include disk potential
    >>> acc_xy_disk = ptd.compute_accelerations(positions, withdisk=True)
    >>> print(f"With disk shape: {acc_xy_disk.shape}")
    With disk shape: (3, 2)

    >>> # Custom halo parameters
    >>> acc_xy_custom = ptd.compute_accelerations(
    ...     positions,
    ...     rs_halo=20.0,  # larger scale radius
    ...     vc_halo=u.Quantity(200, "km/s").ustrip("kpc/Myr"),  # slower
    ...     q1=0.8,  # oblate halo
    ...     q2=0.8,
    ... )
    >>> print(f"Custom halo shape: {acc_xy_custom.shape}")
    Custom halo shape: (3, 2)

    >>> # Rotated coordinate system
    >>> import math
    >>> acc_xy_rotated = ptd.compute_accelerations(
    ...     positions,
    ...     rot_z=math.pi/4,  # 45 degree rotation around z
    ...     rot_x=math.pi/6,  # 30 degree rotation around x
    ...     withdisk=True,
    ... )
    >>> print(f"Rotated system shape: {acc_xy_rotated.shape}")
    Rotated system shape: (3, 2)

    >>> # Translated halo center
    >>> acc_xy_translated = ptd.compute_accelerations(
    ...     positions,
    ...     origin=np.array([2.0, -1.0, 0.5]),  # offset halo center
    ... )
    >>> print(f"Translated halo shape: {acc_xy_translated.shape}")
    Translated halo shape: (3, 2)

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
        pos_prime = jnp.einsum("ij,nj->ni", R, pos)
        # Calculate the acceleration in the disk's frame and convert it back to the halo's frame
        acc_disk_prime = disk_pot.acceleration(pos_prime, t=0)
        acc_disk = jnp.einsum("ji,ni->nj", R, acc_disk_prime)
        acc_halo = halo_pot.acceleration(pos, t=0)

        acc = acc_halo + acc_disk
    else:
        acc = halo_pot.acceleration(pos, t=0)

    acc_unit = acc / jnp.linalg.norm(acc, axis=1, keepdims=True)
    acc_xy_unit = acc_unit[:, :2]  # Extract x-y components
    return acc_xy_unit
