"""Curvature analysis functions."""

__all__: list[str] = []

import logging

import gala.potential as gp  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
import numpy.typing as npt  # type: ignore[import-not-found]
from astropy import units as u  # type: ignore[import-not-found]
from gala.units import galactic  # type: ignore[import-not-found]


def rotation(beta: float, alpha: float) -> npt.NDArray[np.float64]:
    c_1, s_1 = np.cos(beta), np.sin(beta)
    R_y = np.array([[c_1, 0, s_1], [0, 1, 0], [-s_1, 0, c_1]])
    c_2, s_2 = np.cos(alpha), np.sin(alpha)
    R_z = np.array([[c_2, -s_2, 0], [s_2, c_2, 0], [0, 0, 1]])
    return R_z @ R_y


default_origin: u.Quantity = np.array([0.0, 0.0, 0.0]) * u.kpc


def get_acceleration(
    pos: npt.NDArray[np.float64],
    beta: float,
    alpha: float,
    q1: float,
    q2: float,
    rs_halo: u.Quantity = 16 * u.kpc,
    vc_halo: u.Quantity = 250 * u.km / u.s,
    Mdisk: u.Quantity = 1.2e10 * u.Msun,
    origin: u.Quantity = default_origin,
    *,
    withdisk: bool,
) -> npt.NDArray[np.float64]:
    """
    Calculate the planar acceleration (x-y plane, ignoring the z-component along the line-of-sight direction) at each given position.

    The gravitational potentials are modeled using two types: a Logarithmic potential for the halo and a Miyamoto-Nagai potential for the disk, if included.

    Parameters
    ----------
    - pos : array_like
      An array of shape (N, 3) where N is the number of postitons. Each posititon is a 3D coordinate (x, y, z).
    - withdisk : bool
      If True, the graivatioanl potential of the disk is included.
    - beta, alpha : float
      Rotation angle around the y-axis and z-axis, respectively, in the unit of radians.

    Returns
    -------
    - acc_xy_unit : array_like
      An array of shape (N, 2) representing the planar acceleration at each input position.

    Examples
    --------
    >>> import numpy as np

    """

    if withdisk:
        halo_pot = gp.LogarithmicPotential(
            v_c=vc_halo, r_h=rs_halo, q1=q1, q2=q2, q3=1, units=galactic, origin=origin
        )
        disk_pot = gp.MiyamotoNagaiPotential(
            m=Mdisk, a=3 * u.kpc, b=0.5 * u.kpc, units=galactic
        )

        # Calculate the position in the disk's reference frame
        R = rotation(beta, alpha)
        pos_prime = R @ pos
        # Calculate the acceleration in the disk's frame and convert it back to the halp's frame
        acc_disk_prime = disk_pot.acceleration(pos_prime * u.kpc).value
        acc_disk = R.T @ acc_disk_prime
        acc_disk = acc_disk.T
        acc_halo = halo_pot.acceleration(pos * u.kpc).value.T

        acc = acc_halo + acc_disk
    else:
        halo_pot = gp.LogarithmicPotential(
            v_c=vc_halo,
            r_h=rs_halo,
            q1=q1,
            q2=q2,
            q3=1,
            units=galactic,
            origin=np.array([0.0, 0.0, 0.0]) * u.kpc,
        )
        acc = halo_pot.acceleration(pos * u.kpc).value.T

    acc_unit = acc / np.linalg.norm(acc, axis=1, keepdims=True)
    acc_xy_unit = acc_unit[:, :2]  # Extract x-y components
    return acc_xy_unit


def get_angles(
    acc_xy_unit: npt.NDArray[np.float64], kappa_hat: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Calculate the angles between the normal vector at given position along the stream and the acceleration at given position along the stream.

    Parameters:
    ----------
    - acc_xy_unit : array_like
      An array of shape (N, 2) representing the planar acceleration at each input position.
    - kappa_hat : array_like
      An array of shape (N, 2). The unit curvature vector (or named normal vector).

    Returns:
    ----------
    - array_like
      An array of angles in radians in the range ``[-pi, pi]``, with shape (N,).
    """

    dot_product = np.einsum("ij,ij->i", acc_xy_unit, kappa_hat)
    cross_product = np.cross(acc_xy_unit, kappa_hat)
    return np.arctan2(cross_product, dot_product)


def get_likelihood(
    undef_bool: npt.NDArray[np.bool],
    kappa_hat: npt.NDArray[np.float64],
    acc_xy_unit: npt.NDArray[np.float64],
    sigma_theta_deg: float = 10.0,
    *,
    tangent_condition: bool = True,
    debug: bool = True,
) -> npt.NDArray[np.float64]:
    """
    Calculate the likelihood based on the angles between the unit curvature vector at given positions along the stream
    and the acceleration at these positions.

    Parameters:
    ----------
    - curvature : array_like
      An array of shape (N, 2) where N is the number of postitons along the stream. Each posititon is a 2D coordinate (x, y).
    - kappa_hat : array_like
      An array of shape (N, 2). The unit curvature vector (or named normal vector).
    - acc_xy_unit : array_like
      An array of shape (N, 2) representing the planar acceleration at each input position.
    - tangent_condition : bool
      If True, applies a tangent condition that affects the likelihood calculation.
    - thresh_f0 : float
      A  threshold value for curvature calculation. Curvature values smaller than this are considered too close to zero to be reliable.
    - sigma_theta_deg : float
      The standard deviation of the angle between the planar acceleration vectors and the unit curvature vectors, given in degrees.

    Returns:
    ----------
    - log_like : float
      The computed logarithm of the likelihood.
    """
    # 注意thresh_f0和sigma_theta_deg的值是如何算出来的，我还不是很清楚
    #
    N = len(kappa_hat)
    N_def = np.sum(~undef_bool)
    sigma_theta = np.deg2rad(sigma_theta_deg)
    f1 = (
        np.sum(
            0.5
            * np.abs(
                1.0
                + np.sign(
                    np.sum(
                        acc_xy_unit[~undef_bool, :] * kappa_hat[~undef_bool, :], axis=1
                    )
                )
            )
        )
        / N
    )
    f2 = (N_def / N) - f1
    f3 = 1 - f1 - f2

    log_like = -np.inf
    log_gauss: npt.NDArray[np.float64] = np.array(0)
    f1_logf1: npt.NDArray[np.float64] = np.array(0)
    f2_logf2: npt.NDArray[np.float64] = np.array(0)

    if f3 < 0.5 and f1 > f2:
        f1_logf1 = 0.0 if np.isclose(f1, 0.0) else f1 * np.log(f1)
        f2_logf2 = 0.0 if np.isclose(f2, 0.0) else f2 * np.log(f2)
        f3_logf3 = 0.0 if np.isclose(f3, 0.0) else f3 * np.log(f3)
        if tangent_condition:
            theta_T = (
                np.pi / 2
                - np.arccos(
                    np.sum(
                        acc_xy_unit[undef_bool, :] * kappa_hat[undef_bool, :], axis=1
                    )
                )
            )  # 不同的约定下，theta_T的结果可能相差一个正负号，但是后面计算中用的theta_T的平方，所以问题不大。
            log_gauss = -1 / 2 * np.log(2 * np.pi * sigma_theta**2) - (
                theta_T - 0
            ) ** 2 / (2 * sigma_theta**2)
            log_like = N * (f1_logf1 + f2_logf2 + f3_logf3) + np.sum(log_gauss)
        else:
            log_like = N_def * (
                f1_logf1 + f2_logf2 + 0.0
            )  # 这里不考虑tangent condition的意思就是说直接把zero curvature的点全部扔掉

    if debug:
        logger = logging.getLogger("streamcurvature")
        msg = f"f1: {f1:.4f}, f2: {f2:.4f}, f3: {f3:.4f}, log_gauss_sum: {np.sum(log_gauss):.2f}, log_like: {log_like:.2f}, f1logf1: {f1_logf1:.3E}, f2logf2: {f2_logf2:.3E}"
        logger.info(msg)
    return log_like
