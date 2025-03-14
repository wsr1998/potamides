"""Fit smooth spline."""

__all__ = ["make_monotonic_gamma_and_data", "optimize_spline_knots", "split_data"]

from functools import partial

import interpax
import jax
import jax.numpy as jnp
import optax
from jaxopt import OptaxSolver
from jaxtyping import Array, Real

from .custom_types import Sz0, SzData, SzData2, SzN, SzN2

# ============================================================================


def p2p_distance(data: SzData2) -> SzData:
    """Return the distance between points in data."""
    dp2p = jnp.linalg.vector_norm(jnp.diff(data, axis=0), axis=1)
    return dp2p


def make_gamma(data: SzData2) -> SzData:
    """Return gamma, the normalized arc-length of the data.

    Note that this is not guaranteed to be monotonic.
    See `make_monotonic_gamma_and_data` for a monotonic version.

    """
    dp2p = jnp.linalg.vector_norm(jnp.diff(data, axis=0), axis=1)
    gamma = jnp.cumsum(dp2p)
    gamma = 2 * (gamma - gamma.min()) / (gamma.max() - gamma.min()) - 1
    return gamma


# Cut out portions where gamma is not monotonic
def _find_plateau_mask(arr: SzN, /) -> SzN:
    """Return a mask that marks plateaus in the array.

    `True` where it is NOT a plateau. `False` where it is a plateau.
    The first element of a plateau is marked as `True`.

    """
    mask = jnp.ones_like(arr, dtype=bool)
    mask = mask.at[1:].set(
        arr[1:] > arr[:-1]
    )  # Mark as False if the same as previous element
    return mask


def make_monotonic_gamma_and_data(data: SzData2) -> tuple[SzData, SzData2]:
    """Return gamma, the normalized arc-length of the data, and data.

    This version ensures that gamma is monotonically increasing.

    """
    gamma = make_gamma(data)
    points = (data[:-1, :] + data[1:, :]) / 2

    monoticity_mask = _find_plateau_mask(gamma)
    gamma = gamma[monoticity_mask]
    points = points[monoticity_mask]
    return gamma, points


# ============================================================================


def split_data(
    gamma: SzData, data: SzData2, *, num_splits: int
) -> tuple[Real[Array, "{num_splits + 2}"], Real[Array, "{num_splits + 2} data"]]:
    gamma_split = jnp.array_split(gamma, num_splits)
    data_split = jnp.array_split(data, num_splits)

    gamma_median = jnp.array(
        [gamma[0]] + [jnp.median(chunk) for chunk in gamma_split] + [gamma[-1]]
    )
    data_median = jnp.stack(
        [data[0]] + [jnp.median(chunk, axis=0) for chunk in data_split] + [data[-1]]
    )

    return gamma_median, data_median


# ============================================================================


def cost(
    params: SzN2,
    gamma_knots: SzN,
    data_gamma: SzData,
    data_target: SzData,
    sigmas: float = 1.0,
) -> Sz0:
    """Cost function to minimize that compares data to spline fit.

    Parameters
    ----------
    params:
        Output values of spline at gamma_knots -- e.g. x or y values.
        This is the parameter to be optimized to minimize the cost function.
    gamma_knots:
        The gamma values at which the spline is anchored. There are N of these,
        one per `params`. These are fixed while the `params` are optimized.

    data_gamma:
        gamma of the target data.
    data_target:
        The target data. This is the data that the spline is trying to fit.

    sigmas:
        The uncertainty on each datum in `data_target`.

    """
    func = interpax.Interpolator1D(gamma_knots, params, method="cubic2")
    return jnp.sum(((data_target - func(data_gamma)) / sigmas) ** 2)


@partial(jax.jit)
def optimize_spline_knots(
    init_params: SzN2,
    gamma_knots: SzN,
    data_gamma: SzData,
    data_target: SzData2,
    sigmas: float = 1.0,
) -> SzN:
    """Optimize spline knots to fit data.

    Parameters
    ----------
    init_params:
        starting outputs of splines at gamma_knots
    gamma_knots: anchor points for spline
        median gamma in chunk
    data_phi: gamma data
        gamma of every point in data
    data_target: output data
        x,y of every point
    sigmas: uncertainty on each datum

    """

    opt = optax.adam(learning_rate=1e-3)  # TODO: try a better optimizer
    solver = OptaxSolver(opt=opt, fun=cost, maxiter=150_000)

    res = solver.run(
        init_params,
        gamma_knots=gamma_knots,
        data_gamma=data_gamma,
        data_target=data_target,
        sigmas=sigmas,
    )
    return res.params
