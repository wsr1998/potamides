"""Spline-related tools."""

__all__ = [
    # Processing data
    "make_gamma_from_data",
    "make_increasing_gamma_from_data",
    "point_to_point_arclenth",
    "point_to_point_distance",
]

from typing import TypeAlias

import jax.numpy as jnp
from jaxtyping import Array, Real

from potamides._src.custom_types import SzData2, SzN

SzGamma: TypeAlias = Real[Array, "data-1"]
SzGamma2: TypeAlias = Real[Array, "data-1 2"]


# ============================================================================
# Tools for constructing `gamma` from an ordered list of points


def point_to_point_distance(data: SzData2, /) -> SzGamma:
    """Return the distance between points in data.

    The data should be sorted, otherwise this doesn't make a lot of sense.

    Examples
    --------
    >>> import jax.numpy as jnp

    >>> data = jnp.array([[0, 0], [1, 0], [1, 2], [0, 2]])
    >>> point_to_point_distance(data)
    Array([1., 2., 1.], dtype=float64)

    """
    vec_p2p = jnp.diff(data, axis=0)  # vector pointing from p_{i} to p_{i+1}
    d_p2p = jnp.linalg.vector_norm(vec_p2p, axis=1)  # distance = norm of the vecs
    return d_p2p


def point_to_point_arclenth(data: SzData2, /) -> SzGamma:
    """Return a P2P approximation of the arc-length.

    The data should be sorted, otherwise this doesn't make a lot of sense.

    Examples
    --------
    >>> import jax.numpy as jnp

    >>> data = jnp.array([[0, 0], [1, 0], [1, 2], [0, 2]])
    >>> point_to_point_arclenth(data)
    Array([1., 3., 4.], dtype=float64)

    """
    return jnp.cumsum(point_to_point_distance(data))


def make_gamma_from_data(data: SzData2, /) -> SzGamma:
    r"""Return $\gamma$, the normalized arc-length of the data.

    $$ \gamma = 2\frac{s}{L} - 1 , \in [-1, 1] $$

    where $s$ is the arc-length at $\gamma$ and $L$ is the total arc-length.

    Gamma is constructed approximately using a point-to-point approximation (the
    function `point_to_point_arclenth`).

    Notes
    -----
    This is guaranteed to be monotonically non-decreasing since the
    point-to-point arc-length is always non-negative. However, this is not
    guaranteed to be monotonically *increasing* since adjacent data points can
    have 0 distance. See `make_increasing_gamma_from_data` for a function that
    trims the data such that gamma is monotonically increasing.

    Examples
    --------
    >>> import jax.numpy as jnp

    >>> data = jnp.array([[0, 0], [1, 0], [1, 2], [0, 2]])
    >>> make_gamma_from_data(data)
    Array([-1.        ,  0.33333333,  1.        ], dtype=float64)

    """
    s = point_to_point_arclenth(data)  # running arc-length
    s_min = s.min()
    gamma = 2 * (s - s_min) / (s.max() - s_min) - 1  # normalize to range
    return gamma


# -------------------------------------


# Cut out portions where gamma is not monotonic
def _find_plateau_mask(arr: SzN, /) -> SzN:
    """Return a mask that marks plateaus in the array.

    `True` where it is NOT a plateau. `False` where it is a plateau. The first
    element of a plateau is marked as `True`.

    """
    # Mark True where increasing (x_{i+1} > x_i)
    mask = jnp.ones_like(arr, dtype=bool)
    mask = mask.at[1:].set(arr[1:] > arr[:-1])
    return mask


def make_increasing_gamma_from_data(data: SzData2, /) -> tuple[SzGamma, SzGamma2]:
    r"""Return the trimmed data and gamma, the normalized arc-length.

    $$ \gamma = 2\frac{s}{L} - 1 , \in [-1, 1] $$

    where $s$ is the arc-length at $\gamma$ and $L$ is the total arc-length.

    Gamma is constructed approximately using a point-to-point (P2P)
    approximation (the function `point_to_point_arclenth`). Using the P2P
    arc-length is not guaranteed to be monotonically *increasing* since adjacent
    data points can have 0 distance. This function then trims the data such that
    gamma is monotonically increasing, keeping the first point of any plateau.

    Returns
    -------
    gamma : Array[real, (N-1,)]
        Monotonically increasing normalized arc-length.
    data_trimmed : Array[real, (N-1, 2)]
        The data, with points where gamma is non-increasing trimmed out.

    Examples
    --------
    >>> import jax.numpy as jnp

    >>> data = jnp.array([[0, 0], [1, 0], [1, 2], [1, 2], [0, 2]])
    >>> gamma, data2 = make_increasing_gamma_from_data(data)
    >>> gamma, data2
    (Array([-1.        ,  0.33333333,  1.        ], dtype=float64),
     Array([[0.5, 0. ],
           [1. , 1. ],
           [0.5, 2. ]], dtype=float64))

    Note that the second point [1, 2] was removed since it was a repeat,
    resulting in a "plateau" in gamma. Then the point-to-point mean was returned
    as the new data.

    """
    # Define gamma from the data using p2p approximation
    gamma = make_gamma_from_data(data)  # (N,2) -> (N-1,)
    # The length of gamma is N-1 and we need the data to match. The easiest
    # solution is to just take the mean of adjacent points, since gamma is
    # defined from the p2p approximation.
    data_mean = (data[:-1, :] + data[1:, :]) / 2

    # Find where gamma is non-increasing -- has plateaued.
    where_increasing = _find_plateau_mask(gamma)

    # Cut out all plateaus
    gamma = gamma[where_increasing]
    data_mean = data_mean[where_increasing]

    return gamma, data_mean
