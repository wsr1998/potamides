"""Spline-related tools."""

__all__ = [  # noqa: RUF022
    # Processing data
    "make_gamma_from_data",
    "make_increasing_gamma_from_data",
    "point_to_point_arclenth",
    "point_to_point_distance",
    # Optimizing splines
    "reduce_point_density",
    "data_distance_cost_fn",
    "curvature_cost_fn",
    "default_cost_fn",
    "optimize_spline_knots",
    # Utils
    "interpax_PPoly_from_scipy_UnivariateSpline",
]

import functools as ft
from collections.abc import Callable
from typing import Any, Protocol, TypeAlias, runtime_checkable

import interpax
import jax
import jax.numpy as jnp
import optax
import scipy.interpolate
from jaxtyping import Array, Real

from .custom_types import Sz0, SzData, SzData2, SzN, SzN2

SzGamma: TypeAlias = Real[Array, "data-1"]
SzGamma2: TypeAlias = Real[Array, "data-1 2"]


def interpax_PPoly_from_scipy_UnivariateSpline(
    scipy_spl: scipy.interpolate.UnivariateSpline, /
) -> interpax.PPoly:
    """Convert a `scipy.interpolate.UnivariateSpline` to an `interpax.PPoly`."""
    # scipy UnivariateSpline -> scipy PPoly. `_eval_args` is specific to some of
    # the scipy splines, so this doesn't scale to all scipy splines :(.
    scipy_ppoly = scipy.interpolate.PPoly.from_spline(scipy_spl._eval_args)
    # Construct the interpax PPoly from the scipy one.
    return interpax.PPoly(c=scipy_ppoly.c, x=scipy_ppoly.x)


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


# ============================================================================


@runtime_checkable
class ReduceFn(Protocol):
    """Protocol for a function that reduces an array along an axis.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> isinstance(jnp.median, ReduceFn)
    True

    """

    def __call__(self, arr: Real[Array, "chunk"], /, axis: int) -> Sz0: ...


@ft.partial(jax.jit, static_argnames=("num_splits", "reduce_fn"))
def reduce_point_density(
    gamma: SzN,
    data: SzN2,
    *,
    num_splits: int,
    reduce_fn: ReduceFn = jnp.median,
) -> tuple[
    Real[Array, "{num_splits + 2}"],  # gamma
    Real[Array, "{num_splits + 2} 2"],  # data
]:
    """Split and reduce gamma, data into `num_split` blocks, keeping ends.

    A dataset representing the points along a stream's track can have
    problematic small changes in curvature. If we reduce the number of points
    that represents the curve then it necessarily forces a greater degree of
    smoothness. Combining this with `optimize_spline_knots` can produce a spline
    curve that better represents the smooth stream track.

    Parameters
    ----------
    gamma
        The gamma values at which the spline is anchored.
    data
        The data points of the spline.

    num_splits
        The number of splits to make in the data. The spline will be reduced to
        `num_splits + 2` points.
    reduce_fn
        The function to use to reduce the data within each chunk to a single
        point. Defaults to `jnp.median`.

    Examples
    --------
    >>> import jax.numpy as jnp

    >>> gamma = jnp.array([-1, 0, 0.5, 1])
    >>> data = jnp.array([[0, 0], [1, 0], [1, 2], [0, 2]])

    >>> gamma2, data2 = reduce_point_density(gamma, data, num_splits=1)
    >>> gamma2
    Array([-1.  ,  0.25,  1.  ], dtype=float64)
    >>> data2
    Array([[0. , 0. ],
           [0.5, 1. ],
           [0. , 2. ]], dtype=float64)

    """
    # Split and reduce gamma
    gamma_split = jnp.array_split(gamma, num_splits)
    gamma_median = jnp.array(
        [gamma[0]] + [reduce_fn(chunk, axis=0) for chunk in gamma_split] + [gamma[-1]]
    )

    # Split and reduce the data
    data_split = jnp.array_split(data, num_splits)
    data_median = jnp.stack(
        [data[0]] + [reduce_fn(chunk, axis=0) for chunk in data_split] + [data[-1]]
    )

    return gamma_median, data_median


# ---------------------------------------------------------


@ft.partial(jax.jit)
def data_distance_cost_fn(
    knots_y: SzN2,
    knots_gamma: SzN,
    data_gamma: SzData,
    data_y: SzData,
    *,
    sigmas: SzN | float = 1.0,
) -> Sz0:
    """Cost function to minimize that compares data to spline fit.

    $$ \text{cost} = \sum_i \left( \frac{y_i - f(\gamma_i)}{\sigma_i} \right)^2 $$

    where $$y_i$ is the target data, $f(\gamma_i)$ is the spline evaluated at
    $\gamma_i$, and $\sigma_i$ is the uncertainty on $y_i$.

    Parameters
    ----------
    knots_y


    """
    # Compute the cost of the distance from the spline from the data
    spl = interpax.Interpolator1D(knots_gamma, knots_y, method="cubic2")
    data_cost = jnp.sum(((data_y - spl(data_gamma)) / sigmas) ** 2)
    return data_cost


# -------------------------------------


@ft.partial(jax.jit)
def curvature_cost_fn(
    params: SzN2,
    gamma_knots: SzN,
    data_gamma: SzData,
    data_target: SzData,  # noqa: ARG001
) -> Sz0:
    """Cost function to penalize changes in curvature."""
    spline = interpax.Interpolator1D(gamma_knots, params, method="cubic2")
    d2x_d2gamma = jax.vmap(jax.jacfwd(jax.jacfwd(spline)))(data_gamma)
    L = 100.0  # A large constant to make tanh approximate the sign function
    sign_approx = jnp.tanh(L * d2x_d2gamma)
    lambda_sign = 10  # TODO: allow setting this
    sign_flip_cost = lambda_sign * jnp.sum((sign_approx[1:] - sign_approx[:-1]) ** 2)
    return sign_flip_cost


# -------------------------------------


@ft.partial(jax.jit)
def _no_curvature_cost_fn(
    params: SzN2,  # noqa: ARG001
    gamma_knots: SzN,  # noqa: ARG001
    data_gamma: SzData,  # noqa: ARG001
    data_target: SzData,  # noqa: ARG001
) -> Sz0:
    """Return 0.0."""
    return jnp.zeros(())


@ft.partial(jax.jit, static_argnames=("penalize_concavity_changes",))
def default_cost_fn(
    params: SzN2,
    gamma_knots: SzN,
    data_gamma: SzData,
    data_target: SzData,
    *,
    sigmas: float = 1.0,
    penalize_concavity_changes: bool = False,
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
    data_cost = data_distance_cost_fn(
        params, gamma_knots, data_gamma, data_target, sigmas=sigmas
    )

    # Optionally add a penalization for changes in concavity
    curvature_cost = jax.lax.cond(
        penalize_concavity_changes,
        curvature_cost_fn,
        _no_curvature_cost_fn,
        params,
        gamma_knots,
        data_gamma,
        data_target,
    )

    return data_cost + curvature_cost


DEFAULT_OPTIMIZER = optax.adam(learning_rate=1e-3)
StepState: TypeAlias = tuple[dict[str, Any], optax.OptState]


@ft.partial(jax.jit, static_argnums=(0,), static_argnames=("optimizer", "nsteps"))
def optimize_spline_knots(
    cost_fn: Callable[..., Sz0],  # TODO: full type hint
    /,
    init_params: SzN2,
    gamma_knots: SzN,
    data_gamma: SzData,
    data_target: SzData2,
    *,
    sigmas: float = 1.0,
    optimizer: optax.GradientTransformation = DEFAULT_OPTIMIZER,
    nsteps: int = 10_000,
) -> SzN2:
    """Optimize spline knots to fit data.

    .. warning::

        If you use this function to change the locations of the knots then this
        changes the arc-length of the spline. This can be problematic if gamma
        is the normalized arc-length of the data. If you change the knots then
        you should also change gamma accordingly. The easiest way to do this is
        to:

        1. evaluate the optimized spline on a dense array of old gamma values
        2. call `make_gamma_from_data` on the new data to define a new gamma,
        3. create a new spline with the new gamma. This spline will have the
           same shape as the optimized spline but with the new gamma values.

    Parameters
    ----------
    cost_fn
        The cost function.
    init_params
        starting outputs of splines at gamma_knots.
    gamma_knots
        anchor points for spline. median gamma in chunk.

    data_gamma
        gamma of every point in data.
    data_target
        x,y of every data point
    sigmas
        uncertainty on each datum.

    optimizer
        The optimizer to use. Defaults to Adam with a learning rate of 1e-3.
    nsteps
        The number of optimization steps to take. Defaults to 10_000.

    """

    @ft.partial(jax.jit)
    def loss_fn(params: SzN2) -> Sz0:
        return cost_fn(params, gamma_knots, data_gamma, data_target, sigmas=sigmas)

    # Choose an optimizer: Adam or SGD.
    opt_state = optimizer.init(init_params)

    # Define a single optimization step.
    @ft.partial(jax.jit)
    def step_fn(state: StepState, _: Any) -> tuple[StepState, Sz0]:
        """Perform a single optimization step."""
        params, opt_state = state
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    # Run the optimization using jax.lax.scan.
    (final_params, _), _ = jax.lax.scan(
        step_fn, (init_params, opt_state), None, length=nsteps
    )
    return final_params
