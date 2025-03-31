"""Spline-related tools."""

__all__ = [  # noqa: RUF022
    "reduce_point_density",
    "data_distance_cost_fn",
    "curvature_cost_fn",
    "default_cost_fn",
    "optimize_spline_knots",
    "new_gamma_knots_from_spline",
]

import functools as ft
from collections.abc import Callable
from typing import Any, Protocol, TypeAlias, runtime_checkable

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Real

from streamcurvature._src.custom_types import Sz0, SzData, SzData2, SzN, SzN2

from .funcs import speed


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
    init_gamma: SzN,
    data_gamma: SzData,
    data_target: SzData,  # noqa: ARG001
) -> Sz0:
    """Cost function to penalize changes in curvature."""
    spline = interpax.Interpolator1D(init_gamma, params, method="cubic2")
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
    init_gamma: SzN,  # noqa: ARG001
    data_gamma: SzData,  # noqa: ARG001
    data_target: SzData,  # noqa: ARG001
) -> Sz0:
    """Return 0.0."""
    return jnp.zeros(())


@ft.partial(jax.jit, static_argnames=("penalize_concavity_changes",))
def default_cost_fn(
    params: SzN2,
    init_gamma: SzN,
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
        Output values of spline at init_gamma -- e.g. x or y values.
        This is the parameter to be optimized to minimize the cost function.
    init_gamma:
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
        params, init_gamma, data_gamma, data_target, sigmas=sigmas
    )

    # Optionally add a penalization for changes in concavity
    curvature_cost = jax.lax.cond(
        penalize_concavity_changes,
        curvature_cost_fn,
        _no_curvature_cost_fn,
        params,
        init_gamma,
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
    init_knots: SzN2,
    init_gamma: SzN,
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
    init_knots
        starting outputs of splines at init_gamma.
    init_gamma
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
        return cost_fn(params, init_gamma, data_gamma, data_target, sigmas=sigmas)

    # Choose an optimizer: Adam or SGD.
    opt_state = optimizer.init(init_knots)

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
        step_fn, (init_knots, opt_state), None, length=nsteps
    )
    return final_params


@ft.partial(jax.jit, static_argnames=("nknots",))
def new_gamma_knots_from_spline(
    spline: interpax.Interpolator1D, /, *, nknots: int
) -> tuple[Real[Array, "{nknots}"], Real[Array, "{nknots} 2"]]:
    """Define new gamma (and knots) from an existing spline.

    When the knots of a spline are changed the arc-length of the spline changes
    as well. It is often useful to define a new gamma that is the normalized
    arc-length of the spline. This function takes a spline and returns a new
    gamma (and corresponding knots) that is the normalized arc-length of the
    spline so that a new spline can be created with the new gamma (and knots).

    Parameters
    ----------
    spline
        The spline to use to define the new gamma.

    nknots
        The number of knots to use in the new spline.

    Returns
    -------
    gamma_new
        The new gamma values. One is at -1 and one is at 1. The rest are
        evenly spaced in between.
    points_new
        The new points of the spline at the new gamma values.

    """
    # Validate nknots
    nknots = eqx.error_if(
        nknots, nknots < 2 or nknots > 1_000, "nknots must be in [2, 1_000]"
    )

    # Use the quadratic approximation of the spline to get the arc-length
    gamma_old = jnp.linspace(
        spline.x.min(), spline.x.max(), int(1e5), dtype=float
    )  # old gammas
    vs = jax.vmap(speed, in_axes=(None, 0))(spline, gamma_old)
    s = jnp.cumsum(vs)
    gamma_new = 2 * s / s[-1] - 1  # new gamma in [-1, 1]

    # subselect gamma down to nknots
    sel = jnp.linspace(0, len(gamma_new), nknots, dtype=int)
    gamma_new = gamma_new[sel]
    points_new = spline(gamma_old[sel])

    return gamma_new, points_new
