"""Spline-related tools."""

__all__ = [
    "CostFn",
    "concavity_change_cost_fn",
    "data_distance_cost_fn",
    "default_cost_fn",
    "new_gamma_knots_from_spline",
    "optimize_spline_knots",
    "reduce_point_density",
]

import functools as ft
from collections.abc import Callable
from typing import Any, Protocol, TypeAlias, runtime_checkable

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.integrate import trapezoid
from jaxtyping import Array, Real
from xmmutablemap import ImmutableMap

from potamides._src.custom_types import Sz0, SzData, SzN, SzN2

from .funcs import curvature, speed, unit_tangent

SzK2: TypeAlias = Real[Array, "K 2"]

speed_vec_fn = jax.vmap(speed, in_axes=(None, 0))


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


@runtime_checkable
class CostFn(Protocol):
    """Protocol for a cost function.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> isinstance(jnp.sum, CostFn)
    True

    """

    def __call__(
        self,
        knots: SzN2,
        gamma: SzN,
        /,
        *cost_args: Any,
        **kwargs: Any,
    ) -> Sz0: ...


@ft.partial(jax.jit)
def data_distance_cost_fn(
    knots: SzN2,
    gamma: SzN,
    /,
    data_gamma: SzData,
    data_y: SzData,
    *,
    sigmas: SzN | float = 1.0,
) -> Sz0:
    r"""Cost function to minimize that compares data to spline fit.

    $$ \text{cost} = \sum_i \left( \frac{y_i - f(\gamma_i)}{\sigma_i} \right)^2
    $$

    where $$y_i$ is the target data, $f(\gamma_i)$ is the spline evaluated at
    $\gamma_i$, and $\sigma_i$ is the uncertainty on $y_i$.

    Parameters
    ----------
    knots
        Output values of spline at gamma -- e.g. x, y values.
    gamma
        The gamma values at which the spline is anchored. There are N of these,
        one per `knots`. These are fixed while the `knots` are optimized.

    data_gamma
        gamma of the target data.
    data_y
        The target data. This is the data that the spline is trying to fit.

    sigmas
        The uncertainty on each datum in `data_y`.

    """
    # Compute the cost of the distance from the spline from the data
    spl = interpax.Interpolator1D(gamma, knots, method="cubic2")
    data_cost = jnp.sum(((data_y - spl(data_gamma)) / sigmas) ** 2)
    return data_cost / data_gamma.shape[0]


# -------------------------------------


@ft.partial(jax.jit)
def signed_kappa_scalar(spline: interpax.Interpolator1D, g: Sz0) -> Sz0:
    """Signed curvature at a point on the spline.

    The signed curvature is a concept which only makes sense for 2D curves. It
    is the dot product of the curvature vector with the 90° rotated unit tangent
    (left-handed normal).

    """
    t = unit_tangent(spline, g)
    n = jnp.array([-t[1], t[0]])  # left-handed normal
    k = curvature(spline, g)
    return jnp.dot(k, n)


# TODO: speed up a lot
@ft.partial(jax.jit)
def concavity_change_cost_fn(
    knots: SzN2,
    gamma: SzN,
    /,
    data_gamma: SzData,
    scale: float = 1e2,
    num_points: int = 1_000,
) -> Sz0:
    r"""Cost function to penalize changes in signed curvature for 2D curves.

    The integrand of the cost function is the derivative of the arctangent of
    the signed curvature multiplied by a large number $\lambda$.

    $$ \left( \frac{d}{ds} \atan\left(\lambda \kappa_{\text{signed}}(s)\right)
    \right)^2 $$

    where $\kappa_{\text{signed}}(s)$ is the signed curvature at $s$ and
    $\lambda$ is a large number that controls the width of the smoothing. The
    $\atan$ function differentiably mimics the undifferetiable $\text{sign}$
    function. The cost is the integral over the arc-length.

    Parameters
    ----------
    knots
        Output values of spline at gamma -- e.g. x, y values.
    gamma
        The gamma values at which the spline is anchored. There are N of these,
        one per `knots`. These are fixed while the `knots` are optimized. These
        should be from the same distribution as `data_gamma`.
    data_gamma
        gamma of the target data.

    scale
        Inverse Smoothing width.
    num_points
        The number of points to use to compute the cost function. This should be
        large enough to capture the curvature of the spline. Default is 1_000.
    """
    spline = interpax.Interpolator1D(gamma, knots, method="cubic2")

    # Compute the range and step size of the gamma values for integration
    gamma0, gamma1 = data_gamma.min(), data_gamma.max()
    gammas = jnp.linspace(gamma0, gamma1, num=num_points)
    dgamma = (gamma1 - gamma0) / (num_points - 1)

    # Smooth sign indicator
    def smooth_sign_fn(g: Sz0) -> Sz0:
        return jnp.arctan(signed_kappa_scalar(spline, g) * scale)

    # Compute arclength derivative of the smooth sign indicator
    ds_dgamma = speed_vec_fn(spline, gammas)
    d_smooth_sign_dgamma = jax.vmap(jax.grad(smooth_sign_fn))(gammas)
    d_smooth_sign_ds = d_smooth_sign_dgamma / ds_dgamma

    # Cost: penalize sign flips
    return trapezoid(d_smooth_sign_ds**2, dx=dgamma)


# -------------------------------------


@ft.partial(jax.jit)
def _no_concavity_change_cost_fn(*_: Any) -> Sz0:
    """Return 0.0."""
    return jnp.zeros(())


@ft.partial(jax.jit)
def default_cost_fn(
    knots: SzN2,
    gamma: SzN,
    data_gamma: SzData,
    data_y: SzData,
    /,
    *,
    sigmas: float = 1.0,
    data_weight: float = 1e3,  # enlarge gradients for GD.
    concavity_weight: float = 0.0,
    concavity_scale: float = 1e2,
) -> Sz0:
    """Cost function to minimize that compares data to spline fit.

    Parameters
    ----------
    knots
        Output values of spline at gamma -- e.g. x or y values. This is the
        parameter to be optimized to minimize the cost function.
    gamma:
        The gamma values at which the spline is anchored. There are N of these,
        one per `knots`. These are fixed while the `knots` are optimized.
        These should be from the same distribution as `data_gamma`.

    data_gamma:
        gamma of the target data.
    data_y:
        The target data. This is the data that the spline is trying to fit.

    sigmas:
        The uncertainty on each datum in `data_y`.
    concavity_weight
        The weight of the curvature penalization term. This should be tuned
        based on the desired smoothness of the spline. Default is `0.0`.
    concavity_scale
        The scale of the smoothing for the curvature penalization term. This
        should be tuned based on the desired smoothness of the spline. Default
        is `1e2`.

    """
    data_cost = data_distance_cost_fn(knots, gamma, data_gamma, data_y, sigmas=sigmas)

    # Optionally add a penalization for changes in concavity
    delta_concavity_cost = jax.lax.cond(
        concavity_weight > 0,
        concavity_change_cost_fn,
        _no_concavity_change_cost_fn,
        knots,
        gamma,
        data_gamma,
        concavity_scale,
    )

    return data_weight * data_cost + concavity_weight * delta_concavity_cost


DEFAULT_OPTIMIZER = optax.adam(learning_rate=1e-3)
StepState: TypeAlias = tuple[dict[str, Any], optax.OptState]


def _free_and_fixed_params(
    init_knots: Array,
    fixed_mask: tuple[bool, ...] | None,
) -> tuple[Array, Callable[[Array], Array]]:
    """Split flat parameters into free subset and provide a reconstructor."""
    if fixed_mask is None:
        free_knots_init = init_knots

        def reconstruct_knots(free_knots: SzK2) -> SzK2:
            return free_knots

    else:
        fixed_mask_ = np.array(fixed_mask, dtype=bool)
        (fixed_idx,) = np.nonzero(fixed_mask_)
        (free_idx,) = np.nonzero(~fixed_mask_)

        fixed_knots = init_knots[fixed_idx]
        free_knots_init = init_knots[free_idx]

        def reconstruct_knots(free_knots: SzK2) -> SzK2:
            out = jnp.empty_like(init_knots)
            out = out.at[fixed_idx].set(fixed_knots)
            out = out.at[free_idx].set(free_knots)
            return out

    return free_knots_init, reconstruct_knots


@ft.partial(
    jax.jit,
    static_argnums=(0,),
    static_argnames=("cost_kwargs", "optimizer", "nsteps", "fixed_mask"),
)
def optimize_spline_knots(
    cost_fn: CostFn,
    /,
    init_knots: SzN2,
    init_gamma: SzN,
    cost_args: tuple[Any, ...],
    *,
    cost_kwargs: ImmutableMap[str, Any] | tuple[tuple[str, Any], ...] | None = None,
    optimizer: optax.GradientTransformation = DEFAULT_OPTIMIZER,
    nsteps: int = 10_000,
    fixed_mask: tuple[bool, ...] | None = None,
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

    cost_args
        Additional positional arguments to pass to the cost function. For
        example, `cost_fn` can be `default_cost_fn` which takes `data_gamma` and
        `data_y` as arguments.
    cost_kwargs
        Additional keyword arguments to pass to the cost function. E.g.
        `data_distance_cost_fn` can take 'sigmas' and `concavity_change_cost_fn`
        can take `concavity_weight` and `concavity_scale`. `default_cost_fn` can
        take both. JAX treats these as static.
    fixed_mask
        A mask that indicates which knots are fixed. If `None` then all knots
        are free to be optimized. If a mask is provided then the knots at the
        indices where the mask is `True` are fixed and the knots at the indices
        where the mask is `False` are free to be optimized. The fixed knots are
        not optimized and are not included in the cost function. This is useful
        if you want to optimize only a subset of the knots while keeping the
        others fixed. The mask should be the same length as `init_knots`. The
        fixed knots are reconstructed in the final output.

    optimizer
        The optimizer to use. Defaults to Adam with a learning rate of 1e-3.
    nsteps
        The number of optimization steps to take. Defaults to 10_000.

    """
    cost_kw = ImmutableMap({} if cost_kwargs is None else cost_kwargs)

    # Determine fixed/free indices using fixed_mask argument
    free_knots_init, reconstruct_knots = _free_and_fixed_params(
        init_knots, fixed_mask=fixed_mask
    )

    # The loss function is the cost function, reconstructed with the free knots.
    @ft.partial(jax.jit)
    def loss_fn(params: SzN2) -> Sz0:
        knots = reconstruct_knots(params)
        return cost_fn(knots, init_gamma, *cost_args, **cost_kw)

    # Choose an optimizer: e.g. Adam or SGD.
    opt_state = optimizer.init(free_knots_init)

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
        step_fn, (free_knots_init, opt_state), None, length=nsteps
    )
    # Reconstruct the full set of knots (with fixed if needed)
    return reconstruct_knots(final_params)


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
    vs = speed_vec_fn(spline, gamma_old)
    s = jnp.cumsum(vs)
    gamma_new = 2 * s / s[-1] - 1  # new gamma in [-1, 1]

    # subselect gamma down to nknots
    sel = jnp.linspace(0, len(gamma_new), nknots, dtype=int)
    gamma_new = gamma_new[sel]
    points_new = spline(gamma_old[sel])

    return gamma_new, points_new
