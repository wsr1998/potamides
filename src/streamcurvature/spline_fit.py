"""Fit smooth spline."""

__all__ = ["optimize_spline_knots"]

from functools import partial

import interpax
import jax
import jax.numpy as jnp
import optax
from jaxopt import OptaxSolver

from .custom_types import Sz0, SzData, SzN

# def split_up_data(gamma: SzData, target: Real[Array, "batch data"]) -> tuple[SzData, SzData]:


def cost(
    params: SzN,
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
    init_params: SzN,
    gamma_knots: SzN,
    data_gamma: SzData,
    data_target: SzData,
    sigmas: float = 1.0,
) -> SzN:
    """Optimize spline knots to fit data.

    Parameters
    ----------
    init_params: starting outputs of splines at gamma_knots
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
