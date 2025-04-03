"""Spline-related tools."""

__all__ = [  # noqa: RUF022
    # Processing data
    "make_gamma_from_data",
    "make_increasing_gamma_from_data",
    "point_to_point_arclenth",
    "point_to_point_distance",
    # Functions
    "position",
    "tangent",
    "unit_tangent",
    "speed",
    "arc_length_p2p",
    "arc_length_quadtrature",
    "arc_length_odeint",
    "arc_length",
    "dThat_dgamma",
    "curvature",
    "unit_curvature",
    # Optimizing splines
    "reduce_point_density",
    "CostFn",
    "data_distance_cost_fn",
    "curvature_cost_fn",
    "default_cost_fn",
    "optimize_spline_knots",
    "new_gamma_knots_from_spline",
    # Compatibility
    "interpax_PPoly_from_scipy_UnivariateSpline",
]

from .compat import interpax_PPoly_from_scipy_UnivariateSpline
from .data import (
    make_gamma_from_data,
    make_increasing_gamma_from_data,
    point_to_point_arclenth,
    point_to_point_distance,
)
from .funcs import (
    arc_length,
    arc_length_odeint,
    arc_length_p2p,
    arc_length_quadtrature,
    curvature,
    dThat_dgamma,
    position,
    speed,
    tangent,
    unit_curvature,
    unit_tangent,
)
from .opt import (
    CostFn,
    curvature_cost_fn,
    data_distance_cost_fn,
    default_cost_fn,
    new_gamma_knots_from_spline,
    optimize_spline_knots,
    reduce_point_density,
)
