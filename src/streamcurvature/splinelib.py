"""Fit smooth spline."""

__all__ = [  # noqa: RUF022
    # Processing data
    "make_gamma_from_data",
    "make_increasing_gamma_from_data",
    "point_to_point_arclenth",
    "point_to_point_distance",
    # Functions
    "position",
    "spherical_position",
    "tangent",
    "speed",
    "arc_length_p2p",
    "arc_length_quadtrature",
    "arc_length_odeint",
    "arc_length",
    "acceleration",
    "principle_unit_normal",
    "curvature",
    "kappa",
    # Optimizing splines
    "reduce_point_density",
    "CostFn",
    "data_distance_cost_fn",
    "concavity_change_cost_fn",
    "default_cost_fn",
    "optimize_spline_knots",
    "new_gamma_knots_from_spline",
    # Utils
    "interpax_PPoly_from_scipy_UnivariateSpline",
]

from ._src.splinelib import (
    CostFn,
    acceleration,
    arc_length,
    arc_length_odeint,
    arc_length_p2p,
    arc_length_quadtrature,
    concavity_change_cost_fn,
    curvature,
    data_distance_cost_fn,
    default_cost_fn,
    interpax_PPoly_from_scipy_UnivariateSpline,
    kappa,
    make_gamma_from_data,
    make_increasing_gamma_from_data,
    new_gamma_knots_from_spline,
    optimize_spline_knots,
    point_to_point_arclenth,
    point_to_point_distance,
    position,
    principle_unit_normal,
    reduce_point_density,
    speed,
    spherical_position,
    tangent,
)
