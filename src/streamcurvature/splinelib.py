"""Fit smooth spline."""

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

from ._src.splinelib import (
    curvature_cost_fn,
    data_distance_cost_fn,
    default_cost_fn,
    interpax_PPoly_from_scipy_UnivariateSpline,
    make_gamma_from_data,
    make_increasing_gamma_from_data,
    optimize_spline_knots,
    point_to_point_arclenth,
    point_to_point_distance,
    reduce_point_density,
)
