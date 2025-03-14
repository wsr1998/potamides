"""
Copyright (c) 2025 Sirui. All rights reserved.

streamcurvature: Constrain gravitational potential with stream curvature
"""

from __future__ import annotations

__all__ = [  # noqa: RUF022
    "__version__",
    "get_acceleration",
    "get_likelihood",
    "get_angles",
    "optimize_spline_knots",
    "get_unit_tangents_and_curvature",
    "split_data",
    "make_monotonic_gamma_and_data",
    # Modules
    "utils",
    "plot",
]


from . import plot, utils
from ._version import version as __version__
from .accelerations import get_acceleration
from .likelihood import get_angles, get_likelihood, get_unit_tangents_and_curvature
from .spline_tools import (
    make_monotonic_gamma_and_data,
    optimize_spline_knots,
    split_data,
)
