"""
Copyright (c) 2025 Sirui. All rights reserved.

streamcurvature: Constrain gravitational potential with stream curvature
"""

from __future__ import annotations

__all__ = [  # noqa: RUF022
    "__version__",
    "get_acceleration",
    "compute_likelihood",
    "compute_tangent",
    "compute_unit_tangent",
    "get_angles",
    "compute_unit_curvature",
    "optimize_spline_knots",
    "split_data",
    "make_monotonic_gamma_and_data",
    # Modules
    "utils",
    "plot",
]


from . import plot, utils
from ._version import version as __version__
from .accelerations import get_acceleration
from .likelihood import (
    compute_likelihood,
    compute_tangent,
    compute_unit_curvature,
    compute_unit_tangent,
)
from .plot import get_angles
from .spline_tools import (
    make_monotonic_gamma_and_data,
    optimize_spline_knots,
    split_data,
)
