"""
Copyright (c) 2025 EGGS collaboration. All rights reserved.

streamcurvature: Constrain gravitational potential with stream curvature
"""

__all__ = [  # noqa: RUF022
    "__version__",
    "compute_accelerations",
    "compute_dThat_dgamma",
    "compute_likelihood",
    "compute_tangent",
    "compute_unit_tangent",
    "get_angles",
    "compute_unit_curvature",
    # Modules
    "plot",
    "spline_tools",
]


from . import spline_tools
from ._src import plot
from ._src.accelerations import compute_accelerations
from ._src.likelihood import (
    compute_likelihood,
    compute_dThat_dgamma,
    compute_tangent,
    compute_unit_curvature,
    compute_unit_tangent,
)
from ._src.plot import get_angles
from ._version import version as __version__
