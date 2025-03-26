"""
Copyright (c) 2025 EGGS collaboration. All rights reserved.

streamcurvature: Constrain gravitational potential with stream curvature
"""

__all__ = [
    "AbstractTrack",
    "Track",
    "__version__",
    "compute_accelerations",
    "compute_darclength_dgamma",
    "compute_ln_likelihood",
    "get_angles",
    "plot",
    "spline_tools",
]


from . import spline_tools
from ._src import plot
from ._src.accelerations import compute_accelerations
from ._src.core import AbstractTrack, Track, compute_darclength_dgamma
from ._src.likelihood import compute_ln_likelihood
from ._src.plot import get_angles
from ._version import version as __version__
