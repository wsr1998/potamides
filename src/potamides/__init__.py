"""
Copyright (c) 2025 EGGS collaboration. All rights reserved.

potamides: Constrain gravitational potential with stream curvature
"""

__all__ = [
    "AbstractTrack",
    "Track",
    "__version__",
    "combine_ln_likelihoods",
    "compute_accelerations",
    "compute_ln_likelihood",
    "get_angles",
    "plot",
    "splinelib",
]


from . import splinelib
from ._src import plot
from ._src.accelerations import compute_accelerations
from ._src.core import AbstractTrack, Track
from ._src.likelihood import combine_ln_likelihoods, compute_ln_likelihood
from ._src.plot import get_angles
from ._version import version as __version__
