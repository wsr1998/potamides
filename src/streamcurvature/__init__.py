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
    # Modules
    "utils",
]


from . import utils
from ._version import version as __version__
from .accelerations import get_acceleration
from .likelihood import get_angles, get_likelihood
