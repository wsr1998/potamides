"""Spline-related tools."""

__all__ = [
    "interpax_PPoly_from_scipy_UnivariateSpline",
]

from typing import TypeAlias

import interpax
import scipy.interpolate
from jaxtyping import Array, Real

SzGamma: TypeAlias = Real[Array, "data-1"]
SzGamma2: TypeAlias = Real[Array, "data-1 2"]


def interpax_PPoly_from_scipy_UnivariateSpline(
    scipy_spl: scipy.interpolate.UnivariateSpline, /
) -> interpax.PPoly:
    """Convert a `scipy.interpolate.UnivariateSpline` to an `interpax.PPoly`."""
    # scipy UnivariateSpline -> scipy PPoly. `_eval_args` is specific to some of
    # the scipy splines, so this doesn't scale to all scipy splines :(.
    scipy_ppoly = scipy.interpolate.PPoly.from_spline(scipy_spl._eval_args)
    # Construct the interpax PPoly from the scipy one.
    return interpax.PPoly(c=scipy_ppoly.c, x=scipy_ppoly.x)
