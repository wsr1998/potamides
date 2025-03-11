"""Utilities."""

__all__ = ["interpax_ppoly_from_scipy_spline"]

import scipy.interpolate
from interpax import PPoly


def interpax_ppoly_from_scipy_spline(
    scipy_spl: scipy.interpolate.UnivariateSpline, /
) -> PPoly:
    scipy_ppoly = scipy.interpolate.PPoly.from_spline(scipy_spl._eval_args)
    return PPoly(c=scipy_ppoly.c, x=scipy_ppoly.x)
