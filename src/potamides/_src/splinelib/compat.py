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
    """Convert a `scipy.interpolate.UnivariateSpline` to an `interpax.PPoly`.

    Notes
    -----
    This conversion relies on scipy's internal `_eval_args` attribute, which
    may not be available for all scipy spline types. The function works by:
    1. Converting the scipy UnivariateSpline to a scipy PPoly using `from_spline`
    2. Extracting coefficients and breakpoints from the scipy PPoly
    3. Constructing an equivalent interpax PPoly instance.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.interpolate import UnivariateSpline
    >>> import interpax

    >>> x = np.linspace(0, 2*np.pi, 10)
    >>> y = np.sin(x)

    >>> scipy_spline = UnivariateSpline(x, y, s=0)
    >>> interpax_ppoly = interpax_PPoly_from_scipy_UnivariateSpline(scipy_spline)

    >>> x_test = np.linspace(0, 2*np.pi, 50)
    >>> scipy_vals = scipy_spline(x_test)
    >>> interpax_vals = interpax_ppoly(x_test)
    >>> np.allclose(scipy_vals, interpax_vals)
    True

    """
    # scipy UnivariateSpline -> scipy PPoly. `_eval_args` is specific to some of
    # the scipy splines, so this doesn't scale to all scipy splines :(.
    scipy_ppoly = scipy.interpolate.PPoly.from_spline(scipy_spl._eval_args)
    # Construct the interpax PPoly from the scipy one.
    return interpax.PPoly(c=scipy_ppoly.c, x=scipy_ppoly.x)
