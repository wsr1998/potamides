"""Test the compat module."""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.interpolate
from jaxtyping import Array, Real

import potamides.splinelib as splib

# pytest.mark.array_compare generates errors b/c of an old-style hookwrapper
# teardown.
pytestmark = pytest.mark.filterwarnings("ignore::pluggy.PluggyTeardownRaisedWarning")


@pytest.fixture(scope="module")
def scipy_spline() -> scipy.interpolate.UnivariateSpline:
    r"""Fixture to create a scipy UnivariateSpline for testing."""
    # Create some test data with deterministic noise
    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, 20)
    y = np.sin(x) + 0.1 * rng.random(len(x))
    return scipy.interpolate.UnivariateSpline(x, y, s=0)


##############################################################################
# Consistency tests
#
# These are NOT tests of correctness, but rather of consistency -- ensuring that
# the output of the tested functions do not change.


@pytest.mark.array_compare
def test_interpax_PPoly_from_scipy_UnivariateSpline_consistency(
    scipy_spline: scipy.interpolate.UnivariateSpline,
) -> Real[Array, "50"]:
    r"""Test that the interpax_PPoly_from_scipy_UnivariateSpline function is consistent."""
    # Convert scipy spline to interpax PPoly
    interpax_ppoly = splib.interpax_PPoly_from_scipy_UnivariateSpline(scipy_spline)

    # Evaluate at test points
    eval_points = jnp.linspace(0.5, 9.5, 50)
    out = interpax_ppoly(eval_points)

    assert out.shape == (len(eval_points),)
    return out


##############################################################################
# Correctness Tests


def test_interpax_PPoly_from_scipy_UnivariateSpline_correctness() -> None:
    """Test ``interpax_PPoly_from_scipy_UnivariateSpline`` function is correct."""
    x = np.linspace(0, 2 * np.pi, 10)
    y = np.sin(x)

    scipy_spline = scipy.interpolate.UnivariateSpline(x, y, s=0)
    interpax_ppoly = splib.interpax_PPoly_from_scipy_UnivariateSpline(scipy_spline)

    x_test = np.linspace(0, 2 * np.pi, 50)
    scipy_vals = scipy_spline(x_test)
    interpax_vals = interpax_ppoly(x_test)

    np.testing.assert_allclose(scipy_vals, interpax_vals, rtol=1e-5, atol=1e-8)
