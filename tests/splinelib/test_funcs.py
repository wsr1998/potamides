"""Test the package itself."""

import pathlib
from typing import TypeAlias

import interpax
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Real

import streamcurvature.splinelib as splib

SzGamma: TypeAlias = Real[Array, "gamma"]
SzGamma2: TypeAlias = Real[Array, "gamma 2"]

# pytest.mark.array_compare generates errors b/c of an old-style hookwrapper
# teardown.
pytestmark = pytest.mark.filterwarnings("ignore::pluggy.PluggyTeardownRaisedWarning")

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()


@pytest.fixture(scope="module")
def spline() -> interpax.Interpolator1D:
    r"""Fixture to create a spline for testing."""
    with jnp.load(CURRENT_DIR.parent / "data" / "example_spline.npz") as f:
        gamma = f["x"]  # (gamma,)
        xy = f["f"]  # (gamma, 2)

    return interpax.Interpolator1D(gamma, xy, method="cubic2")


@pytest.fixture(scope="module")
def gamma() -> SzGamma:
    return jnp.linspace(-0.95, 0.95, 128)


##############################################################################
# Consistency tests
#
# There is NOT tests of correctness, but rather of consistency -- ensuring that
# the output of the tested functions do not change.


@pytest.mark.array_compare
def test_position_consistency(
    spline: interpax.Interpolator1D, gamma: SzGamma
) -> SzGamma2:
    r"""Test that the spline position function is consistent."""
    out = splib.position(spline, gamma)
    assert out.shape == (len(gamma), 2)
    return out


@pytest.mark.array_compare
def test_tangent_consistency(
    spline: interpax.Interpolator1D, gamma: SzGamma
) -> SzGamma2:
    r"""Test that the spline position function is consistent."""
    out = jax.vmap(splib.tangent, (None, 0))(spline, gamma)
    assert out.shape == (len(gamma), 2)
    return out


@pytest.mark.array_compare
def test_unit_tangent_consistency(
    spline: interpax.Interpolator1D, gamma: SzGamma
) -> SzGamma2:
    r"""Test that the unit tangent function is consistent."""
    out = jax.vmap(splib.unit_tangent, (None, 0))(spline, gamma)
    assert out.shape == (len(gamma), 2)
    return out


@pytest.mark.array_compare
def test_speed_consistency(spline: interpax.Interpolator1D, gamma: SzGamma) -> SzGamma:
    r"""Test that the speed function is consistent."""
    out = jax.vmap(splib.speed, (None, 0))(spline, gamma)
    assert out.shape == gamma.shape
    return out


@pytest.mark.array_compare
def test_arc_length_p2p_consistency(
    spline: interpax.Interpolator1D, gamma: SzGamma
) -> SzGamma:
    r"""Test that the arc length of the spline is consistent."""
    out = jax.vmap(splib.arc_length_p2p, (None, 0))(spline, gamma)
    assert out.shape == gamma.shape
    return out


@pytest.mark.array_compare
def test_arc_length_quadtrature_consistency(
    spline: interpax.Interpolator1D, gamma: SzGamma
) -> SzGamma:
    r"""Test that the arc length from start to end of the spline is consistent."""
    out = jax.vmap(splib.arc_length_quadtrature, (None, 0))(spline, gamma)
    assert out.shape == gamma.shape
    return out


@pytest.mark.array_compare
def test_arc_length_odeint_consistency(
    spline: interpax.Interpolator1D, gamma: SzGamma
) -> SzGamma:
    r"""Test that the arc length from start to end of the spline is consistent."""
    out = jax.vmap(splib.arc_length_odeint, (None, 0))(spline, gamma)
    assert out.shape == gamma.shape
    return out


@pytest.mark.array_compare
def test_arc_length_consistency(
    spline: interpax.Interpolator1D, gamma: SzGamma
) -> SzGamma:
    r"""Test that the arc length from start to end of the spline is consistent."""
    out = jax.vmap(splib.arc_length, (None, 0))(spline, gamma)
    assert out.shape == gamma.shape
    return out


@pytest.mark.array_compare
def test_dThat_dgamma_consistency(
    spline: interpax.Interpolator1D, gamma: SzGamma
) -> SzGamma2:
    r"""Test that the unit tangent function is consistent."""
    out = jax.vmap(splib.dThat_dgamma, (None, 0))(spline, gamma)
    assert out.shape == (len(gamma), 2)
    return out


@pytest.mark.array_compare
def test_curvature_consistency(
    spline: interpax.Interpolator1D, gamma: SzGamma
) -> SzGamma2:
    r"""Test that the unit tangent function is consistent."""
    out = jax.vmap(splib.curvature, (None, 0))(spline, gamma)
    assert out.shape == (len(gamma), 2)
    return out


@pytest.mark.array_compare
def test_unit_curvature_consistency(
    spline: interpax.Interpolator1D, gamma: SzGamma
) -> SzGamma2:
    r"""Test that the unit tangent function is consistent."""
    out = jax.vmap(splib.unit_curvature, (None, 0))(spline, gamma)
    assert out.shape == (len(gamma), 2)
    return out
