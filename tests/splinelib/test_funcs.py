"""Test the functions in the splinelib module."""

import pathlib

import interpax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import potamides.splinelib as splib
from potamides._src.custom_types import SzGamma, SzGamma2

# pytest.mark.array_compare generates errors b/c of an old-style hookwrapper
# teardown.
pytestmark = pytest.mark.filterwarnings("ignore::pluggy.PluggyTeardownRaisedWarning")

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()


@pytest.fixture(scope="module")
def spline() -> interpax.Interpolator1D:
    r"""Fixture to create a spline for testing."""
    with np.load(CURRENT_DIR.parent / "data" / "example_spline.npz") as f:
        gamma = f["x"]  # (gamma,)
        xy = f["f"]  # (gamma, 2)

    return interpax.Interpolator1D(gamma, xy, method="cubic2")


@pytest.fixture(scope="module")
def gamma() -> SzGamma:
    return jnp.linspace(-0.95, 0.95, 128)


@pytest.fixture(scope="module")
def spline_large() -> interpax.Interpolator1D:
    r"""Fixture to create a large spline for testing (unscaled)."""
    gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
    xy = jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
    return interpax.Interpolator1D(gamma, xy, method="cubic2")


@pytest.fixture
def scaled_spline(request) -> interpax.Interpolator1D:
    r"""Fixture to create a scaled spline for testing."""
    scale = getattr(request, "param", 1.0)
    gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
    xy = scale * jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=-1)
    return interpax.Interpolator1D(gamma, xy, method="cubic2")


@pytest.fixture(scope="module")
def gamma_test() -> SzGamma:
    return jnp.array([0, jnp.pi / 2, jnp.pi])


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
def test_spherical_position_consistency(
    spline: interpax.Interpolator1D, gamma: SzGamma
) -> SzGamma2:
    r"""Test that the spline spherical position function is consistent."""
    out = jax.vmap(splib.spherical_position, (None, 0))(spline, gamma)
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


@pytest.mark.array_compare(rtol=5e-6)
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
def test_acceleration_consistency(
    spline: interpax.Interpolator1D, gamma: SzGamma
) -> SzGamma2:
    r"""Test that the acceleration function is consistent."""
    out = jax.vmap(splib.acceleration, (None, 0))(spline, gamma)
    assert out.shape == (len(gamma), 2)
    return out


@pytest.mark.array_compare
def test_principle_unit_normal_consistency(
    spline: interpax.Interpolator1D, gamma: SzGamma
) -> SzGamma2:
    r"""Test that the principle unit normal function is consistent."""
    out = jax.vmap(splib.principle_unit_normal, (None, 0))(spline, gamma)
    assert out.shape == (len(gamma), 2)
    return out


@pytest.mark.array_compare
def test_curvature_consistency(
    spline: interpax.Interpolator1D, gamma: SzGamma
) -> SzGamma2:
    r"""Test that the curvature function is consistent."""
    out = jax.vmap(splib.curvature, (None, 0))(spline, gamma)
    assert out.shape == (len(gamma), 2)
    return out


@pytest.mark.array_compare
def test_kappa_consistency(spline: interpax.Interpolator1D, gamma: SzGamma) -> SzGamma2:
    r"""Test that the kappa function is consistent."""
    out = jax.vmap(splib.kappa, (None, 0))(spline, gamma)
    assert out.shape == (len(gamma),)
    return out


##############################################################################
# Correctness tests


@pytest.mark.parametrize("scaled_spline", [1.0], indirect=True)
def test_position_correctness(
    scaled_spline: interpax.Interpolator1D, gamma_test: SzGamma
) -> None:
    """Test `position` function for correctness."""
    got = jax.vmap(splib.position, (None, 0))(scaled_spline, gamma_test)
    exp = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert jnp.allclose(got, exp)


@pytest.mark.parametrize("scaled_spline", [2.0], indirect=True)
def test_spherical_position_correctness(
    scaled_spline: interpax.Interpolator1D, gamma_test: SzGamma
) -> None:
    """Test `spherical_position` for correctness."""
    got = jax.vmap(splib.spherical_position, (None, 0))(scaled_spline, gamma_test)

    exp = jnp.array([[2.0, 0.0], [2.0, 1.5708], [2.0, 3.1416]])
    assert jnp.allclose(got, exp)


@pytest.mark.parametrize("scaled_spline", [2.0], indirect=True)
def test_tangent_correctness(
    scaled_spline: interpax.Interpolator1D, gamma_test: SzGamma
) -> None:
    """Test `tangent` for correctness."""
    got = jax.vmap(splib.tangent, (None, 0))(scaled_spline, gamma_test)

    exp = 2 * jnp.array([[0, 1], [-1, 0], [0, -1]], dtype=float)
    assert jnp.allclose(got, exp)


# unit_tangent is private
# @pytest.mark.parametrize("scaled_spline", [1.0], indirect=True)
# def test_unit_tangent_correctness(scaled_spline, gamma_test) -> None:
#     """Test `unit_tangent` for correctness."""
#     got = jax.vmap(splib.unit_tangent, (None, 0))(scaled_spline, gamma_test)

#     exp = jnp.array([[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
#     assert jnp.allclose(got, exp)


@pytest.mark.parametrize("scaled_spline", [2.0], indirect=True)
def test_speed_correctness(
    scaled_spline: interpax.Interpolator1D, gamma_test: SzGamma
) -> None:
    """Test `speed` for correctness."""
    got = jax.vmap(splib.speed, (None, 0))(scaled_spline, gamma_test)

    exp = 2 * jnp.array([1.0, 1, 1])
    assert jnp.allclose(got, exp)


@pytest.mark.parametrize("scaled_spline", [2.0], indirect=True)
def test_arc_length_p2p_correctness(scaled_spline: interpax.Interpolator1D) -> None:
    """Test `arc_length_p2p` for correctness."""
    got = splib.arc_length_p2p(scaled_spline, 0, 2 * jnp.pi) / jnp.pi
    exp = 4
    assert jnp.allclose(got, exp)


@pytest.mark.parametrize("scaled_spline", [2.0], indirect=True)
def test_arc_length_quadtrature_correctness(
    scaled_spline: interpax.Interpolator1D,
) -> None:
    """Test `arc_length_quadrature` for correctness."""
    got = splib.arc_length_quadtrature(scaled_spline, 0, 2 * jnp.pi) / jnp.pi
    exp = 4
    assert jnp.allclose(got, exp)


@pytest.mark.parametrize("scaled_spline", [2.0], indirect=True)
def test_arc_length_odeint_correctness(
    scaled_spline: interpax.Interpolator1D,
) -> None:
    """Test `arc_length_odeint` for correctness."""
    got = splib.arc_length_odeint(scaled_spline, 0, 2 * jnp.pi) / jnp.pi
    exp = 4
    assert jnp.allclose(got, exp)


@pytest.mark.parametrize("scaled_spline", [2.0], indirect=True)
@pytest.mark.parametrize("method", ["p2p", "quad", "ode"])
def test_arc_length_correctness(
    scaled_spline: interpax.Interpolator1D, method: str
) -> None:
    """Test `arc_length` for correctness."""
    got = splib.arc_length(scaled_spline, 0, 2 * jnp.pi, method=method)
    exp = 4 * jnp.pi
    assert jnp.allclose(got, exp)


@pytest.mark.parametrize("scaled_spline", [2.0], indirect=True)
def test_acceleration_correctness(
    scaled_spline: interpax.Interpolator1D, gamma_test: SzGamma
) -> None:
    """Test `acceleration` for correctness."""
    got = jax.vmap(splib.acceleration, in_axes=(None, 0))(scaled_spline, gamma_test)
    exp = jnp.array([[-2.0, 0.0], [0.0, -2.0], [2.0, 0.0]])
    assert jnp.allclose(got, exp)


@pytest.mark.parametrize("scaled_spline", [2.0], indirect=True)
def test_principle_unit_normal_correctness(
    scaled_spline: interpax.Interpolator1D, gamma_test: SzGamma
) -> None:
    """Test `principle_unit_normal` for correctness."""
    got = jax.vmap(splib.principle_unit_normal, in_axes=(None, 0))(
        scaled_spline, gamma_test
    )
    exp = jnp.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]])
    assert jnp.allclose(got, exp)


@pytest.mark.parametrize("scaled_spline", [2.0], indirect=True)
def test_curvature_correctness(
    scaled_spline: interpax.Interpolator1D, gamma_test: SzGamma
) -> None:
    """Test `curvature` for correctness."""
    got = jax.vmap(splib.curvature, in_axes=(None, 0))(scaled_spline, gamma_test)
    exp = jnp.array([[-0.5, 0.0], [0.0, -0.5], [0.5, 0.0]])
    assert jnp.allclose(got, exp)


@pytest.mark.parametrize("scaled_spline", [2.0], indirect=True)
def test_kappa_correctness(
    scaled_spline: interpax.Interpolator1D, gamma_test: SzGamma
) -> None:
    """Test `kappa` for correctness."""
    got = jax.vmap(splib.kappa, in_axes=(None, 0))(scaled_spline, gamma_test)
    exp = jnp.array([0.5, 0.5, 0.5])
    assert jnp.allclose(got, exp)
