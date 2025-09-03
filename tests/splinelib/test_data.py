"""Test the data module."""

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Real

import potamides.splinelib as splib

# pytest.mark.array_compare generates errors b/c of an old-style hookwrapper
# teardown.
pytestmark = pytest.mark.filterwarnings("ignore::pluggy.PluggyTeardownRaisedWarning")


@pytest.fixture(scope="module")
def data_simple() -> Real[Array, "4 2"]:
    r"""Fixture for simple test data - a rectangular path."""
    return jnp.array([[0, 0], [1, 0], [1, 2], [0, 2]], dtype=float)


@pytest.fixture(scope="module")
def data_with_plateaus() -> Real[Array, "5 2"]:
    r"""Fixture for test data with duplicate points that create plateaus."""
    return jnp.array([[0, 0], [1, 0], [1, 2], [1, 2], [0, 2]], dtype=float)


@pytest.fixture(scope="module")
def data_complex() -> Real[Array, "20 2"]:
    r"""Fixture for more complex test data with some noise."""
    # Create deterministic test data
    rng = np.random.default_rng(42)
    t = np.linspace(0, 2 * np.pi, 20)
    x = np.cos(t) + 0.1 * rng.random(len(t))
    y = np.sin(t) + 0.1 * rng.random(len(t))
    return jnp.array(np.column_stack([x, y]), dtype=float)


##############################################################################
# Consistency tests
#
# These are NOT tests of correctness, but rather of consistency -- ensuring that
# the output of the tested functions do not change.


@pytest.mark.array_compare
def test_point_to_point_distance_simple_consistency(
    data_simple: Real[Array, "4 2"],
) -> Real[Array, "3"]:
    r"""Test that point_to_point_distance function is consistent for simple data."""
    out = splib.point_to_point_distance(data_simple)
    assert out.shape == (3,)
    return out


@pytest.mark.array_compare
def test_point_to_point_distance_complex_consistency(
    data_complex: Real[Array, "20 2"],
) -> Real[Array, "19"]:
    r"""Test that point_to_point_distance function is consistent for complex data."""
    out = splib.point_to_point_distance(data_complex)
    assert out.shape == (19,)
    return out


@pytest.mark.array_compare
def test_point_to_point_arclenth_simple_consistency(
    data_simple: Real[Array, "4 2"],
) -> Real[Array, "3"]:
    r"""Test that point_to_point_arclenth function is consistent for simple data."""
    out = splib.point_to_point_arclenth(data_simple)
    assert out.shape == (3,)
    return out


@pytest.mark.array_compare
def test_point_to_point_arclenth_complex_consistency(
    data_complex: Real[Array, "20 2"],
) -> Real[Array, "19"]:
    r"""Test that point_to_point_arclenth function is consistent for complex data."""
    out = splib.point_to_point_arclenth(data_complex)
    assert out.shape == (19,)
    return out


@pytest.mark.array_compare
def test_make_gamma_from_data_simple_consistency(
    data_simple: Real[Array, "4 2"],
) -> Real[Array, "3"]:
    r"""Test that make_gamma_from_data function is consistent for simple data."""
    out = splib.make_gamma_from_data(data_simple)
    assert out.shape == (3,)
    return out


@pytest.mark.array_compare
def test_make_gamma_from_data_complex_consistency(
    data_complex: Real[Array, "20 2"],
) -> Real[Array, "19"]:
    r"""Test that make_gamma_from_data function is consistent for complex data."""
    out = splib.make_gamma_from_data(data_complex)
    assert out.shape == (19,)
    return out


@pytest.mark.array_compare
def test_make_gamma_from_data_with_plateaus_consistency(
    data_with_plateaus: Real[Array, "5 2"],
) -> Real[Array, "4"]:
    r"""Test that make_gamma_from_data function is consistent for data with plateaus."""
    out = splib.make_gamma_from_data(data_with_plateaus)
    assert out.shape == (4,)
    return out


@pytest.mark.array_compare
def test_make_increasing_gamma_from_data_simple_consistency(
    data_simple: Real[Array, "4 2"],
) -> Real[Array, "9"]:
    r"""Test that make_increasing_gamma_from_data function is consistent for simple data."""
    gamma, data_trimmed = splib.make_increasing_gamma_from_data(data_simple)
    assert gamma.shape == (3,)
    assert data_trimmed.shape == (3, 2)
    # Stack gamma and flattened data_trimmed for comparison
    return jnp.concatenate([gamma, data_trimmed.flatten()])


@pytest.mark.array_compare
def test_make_increasing_gamma_from_data_with_plateaus_consistency(
    data_with_plateaus: Real[Array, "5 2"],
) -> Real[Array, "9"]:
    r"""Test that make_increasing_gamma_from_data function is consistent for data with plateaus."""
    gamma, data_trimmed = splib.make_increasing_gamma_from_data(data_with_plateaus)
    assert gamma.shape == (3,)
    assert data_trimmed.shape == (3, 2)
    # Stack gamma and flattened data_trimmed for comparison
    return jnp.concatenate([gamma, data_trimmed.flatten()])


@pytest.mark.array_compare
def test_make_increasing_gamma_from_data_complex_consistency(
    data_complex: Real[Array, "20 2"],
) -> Real[Array, "57"]:
    r"""Test that make_increasing_gamma_from_data function is consistent for complex data."""
    gamma, data_trimmed = splib.make_increasing_gamma_from_data(data_complex)
    # Note: for complex data without explicit plateaus, the output should be the same size
    assert gamma.shape == (19,)
    assert data_trimmed.shape == (19, 2)
    # Stack gamma and flattened data_trimmed for comparison
    return jnp.concatenate([gamma, data_trimmed.flatten()])


##############################################################################
# Correctness tests


def test_point_to_point_distance_correctness(data_simple) -> None:
    """Test the point_to_point_distance function is correct."""
    expected_output = jnp.array([1, 2, 1], dtype=float)
    output = splib.point_to_point_distance(data_simple)
    np.testing.assert_allclose(output, expected_output, rtol=1e-5, atol=1e-8)


def test_point_to_point_arclenth_correctness(data_simple) -> None:
    """Test the point_to_point_arclenth function is correct."""
    expected_output = jnp.array([1, 3, 4], dtype=float)
    output = splib.point_to_point_arclenth(data_simple)
    np.testing.assert_allclose(output, expected_output, rtol=1e-5, atol=1e-8)


def test_make_gamma_from_data_correctness(data_simple) -> None:
    """Test the make_gamma_from_data function is correct."""
    expected_output = jnp.array([-1, 1 / 3, 1], dtype=float)
    output = splib.make_gamma_from_data(data_simple)
    np.testing.assert_allclose(output, expected_output, rtol=1e-5, atol=1e-8)


def test_make_increasing_gamma_from_data_correctness(data_with_plateaus) -> None:
    """Test the make_increasing_gamma_from_data function is correct."""
    expected_gamma = jnp.array([-1, 1 / 3, 1], dtype=float)
    gamma, data2 = splib.make_increasing_gamma_from_data(data_with_plateaus)
    np.testing.assert_allclose(gamma, expected_gamma, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        data2, jnp.array([[0.5, 0.0], [1.0, 1.0], [0.5, 2.0]]), rtol=1e-5, atol=1e-8
    )
