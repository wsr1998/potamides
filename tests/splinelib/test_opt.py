"""Test the opt module."""

import interpax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Real

import potamides.splinelib as splib

# pytest.mark.array_compare generates errors b/c of an old-style hookwrapper
# teardown.
pytestmark = pytest.mark.filterwarnings("ignore::pluggy.PluggyTeardownRaisedWarning")


@pytest.fixture(scope="module")
def test_gamma() -> Real[Array, "10"]:
    r"""Fixture for test gamma values."""
    return jnp.linspace(-1, 1, 10, dtype=float)


@pytest.fixture(scope="module")
def test_data() -> Real[Array, "10 2"]:
    r"""Fixture for test data points."""
    # Create a simple curved path
    t = jnp.linspace(0, jnp.pi, 10)
    x = jnp.cos(t)
    y = jnp.sin(t)
    return jnp.column_stack([x, y])


@pytest.fixture(scope="module")
def test_spline(test_gamma: Real[Array, "10"], test_data: Real[Array, "10 2"]):
    r"""Fixture for a test spline."""
    return interpax.Interpolator1D(test_gamma, test_data, method="cubic2")


@pytest.fixture(scope="module")
def test_data_gamma() -> Real[Array, "50"]:
    r"""Fixture for dense gamma values for data fitting."""
    return jnp.linspace(-0.8, 0.8, 50, dtype=float)


@pytest.fixture(scope="module")
def test_data_y(test_spline, test_data_gamma: Real[Array, "50"]) -> Real[Array, "50 2"]:
    r"""Fixture for test target data with some noise."""
    # Generate target data from spline with small noise
    rng = np.random.default_rng(42)
    clean_data = test_spline(test_data_gamma)
    noise = 0.01 * rng.random(clean_data.shape)
    return clean_data + noise


##############################################################################
# Consistency tests
#
# These are NOT tests of correctness, but rather of consistency -- ensuring that
# the output of the tested functions do not change.


@pytest.mark.array_compare
def test_reduce_point_density_consistency(
    test_gamma: Real[Array, "10"], test_data: Real[Array, "10 2"]
) -> Real[Array, "16"]:
    r"""Test that reduce_point_density function is consistent."""
    gamma_reduced, data_reduced = splib.reduce_point_density(
        test_gamma, test_data, num_splits=3
    )
    assert gamma_reduced.shape == (5,)  # num_splits + 2
    assert data_reduced.shape == (5, 2)

    # Stack the results for comparison
    return jnp.concatenate([gamma_reduced, data_reduced.flatten()])


@pytest.mark.array_compare
def test_reduce_point_density_with_mean_consistency(
    test_gamma: Real[Array, "10"], test_data: Real[Array, "10 2"]
) -> Real[Array, "16"]:
    r"""Test that reduce_point_density function is consistent with mean reduction."""
    gamma_reduced, data_reduced = splib.reduce_point_density(
        test_gamma, test_data, num_splits=3, reduce_fn=jnp.mean
    )
    assert gamma_reduced.shape == (5,)  # num_splits + 2
    assert data_reduced.shape == (5, 2)

    # Stack the results for comparison
    return jnp.concatenate([gamma_reduced, data_reduced.flatten()])


@pytest.mark.array_compare
def test_data_distance_cost_fn_consistency(
    test_gamma: Real[Array, "10"],
    test_data: Real[Array, "10 2"],
    test_data_gamma: Real[Array, "50"],
    test_data_y: Real[Array, "50 2"],
) -> Real[Array, "1"]:
    r"""Test that data_distance_cost_fn function is consistent."""
    cost = splib.data_distance_cost_fn(
        test_data, test_gamma, test_data_gamma, test_data_y, sigmas=1.0
    )
    assert cost.shape == ()
    return jnp.array([cost])


@pytest.mark.array_compare
def test_data_distance_cost_fn_with_sigmas_consistency(
    test_gamma: Real[Array, "10"],
    test_data: Real[Array, "10 2"],
    test_data_gamma: Real[Array, "50"],
    test_data_y: Real[Array, "50 2"],
) -> Real[Array, "1"]:
    r"""Test that data_distance_cost_fn function is consistent with custom sigmas."""
    # Create variable sigmas with proper shape - needs to broadcast with data_y
    sigmas = jnp.ones((50, 2)) * 0.5  # Match shape of data_y
    cost = splib.data_distance_cost_fn(
        test_data, test_gamma, test_data_gamma, test_data_y, sigmas=sigmas
    )
    assert cost.shape == ()
    return jnp.array([cost])


@pytest.mark.array_compare
def test_concavity_change_cost_fn_consistency(
    test_gamma: Real[Array, "10"],
    test_data: Real[Array, "10 2"],
    test_data_gamma: Real[Array, "50"],
) -> Real[Array, "1"]:
    r"""Test that concavity_change_cost_fn function is consistent."""
    # Use static (non-traced) parameters to avoid concretization issues
    cost = splib.concavity_change_cost_fn(
        test_data, test_gamma, test_data_gamma, scale=100.0, num_points=100
    )
    assert cost.shape == ()
    return jnp.array([cost])


@pytest.mark.array_compare
def test_default_cost_fn_consistency(
    test_gamma: Real[Array, "10"],
    test_data: Real[Array, "10 2"],
    test_data_gamma: Real[Array, "50"],
    test_data_y: Real[Array, "50 2"],
) -> Real[Array, "1"]:
    r"""Test that ``default_cost_fn`` function is consistent."""
    cost = splib.default_cost_fn(
        test_data,
        test_gamma,
        test_data_gamma,
        test_data_y,
        sigmas=1.0,
        data_weight=1e3,
        concavity_weight=0.0,
    )
    assert cost.shape == ()
    return jnp.array([cost])


@pytest.mark.array_compare
def test_default_cost_fn_with_concavity_consistency(
    test_gamma: Real[Array, "10"],
    test_data: Real[Array, "10 2"],
    test_data_gamma: Real[Array, "50"],
    test_data_y: Real[Array, "50 2"],
) -> Real[Array, "1"]:
    r"""Test that default_cost_fn function is consistent with concavity penalty."""
    cost = splib.default_cost_fn(
        test_data,
        test_gamma,
        test_data_gamma,
        test_data_y,
        sigmas=1.0,
        data_weight=1e3,
        concavity_weight=0.1,
        concavity_scale=1e2,
    )
    assert cost.shape == ()
    return jnp.array([cost])


@pytest.mark.array_compare
def test_optimize_spline_knots_consistency(
    test_gamma: Real[Array, "10"],
    test_data: Real[Array, "10 2"],
    test_data_gamma: Real[Array, "50"],
    test_data_y: Real[Array, "50 2"],
) -> Real[Array, "20"]:
    r"""Test that optimize_spline_knots function is consistent."""
    # Use hashable cost_kwargs (tuple of tuples instead of dict)
    cost_kwargs = (
        ("sigmas", 1.0),
        ("data_weight", 1e3),
        ("concavity_weight", 0.0),
    )

    optimized_knots = splib.optimize_spline_knots(
        splib.default_cost_fn,
        test_data,
        test_gamma,
        (test_data_gamma, test_data_y),
        cost_kwargs=cost_kwargs,
        nsteps=10,  # Small number for consistency
    )
    assert optimized_knots.shape == (10, 2)
    return optimized_knots.flatten()


@pytest.mark.array_compare
def test_optimize_spline_knots_with_fixed_mask_consistency(
    test_gamma: Real[Array, "10"],
    test_data: Real[Array, "10 2"],
    test_data_gamma: Real[Array, "50"],
    test_data_y: Real[Array, "50 2"],
) -> Real[Array, "20"]:
    r"""Test that optimize_spline_knots function is consistent with fixed mask."""
    # Fix the first and last knots
    fixed_mask = (True, False, False, False, False, False, False, False, False, True)

    # Use hashable cost_kwargs (tuple of tuples instead of dict)
    cost_kwargs = (
        ("sigmas", 1.0),
        ("data_weight", 1e3),
        ("concavity_weight", 0.0),
    )

    optimized_knots = splib.optimize_spline_knots(
        splib.default_cost_fn,
        test_data,
        test_gamma,
        (test_data_gamma, test_data_y),
        cost_kwargs=cost_kwargs,
        fixed_mask=fixed_mask,
        nsteps=10,  # Small number for consistency
    )
    assert optimized_knots.shape == (10, 2)
    return optimized_knots.flatten()


@pytest.mark.array_compare
def test_new_gamma_knots_from_spline_consistency(test_spline) -> Real[Array, "14"]:
    r"""Test that new_gamma_knots_from_spline function is consistent."""
    gamma_new, points_new = splib.new_gamma_knots_from_spline(test_spline, nknots=5)
    assert gamma_new.shape == (5,)
    assert points_new.shape == (5, 2)

    # Stack the results for comparison
    return jnp.concatenate([gamma_new, points_new.flatten()])


@pytest.mark.array_compare
def test_new_gamma_knots_from_spline_more_knots_consistency(
    test_spline,
) -> Real[Array, "24"]:
    r"""Test that new_gamma_knots_from_spline function is consistent with more knots."""
    gamma_new, points_new = splib.new_gamma_knots_from_spline(test_spline, nknots=8)
    assert gamma_new.shape == (8,)
    assert points_new.shape == (8, 2)

    # Stack the results for comparison
    return jnp.concatenate([gamma_new, points_new.flatten()])


##############################################################################
# Correctness tests


def test_reduce_point_density_correctness() -> None:
    """Test the correctness of ``reduce_point_density`` function."""
    gamma = jnp.array([-1, 0, 0.5, 1])
    data = jnp.array([[0, 0], [1, 0], [1, 2], [0, 2]])

    gamma2, data2 = splib.reduce_point_density(gamma, data, num_splits=1)

    assert jnp.allclose(gamma2, jnp.array([-1, 0.25, 1]), rtol=1e-5, atol=1e-8)
    assert jnp.allclose(
        data2, jnp.array([[0.0, 0.0], [0.5, 1.0], [0.0, 2.0]]), rtol=1e-5, atol=1e-8
    )


# def test_data_distance_cost_fn_correctness() -> None:
#     """Test the correctness of ``data_distance_cost_fn`` function."""
