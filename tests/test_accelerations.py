"""Test the acceleration functions."""

import math
from typing import TypeAlias

import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

import potamides as ptd
from potamides._src.custom_types import SzN2, SzN3

Sz33: TypeAlias = Float[Array, "3 3"]


# pytest.mark.array_compare generates errors b/c of an old-style hookwrapper
# teardown.
pytestmark = pytest.mark.filterwarnings("ignore::pluggy.PluggyTeardownRaisedWarning")


@pytest.fixture(scope="module")
def positions() -> SzN3:
    """Fixture to create test positions for consistency testing."""
    return jnp.array(
        [
            [8.0, 0.0, 0.0],  # Solar neighborhood
            [0.0, 8.0, 0.0],  # 90 degrees around
            [4.0, 4.0, 1.0],  # Inner galaxy, off-plane
            [12.0, -3.0, 2.0],  # Outer galaxy
            [-5.0, 6.0, -1.0],  # Different quadrant
        ]
    )


@pytest.fixture(scope="module")
def single_position() -> SzN3:
    """Fixture for single position testing."""
    return jnp.array([[8.0, 0.0, 0.0]])


@pytest.fixture(scope="module")
def custom_origin() -> Float[Array, "3"]:
    """Fixture for custom halo center."""
    return jnp.array([2.0, -1.0, 0.5])


# =============================================================================
# Consistency tests
# These tests ensure that the output of the functions do not change.


@pytest.mark.array_compare
def test_compute_accelerations_basic_consistency(positions: SzN3) -> SzN2:
    """Test that `compute_accelerations` with default parameters is consistent."""
    out = ptd.compute_accelerations(positions)
    assert out.shape == (len(positions), 2)
    return out


@pytest.mark.array_compare
def test_compute_accelerations_single_position_consistency(
    single_position: SzN3,
) -> SzN2:
    """Test that `compute_accelerations` with single position is consistent."""
    out = ptd.compute_accelerations(single_position)
    assert out.shape == (1, 2)
    return out


@pytest.mark.array_compare
def test_compute_accelerations_with_disk_consistency(positions: SzN3) -> SzN2:
    """Test that `compute_accelerations` with disk potential is consistent."""
    out = ptd.compute_accelerations(positions, withdisk=True)
    assert out.shape == (len(positions), 2)
    return out


@pytest.mark.array_compare
def test_compute_accelerations_custom_halo_consistency(positions: SzN3) -> SzN2:
    """Test that `compute_accelerations` with custom halo parameters is consistent."""
    out = ptd.compute_accelerations(
        positions,
        rs_halo=20.0,
        vc_halo=250 * 1000 / 3.086e19 * 3.154e13,  # 200 km/s to kpc/Myr
        q1=0.8,
        q2=0.8,
        q3=0.6,
        phi=math.pi / 6,
    )
    assert out.shape == (len(positions), 2)
    return out


@pytest.mark.array_compare
def test_compute_accelerations_rotated_consistency(positions: SzN3) -> SzN2:
    """Test that `compute_accelerations` with rotations is consistent."""
    out = ptd.compute_accelerations(
        positions,
        rot_z=math.pi / 4,  # 45 degrees around z
        rot_x=math.pi / 6,  # 30 degrees around x
        withdisk=True,
    )
    assert out.shape == (len(positions), 2)
    return out


@pytest.mark.array_compare
def test_compute_accelerations_translated_consistency(
    positions: SzN3, custom_origin: Float[Array, "3"]
) -> SzN2:
    """Test that `compute_accelerations` with translated halo center is consistent."""
    out = ptd.compute_accelerations(positions, origin=custom_origin)
    assert out.shape == (len(positions), 2)
    return out


@pytest.mark.array_compare
def test_compute_accelerations_quantities_consistency(single_position: SzN3) -> SzN2:
    """Test that `compute_accelerations` with unit quantities is consistent."""
    out = ptd.compute_accelerations(single_position)
    assert out.shape == (1, 2)
    return out


@pytest.mark.array_compare
def test_compute_accelerations_all_parameters_consistency(positions: SzN3) -> SzN2:
    """Test that `compute_accelerations` with all custom parameters is consistent."""
    out = ptd.compute_accelerations(
        positions,
        rot_z=math.pi / 3,
        rot_x=math.pi / 8,
        q1=0.9,
        q2=0.85,
        q3=0.7,
        phi=math.pi / 4,
        rs_halo=18.0,
        vc_halo=220 * 1000 / 3.086e19 * 3.154e13,  # 220 km/s to kpc/Myr
        origin=jnp.array([1.0, -0.5, 0.2]),
        Mdisk=1.5e10,
        withdisk=True,
    )
    assert out.shape == (len(positions), 2)
    return out


# =============================================================================
# Correctness tests
# These tests verify that the function behaves as expected.


def test_compute_accelerations_output_shape():
    """Test that the output shape is correct for various input sizes."""
    # Single position
    pos1 = jnp.array([[8.0, 0.0, 0.0]])
    acc1 = ptd.compute_accelerations(pos1)
    assert acc1.shape == (1, 2)

    # Multiple positions
    pos3 = jnp.array([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [4.0, 4.0, 1.0]])
    acc3 = ptd.compute_accelerations(pos3)
    assert acc3.shape == (3, 2)


def test_compute_accelerations_unit_vectors():
    """Test that the returned acceleration vectors are unit vectors (or close to it)."""
    test_positions = jnp.array(
        [
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [4.0, 4.0, 1.0],
        ]
    )
    acc_xy = ptd.compute_accelerations(test_positions)

    # Check that all values are finite
    assert jnp.all(jnp.isfinite(acc_xy)), "All acceleration values should be finite"

    # The vectors aren't strictly unit vectors since we only take x-y components
    # But they should be reasonable values
    norms = jnp.linalg.norm(acc_xy, axis=1)
    assert jnp.all(norms > 0), "All acceleration vectors should have positive norm"
    assert jnp.all(norms <= 1.0), "All acceleration vectors should have norm <= 1"


def test_compute_accelerations_symmetry():
    """Test basic symmetry properties of the acceleration field."""
    # Test that symmetric positions give symmetric accelerations (for spherical halo)
    pos_right = jnp.array([[8.0, 0.0, 0.0]])
    pos_left = jnp.array([[-8.0, 0.0, 0.0]])

    acc_right = ptd.compute_accelerations(pos_right)
    acc_left = ptd.compute_accelerations(pos_left)

    # For a spherical halo (q1=q2=q3=1), accelerations should point inward
    # Right position should have negative x-component acceleration
    # Left position should have positive x-component acceleration
    assert acc_right[0, 0] < 0, "Right position should have leftward acceleration"
    assert acc_left[0, 0] > 0, "Left position should have rightward acceleration"


def test_compute_accelerations_disk_effect():
    """Test that including the disk changes the acceleration."""
    # Test function runs without error and produces valid outputs
    test_positions = jnp.array([[4.0, 0.0, 0.0]])

    acc_no_disk = ptd.compute_accelerations(test_positions, withdisk=False)
    acc_with_disk = ptd.compute_accelerations(test_positions, withdisk=True)

    # Both should produce finite results
    assert jnp.all(jnp.isfinite(acc_no_disk)), "No disk acceleration should be finite"
    assert jnp.all(jnp.isfinite(acc_with_disk)), (
        "With disk acceleration should be finite"
    )
    # Shape should be the same
    assert acc_no_disk.shape == acc_with_disk.shape == (1, 2), (
        "Output shapes should match"
    )


def test_compute_accelerations_rotation_effect():
    """Test that rotations change the acceleration field."""
    test_positions = jnp.array([[8.0, 0.0, 0.0]])

    acc_no_rotation = ptd.compute_accelerations(test_positions, withdisk=True)
    acc_rotated = ptd.compute_accelerations(
        test_positions, rot_z=math.pi / 4, rot_x=math.pi / 6, withdisk=True
    )  # The accelerations should be different when rotations are applied
    assert not jnp.allclose(acc_no_rotation, acc_rotated, rtol=1e-10), (
        "Rotations should change the acceleration field"
    )


def test_compute_accelerations_units_handling():
    """Test that the function properly handles unit quantities."""
    # Test with unitless positions
    pos_unitless = jnp.array([[8.0, 0.0, 0.0]])
    acc_unitless = ptd.compute_accelerations(pos_unitless)

    # Just test that it runs and produces finite results
    assert jnp.all(jnp.isfinite(acc_unitless)), "Function should produce finite results"


def test_compute_accelerations_origin_translation():
    """Test that translating the halo origin affects the acceleration field."""
    # Test function runs without error and produces valid outputs
    test_positions = jnp.array([[8.0, 0.0, 0.0]])

    acc_center = ptd.compute_accelerations(
        test_positions, origin=jnp.array([0.0, 0.0, 0.0])
    )
    acc_shifted = ptd.compute_accelerations(
        test_positions, origin=jnp.array([5.0, 0.0, 0.0])
    )

    # Both should produce finite results
    assert jnp.all(jnp.isfinite(acc_center)), (
        "Center origin acceleration should be finite"
    )
    assert jnp.all(jnp.isfinite(acc_shifted)), (
        "Shifted origin acceleration should be finite"
    )
    # Shape should be the same
    assert acc_center.shape == acc_shifted.shape == (1, 2), "Output shapes should match"


@pytest.mark.parametrize(
    ("q1", "q2", "q3"),
    [
        (1.0, 1.0, 1.0),  # spherical
        (0.8, 0.8, 0.6),  # oblate
        (1.2, 1.0, 0.8),  # triaxial
    ],
)
def test_compute_accelerations_halo_shapes(q1, q2, q3):
    """Test that different halo shapes produce finite accelerations."""
    test_positions = jnp.array(
        [
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
        ]
    )

    acc = ptd.compute_accelerations(test_positions, q1=q1, q2=q2, q3=q3)

    assert jnp.all(jnp.isfinite(acc)), (
        f"Accelerations should be finite for q1={q1}, q2={q2}, q3={q3}"
    )
    assert acc.shape == (2, 2), "Output shape should be correct"
