"""Test the likelihood functions."""

from typing import TypeAlias

import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float, Int, Real

import potamides as ptd
from potamides._src.custom_types import BoolSzGamma, SzGamma2

# pytest.mark.array_compare generates errors b/c of an old-style hookwrapper
# teardown.
pytestmark = pytest.mark.filterwarnings("ignore::pluggy.PluggyTeardownRaisedWarning")

Sz1: TypeAlias = Float[Array, "1"]
SzS: TypeAlias = Real[Array, "S"]  # type: ignore[name-defined]
IntSzS: TypeAlias = Int[Array, "S"]  # type: ignore[name-defined]


# Helper function to convert scalar to 1D array for array_compare
def _scalar_to_array(scalar: float) -> Sz1:
    """Convert scalar output to 1D array for array_compare compatibility."""
    out_reshaped = jnp.array([scalar])
    assert out_reshaped.shape == (1,)
    return out_reshaped


@pytest.fixture(scope="module")
def kappa_hat() -> SzGamma2:
    """Fixture to create unit curvature vectors for testing."""
    return jnp.array(
        [
            [1.0, 0.0],  # pointing right
            [0.0, 1.0],  # pointing up
            [-1.0, 0.0],  # pointing left
            [0.0, -1.0],  # pointing down
            [0.707, 0.707],  # diagonal
        ]
    )


@pytest.fixture(scope="module")
def acc_xy_unit() -> SzGamma2:
    """Fixture to create unit acceleration vectors for testing."""
    return jnp.array(
        [
            [1.0, 0.0],  # aligned with first kappa_hat
            [0.0, 1.0],  # aligned with second kappa_hat
            [-0.8, 0.2],  # mostly aligned with third kappa_hat
            [0.1, -0.9],  # mostly aligned with fourth kappa_hat
            [0.6, 0.8],  # somewhat aligned with fifth kappa_hat
        ]
    )


@pytest.fixture(scope="module")
def where_straight() -> BoolSzGamma:
    """Fixture to create a boolean mask for straight segments."""
    return jnp.array([False, False, True, False, False])


@pytest.fixture(scope="module")
def lnliks() -> SzS:
    """Fixture to create log-likelihoods for testing combine_ln_likelihoods."""
    return jnp.array([0.5, 1.0, 1.5, 2.0])


@pytest.fixture(scope="module")
def ngammas() -> IntSzS:
    """Fixture to create number of gamma points for testing."""
    return jnp.array([100, 150, 200, 250])


@pytest.fixture(scope="module")
def arclengths() -> SzS:
    """Fixture to create arc lengths for testing."""
    return jnp.array([1.0, 1.5, 2.0, 2.5])


# Common test data for correctness tests
@pytest.fixture(scope="module")
def simple_kappa_hat() -> SzGamma2:
    """Simple 3-point curvature vectors for correctness tests."""
    return jnp.array(
        [
            [1.0, 0.0],  # pointing right
            [0.0, 1.0],  # pointing up
            [-1.0, 0.0],  # pointing left
        ]
    )


@pytest.fixture(scope="module")
def aligned_acc() -> SzGamma2:
    """Perfectly aligned acceleration vectors."""
    return jnp.array(
        [
            [1.0, 0.0],  # perfectly aligned
            [0.0, 1.0],  # perfectly aligned
            [-1.0, 0.0],  # perfectly aligned
        ]
    )


@pytest.fixture(scope="module")
def anti_aligned_acc() -> SzGamma2:
    """Anti-aligned acceleration vectors."""
    return jnp.array(
        [
            [-1.0, 0.0],  # opposite direction
            [0.0, -1.0],  # opposite direction
            [1.0, 0.0],  # opposite direction
        ]
    )


##############################################################################
# Consistency tests
#
# There is NOT tests of correctness, but rather of consistency -- ensuring that
# the output of the tested functions do not change.


@pytest.mark.array_compare
def test_compute_ln_likelihood_consistency(
    kappa_hat: SzGamma2, acc_xy_unit: SzGamma2
) -> Real[Array, "1"]:
    r"""Test that `compute_ln_likelihood` is consistent."""
    out = ptd.compute_ln_likelihood(kappa_hat, acc_xy_unit)
    return _scalar_to_array(out)


@pytest.mark.array_compare
def test_compute_ln_likelihood_with_straight_consistency(
    kappa_hat: SzGamma2, acc_xy_unit: SzGamma2, where_straight: BoolSzGamma
) -> Real[Array, "1"]:
    r"""Test that `compute_ln_likelihood` with straight segments is consistent."""
    out = ptd.compute_ln_likelihood(
        kappa_hat, acc_xy_unit, where_straight=where_straight
    )
    return _scalar_to_array(out)


@pytest.mark.array_compare
def test_compute_ln_likelihood_custom_sigma_consistency(
    kappa_hat: SzGamma2, acc_xy_unit: SzGamma2
) -> Real[Array, "1"]:
    r"""Test that `compute_ln_likelihood` with custom sigma_theta is consistent."""
    out = ptd.compute_ln_likelihood(
        kappa_hat, acc_xy_unit, sigma_theta=jnp.deg2rad(5.0)
    )
    return _scalar_to_array(out)


@pytest.mark.array_compare
def test_combine_ln_likelihoods_consistency(
    lnliks: SzS, ngammas: IntSzS, arclengths: SzS
) -> Real[Array, "1"]:
    r"""Test that `combine_ln_likelihoods` is consistent."""
    out = ptd.combine_ln_likelihoods(lnliks, ngammas, arclengths)
    return _scalar_to_array(out)


@pytest.mark.array_compare
def test_combine_ln_likelihoods_vectorized_consistency() -> Real[Array, "2"]:
    r"""Test that vectorized `combine_ln_likelihoods` is consistent."""
    # Create 2D input arrays for vectorized operation
    lnliks = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    ngammas = jnp.array([[100, 200, 300], [150, 250, 350]])
    arclengths = jnp.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])

    out = ptd.combine_ln_likelihoods(lnliks, ngammas, arclengths)
    assert out.shape == (2,)  # vector output for vectorized inputs
    return out


##############################################################################
# Correctness tests


def test_compute_ln_likelihood_perfect_alignment(
    simple_kappa_hat: SzGamma2, aligned_acc: SzGamma2
):
    """Test that perfectly aligned curvature and acceleration gives positive likelihood."""
    ln_lik = ptd.compute_ln_likelihood(simple_kappa_hat, aligned_acc)
    assert ln_lik > 0, "Perfect alignment should give positive log-likelihood"


def test_compute_ln_likelihood_anti_alignment(
    simple_kappa_hat: SzGamma2, anti_aligned_acc: SzGamma2
):
    """Test that anti-aligned curvature and acceleration gives -inf likelihood."""
    ln_lik = ptd.compute_ln_likelihood(simple_kappa_hat, anti_aligned_acc)
    assert jnp.isinf(ln_lik), "Anti-alignment should give infinite log-likelihood"
    assert ln_lik < 0, "Anti-alignment should give negative log-likelihood"


def test_compute_ln_likelihood_with_straight_segments(
    simple_kappa_hat: SzGamma2, aligned_acc: SzGamma2
):
    """Test that straight segments parameter works without crashing."""
    # Test that function accepts where_straight parameter without crashing
    where_straight = jnp.array([False, False, False])  # all curved
    ln_lik = ptd.compute_ln_likelihood(
        simple_kappa_hat, aligned_acc, where_straight=where_straight
    )
    assert jnp.isfinite(ln_lik), "Should work when all segments are marked as curved"
    assert ln_lik > 0, "Should give positive likelihood with good alignment"


@pytest.mark.parametrize(
    ("lnliks", "ngammas", "arclengths", "expected"),
    [
        # Equal weights case
        ([1.0, 2.0, 3.0], [100, 100, 100], [1.0, 1.0, 1.0], 6.0),
        # Density weighting case: mean density = 300/2 = 150
        # Weights: [150/100, 150/200] = [1.5, 0.75]
        # Result: 1.5 * 1.0 + 0.75 * 1.0 = 2.25
        ([1.0, 1.0], [100, 200], [1.0, 1.0], 2.25),
    ],
)
def test_combine_ln_likelihoods_weighting(lnliks, ngammas, arclengths, expected):
    """Test combine_ln_likelihoods weighting behavior."""
    result = ptd.combine_ln_likelihoods(
        jnp.array(lnliks), jnp.array(ngammas), jnp.array(arclengths)
    )
    assert jnp.allclose(result, expected, rtol=1e-10), (
        f"Expected {expected}, got {result}"
    )


def test_combine_ln_likelihoods_vectorized():
    """Test vectorized operation of combine_ln_likelihoods."""
    # Two sets of segments
    lnliks = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    ngammas = jnp.array([[100, 200], [150, 250]])
    arclengths = jnp.array([[1.0, 2.0], [1.5, 2.5]])

    result = ptd.combine_ln_likelihoods(lnliks, ngammas, arclengths)
    assert result.shape == (2,), "Should return vector for vectorized inputs"
    assert jnp.all(jnp.isfinite(result)), "All results should be finite"
