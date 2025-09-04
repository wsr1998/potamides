"""Test the plotting functions."""

import galax.potential as gp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
import unxt as u
from jaxtyping import Array, Float

import potamides as ptd
from potamides._src.custom_types import SzGamma, SzN2

# pytest.mark.array_compare generates errors b/c of an old-style hookwrapper
# teardown.
pytestmark = pytest.mark.filterwarnings("ignore::pluggy.PluggyTeardownRaisedWarning")


# =============================================================================
# Fixtures


@pytest.fixture(scope="module")
def sample_accelerations() -> SzN2:
    """Fixture for sample 2D acceleration vectors."""
    return jnp.array(
        [
            [1.0, 0.0],  # +x direction
            [0.0, 1.0],  # +y direction
            [-1.0, 0.0],  # -x direction
            [0.0, -1.0],  # -y direction
            [1.0, 1.0],  # 45° angle
            [-1.0, -1.0],  # 225° angle
        ]
    )


@pytest.fixture(scope="module")
def sample_normals() -> SzN2:
    """Fixture for sample normal vectors."""
    return jnp.array(
        [
            [0.0, 1.0],  # +y direction
            [-1.0, 0.0],  # -x direction
            [0.0, -1.0],  # -y direction
            [1.0, 0.0],  # +x direction
            [-1.0, 1.0],  # Normal to 45° vector
            [1.0, 1.0],  # Another normal
        ]
    )


@pytest.fixture(scope="module")
def sample_gamma() -> SzGamma:
    """Fixture for sample gamma parameter values."""
    return jnp.linspace(-1.0, 1.0, 20)


@pytest.fixture(scope="module")
def sample_params() -> Float[Array, "param"]:  # type: ignore[name-defined]
    """Fixture for sample parameter values."""
    return jnp.array([0.5, 0.7, 0.9, 1.0, 1.2])


@pytest.fixture(scope="module")
def sample_angles(sample_params, sample_gamma) -> Float[Array, "param gamma"]:
    """Fixture for sample angle values."""
    angles = jnp.zeros((len(sample_params), len(sample_gamma)))
    for i, q in enumerate(sample_params):
        angles = angles.at[i].set(
            0.3 * jnp.sin(2 * jnp.pi * sample_gamma) / q + 0.1 * sample_gamma
        )
    return angles


@pytest.fixture(scope="module")
def logarithmic_potential():
    """Fixture for a logarithmic gravitational potential."""
    return gp.LMJ09LogarithmicPotential(
        v_c=u.Quantity(250, "km/s"),
        r_s=u.Quantity(16, "kpc"),
        q1=1.0,
        q2=0.9,
        q3=0.8,
        phi=0.0,
        units="galactic",
    )


@pytest.fixture(scope="module")
def disk_potential():
    """Fixture for a Miyamoto-Nagai disk potential."""
    return gp.MiyamotoNagaiPotential(
        m_tot=u.Quantity(1.2e10, "Msun"),
        a=u.Quantity(3, "kpc"),
        b=u.Quantity(0.5, "kpc"),
        units="galactic",
    )


@pytest.fixture(scope="module")
def composite_potential(logarithmic_potential, disk_potential):
    """Fixture for a composite potential."""
    return gp.CompositePotential(halo=logarithmic_potential, disk=disk_potential)


# =============================================================================
# Consistency tests with array comparison


@pytest.mark.array_compare
def test_get_angles_basic_consistency(sample_accelerations, sample_normals):
    """Test get_angles produces consistent output for basic vectors."""
    return ptd.get_angles(sample_accelerations, sample_normals)


@pytest.mark.array_compare
def test_get_angles_normalized_consistency():
    """Test get_angles handles normalization consistently."""
    # Test with large magnitude vectors
    large_acc = jnp.array([[100.0, 0.0], [50.0, 50.0], [0.0, 200.0]])
    large_normals = jnp.array([[0.0, 300.0], [100.0, 0.0], [150.0, 0.0]])

    return ptd.get_angles(large_acc, large_normals)


@pytest.mark.array_compare
def test_get_angles_edge_cases_consistency():
    """Test get_angles handles edge cases consistently."""
    # Test with very small vectors
    small_acc = jnp.array([[1e-10, 0.0], [0.0, 1e-10]])
    small_normals = jnp.array([[0.0, 1e-10], [1e-10, 0.0]])

    return ptd.get_angles(small_acc, small_normals)


# =============================================================================
# Correctness tests


def test_get_angles_orthogonal_vectors():
    """Test get_angles with orthogonal vectors gives ±π/2."""
    # Acceleration in +x, normal in +y should give π/2
    acc = jnp.array([[1.0, 0.0]])
    normal = jnp.array([[0.0, 1.0]])

    angle = ptd.get_angles(acc, normal)
    assert jnp.allclose(angle, jnp.pi / 2, atol=1e-6)

    # Acceleration in +x, normal in -y should give -π/2
    normal_neg = jnp.array([[0.0, -1.0]])
    angle_neg = ptd.get_angles(acc, normal_neg)
    assert jnp.allclose(angle_neg, -jnp.pi / 2, atol=1e-6)


def test_get_angles_parallel_vectors():
    """Test get_angles with parallel vectors gives 0 or π."""
    # Same direction should give 0
    acc = jnp.array([[1.0, 0.0]])
    normal = jnp.array([[1.0, 0.0]])

    angle = ptd.get_angles(acc, normal)
    assert jnp.allclose(angle, 0.0, atol=1e-6)

    # Opposite direction should give π
    normal_opp = jnp.array([[-1.0, 0.0]])
    angle_opp = ptd.get_angles(acc, normal_opp)
    assert jnp.allclose(jnp.abs(angle_opp), jnp.pi, atol=1e-6)


def test_get_angles_shape_consistency():
    """Test get_angles returns correct shapes."""
    n_points = 10
    acc = jnp.ones((n_points, 2))
    normals = jnp.ones((n_points, 2))

    angles = ptd.get_angles(acc, normals)
    assert angles.shape == (n_points,)


def test_get_angles_normalization_invariance():
    """Test get_angles is invariant to vector magnitudes."""
    acc1 = jnp.array([[1.0, 0.0]])
    acc2 = jnp.array([[100.0, 0.0]])  # Same direction, different magnitude
    normal = jnp.array([[0.0, 1.0]])

    angle1 = ptd.get_angles(acc1, normal)
    angle2 = ptd.get_angles(acc2, normal)

    assert jnp.allclose(angle1, angle2, atol=1e-6)


def test_get_angles_range():
    """Test get_angles returns values in [-π, π]."""
    # Test many random vectors
    key = jax.random.PRNGKey(42)
    acc = jax.random.normal(key, (100, 2))
    normals = jax.random.normal(jax.random.split(key)[1], (100, 2))

    angles = ptd.get_angles(acc, normals)

    assert jnp.all(angles >= -jnp.pi)
    assert jnp.all(angles <= jnp.pi)


def test_plot_theta_of_gamma_return_types(sample_gamma, sample_params, sample_angles):
    """Test plot_theta_of_gamma returns correct types."""
    fig, ax = ptd.plot.plot_theta_of_gamma(sample_gamma, sample_params, sample_angles)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_plot_theta_of_gamma_with_mle(sample_gamma, sample_params, sample_angles):
    """Test plot_theta_of_gamma with MLE highlighting."""
    fig, ax = ptd.plot.plot_theta_of_gamma(
        sample_gamma, sample_params, sample_angles, mle_idx=2
    )

    # Check that the MLE line was added
    lines = ax.get_lines()
    assert len(lines) > 0
    # Should have a red line for MLE
    red_lines = [line for line in lines if line.get_color() == "red"]
    assert len(red_lines) > 0


@pytest.mark.mpl_image_compare(deterministic=True)
def test_plot_theta_of_gamma_exclusion_regions():
    """Test plot showing exclusion regions clearly."""
    gamma = jnp.linspace(-1.0, 1.0, 30)
    params = jnp.array([0.3, 0.6, 1.0])  # Include problematic parameter

    # Create angles that go into exclusion regions
    angles = jnp.zeros((len(params), len(gamma)))
    for i, q in enumerate(params):
        if q < 0.5:
            # Low values produce large angles that enter exclusion regions
            angles = angles.at[i].set(1.8 * jnp.sin(3 * jnp.pi * gamma))
        else:
            angles = angles.at[i].set(0.4 * jnp.sin(jnp.pi * gamma))

    fig, ax = ptd.plot.plot_theta_of_gamma(
        gamma, params, angles, param_label=r"Problem parameter"
    )
    ax.set_title("Angles with Exclusion Regions")

    return fig


# =============================================================================
# Matplotlib image comparison tests


@pytest.mark.mpl_image_compare(deterministic=True)
def test_plot_theta_of_gamma_basic_plot():
    """Test basic plot_theta_of_gamma visualization."""
    # Use deterministic data for reproducible plots
    gamma = jnp.linspace(-1.0, 1.0, 25)
    params = jnp.array([0.6, 0.8, 1.0, 1.2])

    angles = jnp.zeros((len(params), len(gamma)))
    for i, q in enumerate(params):
        angles = angles.at[i].set(0.4 * jnp.sin(jnp.pi * gamma) / q)

    fig, ax = ptd.plot.plot_theta_of_gamma(
        gamma, params, angles, param_label=r"Parameter $q$"
    )
    ax.set_title("Basic Theta vs Gamma Plot")

    return fig


@pytest.mark.mpl_image_compare(deterministic=True)
def test_plot_theta_of_gamma_with_mle_highlight():
    """Test plot_theta_of_gamma with MLE highlighting."""
    gamma = jnp.linspace(-0.8, 0.8, 20)
    params = jnp.array([0.5, 0.7, 0.9, 1.1])

    # Create more varied angle data
    angles = jnp.zeros((len(params), len(gamma)))
    for i, q in enumerate(params):
        angles = angles.at[i].set(0.3 * jnp.sin(2 * jnp.pi * gamma) / q + 0.15 * gamma)

    fig, ax = ptd.plot.plot_theta_of_gamma(
        gamma,
        params,
        angles,
        mle_idx=2,  # Highlight third parameter
        param_label=r"Mass ratio",
    )
    ax.set_title("Theta vs Gamma with MLE")

    return fig


@pytest.mark.mpl_image_compare(deterministic=True)
def test_plot_acceleration_field_logarithmic():
    """Test acceleration field plot for logarithmic potential."""
    # Create a deterministic logarithmic potential
    potential = gp.LMJ09LogarithmicPotential(
        v_c=u.Quantity(220, "km/s"),
        r_s=u.Quantity(15, "kpc"),
        q1=1.0,
        q2=0.9,
        q3=0.8,
        phi=0.0,
        units="galactic",
    )

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    ptd.plot.plot_acceleration_field(
        potential,
        xlim=(-15, 15),
        ylim=(-15, 15),
        grid_size=12,
        ax=ax,
        vec_width=0.004,
        vec_scale=25,
        color="#2E8B57",
    )

    ax.set_title("Logarithmic Potential Acceleration Field")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.grid(True, alpha=0.3)

    return fig


@pytest.mark.mpl_image_compare(deterministic=True)
def test_plot_acceleration_field_disk():
    """Test acceleration field plot for disk potential."""
    # Create a deterministic disk potential
    potential = gp.MiyamotoNagaiPotential(
        m_tot=u.Quantity(1e10, "Msun"),
        a=u.Quantity(3, "kpc"),
        b=u.Quantity(0.4, "kpc"),
        units="galactic",
    )

    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

    ptd.plot.plot_acceleration_field(
        potential,
        xlim=(-8, 8),
        ylim=(-6, 6),
        grid_size=14,
        ax=ax,
        vec_width=0.003,
        vec_scale=30,
        color="blue",
    )

    ax.set_title("Miyamoto-Nagai Disk Acceleration Field")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")

    return fig


@pytest.mark.mpl_image_compare(deterministic=True)
def test_plot_acceleration_field_composite_with_stream():
    """Test acceleration field plot for composite potential with stream overlay."""
    # Create deterministic composite potential
    halo_pot = gp.LMJ09LogarithmicPotential(
        v_c=u.Quantity(200, "km/s"),
        r_s=u.Quantity(18, "kpc"),
        q1=1.0,
        q2=1.0,
        q3=1.0,
        phi=0.0,
        units="galactic",
    )

    disk_pot = gp.MiyamotoNagaiPotential(
        m_tot=u.Quantity(8e9, "Msun"),
        a=u.Quantity(2.5, "kpc"),
        b=u.Quantity(0.3, "kpc"),
        units="galactic",
    )

    composite_pot = gp.CompositePotential(halo=halo_pot, disk=disk_pot)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

    # Add a synthetic stream
    stream_x = np.linspace(-6, 6, 25)
    stream_y = 0.08 * stream_x**2 - 0.5  # Parabolic stream
    ax.plot(stream_x, stream_y, "k-", linewidth=4, label="Stellar Stream", alpha=0.8)

    # Add acceleration field
    ptd.plot.plot_acceleration_field(
        composite_pot,
        xlim=(-8, 8),
        ylim=(-3, 5),
        grid_size=10,
        ax=ax,
        color="gray",
        vec_width=0.004,
        vec_scale=35,
    )

    ax.set_title("Composite Potential with Stellar Stream")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.legend()
    ax.grid(True, alpha=0.2)

    return fig
