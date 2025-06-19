# potamides

```{toctree}
:maxdepth: 2
:hidden:

```

```{toctree}
:maxdepth: 2
:hidden:
:caption: 🔌 API Reference

api/index.md
```

# 🚀 Get Started

Potamides is ... in [JAX][jax].

## Installation

<!-- [![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link] -->

::::{tab-set}

:::{tab-item} pip

```bash
pip install potamides
```

:::

:::{tab-item} uv

```bash
uv add potamides
```

:::

:::{tab-item} source, via pip

```bash
pip install git+https://https://github.com/wsr1998/potamides.git
```

:::

:::{tab-item} building from source

```bash
cd /path/to/parent
git clone https://https://github.com/wsr1998/potamides.git
cd potamides
pip install -e .  # editable mode
```

:::

::::

## Quickstart

### Fitting the Stream

```{code-block} python

>>> import potamides
>>> from potamides import splinelib

>>> num_knots=6
>>> fid_gamma, fid_knots = splinelib.make_increasing_gamma_from_data(xy_centered)
>>> fiducial_spline = interpax.Interpolator1D(fid_gamma, fid_knots, method="cubic2")
>>> ref_gamma = jnp.linspace(fid_gamma.min(), fid_gamma.max(), num=128)
>>> ref_points = fiducial_spline(ref_gamma)
>>> knots = splinelib.optimize_spline_knots(
    splinelib.default_cost_fn,
    fid_knots,
    fid_gamma,
    cost_args=(ref_gamma, ref_points),
    cost_kwargs=ImmutableMap({"concavity_weight": 1e12}),
)
>>> spline = interpax.Interpolator1D(fid_gamma, knots, method="cubic2")
>>> opt_gamma, opt_knots = splinelib.new_gamma_knots_from_spline(
    spline, nknots=num_knots
)
>>> track=ptd.Track(opt_gamma, opt_knots)

```

### visulaize the Stream

```{code-block} python

>>> fig, ax = plt.subplots(figsize=(figsize, figsize),dpi=150)
>>> plot_sparse_gamma = jnp.linspace(track.gamma.min(), track.gamma.max(), num=8)
>>> track.plot_all(plot_sparse_gamma, ax=ax, show_tangents=False)
>>> plt.show()

```

### Initial basic function for calculating the likelihood

```{code-block} python

>>> params_defaults = {
    "rs_halo": 16,  # [kpc]
    "vc_halo": u.Quantity(250, "km/s").ustrip("kpc/Myr"),
    "q1": 1.0,
    "q2": 1.0,
    "q3": 1.0,
    "phi": 0.0,
    "Mdisk": 1.2e10,  # [Msun]
    "origin_x": 0,
    "origin_y": 0,
    "origin_z": 0,
    "rot_z": 0.0,
    "rot_x": 0.0,
}
>>>params_statics = {"withdisk": False}

>>> @jax.jit
>>> def compute_acc_hat(params, pos2d):
    # Positions: 2D -> 3D
    pos3d = jnp.zeros((len(pos2d), 3))
    pos3d = pos3d.at[:, :2].set(pos2d)

    # Accelerations from the potential model
    params = params_defaults | params_statics | params
    params["origin"] = jnp.array(
        [
            params.pop("origin_x", 0),
            params.pop("origin_y", 0),
            params.pop("origin_z", 0),
        ]
    )
    return ptd.compute_accelerations(pos3d, **params)


>>> @jax.jit
>>> def compute_ln_likelihood_scalar(params, pos2d, unit_curvature, where_straight=None):
    unit_acc_xy = compute_acc_hat(params, pos2d)
    where_straight = (
        where_straight
        if where_straight is not None
        else jnp.zeros(len(unit_curvature), dtype=bool)
    )
    return ptd.compute_ln_likelihood(
        unit_curvature, unit_acc_xy, where_straight=where_straight
    ) / len(unit_curvature)


>>> compute_ln_likelihood = jax.vmap(
    compute_ln_likelihood_scalar, in_axes=(0, None, None, None)
)

```

### Calculating the likelihood

```{code-block} python

>>> ranges = {
    "q1": (0.1, 2),
    "phi": (-np.pi/2, np.pi / 2),
}

>>> key = jr.key(0)
>>> skeys = jr.split(key, num=len(ranges))
>>> nsamples = 1_000_000
>>> params = {
    k: jr.uniform(skey, minval=v[0], maxval=v[1], shape=nsamples)
    for skey, (k, v) in zip(skeys, ranges.items(), strict=True)
}

>>> gamma = jnp.linspace(-0.95, 0.95, 128)

>>> lnlik_seg = compute_ln_likelihood(
    params, track(gamma), track.curvature(gamma), None
)

```

### visulaize the likelihood

```{code-block} python

>>> hist2d_kw = {
    "bins": 20,
    "color": "purple",
    "levels": [0.68, 0.95, 0.997],
    "plot_density": True,
    "plot_contours": True,
    "plot_datapoints": False,
}


>>> fig, ax = plt.subplots(figsize=(4, 4))
>>> corner.hist2d(
    params["q1"],
    params["phi"] * 180 / jnp.pi,  # convert to degrees
    weights=np.exp(lnlik_seg - lnlik_seg.max()),
    **hist2d_kw,
)
>>> ax.set_xlabel('q')
>>> ax.set_ylabel('phi')

```

## Citation

...

## Ecosystem

### `potamides`'s Dependencies

- [Equinox][equinox]: one-stop JAX library, for everything that isn't already in
  core JAX.

### `potamides`'s Dependents

- [coordinax][coordinax]: Coordinates in JAX.
- [galax][galax]: Galactic dynamics in JAX.

 <!-- LINKS -->

[coordinax]: https://github.com/GalacticDynamics/coordinax
[equinox]: https://docs.kidger.site/equinox/
[galax]: https://github.com/GalacticDynamics/galax

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
