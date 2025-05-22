---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# potamides

```{toctree}
:maxdepth: 2
:hidden:
:caption: Guides

guides/begin.md
guides/stream_fitting.md
guides/1D-inference.md
guides/2D-inference.md
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

The `potamides` package provides tools for fitting stellar streams with splines
and inferring the gravitational potential from the stream's curvature and
acceleration. The high-level workflow uses the object-oriented interface of the
`potamides.Track` class. Lower-level functions for spline fitting, etc. can be
found in the `potamides.splinelib` module.

```{code-cell} ipython3
import potamides as ptd
import potamides.splinelib as splib
```

We will also assume the following imports for computing and visualization:

```{code-cell} ipython3
import functools as ft
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import unxt as u
```

### Fitting a 'Stream'

Let's create a mock stream in 2D (an oval) and fit it with a spline. In
practice, you would replace this with your actual stream data.

```{code-cell} ipython3
a, b = 2.0, 1.2  # semi-axes (x, y)
gamma = jnp.linspace(0, 2 * jnp.pi, 10_000)
# # Ellipse (oval)
# data = jnp.stack([a * jnp.cos(gamma), b * jnp.sin(gamma)], axis=-1)
# --- Optional: Egg-shape (uncomment to use) ---
k = 0.25  # distortion factor; 0 → pure ellipse
x = a * jnp.cos(gamma) * (1 + k * jnp.cos(gamma))
y = b * jnp.sin(gamma)
data = jnp.stack([x, y], axis=-1)
```

Then we make the stream track. We choose 25 knots for the spline fit. For
demonstration purposes we'll set the knot position directly from the data.

```{code-cell} ipython3
track = ptd.Track(gamma[::400], data[::400])
```

Now we can visualize the 'stream':

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
_gamma = jnp.linspace(track.gamma.min(), track.gamma.max(), num=8)  # for plotting only
track.plot_all(_gamma, ax=ax, show_tangents=False)
plt.show()
```

### Initial basic function for calculating the likelihood

```{code-cell} ipython3

params_defaults = {
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
params_statics = {"withdisk": False}

@jax.jit
def compute_acc_hat(p, /, pos2d):
   # Positions: 2D -> 3D
   pos3d = jnp.zeros((len(pos2d), 3))
   pos3d = pos3d.at[:, :2].set(pos2d)
   # Accelerations from the potential model
   p = params_defaults | params_statics | p
   p["origin"] = jnp.array([p.pop("origin_x", 0), p.pop("origin_y", 0), p.pop("origin_z", 0)])
   return ptd.compute_accelerations(pos3d, **p)


@ft.partial(jax.vmap, in_axes=(0, None, None))
@jax.jit
def compute_ln_likelihood(p, /, pos2d, unit_kappa):
   unit_acc_xy = compute_acc_hat(p, pos2d)
   where_straight = jnp.zeros(len(unit_kappa), dtype=bool)
   return ptd.compute_ln_likelihood(unit_kappa, unit_acc_xy, where_straight) / len(unit_kappa)

```

### Calculating the likelihood

```{code-cell} ipython3

ranges = {"q2": (0.1, 2), "phi": (-jnp.pi/2, jnp.pi / 2)}

key, *skeys = jr.split(jr.key(0), num=len(ranges) + 1)
nsamples = 1_000_000
params = {k: jr.uniform(skey, minval=v[0], maxval=v[1], shape=nsamples)
          for skey, (k, v) in zip(skeys, ranges.items(), strict=True)}

gamma = jnp.linspace(-0.95, 0.95, 128)

lnlik_seg = compute_ln_likelihood(params, track(gamma), track.curvature(gamma))

```

Now we can visualize the inference result in 2D:

```{code-cell} ipython3
import corner

hist2d_kw = {
    "bins": 20,
    "color": "purple",
    "levels": [0.68, 0.95, 0.997],
    "plot_density": True,
    "plot_contours": False,
    "plot_datapoints": False,
}

fig, ax = plt.subplots(figsize=(4, 4))
lnlik_seg = jnp.nan_to_num(lnlik_seg, neginf=jnp.nanmin(lnlik_seg) - 10)
corner.hist2d(
    params["q2"],
    jnp.rad2deg(params["phi"]),
    weights=jnp.exp(lnlik_seg - lnlik_seg.max()),
    **hist2d_kw,
)
ax.set_xlabel('q')
ax.set_ylabel('phi')
fig
```

## Citation

...

## Ecosystem

### `potamides`'s Dependencies

- [Equinox][equinox]: one-stop JAX library, for everything that isn't already in
  core JAX.

 <!-- LINKS -->

[coordinax]: https://github.com/GalacticDynamics/coordinax
[equinox]: https://docs.kidger.site/equinox/
[galax]: https://github.com/GalacticDynamics/galax

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
