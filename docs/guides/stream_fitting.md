# Stream_fitting

First, we need to change the data from numpy to jax numpy

```{code-block} python
>>> import jax
>>> jax.config.update("jax_enable_x64", True)  # noqa: FBT003
>>> import jax.numpy as jnp
>>> x_jax=jnp.array(x_cent,dtype=jnp.float64)
>>> y_jax=jnp.array(y_cent,dtype=jnp.float64)
>>> xy_centered = jnp.stack([x_jax, y_jax], axis=1)
```

Here, `num_knots` is a key parameter controlling the spline fit, inndicating how
many knots the spline is composed of.

```{code-block} python
>>> from potamides import splinelib as splib
>>> import interpax
>>> fid_gamma, fid_knots = splib.make_increasing_gamma_from_data(xy_centered)
>>> fiducial_spline = interpax.Interpolator1D(fid_gamma, fid_knots, method="cubic2")
>>> ref_gamma = jnp.linspace(fid_gamma.min(), fid_gamma.max(), num=128)
>>> ref_points = fiducial_spline(ref_gamma)
```

Result of the preliminary spline fit

```{code-block} python
>>> plt.figure(figsize=(5,5))
>>> plt.plot(X,Y,'.')
>>> plt.plot(0,0,'r*')
>>> plt.plot(ref_points[:,0],ref_points[:,1],'.')
>>> plt.xlim(-40,40)
>>> plt.ylim(-40,40)
>>> plt.grid()
>>> plt.show()
```

In this step, we reorganized the gamma parameter along with the optimized nodes,
with the goal of transforming gamma into a linear parameter with respect to arc
length

```{code-block} python
>>> from xmmutablemap import IMMutableMap
>>> knots = splib.optimize_spline_knots(
>>>     splib.default_cost_fn,
>>>     fid_knots,
>>>     fid_gamma,
>>>     cost_args=(ref_gamma, ref_points),
>>>     cost_kwargs=ImmutableMap({"concavity_weight": 1e12}),
>>> )

>>> # Create a spline from the optimized knots.
>>> spline = interpax.Interpolator1D(fid_gamma, knots, method="cubic2")

>>> # Create a new gamma, proportional to the arc-length from the spline.
>>> # arclength.
>>> opt_gamma, opt_knots = splib.new_gamma_knots_from_spline(
>>>     spline, nknots=num_knots
>>> )

>>> track=ptd.Track(opt_gamma, opt_knots)
```

visualize the result

```{code-block} python
>>> axlim=50
>>> figsize=5
>>> fig, ax = plt.subplots(figsize=(figsize, figsize),dpi=150)
>>> plt.plot(X,Y,'c.',zorder=0)
>>> plt.plot(0,0,'r*')
>>> plt.plot(x_cent,y_cent,'o',color='orange')
>>> plot_sparse_gamma = jnp.linspace(track.gamma.min(), track.gamma.max(), num=8)
>>> track.plot_all(plot_sparse_gamma, ax=ax, show_tangents=False)
>>> ax.set_xlabel("X (pixel)")
>>> ax.set_ylabel("Y (pixel)")
>>> ax.set_xlim(-axlim,axlim)
>>> ax.set_ylim(-axlim,axlim)
>>> fig.tight_layout()
>>> plt.show()
```
