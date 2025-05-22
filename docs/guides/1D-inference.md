# 1D-inference

```{code block} python
>>> import potamides as ptd
>>> import unxt as u
>>> import jax.random as jr

>>>@jax.jit
>>>def compute_acc_hat(params, pos2d):
>>>    # Positions: 2D -> 3D
>>>    pos3d = jnp.zeros((len(pos2d), 3))
>>>    pos3d = pos3d.at[:, :2].set(pos2d)
>>>
>>>    # Accelerations from the potential model
>>>    params = params_defaults | params_statics | params
>>>    params["origin"] = jnp.array(
>>>        [
>>>            params.pop("origin_x", 0),
>>>            params.pop("origin_y", 0),
>>>            params.pop("origin_z", 0),
>>>        ]
>>>    )
>>>    return ptd.compute_accelerations(pos3d, **params)

>>> @jax.jit
>>> def compute_ln_likelihood_scalar(params, pos2d, unit_curvature, where_straight=None):
>>>     unit_acc_xy = compute_acc_hat(params, pos2d)
>>>     where_straight = (
>>>         where_straight
>>>         if where_straight is not None
>>>         else jnp.zeros(len(unit_curvature), dtype=bool)
>>>     )
>>>     return ptd.compute_ln_likelihood(
>>>         unit_curvature, unit_acc_xy, where_straight=where_straight
>>>     ) / len(unit_curvature)

>>> # parameters for the halo
>>> params_defaults = {
>>>     "rs_halo": 16,  # [kpc]
>>>     "vc_halo": u.Quantity(250, "km/s").ustrip("kpc/Myr"),
>>>     "q1": 1.0,
>>>     "q2": 1.0,
>>>     "q3": 1.0,
>>>     "phi": 0.0,
>>>     "Mdisk": 1.2e10,  # [Msun]
>>>     "origin_x": 0,
>>>     "origin_y": 0,
>>>     "origin_z": 0,
>>>     "rot_z": 0.0,
>>>     "rot_x": 0.0,
>>> }
>>> params_statics = {"withdisk": False}


>>> compute_ln_likelihood = jax.vmap(
>>>     compute_ln_likelihood_scalar, in_axes=(0, None, None, None)
>>> )

>>> anges = {
>>>     "q1": (0.1, 2),
>>> }

>>> key = jr.key(0)
>>> skeys = jr.split(key, num=len(ranges))
>>> nsamples = 1_000
>>> params = {
>>>     k: jr.uniform(skey, minval=v[0], maxval=v[1], shape=nsamples)
>>>     for skey, (k, v) in zip(skeys, ranges.items(), strict=True)
>>> }

>>> gamma = jnp.linspace(-0.95, 0.95, 128)

>>> lnlik_seg = compute_ln_likelihood(
>>>     params, track(gamma), track.curvature(gamma), None
>>> )

```
