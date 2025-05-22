# 2D inference

```{code block} python
>>> import corner
>>> ranges = {
>>>     "q1": (0.1, 1),
>>>     "phi": (-np.pi/2, np.pi / 2),
>>> }

>>> key = jr.key(0)
>>> skeys = jr.split(key, num=len(ranges))
>>> nsamples = 1_000_000
>>> params = {
>>>     k: jr.uniform(skey, minval=v[0], maxval=v[1], shape=nsamples)
>>>     for skey, (k, v) in zip(skeys, ranges.items(), strict=True)
>>> }
>>> params

>>> gamma = jnp.linspace(-0.95, 0.95, 128)
>>>
>>> lnlik_seg = compute_ln_likelihood(
>>>     params, track(gamma), track.curvature(gamma), None
>>> )

>>> hist2d_kw = {
>>>     "bins": 20,
>>>     "color": "purple",
>>>     "levels": [0.68, 0.95, 0.997],
>>>     "plot_density": True,
>>>     "plot_contours": True,
>>>     "plot_datapoints": False,
>>> }

>>> fig, ax = plt.subplots(figsize=(4, 4))
>>> corner.hist2d(
>>>     params["q1"],
>>>     params["phi"] * 180 / jnp.pi,  # convert to degrees
>>>     weights=np.exp(lnlik_seg - lnlik_seg.max()),
>>>     **hist2d_kw,
>>> )
>>> ax.set_xlabel('q')
>>> ax.set_ylabel('phi')

```
