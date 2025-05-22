#

Here is the test using the stream from Jake's paper.

The basic work flow is stream fitting, then

## Stream visualization

The stream is 3D. We use only 2D (x-z), for plotting we name it as (x-y)

```{doctest}
:group: stream
>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> name='fig5_streamB.npy'
>>> file_dir='enter_your_file_dir' # enter your own file_dir
>>> data=np.load(file_dir+name,allow_pickle='True')

>>> x=data.item()['x']
>>> y=data.item()['z']

>>> plt.figure(figsize=(5,5))
>>> plt.plot(x,y,'.')

```

Here need to maybe upload the plot of the result.

## get the representative points along the stream

In this step, we providing solution that doing a **median trajectory** for the
stream. As mentioned in the paper (?), our code

From the plot, we can see that the stream exhibits a long, gradually narrowing
tail (?) extending towards the right, while the left side shos a distinct
U-shaped structure. At the U-shaped region, the stream undergoes a pronounced
turn, with stars appearing more densely concentrated in this section.

Because the straight segment is hard to calculate the curvature, and if we set
the curvature to 0, this part will be no contribution to the inference. So we
skip the part

```{doctest}
>>> X, Y = x, y

>>> # Original angles in the range (-π, π]
>>> angles = np.arctan2(Y, X)

>>> # ===== 1. Define starting angle =====
>>> angle_start = -np.pi/2          # Bin #1 lower boundary set to -π/2
>>> total_span  = 2*np.pi           # Full circular coverage

>>> # ===== 2. Shift angles into [0, 2π) =====
>>> angles_shifted = (angles - angle_start) % total_span

>>> # ===== 3. Construct equally spaced bins =====
>>> n_bins_angle = 100
>>> angle_bins = np.linspace(0, total_span, n_bins_angle + 1)

>>> # ===== 4. Assign points to bins =====
>>> angle_idx = np.digitize(angles_shifted, angle_bins, right=False)   # Range: 1…n_bins_angle
>>> radius    = np.hypot(X, Y)
>>> radius_idx = np.ones_like(angle_idx)   # Only one radial bin used

>>> # ===== 5. Group points by (angle_bin, radius_bin) =====
>>> bins = list(zip(angle_idx, radius_idx))

>>> binned = {}
>>> for (ab, rb), x_val, y_val in zip(bins, X, Y):
>>>     binned.setdefault((ab, rb), []).append((x_val, y_val))

>>> # ===== 6. Compute medians and sort by angle bin =====
>>> median_x_ls, median_y_ls = [], []
>>> sorted_bins = sorted(binned.keys(), key=lambda k: k[0])   # Sort by angular order
>>> for b in sorted_bins:
>>>     pts = np.asarray(binned[b])
>>>     median_x_ls.append(np.median(pts[:, 0]))
>>>     median_y_ls.append(np.median(pts[:, 1]))

```

The result we get is here: (In this step, we did not select all the median
trajectories. Instead, following the annotation in the paper, we only chose a
subset, corresponding to the operation `[42:-4]` here.)

```{code-block} python
>>> x_cent=np.array(median_x_ls[42:-4])
>>> y_cent=np.array(median_y_ls[42:-4])
>>> plt.figure(figsize=(5,5))
>>> plt.plot(X,Y,'.')
>>> plt.plot(0,0,'r*')
>>> plt.plot(x_cent,y_cent,'.')
>>> plt.xlim(-40,40)
>>> plt.ylim(-40,40)
>>> plt.grid()
>>> plt.show()
```

(adding the plot)
