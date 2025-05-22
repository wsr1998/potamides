---
title: "Potamides: A Python package for stream curvature analysis"
tags:
  - Python
  - astronomy
  - stellar stream
authors:
  - name: Sirui Wu
    orcid: 0009-0003-4675-3622
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Nathaniel Starkman
    orcid: 0000-0003-3954-3291
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Jake Nibauer
    orcid: 0000-0001-8042-5794
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - name: Sarah Pearson
    orcid: 0000-0003-0256-5446
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1

affiliations:
  - name: Niels Bohr Institute, University of Copenhagen, Denmark
    index: 1
    ror: 035b05819
  - name:
      Brinson Prize Fellow at Kavli Institute for Astrophysics and Space
      Research, Massachusetts Institute of Technology, USA
    index: 2
    ror: 042nb2s44
  - name: Department of Physics, Princeton University, USA
    index: 3
    ror: 00hx57361
date: 18 June 2025
bibliography: paper.bib
---

# Summary

# Statement of need

`Potamides` is an Astropy-affiliated Python package for galactic dynamics.
Python enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by students
in courses on gravitational dynamics or astronomy. It has already been used in a
number of scientific publications [@Pearson:2017] and has also been used in
graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the _Gaia_ mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$
\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.
$$

You can also use plain \LaTeX for equations \begin{equation}\label{eq:fourier}
\hat f(\omega) = \int\_{-\infty}^{\infty} f(x) e^{i\omega x} dx \end{equation}
and refer to \autoref{eq:fourier} from text.

## the curvature

## About the Likelihood

We assume that the parameter we want to constrain is $\vec{m}$, which might
including the halo projected flattening (or axis ratio) q, the orientation of
the flattening $\alpha$, and the center of the baryon $x_{cent}$ and $y_{cent}$.

For the stream, we use $\{\vec{\kappa}_i\}$ to represent the data point along
the stream, especially the curvature. It contains N points, which is the sample
points along the track.

There is three different condition for the data point, about the angle $\theta$
which is between the curvature and its local accelerations

- $C_1$: $\theta<90\degree$
- $C_2$: $\theta>90\degree$
- $C_3$: $\theta=90\degree$

If for a given parameters group $\vec{m}$, all the stream track is in accordance
to the $C_1$, then $P(C_j|\vec{m})=1$. This is our base assumption that, the
given acceleration fields will tends to make the angle between curvature and
local acceleration smaller than $90\degree$. If all the stream track is in
$C_2$, that turns out to be $P(C_j|\vec{m})=0$

For the $C_3$, because if the segment of the stream is very straight, the small
change of the track turns out to be completely different orientation of the
curvature. That is the numericiscontinuties may occur. The direction of
curvature at adjacent positions may undergo a change approaching 180 degrees. We
do not wish for such a situation to raise. <span style="color:red">We set a
threshold value, based on the distance, pixel scale, and the angular resolution
of the telescope (or the seeing)</span>.

Here it contains two part of MLE one is using the MLE to get the
$f_j=P(C_j|\vec{m})$, so this require that the N is big enough. The other is
using the MLE to get the likelihood $L({\kappa_i}|\vec{m})$

The

The probability is
$$L(\{\vec{\kappa}_i\}|\vec{m})=\prod\limits_{i}P(\{\vec{\kappa}_i\}|\vec{m})=\prod\limits_{i}P(\{\vec{\kappa}_i\}|\vec{m})$$

The probability is
$$L(\{\vec{\kappa}_i\}|\vec{m})=\prod\limits_{i}P(\vec{\kappa}_i|\vec{m})=\sum\limits_j\prod\limits_{i}P(\vec{\kappa}_i|\vec{m}, C_j)P(C_j|\vec{m})$$

Don't apply the straight condition here.
$$L(\{\vec{\kappa}_i\}|\vec{m})=f_1^{n_1}\times f_2^{n_2}\$$

After log
$$\log \mathcal{L}(\{\vec{\kappa}_i\}|\vec{m})=n_1\log f_1 + n_2\log f_2$$

Using the definition of $f_1=n_1/N$ and $f_2=n_2/N$, we get

$$\log \mathcal{L}(\{\vec{\kappa}_i\}|\vec{m})=n_1\log f_1 + n_2\log f_2=N(f_1\log f_1 + f_2\log f_2)$$

Apply the straight condition here
$$L(\{\vec{\kappa}_i\}|\vec{m})=f_1^{n_1}\times f_2^{n_2}\times f_3^{n_3}\times \prod\limits_{i\in C_3}\mathcal{N}(\theta_T,i)$$

If we have multiple streams, we weight the likelihood by the unit length.
Because different streams, we the total arc length of the stream will be
different. The number of control point is fixed <span style="color:red">Double
check here.</span> . So we can define a unit point length $\Delta l=L/N$, where
L is the arc length of the stream. N is the number of the sampling point on the
stream. So the weight for the log likelihood is
$$\ln \mathcal{L}_{tot}=\sum\limits_{k}\frac{\Delta l_k}{\Delta l_{tot}}\ln\mathcal{L}$$

### Notes for no straightness condition

The equation here will be
$$\log \mathcal{L}(\{\vec{\kappa}_i\}|\vec{m})=n_1\log f_1 + n_2\log f_2$$

we can write it to be a function of $n_1$, that is
$$\log \mathcal{L}(\{\vec{\kappa}_i\}|\vec{m})=n_1\log \frac{n_1}{N} + (1-n_1)\log \frac{1-n_1}{N}=n_1\ln n_1 + (1-n_1)\ln(1-n_1) - \ln N$$

Because we care about the likelihood value compare to the maximum likelilhood,
that is
$$\ln\frac{\mathcal{L}(\{\vec{\kappa}_i\}|\vec{m})}{\mathcal{L}(\{\vec{\kappa}_i\}|\vec{m})_{max}}=\ln \mathcal{L}(\{\vec{\kappa}_i\}|\vec{m})-\ln\mathcal{L}(\{\vec{\kappa}_i\}|\vec{m})_{max}=n_1\ln n_1 + (1-n_1)\ln(1-n_1) - n'_1\ln n'_1 + (1-n'_1)\ln(1-n'_1) \$$

Where $n'_1$ is when the parameter group $\vec{m}$ can maximize the likelihood.
Here the result is independent of the number of the sampling points N.

<span style="color:red">So what is the best way to combine different streams
likelihood?</span>

## potential and acceleration

### typical acceleration fields (q, $\theta$)

<span style="color:red">Write down how to get acceleration field from the
potential</span>.

$$\Phi_{\rm halo}=v^2_{\rm halo} \ln(r^2_{\rm halo}+C_1x^2+C_2y^2+C_3xy+(\frac{z}{q_z})^2)$$

where
$$C_1 = 1/\left(\frac{\cos^2\phi}{q_1^2} + \frac{\sin^2\phi}{q_2^2}\right)$$
$$C_2 = 1/\left(\frac{\cos^2\phi}{q_2^2} + \frac{\sin^2\phi}{q_1^2}\right)$$
$$C_3 = 2\sin\phi\cos\phi \left(\frac{1}{q_1^2} - \frac{1}{q_2^2}\right) $$

Now the 2D potential we are using is
$$\Phi_{\rm halo}=v^2_{\rm halo}\ln(r^2_{\rm halo}+C_1x^2+C^2y^2+C_3xy)$$

we let $D=r^2_{\rm halo}+C_1x^2+C^2y^2+C_3xy$. We know that
$\mathbf{a}=-\nabla \Phi=-v^2_{\rm halo} \nabla\ln D=-v^2_{\rm halo} \nabla D/D$,
where $\nabla D=(2C_1x+C_3y, 2C_2y+C_3x)$.

So the planner acceleration is
$\mathbf{a}=-\frac{v^2_{\rm halo}}{D}(2C_1x+C_3y, 2C_2y+C_3x)$

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without
a preferred citation) then you can do it with the example BibTeX entry below for
@fidgit.

For a quick reference, the following citation commands can be used:

- `@author:2001` -> "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al.,
  2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png) and referenced
from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

This work was supported by a research grant (VIL53081) from VILLUM FONDEN. Thus
work was also co-funded by the European Union (ERC, BeyondSTREAMS, 101115754)
grant. Views and opinions expressed are however those of the author(s) only and
do not necessarily reflect those of the European Union or the European Research
Council. Neither the European Union nor the granting authority can be held
responsible for them. The Tycho supercomputer hosted at the SCIENCE HPC center
at the University of Copenhagen was used for supporting this work.

# References
