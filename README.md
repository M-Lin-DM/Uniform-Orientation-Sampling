## Overview
<img src="figures/N3125zoom.png" width="250" title="N3125" alt="N3125" align="right" vspace = "0">

Generating evenly distributed points on the surface of a hypersphere of radius 1. 

The function `UniformOrientationSampling.run_optimizer()` in `Uniform_Orientation_Sampling.py` finds a set of `pop_size` points on the surface of a hypersphere in a space of `dimensions` dimensions. The result is obtained using an particle simulation-based optimization process that encourages all vectors to be equidistant from their nearest neighbors. The final embedding consists in unit vectors that uniformly and efficiently sample orientations in the space. 

<img src="figures/N625.PNG" width="1000" title="Comparison at 625 pop_size" alt="comparison N625" vspace = "50">

*Fig. (left) 625 orientation vectors in 3D sampled using a naive approach. These points were obtained by uniformly sampling (x, y, z) coordinates in the range [-0.5, 0.5] and then projecting all points onto the sphere by normalization to length 1. (right) Vectors obtained by the "KNN repulsion" algorithm used in UniformOrientationSampling. Only the top half of points was plotted for clarity.*

This is useful in applications where you are searching for optima in a continuous space. Certain gradient-decent-based algorithms require evaluating a cost function at a local set of points surrounding the current-best location. This current location would then be updated by moving towards the local point with the minimum cost value. In dimensions greater than 2, it is not obvious how to optimally sample a set of local test points. The algorithm developed here offers one solution. 

One obvious, projection-based, approach (see Fig left) leads to an inefficient sampling of directions due to the heterogeneous density of points. For example, there are typically several near-redundant points.

<img src="figures/comparison_over_N.PNG" width="1000" title="Comparison over Number of vectors" alt="comparison over N" vspace = "50">
