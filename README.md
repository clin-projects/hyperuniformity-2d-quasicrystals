# Overview

Repository contains supporting files for the article

[Lin et al. "Hyperuniformity variation with quasicrystal local isomorphism class" (Journal of Physics: Condensed Matter, 2017)](http://iopscience.iop.org/article/10.1088/1361-648X/aa6944/meta)

including:

1. A [**vectorized** implementation](./dualmethod.py) of the [generalized dual method](http://www.physics.princeton.edu/~steinh/ArbSymQC.pdf) for generating 2D direct projection tilings (quasicrystals); and

2. A [**parallelized** implementation](./numbervariance.py) for estimating the [number variance](https://arxiv.org/abs/cond-mat/0311532) (used for estimating the _degree of hyperuniformity_)

The main file can be run using
```
$ python main.py gammasum seed kmax num_windows num_intervals
```
where

`gammasum` is the LI class label

`seed` is the RNG seed

`kmax` is the maximum index for each grid

`num_windows` is the number of windows to use when estimating the number variance N(r)

`num_intervals` is the number of radii to use in the argument for N(r)

# Other files

* [**CalculateLocalMin**](./Other%20files/CalculateLocalMin/): computing the _degree of hyperuniformity_ for LI classes around gammasum = the golden ratio

* [**CalculateVertex**](./Other%20files/CalculateVertex/): computing the frequency of _vertex environments_ for all LI classes, reproducing Fig. 7 of [Zobetz-Preisinger "Vertex frequencies in generalized Penrose patterns" (Acta Cryst., 1990)](http://scripts.iucr.org/cgi-bin/paper?S0108767390008479)

* [**CalculateVoronoi**](./Other%20files/CalculateVoronoi/): computing the standard deviation of the _Voronoi areas_
* [**MakeTiles**](./Other%20files/MakeTiles/): Mathematica code for generating figures of the vertex environments
* [**data**](./dat/): number variance for various LI classes already computed (in `.npz` format)
