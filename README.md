# Overview

Repository contains files related to the article
[Hyperuniformity variation with quasicrystal local isomorphism class](http://iopscience.iop.org/article/10.1088/1361-648X/aa6944/meta)
including:
1. A vectorized implementation of the [generalized dual method](http://www.physics.princeton.edu/~steinh/ArbSymQC.pdf) for generating 2D direct projection tilings (quasicrystals); and
2. A parallelized implementation for estimating the [number variance](https://arxiv.org/abs/cond-mat/0311532)

To operate, run
```
$ python main.py gammasum seed kmax num_windows num_intervals
```
where
`gammasum` is the LI class label
`seed` is the RNG seed
`kmax` is the maximum index for each grid
`num_windows` is the number of windows to use when estimating the number variance N(r)
`num_intervals` is the number of radii to use in the argument for N(r)
