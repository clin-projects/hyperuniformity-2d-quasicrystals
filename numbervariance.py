#!/usr/bin/python
# Parallelized implementation for Monte Carlo estimation of number variance
#
# For background on the number variance, please refer to the early sections of following paper:
# Torquato and Stillinger, "Local density fluctuations, hyperuniformity, and order metrics" (PRE, 2003)
#
# Author: Chaney Lin
# Date: April 2018

import numpy as np
import random as random
from time import time
from joblib import Parallel, delayed

np.seterr(divide='ignore')
np.seterr(invalid='ignore')

def WindowCenters(r, num_windows):
    """
    Creates a uniform grid of window centers within a central, circular region,
    such that windows do not extend outside a circular region of radius r

    Inputs:
    maxrad      --  radius of tiling 
    num_windows --  number of window centers

    Returns:
    centers     --  (x,y) coordinates of window centers, uniformly distributed in 
                    circle of radius maxrad/2
    """
    x = np.sqrt(np.pi * float((r)**2) / float(num_windows))
    n = int(np.sqrt(num_windows))
    a = np.linspace(-n * x,n *x,2*n+1)
    grid = np.array([(i,j) for i in a for j in a])
    dist = np.linalg.norm(grid,axis=1)
    centers = grid[np.where(dist <= r)[0]]
    return centers

def RunCounts(centers, pts, rlist, i):
    """
    returns a list of N(r), the number of points located in a circle of radius r around
    the point centers[i]

    Returns:
    list containing N(r)
    """
    y = centers[i]
    dist = np.sort(np.ascontiguousarray(np.linalg.norm(np.ascontiguousarray(pts-y),axis=1))) # trim zero distance
    return [np.searchsorted(dist,rlist)]

def CalculateNumberVariance(rho, centers, pts, rlist, num_cores = 1):
    """
    Implements estimation of the number variance using a "Monte Carlo" method
    where the number of points contained within a 

    Inputs:
    centers         -- list of window centers
    pts             -- the full set of vertices
    rlist           -- the set of window radii r over which to calculate N(r), the number of points within
                        a window of radius r

    Returns:
    var             -- list containing number variance (as function of rlist)
    """    

    n = len(centers)
    counts = Parallel(n_jobs = num_cores)(delayed(RunCounts)(centers, pts, rlist, i) for i in range(n))
    counts = np.concatenate(counts)
    expected_counts = rho*np.pi*rlist**2
    n = counts - expected_counts # removing the constant value make it more efficient
    n2 = n**2
    var = np.mean(n2,axis=0) - np.mean(n,axis=0)**2

    return var