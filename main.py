#!/usr/bin/python
# Parallelized implementation of number variance calculation
# for 2D direct project tilings

# Author: Chaney Lin
# April 2018

import sys

if len(sys.argv) != 6:
    print 'too many args: want arguments in the form [gammasum, seed, kmax, windows, intervals]'
    sys.exit()

import dualmethod, numbervariance
import numpy as np
import gc

from time import time
from multiprocessing import cpu_count

if __name__ == "__main__":

    # STEP 0: PROCESS USER INPUTS

    gammasum = (float)(sys.argv[1])
    seed = (int)(sys.argv[2])
    kmax = (int)(sys.argv[3])
    num_windows = (int)(sys.argv[4])
    num_intervals = (int)(sys.argv[5])
    kmin = -kmax

    # print inputs
    print '\ngammasum, seed, kmax, numwindows, numintervals', sys.argv[1:]
    print 'beginning..............................'
    
    print 'gammasum  ', gammasum
    print 'seed      ', seed
    print 'kmax      ', kmax

    # STEP 1: CONSTRUCT QC TILING

    dg = dualmethod.DualizeGridParallel(kmin,kmax,gammasum,seed = seed)
    pts = dualmethod.TrimDualGrid(kmin,kmax,dg)
    
    gc.collect() # clear contents of dg , pts -- just in case

    print 'finished constructing tiling'

    # STEP 2: CALCULATE NUMBER VARIANCE

    s = time()

    maxrad = max(pts[:,0])*.95
    tau = (1+5**.5) / 2 # golden ratio
    rho = 2*(tau+1) / (tau+2) / (3-tau)**.5 # density of infinite tiling
    #rho = N / np.pi / maxrad**2 # density of finite tiling
    N = len(np.where(np.linalg.norm(pts,axis=1)<maxrad)[0])

    centers = numbervariance.WindowCenters(maxrad/2, num_windows) # the centers must be contained in a circular region of maxrad/2

    print maxrad
    maxrad = float(kmax*2)
    dr = maxrad / num_intervals
    #rlist = np.linspace(0,maxrad/2,num_intervals)
    rlist = np.arange(dr, maxrad + dr, dr)

    # specify number of cores for parallelization

    ##### uncomment if running on Princeton cluster
    #from os import environ
    #num_cores = int(environ['SLURM_NTASKS_PER_NODE']) 

    num_cores = cpu_count() # use this if running on local machine
    print 'num cores', num_cores, type(num_cores)

    var = numbervariance.CalculateNumberVariance(rho = rho, centers = centers, pts = pts, rlist = rlist, num_cores = num_cores)

    print 'total', time() - s, 'sec'

    print 'done computing number variance'

    # STEP 3: OUTPUT NUMBER VARIANCE

    params = 'g_{:0.2f}_s_{:d}_k_{:d}'.format(gammasum,seed,kmax)

    fout = ('var_' + params +
            '_win_{:d}_int_{:d}_N_{:d}'.format(num_windows, num_intervals, N))
    
    np.savez(fout + '_lin_rmax_{:d}'.format(int(maxrad)),var=var, rho=rho, N=N,
        num_windows=num_windows, num_intervals=num_intervals, rlist=rlist, seed=seed,
        gammasum=gammasum,kmax=kmax)