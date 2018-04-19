#!/usr/bin/python
# Vectorized implementation of dual grid method for 2D pentagonal direct projection tilings.
#
# For background on the dual grid method, please refer to the following paper:
# Socolar, Steinhardt, and Levine, "Quasicrystals with arbitrary orientational symmetry" (PRB, 1985)
#
# Author: Chaney Lin
# Date: April 2018

import numpy as np
import random as random
from time import time

def unique_rows(A):
    """
    returns unique rows in A
    """
    return np.vstack({tuple(row) for row in A})

def StarVectorsExact():
    """
    The pentagonal star vectors are unit vectors that point to the five corners of a pentagon.
    The direction of the orthogonal vectors is designed for the dual grid algorithm implemented here

    Returns:
    a           -- pentagonal star vectors
    aorth       -- unit vectors orthogonal to the star vectors
    """
    a = [[np.cos(2*np.pi / 5 * n), np.sin(2*np.pi / 5 * n)] for n in range(5)]
    aorth = np.array([[-x[1],x[0]] for x in a])
    return np.array(a), aorth

def DualizeGridParallel(kmin,kmax,gammasum,nj=5,seed=1):
    """
    Vectorized implementation of dual grid method that returns points
    Input:
    kmin        -- minimum index for the gridlines
    kmax        -- maximum index for the gridlines [each grid will have (kmax - kmin + 1) gridlines]
    gammasum    -- labels the LI class [distinct classes from 0 to 0.5]
    seed        -- seed for RNG

    Returns:
    tiles       -- 
    tilepoints  -- the indices that correspond to valid vertices, without duplicates
    """

    random.seed(seed)
    
    def step(i, nj=5):
        return np.array([i==j for j in range(nj)])

    def cut_parallel(tij):
        return np.maximum(np.minimum(tij,kmax+1),kmin)

    start_time = time()
    r, rorth = StarVectorsExact()
 
    # shift vectors
    gam = [random.uniform(-0.5,0.5) for x in range(nj-1)]
    gam.append(gammasum-sum(gam))
    gam = np.array(gam)

    # stack the grid labels -- necessary step for vectorization
    k = np.tile(np.array(range(kmin, kmax+1)),(nj,1)) # repeats it
    l = k - np.array([gam]).transpose()



    lro = np.einsum('ia,jv->iajv',l,rorth)

    rorinv = 1. / np.einsum('iv,jv->ij',rorth,r)

    x1 = np.einsum('ij,jaiv->jiav',rorinv,lro)
    x2 = np.einsum('ij,iajv->jiav',rorinv,lro)

    x1 = np.swapaxes(x1,0,2)
    x2 = np.swapaxes(x2,0,2)
    x = x1[:,None,:,:,:] - x2

    t0 = np.einsum('abjiv,kv->abijk',x,r) + gam

    t = np.ceil(t0) #round to nearest line
    
    tiles = []

    dist = 4
    mult = 2*kmax+1
    start_index = np.array(range(dist*mult))
    new_index = (start_index%dist)*mult + start_index/dist

    fix_index_matrix = np.tile(np.array(range(kmin,kmax+1)),(2*kmax+1,1)).transpose()

    for i in range(nj-1):
        for j in range(i+1,nj):
            tij = t[:,:,i,j,:]
            tij[:,:,i] = fix_index_matrix
            tij[:,:,j] = fix_index_matrix.transpose()
            si = step(i)
            sj = step(j)
            tij_cut = np.concatenate((
                                cut_parallel(tij), cut_parallel(tij+si), 
                                cut_parallel(tij+si+sj),cut_parallel(tij+sj)
                                ),axis=1)
            #reindexing
            for k in range(mult):
                tij_cut[k,:] = tij_cut[k,:][new_index]
            tiles.append(tij_cut)

    tiles = np.concatenate(tiles)

    dim = tiles.shape

    tiles = tiles.reshape((dim[0]*dim[1]/dist,dist,nj))

    end_time = time()

    print 'calculate tiles       ', end_time - start_time, 'sec'

    rows = [j for i in tiles for j in i]

    tilepoints = unique_rows(rows)
    
    print 'calculate tilepoints  ', time() - end_time, 'sec'
    
    return tiles, tilepoints

def TrimDualGrid(kmin,kmax,dg):
    """
    Return the coordinates of the vertices after removing surface tiles

    Inputs:
    kmin        -- minimum index for the gridlines
    kmax        -- maximum index for the gridlines [each grid will have (kmax - kmin + 1) gridlines]
    dg          -- output of DualizeGridParallel(), of the form [tiles, tilepoints]

    Returns:
    dualtilepoints -- the coordinates of the vertices, without surface tiles
    """
    
    tiles, tilepoints = dg
    
    surfacetiles = np.logical_or(tilepoints>=kmax, tilepoints<=kmin) # these are tiles that lie at or near the surface
    
    goodrows = np.where(np.invert(np.any(surfacetiles,axis=1)))[0] # these are the indices for tiles that are not surface tiles
    
    dualtilepoints = np.dot(tilepoints[goodrows], StarVectorsExact()[0]) # this returns the coordinates of the non-surface tiles

    print 'number of points      ', len(dualtilepoints)
    return dualtilepoints