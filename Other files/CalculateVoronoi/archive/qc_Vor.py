#!/usr/bin/python

import sys

if len(sys.argv) != 4:
    print 'too many args: want arguments in the form [gammasum, seed, kmax]'
    sys.exit()
else:
    g = (float)(sys.argv[1])
    seed = (int)(sys.argv[2])
    kmax = (int)(sys.argv[3])
    kmin = -kmax

from scipy.spatial import Voronoi
import gc
import numpy as np
import random as random
from time import time
from joblib import Parallel, delayed
import os

num_cores = int(os.environ['SLURM_NTASKS_PER_NODE'])
#num_cores = 4
print 'num cores', num_cores, type(num_cores)

np.seterr(divide='ignore')
np.seterr(invalid='ignore')

print '\ngammasum, seed, kmax, numwindows, numintervals', sys.argv[1:]
print 'beginning..............................'
print 'kmax      ', kmax
print 'gammasum  ', g
print 'seed      ', seed

nj = 5

def unique_rows(a):
    return np.vstack({tuple(row) for row in a})

def StarVectorsExact():
    a = [[np.cos(2*np.pi / 5 * n), np.sin(2*np.pi / 5 * n)] for n in range(5)]
    aorth = np.array([[-x[1],x[0]] for x in a])        
    return np.array(a), aorth

def DualizeGridParallel(kmin,kmax,gammasum,nj=5,seed=1):

    random.seed(seed)
    
    def step(i, nj=5):
        return np.array([i==j for j in range(nj)])

    def cut_parallel(tij):
        return np.maximum(np.minimum(tij,kmax+1),kmin)

    start_time = time()
    r, rorth = StarVectorsExact()
 
    gam = [random.uniform(-0.5,0.5) for x in range(nj-1)]
    gam.append(gammasum-sum(gam))
    gam = np.array(gam)

    k = np.tile(np.array(range(kmin, kmax+1)),(nj,1)) # repeats it
    l = k - np.array([gam]).transpose()

    lro = np.einsum('ia,jv->iajv',l,rorth) # good

    rorinv = 1. / np.einsum('iv,jv->ij',rorth,r)

    x1 = np.einsum('ij,jaiv->jiav',rorinv,lro)
    x2 = np.einsum('ij,iajv->jiav',rorinv,lro)

    x1 = np.swapaxes(x1,0,2)
    x2 = np.swapaxes(x2,0,2)
    x = x1[:,None,:,:,:] - x2

    t0 = np.einsum('abjiv,kv->abijk',x,r) + gam

    t = np.ceil(t0) #round to nearest line
    
    #works up to here...abijk

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

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def PolyAreaPt(r):
    return PolyArea(r[:,0],r[:,1])

def AddPolyAreas(i):
    neighbors = neighborlist[i]
    temp = [vor.vertices[vor.regions[vor.point_region[x]]] for x in neighbors]
    return sum([PolyAreaPt(r) for r in temp])



############################################################
##### main code ####
   
tiles,tilepoints = DualizeGridParallel(kmin,kmax,g, nj, seed)
rad = 2*kmax
tilepoints = np.dot(tilepoints, StarVectorsExact()[0])
dist = np.linalg.norm(tilepoints, axis=1)
pts_ind = np.where(dist<rad/2)[0]

vor = Voronoi(tilepoints)

gc.collect()  

# First Nearest Neighbor Voronoi

print 'calculating first nearest neighbor voronoi...',

s = time()

temp = [vor.vertices[vor.regions[vor.point_region[x]]] for x in pts_ind]
areas = [PolyAreaPt(r) for r in temp]

round_areas = np.round(areas,10)
n = len(round_areas)
un_areas = np.unique(round_areas)
cts = []
freq = []
for j in un_areas:
    counts = len(np.where(round_areas == j)[0])
    cts.append(counts)
    freq.append(float(counts) / n * 100)
fout='./vor1_g_{:0.4f}_s{:d}_k_{:d}'.format(g,seed,kmax)
np.savez(fout,cts=cts,freq=freq,areas=un_areas,g=g,s=seed,k=kmax)

print 'Done', time() - s, 'sec'
# Second Nearest Neighbor Voronoi

print 'calculating second nearest neighbor voronoi...',

s - time()

def GetNeighbors(i):
    return list(set(np.concatenate(vor.ridge_points[np.where(vor.ridge_points==i)[0]])))

neighborlist = Parallel(n_jobs=num_cores)(delayed(GetNeighbors)(i) for i in pts_ind)
areas = [AddPolyAreas(i) for i in range(len(pts_ind))]

round_areas = np.round(areas,10)
n = len(round_areas)
un_areas = np.unique(round_areas)
cts = []
freq = []
for j in un_areas:
    counts = len(np.where(round_areas == j)[0])
    cts.append(counts)
    freq.append(float(counts) / n * 100)
fout='./vor2_g_{:0.4f}_s{:d}_k_{:d}'.format(g,seed,kmax)
np.savez(fout,cts=cts,freq=freq,areas=un_areas,g=g,s=seed,k=kmax)

print 'Done', time() - s, 'sec'