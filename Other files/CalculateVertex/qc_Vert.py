#!/usr/bin/python

import sys

if len(sys.argv) != 4:
    print 'too many args: want arguments in the form [gammasum, seeds, kmax]'
    sys.exit()
else:
    g = (float)(sys.argv[1])
    seeds = (int)(sys.argv[2])
    kmax = (int)(sys.argv[3])
    kmin = -kmax

from scipy.spatial import Voronoi
import gc
import numpy as np
import random as random
from time import time
from joblib import Parallel, delayed
import os
import math

dic = {36 : 'S1', 180 - 36 : 'S2', 72 : 'F1', 180 - 72 : 'F2'}

V = ''.join(['F1','F1','S1','S1','F1','S1','S1'])
W = ''.join(['F1','S1','F1','S1','F1','S1','S1'])
X = ''.join(['F1','S1','S1','F1','S1','S1','S1','S1'])
Y = ''.join(['F1','S1','F1','S1','S1','S1','S1','S1'])

#num_cores = int(os.environ['SLURM_NTASKS_PER_NODE'])
num_cores = 1
print 'num cores', num_cores, type(num_cores)

np.seterr(divide='ignore')
np.seterr(invalid='ignore')

print '\ngammasum, seeds, kmax, numwindows, numintervals', sys.argv[1:]
print 'beginning..............................'
print 'kmax      ', kmax
print 'gammasum  ', g
print 'seeds      ', seeds

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


sve = StarVectorsExact()[0]

rad = 2*kmax

all_cts = []

for seed in range(seeds):

    tiles,tilepoints = DualizeGridParallel(kmin,kmax,g, nj, seed)

    (a,b,c) = tiles.shape

    tilepoints2 = np.dot(tilepoints, sve)
    dist = np.linalg.norm(tilepoints2, axis=1)
    pts_ind = np.where(dist<rad/2)[0]

    print 'calculating vertex distribution...seed',seed

    s = time()

    vor = Voronoi(tilepoints2)

    gc.collect()  

    print 'vor done...', time() - s

    s = time()

    # Get Vertices

    temp = [vor.vertices[vor.regions[vor.point_region[x]]] for x in pts_ind]
    areas = [PolyAreaPt(r) for r in temp]

    round_areas = np.round(areas,10)

    v_types = ['Q', 'R', 'K', 'M', 'D', 'ST', 'Z', 'T', 'U', 'S', 'J', 'L', 'V', 'W', 'X', 'Y']
    v_areas = [0.607, 0.738, 0.758, 0.769, 0.8, 0.812, 0.831, 0.889, 0.901, 0.908, 0.932, 0.963, 0.87, 0.87, 0.851, 0.851]

    m = len(v_types)
    cts = np.zeros(m).astype(int)
    for i in range(m-4): # except for VW and XY
        cts[i] = len(np.where(np.round(round_areas,3) == v_areas[i])[0])

    XY_ind = np.where(np.round(round_areas,3) == 0.851)[0]
    VW_ind = np.where(np.round(round_areas,3) == 0.87)[0]

    tiles_rs = np.reshape(tiles,(a*b,c))

    print 'start degenerate tiles...', time() - s

    s = time()

    for i in range(len(VW_ind)):

        cur_vx = tilepoints[pts_ind[VW_ind[i]]]

        cur_ind = np.where((tiles_rs == cur_vx).all(axis=1))[0]
        cur_tiles = unique_rows(np.concatenate(tiles[cur_ind/b]))
        cur_pts = np.dot(cur_tiles, sve)

        r = np.dot(cur_vx, sve)
        dr = cur_pts - r
        nearest_nbr = dr[np.where(np.abs(np.linalg.norm(dr,axis=1) - 1)<0.01)[0]]

        cur_ang = np.argsort([math.atan2(x[1],x[0]) for x in nearest_nbr])

        nearest_nbr = nearest_nbr[cur_ang]

        nearest_nbr = np.concatenate((nearest_nbr,[nearest_nbr[0]])) # add 0th guy
        angles = np.round(np.arccos(np.einsum('ij,ij->i',nearest_nbr[1:],nearest_nbr[:-1])) / np.pi * 180,1)
    #     print i, #nearest_nbr, angles
        angles = [dic[x] for x in angles]

        if V in ''.join(angles + angles):
            cts[-4] += 1
        elif W in ''.join(angles + angles):
            cts[-3] += 1
        else:
            print 'unidentified'
            
    for i in range(len(XY_ind)):

        cur_vx = tilepoints[pts_ind[XY_ind[i]]]

        cur_ind = np.where((tiles_rs == cur_vx).all(axis=1))[0]
        cur_tiles = unique_rows(np.concatenate(tiles[cur_ind/b]))
        cur_pts = np.dot(cur_tiles, sve)


        r = np.dot(cur_vx, sve)
        dr = cur_pts - r
        nearest_nbr = dr[np.where(np.abs(np.linalg.norm(dr,axis=1) - 1)<0.01)[0]]


        cur_ang = np.argsort([math.atan2(x[1],x[0]) for x in nearest_nbr])

        nearest_nbr = nearest_nbr[cur_ang]

        nearest_nbr = np.concatenate((nearest_nbr,[nearest_nbr[0]])) # add 0th guy
        angles = np.round(np.arccos(np.einsum('ij,ij->i',nearest_nbr[1:],nearest_nbr[:-1])) / np.pi * 180,1)
        #print i, #nearest_nbr, angles
        angles = [dic[x] for x in angles]

        if X in ''.join(angles + angles):
            cts[-2] += 1
        elif Y in ''.join(angles + angles):
            cts[-1] += 1
        else:
            print 'unidentified'

    tot = sum(cts)

    print 'total', tot, len(pts_ind)

    for i in range(m):
        print v_types[i] + '{:6d}   {:6.2f}%'.format(cts[i], float(cts[i]) / tot * 100)

    all_cts.append(cts)

fout='./vert_g_{:0.4f}_ns_{:d}_k_{:d}'.format(g,seeds,kmax)
np.savez(fout,cts=all_cts,vt=v_types,va=v_areas,g=g,s=seeds,k=kmax)

print 'Done', time() - s, 'sec'