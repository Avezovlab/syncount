#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:27:37 2020

@author: pierre
"""
from numpy import zeros, array, mean, where, sum, sqrt, min, max, argmin, NaN
from numpy import ravel_multi_index, bincount
from numpy.random import randint


from skimage.external.tifffile import imread, imsave
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.morphology import remove_small_objects

from collections import Counter

import pickle
from os import path, listdir, makedirs

from matplotlib import pyplot as plt


fname = "/mnt/data/cam/mosab/data/batch2/inhib/ID416_BO_SS ct PV Geph VGAT 18-12-2018.lif - 416 x63 dx field5.tif"


imgs = imread(fname)

pxw = 0.1003842

th_pv = 0.6
sigma_pv = 0.1
comp_size_pv = 50

th_geph = 0.6
sigma_geph = 0.1
comp_size_geph = 50

th_vgat = 0.6
sigma_vgat = 0.1
comp_size_vgat = 50


#imgs_pv = gaussian(imgs[:,0,:,:], sigma=sigma_pv)
#labs_pv = label(imgs_pv > th_pv)
#tmp_pv = zeros(imgs_pv.shape)
#
#cnts = Counter(labs_pv.flatten())
#for k,v in cnts.items():
#    if v >= comp_size_pv:
#        tmp_pv[labs_pv == k] = k
#labs_pv = tmp_pv
#del tmp_pv


imgs_geph = gaussian(imgs[:,1,:,:], sigma=sigma_geph)


def lab_radii(lab_imgs):
    res = []
    for k in (set(lab_imgs.flatten()) - set([0])):
        [pxsx, pxsy, pxsz] = where(lab_imgs == k)
        pxs = array([pxsx, pxsy, pxsz])
        cent = mean(pxs, axis=1)
        res.append(max(sqrt(sum((pxs - cent.reshape((3,1))) ** 2, axis=0)))*pxw)
    return array(res)


def remove_large_objects(ar, max_size):
    component_sizes = bincount(ar.ravel())
    too_big = component_sizes > max_size
    too_big_mask = too_big[ar]
    ar[too_big_mask] = 0
    return ar

labs_geph = remove_small_objects(label(imgs_geph > th_geph), min_size=11, in_place=True)
labs_geph = remove_large_objects(labs_geph, 300)
props = regionprops(labs_geph)
max_clusts = len(props)
best_labs = labs_geph
cnt = 0
while cnt < 15:
    print(cnt)
    labs_geph = remove_small_objects(label(imgs_geph > th_geph), min_size=11, in_place=True)
    labs_geph = remove_large_objects(labs_geph, 300)
    props = regionprops(labs_geph)
    if len(props) > max_clusts:
        max_clusts = len(props)
        best_labs = labs_geph
    th_geph -= 0.02
    cnt += 1

tmp = label2rgb(best_labs, bg_label=0, colors = [[randint(255), randint(255), randint(255)] for i in range(len(props))])
imsave("/tmp/toto.tif", tmp.astype('uint8'))

del tmp

assert(False)

tmp_geph = zeros(imgs_pv.shape)


cnts = Counter(labs_geph.flatten())
for k,v in cnts.items():
    if v >= comp_size_geph:
        tmp_geph[labs_geph == k] = k
labs_geph = tmp_geph
del tmp_geph


imgs_vgat = gaussian(imgs[:,2,:,:], sigma=sigma_vgat)
labs_vgat = label(imgs_vgat > th_vgat)
tmp_vgat = zeros(imgs_pv.shape)

cnts = Counter(labs_vgat.flatten())
for k,v in cnts.items():
    if v >= comp_size_vgat:
        tmp_vgat[labs_vgat == k] = k
labs_vgat = tmp_vgat
del tmp_vgat


tmp = zeros((labs_pv.shape[0], labs_pv.shape[1], labs_pv.shape[2], 3), dtype='uint8')
for k in set(labs_pv.flatten()) - set([0]):
    col = [randint(255), randint(255), randint(255)]
    tmp[labs_pv == k,:] = col
del tmp

tmp = zeros((labs_geph.shape[0], labs_geph.shape[1], labs_geph.shape[2], 3), dtype='uint8')
for k in set(labs_geph.flatten()) - set([0]):
    col = [randint(255), randint(255), randint(255)]
    tmp[labs_geph == k,:] = col
del tmp

tmp = zeros((labs_vgat.shape[0], labs_vgat.shape[1], labs_vgat.shape[2], 3), dtype='uint8')
for k in set(labs_vgat.flatten()) - set([0]):
    col = [randint(255), randint(255), randint(255)]
    tmp[labs_vgat == k,:] = col
del tmp

labs_pv_set = set(labs_pv.flatten()) - set([0])
labs_pv_pxs = {l:set(ravel_multi_index(where(labs_pv == l), labs_pv.shape))
                for l in labs_pv_set}

clusts_vgat = []
clusts_vgat_neurons = []
for k in (set(labs_vgat.flatten()) - set([0])):
    print("vgat: {}".format(k))
    [pxsx, pxsy, pxsz] = where(labs_vgat == k)
    pxs = array([pxsx, pxsy, pxsz])
    cent = mean(pxs, axis=1)
    rad = max(sqrt(sum((pxs - cent.reshape((3,1))) ** 2, axis=0)))

    clusts_vgat.append(array([k, cent[0], cent[1], cent[2], rad]))

    vgat_clust_pxs = set(ravel_multi_index(where(labs_vgat == k), labs_vgat.shape))
    clusts_vgat_neurons = [l for l in labs_pv_set if
                           vgat_clust_pxs & labs_pv_pxs[l] != set([])]

clusts_geph = []
clusts_geph_neurons = []
for k in (set(labs_geph.flatten()) - set([0])):
    print("geph: {}".format(k))
    [pxsx, pxsy, pxsz] = where(labs_geph == k)
    pxs = array([pxsx, pxsy, pxsz])
    cent = mean(pxs, axis=1)
    rad = max(sqrt(sum((pxs - cent.reshape((3,1))) ** 2, axis=0)))

    clusts_geph.append(array([k, cent[0], cent[1], cent[2], rad]))
    
    geph_clust_pxs = set(ravel_multi_index(where(labs_geph == k), labs_geph.shape))
    clusts_geph_neurons = [l for l in labs_pv_set if
                           geph_clust_pxs & labs_pv_pxs[l] != set([])]


near_neighb = []
overlap = []
for k in range(len(clusts_vgat)):
    dists = zeros((len(clusts_geph), 1))
    for l in range(len(clusts_geph)):
        dists[l] = sqrt(sum((clusts_vgat[k][1:4] - clusts_geph[l][1:4])**2))
        if sum((clusts_vgat[k][1:4] - clusts_geph[l][1:4])**2) < (clusts_vgat[k][4] + clusts_geph[l][4])**2:
            overlap.append([k, l])
    if dists.size > 0:
        near_neighb.append([argmin(dists), min(dists)])
    else:
        near_neighb.append([-1, NaN])
print("DONE")