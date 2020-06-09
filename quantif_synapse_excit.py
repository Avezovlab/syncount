#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:13:03 2020

@author: pierre
"""
from numpy import zeros, array, mean, where, sum, sqrt, min, max, argmin, NaN
from numpy import ravel_multi_index, bincount
from numpy.random import randint
from scipy.spatial.distance import cdist

from skimage.external.tifffile import imread, imsave
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.morphology import remove_small_objects


from collections import Counter

import pickle
from os import path, listdir, makedirs


def remove_large_objects(ar, max_size):
    component_sizes = bincount(ar.ravel())
    too_big = component_sizes > max_size
    too_big_mask = too_big[ar]
    ar[too_big_mask] = 0
    return ar

batch = 3
base_dir = "/mnt/data/cam/mosab/data/batch{}/excit".format(batch)
out_dir = "/mnt/data/cam/mosab/analysis/batch{}/excit".format(batch)


force = False
for fname in listdir(base_dir): #[e for e in listdir(base_dir) if e.startswith("510")]:
    if not fname.endswith(".tif"):
        print("Skipped: " + fname + " not a .tif file")
        continue
    print("Processing: " + fname)

    out_exp_dir = path.join(out_dir, fname)
    if not path.isdir(out_exp_dir):
        makedirs(out_exp_dir)

    imgs = imread(path.join(base_dir, fname))

    th_pv = 0.1
    sigma_pv = 0.1

    th_psd95 = 0.6
    sigma_psd95 = 0.1

    th_vglut = 0.6
    sigma_vglut = 0.1

    print("Labeling")
    ran = False
    if not force and path.isfile(path.join(out_exp_dir, "labs_pv.tif")):
        labs_pv = imread(path.join(out_exp_dir, "labs_pv.tif"))
    else:
        print("  pv")
        ran = True
        imgs_pv = gaussian(imgs[:,0,:,:], sigma=sigma_pv)
        labs_pv = remove_small_objects(label(imgs_pv > th_pv), min_size=5000, in_place=True)
        labs_pv = remove_large_objects(labs_pv, 200000)
        props = regionprops(labs_pv)
        best_area = mean([e.area for e in props])
        best_labs = array(labs_pv)
        best_th = th_pv
        cnt = 0
        while cnt < 15:
            labs_pv = remove_small_objects(label(imgs_pv > th_pv), min_size=5000, in_place=True)
            labs_pv = remove_large_objects(labs_pv, 200000)
            props = regionprops(labs_pv)
            cur_area = mean([e.area for e in props])
            if cur_area > best_area:
                best_area = cur_area
                best_labs = array(labs_pv)
                best_th = th_pv
            th_pv += 0.02
            cnt += 1
        labs_pv = best_labs
        th_pv = best_th
        imsave(path.join(out_exp_dir, "labs_pv.tif"), labs_pv.astype("uint16"))

    if not force and path.isfile(path.join(out_exp_dir, "labs_psd95.tif")):
        labs_psd95 = imread(path.join(out_exp_dir, "labs_psd95.tif"))
    else:
        print("  psd95")
        ran = True
        imgs_psd95 = gaussian(imgs[:,1,:,:], sigma=sigma_psd95)
        labs_psd95 = remove_small_objects(label(imgs_psd95 > th_psd95), min_size=11, in_place=True)
        labs_psd95 = remove_large_objects(labs_psd95, 300)
        props = regionprops(labs_psd95)
        max_clusts = len(props)
        best_labs = array(labs_psd95)
        best_th = th_psd95
        cnt = 0
        while cnt < 15:
            labs_psd95 = remove_small_objects(label(imgs_psd95 > th_psd95), min_size=11, in_place=True)
            labs_psd95 = remove_large_objects(labs_psd95, 300)
            props = regionprops(labs_psd95)
            if len(props) > max_clusts:
                max_clusts = len(props)
                best_labs = array(labs_psd95)
                best_th = th_psd95
            th_psd95 -= 0.02
            cnt += 1
        labs_psd95 = best_labs
        th_psd95 = best_th
        imsave(path.join(out_exp_dir, "labs_psd95.tif"), labs_psd95.astype("uint16"))

    if not force and path.isfile(path.join(out_exp_dir, "labs_vglut.tif")):
        labs_vglut = imread(path.join(out_exp_dir, "labs_vglut.tif"))
    else:
        print("  vglut")
        ran = True
        imgs_vglut = gaussian(imgs[:,2,:,:], sigma=sigma_vglut)
        labs_vglut = remove_small_objects(label(imgs_vglut > th_vglut), min_size=11, in_place=True)
        labs_vglut = remove_large_objects(labs_vglut, 300)
        props = regionprops(labs_vglut)
        max_clusts = len(props)
        best_labs = labs_vglut
        best_th = th_vglut
        cnt = 0
        while cnt < 15:
            labs_vglut = remove_small_objects(label(imgs_vglut > th_vglut), min_size=11, in_place=True)
            labs_vglut = remove_large_objects(labs_vglut, 300)
            props = regionprops(labs_vglut)
            if len(props) > max_clusts:
                max_clusts = len(props)
                best_labs = array(labs_vglut)
                best_th = th_vglut
            th_vglut -= 0.02
            cnt += 1
        labs_vglut = best_labs
        th_vglut = best_th
        imsave(path.join(out_exp_dir, "labs_vglut.tif"), labs_vglut.astype("uint16"))

    print("Generating color images")
    if force or not path.isfile(path.join(out_exp_dir, "labs_pv_col.tif")):
        props = regionprops(labs_pv)
        imsave(path.join(out_exp_dir, "labs_pv_col.tif"), label2rgb(labs_pv, bg_label=0,
               colors=[[randint(255), randint(255), randint(255)] for i in range(len(props))]).astype('uint8'))

    if force or not path.isfile(path.join(out_exp_dir, "labs_psd95_col.tif")):
        props = regionprops(labs_psd95)
        imsave(path.join(out_exp_dir, "labs_psd95_col.tif"), label2rgb(labs_psd95, bg_label=0,
               colors=[[randint(255), randint(255), randint(255)] for i in range(len(props))]).astype('uint8'))

    if force or not path.isfile(path.join(out_exp_dir, "labs_vglut_col.tif")):
        props = regionprops(labs_vglut)
        imsave(path.join(out_exp_dir, "labs_vglut_col.tif"), label2rgb(labs_vglut, bg_label=0,
               colors=[[randint(255), randint(255), randint(255)] for i in range(len(props))]).astype('uint8'))

    params = None
    if path.isfile(path.join(out_exp_dir, "analysis_params.pkl")):
        with open(path.join(out_exp_dir, "analysis_params.pkl"), 'rb') as f:
            params = pickle.load(f)
    if (params is None or params["th_pv"] != th_pv or params["sigma_pv"] != sigma_pv or
        params["th_psd95"] != th_psd95 or params["sigma_psd95"] != sigma_psd95 or
        params["th_vglut"] != th_vglut or params["sigma_vglut"] != sigma_vglut) and ran:
        print("Saving params")
           params = {"th_pv": th_pv, "sigma_pv": sigma_pv,
                     "th_psd95": th_psd95, "sigma_psd95": sigma_psd95,
                     "th_vglut": th_vglut, "sigma_vglut": sigma_vglut}
           with open(path.join(out_exp_dir, "analysis_params.pkl"), 'wb') as f:
               pickle.dump(params, f)

    if force or not path.isfile(path.join(out_exp_dir, "clusts.pkl")):
        print("Finding synapses")

        props = regionprops(labs_pv)
        labs_pv_set = [e.label for e in props]
        labs_pv_pxs = {e.label:set([ravel_multi_index(f, labs_pv.shape) for f in e.coords])
                        for e in props}

        print("  vglut")
        clusts_vglut = []
        clusts_vglut_neurons = []
        for e in regionprops(labs_vglut):
            pxs = e.coords
            cent = mean(pxs, axis=0)
            rad = e.equivalent_diameter/2

            clusts_vglut.append(array([e.label, cent[0], cent[1], cent[2], rad]))

            vglut_clust_pxs = set([ravel_multi_index(f, labs_vglut.shape) for f in e.coords])
            clusts_vglut_neurons.extend([[e.label, l] for l in labs_pv_set if
                                        vglut_clust_pxs & labs_pv_pxs[l] != set([])])

        print("  psd95")
        clusts_psd95 = []
        clusts_psd95_neurons = []
        for e in regionprops(labs_psd95):
            pxs = e.coords
            cent = mean(pxs, axis=0)
            rad = e.equivalent_diameter/2

            clusts_psd95.append(array([e.label, cent[0], cent[1], cent[2], rad]))

            psd95_clust_pxs = set([ravel_multi_index(f, labs_psd95.shape) for f in e.coords])
            clusts_psd95_neurons.extend([[e.label, l] for l in labs_pv_set if
                                        psd95_clust_pxs & labs_pv_pxs[l] != set([])])


        print("  colocalization")
        near_neighb = []
        overlap = []
        for k in range(len(clusts_vglut)):
            dists = cdist(clusts_vglut[k][1:4].reshape((1,3)), array([e[1:4] for e in clusts_psd95]))
            overlap.extend([[k,l] for l in range(len(clusts_psd95))
                            if dists[0,l] < clusts_vglut[k][4] + clusts_psd95[l][4]])

            if dists.size > 0:
                near_neighb.append([argmin(dists), min(dists)])
            else:
                near_neighb.append([-1, NaN])

        with open(path.join(out_exp_dir, "clusts.pkl"), "wb") as f:
            pickle.dump({"psd95": clusts_psd95, "psd95_neurons": clusts_psd95_neurons,
                         "vglut": clusts_vglut, "vglut_neurons": clusts_vglut_neurons,
                         'overlap': overlap, 'near_neighb': near_neighb}, f)

    del imgs
print("DONE")