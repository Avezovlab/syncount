#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:13:03 2020

@author: pierre
"""
import argparse
import logging

from numpy import array, mean, min, argmin, NaN, isnan, arange
from numpy import ravel_multi_index, bincount
from numpy.random import randint
from scipy.spatial.distance import cdist
from skimage.feature import blob_log


from skimage.external.tifffile import imread, imsave
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.morphology import remove_small_objects

from random import shuffle
import pickle
from os import path, listdir, makedirs, remove
from pathlib import Path


def remove_large_objects(ar, max_size):
    component_sizes = bincount(ar.ravel())
    too_big = component_sizes > max_size
    too_big_mask = too_big[ar]
    ar[too_big_mask] = 0
    return ar


#"C:/pierre/data"
in_base_dir =  "/mnt/data/mosab/data" #replace by the path to the folder
#containing the data

#C:/pierre/analysis
out_base_dir = "/tmp/a" #replace by the path to the folder
                                    #where you want the output files
                                    #to be generated
batch = 1 #replace batch number here
syn_type = "excit" #or "inhib"
generate_RGB = False #whether or not to generate RGB images of
                     #detected synapses
force = False #re-run already generated files


assert(syn_type in ["excit", "inhib"])

base_dir = "{}/batch{}/{}".format(in_base_dir, batch, syn_type)
out_dir = "{}/batch{}/{}".format(out_base_dir, batch, syn_type)

chan_names = {"inhib": {0: "pv", 1: "geph", 2: "vgat"},
              "excit": {0: "pv", 1: "psd95", 2: "vglut"}}

default_sigmas = {"inhib": [0.1, 0.1, 0.1],
                  "excit": [0.1, 0.1, 0.1]}


intens_range = list([0.1 + k * 0.02 for k in range(44)])

min_syn_marker_size = 50
max_syn_marker_size = 150
n_syn_marker_it = 30

print("Processing: batch{}: {}".format(batch, syn_type))

def shuffle_ret(l):
    shuffle(l)
    return l

def log(m, f):
    f.write(m + "\n")
    print(m)

exp_dirs = listdir(base_dir)
for cnt, fname in enumerate(shuffle_ret(exp_dirs)):
    if not fname.endswith(".tif"):
        print("Skipped[{}/{}]: {} is not a .tif file"
              .format(fname, cnt+1, len(exp_dirs)))
        continue

    out_exp_dir = path.join(out_dir, fname)
    if not path.isdir(out_exp_dir):
        makedirs(out_exp_dir)

    if path.isfile(path.join(out_exp_dir, ".lock")):
        print("Skipped[{}/{}] {}: found lock file"
              .format(cnt+1, len(exp_dirs), fname))
        continue

    print("Processing[{}/{}]: {}".format(cnt+1, len(exp_dirs), fname))

    #Put a lock on this analysis so that if multiple processings
    # happen in parallel they don't overlap
    Path(path.join(out_exp_dir, ".lock")).touch()

    logf = open(path.join(out_exp_dir, "log.txt"), 'w')

    imgs = imread(path.join(base_dir, fname))
    img_shape = (imgs.shape[0],) + imgs.shape[2:]

    ths = [None, None, None]
    sigmas = default_sigmas[syn_type]

    labs_props = [None, None, None]
    ran = [False, False, False]

    log("Labeling", logf)
    chan_file = path.join(out_exp_dir, "labs_{}.tif"
                          .format(chan_names[syn_type][0]))
    if not force and path.isfile(chan_file):
        labs = imread(chan_file)
    elif False: #else
        log("  " + chan_names[syn_type][0], logf)
        ran[0] = True
        imgs_0 = gaussian(imgs[:,0,:,:], sigma=sigmas[0])
        M = max(imgs_0.flatten())

        labs_0 = remove_small_objects(label(imgs_0 > intens_range[0] * M),
                                      min_size=5000, in_place=True)
        labs_0 = remove_large_objects(labs_0, 300000)
        props = regionprops(labs_0)
        best_area = mean([e.area for e in props])
        best_labs = array(labs_0)
        best_th = intens_range[0] * M
        log("   {}% ({:.2f}): {} (area={:.3f})"
            .format(int(intens_range[0]*100), intens_range[0]*M, len(props),
                    best_area), logf)
        for cnt in range(1, len(intens_range)):
            labs_0 = remove_small_objects(label(imgs_0 > intens_range[cnt] * M),
                                          min_size=5000, in_place=True)
            labs_0 = remove_large_objects(labs_0, 300000)
            props = regionprops(labs_0)
            cur_area = mean([e.area for e in props])
            log("   {}% ({:.2f}): {} (area={:.3f})"
                .format(int(intens_range[cnt]*100), intens_range[cnt]*M,
                        len(props), cur_area), logf)
            if isnan(best_area) or cur_area > best_area:
                best_area = cur_area
                best_labs = array(labs_0)
                best_th = intens_range[cnt] * M
        labs = best_labs
        ths[0] = best_th
        imsave(chan_file, labs.astype("uint16"))
    #labs_props[0] = regionprops(labs)
    #log("    Found {} Neurons".format(len(labs_props[0])), logf)

    if generate_RGB:
        col_file = path.join(out_exp_dir, "labs_{}_col.tif"
                             .format(chan_names[syn_type][0]))
        if force or not path.isfile(col_file):
            imsave(col_file,
                   label2rgb(labs, bg_label=0, colors=[[randint(255), randint(255), randint(255)] for i in range(len(labs_props[0]))]).astype('uint8'))
    #del labs

    chan_file = path.join(out_exp_dir, "labs_{}.tif"
                          .format(chan_names[syn_type][1]))
    if not force and path.isfile(chan_file):
        labs = imread(chan_file)
    else:
        log("  " + chan_names[syn_type][1], logf)
        ran[1] = True

        blbs = blob_log(imgs[:,1,:,:], max_sigma=10, threshold=mean(imgs[:,1,:,:].flatten())/2)


        assert(False)
        imgs_1 = gaussian(imgs[:,1,:,:], sigma=sigmas[1])
        M = max(imgs_1.flatten())



        labs_1 = remove_small_objects(label(imgs_1 > intens_range[-1] * M),
                                      min_size=min_syn_marker_size, in_place=True)
        labs_1 = remove_large_objects(labs_1, max_syn_marker_size)
        props = regionprops(labs_1)
        max_clusts = len(props)
        best_labs = array(labs_1)
        best_th = intens_range[-1] * M
        log("   {}% ({:.2f}): {}".format(int(intens_range[-1]*100),
                                         intens_range[-1]*M, len(props)), logf)
        for cnt in range(len(intens_range)-2, -1, -1):
            labs_1 = remove_small_objects(label(imgs_1 > intens_range[cnt] * M),
                                          min_size=min_syn_marker_size, in_place=True)
            labs_1 = remove_large_objects(labs_1, max_syn_marker_size)
            props = regionprops(labs_1)
            log("   {}% ({:.2f}): {}"
                .format(int(intens_range[cnt]*100), intens_range[cnt]*M,
                        len(props)), logf)
            if len(props) > max_clusts:
                max_clusts = len(props)
                best_labs = array(labs_1)
                best_th = intens_range[cnt] * M
        labs = best_labs
        ths[1] = best_th
        imsave(chan_file, labs.astype("uint16"))
    labs_props[1] = regionprops(labs)
    log("    Found {} {} clusters"
        .format(len(labs_props[1]), chan_names[syn_type][1]), logf)

    if generate_RGB:
        col_file = path.join(out_exp_dir, "labs_{}_col.tif"
                             .format(chan_names[syn_type][1]))
        if force or not path.isfile(col_file):
            imsave(col_file, label2rgb(labs, bg_label=0,
                                       colors=[[randint(255), randint(255), randint(255)] for i in range(len(labs_props[1]))]).astype('uint8'))
    del labs

    chan_file = path.join(out_exp_dir, "labs_{}.tif"
                          .format(chan_names[syn_type][2]))
    if not force and path.isfile(chan_file):
        labs = imread(chan_file)
    else:
        log("  " + chan_names[syn_type][2], logf)
        ran[2] = True
        imgs_2 = gaussian(imgs[:,2,:,:], sigma=sigmas[2])
        M= max(imgs_2.flatten())

        labs_2 = remove_small_objects(label(imgs_2 > intens_range[-1] * M),
                                      min_size=min_syn_marker_size, in_place=True)
        labs_2 = remove_large_objects(labs_2, max_syn_marker_size)
        props = regionprops(labs_2)
        max_clusts = len(props)
        best_labs = labs_2
        best_th = intens_range[-1] * M
        log("   {}% ({:.2f}): {}".format(int(intens_range[-1]*100),
                                         intens_range[-1]*M, len(props)), logf)
        for cnt in range(len(intens_range)-2, -1, -1):
            labs_2 = remove_small_objects(label(imgs_2 > intens_range[cnt] * M),
                                          min_size=min_syn_marker_size, in_place=True)
            labs_2 = remove_large_objects(labs_2, max_syn_marker_size)
            props = regionprops(labs_2)
            log("   {}% ({:.2f}): {}"
                .format(int(intens_range[cnt]*100), intens_range[cnt]*M,
                        len(props)), logf)
            if len(props) > max_clusts:
                max_clusts = len(props)
                best_labs = array(labs_2)
                best_th = intens_range[cnt] * M
        labs = best_labs
        ths[2] = best_th
        imsave(chan_file, labs.astype("uint16"))
    labs_props[2] = regionprops(labs)
    log("    Found {} {} clusters"
        .format(len(labs_props[2]), chan_names[syn_type][2]), logf)

    if generate_RGB:
        col_file = path.join(out_exp_dir, "labs_{}_col.tif"
                             .format(chan_names[syn_type][2]))
        if force or not path.isfile(col_file):
            imsave(col_file, label2rgb(labs, bg_label=0,
                                       colors=[[randint(255), randint(255), randint(255)] for i in range(len(labs_props[2]))]).astype('uint8'))
    del labs

    if path.isfile(path.join(out_exp_dir, "analysis_params.pkl")):
        with open(path.join(out_exp_dir, "analysis_params.pkl"), 'rb') as f:
            params = pickle.load(f)
    else:
        params = {}

    if params == {} or any(ran):
        for k in range(3):
            if ran[k]:
                params["th_{}".format(chan_names[syn_type][k])] = ths[k]
                params["sigma_{}".format(chan_names[syn_type][k])] = sigmas[k]
        with open(path.join(out_exp_dir, "analysis_params.pkl"), 'wb') as f:
                  pickle.dump(params, f)

    if force or not path.isfile(path.join(out_exp_dir, "clusts.pkl")):
        log("Finding synapses", logf)

        labs_0_set = [e.label for e in labs_props[0]]
        labs_0_pxs = {e.label:set([ravel_multi_index(f, img_shape)
                                   for f in e.coords])
                        for e in labs_props[0]}

        clusts = {}
        clusts_neurons = {}
        for k in [1, 2]:
            name = chan_names[syn_type][k]
            log("  " + name, logf)
            clusts[name] = []
            clusts_neurons[name] = []
            for e in labs_props[k]:
                pxs = e.coords
                cent = mean(pxs, axis=0)
                rad = e.equivalent_diameter/2

                clusts[name].append(array([e.label, cent[0], cent[1], cent[2],
                                           rad, pxs.shape[0]]))
                clust_pxs = set([ravel_multi_index(f, img_shape)
                                 for f in e.coords])
                clusts_neurons[name].extend([[e.label, l] for l in labs_0_set if
                                            clust_pxs & labs_0_pxs[l] != set([])])


        log("  colocalization", logf)
        name1 = chan_names[syn_type][1]
        name2 = chan_names[syn_type][2]
        overlap = []
        for k in range(len(clusts[name1])):
            c = array(clusts[name1][k][1:4])
            overlap.extend([[k,l] for l in range(len(clusts[name2]))
                            if sum((c - array(clusts[name2][l][1:4]))**2) < (clusts[name1][k][4] + clusts[name2][l][4])**2])
        log("  Found {} synapses".format(len(overlap)), logf)


        with open(path.join(out_exp_dir, "clusts.pkl"), "wb") as f:
            pickle.dump({"sigmas": default_sigmas[syn_type],
                         "intensities": intens_range,
                         "min_syn_marker_size": min_syn_marker_size,
                         "max_syn_marker_size": max_syn_marker_size,
                         "n_syn_marker_it": n_syn_marker_it,
                         name1: clusts[name1], "{}_neurons".format(name1): clusts_neurons[name1],
                         name2: clusts[name2], "{}_neurons".format(name2): clusts_neurons[name2],
                         'overlap': overlap}, f)   #, 'near_neighb': near_neighb}, f)
        del overlap
        del clusts
        del clusts_neurons

    logf.close()
    #analysis is done, remove lock file
    remove(path.join(out_exp_dir, ".lock"))

    del imgs

print("DONE")
