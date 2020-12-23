#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:34:30 2020

@author: pierre
"""

from numpy import array, mean, min, argmin, NaN, isnan, arange
from numpy import ravel_multi_index, bincount
from numpy.random import randint
from scipy.spatial.distance import cdist


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

ths = [None, None, None]
sigmas = default_sigmas[syn_type]

labs_props = [None, None, None]
ran = [False, False, False]


exp_dirs = listdir(base_dir)


fname = exp_dirs[0]
imgs = imread(path.join(base_dir, fname))
img_shape = (imgs.shape[0],) + imgs.shape[2:]

imgs_1 = gaussian(imgs[:,1,:,:], sigma=sigmas[1])
M = max(imgs_1.flatten())

labs_1 = remove_small_objects(label(imgs_1 > intens_range[-1] * M),
                              min_size=min_syn_marker_size, in_place=True)
labs_1 = remove_large_objects(labs_1, max_syn_marker_size)
props = regionprops(labs_1)
max_clusts = len(props)
best_labs = array(labs_1)
best_th = intens_range[-1] * M
print("   {}% ({:.2f}): {}".format(int(intens_range[-1]*100),
                                 intens_range[-1]*M, len(props)))
for cnt in range(len(intens_range)-2, -1, -1):
    labs_1 = remove_small_objects(label(imgs_1 > intens_range[cnt] * M),
                                  min_size=min_syn_marker_size, in_place=True)
    labs_1 = remove_large_objects(labs_1, max_syn_marker_size)
    props = regionprops(labs_1)
    print("   {}% ({:.2f}): {}"
        .format(int(intens_range[cnt]*100), intens_range[cnt]*M,
                len(props)))
    if len(props) > max_clusts:
        max_clusts = len(props)
        best_labs = array(labs_1)
        best_th = intens_range[cnt] * M
labs = best_labs
ths[1] = best_th
#imsave(chan_file, labs.astype("uint16"))
labs_props[1] = regionprops(labs)
print("    Found {} {} clusters"
      .format(len(labs_props[1]), chan_names[syn_type][1]))
