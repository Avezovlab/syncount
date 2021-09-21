#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 12:24:33 2021

@author: pierre
"""

import argparse

from math import sqrt, floor, ceil
from numpy import array, mean, isnan, ones, stack, where, vstack, amax
from numpy import bincount, arange
from numpy.random import randint
from numpy import isscalar, full, asarray, argmax, log10, logspace, max, newaxis, linspace, empty, float64, hstack


from skimage.io import imread, imsave
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.feature import blob_log
from skimage.color import label2rgb
from skimage.morphology import remove_small_objects, erosion
from skimage.exposure import equalize_adapthist
from skimage.util import img_as_float
from skimage.feature.peak import peak_local_max
from skimage.feature.blob import _format_exclude_border, _prune_blobs


from scipy.ndimage import gaussian_laplace

from random import shuffle
import pickle
from os import path, listdir, makedirs, remove
from pathlib import Path


parser = argparse.ArgumentParser(description="Compute imaging volume")

parser.add_argument("in_dir")
parser.add_argument("out_dir")
parser.add_argument("batch")
parser.add_argument("syn_type")

args = parser.parse_args()

batch = int(args.batch)
syn_type = args.syn_type
assert(syn_type in ["excit", "inhib"])


base_dir = "{}{}/{}".format(args.in_dir, batch, syn_type)
out_dir = "{}/batch{}/{}".format(args.out_dir, batch, syn_type)

chan_names = {"inhib": {0: "pv", 1: "geph", 2: "vgat"},
              "excit": {0: "pv", 1: "psd95", 2: "vglut"}}

pxsize = 0.1 #micron

print("Processing: batch{}: {}".format(batch, syn_type))


exp_dirs = listdir(base_dir)

files_to_treat = enumerate(exp_dirs)

for cnt, fname in files_to_treat:
    if not fname.endswith(".tif"):
        print("Skipped[{}/{}]: {} is not a tif file"
              .format(cnt+1, len(exp_dirs), fname))
        continue

    out_exp_dir = path.join(out_dir, fname)

    if not path.isdir(out_exp_dir):
        makedirs(out_exp_dir)

    print("Processing[{}/{}]: {}".format(cnt+1, len(exp_dirs), fname))

    imgs = imread(path.join(base_dir, fname))
    img_shape = (imgs.shape[0],) + imgs.shape[2:]

    with open(path.join(out_exp_dir, "clusts.pkl"), "rb") as f:
        dats = pickle.load(f)

    dats["volume"] = img_shape[0] * img_shape[1] * img_shape[2] * pxsize
    dats["pxsize"] = pxsize

    with open(path.join(out_exp_dir, "clusts.pkl"), "wb") as f:
        pickle.dump(dats, f)
