#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 12:59:20 2021

@author: pierre
"""
from numpy import zeros, array, mean, std, where, sum, sqrt, min, max, argmin, histogram, ones, vstack
from matplotlib import pyplot as plt
from math import ceil

from collections import Counter
from numpy.random import randint

from os import path, listdir, makedirs

import pickle

def log(m, f):
    f.write(m + "\n")
    print(m)

chan_names = {"inhib": {0: "pv", 1: "geph", 2: "vgat"},
              "excit": {0: "pv", 1: "psd95", 2: "vglut"}}

batch = 1 #batch number
neur_type = "inhib" #excit" #or "inhib"
in_base_dir = "/mnt/data/mosab/analysis3" #"/mnt/data/mosab/analysis2/" #This corresponds to the
                                   #out_base_dir of the
                                   #quantif_synapse.py script

f_th = 1.5 

base_dir = "{}/batch{}/{}".format(in_base_dir, batch, neur_type)

dirs = listdir(base_dir)
nsynapses = {}
nclusts = [{}, {}]
clust_sizes = [{}, {}]
for i in range(len(dirs)):
    fname = dirs[i]
    cur_path = path.join(base_dir, fname)

    print("Processing[{}/{}]: {}".format(i+1, len(dirs), fname))

    with open(path.join(cur_path, "clusts.pkl"), 'rb') as f:
        ana = pickle.load(f)

    pks = [None, ana[chan_names[neur_type][1]], ana[chan_names[neur_type][2]]]
    print("  Overlaping {} x {} clusters with f={}".format(pks[1].shape[0], pks[2].shape[0], f_th))

    overlap = []
    for k in range(pks[1].shape[0]):
        idxs = where(((pks[1][k,0] - pks[2][:,0]) / max(vstack([pks[1][k,3] * ones(pks[2].shape[0]), pks[2][:,3]]).T, axis=1))**2 +
                     ((pks[1][k,1] - pks[2][:,1]) / max(vstack([pks[1][k,3] * ones(pks[2].shape[0]), pks[2][:,3]]).T, axis=1))**2 +
                     ((pks[1][k,2] - pks[2][:,2]) / max(vstack([pks[1][k,5] * ones(pks[2].shape[0]), pks[2][:,5]]).T, axis=1))**2 < f_th)
        overlap.extend([[k, l] for l in idxs[0].astype("int")])

    print("  Found {} synapses (was {} for f=1)".format(len(overlap), len(ana["overlap"])))

    ana["overlap_f={}".format(f_th)] = overlap
    with open(path.join(cur_path, "clusts.pkl"), "wb") as f:
        pickle.dump(ana, f)