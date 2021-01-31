#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:09:29 2021

@author: pierre
"""
import argparse

from numpy import zeros, array, mean, std, where, sum, sqrt, min, max, argmin, histogram, ones, vstack
from math import ceil

from collections import Counter
from numpy.random import randint

from os import path, listdir, makedirs

from skimage.io import imread, imsave
from skimage.measure import label, regionprops

import pickle

def log(m, f):
    f.write(m + "\n")
    print(m)

parser = argparse.ArgumentParser(description="Find synapses near somas")

parser.add_argument("--force", action="store_true", default=False,
                    help="Force re-run analysis for already completed files")
parser.add_argument("batch")
parser.add_argument("syn_type")

args = parser.parse_args()

batch = int(args.batch)
neur_type = args.syn_type
assert(neur_type in ["excit", "inhib"])

chan_names = {"inhib": {0: "pv", 1: "geph", 2: "vgat"},
              "excit": {0: "pv", 1: "psd95", 2: "vglut"}}


in_base_dir = "/mnt/data/mosab/analysis3" #"/mnt/data/mosab/analysis2/" #This corresponds to the
                                   #out_base_dir of the
                                   #quantif_synapse.py script

soma_f_th = 1
f_th = 1.5 
force = args.force

base_dir = "{}/batch{}/{}".format(in_base_dir, batch, neur_type)

dirs = listdir(base_dir)
nsynapses = {}
nclusts = [{}, {}]
clust_sizes = [{}, {}]
for i in range(len(dirs)):
    fname = dirs[i]
    cur_path = path.join(base_dir, fname)
    
    imgs = imread(path.join(base_dir, dirs[i], "labs_{}.tif".format(chan_names[neur_type][0])))
    neurs = regionprops(label(imgs))

    try:
        with open(path.join(cur_path, "clusts.pkl"), 'rb') as f:
            ana = pickle.load(f)
    except:
        print("Fail: {}".format(fname))
        continue

    if not force and "soma_f={}_overlap_f={}".format(soma_f_th, f_th) in ana.keys():
        print("Skipped[{}/{}]: {}".format(i+1, len(dirs), fname))
        continue

    print("Processing[{}/{}]: {}".format(i+1, len(dirs), fname))

    pks = [None, ana[chan_names[neur_type][1]], ana[chan_names[neur_type][2]]]

    print("  Filtering clusters on soma")

    pks_filt = [None, [], []]
    for k in [1, 2]:
        for pk in pks[k]:
            for neur in neurs:
                cs = neur.coords
                if sum(((pk[0] - cs[:,0]) / pk[3])**2 + ((pk[1] - cs[:,1]) / pk[3])**2 + ((pk[2] - cs[:,2]) / pk[5])**2 < soma_f_th) > 1:
                    pks_filt[k].append(pk)
        pks_filt[k] = array(pks_filt[k])

    print("  Overlaping {} (was {}) x {} (was {}) clusters with f={}".format(pks_filt[1].shape[0], pks[1].shape[0],
                                                                             pks_filt[2].shape[0], pks[2].shape[0], f_th))

    overlap = []
    if pks_filt[1].size > 0 and pks_filt[2].size > 0:
        for k, pk in enumerate(pks_filt[1]):
            idxs = where(((pk[0] - pks_filt[2][:,0]) / max(vstack([pk[3] * ones(pks_filt[2].shape[0]), pks_filt[2][:,3]]).T, axis=1))**2 +
                         ((pk[1] - pks_filt[2][:,1]) / max(vstack([pk[3] * ones(pks_filt[2].shape[0]), pks_filt[2][:,3]]).T, axis=1))**2 +
                         ((pk[2] - pks_filt[2][:,2]) / max(vstack([pk[5] * ones(pks_filt[2].shape[0]), pks_filt[2][:,5]]).T, axis=1))**2 < f_th)
            overlap.extend([[k, l] for l in idxs[0].astype("int")])

    print("  Found {} synapses".format(len(overlap)))

    ana[chan_names[neur_type][1] + "_soma_f={}".format(soma_f_th)] = pks_filt[1]
    ana[chan_names[neur_type][2] + "_soma_f={}".format(soma_f_th)] = pks_filt[2]
    ana["soma_f={}_overlap_f={}".format(soma_f_th, f_th)] = overlap
    with open(path.join(cur_path, "clusts.pkl"), "wb") as f:
        pickle.dump(ana, f)

print("DONE")