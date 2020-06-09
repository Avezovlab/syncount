#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:38:06 2020

@author: pierre
"""
from numpy import zeros, array, mean, std, where, sum, sqrt, min, max, argmin

from matplotlib import pyplot as plt

from collections import Counter
from numpy.random import randint

from os import path, listdir, makedirs

import pickle

#imread("/mnt/data/cam/ratiometry/sec61_ERmcherry/Image 3_Airyscan Processing.tiff")

#batch = 2
batch = 1

#neur_type = "excit"
neur_type = "excit"

base_dir = "/mnt/data/cam/mosab/analysis/batch{}/{}".format(batch, neur_type)

dirs = listdir(base_dir)
nclusters = {}
for i in range(len(dirs)):
    fname = dirs[i]
    cur_path = path.join(base_dir, fname)

    if neur_type == "excit" and batch != 1:
        cat = fname.split(" ")[0]
    else:
        cat = fname.split("_")[0]

    if not path.isfile(path.join(cur_path, "clusts.pkl")):
        print("Missing " + cur_path)
        continue

    with open(path.join(cur_path, "clusts.pkl"), 'rb') as f:
        ana = pickle.load(f)

    if cat not in nclusters.keys():
        nclusters[cat] = []
    nclusters[cat].append(len(ana["overlap"]))


plt.figure()
plt.bar(range(len(nclusters.keys())), [mean(e) for e in nclusters.values()])
ks = list(nclusters.keys())
for k in range(len(nclusters.keys())):
    print("{} {}".format(mean(nclusters[ks[k]]), std(nclusters[ks[k]])))
    plt.plot([k, k], mean(nclusters[ks[k]]) + std(nclusters[ks[k]]) / sqrt(len(nclusters[ks[k]])) * array([-1, 1]), 'black')
plt.xticks(range(len(nclusters.keys())), nclusters.keys())
plt.ylabel('AVG Number of synapses')
plt.savefig("/tmp/batch{}_{}_synapse_count.pdf".format(batch, neur_type), bbox_inches='tight',pad_inches = 0)
