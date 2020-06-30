#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:38:06 2020

@author: pierre
"""
from numpy import zeros, array, mean, std, where, sum, sqrt, min, max, argmin, histogram

from matplotlib import pyplot as plt
from math import ceil

from collections import Counter
from numpy.random import randint

from os import path, listdir, makedirs

import pickle

def log(m, f):
    f.write(m + "\n")
    print(m)

#Bologna4_Inhibitory_509.lif - 509_Field1_Right.tif -> no clear synapses

chan_names = {"inhib": {0: "pv", 1: "geph", 2: "vgat"},
              "excit": {0: "pv", 1: "psd95", 2: "vglut"}}

#batch = 2
batch = 3

#neur_type = "inhib"
neur_type = "excit"

base_dir = "/mnt/data/mosab/analysis/batch{}/{}".format(batch, neur_type)

dirs = listdir(base_dir)
nsynapses = {}
nclusts = [{}, {}]
clust_sizes = [{}, {}]
for i in range(len(dirs)):
    fname = dirs[i]
    cur_path = path.join(base_dir, fname)

    if batch == 3 or neur_type == "excit" and batch not in [1,4]:
        cat = fname.split(" ")[0]
    elif batch == 4:
        cat = fname.split("_")[2].split(".")[0]
    else:
        cat = fname.split("_")[0]

    if not path.isfile(path.join(cur_path, "clusts.pkl")):
        print("Missing " + cur_path)
        continue

    with open(path.join(cur_path, "clusts.pkl"), 'rb') as f:
        ana = pickle.load(f)

    if cat not in nsynapses.keys():
        nsynapses[cat] = []
        nclusts[0][cat] = []
        nclusts[1][cat] = []
        clust_sizes[0][cat] = []
        clust_sizes[1][cat] = []
    nsynapses[cat].append(len(ana["overlap"]))
    nclusts[0][cat].append(len(ana[chan_names[neur_type][1]]))
    nclusts[1][cat].append(len(ana[chan_names[neur_type][2]]))
    clust_sizes[0][cat].extend([e[4] for e in ana[chan_names[neur_type][1]]])
    clust_sizes[1][cat].extend([e[4] for e in ana[chan_names[neur_type][2]]])

print("Cluster Sizes:")
plt.figure(figsize=(10,10))
for cat,v in clust_sizes[0].items():
    print("{}/{}: {} ± {}".format(cat, chan_names[neur_type][0], mean(v), std(v)))
    o, b = histogram(v, bins=[1+i*0.05 for i in range(0, 100)])
    plt.plot(b[:-1], o / sum(o))
for cat,v in clust_sizes[1].items():
    print("{}/{}: {} ± {}".format(cat, chan_names[neur_type][1], mean(v), std(v)))
    o, b = histogram(v, bins=[1+i*0.05 for i in range(0, 100)])
    plt.plot(b[:-1], o / sum(o), '--')
plt.xlabel('Cluster radius')
plt.ylabel('Frequency')

print()
logf = open("/tmp/batch{}_{}_quantif.txt".format(batch, neur_type), "w")
log("Number of {} clusters:".format(chan_names[neur_type][1]), logf)

plt.figure(figsize=(10,10))
plt.subplot(3, 1, 1)
plt.bar(range(len(nsynapses.keys())), [mean(e) for e in nclusts[0].values()])
ks = list(nsynapses.keys())
for k in range(len(nsynapses.keys())):
    log("{}: AVG={:.3f} SEM={:.3f} n={} (values:[{}])".format(list(nclusts[0].keys())[k], mean(nclusts[0][ks[k]]), std(nclusts[0][ks[k]]), len(nclusts[0][ks[k]]), ",".join([str(e) for e in nclusts[0][ks[k]]])), logf)
    plt.plot([k, k], mean(nclusts[0][ks[k]]) + std(nclusts[0][ks[k]]) / sqrt(len(nclusts[0][ks[k]])) * array([-1, 1]), 'black')
plt.xticks(range(len(nclusts[0].keys())), nclusts[0].keys())
plt.ylabel('AVG Number of {} clusters'.format(chan_names[neur_type][1]))

log("", logf)
log("Number of {} clusters:".format(chan_names[neur_type][2]), logf)
plt.subplot(3, 1, 2)
plt.bar(range(len(nsynapses.keys())), [mean(e) for e in nclusts[1].values()])
ks = list(nsynapses.keys())
for k in range(len(nsynapses.keys())):
    log("{}: AVG={:.3f} SEM={:.3f} n={} (values:[{}])".format(list(nclusts[1].keys())[k], mean(nclusts[1][ks[k]]), std(nclusts[1][ks[k]]), len(nclusts[1][ks[k]]), ",".join([str(e) for e in nclusts[1][ks[k]]])), logf)
    plt.plot([k, k], mean(nclusts[1][ks[k]]) + std(nclusts[1][ks[k]]) / sqrt(len(nclusts[1][ks[k]])) * array([-1, 1]), 'black')
plt.xticks(range(len(nclusts[1].keys())), nclusts[1].keys())
plt.ylabel('AVG Number of {} clusters'.format(chan_names[neur_type][2]))

log("", logf)
log("Number of synapses:", logf)
plt.subplot(3, 1, 3)
plt.bar(range(len(nsynapses.keys())), [mean(e) for e in nsynapses.values()])
ks = list(nsynapses.keys())
Msyn = 0
for k in range(len(nsynapses.keys())):
    log("{}: AVG={:.3f} SEM={:.3f} n={} (values:[{}])".format(list(nsynapses.keys())[k], mean(nsynapses[ks[k]]), std(nsynapses[ks[k]]), len(nsynapses[ks[k]]), ",".join([str(e) for e in nsynapses[ks[k]]])), logf)
    plt.plot([k, k], mean(nsynapses[ks[k]]) + std(nsynapses[ks[k]]) / sqrt(len(nsynapses[ks[k]])) * array([-1, 1]), 'black')
    Msyn = max([Msyn, mean(nsynapses[ks[k]])])
plt.xticks(range(len(nsynapses.keys())), nsynapses.keys())
plt.yticks([e*100 for e in range(int(ceil(Msyn / 100))+1)])
plt.ylabel('AVG Number of synapses')
plt.savefig("/tmp/batch{}_{}_results.pdf".format(batch, neur_type), bbox_inches='tight',pad_inches = 0)

logf.close()
