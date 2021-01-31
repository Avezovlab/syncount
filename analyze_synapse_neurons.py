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

chan_names = {"inhib": {0: "pv", 1: "geph", 2: "vgat"},
              "excit": {0: "pv", 1: "psd95", 2: "vglut"}}

batch = 4 #batch number
neur_type = "excit" #excit" #or "inhib"
in_base_dir = "/mnt/data/mosab/analysis3" #"/mnt/data/mosab/analysis2/" #This corresponds to the
                                   #out_base_dir of the
                                   #quantif_synapse.py script
f_th = 1.5
soma = "soma_f=1_overlap_f={}".format(f_th)

base_dir = "{}/batch{}/{}".format(in_base_dir, batch, neur_type)

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

    print("Loading {}/{}: {}".format(i, len(dirs), fname))


    with open(path.join(cur_path, "clusts.pkl"), 'rb') as f:
        ana = pickle.load(f)

    if cat not in nsynapses.keys():
        if cat.startswith("ID"):
            cat = cat[2:]

        nsynapses[cat] = []
        nclusts[0][cat] = []
        nclusts[1][cat] = []
        clust_sizes[0][cat] = []
        clust_sizes[1][cat] = []
    if f_th == 1:
        nsynapses[cat].append(len(ana["overlap"]))
    elif soma != "":
        nsynapses[cat].append(len(ana[soma]))
    else:
        nsynapses[cat].append(len(ana["overlap_f={}".format(f_th)]))

    if soma != "":
        nclusts[0][cat].append(len(ana[chan_names[neur_type][1] + "_soma_f=1"]))
        nclusts[1][cat].append(len(ana[chan_names[neur_type][2] + "_soma_f=1"]))
    else:
        nclusts[0][cat].append(len(ana[chan_names[neur_type][1]]))
        nclusts[1][cat].append(len(ana[chan_names[neur_type][2]]))
#    clust_sizes[0][cat].extend([e[4] for e in ana[chan_names[neur_type][1]]])
#    clust_sizes[1][cat].extend([e[4] for e in ana[chan_names[neur_type][2]]])

cats = sorted(nsynapses.keys(), key=lambda e: int(e))


#print("Cluster Sizes:")
#plt.figure(figsize=(10,10))
#for cat in cats:
#    v = clust_sizes[0][cat]
#    print("{}/{}: {} ± {}".format(cat, chan_names[neur_type][0], mean(v), std(v)))
#    o, b = histogram(v, bins=[1+i*0.05 for i in range(0, 100)])
#    plt.plot(b[:-1], o / sum(o))
#for cat in cats:
#    v = clust_sizes[1][cat]
#    print("{}/{}: {} ± {}".format(cat, chan_names[neur_type][1], mean(v), std(v)))
#    o, b = histogram(v, bins=[1+i*0.05 for i in range(0, 100)])
#    plt.plot(b[:-1], o / sum(o), '--')
#plt.xlabel('Cluster radius')
#plt.ylabel('Frequency')

print()
logf = open("/tmp/batch{}_{}_quantif.txt".format(batch, neur_type), "w")
log("Number of {} clusters:".format(chan_names[neur_type][1]), logf)

plt.figure(figsize=(10,10))
plt.subplot(3, 1, 1)
plt.bar(range(len(cats)), [mean(nclusts[0][cat]) for cat in cats])
for k,cat in enumerate(cats):
    log("{}: AVG={:.3f} SEM={:.3f} n={} (values:[{}])".format(cat, mean(nclusts[0][cat]), std(nclusts[0][cat]), len(nclusts[0][cat]), ",".join([str(e) for e in nclusts[0][cat]])), logf)
    plt.plot([k, k], mean(nclusts[0][cat]) + std(nclusts[0][cat]) / sqrt(len(nclusts[0][cat])) * array([-1, 1]), 'black')
plt.xticks(range(len(cats)), cats)
plt.ylabel('AVG Number of {} clusters'.format(chan_names[neur_type][1]))

log("", logf)
log("Number of {} clusters:".format(chan_names[neur_type][2]), logf)
plt.subplot(3, 1, 2)
plt.bar(range(len(cats)), [mean(nclusts[1][cat]) for cat in cats])
for k,cat in enumerate(cats):
    log("{}: AVG={:.3f} SEM={:.3f} n={} (values:[{}])".format(cat, mean(nclusts[1][cat]), std(nclusts[1][cat]), len(nclusts[1][cat]), ",".join([str(e) for e in nclusts[1][cat]])), logf)
    plt.plot([k, k], mean(nclusts[1][cat]) + std(nclusts[1][cat]) / sqrt(len(nclusts[1][cat])) * array([-1, 1]), 'black')
plt.xticks(range(len(cats)), cats)
plt.ylabel('AVG Number of {} clusters'.format(chan_names[neur_type][2]))

log("", logf)
log("Number of synapses:", logf)
plt.subplot(3, 1, 3)
plt.bar(range(len(cats)), [mean(nsynapses[cat]) for cat in cats])
Msyn = 0
for k,cat in enumerate(cats):
    log("{}: AVG={:.3f} SEM={:.3f} n={} (values:[{}])".format(cat, mean(nsynapses[cat]), std(nsynapses[cat]) / len(nsynapses[cat]), len(nsynapses[cat]), ",".join([str(e) for e in nsynapses[cat]])), logf)
    plt.plot([k, k], mean(nsynapses[cat]) + std(nsynapses[cat]) / sqrt(len(nsynapses[cat])) * array([-1, 1]), 'black')
    Msyn = max([Msyn, mean(nsynapses[cat])])
plt.xticks(range(len(cats)), cats)
plt.yticks([e*100 for e in range(int(ceil(Msyn / 100))+1)])
plt.ylabel('AVG Number of synapses')
plt.savefig("/tmp/batch{}_{}_results.pdf".format(batch, neur_type), bbox_inches='tight',pad_inches = 0)

logf.close()
