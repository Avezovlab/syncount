#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 19:59:31 2021

@author: pierre
"""
import argparse

from math import sqrt, floor
from numpy import array, mean, isnan, ones, stack, where, vstack, amax
from numpy import bincount
from numpy.random import randint


from skimage.io import imread, imsave
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.feature import blob_log
from skimage.color import label2rgb
from skimage.morphology import remove_small_objects, erosion

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

parser = argparse.ArgumentParser(description="Detect synapses")

parser.add_argument("--force", action="store_true", default=False,
                    help="Force re-run analysis for already completed files")
parser.add_argument("--force-lock", action="store_true", default=False,
                    help="Force re-run analysis for locked files")
parser.add_argument("--shuffle", action="store_true", default=False,
                    help="shuffle input files")
parser.add_argument("--process-only", type=str,
                    help="part of name to recognize files to process",
                    dest="filter", default=None)
parser.add_argument("--no-RGB", action="store_true",
                    help="do not generate and save RGB images", dest="noRGB")
parser.add_argument("batch")
parser.add_argument("syn_type")

args = parser.parse_args()

batch = int(args.batch)
syn_type = args.syn_type
assert(syn_type in ["excit", "inhib"])


base_dir = "/mnt/data/mosab/data/batch{}/{}".format(batch, syn_type)
out_dir = "/mnt/data/mosab/analysis3/batch{}/{}".format(batch, syn_type)

chan_names = {"inhib": {0: "pv", 1: "geph", 2: "vgat"},
              "excit": {0: "pv", 1: "psd95", 2: "vglut"}}

intens_range = list([0.1 + k * 0.02 for k in range(44)])

min_neur_marker_size = 10000
max_neur_marker_size = 300000

#Batch 2
#syn_ths = {"inhib": [None, 0.2, 0.05],
#           "excit": [None, 0.1, 0.1]}

#Batch 3
#syn_ths = {"inhib": [None, 0.03, 0.05],
#           "excit": [None, 0.1, 0.05]}

#Batch 4
syn_ths = {"inhib": [None, 0.05, 0.05],
           "excit": [None, 0.05, 0.05]}

print("Processing: batch{}: {}".format(batch, syn_type))

def shuffle_ret(l):
    shuffle(l)
    return l

def log(m, f):
    f.write(m + "\n")
    print(m)

force = args.force
force_lock = args.force_lock
exp_dirs = listdir(base_dir)

if args.shuffle:
    files_to_treat = enumerate(shuffle_ret(exp_dirs))
else:
    files_to_treat = enumerate(exp_dirs)

for cnt, fname in files_to_treat:
    if not fname.endswith(".tif"):
        print("Skipped[{}/{}]: {} is not a .tif file"
              .format(cnt+1, len(exp_dirs), fname))
        continue

    if args.filter is not None and args.filter not in fname:
        print("Skipped[{}/{}]: {} filtered out on name"
              .format(cnt+1, len(exp_dirs), fname))
        continue

    out_exp_dir = path.join(out_dir, fname)

    if not (force or force_lock) and path.isfile(path.join(out_exp_dir, ".lock")):
        print("Skipped[{}/{}] {}: found lock file"
              .format(cnt+1, len(exp_dirs), fname))
        continue

    if not force and path.isdir(out_exp_dir):
        print("Skipped[{}/{}]: {} file already processed"
              .format(cnt+1, len(exp_dirs), fname))
        continue

    if not path.isdir(out_exp_dir):
        makedirs(out_exp_dir)

    print("Processing[{}/{}]: {}".format(cnt+1, len(exp_dirs), fname))

    #Put a lock on this analysis so that if multiple processings
    # happen in parallel they don't overlap
    Path(path.join(out_exp_dir, ".lock")).touch()

    logf = open(path.join(out_exp_dir, "log.txt"), 'w')

    imgs = imread(path.join(base_dir, fname))
    img_shape = (imgs.shape[0],) + imgs.shape[2:]

    ran = [False, False, False]
    pks = [None, [], []]

    th_soma = 0.0
    soma_labs_props = None

    log("Labeling", logf)
    chan_file = path.join(out_exp_dir, "labs_{}.tif".format(chan_names[syn_type][0]))
    if not force and path.isfile(chan_file):
        labs = imread(chan_file)
    else:
        log("  " + chan_names[syn_type][0], logf)
        ran[0] = True
        imgs_0 = gaussian(imgs[:,:,:,0], sigma=0.3)
        M = max(imgs_0.flatten())

        labs_0 = remove_small_objects(label(imgs_0 > intens_range[0] * M), min_size=min_neur_marker_size, in_place=True)
        labs_0 = remove_large_objects(labs_0, max_neur_marker_size)
        labs_0 = erosion(labs_0, ones((5,5,5)))
        props = regionprops(labs_0)
        best_area = mean([e.area for e in props])
        best_labs = array(labs_0)
        best_th = intens_range[0] * M
        log("   {}% ({:.2f}): {} (area={:.3f})".format(int(intens_range[0]*100), intens_range[0]*M, len(props), best_area), logf)
        for cnt in range(1, len(intens_range)):
            labs_0 = remove_small_objects(label(imgs_0 > intens_range[cnt] * M), min_size=min_neur_marker_size, in_place=True)
            labs_0 = remove_large_objects(labs_0, max_neur_marker_size)
            props = regionprops(labs_0)
            cur_area = mean([e.area for e in props])
            log("   {}% ({:.2f}): {} (area={:.3f})".format(int(intens_range[cnt]*100), intens_range[cnt]*M, len(props), cur_area), logf)
            if isnan(best_area) or cur_area > best_area:
                best_area = cur_area
                best_labs = array(labs_0)
                best_th = intens_range[cnt] * M
        labs = best_labs
        th_soma = best_th
        imsave(chan_file, labs.astype("uint16"))
    soma_labs_props = regionprops(labs)


    log("    Found {} Neurons".format(len((soma_labs_props))), logf)

    if not args.noRGB:
        col_file = path.join(out_exp_dir, "labs_{}_col.tif".format(chan_names[syn_type][0]))
        if force or not path.isfile(col_file):
            imsave(col_file, label2rgb(labs, bg_label=0,
                                       colors=[[randint(255), randint(255), randint(255)] for i in range(len(soma_labs_props))]).astype('uint8'))
    del labs

    for c_idx in [1, 2]:
        chan_file = path.join(out_exp_dir, "labs_{}.tif".format(chan_names[syn_type][c_idx]))
        if not force and path.isfile(chan_file):
            labs = imread(chan_file)
        else:
            log("  " + chan_names[syn_type][c_idx], logf)
            ran[c_idx] = True

            M = max(imgs[:,:,:,c_idx].flatten())
            pks[c_idx] = blob_log(((imgs[:,:,:,c_idx] / M) * 255).astype("uint8"), min_sigma=[2, 2, 1], max_sigma=[3, 3, 2], threshold=syn_ths[syn_type][c_idx], overlap=0)
            log("    Found {} {} clusters".format(pks[c_idx].shape[0], chan_names[syn_type][c_idx]), logf)

        if not args.noRGB:
            res = stack([(imgs[:,:,:,c_idx] / M) * 255, (imgs[:,:,:,c_idx] / M) * 255, (imgs[:,:,:,c_idx] / M) * 255], axis=3)
            s3 = sqrt(3)
            for pk in pks[c_idx]:
                rxy = floor(s3 * pk[3])
                rz = floor(s3 * pk[5])
                col = [randint(255), randint(255), randint(255)]
                for i in range(-rxy, rxy+1):
                    for j in range(-rxy, rxy+1):
                        for k in range(-rz, rz+1):
                            pos = pk[:3].astype("int") + array([k, i, j])
                            if (i/rxy)**2 + (j/rxy)**2 + (k/rz)**2 <= 1 and  0 <= pos[0] < res.shape[0] and 0 <= pos[1] < res.shape[1] and 0 <= pos[2] < res.shape[2]:
                                res[pos[0], pos[1], pos[2], :] = col
            col_file = path.join(out_exp_dir, "labs_{}_col.tif".format(chan_names[syn_type][c_idx]))
            if force or not path.isfile(col_file):
                imsave(col_file, res.astype("uint8"))

    if force or not path.isfile(path.join(out_exp_dir, "clusts.pkl")):
        log("  colocalization", logf)
        overlap = []
        for k in range(pks[1].shape[0]):
            idxs = where(((pks[1][k,0] - pks[2][:,0]) / amax(vstack([pks[1][k,3] * ones(pks[2].shape[0]), pks[2][:,3]]).T, axis=1))**2 +
                         ((pks[1][k,1] - pks[2][:,1]) / amax(vstack([pks[1][k,3] * ones(pks[2].shape[0]), pks[2][:,3]]).T, axis=1))**2 +
                         ((pks[1][k,2] - pks[2][:,2]) / amax(vstack([pks[1][k,5] * ones(pks[2].shape[0]), pks[2][:,5]]).T, axis=1))**2 < 1)
            overlap.extend([[k, l] for l in idxs[0].astype("int")])
        log("  Found {} synapses".format(len(overlap)), logf)


        with open(path.join(out_exp_dir, "clusts.pkl"), "wb") as f:
            pickle.dump({"th_" + chan_names[syn_type][0]: th_soma,
                         chan_names[syn_type][1]: pks[1], chan_names[syn_type][2]: pks[2],
                         'overlap': overlap}, f)
        del overlap
        del pks

    logf.close()
    #analysis is done, remove lock file
    remove(path.join(out_exp_dir, ".lock"))

    del imgs

print("DONE")
