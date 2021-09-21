#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 19:59:31 2021

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

def remove_large_objects(ar, max_size):
    component_sizes = bincount(ar.ravel())
    too_big = component_sizes > max_size
    too_big_mask = too_big[ar]
    ar[too_big_mask] = 0
    return ar

def my_blob_log(image, min_sigma=1, max_sigma=50, num_sigma=10, th=.2,
             overlap=.5):
    image = img_as_float(image)

    # if both min and max sigma are scalar, function returns only one sigma
    scalar_sigma = (
        True if isscalar(max_sigma) and isscalar(min_sigma) else False
    )

    # Gaussian filter requires that sequence-type sigmas have same
    # dimensionality as image. This broadcasts scalar kernels
    if isscalar(max_sigma):
        max_sigma = full(image.ndim, max_sigma, dtype=float)
    if isscalar(min_sigma):
        min_sigma = full(image.ndim, min_sigma, dtype=float)

    # Convert sequence types to array
    min_sigma = asarray(min_sigma, dtype=float)
    max_sigma = asarray(max_sigma, dtype=float)


    scale = linspace(0, 1, num_sigma)[:, newaxis]
    sigma_list = scale * (max_sigma - min_sigma) + min_sigma

    # computing gaussian laplace
    # average s**2 provides scale invariance
    gl_images = [-gaussian_laplace(image, s) * mean(s) ** 2
                 for s in sigma_list]

    image_cube = stack(gl_images, axis=-1)

    exclude_border = _format_exclude_border(image.ndim, True)
    loc_max = peak_local_max(image_cube,
        min_distance = 5,
        threshold_abs=0.3,
        footprint=ones((3, 13, 13, 3)),
        threshold_rel=th,
        exclude_border=exclude_border)

    # Catch no peaks
    if loc_max.size == 0:
        return empty((0, 3))

    # Convert local_maxima to float64
    lm = loc_max.astype(float64)

    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma
    sigmas_of_peaks = sigma_list[loc_max[:, -1]]

    if scalar_sigma:
        # select one sigma column, keeping dimension
        sigmas_of_peaks = sigmas_of_peaks[:, 0:1]

    # Remove sigma index and replace with sigmas
    lm = hstack([lm[:, :-1], sigmas_of_peaks])

    return _prune_blobs(lm, 0, sigma_dim=sigmas_of_peaks.shape[1])

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

intens_range = list([0.1 + k * 0.02 for k in range(44)])


#params BATCH 1 TO 3 excit
#kernel_size=(11, 101, 101), clip_limit=0.01
#min_sigma=[0.2, 0.2, 1], max_sigma=[1.1, 1.1, 2], th=0.25

#params BATCH 4 excit
#pks[c_idx] = my_blob_log(cur_img, min_sigma=[0.2, 0.2, 1], max_sigma=[1.1, 1.1, 2], th=0.15)

blob_ths = {1 : {"inhib": {"geph": 0.2, "vgat": 0.2},
                 "excit": {"psd95": 0.15, "vglut": 0.15}},
            2 : {"inhib": {"geph": 0.1, "vgat": 0.5},
                 "excit": {"psd95": 0.15, "vglut": 0.15}},
            3 : {"inhib": {"geph": 0.05, "vgat": 0.15},
                 "excit": {"psd95": 0.15, "vglut": 0.15}},
            4 : {"inhib": {"geph": 0.01, "vgat": 0.2},
                 "excit": {"psd95": 0.15, "vglut": 0.15}}}

min_sigs = {1: {"inhib": {"geph": [0.6, 0.6, 0.7], "vgat": [0.4, 0.4, 0.7]},
                "excit": {"psd95": [0.2, 0.2, 1], "vglut": [0.2, 0.2, 1]}},
            2: {"inhib": {"geph": [0.6, 0.6, 0.7], "vgat": [0.4, 0.4, 0.7]},
                "excit": {"psd95": [0.2, 0.2, 1], "vglut": [0.2, 0.2, 1]}},
            3: {"inhib": {"geph": [0.6, 0.6, 0.7], "vgat": [0.6, 0.6, 0.7]},
                "excit": {"psd95": [0.2, 0.2, 1], "vglut": [0.2, 0.2, 1]}},
            4: {"inhib": {"geph": [0.6, 0.6, 0.7], "vgat": [0.6, 0.6, 0.7]},
                "excit": {"psd95": [0.2, 0.2, 1], "vglut": [0.2, 0.2, 1]}}}

max_sigs = {1: {"inhib": {"geph": [1.1, 1.1, 1.5], "vgat": [1, 1, 1.5]},
                "excit": {"psd95": [1.1, 1.1, 2], "vglut": [1.1, 1.1, 2]}},
            2: {"inhib": {"geph": [1.1, 1.1, 1.5], "vgat": [1, 1, 1.5]},
                "excit": {"psd95": [1.1, 1.1, 2], "vglut": [1.1, 1.1, 2]}},
            3: {"inhib": {"geph": [1.1, 1.1, 1.5], "vgat": [1.6, 1.6, 1.5]},
                "excit": {"psd95": [1.1, 1.1, 2], "vglut": [1.1, 1.1, 2]}},
            4: {"inhib": {"geph": [1.1, 1.1, 1.5], "vgat": [1.6, 1.6, 1.5]},
                "excit": {"psd95": [1.1, 1.1, 2], "vglut": [1.1, 1.1, 2]}},}

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
        print("Skipped[{}/{}]: {} is not a min_sigma=[0.2, 0.2, 1], max_sigma=[1.1, 1.1, 2], th=0.25).tif file"
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
        imsave(chan_file, labs.astype("uint16"), check_contrast=False)
    soma_labs_props = regionprops(labs)


    log("    Found {} Neurons".format(len((soma_labs_props))), logf)

    if not args.noRGB:
        col_file = path.join(out_exp_dir, "labs_{}_col.tif".format(chan_names[syn_type][0]))
        if force or not path.isfile(col_file):
            imsave(col_file, label2rgb(labs, bg_label=0,
                                        colors=[[randint(255), randint(255), randint(255)] for i in range(len(soma_labs_props))]).astype('uint8'),
                    check_contrast=False)
    del labs

    for c_idx in [1, 2]:
        chan_name = chan_names[syn_type][c_idx]
        chan_file = path.join(out_exp_dir, "labs_{}.tif".format(chan_name))

        if not force and path.isfile(chan_file):
            labs = imread(chan_file)
        else:
            log("  " + chan_names[syn_type][c_idx], logf)
            ran[c_idx] = True

            M = max(imgs[:,:,:,c_idx].flatten())
            cur_img = equalize_adapthist(imgs[:,:,:,c_idx], kernel_size=(11, 101, 101), clip_limit=0.01)
            #img_filt = gaussian(cur_img, sigma=3)
            #cur_img = cur_img - img_filt

            # s = 0.3
            # gi = -gaussian_laplace(cur_img, s) * mean(s) ** 2
            # imsave(path.join(out_exp_dir, "gi_{}.tif".format(chan_names[syn_type][c_idx])), gi, check_contrast=False)

            # loc_max = peak_local_max(gi,
            #                         min_distance = 15,
            #                         footprint=ones((3, 9, 9)),
            #                         threshold_abs=0.01,
            #                         exclude_border=False)
            # pks[c_idx] = hstack([loc_max,  1.5 * ones((loc_max.shape[0], 2)), 1 * ones((loc_max.shape[0], 1))])

            print("    Blob th:", blob_ths[batch][syn_type][chan_name],
                  "min sigs:", min_sigs[batch][syn_type][chan_name],
                  "max sigs:", max_sigs[batch][syn_type][chan_name])
            pks[c_idx] = my_blob_log(cur_img, min_sigma=min_sigs[batch][syn_type][chan_name],
                                     max_sigma=max_sigs[batch][syn_type][chan_name],
                                     th=blob_ths[batch][syn_type][chan_name])
            log("    Found {} {} clusters".format(pks[c_idx].shape[0], chan_name), logf)

        if not args.noRGB:
            res = stack([(imgs[:,:,:,c_idx] / M) * 255, (imgs[:,:,:,c_idx] / M) * 255, (imgs[:,:,:,c_idx] / M) * 255], axis=3)
            s3 = sqrt(3)
            for pk in pks[c_idx]:
                rxy = ceil(s3 * pk[3])
                rz = ceil(s3 * pk[5])
                col = [randint(255), randint(255), randint(255)]
                for i in range(-rxy, rxy+1):
                    for j in range(-rxy, rxy+1):
                        for k in range(-rz, rz+1):
                            pos = pk[:3].astype("int") + array([k, i, j])
                            if (i/rxy)**2 + (j/rxy)**2 + (k/rz)**2 <= 1 and  0 <= pos[0] < res.shape[0] and 0 <= pos[1] < res.shape[1] and 0 <= pos[2] < res.shape[2]:
                                res[pos[0], pos[1], pos[2], :] = col
            col_file = path.join(out_exp_dir, "labs_{}_col.tif".format(chan_names[syn_type][c_idx]))
            if force or not path.isfile(col_file):
                imsave(col_file, res.astype("uint8"), check_contrast=False)
            del res

    overlap_dist = 2
    if force or not path.isfile(path.join(out_exp_dir, "clusts.pkl")):
        log("  colocalization", logf)
        overlap = []
        for k in range(pks[1].shape[0]):
            idxs = where(((pks[1][k,0] - pks[2][:,0]) / amax(vstack([pks[1][k,3] * ones(pks[2].shape[0]), pks[2][:,3]]).T, axis=1))**2 +
                         ((pks[1][k,1] - pks[2][:,1]) / amax(vstack([pks[1][k,3] * ones(pks[2].shape[0]), pks[2][:,3]]).T, axis=1))**2 +
                         ((pks[1][k,2] - pks[2][:,2]) / amax(vstack([pks[1][k,5] * ones(pks[2].shape[0]), pks[2][:,5]]).T, axis=1))**2 < overlap_dist)
            overlap.extend([[k, l] for l in idxs[0].astype("int")])
        log("  Found {} synapses".format(len(overlap)), logf)


        with open(path.join(out_exp_dir, "clusts.pkl"), "wb") as f:
            pickle.dump({"th_" + chan_names[syn_type][0]: th_soma,
                         chan_names[syn_type][1]: pks[1], chan_names[syn_type][2]: pks[2],
                         'overlap': overlap, "overlap_dist": overlap_dist}, f)
        del overlap
        del pks

    logf.close()
    #analysis is done, remove lock file
    remove(path.join(out_exp_dir, ".lock"))

    del imgs

print("DONE")
