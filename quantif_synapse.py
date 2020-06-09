#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:13:03 2020

@author: pierre
"""
import argparse

from numpy import array, mean, min, argmin, NaN
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
from os import path, listdir, makedirs


def remove_large_objects(ar, max_size):
    component_sizes = bincount(ar.ravel())
    too_big = component_sizes > max_size
    too_big_mask = too_big[ar]
    ar[too_big_mask] = 0
    return ar

parser = argparse.ArgumentParser(description="Detect synapses")

parser.add_argument("batch")
parser.add_argument("syn_type")

args = parser.parse_args()

batch = int(args.batch)
syn_type = args.syn_type
assert(syn_type in ["excit", "inhib"])


base_dir = "/mnt/data/cam/mosab/data/batch{}/{}".format(batch, syn_type)
out_dir = "/mnt/data/cam/mosab/analysis/batch{}/{}".format(batch, syn_type)

chan_names = {"inhib": {0: "pv", 1: "geph", 2: "vgat"},
              "excit": {0: "pv", 1: "psd95", 2: "vglut"}}

default_params = {"inhib": {"ths": [0.1, 0.6, 0.6],
                            "sigmas": [0.1, 0.1, 0.1]},
                  "excit": {"ths": [0.1, 0.6, 0.6],
                            "sigmas": [0.1, 0.1, 0.1]}}

print("Processing: batch{}: {}".format(batch, syn_type))

def shuffle_ret(l):
    shuffle(l)
    return l

force = False
for fname in shuffle_ret(listdir(base_dir)): #[e for e in listdir(base_dir) if e.startswith("522")]:
    if not fname.endswith(".tif"):
        print("Skipped: " + fname + " not a .tif file")
        continue
    print("Processing: " + fname)

    out_exp_dir = path.join(out_dir, fname)
    if not path.isdir(out_exp_dir):
        makedirs(out_exp_dir)

    imgs = imread(path.join(base_dir, fname))
    img_shape = (imgs.shape[0],) + imgs.shape[2:]

    ths = default_params[syn_type]["ths"]
    sigmas = default_params[syn_type]["sigmas"]

    labs_props = [None, None, None]
    ran = [False, False, False]

    print("Labeling")
    chan_file = path.join(out_exp_dir, "labs_{}.tif".format(chan_names[syn_type][0]))
    if not force and path.isfile(chan_file):
        labs = imread(chan_file)
    else:
        print("  " + chan_names[syn_type][0])
        ran[0] = True
        imgs_0 = gaussian(imgs[:,0,:,:], sigma=sigmas[0])
        labs_0 = remove_small_objects(label(imgs_0 > ths[0]), min_size=5000, in_place=True)
        labs_0 = remove_large_objects(labs_0, 200000)
        props = regionprops(labs_0)
        best_area = mean([e.area for e in props])
        best_labs = array(labs_0)
        best_th = ths[0]
        cnt = 0
        while cnt < 15:
            labs_0 = remove_small_objects(label(imgs_0 > ths[0]), min_size=5000, in_place=True)
            labs_0 = remove_large_objects(labs_0, 200000)
            props = regionprops(labs_0)
            cur_area = mean([e.area for e in props])
            if cur_area > best_area:
                best_area = cur_area
                best_labs = array(labs_0)
                best_th = ths[0]
            ths[0] += 0.02
            cnt += 1
        labs = best_labs
        ths[0] = best_th
        imsave(chan_file, labs.astype("uint16"))
    labs_props[0] = regionprops(labs)
 

    col_file = path.join(out_exp_dir, "labs_{}_col.tif".format(chan_names[syn_type][0]))
    if force or not path.isfile(col_file):
        imsave(col_file, label2rgb(labs, bg_label=0,
                colors=[[randint(255), randint(255), randint(255)] for i in range(len(labs_props[0]))]).astype('uint8'))
    del labs

    chan_file = path.join(out_exp_dir, "labs_{}.tif".format(chan_names[syn_type][1]))
    if not force and path.isfile(chan_file):
        labs = imread(chan_file)
    else:
        print("  " + chan_names[syn_type][1])
        ran[1] = True
        imgs_1 = gaussian(imgs[:,1,:,:], sigma=sigmas[1])
        labs_1 = remove_small_objects(label(imgs_1 > ths[1]), min_size=11, in_place=True)
        labs_1 = remove_large_objects(labs_1, 300)
        props = regionprops(labs_1)
        max_clusts = len(props)
        best_labs = array(labs_1)
        best_th = ths[1]
        cnt = 0
        while cnt < 15:
            labs_1 = remove_small_objects(label(imgs_1 > ths[1]), min_size=11, in_place=True)
            labs_1 = remove_large_objects(labs_1, 300)
            props = regionprops(labs_1)
            if len(props) > max_clusts:
                max_clusts = len(props)
                best_labs = array(labs_1)
                best_th = ths[1]
            ths[1] -= 0.02
            cnt += 1
        labs = best_labs
        ths[1] = best_th
        imsave(chan_file, labs.astype("uint16"))
    labs_props[1] = regionprops(labs)

    col_file = path.join(out_exp_dir, "labs_{}_col.tif".format(chan_names[syn_type][1]))
    if force or not path.isfile(col_file):
        imsave(col_file, label2rgb(labs, bg_label=0,
                colors=[[randint(255), randint(255), randint(255)] for i in range(len(labs_props[1]))]).astype('uint8'))
    del labs

    chan_file = path.join(out_exp_dir, "labs_{}.tif".format(chan_names[syn_type][2]))
    if not force and path.isfile(chan_file):
        labs = imread(chan_file)
    else:
        print("  " + chan_names[syn_type][2])
        ran[2] = True
        imgs_2 = gaussian(imgs[:,2,:,:], sigma=sigmas[2])
        labs_2 = remove_small_objects(label(imgs_2 > ths[2]), min_size=11, in_place=True)
        labs_2 = remove_large_objects(labs_2, 300)
        props = regionprops(labs_2)
        max_clusts = len(props)
        best_labs = labs_2
        best_th = ths[2]
        cnt = 0
        while cnt < 15:
            labs_2 = remove_small_objects(label(imgs_2 > ths[2]), min_size=11, in_place=True)
            labs_2 = remove_large_objects(labs_2, 300)
            props = regionprops(labs_2)
            if len(props) > max_clusts:
                max_clusts = len(props)
                best_labs = array(labs_2)
                best_th = ths[2]
            ths[2] -= 0.02
            cnt += 1
        labs = best_labs
        ths[2] = best_th
        imsave(chan_file, labs.astype("uint16"))
    labs_props[2] = regionprops(labs)
 
    col_file = path.join(out_exp_dir, "labs_{}_col.tif".format(chan_names[syn_type][2]))
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
        print("Finding synapses")

        labs_0_set = [e.label for e in labs_props[0]]
        labs_0_pxs = {e.label:set([ravel_multi_index(f, img_shape) for f in e.coords])
                        for e in labs_props[0]}

        clusts = {}
        clusts_neurons = {}
        for k in [1, 2]:
            name = chan_names[syn_type][k]
            print("  " + name)
            clusts[name] = []
            clusts_neurons[name] = []
            for e in labs_props[k]:
                pxs = e.coords
                cent = mean(pxs, axis=0)
                rad = e.equivalent_diameter/2
    
                clusts[name].append(array([e.label, cent[0], cent[1], cent[2], rad]))
                clust_pxs = set([ravel_multi_index(f, img_shape) for f in e.coords])
                clusts_neurons[name].extend([[e.label, l] for l in labs_0_set if
                                            clust_pxs & labs_0_pxs[l] != set([])])


        print("  colocalization")
        name1 = chan_names[syn_type][1]
        name2 = chan_names[syn_type][2]
        #near_neighb = []
        overlap = []
        for k in range(len(clusts[name1])):
            c = array(clusts[name1][k][1:4])
            overlap.extend([[k,l] for l in range(len(clusts[name2]))
                            if sum((c - array(clusts[name2][l][1:4]))**2) < (clusts[name1][k][4] + clusts[name2][l][4])**2])

            #if dists.size > 0:
            #    near_neighb.append([argmin(dists), min(dists)])
            #else:
            #    near_neighb.append([-1, NaN])

        with open(path.join(out_exp_dir, "clusts.pkl"), "wb") as f:
            pickle.dump({name1: clusts[name1], "{}_neurons".format(name1): clusts_neurons[name1],
                         name2: clusts[name2], "{}_neurons".format(name2): clusts_neurons[name2],
                         'overlap': overlap}, f)   #, 'near_neighb': near_neighb}, f)
        del overlap
        del clusts
        del clusts_neurons

    del imgs
print("DONE")
