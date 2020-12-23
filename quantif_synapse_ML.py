#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:32:42 2020

@author: pierre
"""
from os import listdir, makedirs
from os import path

import subprocess
import tempfile
import pickle

from skimage.external.tifffile import imread, imsave
from skimage.segmentation import join_segmentations
from skimage.measure import label, regionprops

from skimage.color import label2rgb
from numpy.random import randint

from random import shuffle
from pathlib import Path


def shuffle_ret(l):
    shuffle(l)
    return l

def log(m, f):
    f.write(m + "\n")
    print(m)


prog_bin = "/home/pierre/cam/ilastik-1.3.3post3-Linux/run_ilastik.sh"
prog_ch1_params = ["--headless",
                   "--project=/mnt/data/mosab/data/batch1/inhib/batch1_inhib_train_C2.ilp",
                   "--output_format=multipage tiff",
                   "--export_source=Simple Segmentation"]

prog_ch2_params = ["--headless",
                   "--project=/mnt/data/mosab/data/batch1/inhib/batch1_inhib_train_C3.ilp",
                   "--output_format=multipage tiff",
                   "--export_source=Simple Segmentation"]


#"C:/pierre/data"
in_base_dir =  "/mnt/data/mosab/data" #replace by the path to the folder
#containing the data

#C:/pierre/analysis
out_base_dir = "/mnt/data/mosab/analysis2" #replace by the path to the folder
                                    #where you want the output files
                                    #to be generated
batch = 1 #replace batch number here
syn_type = "inhib" #or "inhib"
generate_RGB = True #whether or not to generate RGB images of
                     #detected synapses
force = False #re-run already generated files

assert(syn_type in ["excit", "inhib"])

base_dir = "{}/batch{}/{}".format(in_base_dir, batch, syn_type)
out_dir = "{}/batch{}/{}".format(out_base_dir, batch, syn_type)

chan_names = {"inhib": {0: "pv", 1: "geph", 2: "vgat"},
              "excit": {0: "pv", 1: "psd95", 2: "vglut"}}


ran = [False, False, False]

exp_dirs = listdir(base_dir)

print("Processing: batch{}: {}".format(batch, syn_type))


exp_dirs = listdir(base_dir)
for cnt, fname in enumerate(shuffle_ret(exp_dirs)):
    if not fname.endswith(".tif"):
        print("Skipped[{}/{}]: {} is not a .tif file"
              .format(fname, cnt+1, len(exp_dirs)))
        continue

    out_exp_dir = path.join(out_dir, fname)
    if not path.isdir(out_exp_dir):
        makedirs(out_exp_dir)

    if path.isfile(path.join(out_exp_dir, ".lock")):
        print("Skipped[{}/{}] {}: found lock file"
              .format(cnt+1, len(exp_dirs), fname))
        continue

    print("Processing[{}/{}]: {}".format(cnt+1, len(exp_dirs), fname))
    tmp_dir = tempfile.TemporaryDirectory(dir="/tmp") 

    #Put a lock on this analysis so that if multiple processings
    # happen in parallel they don't overlap
    Path(path.join(out_exp_dir, ".lock")).touch()

    logf = open(path.join(out_exp_dir, "log.txt"), 'w')

    imgs = imread(path.join(base_dir, fname))
    img_shape = (imgs.shape[0],) + imgs.shape[2:]

    labs_props = [None, None, None]
    ran = [False, False, False]

    imsave(path.join(tmp_dir.name, "ch1.tiff"), imgs[:,1,:,:])
    imsave(path.join(tmp_dir.name, "ch2.tiff"), imgs[:,2,:,:])

    subp_res = subprocess.run([prog_bin] + prog_ch1_params + [path.join(tmp_dir.name, "ch1.tiff")], capture_output=True)
    assert(subp_res.returncode == 0)

    subp_res = subprocess.run([prog_bin] + prog_ch1_params + [path.join(tmp_dir.name, "ch2.tiff")], capture_output=True)
    assert(subp_res.returncode == 0)

    imgs_ch1_lab = imread(path.join(tmp_dir.name, "ch1_Simple Segmentation.tiff"))
    imgs_ch2_lab = imread(path.join(tmp_dir.name, "ch2_Simple Segmentation.tiff"))

    ch1_labs = label(imgs_ch1_lab == 1)
    ch2_labs = label(imgs_ch2_lab == 1)
    
    imsave(path.join(out_exp_dir, "labs_{}.tif".format(chan_names[syn_type][1])),
           ch1_labs.astype("uint16"))
    imsave(path.join(out_exp_dir, "labs_{}.tif".format(chan_names[syn_type][2])),
           ch2_labs.astype("uint16"))

    ch1_props = regionprops(ch1_labs)
    ch2_props = regionprops(ch2_labs)


    log("    Found {} {} clusters"
        .format(len(ch1_props), chan_names[syn_type][1]), logf)

    log("    Found {} {} clusters"
        .format(len(ch2_props), chan_names[syn_type][2]), logf)


    merged_labs = join_segmentations(ch1_labs, ch2_labs)

    merged_labs = merged_labs * (ch2_labs > 0) * (ch1_labs > 0)

    merged_props = regionprops(merged_labs)

    log("  Found {} synapses".format(len(merged_props)), logf)

    if generate_RGB:
        imsave(path.join(out_exp_dir, "merged_labs_col.tiff"),
               label2rgb(merged_labs, bg_label=0, colors=[[randint(255), randint(255), randint(255)] for i in range(len(merged_props))]).astype('uint8'))

    with open(path.join(out_exp_dir, "clusts.pkl"), "wb") as f:
        pickle.dump({"overlap": merged_props,
                     chan_names[syn_type][1]: ch1_props,
                     chan_names[syn_type][2]: ch2_props}, f)
