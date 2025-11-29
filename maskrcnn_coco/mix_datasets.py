from PIL import Image
import numpy as np
import os
import glob
import shutil


imgs_envy = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/np_segs_small_renamed", "*.npy"), recursive=True))
segs_envy = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/np_segs_small_renamed", "*.npy"), recursive=True))
imgs_fuji = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/np_segs_small_renamed", "*.npy"), recursive=True))
masks_fruits = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/np_segs_small_renamed", "*.npy"), recursive=True))
masks_leaves = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/np_segs_small_renamed", "*.npy"), recursive=True))

# 
