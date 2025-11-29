from PIL import Image
import numpy as np
import os
import glob
import shutil

LABEL_DIR = "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/np_segs_small"
SAVE_DIR = "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/np_segs_small_renamed"

segments = glob.glob(os.path.join(LABEL_DIR, "*.npy"), recursive=True)
# change all name format from "0003_label_rgb_0001.npy" to "0003_rgb_0001.npy" and save in new folder
for seg in segments:
    base = seg.split("/")[-1]
    new_name = base.replace("_label","")
    shutil.copy(seg, os.path.join(SAVE_DIR, new_name))