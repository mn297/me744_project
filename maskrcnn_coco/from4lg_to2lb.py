from PIL import Image
import numpy as np
import os
import glob
import shutil


lb4_masks = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/np_segs_test", "*.npy"), recursive=True))
lb2_save = "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/mixed/test/seg"


for i in range(len(lb4_masks)):
    seg4 = np.load(lb4_masks[i])
    label_map = np.zeros((seg4.shape[0], seg4.shape[1]), dtype=np.uint8)
    label_map[seg4 != 0] = 1  # Map all fruit classes to 1  
    np.save(os.path.join(lb2_save, lb4_masks[i].split("/")[-1]), label_map.astype(np.uint8))
    print("Processed and saved:", lb4_masks[i].split("/")[-1])




