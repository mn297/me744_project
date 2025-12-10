from PIL import Image
import numpy as np
import os
import glob
import shutil



gts = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/np_segs_small_renamed", "*.npy"), recursive=True))
print(gts)
for g, gt in enumerate(gts):
    print(gt)
    annot = np.load(gt)
    assert len(np.unique(annot)) == 4
    annot[annot != 0] = 1
    np.save(os.path.join("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/np_segs_2lbs", gt.split("/")[-1]), annot)


