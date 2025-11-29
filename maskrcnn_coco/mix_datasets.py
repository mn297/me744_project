from PIL import Image
import numpy as np
import os
import glob
import shutil
# import scikit
# import skimage label function
from scipy.ndimage import label
import random

imgs_save = "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/mixed/imgs"
segs_save = "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/mixed/segs"
save_seg_vis = "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/mixed/seg_vis_test"
save_img_vis = "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/mixed/img_vis_test"
vis_only = True 

imgs_envy = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/np_imgs_test", "*.npy"), recursive=True))
segs_envy = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/np_segs_2lbs", "*.npy"), recursive=True))
imgs_fuji = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/me744_project/datasets/Fuji-Apple-Segmentation_unet/np_imgs_train", "*.npy"), recursive=True))
masks_fruits = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/me744_project/datasets/Fuji-Apple-Segmentation_unet/np_segs_train", "*.npy"), recursive=True))
# masks_leaves = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/np_segs_small_renamed", "*.npy"), recursive=True))

label_fruit = 0

for i in range(10):
    inst_masks = 0
    img_e = np.load(imgs_envy[i]) # rgb
    seg_e = np.load(segs_envy[i])
    n_fruits = random.randint(1, 10)
    while inst_masks < n_fruits:
        load_ind = random.randint(0, len(imgs_fuji)-1)
        img_f = np.load(imgs_fuji[load_ind])
        seg_f = np.load(masks_fruits[load_ind])
        # seg_l = np.load(masks_leaves[i])
        apples = label(seg_f != 0)[0]
        # discard the background component
        apples[seg_f == 0] = 0
        for inst in range(1, np.unique(apples).shape[0]):
            inst_mask = apples == inst
            # get bounding box for the instance
            ys, xs = np.where(inst_mask)
            y1, y2 = np.min(ys), np.max(ys)
            x1, x2 = np.min(xs), np.max(xs)
            h, w = y2 - y1, x2 - x1
        
            # select a random spot in the envy image to paste
            max_y = img_e.shape[0] - h - 1
            max_x = img_e.shape[1] - w - 1
            # if max_y <= 0 or max_x <= 0:
            #     continue
            offset_y = random.randint(0, max_y)
            offset_x = random.randint(0, max_x)
            # paste the instance onto the envy image (only past the instance pixels, not teh square)
            img_e[offset_y:offset_y+h, offset_x:offset_x+w, :][inst_mask[y1:y2, x1:x2]] = img_f[y1:y2, x1:x2, :][inst_mask[y1:y2, x1:x2]]
            # update the segmentation mask accordingly (make sure to offset the class values)
            seg_e[offset_y:offset_y+h, offset_x:offset_x+w][inst_mask[y1:y2, x1:x2]] = label_fruit

            inst_masks += 1

    # save the mixed image and segmentation mask
    if not vis_only:
        np.save(os.path.join(imgs_save, imgs_envy[i].split("/")[-1]), img_e)
        np.save(os.path.join(segs_save, segs_envy[i].split("/")[-1]), seg_e)

    # save a visualization of the segmentation mask, with different colors for each class
    seg_vis = np.zeros((seg_e.shape[0], seg_e.shape[1], 3), dtype=np.uint8)
    seg_vis[seg_e == 0] = [0, 0, 0]          # Background: Black
    seg_vis[seg_e == 1] = [255, 0, 0]      # Class 1: Red
    seg_vis[seg_e == 2] = [255, 255, 0]  # Class 2: yellow 
    # seg_vis[seg_e == 3] = [0, 255, 0]  # Class 3: green
    # seg_vis[seg_e == 4] = [0, 0, 255]      # Class 4: Blue
    # Image.fromarray(seg_vis).save(os.path.join(save_seg_vis, segs_envy[i].split("/")[-1].replace(".npy", ".png")))
    Image.fromarray(img_e).save(os.path.join(save_img_vis, imgs_envy[i].split("/")[-1].replace(".npy", ".png")))
        

            