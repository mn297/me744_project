import argparse
from fileinput import filename
import logging
import os
import glob
import pickle
import numpy as np
import torch


rcnn_out = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/visualizations/mixed/test_on_good_mix/rcnn_out", "*.pkl"), recursive=True))
unet_out = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/visualizations/mixed/test_on_good_mix/results", "*.npy"), recursive=True))
gt_semantic = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/visualizations/mixed/test_on_good_mix/gt", "*.npy"), recursive=True))


# rcnn_out = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/mixed/test_common/mixed_training/rcnn_out", "*.pkl"), recursive=True))
# unet_out = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/mixed/test_common/mixed_training/results", "*.npy"), recursive=True))
# gt_semantic = sorted(glob.glob(os.path.join("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/mixed/test_common/seg", "*.npy"), recursive=True))


num_classes = 3  # background, non-fruit, fruit
save_seg = True
seg_save_path = "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/visualizations/mixed/test_on_good_mix/merged"

assert len(rcnn_out) == len(unet_out) == len(gt_semantic), f"Mismatched number of files, {len(rcnn_out), len(unet_out), len(gt_semantic)}"

ious_all = []
dice_all = []

for i, (r, u, g) in enumerate(zip(rcnn_out, unet_out, gt_semantic)):
    # with open(r, "rb") as f:
    # r_out = torch.load(r, map_location=torch.device('cpu'))
    print(r)
    print(u)
    print(g)
    with open(r, "rb") as f:
        r_out = pickle.load(f)
    r_keep = r_out["scores"] >= 0.5
    print(r_out["masks"][r_keep].shape)
    r_mask = np.squeeze(r_out["masks"][r_keep] > 0.5, axis=1)
    print(np.sum(r_mask))
    print(r_mask.shape)
    u_mask = np.load(u)
    print(np.unique(u_mask))
    merged_mask = u_mask.copy()
    if r_mask.shape[0] != 0:
        assert r_mask.shape[-2:] == u_mask.shape
        assert (len(np.unique(r_mask)) <= 2) and (np.max(r_mask) == 1)
        # print(r_mask.shape, u_mask.shape)

        # merge semantic masks
        r_mask = np.max(r_mask, axis=0)
        assert (len(np.unique(r_mask)) <= 2) and (np.max(r_mask) == 1)
        merged_mask[r_mask != 0] = 2  # all fruits
        

    # save the merged mask as a npy file
    if save_seg:
        out_filename = os.path.join(seg_save_path, os.path.basename(u).replace('.npy', '_merged.npy'))
        np.save(out_filename, merged_mask.astype(np.uint8))
        print(f"Saved merged mask to {out_filename}")

    # calculate multi-class mIoU and Dice
    gt_mask = np.load(g)
    assert merged_mask.shape == gt_mask.shape   
    ious = []
    dices = []
    for cls in range(num_classes):
        pred_cls = merged_mask == cls
        gt_cls = gt_mask == cls
        print(np.sum(pred_cls))
        print(np.sum(gt_cls))

        intersection = np.logical_and(pred_cls, gt_cls)
        union = np.logical_or(pred_cls, gt_cls)
        if np.sum(union) == 0:
            iou_score = 1.0  # both pred and gt are empty
        else:
            iou_score = np.sum(intersection) / np.sum(union)
        ious.append(iou_score)
        # print(np.sum(intersection))
        # print(np.sum(np.sum(union)))
         

        if np.sum(pred_cls) + np.sum(gt_cls) == 0:
            dice_score = 1.0  # both pred and gt are empty
        else:
            dice_score = 2 * np.sum(intersection) / (np.sum(pred_cls) + np.sum(gt_cls))
        dices.append(dice_score)
        print(f"Image {i}, Class {cls}: mIoU = {iou_score:.4f}, Dice = {dice_score:.4f}")

    ious_all.append(ious)
    dice_all.append(dices)

ious_all = np.array(ious_all)
dice_all = np.array(dice_all)

print(ious_all.shape)
print(dice_all.shape)

# calculate mious and mean dices per class
mean_ious = np.mean(ious_all, axis=0)
mean_dices = np.mean(dice_all, axis=0)

# calculate mious and mean dices per image
mean_ious_per_image = np.mean(ious_all[:, 1:], axis=1)
mean_dices_per_image = np.mean(dice_all[:, 1:], axis=1)

mean_mious = np.mean(mean_ious[1:])
mean_dices_avg = np.mean(mean_dices[1:])

print("Mean mIoU per class:", mean_ious)
print("Mean Dice per class:", mean_dices)
print("Mean mIoU per image (no background):", mean_ious_per_image)
print("Mean Dice per image (no background):", mean_dices_per_image)
print("Overall Mean mIoU (no background):", mean_mious)
print("Overall Mean Dice (no background):", mean_dices_avg)





