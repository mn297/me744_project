from PIL import Image
import numpy as np
import os
import glob
import shutil

# https://patrickwasp.com/create-your-own-coco-style-dataset/?utm_source=chatgpt.com
# https://github.com/waspinator/pycococreator/blob/114df401e5310c602178b31a48d3bb4cef876258/pycococreatortools/pycococreatortools.py#L25

# the folders containing the images / labels
IMAGE_DIR = "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/img"
LABEL_DIR = "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/gt"
# the folders to save the numpy arrays (train and evaluation)
NP_SEGDIR_TRAIN_VAL = "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/np_segs_train_val"
NP_IMGDIR_TRAIN_VAL = "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/np_imgs_train_val"
# the folders to save the numpy arrays (test)
NP_SEGDIR_TEST = "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/np_segs_test"
NP_IMGDIR_TEST = "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/np_imgs_test"



colors = [
[[255, 0, 0],[255, 1, 1]],
[[255, 255, 0],[255, 255, 1]],
[[0, 255, 0],[1, 255, 1]]
]

# go through each image
segmentation_id = 0
numclasses = len(colors) # excluding background
images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png"), recursive=True))
gts = sorted(glob.glob(os.path.join(LABEL_DIR, "*.png"), recursive=True))
print(len(gts))
for g, (gt, im) in enumerate(zip(gts, images)):
    assert gt.split("/")[-1].split("_")[0] == im.split("/")[-1].split("_")[0]
    print(gt.split("/")[-1].split("_")[0])
    annot = np.asarray(Image.open(gt))
    assert np.all(annot[:,:,3]==255)
    annot = annot[:,:,:3] # ignore alpha
    im_cur = np.asarray(Image.open(im))
    assert np.all(im_cur[:,:,3]==255)
    im_cur = im_cur[:,:,:3] # ignore alpha
    # all_labels = np.unique(annot.reshape(annot.shape[0] * annot.shape[1], 3), axis=0, return_counts=True)
    # print(annotation_filename)
    # print(annot.shape)
    # print(all_labels)
    # print(annot.shape)
    masks = []
    for lab in range(numclasses):
        colors_cur = colors[lab]
        matches = []
        for col in colors_cur:
            matches.append(np.all(annot == col, axis=-1))
        masks.append(np.logical_or.reduce(matches))
    masks = np.stack(masks,axis=0).astype(np.uint8)
    assert np.max(np.sum(masks, axis=0)) == 1

    label_map = np.zeros((annot.shape[0], annot.shape[1]), dtype=np.uint8)
    for cur_label in range(masks.shape[0]):
        label_map += masks[cur_label] * (cur_label + 1)
    
    np.save(os.path.join(NP_SEGDIR_TRAIN_VAL, gt.split("/")[-1].replace(".png",".npy")), label_map.astype(np.uint8))
    np.save(os.path.join(NP_IMGDIR_TRAIN_VAL, im.split("/")[-1].replace(".png",".npy")), im_cur)
    # print(np.asarray(Image.open(im)).shape)

    # label_map[label_map==3] = 254
    # label_map = label_map.astype(np.uint8)
    # Image.fromarray(label_map).save(os.path.join(BIN_SEGDIR, gt.split("/")[-1]))
    # print(label_map.shape)
    # print(np.sum(label_map==254))
    # print(annot[0,0])


images_tv = sorted(glob.glob(os.path.join(NP_SEGDIR_TRAIN_VAL, "*.npy"), recursive=True))
gts_tv = sorted(glob.glob(os.path.join(NP_IMGDIR_TRAIN_VAL, "*.npy"), recursive=True))

# move the last 1000 to test set
for g, (gt, im) in enumerate(zip(images_tv[-1000:], gts_tv[-1000:])):
    shutil.move(gt, os.path.join(NP_SEGDIR_TEST, gt.split("/")[-1]))
    shutil.move(im, os.path.join(NP_IMGDIR_TEST, im.split("/")[-1]))


