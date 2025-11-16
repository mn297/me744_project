from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
import json
import datetime
import os
import glob
from skimage import morphology, filters
from skimage.morphology import disk
from skimage.measure import label

# https://patrickwasp.com/create-your-own-coco-style-dataset/?utm_source=chatgpt.com
# https://github.com/waspinator/pycococreator/blob/114df401e5310c602178b31a48d3bb4cef876258/pycococreatortools/pycococreatortools.py#L25

from pycocotools import mask as maskUtils

def ann_from_mask_rle(seg_id, image_id, category_id, binary_mask, iscrowd=0):
    m = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = maskUtils.encode(m)
    rle["counts"] = rle["counts"].decode("ascii")
    area = float(maskUtils.area(rle))
    bbox = maskUtils.toBbox(rle).tolist()
    return {
        "id": seg_id,
        "image_id": image_id,
        "category_id": int(category_id),
        "iscrowd": int(iscrowd),
        "segmentation": rle,
        "area": area,
        "bbox": bbox,
    }

IMAGE_DIR = "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/img"
LABEL_DIR = "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/gt"

# BIN_SEGDIR = "/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/bin_segs"

coco_output = {
    "info": {
        "description": "envy Dataset",
        "url": "https://github.com/waspinator/pycococreator",
        "version": "0.1.0",
        "year": 2025,
        "contributor": "waspinator",
        "date_created": str(datetime.datetime.now())
    },
    "licenses": [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
    ],
    "categories": [
        {
            'id': 1,
            'name': 'branches',
            'supercategory': 'tree',
        },
    ],
    "images": [],
    "annotations": []
}

# go through each image
segmentation_id = 0
numclasses = [1] # excluding background
images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png"), recursive=True))[3000:3999]
gts = sorted(glob.glob(os.path.join(LABEL_DIR, "*.png"), recursive=True))[3000:3999]
print(len(images))
print(len(gts))
for image_id, image_filename in enumerate(images):
    image = Image.open(image_filename)
    image_info = pycococreatortools.create_image_info(
        image_id, os.path.basename(image_filename), image.size)
    coco_output["images"].append(image_info)
    print(image_info)
    # exit(0)
    
    annotation_filename = gts[image_id]
    annot = np.asarray(Image.open(annotation_filename))

    print(image_filename.split('/')[-1])
    print(annotation_filename.split('/')[-1])

    assert np.all(annot[:,:,3]==255)
    annot = annot[:,:,:3] # ignore alpha
    # all_labels = np.unique(annot.reshape(annot.shape[0] * annot.shape[1], 3), axis=0, return_counts=True)
    # print(annotation_filename)
    # print(annot.shape)
    # print(all_labels)
    # print(annot.shape)
    # def rgbtolb(rgb):
    #     if np.all(rgb == [255,0,0]) or np.all(rgb == [255,1,1]):
    #         return 1
    #     elif np.all(rgb == [255,255,0]) or np.all(rgb == [255,255,1]):
    #         return 1
    #     elif np.all(rgb == [0,255,0]) or np.all(rgb == [1,255,1]):
    #         return 1
    #     else:
    #         return -1

    # annot = np.apply_along_axis(rgbtolb, -1, annot)
    matches = []
    colors = [[255, 0, 0],
    [255, 1, 1],
    [255, 255, 0],
    [255, 255, 1],
    [0, 255, 0],
    [1, 255, 1]]
    for col in colors:
        matches.append(np.all(annot == col, axis=-1))
    
    annot = np.logical_or.reduce(matches)
    # print(annot.shape)
    # print(annot[0,0])
    # print()
    # exit(0)
    # continue
    for class_id in numclasses:
        category_info = {'id': class_id, 'is_crowd': 0}
        binary_mask = annot == class_id
        # binary_mask = morphology.remove_small_objects(binary_mask, min_size=50, connectivity=1)
        # binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=50)
        # binary_mask = morphology.binary_closing(binary_mask, footprint=disk(1))
        # instance_map = label(binary_mask, connectivity=1) 
        # print(np.unique(instance_map, return_counts=True)) 
        # instance_map = label(binary_mask, connectivity=1) 
        # print(np.unique(instance_map, return_counts=True))
        # for u in np.unique(instance_map)[1:]: # exclude background
        # subinstance_mask = instance_map == u
        # print(np.sum(subinstance_mask))
        # Image.fromarray(binary_mask.astype(np.uint8) * 255).save(os.path.join(BIN_SEGDIR, f"mask_{image_id}_{class_id}_{segmentation_id}.png"))
        
        # annotation_info = pycococreatortools.create_annotation_info(
        #     segmentation_id, image_id, category_info, subinstance_mask,
        #     image.size, tolerance=2)
        annotation_info = ann_from_mask_rle(segmentation_id, image_id, class_id, binary_mask, iscrowd=0)
        if annotation_info is not None:
            coco_output["annotations"].append(annotation_info)

        segmentation_id += 1

    
with open("/Users/ziyaoshang/Desktop/MEproject/DAFormer/data/source/envy_coco.json", "w") as f:
    json.dump(coco_output, f, indent=4)