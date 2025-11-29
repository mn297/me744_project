from pathlib import Path
from PIL import Image
import numpy as np
import shutil
from tqdm import tqdm

# https://patrickwasp.com/create-your-own-coco-style-dataset/?utm_source=chatgpt.com
# https://github.com/waspinator/pycococreator/blob/114df401e5310c602178b31a48d3bb4cef876258/pycococreatortools/pycococreatortools.py#L25

BASE_DIR = Path(__file__).parent.parent

# the folders containing the images / labels
IMAGE_DIR = BASE_DIR / "datasets" / "image_envy_5000_before_unet" / "img"
LABEL_DIR = BASE_DIR / "datasets" / "image_envy_5000_before_unet" / "gt"
# intermediate folder for all samples before splitting
NP_SEGDIR_ALL = BASE_DIR / "datasets" / "image_envy_5000_unet" / "np_segs_all"
NP_IMGDIR_ALL = BASE_DIR / "datasets" / "image_envy_5000_unet" / "np_imgs_all"

# final split folders
NP_SEGDIR_TRAIN = BASE_DIR / "datasets" / "image_envy_5000_unet" / "np_segs_train"
NP_IMGDIR_TRAIN = BASE_DIR / "datasets" / "image_envy_5000_unet" / "np_imgs_train"
NP_SEGDIR_VAL = BASE_DIR / "datasets" / "image_envy_5000_unet" / "np_segs_val"
NP_IMGDIR_VAL = BASE_DIR / "datasets" / "image_envy_5000_unet" / "np_imgs_val"

for path in (
    NP_SEGDIR_ALL,
    NP_IMGDIR_ALL,
    NP_SEGDIR_TRAIN,
    NP_IMGDIR_TRAIN,
    NP_SEGDIR_VAL,
    NP_IMGDIR_VAL,
):
    path.mkdir(parents=True, exist_ok=True)


colors = [
    [[255, 0, 0], [255, 1, 1]],
    [[255, 255, 0], [255, 255, 1]],
    [[0, 255, 0], [1, 255, 1]],
]

# go through each image
numclasses = len(colors)  # excluding background
images = sorted(IMAGE_DIR.glob("*.png"))
gts = sorted(LABEL_DIR.glob("*.png"))
print(f"Found {len(gts)} label files and {len(images)} image files.")

if len(images) != len(gts):
    raise ValueError("Mismatched image/label counts; cannot pair files reliably.")

for gt_path, im_path in tqdm(
    zip(gts, images), total=len(images), desc="Converting image/label pairs"
):
    gt_key = gt_path.stem.split("_")[0]
    im_key = im_path.stem.split("_")[0]
    assert gt_key == im_key, f"Label/image mismatch: {gt_path.name} vs {im_path.name}"

    annot = np.asarray(Image.open(gt_path))
    assert np.all(annot[:, :, 3] == 255)
    annot = annot[:, :, :3]  # ignore alpha
    im_cur = np.asarray(Image.open(im_path))
    assert np.all(im_cur[:, :, 3] == 255)
    im_cur = im_cur[:, :, :3]  # ignore alpha
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
    masks = np.stack(masks, axis=0).astype(np.uint8)
    assert np.max(np.sum(masks, axis=0)) == 1

    # Create a binary mask: 1 if any fruit color is present, 0 otherwise
    label_map = np.zeros((annot.shape[0], annot.shape[1]), dtype=np.uint8)
    for cur_label in range(masks.shape[0]):
        # If any of the fruit classes are present, mark as 1
        label_map[masks[cur_label] > 0] = 1

    seg_out = NP_SEGDIR_ALL / f"{im_path.stem}.npy"
    img_out = NP_IMGDIR_ALL / f"{im_path.stem}.npy"

    np.save(seg_out, label_map.astype(np.uint8))
    np.save(img_out, im_cur)
    # print(np.asarray(Image.open(im)).shape)

    # label_map[label_map==3] = 254
    # label_map = label_map.astype(np.uint8)
    # Image.fromarray(label_map).save(os.path.join(BIN_SEGDIR, gt.split("/")[-1]))
    # print(label_map.shape)
    # print(np.sum(label_map==254))
    # print(annot[0,0])


all_seg_paths = sorted(NP_SEGDIR_ALL.glob("*.npy"))
all_img_paths = sorted(NP_IMGDIR_ALL.glob("*.npy"))

if len(all_seg_paths) != len(all_img_paths):
    raise ValueError("Mismatch between saved segmentation and image counts.")

train_cutoff = min(900, len(all_seg_paths))
train_seg = all_seg_paths[:train_cutoff]
train_img = all_img_paths[:train_cutoff]
val_seg = all_seg_paths[train_cutoff:]
val_img = all_img_paths[train_cutoff:]


# move the pairs to the destination folders
def move_pairs(seg_list, img_list, seg_dest, img_dest, desc):
    for seg_path, img_path in tqdm(
        zip(seg_list, img_list),
        total=len(seg_list),
        desc=desc,
    ):
        shutil.move(seg_path, seg_dest / seg_path.name)
        shutil.move(img_path, img_dest / img_path.name)


move_pairs(train_seg, train_img, NP_SEGDIR_TRAIN, NP_IMGDIR_TRAIN, "Moving train split")
move_pairs(val_seg, val_img, NP_SEGDIR_VAL, NP_IMGDIR_VAL, "Moving val split")
