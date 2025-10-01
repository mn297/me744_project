# ME744 Project - Apple Detection and Segmentation

Apple detection and segmentation using SAM2 on the Fuji-SfM dataset.

## Setup

Install dependencies with uv:

```bash
uv sync
```

Download datasets and install SAM2:

```bash
source .venv/bin/activate
python setup.py
```

This will:
- Download and extract the Fuji-SfM dataset
- Clone SAM2 to `~/sam2` and install it
- You'll need to manually download the checkpoint: `sam2.1_hiera_large.pt` to `~/sam2/checkpoints/`

## Usage

### Visualize ground truth annotations

```bash
python fuji_visualize_ground_truth_masks.py
```

Loads polygon annotations from CSV files and creates visualizations.

Output: `fuji_sfm_data/Fuji-SfM_dataset/ground_truth_visualizations/`

### SAM2 segmentation

```bash
python fuji_sam2.py
```

Interactive and automatic segmentation examples using SAM2.

Output: `fuji_sfm_data/Fuji-SfM_dataset/sam2_results/`

### Train custom CNN

```bash
python fuji_train.py
```

Trains a simple U-Net on the dataset, validates, and visualizes results.

Output: `training_outputs/`

All scripts work as Jupyter notebooks (use the `# %%` cell markers).

## Dataset Structure

```
fuji_sfm_data/Fuji-SfM_dataset/
├── 1-Mask-set/
│   ├── training_images_and_annotations/
│   │   ├── *.jpg                     # Images
│   │   └── mask_*.csv                # Polygon annotations
│   └── validation_images_and_annotations/
├── 2-SfM-set/
│   ├── raw_images_east_side/
│   └── raw_images_west_side/
└── 3-3D_data/
    └── Fuji_apple_trees_point_cloud.txt
```

## Requirements

- Python >= 3.12
- PyTorch >= 2.8.0 (CUDA 12.8)
- Open3D, OpenCV, matplotlib

Optional tools for faster setup:
- `aria2`, `wget` for downloads
- `unrar` or `unar` for RAR extraction

## Scripts

**setup.py** - Downloads datasets and installs SAM2

**fuji_sam2.py** - SAM2 segmentation (interactive prompts and automatic masks)

**fuji_visualize_ground_truth_masks.py** - Visualize polygon annotations and create masks

**fuji_train.py** - Train custom U-Net CNN for apple segmentation
