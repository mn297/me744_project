# ME744 Project: Apple Segmentation Pipeline

This repository contains a complete pipeline for training and evaluating segmentation models (Mask R-CNN, YOLO, U-Net) on apple datasets (Fuji, Envy).

## 📂 Project Structure

- **`maskrcnn/`**: Core logic for Mask R-CNN and YOLO workflows.
  - Contains training scripts, converters, and MLOps utilities.
- **`unet/`**: Standalone U-Net implementation (`Pytorch-UNet`).
- **`datasets/`**: **IMPORTANT**: All data lives here.
  - Raw data (e.g., `Fuji-SfM_dataset`) should be placed here.
  - Processed datasets (COCO JSONs, YOLO yamls, U-Net masks) are generated here.
  - Model checkpoints and predictions are saved here.

---

## 🔧 Setup & Installation

1.  **Environment**: Ensure you have Python 3.10+ installed.
2.  **Dependencies**:
    ```bash
    # Using uv (fast, recommended)
    ./setup_uv.sh

    # OR using pip directly
    pip install -e .
    ```

---

## 📁 Dataset Assumptions

The scripts assume specific locations for raw and processed data.

1.  **Raw Data**:
    - Place the raw Fuji SfM dataset in: `datasets/Fuji-SfM_dataset/`
    - It should contain `1-Mask-set/training_images_and_annotations/` and `validation...`.

2.  **Processed Data**:
    - Converters will generate folders like:
        - `datasets/Fuji-Apple-Segmentation_coco/` (for Mask R-CNN)
        - `datasets/Fuji-Apple-Segmentation_yolo/` (for YOLO)
        - `datasets/Fuji-Apple-Segmentation_unet/` (for U-Net)

---

## 🚀 Running Instructions

### 1. Data Preparation (Required First Step)

Convert raw data into the format required by your model.

**For Mask R-CNN (COCO Format):**
```bash
python maskrcnn/convert_fuji_to_coco.py
```
*Creates `datasets/Fuji-Apple-Segmentation_coco/`.*

**For YOLO:**
```bash
python maskrcnn/convert_coco_to_yolo.py
```
*Requires COCO dataset first. Creates `datasets/Fuji-Apple-Segmentation_yolo/`.*

**For U-Net:**
```bash
python maskrcnn/convert_coco_to_unet.py
```
*Creates `datasets/Fuji-Apple-Segmentation_unet/`.*

---

### 2. Training Models

#### A. Mask R-CNN
You can run using a CLI command or a YAML config file.

**Option 1: YAML Config (Recommended)**
```bash
python maskrcnn/train_maskrcnn.py --config maskrcnn/experiment_fuji.yaml
```

**Option 2: CLI Arguments**
```bash
python maskrcnn/train_maskrcnn.py \
  --train-images datasets/Fuji-Apple-Segmentation_coco/trainingset/JPEGImages \
  --train-anno datasets/Fuji-Apple-Segmentation_coco/trainingset/annotations.json \
  --val-images datasets/Fuji-Apple-Segmentation_coco/testset/JPEGImages \
  --val-anno datasets/Fuji-Apple-Segmentation_coco/testset/annotations.json \
  --epochs 20 --batch-size 2
```
*Outputs are saved to `datasets/checkpoints` and `runs/maskrcnn`.*

#### B. YOLO
The training script currently points to a specific dataset path inside the code.
1.  Open `maskrcnn/train_yolo.py`.
2.  Verify/Edit the `dataset_dir` variable to point to your YOLO dataset (e.g., `datasets/Fuji-Apple-Segmentation_yolo`).
3.  Run:
    ```bash
    python maskrcnn/train_yolo.py
    ```

#### C. U-Net
The U-Net training script has **hardcoded paths** that must be updated before running.
1.  Open `unet/Pytorch-UNet/train.py`.
2.  Edit lines 25-27 to point to your generated U-Net data:
    ```python
    dir_img = Path('../../datasets/Fuji-Apple-Segmentation_unet/np_imgs_train')
    dir_mask = Path('../../datasets/Fuji-Apple-Segmentation_unet/np_segs_train')
    ```
3.  Run:
    ```bash
    python unet/Pytorch-UNet/train.py --epochs 20 --batch-size 4
    ```

---

### 3. Inference & Visualization

**Mask R-CNN Prediction:**
```bash
python maskrcnn/predict_maskrcnn.py
```
*Check the script to ensure it points to the correct checkpoint file.*

**YOLO Prediction:**
```bash
python maskrcnn/predict_yolo.py
```
*Or use `visualize_yolo.py` for visual debugging.*

**U-Net Prediction:**
```bash
python maskrcnn/predict_unet.py
```

---

## 📊 Fusion & Evaluation

To combine outputs from different models or evaluate ensemble performance:

```bash
python maskrcnn/merge_and_eval.py
```

## 📝 References

- [`maskrcnn/torchvision_references/`](maskrcnn/torchvision_references/): Upstream utilities from PyTorch
- [`maskrcnn/Mask_RCNN/`](maskrcnn/Mask_RCNN/): Legacy references from the Mask R-CNN paper
- [Pytorch-UNet by milesial](https://github.com/milesial/Pytorch-UNet)
- [PyTorch Mask R-CNN documentation](https://docs.pytorch.org/vision/main/models/mask_rcnn.html)
