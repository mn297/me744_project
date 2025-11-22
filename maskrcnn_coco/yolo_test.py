from ultralytics import YOLO
import os
import random

# Load a pretrained YOLOv8 segmentation model
model = YOLO("yolo11s-seg.pt")

# Define the path to the dataset YAML file
dataset_yaml = "Fuji-Apple-Segmentation_yolo/dataset.yaml"

# Train the model on the Fuji Apple dataset
# Note: You might want to adjust epochs, imgsz, etc. for your specific needs.
results = model.train(
    data=dataset_yaml,
    epochs=2,  # Example: fine-tune for 50 epochs
    imgsz=640,
    device=0,  # Use GPU 0
    name="yolov8m-seg-fuji-apple",  # Custom name for the training run
)

# Validate the model
print("Validation metrics:")
metrics = model.val()

# --- Run inference on a random test image ---
# Find a random image from the validation set to test prediction
val_dir = "Fuji-Apple-Segmentation_yolo/images/val"
if os.path.exists(val_dir):
    test_image_name = random.choice(os.listdir(val_dir))
    test_image_path = os.path.join(val_dir, test_image_name)

    print(f"\nRunning prediction on a random test image: {test_image_path}")
    predictions = model.predict(test_image_path, conf=0.5, save=True)

    # The results with plotted masks will be saved in the 'runs/segment/predict' directory
    print(f"\nPrediction results saved. Check the latest directory in 'runs/segment/predict'.")
    print("Each prediction contains bounding boxes, masks, and class probabilities.")

else:
    print(f"Validation directory not found at: {val_dir}")
    print("Skipping prediction on a test image.")
