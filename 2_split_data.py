# _____________________________________________________
# Phase 2: Split Data for YOLO
# _____________________________________________________

import splitfolders

print("Splitting data into Training and Validation sets...")

# This takes your cleaned 'Crop_dataset' and creates a new folder ready for YOLO
# Ratio (0.8, 0.2) means 80% for training, 20% for validation/testing
splitfolders.ratio(
    'Crop_dataset', 
    output='yolo_dataset', 
    seed=42, 
    ratio=(0.8, 0.2), 
    group_prefix=None
)

print("Splitting complete! Data is ready for YOLOv8 in the 'yolo_dataset' folder.")