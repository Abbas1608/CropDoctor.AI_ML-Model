# ----------------------------------------
# Phase 3: Train the YOLOv8-cls Model
# ----------------------------------------

from ultralytics import YOLO

if __name__ == '__main__':
    # Load the pre-trained classification model
    model = YOLO('yolov8n-cls.pt')

    print("Starting YOLOv8 training...")
    
    # Train the model
    # Note: On Windows, use absolute paths to avoid directory issues
    results = model.train(
        data='yolo_dataset',
        epochs=25,       # Number of times it loops through the data
        imgsz=224,       # Image size (224 is standard for classification)
        batch=16,        # Adjust to 8 if your computer runs out of memory, or 32 if you have a strong GPU
        name='crop_doctor_run_1'
    )