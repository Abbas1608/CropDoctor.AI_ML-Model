#______________________________________________________
# Phase 1: Data Extraction & Cleaning ("Removing Nulls")
# ______________________________________________________
import os
from datasets import load_dataset
from PIL import Image

# 1. Load the Parquet file
print("Loading Hugging Face dataset...")
# Update this path if your parquet file is located elsewhere
dataset = load_dataset("parquet", data_files="train-00000-of-00001.parquet")

raw_dir = "Crop_dataset"
os.makedirs(raw_dir, exist_ok=True)

corrupted_count = 0
valid_count = 0

print("Extracting and cleaning images...")
for i, item in enumerate(dataset['train']):
    try:
        img = item['image']
        label_name = str(item['label']) 
        
        class_dir = os.path.join(raw_dir, label_name)
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, f"img_{i}.jpg")
        
        # --- DATA CLEANING (The "Null" Check) ---
        # 1. Check if the image object actually exists
        if img is None:
            raise ValueError("Image is Null")
            
        # 2. Convert to RGB to ensure color consistency
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # 3. Verify the image isn't broken by trying to verify its contents
        img.verify() 
        
        # If it passes, save it
        img.save(img_path)
        valid_count += 1
        
    except Exception as e:
        corrupted_count += 1
        # We silently skip and do not save the corrupted/"null" image

print(f"Data Cleaning Complete!")
print(f"Valid images saved: {valid_count}")
print(f"Corrupted/Null images removed: {corrupted_count}")