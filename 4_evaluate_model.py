# --------------------------------------
# Phase 5: Check Accuracy and Test
# ---------------------------------------
from ultralytics import YOLO

# 1. Load your custom trained model
model = YOLO('runs/classify/crop_doctor_run_1/weights/best.pt')

# 2. Your exact dataset dictionary
DISEASE_MAP = {
    0: 'Apple__Apple_scab',
    1: 'Apple__Black_rot',
    2: 'Apple__Cedar_apple_rust',
    3: 'Apple__healthy',
    4: 'Bellpepper__Bacterial_spot',
    5: 'Bellpepper__healthy',
    6: 'Corn__Common_rust',
    7: 'Corn__Gray_leaf_spot',
    8: 'Corn__Northern_Leaf_Blight',
    9: 'Corn__healthy',
    10: 'Grape__Black_Measles',
    11: 'Grape__Black_rot',
    12: 'Grape__Leaf_blight',
    13: 'Grape__healthy',
    14: 'Potato__Early_blight',
    15: 'Potato__Late_blight',
    16: 'Potato__healthy',
    17: 'Rice__Brown_Spot',
    18: 'Rice__Healthy',
    19: 'Rice__Leaf_Blast',
    20: 'Rice__Neck_Blast',
    21: 'Wheat__Brown_Rust',
    22: 'Wheat__Healthy',
    23: 'Wheat__Yellow_Rust'
}

# Define the test image
image_path = 'image1.png' 
print(f"\nAnalyzing '{image_path}'...")

# 3. Run the AI
results = model(image_path, verbose=False)

# 4. Extract predictions
top_class_id = results[0].probs.top1              
confidence = results[0].probs.top1conf.item()     

# Look up the ID in our dictionary
raw_disease_name = DISEASE_MAP.get(top_class_id, "Unknown Disease")

# Clean up the text for the website UI (e.g., "Apple__Apple_scab" -> "Apple - Apple scab")
formatted_disease_name = raw_disease_name.replace('__', ' - ').replace('_', ' ')

# 5. Print the final results for the web UI
print("\n--- AI CROP DOCTOR DIAGNOSIS ---")
print(f"Result: {formatted_disease_name}")
print(f"Confidence: {confidence * 100:.2f}%")
print("--------------------------------\n")


# ==========================================
# OPTIONAL: Evaluate overall accuracy
# (Commented out to save time. Uncomment if you want to run the full test again)
# ==========================================
# print("Calculating accuracy metrics...")
# metrics = model.val()
# print(f"Top-1 Accuracy: {metrics.top1 * 100:.2f}%")
# print(f"Top-5 Accuracy: {metrics.top5 * 100:.2f}%")

# ==========================================
# SINGLE IMAGE TEST
# ==========================================
# View the entire dictionary of your 24 diseases (Optional, but good to see!)
# print(model.names) 