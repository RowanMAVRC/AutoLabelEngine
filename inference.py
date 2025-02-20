import os
import random
from ultralytics import YOLO  # Ensure the YOLOv8 package is correctly imported

# Load the model
model = YOLO("/data/TGSSE/weights/coco_2_ijcnn_vr_full_2_real_world_combination_2_hololens_finetune-v3.pt")

# Define directories
image_dir = "/data/TGSSE/HololensCombined/images/"
output_dir = "temp"
os.makedirs(output_dir, exist_ok=True)

# Get list of all images in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Randomly sample 5 images
sampled_images = random.sample(image_files, 5)

# Process and save results
for i, image_name in enumerate(sampled_images):
    image_path = os.path.join(image_dir, image_name)
    results = model(image_path)
    results[0].save(os.path.join(output_dir, f"result_{i + 1}.png"))