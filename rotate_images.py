import os
import json
import argparse
from PIL import Image
from tqdm import tqdm

def rotate_image(image_path, rotation_type):
    """Rotates an image based on the specified rotation type and saves it in place."""
    try:
        with Image.open(image_path) as img:
            if rotation_type == 'CW':
                rotated_img = img.rotate(-90, expand=True)
            elif rotation_type == 'CCW':
                rotated_img = img.rotate(90, expand=True)
            elif rotation_type == '180':
                rotated_img = img.rotate(180, expand=True)
            else:
                print(f"Invalid rotation type: {rotation_type}")
                return
            rotated_img.save(image_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_images(directory, rotation_type):
    """Processes all images in the directory and applies the specified rotation."""
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]
    if not image_files:
        print("No valid image files found in the directory.")
        return

    for filename in tqdm(image_files, desc="Processing Images", unit="file"):
        image_path = os.path.join(directory, filename)
        rotate_image(image_path, rotation_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotate images in multiple datasets using JSON file input.")
    parser.add_argument(
        "--json_file", 
        type=str, 
        required=True,
        help="Path to the JSON file containing a list of dataset entries with 'directory' and 'rotation' keys."
    )
    args = parser.parse_args()
    
    try:
        with open(args.json_file, "r") as f:
            datasets = json.load(f)
        # Delete the temporary file after loading.
        os.remove(args.json_file)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        exit(1)
    
    if not isinstance(datasets, list):
        print("Datasets JSON must be a list of dictionaries.")
        exit(1)
    
    for entry in datasets:
        directory = entry.get("directory")
        rotation = entry.get("rotation")
        if directory is None or rotation is None:
            print(f"Skipping invalid dataset entry: {entry}")
            continue
        print(f"Processing dataset: {directory} with rotation: {rotation}")
        process_images(directory, rotation)
