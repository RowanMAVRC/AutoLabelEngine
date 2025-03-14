import os
import argparse
from PIL import Image
from tqdm import tqdm

def rotate_image(image_path, rotation_type):
    """Rotates an image based on the specified rotation type and saves it in place."""
    try:
        with Image.open(image_path) as img:
            if rotation_type == 'CW':
                rotated_img = img.rotate(-90, expand=True)  # Clockwise 90 degrees
            elif rotation_type == 'CCW':
                rotated_img = img.rotate(90, expand=True)  # Counterclockwise 90 degrees
            elif rotation_type == '180':
                rotated_img = img.rotate(180, expand=True)  # 180-degree rotation
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
    parser = argparse.ArgumentParser(description="Rotate images in a directory")
    parser.add_argument("--directory", type=str, required=True, help="Path to the directory containing images")
    parser.add_argument("--rotation", type=str, choices=['CW', 'CCW', '180'], required=True, 
                        help="Rotation type: CW (90° Clockwise), CCW (90° Counterclockwise), or 180 (180° Rotation)")

    args = parser.parse_args()
    process_images(args.directory, args.rotation)
