import os
import shutil
import argparse
from tqdm import tqdm

def is_valid_line(line):
    """Check if a line contains valid YOLO format data."""
    parts = line.strip().split()
    # A valid line should have at least one number (class ID)
    return len(parts) > 0 and all(part.replace('.', '', 1).isdigit() for part in parts)

def categorize_images(yolo_dir, output_dir_with_objects, output_dir_no_objects, images_subdir="images", labels_subdir="labels"):
    # Create output directories with subdirectories for images and labels
    output_images_with_objects = os.path.join(output_dir_with_objects, images_subdir)
    output_labels_with_objects = os.path.join(output_dir_with_objects, labels_subdir)
    output_images_no_objects = os.path.join(output_dir_no_objects, images_subdir)
    output_labels_no_objects = os.path.join(output_dir_no_objects, labels_subdir)
    
    os.makedirs(output_images_with_objects, exist_ok=True)
    os.makedirs(output_labels_with_objects, exist_ok=True)
    os.makedirs(output_images_no_objects, exist_ok=True)
    os.makedirs(output_labels_no_objects, exist_ok=True)

    # Get paths to images and labels
    images_path = os.path.join(yolo_dir, images_subdir)
    labels_path = os.path.join(yolo_dir, labels_subdir)

    if not os.path.exists(images_path):
        print("Images directory is missing. Please check the input path.")
        return

    # Process each image file in the images directory
    for image_file in tqdm(os.listdir(images_path), desc="Categorizing image/label pairs"):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_path, image_file)
            # Assume corresponding label file has the same name but with .txt extension
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_path, label_file)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as file:
                    lines = file.readlines()
                valid_lines = [line for line in lines if is_valid_line(line)]
                if valid_lines:  # Label file contains valid objects
                    shutil.copy(image_path, os.path.join(output_images_with_objects, image_file))
                    shutil.copy(label_path, os.path.join(output_labels_with_objects, label_file))
                else:  # Label file exists but no valid objects
                    shutil.copy(image_path, os.path.join(output_images_no_objects, image_file))
                    # Create an empty label file in the no-objects directory
                    empty_label_path = os.path.join(output_labels_no_objects, label_file)
                    open(empty_label_path, 'w').close()
            else:
                # No label file exists: treat as image with no objects
                shutil.copy(image_path, os.path.join(output_images_no_objects, image_file))
                # Create a blank label file in the no-objects directory
                empty_label_path = os.path.join(output_labels_no_objects, label_file)
                open(empty_label_path, 'w').close()

def main():
    parser = argparse.ArgumentParser(
        description="Categorize YOLO images into two groups: with valid objects and without valid objects."
    )
    parser.add_argument('--data_path', help="Path to the YOLO directory containing 'images' and 'labels' subdirectories")
    parser.add_argument('--save_path', help="Directory where the categorized outputs will be saved")
    args = parser.parse_args()

    yolo_dir = args.data_path
    output_dir_with_objects = os.path.join(args.save_path, "output_with_objects")
    output_dir_no_objects = os.path.join(args.save_path, "output_no_objects")
    
    categorize_images(yolo_dir, output_dir_with_objects, output_dir_no_objects)

if __name__ == "__main__":
    main()
