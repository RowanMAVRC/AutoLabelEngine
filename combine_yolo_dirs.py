import os
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

def combine_yolo_dirs(yolo_dirs, dst_dir):
    """
    Combine two YOLO directories into a single destination directory with prefixed filenames.
    Resolves duplicate prefixes by appending a higher-level directory name to make them unique.

    Args:
        yolo_dirs (list of str): List of two YOLO directories to combine. Each directory must contain 'images' and 'labels' subdirectories.
        dst_dir (str): Destination directory where combined files will be saved.
    """
    dst_images_dir = Path(dst_dir) / "images"
    dst_labels_dir = Path(dst_dir) / "labels"

    # Create destination subdirectories if they don't exist
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    # Determine unique prefixes for each dataset
    prefixes = {}
    for yolo_dir in yolo_dirs:
        yolo_path = Path(yolo_dir)
        prefix = f"{yolo_path.parent.name}_{yolo_path.name}"
        # Ensure the prefix is unique
        while prefix in prefixes.values():
            prefix = f"{Path(yolo_path.parent.parent).name}_{prefix}"
        if prefix.startswith("_"):
            prefix = prefix[1:]
        prefixes[yolo_dir] = prefix

    # Process each YOLO directory
    for yolo_dir, prefix in prefixes.items():
        yolo_path = Path(yolo_dir)
        images_dir = yolo_path / "images"
        labels_dir = yolo_path / "labels"

        # Check if subdirectories exist
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Skipping {yolo_dir}: Missing 'images' or 'labels' directory.")
            continue

        # Copy images with new prefixed filenames
        image_files = list(images_dir.glob("*.*"))  # Match all files
        for image_file in tqdm(image_files, desc=f"Copying images from {yolo_dir}", unit="file"):
            new_name = f"{prefix}_{image_file.name}"
            shutil.copy(image_file, dst_images_dir / new_name)

        # Copy labels with new prefixed filenames
        label_files = list(labels_dir.glob("*.*"))  # Match all files
        for label_file in tqdm(label_files, desc=f"Copying labels from {yolo_dir}", unit="file"):
            new_name = f"{prefix}_{label_file.name}"
            shutil.copy(label_file, dst_labels_dir / new_name)

    print(f"All files have been combined and saved to {dst_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine two YOLO dataset directories into one.")
    parser.add_argument("--dataset1", help="Path to the first YOLO dataset directory")
    parser.add_argument("--dataset2", help="Path to the second YOLO dataset directory")
    parser.add_argument("--dst_dir", help="Destination directory where combined files will be saved")
    args = parser.parse_args()

    combine_yolo_dirs([args.dataset1, args.dataset2], args.dst_dir)
