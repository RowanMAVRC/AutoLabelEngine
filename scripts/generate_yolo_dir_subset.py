#!/usr/bin/env python3
import os
import random
import shutil

def is_image_file(filename):
    # Check for common image extensions
    return any(filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"])

def main():
    # Hard-coded variables
    data_dir = '/data/TGSSE/HololensCombined/'    # Path to your YOLO dataset directory
    n_samples = 100                       # Number of images to sample
    save_dir = '/data/TGSSE/HololensCombined/random_subset'    # Destination directory to copy the samples

    # Define source subdirectories
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    # Verify source directories exist
    if not os.path.isdir(images_dir):
        print(f"Error: Images directory '{images_dir}' does not exist.")
        return
    if not os.path.isdir(labels_dir):
        print(f"Warning: Labels directory '{labels_dir}' does not exist. Only images will be copied.")

    # List all image files in the images directory
    all_images = [f for f in os.listdir(images_dir) if is_image_file(f)]
    if not all_images:
        print("No image files found in the images directory.")
        return

    # Adjust sample count if necessary
    if n_samples > len(all_images):
        print(f"Requested {n_samples} samples, but only {len(all_images)} images available. Using all images.")
        n_samples = len(all_images)

    sampled_images = random.sample(all_images, n_samples)

    # Create destination subdirectories
    save_images_dir = os.path.join(save_dir, "images")
    save_labels_dir = os.path.join(save_dir, "labels")
    os.makedirs(save_images_dir, exist_ok=True)
    os.makedirs(save_labels_dir, exist_ok=True)

    # Copy sampled images and their corresponding label files if they exist
    for image_file in sampled_images:
        src_image_path = os.path.join(images_dir, image_file)
        dst_image_path = os.path.join(save_images_dir, image_file)
        shutil.copy2(src_image_path, dst_image_path)

        # Look for a corresponding label file (assumed to have the same base name with .txt extension)
        base_name, _ = os.path.splitext(image_file)
        label_file = base_name + ".txt"
        src_label_path = os.path.join(labels_dir, label_file)
        if os.path.isfile(src_label_path):
            dst_label_path = os.path.join(save_labels_dir, label_file)
            shutil.copy2(src_label_path, dst_label_path)
        else:
            print(f"Warning: No label file found for image '{image_file}'.")

    print(f"Successfully sampled {n_samples} images (and available labels) to '{save_dir}'.")

if __name__ == "__main__":
    main()
