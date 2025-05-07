#!/usr/bin/env python3
import os
import cv2
import argparse
from collections import defaultdict

def find_image_file(images_dir, base_name):
    """
    Given the images directory and a base file name,
    search for an image with common extensions.
    """
    for ext in ['.jpg', '.jpeg', '.png']:
        image_path = os.path.join(images_dir, base_name + ext)
        if os.path.exists(image_path):
            return image_path
    return None

def parse_yolo_annotation(label_file_path):
    """
    Parse a YOLO annotation file.
    Each line is expected to be in the format:
      <class> <x_center> <y_center> <width> <height>
    where the coordinates are normalized (i.e. 0 to 1).
    Returns a dict mapping label (as a string) to a list of bounding boxes.
    Each bounding box is a tuple: (x_center, y_center, width, height).
    """
    boxes_by_label = defaultdict(list)
    with open(label_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"Skipping invalid line in {label_file_path}: {line}")
                continue
            cls, x_center, y_center, w, h = parts
            try:
                cls = str(cls)
                x_center = float(x_center)
                y_center = float(y_center)
                w = float(w)
                h = float(h)
                boxes_by_label[cls].append((x_center, y_center, w, h))
            except ValueError:
                print(f"Skipping invalid values in {label_file_path}: {line}")
                continue
    return boxes_by_label

def draw_bboxes_on_image(image, bboxes, label):
    """
    Draw the given bounding boxes (in YOLO format) on the image.
    The YOLO format uses normalized coordinates.
    """
    img_h, img_w = image.shape[:2]
    for bbox in bboxes:
        x_center, y_center, w, h = bbox
        # Convert to absolute pixel coordinates (top-left and bottom-right)
        x1 = int((x_center - w / 2) * img_w)
        y1 = int((y_center - h / 2) * img_h)
        x2 = int((x_center + w / 2) * img_w)
        y2 = int((y_center + h / 2) * img_h)
        # Draw rectangle and put label text
        color = (0, 255, 0)  # Green color
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def main(dataset_dir, output_dir):
    """
    The main function that:
      - Looks for images in dataset_dir/images and annotations in dataset_dir/labels.
      - For each annotation file, finds the corresponding image and parses bounding boxes.
      - For each label, collects up to 5 images.
      - Draws the bounding boxes for that label on the image and saves the result.
    """
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    if not os.path.exists(images_dir):
        print(f"Images directory does not exist: {images_dir}")
        return
    if not os.path.exists(labels_dir):
        print(f"Labels directory does not exist: {labels_dir}")
        return

    # Dictionary mapping label to a list of (image_path, bounding boxes for that label)
    label_to_images = defaultdict(list)

    # Iterate over all annotation files
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue
        label_file_path = os.path.join(labels_dir, label_file)
        base_name = os.path.splitext(label_file)[0]
        image_file_path = find_image_file(images_dir, base_name)
        if image_file_path is None:
            print(f"Image file not found for {label_file}")
            continue

        boxes_by_label = parse_yolo_annotation(label_file_path)
        # For each label found in this file, add the image (if we haven't reached 5 images yet)
        for label, bboxes in boxes_by_label.items():
            if len(label_to_images[label]) < 5:
                label_to_images[label].append((image_file_path, bboxes))

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process and save images for each label
    for label, image_entries in label_to_images.items():
        # Create a subdirectory for the label
        label_output_dir = os.path.join(output_dir, f"label_{label}")
        if not os.path.exists(label_output_dir):
            os.makedirs(label_output_dir)
        for idx, (img_path, bboxes) in enumerate(image_entries):
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue
            # Draw bounding boxes (only for the current label)
            image_with_boxes = draw_bboxes_on_image(image, bboxes, label)
            base_image_name = os.path.basename(img_path)
            output_path = os.path.join(label_output_dir, f"{os.path.splitext(base_image_name)[0]}_{idx}.jpg")
            cv2.imwrite(output_path, image_with_boxes)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Grab 5 images with bounding boxes for each label from a YOLO dataset."
    # )
    # parser.add_argument(
    #     "--dataset_dir",
    #     type=str,
    #     required=True,
    #     help="Path to the YOLO dataset directory containing 'images' and 'labels' folders."
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     required=True,
    #     help="Directory to save the output images with bounding boxes."
    # )
    # args = parser.parse_args()

    # dataset_dir = args.dataset_dir 
    # output_dir = args.output_dir

    dataset_dir = "/data/TGSSE/hololens_drone_only/"
    output_dir = "temp2/"

    main(dataset_dir, output_dir)
