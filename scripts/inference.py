#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from tqdm import tqdm
import cv2
from tqdm import tqdm
from ultralytics import YOLO 

def convert_to_yolo_format(box, img_width, img_height):
    """
    Convert box coordinates from (x1, y1, x2, y2) to YOLO format:
    (class, center_x, center_y, width, height) with normalized values.
    """
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    center_x = (x1 + x2) / 2.0 / img_width
    center_y = (y1 + y2) / 2.0 / img_height
    box_width = (x2 - x1) / img_width
    box_height = (y2 - y1) / img_height
    cls = int(box.cls[0].item())
    return cls, center_x, center_y, box_width, box_height

def process_images_dirs(model_weights, base_dir, label_replacement, gpu_number, threshold, use_tracking):
    """
    Recursively finds all "images" directories under base_dir, runs YOLO inference
    with a confidence threshold, and writes normalized YOLO-format labels into
    parallel "labels" directories (or whatever you name with label_replacement).
    """
    # Select device
    device = f"cuda:{gpu_number}" if gpu_number >= 0 else "cpu"
    print(f"Loading model weights from {model_weights} on device {device}...")
    model = YOLO(model_weights).to(device)
    print("Model loaded successfully.\n")

    base_dir = Path(base_dir)
    images_dirs = [d for d in base_dir.rglob("*") if d.is_dir() and d.name.lower() == "images"]

    if not images_dirs:
        print("No images directories found in the base directory.")
        return

    print(f"Found {len(images_dirs)} 'images' folder(s). Starting processing with threshold {threshold}...\n")

    total_preds = 0
    for images_folder in tqdm(images_dirs, desc="Processing folder(s)", unit="folder"):
        print(f"--- Processing folder: {images_folder} ---")
        label_folder = images_folder.parent / label_replacement
        os.makedirs(label_folder, exist_ok=True)

        image_files = [
            p for p in images_folder.iterdir()
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
        print(f"Found {len(image_files)} images in {images_folder}")

        for img_path in tqdm(image_files, desc=f"  Images in {images_folder.name}", leave=False, unit="img"):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Unable to read {img_path}")
                continue
            img_height, img_width = image.shape[:2]

            # Run YOLO inference with confidence threshold
            if use_tracking:
                results = model.track(
                    str(img_path),
                    conf=threshold,
                    verbose=False
                )
            else:
                results = model.predict(
                    str(img_path),
                    conf=threshold,
                    verbose=False
                )
            total_preds += len(results[0].boxes)

            label_file = label_folder / (img_path.stem + ".txt")
            os.makedirs(label_file.parent, exist_ok=True)
            with open(label_file, "w") as f:
                for box in results[0].boxes:
                    cls, cx, cy, w, h = convert_to_yolo_format(box, img_width, img_height)
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        print(f"Finished processing folder: {images_folder}\n")

    print(f"Total predictions made: {total_preds}")

    print("All image directories processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Auto-label images using a YOLO model. "
            "Automatically mirrors 'images' folders into corresponding label directories."
        )
    )
    parser.add_argument(
        "--model_weights_path", required=True,
        help="Path to the YOLO model weights file"
    )
    parser.add_argument(
        "--images_dir_path", required=True,
        help="Base directory under which to search for 'images' subfolders"
    )
    parser.add_argument(
        "--label_replacement", required=True,
        help="Replacement name for 'images' folders (e.g., 'labels')"
    )
    parser.add_argument(
        "--gpu_number", type=int, default=0,
        help="GPU number to use (0, 1, ...). Use -1 for CPU."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.25,
        help="Confidence threshold for YOLO detections (0.0–1.0)"
    )
    parser.add_argument(
        "--method",
        choices=["track", "detect"],
        default="detect",
        help="Choose 'track' for YOLO tracking or 'detect' for per-image detection"
    )
    args = parser.parse_args()
    process_images_dirs(
        args.model_weights_path,
        args.images_dir_path,
        args.label_replacement,
        args.gpu_number,
        args.threshold,
        True if args.method == "track" else False
    )
