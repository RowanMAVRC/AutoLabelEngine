import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO  # Ensure you have installed ultralytics (pip install ultralytics)
from tqdm import tqdm
import os

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

from pathlib import Path
import cv2
import os
from tqdm import tqdm
from ultralytics import YOLO  # Ensure YOLO (ultralytics) is installed

def convert_to_yolo_format(box, img_width, img_height):
    """
    Convert box coordinates from (x1, y1, x2, y2) to YOLO format:
    (class, center_x, center_y, width, height) with normalized values.
    """
    # Extract coordinates from the box (assumes single box per result)
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    center_x = (x1 + x2) / 2.0 / img_width
    center_y = (y1 + y2) / 2.0 / img_height
    box_width = (x2 - x1) / img_width
    box_height = (y2 - y1) / img_height
    cls = int(box.cls[0].item())
    return cls, center_x, center_y, box_width, box_height

def process_images_dirs(model_weights, base_dir, label_replacement, gpu_number):
    """
    Searches recursively within the provided base directory for subdirectories 
    named exactly "images" (case-insensitive). For each such images directory found,
    it computes a new save path by replacing the "images" folder with the provided 
    label_replacement (e.g. "labels") in that directory's parent, then auto-labels
    all image files within the images directory using the YOLO model. Each image 
    file's label file is stored in the corresponding labels directory so that the 
    original folder structure is preserved.
    
    Args:
        model_weights (str): Path to the YOLO model weights file.
        base_dir (str): Base directory under which to search for subdirectories named "images".
        label_replacement (str): Folder name to use in place of "images" (e.g. "labels").
        gpu_number (int): GPU device number to use (or -1 to use the CPU).
    """
    # Load the YOLO model on the specified device.
    device = f"cuda:{gpu_number}" if gpu_number >= 0 else "cpu"
    print(f"Loading model weights from {model_weights} on device {device}...")
    model = YOLO(model_weights).to(device)
    print("Model loaded successfully.\n")
    
    base_dir = Path(base_dir)
    # Find all subdirectories whose name is exactly "images" (case-insensitive)
    images_dirs = [d for d in base_dir.rglob("*") if d.is_dir() and d.name.lower() == "images"]
    
    if not images_dirs:
        print("No images directories found in the base directory.")
        return

    print(f"Found {len(images_dirs)} images directories. Starting processing...")
    
    for images_folder in images_dirs:
        print(f"\n--- Processing folder: {images_folder} ---")
        # Compute the corresponding labels folder by replacing the images folder with the replacement.
        # For example, if images_folder is /.../Copy of area_denial_001/images,
        # then label_folder will be /.../Copy of area_denial_001/labels.
        label_folder = images_folder.parent / label_replacement
        os.makedirs(label_folder, exist_ok=True)
        
        # Process all image files (non-recursively) within this images folder.
        image_files = [p for p in images_folder.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        print(f"Found {len(image_files)} images in {images_folder}")
        
        for img_path in tqdm(image_files, desc=f"Processing {images_folder}", leave=False):
            # Read image for dimensions.
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Unable to read {img_path}")
                continue
            img_height, img_width = image.shape[:2]
            
            # Run YOLO inference on the image.
            results = model.predict(str(img_path), verbose=False)
            
            # Build the label file path inside the computed label_folder using the image's base name.
            label_file = label_folder / (img_path.stem + ".txt")
            os.makedirs(label_file.parent, exist_ok=True)
            
            with open(label_file, "w") as f:
                # Loop over each detected box in the first result (assumes results[0] corresponds to the image)
                boxes = results[0].boxes
                for box in boxes:
                    cls, center_x, center_y, box_width, box_height = convert_to_yolo_format(box, img_width, img_height)
                    f.write(f"{cls} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n")
        print(f"Finished processing folder: {images_folder}")
    print("All images directories processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-label images using a YOLO model. " +
                    "The labels directory is automatically computed by replacing the last folder in the images path ('images') with the provided label replacement."
    )
    parser.add_argument("--model_weights_path", required=True, help="Path to the YOLO model weights file")
    parser.add_argument("--images_dir_path", required=True, help="Path to the directory containing images (should be the 'images' folder)")
    parser.add_argument("--label_replacement", required=True, help="Replacement for the 'images' folder (e.g., 'labels')")
    parser.add_argument("--gpu_number", type=int, default=0, help="GPU number to use. Set to -1 to use CPU.")
    args = parser.parse_args()

    process_images_dirs(args.model_weights_path, args.images_dir_path, args.label_replacement, args.gpu_number)
