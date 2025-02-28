import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO  # Ensure you have installed ultralytics (pip install ultralytics)
from tqdm import tqdm  # Progress bar

def convert_to_yolo_format(box, img_width, img_height):
    """
    Convert box coordinates from (x1, y1, x2, y2) to YOLO format:
    (class, center_x, center_y, width, height) where coordinates are normalized.
    """
    # Extract coordinates
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    # Calculate center, width, and height
    center_x = (x1 + x2) / 2.0 / img_width
    center_y = (y1 + y2) / 2.0 / img_height
    box_width = (x2 - x1) / img_width
    box_height = (y2 - y1) / img_height
    # Get the predicted class (as integer)
    cls = int(box.cls[0].item())
    return cls, center_x, center_y, box_width, box_height

def process_images(model_weights, images_dir, labels_dir, gpu_number):
    # Create the labels directory if it doesn't exist
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"Labels directory is set to: {labels_dir.resolve()}")

    # Load the YOLO model on the specified GPU device
    device = f"cuda:{gpu_number}" if gpu_number >= 0 else "cpu"
    print(f"Loading model weights from {model_weights} on device {device}...")
    model = YOLO(model_weights).to(device)
    print("Model loaded successfully.\n")

    # Process each image with extension png, jpg, or jpeg
    images_path = Path(images_dir)
    image_files = [p for p in images_path.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    if not image_files:
        print("No images found in the specified directory.")
        return

    print(f"Found {len(image_files)} images. Starting processing...")
    for img_path in tqdm(image_files, desc="Processing images"):
        # Read image to get dimensions
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Unable to read {img_path}")
            continue
        img_height, img_width = image.shape[:2]

        # Run inference on the image using the model
        results = model.predict(str(img_path), verbose=False)
        # For each detected box, convert and save in YOLO format
        label_file = labels_dir / f"{img_path.stem}.txt"
        with open(label_file, "w") as f:
            # Assuming results[0] corresponds to the processed image
            boxes = results[0].boxes
            for box in boxes:
                cls, center_x, center_y, box_width, box_height = convert_to_yolo_format(box, img_width, img_height)
                # Write the detection in YOLO format: class center_x center_y width height
                f.write(f"{cls} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n")
    print("All images processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-label images using a YOLO model and save detections in YOLO format."
    )
    parser.add_argument("--model_weights_path", help="Path to the YOLO model weights file", required=True)
    parser.add_argument("--images_dir_path", help="Path to the directory containing images (PNG/JPG)", required=True)
    parser.add_argument("--labels_save_path", help="Path to the directory where label files will be saved", required=True)
    parser.add_argument("--gpu_number", type=int, default=0, help="GPU number to use. Set to -1 to use CPU.")
    args = parser.parse_args()

    process_images(args.model_weights_path, args.images_dir_path, args.labels_save_path, args.gpu_number)
