#!/usr/bin/env python3
import os
import shutil
import argparse

def has_objects(label_path):
    """
    Return True if the label file exists and contains at least one valid YOLO line
    (at least 5 whitespace-separated entries).
    """
    if not os.path.isfile(label_path):
        return False
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                return True
    return False

def split_yolo_by_object(data_path, save_path):
    # Prepare output directories
    out_with = os.path.join(save_path, "with_objects")
    out_without = os.path.join(save_path, "without_objects")
    for base in (out_with, out_without):
        os.makedirs(os.path.join(base, "images"), exist_ok=True, mode=0o777)
        os.makedirs(os.path.join(base, "labels"), exist_ok=True, mode=0o777)

    # Walk all subdirectories
    for dirpath, dirnames, _ in os.walk(data_path):
        if "images" in dirnames and "labels" in dirnames:
            images_dir = os.path.join(dirpath, "images")
            labels_dir = os.path.join(dirpath, "labels")
            # Build underscore-joined prefix from the path relative to data_path
            rel = os.path.relpath(images_dir, data_path)
            prefix = "" if rel == "." else rel.replace(os.sep, "_") + "_"

            # Process each image file
            for fname in os.listdir(images_dir):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img_src = os.path.join(images_dir, fname)
                lbl_name = os.path.splitext(fname)[0] + ".txt"
                lbl_src = os.path.join(labels_dir, lbl_name)

                # Decide destination based on whether label has objects
                if has_objects(lbl_src):
                    dest_img_dir = os.path.join(out_with, "images")
                    dest_lbl_dir = os.path.join(out_with, "labels")
                else:
                    dest_img_dir = os.path.join(out_without, "images")
                    dest_lbl_dir = os.path.join(out_without, "labels")

                out_fname     = prefix + fname
                out_lbl_name  = prefix + lbl_name

                # Copy image
                shutil.copy2(img_src, os.path.join(dest_img_dir, out_fname))
                # Copy or touch label
                if os.path.isfile(lbl_src):
                    shutil.copy2(lbl_src, os.path.join(dest_lbl_dir, out_lbl_name))
                else:
                    open(os.path.join(dest_lbl_dir, out_lbl_name), 'w').close()

    print("Done splitting YOLO dataset.")

def main():
    parser = argparse.ArgumentParser(
        description="Split a YOLO-format dataset into with_objects/without_objects"
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Root directory containing subfolders with images/ and labels/"
    )
    parser.add_argument(
        "--save_path",
        required=True,
        help="Directory under which 'with_objects' and 'without_objects' will be created"
    )
    args = parser.parse_args()
    split_yolo_by_object(args.data_path, args.save_path)

if __name__ == "__main__":
    main()
