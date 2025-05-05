#!/usr/bin/env python3
import os
import shutil
import argparse
import re
from collections import defaultdict

def build_mapping(data_path):
    """
    Walk data_path/**/(images, labels) to build a dict mapping
    each split label filename (prefix_base.txt) → original labels/base.txt path.
    """
    mapping = {}
    for dirpath, dirnames, _ in os.walk(data_path):
        if "images" in dirnames and "labels" in dirnames:
            images_dir = os.path.join(dirpath, "images")
            labels_dir = os.path.join(dirpath, "labels")
            rel = os.path.relpath(images_dir, data_path)
            prefix = "" if rel == "." else rel.replace(os.sep, "_") + "_"
            for img_fname in os.listdir(images_dir):
                if not img_fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                base = os.path.splitext(img_fname)[0]
                split_lbl = prefix + base + ".txt"
                orig_lbl = os.path.join(labels_dir, base + ".txt")
                mapping[split_lbl] = orig_lbl
    return mapping

def extract_frame_index(filename):
    """
    Extract the last numeric group from 'prefix_frame032.txt' → '32'
    """
    name = os.path.splitext(filename)[0]
    nums = re.findall(r"(\d+)", name)
    return str(int(nums[-1])) if nums else name

def unsplit_labels(data_path, split_path):
    """
    For each non‐empty label in split_path/without_objects/labels:
      1. Map it via build_mapping() back to the original labels/ folder
      2. Copy it into place
      3. Record its frame index (unpadded) in that folder’s updated_frames.csv
    """
    mapping = build_mapping(data_path)
    src_lbl_dir = os.path.join(split_path, "without_objects", "labels")
    updated = defaultdict(set)

    for lbl_fname in os.listdir(src_lbl_dir):
        src_path = os.path.join(src_lbl_dir, lbl_fname)
        # skip truly empty files
        with open(src_path) as f:
            if not any(line.strip() for line in f):
                continue

        orig_lbl = mapping.get(lbl_fname)
        if not orig_lbl:
            print(f"Warning: no mapping for {lbl_fname}")
            continue

        os.makedirs(os.path.dirname(orig_lbl), exist_ok=True)
        shutil.copy2(src_path, orig_lbl)

        idx = extract_frame_index(lbl_fname)
        parent_dir = os.path.dirname(os.path.dirname(orig_lbl))
        updated[parent_dir].add(idx)

    for parent_dir, indices in updated.items():
        csv_path = os.path.join(parent_dir, "updated_frames.csv")
        with open(csv_path, "w") as f:
            for i in sorted(indices, key=lambda x: int(x)):
                f.write(f"{i}\n")
        print(f"Wrote {len(indices)} entries to {csv_path}")

    print(f"Done. Restored labels in {len(updated)} directories.")

def main():
    parser = argparse.ArgumentParser(
        description="Unsplit YOLO labels: restore new 'no-object' labels back to original structure"
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Root of your original YOLO dataset"
    )
    parser.add_argument(
        "--split_path",
        required=True,
        help="Path where you ran split_yolo_by_object (contains without_objects/labels)"
    )
    args = parser.parse_args()
    unsplit_labels(args.data_path, args.split_path)

if __name__ == "__main__":
    main()
