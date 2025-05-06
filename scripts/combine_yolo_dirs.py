#!/usr/bin/env python3
import os
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm

def combine_yolo_dirs(yolo_dirs, dst_dir):
    """
    Combine multiple YOLO directories into a single destination directory with prefixed filenames.
    Strips any leading/trailing brackets, commas or quotes from each path.
    """
    # --- sanitize incoming list entries ---
    clean = []
    for p in yolo_dirs:
        c = p.strip().strip("[],'\"")
        if c:
            clean.append(c)
    yolo_dirs = clean

    dst_images_dir = Path(dst_dir) / "images"
    dst_labels_dir = Path(dst_dir) / "labels"
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    # Build unique prefixes
    prefixes = {}
    for yolo_dir in yolo_dirs:
        p = Path(yolo_dir)
        prefix = f"{p.parent.name}_{p.name}"
        while prefix in prefixes.values():
            prefix = f"{p.parent.parent.name}_{prefix}"
        prefixes[yolo_dir] = prefix.lstrip("_")

    # Copy with prefixes
    for yolo_dir, prefix in prefixes.items():
        p = Path(yolo_dir)
        imgs = p / "images"
        lbls = p / "labels"
        if not imgs.exists() or not lbls.exists():
            print(f"Skipping {yolo_dir!r}: missing 'images' or 'labels'")
            continue

        for img in tqdm(list(imgs.glob("*.*")), desc=f"Images from {yolo_dir}", unit="file"):
            shutil.copy(img, dst_images_dir / f"{prefix}_{img.name}")

        for label in tqdm(list(lbls.glob("*.*")), desc=f"Labels from {yolo_dir}", unit="file"):
            shutil.copy(label, dst_labels_dir / f"{prefix}_{label.name}")

    print(f"[✓] Combined all datasets into {dst_dir}")

def split_dataset(dst_dir, val_size=5000, test_size=1000, seed=42):
    """
    After combining, split images+labels into top-level train/val/test directories,
    each with its own images/ and labels/ subfolders.
    """
    dst = Path(dst_dir)
    combined_imgs = dst / "images"
    combined_lbls = dst / "labels"

    # Gather and shuffle
    all_imgs = [f for f in combined_imgs.glob("*.*") if f.is_file()]
    random.seed(seed)
    random.shuffle(all_imgs)

    n = len(all_imgs)
    if n < val_size + test_size:
        raise ValueError(f"Not enough images ({n}) for val_size={val_size} + test_size={test_size}")

    n_train = n - val_size - test_size
    splits = {
        "train": all_imgs[:n_train],
        "val":   all_imgs[n_train:n_train + val_size],
        "test":  all_imgs[n_train + val_size:]
    }

    # Move files into top-level split dirs
    for split_name, img_list in splits.items():
        img_dest = dst / split_name / "images"
        lbl_dest = dst / split_name / "labels"
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(img_list, desc=f"Moving {split_name}", unit="file"):
            # move image
            shutil.move(str(img_path), img_dest / img_path.name)
            # move label if exists
            lbl_src = combined_lbls / f"{img_path.stem}.txt"
            if lbl_src.exists():
                shutil.move(str(lbl_src), lbl_dest / lbl_src.name)

    # clean up the now-empty combined dirs
    try:
        shutil.rmtree(combined_imgs)
        shutil.rmtree(combined_lbls)
    except Exception:
        pass

    print(f"[✓] Split into: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine multiple YOLO datasets then split into train/val/test."
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="One or more YOLO dataset dirs (each with 'images' & 'labels')."
    )
    parser.add_argument(
        "--dst_dir", required=True,
        help="Destination dir for combined + split data."
    )
    parser.add_argument(
        "--val_size", type=int, default=5000,
        help="Number of samples to use for the validation set."
    )
    parser.add_argument(
        "--test_size", type=int, default=1000,
        help="Number of samples to use for the test set."
    )
    args = parser.parse_args()

    combine_yolo_dirs(args.datasets, args.dst_dir)
    split_dataset(args.dst_dir, val_size=args.val_size, test_size=args.test_size)
