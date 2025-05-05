#!/usr/bin/env python3
# cluster_objects.py

import argparse
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

def compute_ahash_bits(img, hash_size=8):
    """
    Compute an average-hash bit-vector of length hash_size*hash_size.
    """
    gray = img.convert("L")
    small = gray.resize((hash_size, hash_size), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.uint8).flatten()
    avg = arr.mean()
    return (arr > avg).astype(np.uint8)  # 1D array, length hash_size^2

def hamming_dist(a, b):
    return int(np.count_nonzero(a != b))

def load_cluster_csv(path):
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path, header=None)
        return df[0].astype(int).tolist()
    except pd.errors.EmptyDataError:
        return []

def save_cluster_csv(path, indices):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(indices).to_csv(path, index=False, header=False)

def build_mapping(images_dir):
    labels_dir = images_dir.replace(os.sep + "images", os.sep + "labels")
    rows = []
    idx = 0
    for img_name in tqdm(sorted(os.listdir(images_dir)), desc="Building mapping"):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.isfile(label_path):
            continue

        img = Image.open(img_path)
        W, H = img.size

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, xc, yc, wn, hn = map(float, parts[:5])
                w = wn * W; h = hn * H
                x = xc * W - w/2; y = yc * H - h/2

                rows.append({
                    "global_index": idx,
                    "image_path":   img_path,
                    "label_path":   label_path,
                    "class":        int(cls),
                    "bbox_x":       x,
                    "bbox_y":       y,
                    "bbox_w":       w,
                    "bbox_h":       h
                })
                idx += 1

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(
        description="Cluster objects by average-hash + Hamming distance"
    )
    parser.add_argument("--images_dir",   type=str, required=True,
                        help="Path to your images/ folder")
    parser.add_argument("--cluster_csv",  type=str, default="cluster.csv",
                        help="Path to load/save clustered indices")
    parser.add_argument("--threshold",    type=int, default=10,
                        help="Max Hamming distance (0–64) allowed for similarity")
    parser.add_argument("--gpu",          type=int, default=-1,
                        help="GPU index (unused by this method)")
    args = parser.parse_args()

    # 1) Build mapping
    mapping = build_mapping(args.images_dir)
    N = len(mapping)
    print(f"[cluster] Mapped {N} objects")

    # 2) Load reference indices
    refs = load_cluster_csv(args.cluster_csv)
    R = len(refs)
    print(f"[cluster] Loaded {R} reference indices from {args.cluster_csv}")

    if R == 0:
        print("[cluster] No references → writing empty cluster_csv and exiting")
        save_cluster_csv(args.cluster_csv, [])
        return

    # 3) Compute reference hashes
    ref_hashes = []
    for idx in tqdm(refs, desc="Hashing references"):
        r = mapping.iloc[idx]
        img = Image.open(r.image_path)
        crop = img.crop((
            int(r.bbox_x), int(r.bbox_y),
            int(r.bbox_x + r.bbox_w),
            int(r.bbox_y + r.bbox_h)
        ))
        ref_hashes.append(compute_ahash_bits(crop))

    # 4) Hash all objects and apply threshold
    selected = []
    for i, r in tqdm(mapping.iterrows(), total=N, desc="Clustering by aHash"):
        img = Image.open(r.image_path)
        crop = img.crop((
            int(r.bbox_x), int(r.bbox_y),
            int(r.bbox_x + r.bbox_w),
            int(r.bbox_y + r.bbox_h)
        ))
        h = compute_ahash_bits(crop)
        min_dist = min(hamming_dist(h, rh) for rh in ref_hashes)
        if min_dist <= args.threshold:
            selected.append(i)

    print(f"[cluster] {len(selected)} / {N} objects within Hamming ≤ {args.threshold}")
    save_cluster_csv(args.cluster_csv, selected)
    print(f"[cluster] Saved clustered indices to {args.cluster_csv}")

if __name__ == "__main__":
    main()
