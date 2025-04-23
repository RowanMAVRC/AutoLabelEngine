#!/usr/bin/env python3
# cluster_objects.py

import argparse
import os
import pandas as pd
import torch
import numpy as np
from PIL import Image

def extract_features_tensor(img_crop):
    """
    Compute a simple HSV‐H‐channel histogram as a torch tensor.
    """
    hsv = np.array(img_crop.convert("HSV"), dtype=np.float32)
    h = hsv[..., 0]
    hist = np.histogram(h, bins=256, range=(0, 255))[0].astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return torch.from_numpy(hist)

def load_cluster_csv(path):
    """
    Load a list of reference global indices from cluster_csv.
    Returns an empty list if file is missing or empty.
    """
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path, header=None)
        return df[0].tolist()
    except pd.errors.EmptyDataError:
        return []

def save_cluster_csv(path, indices):
    """
    Save the selected global indices back to cluster_csv.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(indices).to_csv(path, index=False, header=False)

def build_mapping(images_dir):
    """
    Walk images_dir and its parallel labels/ folder to build a DataFrame
    with columns: global_index, image_path, label_path, bbox_x, bbox_y, bbox_w, bbox_h.
    """
    labels_dir = images_dir.replace(os.sep + "images", os.sep + "labels")
    rows = []
    idx = 0

    for img_name in sorted(os.listdir(images_dir)):
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
                w = wn * W
                h = hn * H
                x = xc * W - w / 2
                y = yc * H - h / 2

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
        description="Cluster objects by cosine similarity (auto-mapping + clustering)"
    )
    parser.add_argument(
        "--images_dir", type=str, required=True,
        help="Path to your images/ folder"
    )
    parser.add_argument(
        "--cluster_csv", type=str, default="cluster.csv",
        help="Path to load/save clustered indices"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7,
        help="Cosine similarity cutoff"
    )
    parser.add_argument(
        "--gpu", type=int, default=-1,
        help="GPU index (or -1 for CPU)"
    )
    args = parser.parse_args()

    # 1) Build the object mapping on the fly
    mapping = build_mapping(args.images_dir)
    num_objs = len(mapping)
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    print(f"[cluster] Mapped {num_objs} objects; using device {device}")

    # 2) Load existing reference indices (if any)
    refs = load_cluster_csv(args.cluster_csv)
    print(f"[cluster] Loaded {len(refs)} reference indices from {args.cluster_csv}")

    # 3) Extract features for reference objects
    ref_feats = []
    for idx in refs:
        row = mapping.iloc[idx]
        img = Image.open(row.image_path)
        crop = img.crop((
            int(row.bbox_x), int(row.bbox_y),
            int(row.bbox_x + row.bbox_w),
            int(row.bbox_y + row.bbox_h)
        ))
        ref_feats.append(extract_features_tensor(crop).to(device))

    if not ref_feats:
        print("[cluster] No references → writing empty cluster_csv and exiting.")
        save_cluster_csv(args.cluster_csv, [])
        return

    ref_stack = torch.stack(ref_feats)  # R x D

    # 4) Extract features for all objects
    all_feats = []
    for _, row in mapping.iterrows():
        img = Image.open(row.image_path)
        crop = img.crop((
            int(row.bbox_x), int(row.bbox_y),
            int(row.bbox_x + row.bbox_w),
            int(row.bbox_y + row.bbox_h)
        ))
        all_feats.append(extract_features_tensor(crop).to(device))
    all_stack = torch.stack(all_feats)  # N x D

    # 5) Normalize and compute cosine similarities
    ref_norm = ref_stack / ref_stack.norm(dim=1, keepdim=True)
    all_norm = all_stack / all_stack.norm(dim=1, keepdim=True)
    sims = all_norm @ ref_norm.t()  # N x R
    max_sims, _ = sims.max(dim=1)

    # 6) Apply threshold and save selected indices
    keep_mask = (max_sims >= args.threshold).cpu().tolist()
    selected = [i for i, keep in enumerate(keep_mask) if keep]
    print(f"[cluster] {len(selected)} objects passed threshold {args.threshold}")
    save_cluster_csv(args.cluster_csv, selected)
    print(f"[cluster] Saved clustered indices to {args.cluster_csv}")

if __name__ == "__main__":
    main()
