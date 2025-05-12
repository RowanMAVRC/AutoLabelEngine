#!/usr/bin/env python3
"""
convert_labels_to_zero.py

Recursively walk through a root directory and, for every .txt file,
set the class ID in each line to 0 while preserving all bounding-box values.
"""

import os
import argparse

def process_label_file(path: str) -> None:
    """Read a YOLO-format .txt file and rewrite every class ID to 0."""
    with open(path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # replace the class index (first element) with '0'
        new_line = '0'
        if len(parts) > 1:
            new_line += ' ' + ' '.join(parts[1:])
        new_lines.append(new_line)

    # overwrite the file in-place
    with open(path, 'w') as f:
        for nl in new_lines:
            f.write(nl + '\n')

def main(root_dir: str) -> None:
    count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith('.txt'):
                full_path = os.path.join(dirpath, fn)
                process_label_file(full_path)
                count += 1
                print(f"[{count}] updated {full_path}")
    print(f"Done. Processed {count} label files.")

if __name__ == "__main__":
   
    main("/data/TGSSE/AutoLabelEngine/yolo_format_data/to_be_reviewed/Batch 2/Recon/IMG_7189(1)/")
