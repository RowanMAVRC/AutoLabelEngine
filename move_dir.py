#!/usr/bin/env python3
"""
move_dir.py

Recursively move a directory and all its contents, overwriting the destination if it already exists.

Usage:
    python move_dir.py --src_dir /path/to/source --dst_dir /path/to/destination
"""

import argparse
import shutil
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Move a directory (overwrite destination if it exists)"
    )
    parser.add_argument(
        "--src_dir",
        required=True,
        help="Path to the source directory to move"
    )
    parser.add_argument(
        "--dst_dir",
        required=True,
        help="Path to the destination (will be overwritten if it exists)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    src = os.path.abspath(args.src_dir)
    dst = os.path.abspath(args.dst_dir)

    # Check source exists and is a directory
    if not os.path.isdir(src):
        print(f"Error: source directory '{src}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Ensure destination's parent directory exists
    parent = os.path.dirname(dst.rstrip(os.sep))
    if parent and not os.path.exists(parent):
        try:
            os.makedirs(parent, exist_ok=True)
        except Exception as e:
            print(f"Error creating destination parent '{parent}': {e}", file=sys.stderr)
            sys.exit(1)

    # If destination exists, remove it to allow overwrite
    if os.path.exists(dst):
        try:
            shutil.rmtree(dst)
            print(f"Removed existing destination '{dst}'")
        except Exception as e:
            print(f"Error removing existing destination '{dst}': {e}", file=sys.stderr)
            sys.exit(1)

    # Perform the move
    try:
        shutil.move(src, dst)
        print(f"Successfully moved '{src}' → '{dst}'")
    except Exception as e:
        print(f"Failed to move directory: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
