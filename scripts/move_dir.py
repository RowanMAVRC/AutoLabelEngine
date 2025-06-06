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


def directory_contains_files(directory):
    """Return True if any file exists within directory tree."""
    for _, _, files in os.walk(directory):
        if files:
            return True
    return False


def prune_empty_parents(start_dir, stop_dir):
    """Delete parent directories up to stop_dir when no files remain."""
    current = os.path.abspath(start_dir)
    stop_dir = os.path.abspath(stop_dir) if stop_dir else ""
    while True:
        if current in ("/", "") or current == stop_dir or not current.startswith(stop_dir):
            break
        if directory_contains_files(current):
            break
        try:
            shutil.rmtree(current)
            print(f"Pruned empty directory: {current}")
        except Exception as e:
            print(f"Failed to prune {current}: {e}", file=sys.stderr)
            break
        current = os.path.dirname(current)

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
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Prune empty parent directories after moving"
    )
    parser.add_argument(
        "--prune_height",
        type=str,
        default=None,
        help="Stop pruning when this directory is reached (it is not removed)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir

    src = os.path.abspath(src_dir)
    dst = os.path.abspath(dst_dir)

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
        print("[ABORTED] Destination path already exists, please pick a new path or move the current one.")
        sys.exit(1)

    # Perform the move
    try:
        shutil.move(src, dst)
        print(f"Successfully moved '{src}' → '{dst}'")
        if args.prune and args.prune_height:
            prune_empty_parents(os.path.dirname(src), args.prune_height)
    except Exception as e:
        print(f"Failed to move directory: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
