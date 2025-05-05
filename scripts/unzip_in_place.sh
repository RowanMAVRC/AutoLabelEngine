#!/usr/bin/env bash
#
# scripts/unzip_in_place.sh
#
# Usage: unzip_in_place.sh <path_to_zip_file>
# Unzips the given archive in its own directory.

if [ $# -lt 1 ]; then
    echo "Usage: $0 <zip_file>"
    exit 1
fi

ZIP_FILE="$1"

if [ ! -f "$ZIP_FILE" ]; then
    echo "Error: File not found - $ZIP_FILE"
    exit 1
fi

TARGET_DIR="$(dirname "$ZIP_FILE")"

echo "Unzipping '$ZIP_FILE' into '$TARGET_DIR'..."
unzip -o "$ZIP_FILE" -d "$TARGET_DIR"

echo "Done."
