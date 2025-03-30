#!/bin/bash
# setup_venv.sh
# This script creates a Python virtual environment at the given venv_path,
# activates it, and installs requirements from a specified file or from
# "requirements.txt" if available.

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <venv_path> [requirements_file]"
    exit 1
fi

VENV_PATH=$1

# Create the virtual environment
python3 -m venv "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment at $VENV_PATH"
    exit 1
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Determine the requirements file to use
if [ "$#" -ge 2 ]; then
    REQUIREMENTS_FILE=$2
elif [ -f requirements.txt ]; then
    REQUIREMENTS_FILE=requirements.txt
else
    REQUIREMENTS_FILE=""
fi

# Install requirements if a file was provided or found
if [ -n "$REQUIREMENTS_FILE" ]; then
    pip install -r "$REQUIREMENTS_FILE"
fi

echo "Virtual environment setup complete."
