#!/bin/bash
# setup_venv.sh
# This script creates a Python virtual environment at the given venv_path,
# installs python3/python3-venv if missing, activates it, and installs
# requirements from a specified file (default: requirements.txt).

set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <venv_path> [requirements_file]"
    exit 1
fi

VENV_PATH=$1
REQ_FILE=${2:-requirements.txt}

install_pkg() {
    PKG_NAME=$1
    if command -v apt-get &> /dev/null; then
        echo "Installing $PKG_NAME via apt-get..."
        sudo apt-get update && sudo apt-get install -y "$PKG_NAME"
    elif command -v yum &> /dev/null; then
        echo "Installing $PKG_NAME via yum..."
        sudo yum install -y "$PKG_NAME"
    else
        echo "Error: No supported package manager found (apt-get or yum)."
        echo "Please install $PKG_NAME manually."
        exit 1
    fi
}

# 1) Ensure python3 is available (install if not)
if ! command -v python3 &> /dev/null; then
    echo "python3 not found. Attempting to install python3..."
    install_pkg python3
fi

# 2) Ensure the venv module is available (install python3-venv if needed)
if ! python3 -c "import venv" &> /dev/null; then
    echo "Python venv module not found. Attempting to install python3-venv..."
    install_pkg python3-venv
fi

# 3) Create the virtual environment
python3 -m venv "$VENV_PATH"

# 4) Activate it
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

# 5) Upgrade pip
pip install --upgrade pip

# 6) Install requirements if the file exists
if [ -f "$REQ_FILE" ]; then
    pip install -r "$REQ_FILE"
else
    echo "Warning: Requirements file '$REQ_FILE' not found — skipping package installation."
fi

echo "✅ Virtual environment ready at '$VENV_PATH'"
