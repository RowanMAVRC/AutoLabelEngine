#!/bin/bash

# Check if the virtual environment activation file exists
if [ ! -f "../envs/auto-label-engine/bin/activate" ]; then
    echo "Virtual environment not found. Setting up environment..."
    bash setup_venv.sh ../envs/auto-label-engine
fi

# Activate the virtual environment
source ../envs/auto-label-engine/bin/activate

# Run the Streamlit application in headless mode
streamlit run --server.headless True --server.fileWatcherType none autolabel_gui.py
