#!/bin/bash

# Activate the virtual environment
source /home/naddeok5/envs/auto-label-engine/bin/activate

# Run the Streamlit application in headless mode with file watcher disabled
streamlit run --server.headless True --server.fileWatcherType none autolabel_gui.py
