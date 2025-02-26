#!/bin/bash
source /home/naddeok5/envs/auto-label-engine/bin/activate
streamlit run --server.headless True --server.fileWatcherType none autolabel_gui.py