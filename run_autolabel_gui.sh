#!/bin/bash
source /home/naddeok5/envs/autolabel//bin/activate
streamlit run --server.headless True --server.fileWatcherType none autolabel_gui.py