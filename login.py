#!/usr/bin/env python3
import os
import re
import socket
import subprocess
import time


import streamlit as st
from streamlit.components.v1 import html


def sanitize_username(name: str) -> str:
    """Return a safe version of the username for filesystem usage."""
    return re.sub(r"[^0-9A-Za-z_-]", "_", name.strip()).lower()


def find_free_port(start: int = 8600, end: int = 8700) -> int:
    """Find an available TCP port in the given range."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port available")


def start_session(username: str) -> int:
    """Launch autolabel_gui.py in a new tmux session and return its port."""
    safe = sanitize_username(username)
    session = f"ale_{safe}"
    port = find_free_port()
    cmd = (
        f"streamlit run --server.headless True --server.fileWatcherType none "
        f"--server.port {port} autolabel_gui.py"
    )
    subprocess.check_call(["tmux", "new-session", "-d", "-s", session, cmd])
    return port


st.title("AutoLabelEngine Login")
user = st.text_input("Username")
if st.button("Start Session"):
    if not user.strip():
        st.error("Please enter a username")
    else:
        try:
            port = start_session(user)
            st.success(f"Session started for {user} on port {port}.")
            base_url = os.getenv("AUTO_LABEL_BASE_URL", "http://<server-ip>")
            full_url = f"{base_url}:{port}"
            st.write(f"Open {full_url} in a new tab.")
            html(f"<script>window.open('{full_url}', '_blank');</script>", height=0)
        except Exception as e:
            st.error(f"Failed to start session: {e}")
