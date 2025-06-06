#!/usr/bin/env python3
import re
import os
import socket
import subprocess
import time
import uuid


import streamlit as st


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


def get_network_ip() -> str:
    """Best-effort attempt to obtain the machine's LAN IP."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def start_session(username: str) -> tuple[int, str]:
    """Launch autolabel_gui.py in a new tmux session and return (port, url).

    Each session name uses a random hash so multiple logins by the same user
    do not clash.
    """
    safe = sanitize_username(username)
    unique = uuid.uuid4().hex[:8]
    session = f"ale_{unique}"
    port = find_free_port()
    log = f"/tmp/{session}.log"
    cmd = (
        f"streamlit run --server.headless True --server.fileWatcherType none "
        f"--server.port {port} autolabel_gui.py -- --user {safe} > {log} 2>&1"
    )
    subprocess.check_call(["tmux", "new-session", "-d", "-s", session, cmd])

    url = None
    for _ in range(40):
        if os.path.exists(log):
            text = open(log).read()
            m = re.search(r"Network URL:\s*(https?://[^\s]+)", text)
            if m:
                url = m.group(1)
                break
        time.sleep(0.5)

    if not url:
        base = os.environ.get("AUTO_LABEL_BASE_URL")
        if not base:
            base = get_network_ip()
        if not base.startswith("http"):
            base = "http://" + base
        url = f"{base}:{port}"

    return port, url


st.title("AutoLabelEngine Login")
user = st.text_input("Username")
if st.button("Start Session"):
    if not user.strip():
        st.error("Please enter a username")
    else:
        try:
            port, url = start_session(user)
            st.success(f"Session started for {user} on port {port}.")
            st.markdown(
                f"<script>window.open('{url}', '_blank');</script>",
                unsafe_allow_html=True,
            )
            st.write(f"Opened {url} in a new tab.")
        except Exception as e:
            st.error(f"Failed to start session: {e}")
