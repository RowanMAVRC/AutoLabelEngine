#!/usr/bin/env python3
"""Streamlit login app that launches per-user GUI instances."""
import os
import subprocess
import time
import re
import shutil
import streamlit as st
from get_login import allocate_cpu, load_json, save_json

PORT_FILE = "port_allocation.json"

NGROK_PROCS = []


def start_ngrok(port: int) -> str:
    """Start an ngrok tunnel for the given port and return the public URL."""
    if shutil.which("ngrok") is None:
        return ""
    cmd = ["ngrok", "http", str(port), "--log=stdout", "--log-format=logfmt"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    NGROK_PROCS.append(proc)

    url = ""
    # Parse ngrok stdout for the public URL
    for _ in range(40):
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        if "started tunnel" in line and "url=" in line:
            m = re.search(r"url=(https?://[^\s]+)", line)
            if m:
                url = m.group(1)
                break
    return url


def allocate_port(username, base=8600):
    ports = load_json(PORT_FILE)
    if username in ports:
        return int(ports[username])
    port = base
    used = {int(p) for p in ports.values()}
    while port in used:
        port += 1
    ports[username] = str(port)
    save_json(PORT_FILE, ports)
    return port


st.title("Auto Label Engine Login")
username = st.text_input("Username")
login_pressed = st.button("Login")
if login_pressed:
    if not username:
        st.error("Please enter a username")
    else:
        cpu = allocate_cpu(username)
        port = allocate_port(username)
        env = os.environ.copy()
        env["ALE_USER"] = username
        env["CPU_CORE"] = cpu
        env["STREAMLIT_PORT"] = str(port)
        subprocess.Popen(["bash", "run_autolabel_gui.sh"], env=env)

        host = os.environ.get("NETWORK_HOST", "localhost")
        protocol = "https" if host.endswith(".ngrok.io") else "http"

        if host.endswith(".ngrok.io") and shutil.which("ngrok"):
            st.info("Starting personal ngrok tunnel...")
            ngrok_url = start_ngrok(port)
            url = ngrok_url if ngrok_url else f"{protocol}://{host}:{port}"
        else:
            url = f"{protocol}://{host}:{port}"

        st.success(f"Launching GUI for {username} on {url}")
        st.markdown(
            f'<meta http-equiv="refresh" content="0; URL={url}" />',
            unsafe_allow_html=True,
        )

