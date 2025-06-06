#!/usr/bin/env python3
"""Streamlit login app that launches per-user GUI instances."""
import os
import subprocess
import streamlit as st
from get_login import allocate_cpu, load_json, save_json

PORT_FILE = "port_allocation.json"


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
        url = f"{protocol}://{host}:{port}"
        st.success(f"Launching GUI for {username} on {url}")
        st.markdown(
            f'<meta http-equiv="refresh" content="0; URL={url}" />',
            unsafe_allow_html=True,
        )

