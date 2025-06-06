#!/usr/bin/env python3
"""Streamlit login app that launches per-user GUI instances."""
import os
import subprocess
import streamlit as st
from get_login import verify_user, allocate_cpu, load_json, save_json

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
password = st.text_input("Password", type="password")
if st.button("Login"):
    if not username or not password:
        st.error("Please enter a username and password")
    elif verify_user(username, password):
        cpu = allocate_cpu(username)
        port = allocate_port(username)
        env = os.environ.copy()
        env["ALE_USER"] = username
        env["CPU_CORE"] = cpu
        env["STREAMLIT_PORT"] = str(port)
        subprocess.Popen(["bash", "run_autolabel_gui.sh"], env=env)
        host = os.environ.get("NGROK_HOST", "localhost")
        url = f"http://{host}:{port}"
        st.success(f"Launching GUI for {username} on {url}")
        st.markdown(
            f'<meta http-equiv="refresh" content="0; URL={url}" />',
            unsafe_allow_html=True,
        )
    else:
        st.error("Invalid username or password")

