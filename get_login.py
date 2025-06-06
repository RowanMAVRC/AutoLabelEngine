#!/usr/bin/env python3
"""Simple login script that launches the GUI with per-user CPU pinning."""
import os
import json
import subprocess
import sys

CPU_FILE = "cpu_allocation.json"


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)




def allocate_cpu(username):
    cpu_map = load_json(CPU_FILE)
    if username in cpu_map:
        return cpu_map[username]
    cores = os.cpu_count() or 1
    used = set(cpu_map.values())
    for i in range(cores):
        if str(i) not in used:
            cpu_map[username] = str(i)
            save_json(CPU_FILE, cpu_map)
            return str(i)
    # fallback if all cores used
    cpu_map[username] = "0"
    save_json(CPU_FILE, cpu_map)
    return "0"


def main():
    username = input("Username: ").strip()
    if not username:
        print("Username required.")
        sys.exit(1)
    cpu = allocate_cpu(username)
    print(f"Launching GUI for {username} on CPU core {cpu}...")
    env = os.environ.copy()
    env["ALE_USER"] = username
    env["CPU_CORE"] = cpu
    subprocess.Popen(["bash", "run_autolabel_gui.sh"], env=env)
    print("GUI started. You can close this window; the process continues in the background.")


if __name__ == "__main__":
    main()
