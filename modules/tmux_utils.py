# modules/tmux_utils.py
import subprocess
import time
import streamlit as st

def run_command(command):
    output = []
    terminal_output = st.empty()
    try:
        with subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        ) as process:
            for line in process.stdout:
                output.append(line)
                terminal_output.code("".join(output).replace('\r', '\n'), language="bash")
            for line in process.stderr:
                output.append(line)
                terminal_output.code("".join(output).replace('\r', '\n'), language="bash")
    except Exception as e:
        st.error(f"Error running command: {e}")
    return "".join(output).replace('\r', '\n')

def update_tmux_terminal(session_key):
    try:
        capture_cmd = f"tmux capture-pane -pt {session_key}:0.0"
        output = subprocess.check_output(capture_cmd, shell=True)
        decoded_output = output.decode("utf-8").replace('\r', '\n')
        return decoded_output
    except subprocess.CalledProcessError:
        st.warning("No tmux session")
        return None

def kill_tmux_session(session_key):
    kill_cmd = f"tmux kill-session -t {session_key}"
    subprocess.call(kill_cmd, shell=True)
    st.success(f"tmux session '{session_key}' has been killed.")

def run_in_tmux(session_key, py_file_path, venv_path, args=""):
    try:
        subprocess.check_call(f"tmux kill-session -t {session_key}", shell=True)
    except subprocess.CalledProcessError:
        pass

    if not os.path.exists(py_file_path):
        st.error(f"Python file not found: {py_file_path}")
        return None

    activate_script = os.path.join(venv_path, "bin", "activate")
    if not os.path.exists(activate_script):
        st.error(f"Virtual environment activation script not found: {activate_script}")
        return None

    if isinstance(args, dict):
        args = " ".join(f"--{key} {value}" for key, value in args.items())

    inner_command = f"source {activate_script} && python {py_file_path} {args}; exec bash"
    tmux_cmd = f'tmux new-session -d -s {session_key} "bash -c \'{inner_command}\'"'
    try:
        subprocess.check_call(tmux_cmd, shell=True)
        time.sleep(2)
        capture_cmd = f"tmux capture-pane -pt {session_key}:0.0"
        output = subprocess.check_output(capture_cmd, shell=True)
        decoded_output = output.decode("utf-8")
        return decoded_output
    except subprocess.CalledProcessError as e:
        st.error(f"Error running command in tmux: {e}")
        return None

def check_gpu_status(button_key):
    if st.button("Check GPU Status", key=button_key):
        try:
            output = subprocess.check_output(["gpustat"]).decode("utf-8")
            return output
        except Exception as e:
            st.error(f"Failed to run gpustat: {e}")
