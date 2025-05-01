# ------------------------------------------------------------------------------------------------------------------------
## Imports
# ------------------------------------------------------------------------------------------------------------------------
## Standard Library

import os
os.umask(0) # remove any default umask restrictions so that os.makedirs(…, mode=0o777) really gets 777
import re
import glob
import time
import base64
import random
import shutil
import subprocess
import zipfile
import hashlib
import uuid
import io

## Third-Party Libraries 
import json
import pandas as pd
from io import BytesIO
import cv2
import yaml
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageSequenceClip, VideoClip, clips_array
from pathlib import Path
import tempfile, json, shlex
import matplotlib.pyplot as plt

## Streamlit-Specific

from streamlit_label_kit import detection as _orig_detection
from streamlit_ace import st_ace

#-------------------------------------------------------------------------------------------------------------------------#
## Functions
#-------------------------------------------------------------------------------------------------------------------------#

## Save and load from session_state

SELECTED_KEYS = [
    "auto_label_gpu",
    "data_cfg",
    "frame_index",
    "gpu_list",
    "images_dir",
    "label_list",
    "paths",
    "python_codes",
    "yamls",
    "unverified_image_scale",
    "global_object_index",
    "video_image_scale"

]

def load_session_state(default_yaml_path="cfgs/gui/session_state/default.yaml"):
    """
    Load selected session state keys from a YAML file.
    If the file doesn't exist, create it with the current session state.
    This version removes null characters that might corrupt the file.
    """
    if os.path.exists(default_yaml_path):
        try:
            with open(default_yaml_path, "r") as f:
                content = f.read()
                # Remove null characters that can cause YAML parsing errors
                content = content.replace("\x00", "")
                saved_state = yaml.safe_load(content)
            if saved_state:
                for key in SELECTED_KEYS:
                    if key in saved_state:
                        st.session_state[key] = saved_state[key]
        except Exception as e:
            st.error(f"Error loading session state: {e}")
            # Optionally delete the corrupt file to allow regeneration
            os.remove(default_yaml_path)
    else:
        save_session_state(default_yaml_path)
        
def save_session_state(default_yaml_path="cfgs/gui/session_state/default.yaml"):
    """
    Save only the selected session state keys to a YAML file.
    """
    # Ensure the directory exists; if not, create it
    directory = os.path.dirname(default_yaml_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True, mode=0o777)

    try:
        # Extract the state for the selected keys that exist in session state
        state_to_save = {key: st.session_state[key] for key in SELECTED_KEYS if key in st.session_state}
        
        # Dump the state to the YAML file
        with open(default_yaml_path, "w") as f:
            yaml.dump(state_to_save, f)
    except Exception as e:
        st.error(f"Error saving session state: {e}")

## Path & File Management

def infer_image_pattern(images_dir, extensions=(".jpg", ".png")):
    """
    Infers a naming pattern for images in the given directory.
    Returns a tuple (pattern, start_index, end_index) if successful,
    but only if all files in the directory follow the same numeric pattern.
    Otherwise, a warning is set in st.session_state.naming_pattern_warning and None is returned.

    For example, if files are like frame001.jpg, frame002.jpg, etc.,
    it returns ("frame{:03d}.jpg", 1, N). If no valid pattern is found,
    a warning is issued and None is returned.
    """
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    if not files:
        st.session_state.no_images_warning = "No images found in directory."
                   
        return None
    else:
        st.session_state.no_images_warning = None

    patterns = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        # Match filenames that contain a numeric sequence.
        match = re.search(r"^(.*?)(\d+)(\.[^.]+)$", filename)
        if match:
            prefix, number_str, ext = match.groups()
            num_length = len(number_str)
            pattern = f"{prefix}{{:0{num_length}d}}{ext}"
            num = int(number_str)
            patterns.setdefault(pattern, []).append(num)
        else:
            return None
    
    if not patterns:
        return None

    # Choose the pattern with the most files.
    best_pattern, numbers = max(patterns.items(), key=lambda item: len(item[1]))
    
    # Check if the best pattern covers ALL files.
    if len(numbers) != len(files):
        return None

    numbers.sort()

    # Check that the numbers form a consecutive sequence.
    for i in range(len(numbers) - 1):
        if numbers[i] + 1 != numbers[i+1]:
            return None

    start_index = numbers[0]
    end_index = numbers[-1]

    # Clear any previous naming pattern warning.
    st.session_state.naming_pattern_warning = None

    return best_pattern, start_index, end_index

def upload_to_dir(save_dir):
    """
    Allows the user to upload a single file or a ZIP archive (representing a directory) 
    and saves/extracts it to the specified directory. Resets the upload field after processing.
    
    Args:
        save_dir (str): The directory where the uploaded file(s) will be saved.
    """
    
    # Define a unique key for the uploader
    uploader_key = "file_uploader"

    # File uploader widget
    uploaded_file = st.file_uploader(
        "Upload a file or a ZIP archive containing a directory",
        type=["txt", "csv", "jpg", "png", "pdf", "py", "yaml", "zip", "mp4", "mov"],
        key=uploader_key
    )
    
    if uploaded_file is not None:
        # Ensure the save directory exists.
        os.makedirs(save_dir, exist_ok=True, mode=0o777)
        
        # If the file is a ZIP archive, extract it.
        if uploaded_file.name.endswith(".zip"):
            temp_zip_path = os.path.join(save_dir, uploaded_file.name)
            with open(temp_zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                os.chmod(temp_zip_path, 0o777)
            except:
                pass
            
            # Extract all contents of the ZIP file to the save directory.
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(save_dir)
            
            os.remove(temp_zip_path)  # Remove ZIP file after extraction
            st.success(f"ZIP file extracted to: {save_dir}")
        else:
            # Save a single uploaded file.
            file_path = os.path.join(save_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                os.chmod(file_path, 0o777)
            except:
                pass
            
        # Reset the file uploader state
        if uploader_key in st.session_state:
            del st.session_state[uploader_key]  # Properly reset uploader key
    
        st.rerun()  # Refresh the UI

def path_navigator(
    key,
    radio_button_prefix="",
    button_and_selectbox_display_size=[0.05, 0.95],
    must_exist: bool = True,
):
    """
    A file/directory navigator that can operate in two modes:
    1) Enter Path as Text: you can type any path. If must_exist is True,
       non‑existent paths trigger Create/Go Up prompts; if False, they
       are accepted immediately.
    2) File Explorer: Browse with a '..' button and selectbox.

    Args:
        key (str): session_state.paths key to read/write.
        radio_button_prefix (str): prefix for widget keys.
        button_and_selectbox_display_size (list): column widths for explorer.
        must_exist (bool): whether the chosen path must already exist.
    """

    # Retrieve or set initial path in session state
    current_path = st.session_state.paths.get(key, "/")
    current_path = os.path.normpath(current_path)

    # Mode switch
    save_path_option = st.radio(
        "Choose save path option:",
        ["Enter Path as Text", "File Explorer"],
        key=f"{radio_button_prefix}_{key}_radio",
        label_visibility="collapsed",
    )

    if save_path_option == "Enter Path as Text":
        # -- CUSTOM PATH MODE --
        custom_path = st.text_input(
            "Enter custom save path:",
            value=current_path,
            key=f"{radio_button_prefix}_{key}_custom_path_input",
            label_visibility="collapsed",
        ).strip()
        st.write(f"**Current {' '.join(w.capitalize() for w in key.split('_'))}:** {current_path}")

        if custom_path:
            custom_path = os.path.normpath(custom_path)
            # If must_exist is False, accept immediately
            if not must_exist or os.path.exists(custom_path):
                st.session_state.paths[key] = custom_path
                return custom_path

            # Otherwise, fall back to original create / go‑up logic
            st.warning(f"Path '{custom_path}' does not exist. Choose an option below:")
            create_col, up_col = st.columns(2)
            with create_col:
                if st.button("Create this path", key=f"{radio_button_prefix}_{key}_create_custom"):
                    new_name = st.text_input(
                        "Optionally enter a different name for the new path:",
                        value=custom_path,
                        key=f"{radio_button_prefix}_{key}_new_path_name"
                    ).strip()
                    if new_name:
                        try:
                            root, ext = os.path.splitext(new_name)
                            if ext:
                                parent = os.path.dirname(new_name)
                                os.makedirs(parent, exist_ok=True, mode=0o777)
                                open(new_name, "a").close()
                            else:
                                os.makedirs(new_name, exist_ok=True, mode=0o777)
                            st.session_state.paths[key] = new_name
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to create: {e}")
            with up_col:
                if st.button("Go up until path exists", key=f"{radio_button_prefix}_{key}_go_up_custom"):
                    temp = custom_path
                    while not os.path.exists(temp) and temp not in ("/", ""):
                        temp = os.path.dirname(temp)
                    if os.path.exists(temp):
                        st.session_state.paths[key] = temp
                        st.rerun()
                    else:
                        st.error("No valid parent directory found.")
            return current_path  # keep the old path until created

        return current_path

    # -- DEFAULT NAVIGATION MODE --
    if os.path.isfile(current_path):
        directory_to_list = os.path.dirname(current_path)
    else:
        directory_to_list = current_path

    col1, col2 = st.columns(button_and_selectbox_display_size, gap="small")
    with col1:
        if st.button("..", key=f"go_up_button_{radio_button_prefix}_{key}"):
            parent = os.path.dirname(current_path) if os.path.isdir(current_path) else os.path.dirname(os.path.dirname(current_path))
            st.session_state.paths[key] = os.path.normpath(parent)
            st.rerun()

    if not os.path.exists(directory_to_list):
        st.warning(f"Path '{directory_to_list}' does not exist.")
        return current_path

    try:
        entries = os.listdir(directory_to_list)
    except Exception as e:
        st.error(f"Error reading directory: {e}")
        return current_path

    dirs = [(e, os.path.join(directory_to_list, e)) for e in entries if os.path.isdir(os.path.join(directory_to_list, e))]
    files = [(e, os.path.join(directory_to_list, e)) for e in entries if os.path.isfile(os.path.join(directory_to_list, e))]

    dirs.sort(key=lambda x: x[0].lower())
    files.sort(key=lambda x: x[0].lower())

    options_list = [directory_to_list]
    options_mapping = {directory_to_list: None}
    indent = "└"

    for name, full in dirs:
        label = f"{indent} 📁 {name}"
        options_list.append(label)
        options_mapping[label] = full

    for name, full in files:
        ext = os.path.splitext(name)[1].lower()
        emoji = "🖼️" if ext in ['.jpg','.png','.gif'] else "🐍" if ext=='.py' else "📄"
        label = f"{indent} {emoji} {name}"
        options_list.append(label)
        options_mapping[label] = full

    default_index = next((i for i,l in enumerate(options_list)
                          if options_mapping.get(l)==os.path.normpath(current_path)), 0)

    widget_key = f"navigator_select_{radio_button_prefix}_{key}"
    def on_change():
        sel = st.session_state[widget_key]
        path = options_mapping.get(sel)
        if path is not None:
            st.session_state.paths[key] = path

    with col2:
        st.selectbox(
            "Select a subdirectory or file:",
            options_list,
            index=default_index,
            key=widget_key,
            on_change=on_change,
            label_visibility="collapsed",
        )

    st.write(f"**Current {' '.join(w.capitalize() for w in key.split('_'))}:** {st.session_state.paths[key]}")
    return st.session_state.paths[key]

def safe_rename_images(images_dir):
    """
    Safely renames all image files in images_dir to a common pattern (image_0000, image_0001, …)
    without conflicts by using a two-phase renaming process. It also renames corresponding label files,
    ensuring that each image and its label remain paired throughout the process.
    
    The function assumes that:
      - Image files can be .jpg, .jpeg, or .png and will be forced to have a .jpg extension.
      - Label files are stored in a parallel directory obtained by replacing "images" with "labels"
        and have a .txt extension.
    
    If multiple images share the same original label file, the first occurrence moves the file and
    subsequent images get a copy of that label.
    """
    # Define the image extensions to process.
    extensions = [".jpg", ".jpeg", ".png"]
    
    # Collect all image paths.
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    image_paths.sort()

    # Determine the labels directory (assumes "images" is part of the path).
    labels_dir = images_dir.replace("images", "labels")
    os.makedirs(labels_dir, exist_ok=True, mode=0o777)
    
    # Calculate total steps (phase 1 and phase 2)
    total_steps = len(image_paths) * 2
    current_step = 0

    # Create Streamlit placeholders for the progress bar and status text.
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # Phase 1: Rename images and corresponding labels to temporary names.
    temp_image_label_pairs = []  # List of tuples: (temp_image_path, temp_label_path, original_image_name)
    label_temp_mapping = {}

    for i, orig_image_path in enumerate(image_paths):
        orig_dir, orig_image_name = os.path.split(orig_image_path)
        
        # Build the temporary image name (force .jpg extension).
        temp_image_name = "__tmp__image_{:04d}.jpg".format(i)
        temp_image_path = os.path.join(orig_dir, temp_image_name)
        os.rename(orig_image_path, temp_image_path)
        
        # Determine the corresponding original label file.
        orig_label_name = os.path.splitext(orig_image_name)[0] + ".txt"
        orig_label_path = os.path.join(labels_dir, orig_label_name)
        
        # Build the temporary label name.
        temp_label_name = "__tmp__image_{:04d}.txt".format(i)
        temp_label_path = os.path.join(labels_dir, temp_label_name)
        
        # If this label file was already processed for a previous image, copy it.
        if orig_label_name in label_temp_mapping:
            shutil.copy2(label_temp_mapping[orig_label_name], temp_label_path)
        elif os.path.exists(orig_label_path):
            os.rename(orig_label_path, temp_label_path)
            label_temp_mapping[orig_label_name] = temp_label_path
        else:
            # If no corresponding label file exists, create an empty one.
            with open(temp_label_path, "w") as f:
                pass
        
        temp_image_label_pairs.append((temp_image_path, temp_label_path, orig_image_name))
        
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        progress_text.text(f"Phase 1: Renaming images {i+1} of {len(image_paths)}")
    
    # Phase 2: Rename temporary files to final names.
    new_pattern_image = "image_{:04d}.jpg"
    new_pattern_label = "image_{:04d}.txt"
    
    for i, (temp_image_path, temp_label_path, orig_image_name) in enumerate(temp_image_label_pairs):
        # Final image name and path.
        final_image_name = new_pattern_image.format(i)
        final_image_path = os.path.join(os.path.dirname(temp_image_path), final_image_name)
        os.rename(temp_image_path, final_image_path)
        
        # Final label name and path.
        final_label_name = new_pattern_label.format(i)
        final_label_path = os.path.join(os.path.dirname(temp_label_path), final_label_name)
        os.rename(temp_label_path, final_label_path)
        
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        progress_text.text(f"Phase 2: Renaming files {i+1} of {len(temp_image_label_pairs)}")
    
    progress_text.text("Renaming complete!")
    return new_pattern_image, len(image_paths)

def load_subset_frames(csv_path):
    """
    Reads unique frame indexes from a CSV (one index per line).
    Returns them as a sorted list of unique integers.
    """
    frames = []
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    frames.append(int(line))
        try:
            with open(csv_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.isdigit():
                        frames.append(int(line))
        except:
            st.warning("CSV Path does not exist.")

    return sorted(set(frames))

def save_subset_csv(csv_path, frames):
    """
    Writes frame indexes to CSV, one per line, ensuring they are unique and sorted.
    """
    frames_sorted = sorted(set(frames))
    with open(csv_path, "w") as f:
        for frame in frames_sorted:
            f.write(f"{frame}\n")
    
    try:
        os.chmod(csv_path, 0o777)
    except:
        pass
    
## TMUX Terminal Commands

def update_tmux_terminal(session_key):
    try:
        capture_cmd = f"tmux capture-pane -pt {session_key}:0.0"
        output = subprocess.check_output(capture_cmd, shell=True)
        # Replace carriage returns with newlines.
        decoded_output = output.decode("utf-8").replace('\r', '\n')
        return decoded_output
    except subprocess.CalledProcessError as e:
        st.warning("No tmux session")
        return None

def kill_tmux_session(session_key):
    """
    Kills the tmux session identified by session_key.

    Args:
        session_key (str): The tmux session name.
    """
    kill_cmd = f"tmux kill-session -t {session_key}"
    subprocess.call(kill_cmd, shell=True)
    st.success(f"tmux session '{session_key}' has been killed.")

def run_in_tmux(session_key, script_path, venv_path=None, args="", script_type="python"):
    """
    Opens a new tmux session with the given session_key, optionally activates the virtual environment from venv_path,
    runs the specified script (Python or Bash) with additional arguments, captures its terminal output,
    displays it in Streamlit, and then kills the session.

    If a tmux session with the same session_key already exists, it will be killed first.

    Args:
        session_key (str): Unique key used as the tmux session name.
        script_path (str): Path to the script file to be executed.
        venv_path (str or None): Path to the virtual environment directory. If None, no virtual environment is activated.
        args (str or dict): Additional command-line arguments to be appended to the command.
                            If a dictionary is provided, it will be converted to a string of command-line arguments.
        script_type (str): Type of script to run. Either "python" or "bash".
    
    Returns:
        str or None: The decoded terminal output if successful; otherwise, None.
    """
    
    # If a session with the given key already exists, kill it.
    try:
        subprocess.check_call(f"tmux kill-session -t {session_key}", shell=True)
    except subprocess.CalledProcessError:
        # If there's no such session, ignore the error.
        pass

    # Check if the script file exists
    if not os.path.exists(script_path):
        st.error(f"Script file not found: {script_path}")
        return None

    # Convert args to string if it's a dictionary
    if isinstance(args, dict):
        args = " ".join(f"--{key} {value}" for key, value in args.items())

    if venv_path is not None:
        activate_script = os.path.join(venv_path, "bin", "activate")
        if not os.path.exists(activate_script):
            st.error(f"Virtual environment activation script not found: {activate_script}")
            return None
        # Build the command to activate the venv
        activation_cmd = f"source {activate_script} && "
    else:
        activation_cmd = ""
    
    # Build the inner command based on script type.
    if script_type == "python":
        cmd = f"{activation_cmd}python {script_path} {args}; exec bash"
    elif script_type == "bash":
        cmd = f"{activation_cmd}bash {script_path} {args}; exec bash"
    else:
        st.error("Invalid script type specified. Use 'python' or 'bash'.")
        return None

    # Build the complete tmux command using bash -c to handle quoting correctly.
    tmux_cmd = (
        f'tmux new-session -d -s {session_key} '
        f'"bash -c \'umask 000 && {cmd}; exec bash\'"'
    )

    try:
        # Create the tmux session and run the command
        subprocess.check_call(tmux_cmd, shell=True)

        # Wait briefly to allow the command to execute and produce output.
        time.sleep(2)

        # Capture the output from the first pane (pane 0 of window 0)
        capture_cmd = f"tmux capture-pane -pt {session_key}:0.0"
        output = subprocess.check_output(capture_cmd, shell=True)
        decoded_output = output.decode("utf-8")

        return decoded_output
    except subprocess.CalledProcessError as e:
        st.error(f"Error running command in tmux: {e}")
        return None
    
def wait_for_conversion(session_key, check_interval=3):
    """
    Poll the tmux session output until the terminal prompt is detected,
    indicating the conversion is complete.
    
    We assume that a shell prompt appears in the output that ends with a colon (":")
    followed by any characters and ends with a "$" symbol, like:
    "naddeok5@lambda03:~/AutoLabelEngine$"
    """
    # Pattern explanation:
    # ^           : start of the line
    # .+@.+       : at least one character, an '@', then at least one character (username@hostname)
    # :.*         : a colon followed by any characters (the current directory)
    # \$\s*$     : a literal "$" followed by optional whitespace till the end of line
    prompt_pattern = r"^.+@.+:.*\$\s*$"

    while True:
        output = update_tmux_terminal(session_key)
        # Check each line in the output to see if one matches the prompt pattern.
        for line in output.splitlines():
            if re.search(prompt_pattern, line.strip()):
                return output
        time.sleep(check_interval)

## GPU Tools

def check_gpu_status(button_key):
    # Add the description for the user
    st.write("Click the 'Check GPU Status' button to check the current status the gpus on the server. ")

    # Check if the gpustat command is available
    if st.button("Check GPU Status", key=button_key):
        try:
            # Run the gpustat command and capture its output
            output = subprocess.check_output(["gpustat"]).decode("utf-8")
            st.write("NOTE: This output is not live. Please click again for an updated status.")
            return output
            
        except Exception as e:
            st.error(f"Failed to run gpustat: {e}")
    
def display_terminal_output(output):
    # Ensure terminal_text exists in session state
    if "terminal_text" not in st.session_state:
        st.session_state.terminal_text = ""
    # Process the output to replace carriage returns with newlines
    processed_output = output.replace('\r', '\n')
    # Update the session state with the processed output
    st.session_state.terminal_text = processed_output
    # Display the accumulated output as bash code
    st.code(st.session_state.terminal_text, language="bash")

## Editors (YAML & Python)

def yaml_editor(yaml_key):
    """
    Display a YAML file in two columns: one for editing (with auto-save) and one for the currently saved YAML.
    Uses st.session_state.paths[yaml_key] as the YAML file path and st.session_state.yamls as a dict to store last saved content.
    
    Also allows copying the YAML to a new file by entering a new save path. The text input auto-fills with the current
    file path modified to include "_copy" before the extension.
    
    Args:
        yaml_key (str): Unique key to index this YAML file in st.session_state.paths and st.session_state.yamls.
    """
    # Retrieve the file path from session state
    if "paths" not in st.session_state or yaml_key not in st.session_state.paths:
        st.error(f"Path for key '{yaml_key}' not found in st.session_state.paths")
        return
    file_path = st.session_state.paths[yaml_key]
   

    # Check if the file exists
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return

    # Read the YAML file content
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # Initialize the session state dictionary for YAML contents if not already present
    if "yamls" not in st.session_state:
        st.session_state.yamls = {}

    # Initialize the last saved content for this YAML file if not set
    st.session_state.yamls[yaml_key] = file_content

    content_hash = hashlib.md5(file_content.encode('utf-8')).hexdigest()
    ace_key = f"edited_content_{yaml_key}_{content_hash}"
    
    st.markdown("Below is the YAML content. Edit and press the apply button at the bottom to save the changes.")

    lines = file_content.splitlines()
    line_count = len(lines) if len(lines) > 0 else 1
    calculated_height = max(100, line_count * 20 + 25)

     # Auto-save if the edited content has changed compared to the last saved version
    
    edited_content = st_ace(
        value=file_content,
        language="yaml",
        theme="",
        height=calculated_height,
        font_size=17, 
        key=ace_key,
    )

    if edited_content != st.session_state.yamls[yaml_key]:
        try:
            # Validate the YAML content
            parsed_yaml = yaml.safe_load(edited_content)

        except yaml.YAMLError as e:
            st.error(f"Invalid YAML format: {e}")
        else:
            try:
                # Save the validated YAML back to the file
                with open(file_path, 'w') as file:
                    yaml.dump(parsed_yaml, file, default_flow_style=False, sort_keys=False)
                # Update the stored content for this YAML file
                st.session_state.yamls[yaml_key] = edited_content
                update_unverified_data_path()
                st.rerun()  # Re-run to update the displayed current YAML content
            except Exception as e:
                st.error(f"Error saving file: {e}")
  
    # Compute default copy path by inserting "_copy" before the extension.
    base, ext = os.path.splitext(file_path)
    default_copy_path = base + "_copy" + ext
    new_save_path = st.text_input("Enter new file path", key=f"copy_path_{yaml_key}", value=default_copy_path)
    if st.button("Copy YAML to new file", key=f"copy_button_{yaml_key}"):
        if new_save_path:
            st.session_state.paths[yaml_key] = new_save_path
            try:
                # Validate the YAML content again
                parsed_yaml = yaml.safe_load(edited_content)
            except yaml.YAMLError as e:
                st.error(f"Invalid YAML format, cannot copy: {e}")
            else:
                try:
                    with open(new_save_path, 'w') as new_file:
                        yaml.dump(parsed_yaml, new_file, default_flow_style=False, sort_keys=False)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error copying file: {e}")
        else:
            st.error("Please enter a valid new file path")

def python_code_editor(code_key):
    """
    Display a Python file in an editor with auto-save and copy functionality.
    Uses st.session_state.paths[code_key] as the Python file path and st.session_state.python_codes
    as a dict to store the last saved content.

    Also allows copying the Python code to a new file by entering a new save path.
    The text input auto-fills with the current file path modified to include "_copy" before the extension.

    Args:
        code_key (str): Unique key to index this Python file in st.session_state.paths and st.session_state.python_codes.
    """
    
    # Retrieve the file path from session state
    if "paths" not in st.session_state or code_key not in st.session_state.paths:
        st.error(f"Path for key '{code_key}' not found in st.session_state.paths")
        return
    file_path = st.session_state.paths[code_key]

    # Check if the file exists
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return

    # Read the Python file content
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # Initialize the session state dictionary for Python code if not already present
    if "python_codes" not in st.session_state:
        st.session_state.python_codes = {}

    # Initialize the last saved content for this Python file if not set
    if code_key not in st.session_state.python_codes:
        st.session_state.python_codes[code_key] = file_content

    ace_key = f"edited_content_{code_key}"
    
    st.markdown("Below is the Python script. Edit and press the apply button at the bottom to save the changes.")

    lines = file_content.splitlines()
    line_count = len(lines) if len(lines) > 0 else 1
    calculated_height = max(300, line_count * 19)

    edited_content = st_ace(
        value=file_content,
        language="python",
        theme="",
        height=calculated_height,
        font_size=17, 
        key=ace_key
    )

    # Auto-save if the edited content has changed compared to the last saved version
    if edited_content != st.session_state.python_codes[code_key]:
        try:
            # Validate the Python code by attempting to compile it
            compile(edited_content, file_path, 'exec')
        except SyntaxError as e:
            st.error(f"Invalid Python syntax: {e}")
        else:
            try:
                # Save the validated Python code back to the file
                with open(file_path, 'w') as file:
                    file.write(edited_content)
                # Update the stored content for this Python file
                st.session_state.python_codes[code_key] = edited_content
                st.rerun()  # Re-run to update the displayed current content
            except Exception as e:
                st.error(f"Error saving file: {e}")
  
    # Compute default copy path by inserting "_copy" before the extension.
    base, ext = os.path.splitext(file_path)
    default_copy_path = base + "_copy" + ext
    new_save_path = st.text_input("Enter new file path", key=f"copy_path_{code_key}", value=default_copy_path)
    
    if st.button("Copy Python code to new file", key=f"copy_button_{code_key}"):
        if new_save_path:
            # Optionally update the session state path to the new copy
            st.session_state.paths[code_key] = new_save_path
            try:
                # Validate the Python code again before copying
                compile(edited_content, new_save_path, 'exec')
            except SyntaxError as e:
                st.error(f"Invalid Python syntax, cannot copy: {e}")
            else:
                try:
                    with open(new_save_path, 'w') as new_file:
                        new_file.write(edited_content)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error copying file: {e}")
        else:
            st.error("Please enter a valid new file path")

## Bounding Box / IoU & Label Comparison

def iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in [x, y, width, height] format, where (x, y) is the top-left corner.
    """
    # Unpack the boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the (x, y) coordinates of the bottom-right corner of each box
    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2

    # Determine the coordinates of the intersection rectangle
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    # Compute the width and height of the intersection rectangle
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Compute the area of each bounding box
    area_box1 = w1 * h1
    area_box2 = w2 * h2

    # Compute the union area
    union_area = area_box1 + area_box2 - inter_area

    if union_area == 0:
        return 0.0

    # Compute the IoU
    return inter_area / union_area

def are_bboxes_equal(current_bboxes, current_labels, bboxes_xyxy, labels_xyxy, threshold=0.9):
    """
    Compare two sets of bounding boxes with their associated labels.
    
    Two boxes match if:
      - IoU(box1, box2) >= threshold, and
      - Their associated labels are equal.
    
    The function returns True if every bbox in current_bboxes (with its label)
    can be matched with a unique bbox in bboxes_xyxy (with its label) and vice versa.
    """
    # Early exit if the number of boxes or labels do not match
    if len(current_bboxes) != len(bboxes_xyxy) or len(current_labels) != len(labels_xyxy):
        return False

    # Create a list of tuples for the second set for easy removal when matched
    unmatched = list(zip(bboxes_xyxy, labels_xyxy))
    
    # For each box in current_bboxes, try to find a matching box in the second set
    for box, label in zip(current_bboxes, current_labels):
        found_match = False
        for i, (box2, label2) in enumerate(unmatched):
            if iou(box, box2) >= threshold and label == label2:
                found_match = True
                del unmatched[i]  # Remove the matched box so it is not reused
                break
        if not found_match:
            return False  # A box in current_bboxes has no matching counterpart

    # If there are any leftover boxes in unmatched, they are extra in the second set
    if unmatched:
        return False
    return True

## Unverified Data & Frame Management

def update_unverified_data_path():
    data_yaml_path = st.session_state.paths.get("unverified_names_yaml_path")
    if not data_yaml_path or not os.path.exists(data_yaml_path):
        data_cfg = {"names": {0: "default_label"}}
    else:
        if Path(data_yaml_path).suffix.lower() not in ['.yaml', '.yml']:
            return
        with open(data_yaml_path, 'r') as file:
            data_cfg = yaml.safe_load(file)

    images_dir = st.session_state.paths["unverified_images_path"]
    # Instead of building a full list, infer the naming pattern.
    pattern_info = infer_image_pattern(images_dir)
    if st.session_state.automatic_generate_list:
        st.session_state.naming_pattern_warning = None

        images = []
        for ext in [".jpg", ".png"]:
            images.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
        images.sort()
        st.session_state.image_list = images
        st.session_state.max_images = len(images)
        st.session_state.start_index = 0
        st.session_state.image_pattern = None
        st.session_state.naming_pattern_warning = None
        # Set the flag so that next runs do not redo this process.
        st.session_state.image_list_stored = True

    elif pattern_info is None:
        # Set a flag instead of immediately showing a warning.
        st.session_state.naming_pattern_warning = (
            "Could not infer an image naming pattern or sequential numeric sequence found."
        )
        st.session_state.max_images = 0
        st.session_state.start_index = 0
        st.session_state.image_pattern = None
    else:
        st.session_state.naming_pattern_warning = None
        image_pattern, start_index, end_index = pattern_info
        st.session_state.image_pattern = image_pattern
        st.session_state.start_index = start_index
        st.session_state.max_images = end_index - start_index + 1

    label_list = list(data_cfg.get("names", {0: "default_label"}).values())
    st.session_state.data_cfg = data_cfg
    st.session_state.label_list = label_list
    st.session_state.images_dir = images_dir
    st.session_state.frame_index = st.session_state.start_index

    # If the user has chosen to use a subset (stored in a CSV), override max_images
    # and any existing frame index logic with the subset frames.
    if st.session_state.use_subset:
        subset_csv = st.session_state.paths["unverified_subset_csv_path"]
        st.session_state.subset_frames = load_subset_frames(subset_csv)
        if len(st.session_state.subset_frames) == 0:
            # If the CSV is empty, there are effectively no images
            st.session_state.max_images = 0
            st.session_state.frame_index = 0
        else:
            # We now rely on the length of subset_frames instead of the entire dataset
            st.session_state.max_images = len(st.session_state.subset_frames)
            st.session_state.frame_index = 0

def update_unverified_frame():
    # Always clamp once by the full dataset:
    if st.session_state.frame_index < 0:
        st.session_state.frame_index = st.session_state.max_images - 1
    if st.session_state.frame_index >= st.session_state.max_images:
        st.session_state.frame_index = 0

    # Now handle subsets
    if st.session_state.use_subset:
        # Load/reload subset if needed
        st.session_state.subset_frames = load_subset_frames(
            st.session_state.paths["unverified_subset_csv_path"]
        )
        if not st.session_state.subset_frames:
            st.session_state.frame_index = 0
            st.session_state.max_images = 0
            st.error("Subset CSV is empty. No frames to load.")
            return

        # Re-clamp to the subset length
        if st.session_state.frame_index >= len(st.session_state.subset_frames):
            st.session_state.frame_index = len(st.session_state.subset_frames) - 1
        if st.session_state.frame_index < 0:
            st.session_state.frame_index = 0

        # Actual frame to load from the full dataset:
        actual_frame_index = st.session_state.subset_frames[st.session_state.frame_index]
    else:
        # Use the full dataset approach
        actual_frame_index = st.session_state.frame_index

    # NEW: Save the actual (original) frame index in session state so it can be displayed later.
    st.session_state.actual_frame_index = actual_frame_index

    # Get Image Path
    images_dir = st.session_state.images_dir
    if st.session_state.image_pattern:
        image_path = os.path.join(images_dir, st.session_state.image_pattern.format(actual_frame_index))
    elif "image_list" in st.session_state and st.session_state.image_list:
        try:
            image_path = st.session_state.image_list[actual_frame_index]
        except IndexError:
            st.error("Frame index out of range for image list.")
            return
    else:
        st.error("No image naming pattern or image list set.")
        return

    # Open Image
    if os.path.exists(image_path):
        image = Image.open(image_path)
        image_width, image_height = image.size
    else:
        update_unverified_data_path()
        st.rerun()

    # Get Labels
    labels_dir = images_dir.replace("images", "labels")
    label_path = (
        image_path
        .replace("images", "labels")
        .replace("jpg", "txt")
        .replace("png", "txt")
    )

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir, mode=0o777)

    bboxes_xyxy = []
    labels = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])
                x_center_abs = x_center * image_width
                y_center_abs = y_center * image_height
                w_abs = w * image_width
                h_abs = h * image_height
                bbox_xyxy = [
                    x_center_abs - w_abs / 2,
                    y_center_abs - h_abs / 2,
                    w_abs,
                    h_abs,
                ]
                bboxes_xyxy.append(bbox_xyxy)
                labels.append(cls)
    else:
        with open(label_path, "w") as f:
            f.write("")
        
        try:
            os.chmod(label_path, 0o777)
        except:
            pass
        

    bbox_ids = ["bbox-" + str(i) for i in range(len(bboxes_xyxy))]

    # Store Internally
    st.session_state.image_path = image_path
    st.session_state.image = image
    st.session_state.labels_dir = labels_dir
    st.session_state.label_path = label_path
    st.session_state.image_width = image_width
    st.session_state.image_height = image_height
    st.session_state.bboxes_xyxy = bboxes_xyxy
    st.session_state.labels = labels
    st.session_state.bbox_ids = bbox_ids

    # Add unknown labels to display
    known_labels = st.session_state.label_list
    display_labels = known_labels.copy()  # Start with known labels
    unknown_label_map = {}

    updated_labels = []
    for label in labels:
        if label < len(known_labels):
            updated_labels.append(label)
        else:
            if label not in unknown_label_map:
                unknown_label_map[label] = len(display_labels)
                display_labels.append(f"Unknown: {label}")
            updated_labels.append(unknown_label_map[label])

    st.session_state["detection_config"] = {
        "image_path": st.session_state.image_path,
        "image_height": int(st.session_state.unverified_image_scale * st.session_state.image_height),
        "image_width": int(st.session_state.unverified_image_scale * st.session_state.image_width),
        "label_list": display_labels,
        "bboxes": bboxes_xyxy,
        "labels": updated_labels,
        "bbox_show_label": True,
        "info_dict": [],
        "meta_data": [],
        "ui_size": "small",
        "ui_left_size": None,
        "ui_bottom_size": None,
        "ui_right_size": None,
        "component_alignment": "left",
        "ui_position": "left",
        "line_width": 1.0,
        "read_only": False,
        "class_select_type": "radio",
        "class_select_position": None,
        "item_editor": False,
        "item_editor_position": "right",
        "edit_description": False,
        "edit_meta": False,
        "item_selector": True,
        "item_selector_position": "right",
        "bbox_format": "XYWH",
        "bbox_show_info": True,
        "key": st.session_state.detector_key,
    }

def update_labels_from_detection():
    if "out" not in st.session_state or "skip_label_update" not in st.session_state:
        st.session_state["skip_label_update"] = True
        return None

    elif st.session_state["skip_label_update"]:
        st.session_state["skip_label_update"] = False
        return None
    
    else:
        current_bboxes = []
        current_labels = []
        for bbox in st.session_state.out['bbox']:
            current_bboxes.append(bbox['bboxes'])
            current_labels.append(bbox['labels'])

        label_path = st.session_state.label_path
        image_width = st.session_state.image_width
        image_height = st.session_state.image_height
        labels = st.session_state.labels
        bboxes_xyxy = st.session_state.bboxes_xyxy

        # If the user-changed bboxes differ from what's currently stored, write out to disk
        if (not are_bboxes_equal(current_bboxes, current_labels, bboxes_xyxy, labels, threshold=0.9)) and not st.session_state["skip_label_update"]:
            st.session_state["skip_label_update"] = False
            # Write normalized YOLO-format labels to file
            with open(label_path, "w") as f:
                for label, bbox in zip(current_labels, current_bboxes):
                    x_min, y_min, width, height = bbox
                    # Convert the absolute coordinates back to normalized YOLO format:
                    x_center_norm = (x_min + width / 2) / image_width
                    y_center_norm = (y_min + height / 2) / image_height
                    width_norm = width / image_width
                    height_norm = height / image_height
                    f.write(f"{label} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
            
            # Re-run to refresh the newly saved label data
            st.rerun()
        else:
            if st.session_state["skip_label_update"]:
                st.session_state["skip_label_update"] = False

def next_callback():
    st.session_state.prev_out = None
    st.session_state.detection_modified = False
    st.session_state.frame_index += 1
    st.session_state["skip_label_update"] = True

def prev_callback():
    st.session_state.prev_out = None
    st.session_state.detection_modified = False
    st.session_state.frame_index -= 1
    st.session_state["skip_label_update"] = True

def frame_slider_frame_by_frame_callback():
    # Get the new value from the slider (using the key "slider_det")
    new_frame_index = st.session_state.slider_det
    # Compare with the current frame_index stored in session_state
    if new_frame_index != st.session_state.frame_index:
        st.session_state.frame_index = new_frame_index
        st.session_state["skip_label_update"] = True

def jump_frame_frame_by_frame_callback():
    # Retrieve the new value from the number input via its key "jump_page"
    new_frame_index = st.session_state.jump_page
    # Compare it with the current frame_index in session_state
    if st.session_state.frame_index != new_frame_index:
        st.session_state.frame_index = new_frame_index
        st.session_state["skip_label_update"] = True

def jump_frame_object_by_object_callback():
    # Get the desired frame number from the number input.
    raw = st.session_state.jump_to_frame_input
    subset_active = st.session_state.use_subset and bool(st.session_state.subset_frames)
    max_idx = (len(st.session_state.subset_frames) - 1) if subset_active else (st.session_state.max_images - 1)

    jump_frame = max(0, min(raw, max_idx))

    # Build the image list either using a naming pattern or a stored list.
    if st.session_state.get("image_pattern") is not None:
        image_list = [
            os.path.join(
                st.session_state.paths["unverified_images_path"],
                st.session_state.image_pattern.format(i)
            )
            for i in range(st.session_state.start_index, st.session_state.start_index + st.session_state.max_images)
        ]
    else:
        image_list = st.session_state.get("image_list", [])
    
    # Initialize the global index and a flag to check if the frame has any objects.
    global_index = 0
    found = False

    # Iterate over images to compute the global index.
    for idx, image_path in enumerate(image_list):
        try:
            label_path_temp = image_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
            with open(label_path_temp, "r") as f:
                # Count only non-empty lines (indicating objects)
                lines = [line for line in f if line.strip() != ""]
        except Exception:
            lines = []
        
        if idx < jump_frame:
            # Add the number of objects from frames before the target frame.
            global_index += len(lines)
        elif idx == jump_frame:
            if len(lines) > 0:
                found = True
            break  # Stop processing once the target frame is reached.

    # If the frame is valid and has objects, update session state and set a flag for rerun.
    if found:
        st.session_state.frame_index = jump_frame
        st.session_state.global_object_index = global_index
        st.session_state.object_by_object_jump_valid = True
        st.session_state.object_by_object_jump_warning = None  # Set flag to trigger rerun outside callback.
    else:
        
        st.session_state.object_by_object_jump_valid = False
        st.session_state.object_by_object_jump_warning = "Frame is either not valid or has no objects."

def copy_labels_from_slide(source_index):
    if st.session_state.image_pattern:
        src_image_path = os.path.join(st.session_state.images_dir, st.session_state.image_pattern.format(source_index))
    elif "image_list" in st.session_state and st.session_state.image_list:
        try:
            src_image_path = st.session_state.image_list[source_index]
        except IndexError:
            st.warning(f"Source index {source_index} is out of range for image list.")
            return
    else:
        st.error("No image naming pattern or image list set.")
        return

    src_label_path = src_image_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    if os.path.exists(src_label_path):
        with open(src_label_path, "r") as f:
            src_labels = f.read()
    else:
        st.warning(f"No labels found in slide {source_index}.")
        src_labels = ""
    
    if st.session_state.image_pattern:
        curr_image_path = os.path.join(st.session_state.images_dir, st.session_state.image_pattern.format(st.session_state.frame_index))
    elif "image_list" in st.session_state and st.session_state.image_list:
        try:
            curr_image_path = st.session_state.image_list[st.session_state.frame_index]
        except IndexError:
            st.error("Current frame index is out of range for image list.")
            return
    else:
        st.error("No image naming pattern or image list set.")
        return

    curr_label_path = curr_image_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    with open(curr_label_path, "w") as f:
        f.write(src_labels)

    st.session_state["skip_label_update"] = True

def copy_prev_labels():
    """Copies labels from the previous slide, if it exists."""
    if st.session_state.frame_index > 0:
        source_index = st.session_state.frame_index - 1
        copy_labels_from_slide(source_index)
    else:
        source_index = st.session_state.max_images - 1
        copy_labels_from_slide(source_index)

def copy_next_labels():
    """Copies labels from the next slide, if it exists."""
    if st.session_state.frame_index < st.session_state.start_index + st.session_state.max_images - 1:
        source_index = st.session_state.frame_index + 1
        copy_labels_from_slide(source_index)
    else:
        source_index = 0
        copy_labels_from_slide(source_index)

def manual_label_subset_checkbox_callback():
    # Retrieve the checkbox value from session state using its key.
    use_subset_val = st.session_state.get("manual_label_subset_btn", False)
    
    # Update session state as required.]
    st.session_state.use_subset = use_subset_val
    st.session_state.use_subset_changed = True
    st.session_state.automatic_generate_list = True
    st.session_state.frame_index = 0
    st.session_state["skip_label_update"] = True
    
    # Call the functions.
    update_unverified_data_path()
    update_unverified_frame()

def handle_image_list_update(prefix=""):
    """
    Checks for image warnings or naming pattern warnings and handles
    image list updates. Returns True if it's okay to proceed,
    otherwise returns False (after handling warnings and calling st.rerun()).
    """
    # If image list has already been stored, do not re-run the process.
    if st.session_state.get("image_list_stored", False):
        return True
    else:
        if st.session_state.get("no_images_warning"):
            st.warning(st.session_state.no_images_warning)
            
            return False
        elif st.session_state.get("naming_pattern_warning"):
            if st.session_state.automatic_generate_list:
                images_dir = st.session_state.paths["unverified_images_path"]
                images = []
                for ext in [".jpg", ".png"]:
                    images.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
                images.sort()
                st.session_state.image_list = images
                st.session_state.max_images = len(images)
                st.session_state.start_index = 0
                st.session_state.image_pattern = None
                st.session_state.naming_pattern_warning = None
                # Set the flag so that next runs do not redo this process.
                st.session_state.image_list_stored = True
                st.rerun()
            else:
                st.warning(st.session_state.naming_pattern_warning)
                option = st.radio(
                    "Select an option to proceed:",
                    options=[
                        "Rename files with a pattern (lose original file names for faster performance)",
                        "Store list of all images (retain original file names but lose performance)"
                    ],
                    key=prefix+"naming_pattern_choice"
                )
                if option == "Rename files with a pattern (lose original file names for faster performance)":
                    if st.button("Apply Rename", key=prefix+"apply_rename"):
                        images_dir = st.session_state.paths["unverified_images_path"]
                        new_pattern, total_images = safe_rename_images(images_dir)
                        if new_pattern is not None:
                            st.session_state.image_pattern = new_pattern
                            st.session_state.start_index = 0
                            st.session_state.max_images = total_images
                            st.session_state.naming_pattern_warning = None
                            st.session_state.image_list = None
                            st.success("Session state updated after renaming.")
                            # Reset the flag since the list (pattern-based) is now in use.
                            st.session_state.image_list_stored = True
                            st.rerun()
                else:
                    if st.button("Store Image List", key=prefix+"store_image_list"):
                        images_dir = st.session_state.paths["unverified_images_path"]
                        images = []
                        for ext in [".jpg", ".png"]:
                            images.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
                        images.sort()
                        st.session_state.image_list = images
                        st.session_state.max_images = len(images)
                        st.session_state.start_index = 0
                        st.session_state.image_pattern = None
                        st.session_state.naming_pattern_warning = None
                        st.session_state.automatic_generate_list = True
                        st.success("Image list stored. Note: This may slow down performance for large datasets. 3 ")
                        st.session_state.image_list_stored = True
                        st.rerun()
                return False
        return True

def get_frame_index_from_filename(filename):
    """
    Determines the frame index corresponding to the given image filename.

    This function works in various scenarios:
      - If an image list (st.session_state.image_list) is stored, it returns the index of the filename in that list.
      - If an image naming pattern (st.session_state.image_pattern) is used, it extracts the numeric part 
        from the filename and adjusts it by st.session_state.start_index.
      - If neither is available, it falls back to extracting the first numeric sequence from the basename.
      - Additionally, if a subset is used (st.session_state.use_subset is True and st.session_state.subset_frames exists),
        the function returns the index within the subset list corresponding to the actual frame index.

    Args:
        filename (str): The full path to the image file.

    Returns:
        int or None: The frame index as an integer, or None if it cannot be determined.
    """
    abs_filename = os.path.abspath(filename)
    base_index = None

    # If an image list exists, try to find the filename in that list.
    if "image_list" in st.session_state and st.session_state.image_list:
        for idx, path in enumerate(st.session_state.image_list):
            if os.path.abspath(path) == abs_filename:
                base_index = idx
                break

    # If not found and a naming pattern is available, extract the numeric sequence.
    if base_index is None and st.session_state.get("image_pattern"):
        basename = os.path.basename(filename)
        # Extract digits that appear immediately before the file extension.
        match = re.search(r"(\d+)(?=\.[^.]+$)", basename)
        if match:
            extracted = int(match.group(1))
            start_index = st.session_state.get("start_index", 0)
            base_index = extracted - start_index

    # Fallback: simply extract the first numeric sequence from the basename.
    if base_index is None:
        basename = os.path.basename(filename)
        match = re.search(r"(\d+)", basename)
        if match:
            base_index = int(match.group(1))
        else:
            return None

    # If using a subset, convert the actual frame index to the subset index.
    if st.session_state.get("use_subset", False) and st.session_state.get("subset_frames"):
        subset_frames = st.session_state.subset_frames
        if base_index in subset_frames:
            subset_index = subset_frames.index(base_index)
            return subset_index
        else:
            # If the actual index isn't found in the subset list, return the base_index as fallback.
            return base_index

    return base_index

def add_frame_callback(key):
    add_val = st.session_state[key]
    if add_val not in st.session_state.subset_frames:
        st.session_state.subset_frames.append(add_val)
        save_subset_csv(csv_file, st.session_state.subset_frames)
    st.session_state["skip_label_update"] = True

def remove_frame_callback(key):
    remove_val = st.session_state[key]
    if remove_val in st.session_state.subset_frames:
        st.session_state.subset_frames.remove(remove_val)
        save_subset_csv(csv_file, st.session_state.subset_frames)
        st.session_state["skip_label_update"] = True

def _bboxes_changed(prev_out, curr_out):
    """Return True if bboxes were added, removed, or moved."""
    # Count changed?
    if len(prev_out["bbox"]) != len(curr_out["bbox"]):
        return True
    prev_ids = {item["bbox_ids"] for item in prev_out["bbox"]}
    curr_ids = {item["bbox_ids"] for item in curr_out["bbox"]}
    if prev_ids != curr_ids:
        return True
    # Coordinates changed?
    for curr in curr_out["bbox"]:
        for prev in prev_out["bbox"]:
            if prev["bbox_ids"] == curr["bbox_ids"] and prev["bboxes"] != curr["bboxes"]:
                return True
    return False

def detection(*args, **kwargs):
    # initialize flags
    st.session_state.setdefault("detection_running", False)
    st.session_state.setdefault("detection_modified", False)

    # mark start
    st.session_state.detection_running = True
    curr = _orig_detection(*args, **kwargs)
    # mark end
    st.session_state.detection_running = False

    prev = st.session_state.get("prev_out")
    if prev is not None:
        st.session_state.detection_modified = _bboxes_changed(prev, curr)
    else:
        st.session_state.detection_modified = False

    st.session_state.prev_out = curr
    return curr

def set_scale():
    st.session_state.unverified_image_scale = st.session_state.unverified_image_scale_input
    st.session_state.skip_label_update = True

    st.session_state.detector_key = f"detector_{uuid.uuid4().hex}"

## Zoom & Object Edit Callbacks

def zoom_edit_callback(i):

    # If the box at index i has been deleted, just return.
    if i < 0 or i >= len(st.session_state.bboxes_xyxy):
        return

    old_bbox = st.session_state.bboxes_xyxy[i]

    try:
        new_center_x = st.session_state[f"bbox_{i}_cx"]
        flipped_center_y = st.session_state[f"bbox_{i}_cy"]
        image_height = st.session_state.image_height
        actual_center_y = image_height - flipped_center_y
        new_w = st.session_state[f"bbox_{i}_w"]
        new_h = st.session_state[f"bbox_{i}_h"]

        new_x = new_center_x - new_w/2
        new_y = actual_center_y - new_h/2

        if new_x < 0:
            new_x = 0.0
        if new_y < 0:
            new_y = 0.0
        if new_x + new_w > st.session_state.image_width:
            new_x = st.session_state.image_width - new_w
        if new_y + new_h > image_height:
            new_y = image_height - new_h

        if (new_x, new_y, new_w, new_h) != tuple(old_bbox):
            st.session_state.bboxes_xyxy[i] = [new_x, new_y, new_w, new_h]
            label_path = st.session_state.label_path
            image_width = st.session_state.image_width
            with open(label_path, "w") as f:
                for label, bbox in zip(st.session_state.labels, st.session_state.bboxes_xyxy):
                    bx, by, bw, bh = bbox
                    x_center_norm = (bx + bw/2) / image_width
                    y_center_norm = (by + bh/2) / image_height
                    width_norm = bw / image_width
                    height_norm = bh / image_height
                    f.write(f"{label} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
            st.session_state["skip_label_update"] = True
    except:
        pass

def get_object_by_global_index(global_index):
    """
    Iterates over the image list (built using the naming pattern if available)
    and returns the object (bounding box and label) corresponding to the given global index.
    Returns a dictionary with keys: img, image_path, label_path, bbox, label, local_index, global_index.
    """
    
    # Build image list based on naming pattern if available; otherwise, use stored list.
    if st.session_state.get("image_pattern") is not None:
        image_list = [
            os.path.join(st.session_state.paths["unverified_images_path"],
                         st.session_state.image_pattern.format(i))
            for i in range(st.session_state.start_index, st.session_state.start_index + st.session_state.max_images)
        ]
    else:
        image_list = st.session_state.image_list if "image_list" in st.session_state else []

    current_obj = None
    count = 0
    for image_path in image_list:
        try:
            img = Image.open(image_path)
        except Exception:
            continue
        label_file = image_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        bboxes = []
        labels = []
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls = int(parts[0])
                            x_center, y_center, w_norm, h_norm = map(float, parts[1:5])
                        except Exception:
                            continue
                        w_abs = w_norm * img.width
                        h_abs = h_norm * img.height
                        x_abs = (x_center * img.width) - (w_abs / 2)
                        y_abs = (y_center * img.height) - (h_abs / 2)
                        bboxes.append([x_abs, y_abs, w_abs, h_abs])
                        labels.append(cls)
                        
        for local_index, bbox in enumerate(bboxes):
            if count == global_index:
                current_obj = {
                    "img": img,
                    "image_path": image_path,
                    "label_path": label_file,
                    "bbox": bbox,
                    "label": labels[local_index] if local_index < len(labels) else None,
                    "local_index": local_index,
                    "global_index": global_index
                }
            count += 1

    if current_obj:
        current_obj["num_labels"] = count
        
    return current_obj

def load_objects_from_image(image_path):
    try:
        img = Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None, None, None, None
    width, height = img.size
    label_path = image_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    bboxes = []
    labels = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls = int(parts[0])
                        x_center, y_center, w_norm, h_norm = map(float, parts[1:5])
                    except Exception:
                        continue
                    w_abs = w_norm * width
                    h_abs = h_norm * height
                    x_abs = (x_center * width) - (w_abs / 2)
                    y_abs = (y_center * height) - (h_abs / 2)
                    bboxes.append([x_abs, y_abs, w_abs, h_abs])
                    labels.append(cls)
    return img, bboxes, labels, label_path

def object_by_object_edit_callback():
    """
    Reads new values from session_state keys for the current object,
    recalculates the bounding box, updates the label file if changed,
    and sets a flag to freeze the UI while running.
    """
    st.session_state["object_running"] = True
    try:
        current_obj = get_object_by_global_index(st.session_state.global_object_index)
        if current_obj is None:
            st.warning("No object found to update.")
            return

        img = current_obj["img"]
        old_bbox = current_obj["bbox"]
        global_idx = current_obj["global_index"]

        # Retrieve new values (falling back on old if missing)
        new_center_x = st.session_state.get(f"object_{global_idx}_center_x", old_bbox[0] + old_bbox[2]/2)
        new_center_y = st.session_state.get(f"object_{global_idx}_center_y", old_bbox[1] + old_bbox[3]/2)
        new_w = st.session_state.get(f"object_{global_idx}_w", old_bbox[2])
        new_h = st.session_state.get(f"object_{global_idx}_h", old_bbox[3])

        # Compute and clamp new bbox
        new_x = max(0.0, new_center_x - new_w/2)
        new_y = max(0.0, new_center_y - new_h/2)
        new_w = min(new_w, img.width)
        new_h = min(new_h, img.height)
        if new_x + new_w > img.width:
            new_x = img.width - new_w
        if new_y + new_h > img.height:
            new_y = img.height - new_h

        new_bbox = [new_x, new_y, new_w, new_h]

        # If changed, rewrite the label file
        if new_bbox != old_bbox:
            label_file = current_obj["label_path"]
            with open(label_file, "r") as f:
                lines = f.readlines()
            local_idx = current_obj["local_index"]
            x_center_norm = (new_x + new_w/2) / img.width
            y_center_norm = (new_y + new_h/2) / img.height
            width_norm = new_w / img.width
            height_norm = new_h / img.height
            lines[local_idx] = f"{current_obj['label']} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n"
            with open(label_file, "w") as f:
                f.writelines(lines)
    except Exception as e:
        st.error(f"Error updating object: {e}")
    finally:
        # always clear the “running” flag
        st.session_state["object_running"] = False

@st.cache_data(show_spinner=False)
def extract_features(img_crop):
    """
    Compute a normalized 3D HSV histogram feature for an image crop.
    """
    hsv = cv2.cvtColor(np.array(img_crop), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

@st.cache_data(show_spinner=False)
def _get_thumbnail_b64(idx: int, thumb_width: int) -> str:
    # Assumes existing function to fetch base64 thumbnail string for object idx
    obj = get_object_by_global_index(idx)
    bx, by, bw, bh = obj["bbox"]
    crop = obj["img"].crop((int(bx), int(by), int(bx + bw), int(by + bh)))
    buf = BytesIO()
    crop.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def render_checkbox_grid(predicted_indices, thumb_width: int = 150, cols: int = 4):
    """
    Displays a grid of thumbnails with checkboxes.
    Returns list of indices for which the checkbox was checked.
    """

    selected = []
    # Split into rows
    rows = [predicted_indices[i:i+cols] for i in range(0, len(predicted_indices), cols)]
    for row in rows:
        columns = st.columns(cols)
        for idx, col in zip(row, columns):
            with col:
                b64 = _get_thumbnail_b64(idx, thumb_width)
                checked = st.checkbox(f"{idx}", key=f"remove_{idx}")
                st.image(f"data:image/png;base64,{b64}", width=thumb_width, use_container_width=False)
                if checked:
                    selected.append(idx)

    return selected

## Image / Video Processing & Creation

def remove_white_background(img: Image.Image, threshold=240):
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] > threshold and item[1] > threshold and item[2] > threshold:
            # Set white-ish pixels to transparent
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img

def center_image_transparent(image_path, width):
    img = Image.open(image_path)
    img = remove_white_background(img)

    # Save to buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()

    st.markdown(
        f'<div style="text-align: center;"><img src="data:image/png;base64,{encoded}" width="{width}"></div>',
        unsafe_allow_html=True
    )

def add_labels(frame, image_path):
    """
    Overlay YOLO-format labels onto a video frame.
    
    For the given image file path, this function constructs the corresponding
    text file path (by replacing the image extension with .txt). If the text file exists,
    it reads each line (assumed to be in YOLO format: class x_center y_center width height)
    and draws a red bounding box and the class id onto the image.
    
    Args:
        frame (np.array): The video frame as a NumPy array.
        image_path (str): Path to the current image.
    
    Returns:
        np.array: The frame with drawn labels.
    """
    # Convert frame (NumPy array) to a PIL Image for drawing.
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    
    # Construct the label file path by replacing the image extension with .txt
    base, _ = os.path.splitext(image_path)
    label_file = base + ".txt"
    label_file = label_file.replace("/images/", "/labels/")
    
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, x_center, y_center, w, h = parts[:5]
            try:
                x_center, y_center, w, h = map(float, (x_center, y_center, w, h))
            except Exception:
                continue
            # Get image dimensions
            img_w, img_h = pil_img.size
            # Convert normalized coordinates to absolute pixel values.
            box_w = w * img_w
            box_h = h * img_h
            top_left_x = (x_center * img_w) - (box_w / 2)
            top_left_y = (y_center * img_h) - (box_h / 2)
            bottom_right_x = top_left_x + box_w
            bottom_right_y = top_left_y + box_h
            # Draw the bounding box.
            draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], outline="red", width=2)
            # Draw the class id (you can later map this to a class name if needed).
            draw.text((top_left_x, top_left_y), str(cls), fill="red")
    # If no label file exists, leave the frame unchanged.
    return np.array(pil_img)

def overlay_frame_text(frame, index):
    # Convert the original frame (numpy array) to a PIL Image.
    original_img = Image.fromarray(frame)
    orig_width, orig_height = original_img.size

    # Define blank area height (50% of original height) and create a new image.
    blank_height = orig_height // 2
    new_height = orig_height + blank_height
    new_img = Image.new("RGB", (orig_width, new_height), color="black")
    
    # Paste the original frame at the top.
    new_img.paste(original_img, (0, 0))
    draw = ImageDraw.Draw(new_img)

    # Determine the text to display.
    if st.session_state.get("use_subset", False) and st.session_state.get("subset_frames"):
        try:
            actual_frame = st.session_state.subset_frames[index]
            line1 = f"Frame: {index}"
            line2 = f"Subset Index: {actual_frame}"
            text = line1 + "\n" + line2
        except Exception:
            text = f"Frame: {index}"
    else:
        text = f"Frame: {index}"

    # Define margins (5% horizontal, 10% vertical of blank area).
    margin_x = int(orig_width * 0.05)
    margin_y = int(blank_height * 0.1)
    available_width = orig_width - 2 * margin_x
    available_height = blank_height - 2 * margin_y

    # Estimate initial font size based on available height and number of text lines.
    lines = text.split("\n")
    num_lines = len(lines)
    font_size = int(available_height / (num_lines + 0.5))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Auto-scale the font so the text fits within the available width.
    text_bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=5)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    while text_width > available_width and font_size > 10:
        font_size -= 1
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        text_bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=5)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

    # Center the text within the blank area.
    text_x = (orig_width - text_width) / 2
    text_y = orig_height + (blank_height - text_height) / 2

    # Draw text outline for better visibility.
    outline_range = 2
    for dx in range(-outline_range, outline_range + 1):
        for dy in range(-outline_range, outline_range + 1):
            if dx != 0 or dy != 0:
                draw.multiline_text((text_x + dx, text_y + dy), text, font=font, fill="black", spacing=5, align="center")
    # Draw the main text in white.
    draw.multiline_text((text_x, text_y), text, font=font, fill="white", spacing=5, align="center")

    return np.array(new_img)

def create_video_file(image_paths, fps, scale=1.0, output_path="generated_videos/current.mp4"):
    """
    Create a composite video with two side-by-side panels:
      - Left panel: original video frames.
      - Right panel: video frames with YOLO labels overlaid (if a corresponding .txt file exists).
      
    After combining the panels, the final composite image has an overlay at its center showing the frame number.
    
    Args:
        image_paths (list): List of file paths to the images.
        fps (float): Frames per second.
        scale (float): Scaling factor for the video size.
        output_path (str): File path to save the generated MP4 video.
    
    Returns:
        str: The output video file path.
    """
    duration = len(image_paths) / fps

    # Create the original clip without any overlay.
    clip_original = ImageSequenceClip(image_paths, fps=fps)
    
    # Create the labeled clip using a custom frame function.
    def make_labeled_frame(t):
        index = int(t * fps)
        if index >= len(image_paths):
            index = len(image_paths) - 1
        current_path = image_paths[index]
        frame = np.array(Image.open(current_path))
        # add_labels should be defined elsewhere to overlay YOLO labels.
        frame_with_labels = add_labels(frame, current_path)
        return frame_with_labels
    
    clip_labeled = VideoClip(make_labeled_frame, duration=duration).set_fps(fps)
    
    # Combine the two clips side by side.
    final_clip = clips_array([[clip_original, clip_labeled]])
    
    # Apply a final overlay to the composite image.
    def add_overlay(get_frame, t):
        frame = get_frame(t)
        index = int(t * fps)
        return overlay_frame_text(frame, index)
    
    final_clip = final_clip.fl(add_overlay, apply_to=['mask', 'video'])
    
    # Write the final composite clip to an MP4 file.
    os.makedirs(os.path.dirname(output_path), mode=0o777, exist_ok=True)
    final_clip.write_videofile(output_path, codec="libx264", audio=False, verbose=False, logger=None)
    
    return output_path

def parse_label_data(labels_path):

    # Define a map for class IDs and arrays for bounding box coordinates
    class_counts = {}
    x_vals, y_vals, w_vals, h_vals = [], [], [], []

    # For each label file in the labels directory
    for label_file in os.listdir(labels_path):
        if label_file.endswith(".txt"):
            file_has_data = False  # Flag to track if the file has valid data

            # Read the label file
            with open(os.path.join(labels_path, label_file), "r") as f:
                for line in f:
                    # Skip empty lines
                    line = line.strip()
                    if not line:  # If the line is empty
                        continue
                    
                    # Split the line into parts and check if it has enough data
                    parts = line.split()
                    if len(parts) >= 5:  # Check if the line has enough data
                        cls, x, y, bw, bh = map(float, parts)

                        # Check if the class ID is new and define
                        if str(cls) not in class_counts:
                            class_counts[str(cls)] = 0

                        # Append the class counts and bounding box coordinates
                        class_counts[str(cls)] += 1
                        x_vals.append(x)
                        y_vals.append(y)
                        w_vals.append(bw)
                        h_vals.append(bh)
                        file_has_data = True  # Set the flag to True if there's valid data
                    else:
                        continue  # This case is handled in other function

            # If the file has no valid data at all, consider it an unlabeled file
            if not file_has_data:
                if "unlabeled" not in class_counts:
                    class_counts["unlabeled"] = 0
                class_counts["unlabeled"] += 1

    return class_counts, x_vals, y_vals, w_vals, h_vals

def generate_label_plots(class_counts, x_vals, y_vals, w_vals, h_vals):

    # Check if class_counts is empty
    if not class_counts:
        st.warning("No class labels found. The label files may be empty.")
        return

    # Check if x_vals, y_vals, w_vals, h_vals are empty
    if not x_vals or not y_vals or not w_vals or not h_vals:
        st.warning("No label data found for coordinate plots.")
        return

    # Pull the class keys and values for plotting and make them unique colors
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
    
    # Define 3 col subplot (1 row)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # X vs Y (Centers)
    axs[0].hexbin(x_vals, y_vals, gridsize=30, cmap="Blues")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("X vs Y (Bounding Box Centers)")
    axs[0].invert_yaxis()

    # Width vs Height (Sizes)
    axs[1].hexbin(w_vals, h_vals, gridsize=30, cmap="Blues")
    axs[1].set_xlabel("width")
    axs[1].set_ylabel("height")
    axs[1].set_title("Width vs Height (Bounding Box Sizes)")

    # Class Distribution
    axs[2].bar(classes, counts, color=colors)
    axs[2].set_xlabel("Class ID")
    axs[2].set_ylabel("Count")
    axs[2].set_title("Class Distribution")
    axs[2].tick_params(axis='x', rotation=45)

    st.pyplot(fig)

def draw_bboxes_on_image(image_path, label_path):

    # Open image and initialize
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size

    # Read the label file and draw bounding boxes in red
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, x, y, bw, bh = map(float, parts)
            x1 = (x - bw / 2) * w
            y1 = (y - bh / 2) * h
            x2 = (x + bw / 2) * w
            y2 = (y + bh / 2) * h
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), str(int(cls)), fill="red")

    return image

def display_images(images_path, labels_path):

    # Check if the images path exists
    if not os.path.exists(images_path):
        st.error("Images path does not exist.")
        return

    # Check if the labels path exists
    if not os.path.exists(labels_path):
        st.error("Labels path does not exist.")
        return

    # Find all the image files in the path
    image_files = [f for f in os.listdir(images_path) if f.endswith(('png', 'jpg', 'jpeg'))]

    # Check if there are any image files
    if not image_files:
        st.warning("No images found in the images folder.")
        return

    # Randomly select 3 images
    sample_images = random.sample(image_files, min(3, len(image_files)))

    # Display the images in a 3-column layout
    cols = st.columns(3)
    for i, image_file in enumerate(sample_images):
        image_path = os.path.join(images_path, image_file)
        label_path = os.path.join(labels_path, image_file.rsplit('.', 1)[0] + '.txt')
        annotated_img = draw_bboxes_on_image(image_path, label_path)
        cols[i % 3].image(annotated_img, caption=image_file, use_container_width=True)

@st.cache_data(show_spinner=True)
def generating_mp4(image_paths, fps, regenerate=False, output_path = "generated_videos/current.mp4"):
 
    # If the file exists and we're not forcing regeneration, just return the existing file path.
    if os.path.exists(output_path) and not regenerate:
        return output_path
    
    # Otherwise, generate a new video file.
    return create_video_file(image_paths, fps, output_path=output_path)

def change_video_path_callback():
    # Construct the new video path by joining the new folder path and the basename of the current video path.
    new_video_path = os.path.join(
        st.session_state.paths["convert_video_copy_path"],
        os.path.basename(st.session_state.paths["convert_video_path"])
    )

    # Move the video file to the new location.
    os.makedirs(st.session_state.paths["convert_video_copy_path"], exist_ok=True, mode=0o777)
    shutil.move(st.session_state.paths["convert_video_path"], new_video_path.replace(" ", "_"))

    # Update the session state with the new path.
    st.session_state.paths["convert_video_path"] = new_video_path

def get_random_image(directory):
    """
    Returns a random image filepath from the given directory.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png')
    try:
        files = os.listdir(directory)
    except Exception as e:
        return None
    images = [os.path.join(directory, f) for f in files if f.lower().endswith(valid_extensions)]
    if images:
        return random.choice(images)
    return None

def _frame_label_count(idx):
    """Return number of non‑empty lines in that frame's label file."""
    # build image path
    if st.session_state.image_pattern:
        img = os.path.join(
            st.session_state.images_dir,
            st.session_state.image_pattern.format(idx)
        )
    else:
        img = st.session_state.image_list[idx]
    lbl = img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    if os.path.exists(lbl):
        with open(lbl) as f:
            return sum(1 for l in f if l.strip())
    return 0

#--------------------------------------------------------------------------------------------------------------------------------#
##  Configuration Tracking
#--------------------------------------------------------------------------------------------------------------------------------#

if "session_running" not in st.session_state:
    st.session_state.session_running = True
    st.session_state.do_rerun = False

    st.session_state.object_by_object_jump_valid = False
    st.session_state.object_by_object_jump_warning = None
    
    st.session_state.video_saved_for_current_run = False
    st.session_state.playback_active = False
    st.session_state.fps = 1
    st.session_state.video_index = 0
    st.session_state.include_labels = True

    st.set_page_config(
        page_title="Autolabel Engine",
        layout="wide"
    )

    st.session_state.paths = {

        "venv_path" : "../envs/auto-label-engine/",
        "generate_venv_script_path": "setup_venv.sh",

        "prev_unverified_images_path" : "example_data/images",
        "unverified_images_path" : "example_data/images",
        "prev_unverified_names_yaml_path" : "cfgs/gui/manual_labels/default.yaml",
        "unverified_names_yaml_path" : "cfgs/gui/manual_labels/default.yaml",

        "upload_save_path": "",

        "convert_video_path" : "",
        "convert_video_save_path" : "",
        "convert_video_script_path" : "convert_mp4_2_png.py",
        "convert_video_copy_path" : "",

        "rotate_images_path":  "example_data",
        "rotate_images_script_path" : "rotate_images.py",

        "split_data_path" : "example_data",
        "split_data_save_path" : "",
        "split_data_script_path" : "split_yolo_data_by_object.py",

        "auto_label_save_path" : "example_data/labels/",
        "auto_label_model_weight_path" : "weights/coco_2_ijcnn_vr_full_2_real_world_combination_2_hololens_finetune-v3.pt",
        "auto_label_data_path" :  "example_data/images/",
        "auto_label_script_path" : "inference.py",
     
        "combine_dataset_1_path": "example_data/",
        "combine_dataset_2_path": "example_data/",
        "combine_dataset_save_path": "example_data_combined/",
        "combine_dataset_script_path" : "combine_yolo_dirs.py",

        "train_data_yaml_path": "cfgs/yolo/data/default.yaml",
        "train_model_yaml_path": "cfgs/yolo/model/default.yaml",
        "train_train_yaml_path": "cfgs/yolo/train/default.yaml",
        "train_script_path" : "train_yolo.py",

        "unverified_subset_csv_path" : "cfgs/gui/subset/default.csv",
        "subset_save_path": "cfgs/gui/subset/new_subset.csv",

        "video_file_path": "generated_videos/current.mp4",

        "move_src_path": "",
        "move_dest_path": "",
        "move_dir_script_path": "move_dir.py",

        "open_workspace": ".",

        "cluster_script_path": "cluster_objects.py",

    }
    st.session_state.detector_key = f"detector_{uuid.uuid4().hex}"
    st.session_state.global_object_index = 0

    st.session_state.use_subset = False
    st.session_state.use_subset_changed = False
    st.session_state.subset_frames = []
    st.session_state.subset_index = 0
    st.session_state.automatic_generate_list = False

    st.session_state.unverified_image_scale = 1.0

    load_session_state()
    
    update_unverified_data_path()

    gpu_info = subprocess.check_output("nvidia-smi -L", shell=True).decode("utf-8")
    st.session_state.gpu_list = [line.strip() for line in gpu_info.splitlines() if line.strip()]

save_session_state()


## Define the title bar and brief description
st.title("Auto Label Engine")
st.write("This is a GUI for the Auto Label Engine. It allows you to generate datasets, auto label images,"
" manually label images, see dataset statistics/figures, finetune models, and run Linux commands on the server.")

with st.expander("Recommended Auto Label Engine Workflow"):
    st.write(
        "Below is a diagram of the Auto Label Engine workflow. Components are color-coded into human interaction, files and folders, and "
        "processes"
    )
    
    center_image_transparent("figures/ale_workflow.png", 800)

    st.write("See the [Auto Label Engine GitHub repo](https://github.com/RowanMAVRC/AutoLabelEngine) for more information.")

# GUI
#--------------------------------------------------------------------------------------------------------------------------------#

if not os.path.exists(os.path.join(st.session_state.paths["venv_path"], "bin/activate")):
    with st.expander("Please generate a venv before running any modules"):
        

        action_option = st.radio(
            "Choose save path option:", 
            [
                "Generate New Virtual Enviornement",
                "Choose Another Path"
            ],
            key=f"venv_radio",
            label_visibility="collapsed"
        )

        if action_option == "Generate New Virtual Enviornement":
            st.subheader("Virtual Environment Save Path")
            path_navigator("venv_path", radio_button_prefix="generate_new_venv")

            output = None
            c1, c2, c3, c4 = st.columns(4, gap="small")
            with c1:
                if st.button("Generate Virtual Enviornment", key="generate_venv_btn"):
                    run_in_tmux(
                        session_key="generate_venv", 
                        script_path=st.session_state.paths["generate_venv_script_path"], 
                        args=st.session_state.paths["venv_path"],
                        script_type="bash"
                    )
                    time.sleep(3)
                    output = update_tmux_terminal("generate_venv")

            with c2:
                if st.button("🔄 Refresh Terminal", key="check_generate_venv_btn"):
                    output = update_tmux_terminal("generate_venv")

            with c3:
                if st.button("🧹 Clear Output", key="generate_venv_clear_terminal_btn"):
                    output = None

            with c4:
                if st.button("❌ Kill Session", key="generate_venv_kill_tmux_session_btn"):
                    output = kill_tmux_session("generate_venv")


            terminal_output = st.empty()
            if output is not None:
                display_terminal_output(output)
                    
        else:
            st.subheader("Choose Virtual Environment Path")
            path_navigator("venv_path", radio_button_prefix="find_new_venv")
    
    st.warning("Virtual environment has not be generated on this device. Please choose one of the following options.")

## Main Tabs
tabs = st.tabs(["Generate Data", "Linux Terminal", "Unrestrict Workspace"])

# ----------------------- Generate Data Tab -----------------------
with tabs[0]:  

    output = None
    action_option = st.radio(
        "Choose:", 
        [
            "Upload Data", 
            "Convert Video to Frames", 
            "Rotate Image Dataset",
            "Auto Label",
            "Generate Labeled Video",
            "Video Review",
            "Object by Object Review", 
            "Frame by Frame Review", 
            "Move Directory",
            "Split YOLO Dataset into Objects / No Objects", 
            "Combine YOLO Datasets",
            "Dataset Statistics",
            "Finetune Model"
        ],
        key=f"generate_data_radio",
        label_visibility="collapsed"
    )

    if action_option == "Upload Data":
        with st.expander("Upload Data"):
            st.write("### Save Path")
            st.write("The path to upload image data on the server.")
            path_navigator("upload_save_path")
            upload_to_dir(st.session_state.paths["upload_save_path"])

    elif action_option == "Convert Video to Frames":
        with st.expander("Settings"):
            st.subheader("Video Path")
            st.write("Enter the path to an MP4/MOV video file **or directory** containing video files (all subdirectories will be scanned).")
            path_navigator("convert_video_path")
            
            video_path = st.session_state.paths.get('convert_video_path', '')
            video_files = []
            if os.path.isdir(video_path):
                st.write("Found videos...")
                for root, dirs, files in os.walk(video_path):
                    for f in files:
                        if f.lower().endswith(('.mp4', '.mov')):
                            video_files.append(os.path.join(root, f))
                if not video_files:
                    st.warning("No MP4/MOV video files found in the selected directory.")
                else:
                    base_dir = os.path.abspath(video_path)
                    rel_paths = [os.path.relpath(v, base_dir) for v in sorted(video_files)]
                    st.selectbox("Found Videos", options=rel_paths, key="found_video_select", label_visibility="collapsed")
            else:
                st.write("**Single video file selected.**")
            
            st.subheader("Frame Save Path")
            st.write("The global default output directory is determined as follows:")
            key = "convert_video_save_path"
            save_path_option = st.radio("Choose save path option:", ["Default", "Custom"],
                                        key="frame_save_path_radio", label_visibility="collapsed")
            if save_path_option == "Default":
                if os.path.isdir(video_path):
                    if "to_be_converted" in video_path:
                        default_output_dir = video_path.replace("to_be_converted", "to_be_reviewed").replace("video_data", "yolo_format_data")
                    else:
                        default_output_dir = video_path.rstrip(os.sep) + "_images"
                    st.session_state.paths[key] = default_output_dir
                else:
                    st.session_state.paths[key] = (
                        video_path.replace('.mp4', '/images/')
                        .replace('video_data', 'yolo_format_data')
                        .replace('.MOV', '/images/')
                        .replace("to_be_converted", "to_be_reviewed")
                    )
                st.write(f"**Current Frame Save Path:** {st.session_state.paths[key]}")
            else:
                path_navigator(key)

            st.subheader("New Video Path Save Path")
            st.write("The base folder to move the original video after conversion.")
            key = "convert_video_copy_path"
            save_path_option = st.radio("Choose save path option:", ["Default", "Custom"],
                                        key="convert_video_copy_radio", label_visibility="collapsed")
            if save_path_option == "Default":
                if os.path.isdir(video_path):
                    if "to_be_converted" in video_path:
                        default_copy_dir = video_path.replace("to_be_converted", "converted")
                    else:
                        default_copy_dir = video_path.rstrip(os.sep) + "/converted_videos"
                    st.session_state.paths[key] = default_copy_dir
                else:
                    st.session_state.paths[key] = os.path.dirname(
                        video_path.replace("to_be_converted", "converted")
                    )
                st.write(f"**Current New Video Path Save Path:** {st.session_state.paths[key]}")
            else:
                path_navigator(key, radio_button_prefix="change_after_conversion_")
        
        with st.expander("Virtual Environment Path"):
            st.write("Provide the path to the virtual environment containing the required packages.")
            path_navigator("venv_path", radio_button_prefix="convert_mp4")
        
        with st.expander("Script"):
            path_navigator("convert_video_script_path")
            python_code_editor("convert_video_script_path")
        
        with st.expander("Convert Video to Frames"):
            st.write(
                "Click **Begin Converting** to start processing your video file(s) in a background tmux session.\n\n"
                "• For directories, a single tmux session will be launched that loops over each video internally, so the GUI doesn't wait for each conversion.")
            st.write(
                "• Once conversion completes, each video file is automatically moved to the new location."
            )

            output = None
            c1_conv, c2, c3, c4 = st.columns(4, gap="small")
            session_key = "convert_video_background"
            with c1_conv:

                if st.button("▶ Begin Converting", key="begin_converting_data_btn"):
                    video_path = st.session_state.paths.get('convert_video_path', '')
                    # Launch the conversion script as a single background task,
                    # passing the "copy_destination" so the script can move each video when done.
                    
                    run_in_tmux(
                        session_key=session_key,
                        script_path=st.session_state.paths["convert_video_script_path"],
                        venv_path=st.session_state.paths["venv_path"],
                        args={
                            "video_path": video_path.replace(" ", "\ "),
                            "output_folder": st.session_state.paths["convert_video_save_path"].replace(" ", "\ "),
                            "copy_destination": st.session_state.paths["convert_video_copy_path"].replace(" ", "\ ")
                        }
                    )
                    st.success("Conversion started in the background. Check tmux for progress.")
                    converted = True

            with c2:
                if st.button("🔄 Refresh Terminal", key="check_convert_video_btn"):
                    output = update_tmux_terminal(session_key)

            with c3:
                if st.button("🧹 Clear Output", key="convert_video_terminal_btn"):
                    output = None

            with c4:
                if st.button("❌ Kill Session", key="convert_video_kill_tmux_session_btn"):
                    output = kill_tmux_session(session_key)

    elif action_option == "Rotate Image Dataset":
        with st.expander("Settings"):
            st.subheader("Image Path")
            st.write("The path to the image datasets. If subdirectories are found, each subdirectory will be treated as an individual dataset.")
            image_directory = path_navigator("rotate_images_path")

            enable_option = st.radio(
                "Choose save path option:", 
                [
                    "Disable Preview (Better Performance)",
                    "Preview Datasets and Set Rotations", 
                ],
                key=f"enable_radio",
                label_visibility="collapsed"
            )

            if enable_option == "Preview Datasets and Set Rotations":
                datasets_list = []  # New list to collect each dataset's info.
                if image_directory and os.path.isdir(image_directory):
                    subdirs = [os.path.join(image_directory, d, "images") for d in os.listdir(image_directory)
                            if os.path.isdir(os.path.join(image_directory, d))]
                    datasets = subdirs if subdirs else [image_directory]

                    for dataset in datasets:
                        st.markdown(f"#### Dataset: {dataset}")
                        random_img_path = get_random_image(dataset)

                        if random_img_path:
                            c1, c2, c3 = st.columns([.37, .16, .37])
                            
                            with c1:
                                st.image(random_img_path, caption="Randomly Sampled Image")
                            
                            with c2:
                                if st.button("Randomize Image Path", key=f"randomize_{dataset}"):
                                    st.rerun()

                                rotation_option = st.radio(
                                    "Choose Rotation:", 
                                    ["None", "CW", "CCW", "180"],
                                    key=f"rotate_images_radio_{dataset}",
                                    label_visibility="collapsed"
                                )
                            
                            with c3:
                                image = Image.open(random_img_path)
                                if rotation_option == "CW":
                                    rotated_image = image.rotate(-90, expand=True)
                                elif rotation_option == "CCW":
                                    rotated_image = image.rotate(90, expand=True)
                                elif rotation_option == "180":
                                    rotated_image = image.rotate(180, expand=True)
                                else:
                                    rotated_image = image
                                st.image(rotated_image, caption=f"Rotated Image ({rotation_option})")
                            
                            datasets_list.append({
                                "directory": dataset.replace("\ ", " "),
                                "rotation": rotation_option
                            })
                        else:
                            st.warning(f"No valid image found in dataset: {os.path.basename(dataset)}")
                else:
                    st.error("Please select a valid image dataset directory.")
                
        with st.expander("Virtual Environment Path"):
            st.write("The path to the virtual environment to run the script in. This contains all python packages needed to run the script.")
            path_navigator("venv_path", radio_button_prefix="rotate_images")
        
        with st.expander("Script"):
            path_navigator("rotate_images_script_path")
            python_code_editor("rotate_images_script_path")
        
        with st.expander("Rotate Images"):
            st.write("Press 'Begin Rotating Images' to perform the desired action on the images.")
            output = None
            c1, c2, c3, c4 = st.columns(4, gap="small")
            
            with c1:
                if st.button("▶ Begin Rotating Images", key="begin_rotating_data_btn"):
                    # Filter the datasets_list to include only entries with a rotation other than "None"
                    filtered_datasets = [entry for entry in datasets_list if entry["rotation"] != "None"]

                    if filtered_datasets:
                        # Write the filtered datasets list to a temporary JSON file.
                        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tf:
                            json.dump(filtered_datasets, tf)
                            temp_json_path = tf.name
                        
                        # Build command-line arguments by passing the JSON file path.
                        # Use shlex.quote to ensure the path is safely escaped.
                        cmd_args = f'--json_file {shlex.quote(temp_json_path)}'
                        
                        # Launch the tmux command with the --json_file argument.
                        run_in_tmux(
                            session_key="rotate_images", 
                            script_path=st.session_state.paths["rotate_images_script_path"], 
                            venv_path=st.session_state.paths["venv_path"],
                            args=cmd_args
                        )
                        time.sleep(3)
                        output = update_tmux_terminal("rotate_images")
                    else:
                        st.warning("No rotations required as all settings are None.")
            
            with c2:
                if st.button("🔄 Refresh Terminal", key="check_rotate_images_btn"):
                    output = update_tmux_terminal("rotate_images")
            
            with c3:
                if st.button("🧹 Clear Output", key="rotate_images_clear_terminal_btn"):
                    output = None
                    st.text("Terminal output cleared.")
            
            with c4:
                if st.button("❌ Kill Session", key="rotate_images_kill_tmux_session_btn"):
                    output = kill_tmux_session("rotate_images")
                    st.text(output)

    elif action_option == "Auto Label":
        with st.expander("Auto Label Settings"):
            st.subheader("Model Weights Path")
            st.write("The path to the model weights.")
            path_navigator("auto_label_model_weight_path")

            st.subheader("Images Path")
            st.write("The path to the images.")
            path_navigator("auto_label_data_path")
            
            # Check whether the provided images path is a directory.
            images_path = st.session_state.paths.get("auto_label_data_path", "")
            if os.path.isdir(images_path):
                # Build a dictionary with counts of images per directory.
                dir_image_counts = {}
                base_dir = os.path.abspath(images_path)
                for root, dirs, files in os.walk(images_path):
                    count = sum(1 for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg')))
                    if count > 0:
                        rel_dir = os.path.relpath(root, base_dir)
                        # If the relative directory is just ".", show "Base Directory".
                        key_name = "Base Directory" if rel_dir == "." else rel_dir
                        dir_image_counts[key_name] = count

                if not dir_image_counts:
                    st.warning("No images (PNG/JPG/JPEG) found in the selected directory.")
                else:
                    # Build a sorted list of options like "FolderName (N images)"
                    options = sorted([f"{k} ({v} images)" for k, v in dir_image_counts.items()])
                    st.selectbox("Found Image Directories", options=options,
                                key="found_autolabel_images", label_visibility="collapsed")
            else:
                st.write("**Single image file selected** (if this is not intended, please choose a directory).")

            st.subheader("Label Save Path Replacement")
            st.write("Instead of specifying a full path, enter the folder name that will replace the last folder in the images path. For example, use 'labels' to automatically replace the 'images' folder.")
            # Provide a radio button to choose between the default replacement and a custom value.
            key = "auto_label_replacement"
            replacement_option = st.radio("Choose label replacement option:", ["Default", "Custom"],
                                        key="autolabel_replacement_radio", label_visibility="collapsed")
            if replacement_option == "Default":
                st.session_state.paths[key] = "labels"
            else:
                st.session_state.paths[key] = st.text_input("Enter custom label replacement", value="labels")
            st.write(f"**Label Replacement:** {st.session_state.paths[key]}")

        with st.expander("Virtual Environment Path"):
            st.subheader("Venv Path")
            st.write("The path to the virtual environment to run the script in. This contains all python packages needed to run the script.")
            path_navigator("venv_path", radio_button_prefix="auto_label")

        with st.expander("Script"):
            path_navigator("auto_label_script_path")
            python_code_editor("auto_label_script_path")

        with st.expander("Auto Label Data"):
            st.write("Click the 'Begin Auto Labeling Data' button to start the auto-labeling process.")
            output = None
            c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")

            with c1:
                output = check_gpu_status("auto_label_check_gpu_status_button")

            with c2:
                try:
                    if st.session_state.gpu_list:
                        selected_gpu = st.selectbox("Select GPU", options=list(range(len(st.session_state.gpu_list))),
                                                    format_func=lambda x: f"GPU {x}", label_visibility="collapsed")
                        st.session_state.auto_label_gpu = int(selected_gpu)
                        st.write(f"Selected GPU: GPU {st.session_state.auto_label_gpu}")
                    else:
                        st.warning("No GPUs found, defaulting to CPU")
                        st.session_state.auto_label_gpu = -1
                except Exception as e:
                    st.error(f"Error checking GPUs: {e}")
                    st.session_state.auto_label_gpu = -1

            with c3:
                if st.button("▶ Begin Auto Labeling Data", key="begin_auto_labeling_data_btn"):
                    run_in_tmux(
                        session_key="auto_label_data", 
                        script_path=st.session_state.paths["auto_label_script_path"], 
                        venv_path=st.session_state.paths["venv_path"],
                        args={
                            "model_weights_path": st.session_state.paths["auto_label_model_weight_path"].replace(" ", "\ "),
                            "images_dir_path": st.session_state.paths["auto_label_data_path"].replace(" ", "\ "),
                            "label_replacement": st.session_state.paths["auto_label_replacement"].replace(" ", "\ "),
                            "gpu_number": st.session_state.auto_label_gpu
                        }
                    )
                    time.sleep(3)
                    output = update_tmux_terminal("auto_label_data")

            with c4:
                if st.button("🔄 Refresh Terminal", key="check_auto_labeling_data_btn"):
                    output = update_tmux_terminal("auto_label_data")

            with c5:
                if st.button("🧹 Clear Output", key="auto_labeling_clear_terminal_btn"):
                    output = None

            with c6:
                if st.button("❌ Kill Session", key="auto_labeling_kill_tmux_session_btn"):
                    output = kill_tmux_session("auto_label_data")

            terminal_output = st.empty()
            if output is not None:
                terminal_output.text(output)

    elif action_option == "Generate Labeled Video":
        # — defaults
        st.session_state.paths.setdefault(
            "gen_vid_input_path", 
            st.session_state.paths["unverified_images_path"]
        )
        default_out = os.path.join(
            st.session_state.paths["gen_vid_input_path"],
            "videos_with_labels"
        )
        st.session_state.paths.setdefault("gen_vid_output_path", default_out)
        st.session_state.paths.setdefault("gen_vid_script_path", "generate_video.py")

        # — SETTINGS —
        with st.expander("Settings"):
            st.subheader("Input Path")
            st.write("Parent folder containing one or more subdirs with `images/` + `labels/`.")
            path_navigator("gen_vid_input_path", radio_button_prefix="gen_vid")

            input_root = st.session_state.paths.get("gen_vid_input_path", "")
            valid_dirs = []
            if os.path.isdir(input_root):
                for entry in sorted(os.listdir(input_root)):
                    imgs = os.path.join(input_root, entry, "images")
                    lbls = os.path.join(input_root, entry, "labels")
                    if os.path.isdir(imgs) and os.path.isdir(lbls):
                        valid_dirs.append(entry)

            st.selectbox(
                "Found subfolders with both `images/` and `labels/` (selection does nothing)",
                options=valid_dirs,
                index=0 if valid_dirs else None,
            )

            st.subheader("FPS")
            st.session_state.paths["gen_vid_fps"] = st.number_input(
                "Frames per second", 
                min_value=1.0, 
                value=float(st.session_state.paths.get("gen_vid_fps", 5.0)),
                step=1.0,
                key="gen_vid_fps_input"    
            )

            st.subheader("Mode")
            st.session_state.paths["gen_vid_mode"] = st.radio(
                "Which video(s) to generate?",
                options=["Frame by Frame", "Object by Object", "Both"],
                index=["Frame by Frame","Object by Object","Both"]
                    .index(st.session_state.paths.get("gen_vid_mode","Both")),
                key="gen_vid_mode_radio"
            )

        # — VIRTUAL ENV —
        with st.expander("Virtual Environment"):
            st.write("Path to the venv containing Python + dependencies.")
            path_navigator("venv_path", radio_button_prefix="gen_vid")

        # — SCRIPT —
        with st.expander("Script"):
            st.write("Your `generate_video.py` (must accept --input_path, --output_path, --fps, --mode).")
            path_navigator("gen_vid_script_path", radio_button_prefix="gen_vid")
            python_code_editor("gen_vid_script_path")

        # — ACTIONS —
        with st.expander("Generate Videos"):
            c1, c2, c3, c4 = st.columns(4, gap="small")
            with c1:
                
                if st.button("▶ Start Generating", key="begin_gen_vid_btn"):
                    output = run_in_tmux(
                        session_key="gen_vid",
                        script_path=st.session_state.paths["gen_vid_script_path"],
                        venv_path=st.session_state.paths["venv_path"],
                        args={
                            "input_path":    st.session_state.paths["gen_vid_input_path"].replace(" ", "\ "),
                            "fps":           st.session_state.paths["gen_vid_fps"],
                            "mode":          st.session_state.paths["gen_vid_mode"].replace(" ", "\ ")
                        },
                        script_type="python"
                    )
            with c2:
                if st.button("🔄 Refresh Terminal", key="update_gen_vid_btn"):
                    output = update_tmux_terminal("gen_vid")
            with c3:
                if st.button("🧹 Clear Output", key="clear_gen_vid_btn"):
                    output = None
            with c4:
                if st.button("❌ Kill Session", key="kill_gen_vid_btn"):
                    output = kill_tmux_session("gen_vid")
        
    elif action_option == "Video Review":

        with st.expander("Settings"):
            
            st.write("### Images Path")
            st.write("The path to the images.")
            path_navigator("unverified_images_path", radio_button_prefix="video_review")

            if st.session_state.paths["prev_unverified_images_path"] != st.session_state.paths["unverified_images_path"] or st.session_state.paths["prev_unverified_names_yaml_path"] != st.session_state.paths["unverified_names_yaml_path"]:
                st.session_state.paths["prev_unverified_images_path"] = st.session_state.paths["unverified_images_path"]
                st.session_state.paths["prev_unverified_names_yaml_path"] = st.session_state.paths["unverified_names_yaml_path"]
                update_unverified_data_path()

                if st.session_state.max_images > 0:
                    update_unverified_frame()
                    
                st.rerun()
            
        with st.expander("Subset Selection"):
            st.write("Select only a small subset of images to review or manually label.")
            if handle_image_list_update(prefix="subset_"):
                # --- CSV Path Selection ---
                st.subheader("Choose CSV Path for Subset")
                
                path_option = st.radio("Please Choose:", ["Default", "Custom"], key="default_subset", label_visibility="collapsed")
                if path_option == "Default":
                    st.session_state.paths["unverified_subset_csv_path"] = os.path.join(st.session_state.paths["unverified_images_path"], "subset.csv")
                else:
                    path_navigator("unverified_subset_csv_path", radio_button_prefix="video_review")

                csv_file = st.session_state.paths["unverified_subset_csv_path"]
                if os.path.exists(csv_file):
                    # Reload the subset frames from the CSV file
                    st.session_state.subset_frames = load_subset_frames(csv_file)

                    st.subheader("Modify/View Subset")

                    # Add/Remove Frames
                    if st.session_state.max_images > 0:
                        c1, c2, c3 = st.columns([10, 10, 10])
                        with c1:
                            st.number_input(
                                "Add Frame Index",
                                min_value=0,
                                max_value=st.session_state.max_images - 1,
                                value=None,
                                step=1,
                                key="video_review_subset_add_frame",
                                on_change=add_frame_callback,
                                args=("subset_add_frame",)
                            )

                        with c2:
                            st.selectbox("View Frames in Subset (Selection Does Nothing)", st.session_state.subset_frames, key="video_review_subset_view_frames_in_subset")

                        with c3:
                            st.number_input(
                                "Remove Frame Index",
                                min_value=0,
                                max_value=st.session_state.max_images - 1,
                                value=None,
                                step=1,
                                key="video_review_subset_remove_frame",
                                on_change=remove_frame_callback,
                                args=("subset_remove_frame",)
                            )
                    else:
                        st.warning("No images available.")

                    # --- Copy CSV to a New File ---
                    base, ext = os.path.splitext(csv_file)
                    default_copy_path = base + "_copy" + ext
                    new_save_path = st.text_input("Enter path for new CSV copy", value=default_copy_path)
                    if st.button("Copy CSV to new file"):
                        if new_save_path:
                            try:
                                st.session_state["skip_label_update"] = True
                                save_subset_csv(new_save_path, st.session_state.subset_frames)
                                st.success(f"Subset CSV copied to {new_save_path}")
                            except Exception as e:
                                st.error(f"Error copying file: {e}")
                        else:
                            st.error("Please enter a valid new file path.")
                
                    # ---  Bulk Operations ---
                    st.markdown("---")
                    st.subheader("Bulk Subset Operations")

                    if st.button("Clear Subset Selection", key="video_review_clear_subset_selection", help="Clears all frames from the subset selection list."):
                        st.session_state.subset_frames = []
                        save_subset_csv(csv_file, st.session_state.subset_frames)
                        st.success("Cleared all frames from subset selection.")
                        st.rerun()

                    if st.button("Add ALL Labeled Frames", help="Adds every frame containing at least one label to the subset."):
                        st.session_state["skip_label_update"] = True
                        labeled = [i for i in range(st.session_state.max_images) if _frame_label_count(i)>0]
                        for i in labeled:
                            if i not in st.session_state.subset_frames:
                                st.session_state.subset_frames.append(i)
                        save_subset_csv(csv_file, st.session_state.subset_frames)
                        st.success(f"Added {len(labeled)} labeled frames.")
                
                    if st.button("Remove ALL Labeled Frames", help="Removes every frame containing labels from the subset."):
                        st.session_state["skip_label_update"] = True
                        before = set(st.session_state.subset_frames)
                        st.session_state.subset_frames = [i for i in st.session_state.subset_frames if _frame_label_count(i)==0]
                        save_subset_csv(csv_file, st.session_state.subset_frames)
                        removed = len(before) - len(st.session_state.subset_frames)
                        st.success(f"Removed {removed} frames.")

                    if st.button("Add ALL Unlabeled Frames", help="Adds every frame with no labels to the subset."):
                        st.session_state["skip_label_update"] = True
                        unlabeled = [i for i in range(st.session_state.max_images) if _frame_label_count(i)==0]
                        for i in unlabeled:
                            if i not in st.session_state.subset_frames:
                                st.session_state.subset_frames.append(i)
                        save_subset_csv(csv_file, st.session_state.subset_frames)
                        st.success(f"Added {len(unlabeled)} unlabeled frames.")
                    
                    if st.button("Remove ALL Unlabeled Frames", help="Removes every frame with no labels from the subset."):
                        st.session_state["skip_label_update"] = True
                        before = set(st.session_state.subset_frames)
                        st.session_state.subset_frames = [i for i in st.session_state.subset_frames if _frame_label_count(i)>0]
                        save_subset_csv(csv_file, st.session_state.subset_frames)
                        removed = len(before) - len(st.session_state.subset_frames)
                        st.success(f"Removed {removed} frames.")

                    if st.button("Invert Subset Selection", help="Selects all frames not currently in the subset and deselects those that are."):
                        st.session_state["skip_label_update"] = True
                        # all frame indices 0 … max_images-1
                        all_frames = list(range(st.session_state.max_images))
                        # keep only those not already in subset
                        new_subset = [i for i in all_frames if i not in st.session_state.subset_frames]
                        st.session_state.subset_frames = new_subset
                        # persist to CSV
                        save_subset_csv(csv_file, st.session_state.subset_frames)
                        st.success(f"Inverted subset: {len(new_subset)} frames selected.")
                        st.rerun()

                    if st.button("Delete ALL Labels in Subset", help="Deletes all label files for the currently selected subset frames."):
                        st.session_state["skip_label_update"] = True
                        for idx in st.session_state.subset_frames:
                            # construct label path
                            if st.session_state.image_pattern:
                                img = os.path.join(
                                    st.session_state.images_dir,
                                    st.session_state.image_pattern.format(idx)
                                )
                            else:
                                img = st.session_state.image_list[idx]
                            lbl = img.replace("/images/", "/labels/").rsplit(".",1)[0] + ".txt"
                            open(lbl, "w").close()
                        st.success("Cleared all label files for frames in subset.")

                    st.markdown("---")
                    # --- Save subset to directory ---
                    st.subheader("Save Subset to New Directory")
                    save_dir = path_navigator("subset_save_path", radio_button_prefix="video_review_save_subset")
                    if st.button("Save images & labels here"):
                        st.session_state["skip_label_update"] = True
                        out_imgs = os.path.join(save_dir, "images")
                        out_lbls = os.path.join(save_dir, "labels")
                        os.makedirs(out_imgs, exist_ok=True, mode=0o777)
                        os.makedirs(out_lbls, exist_ok=True, mode=0o777)

                        for idx in st.session_state.subset_frames:
                            # resolve source image
                            if st.session_state.image_pattern:
                                src = os.path.join(
                                    st.session_state.images_dir,
                                    st.session_state.image_pattern.format(idx)
                                )
                            else:
                                src = st.session_state.image_list[idx]
                            # copy image
                            shutil.copy2(src, os.path.join(out_imgs, os.path.basename(src)))
                            # copy or touch label
                            lbl = src.replace("/images/", "/labels/").rsplit(".",1)[0] + ".txt"
                            dst_lbl = os.path.join(out_lbls, os.path.basename(lbl))
                            if os.path.exists(lbl):
                                shutil.copy2(lbl, dst_lbl)
                            else:
                                open(dst_lbl, "w").close()
                        st.success(f"Saved {len(st.session_state.subset_frames)} frames to `{save_dir}`")
                
                else:
                    st.info("No CSV found. Create or upload a CSV to begin using a subset.")

        with st.expander("Video Label Review"):
            update_unverified_data_path()
            if handle_image_list_update(prefix="video_"):
                video_base_path = st.session_state.paths["unverified_images_path"].replace("/images", "/videos_with_labels")
                if os.path.exists(video_base_path):
                        
                    video_paths = os.listdir(video_base_path)

                    for i in range(len(video_paths) + 1):
                        video_path = video_paths[i-1]

                        if i > 0:
                            if "frame_by_frame" in video_path or "object_by_object" in video_path:
                                st.video(os.path.join(video_base_path, video_path), start_time=0, format="video/mp4")

                        # Add/Remove Frames (single)
                        if st.session_state.max_images > 0:
                                c1, c2, c3 = st.columns([10, 10, 10])
                                with c1:
                                    st.number_input(
                                        "Add Frame Index",
                                        min_value=0,
                                        max_value=st.session_state.max_images - 1,
                                        value=None,
                                        step=1,
                                        key=f"{i}_video_subset_add_frame",
                                        on_change=add_frame_callback,
                                        args=(f"{i}_video_subset_add_frame",),
                                        label_visibility="visible"     
                                    )
                                with c2:
                                    st.selectbox(
                                        "View Frames in Subset (selection does nothing)",
                                        options=sorted(st.session_state.subset_frames),
                                        key=f"{i}_video_subset_view_frames_in_subset",
                                        label_visibility="visible"     
                                    )
                                with c3:
                                    st.number_input(
                                        "Remove Frame Index",
                                        min_value=0,
                                        max_value=st.session_state.max_images - 1,
                                        value=None,
                                        step=1,
                                        key=f"{i}_video_subset_remove_frame",
                                        on_change=remove_frame_callback,
                                        args=(f"{i}_video_subset_remove_frame",),
                                        label_visibility="visible"     
                                    )

                                # Add/Remove Frames (range)
                                max_idx = st.session_state.max_images - 1
                                r1, r2, r3, r4 = st.columns([8, 8, 4, 4])
                                with r1:
                                    range_start = st.number_input(
                                        "Range Start",
                                        min_value=0,
                                        max_value=max_idx,
                                        value=0,
                                        key=f"{i}_video_subset_range_start",
                                        label_visibility="visible"     
                                    )
                                with r2:
                                    range_end = st.number_input(
                                        "Range End",
                                        min_value=0,
                                        max_value=max_idx,
                                        value=max_idx,
                                        key=f"{i}_video_subset_range_end",
                                        label_visibility="visible"     
                                    )
                                with r3:
                                    if st.button("Add Range", key=f"{i}_add_range_btn"):
                                        lo, hi = sorted((range_start, range_end))
                                        for i in range(lo, hi + 1):
                                            if i not in st.session_state.subset_frames:
                                                st.session_state.subset_frames.append(i)
                                        save_subset_csv(csv_file, st.session_state.subset_frames)
                                        st.success(f"Added frames {lo}–{hi}")
                                with r4:
                                    if st.button("Remove Range", key=f"{i}_remove_range_btn"):
                                        lo, hi = sorted((range_start, range_end))
                                        before = set(st.session_state.subset_frames)
                                        st.session_state.subset_frames = [
                                            i for i in st.session_state.subset_frames if not (lo <= i <= hi)
                                        ]
                                        save_subset_csv(csv_file, st.session_state.subset_frames)
                                        removed = len(before) - len(st.session_state.subset_frames)
                                        st.success(f"Removed {removed} frames")
                
                else:
                    st.warning("No videos generated please go to Generate Data Tab.")

    elif action_option == "Object by Object Review":

        with st.expander("Settings"):
            
            st.write("### Images Path")
            st.write("The path to the images.")
            path_navigator("unverified_images_path", radio_button_prefix="object_")

            if st.session_state.paths["prev_unverified_images_path"] != st.session_state.paths["unverified_images_path"] or st.session_state.paths["prev_unverified_names_yaml_path"] != st.session_state.paths["unverified_names_yaml_path"]:
                st.session_state.paths["prev_unverified_images_path"] = st.session_state.paths["unverified_images_path"]
                st.session_state.paths["prev_unverified_names_yaml_path"] = st.session_state.paths["unverified_names_yaml_path"]
                update_unverified_data_path()

                if st.session_state.max_images > 0:
                    update_unverified_frame()
                    
                st.rerun()
            
        with st.expander("Object by Object Label Review"):
            st.write( "Review the labels in an object by object sequence.")

            # Call the helper function to ensure image list/naming pattern is up-to-date.
            if handle_image_list_update(prefix="object_by_object_"):
                object_running = st.session_state.get("object_running", False)

                current_obj = get_object_by_global_index(st.session_state.global_object_index)
                if current_obj is None:
                    st.session_state.global_object_index = 0
                    current_obj = get_object_by_global_index(st.session_state.global_object_index)
                    if current_obj is None:
                        st.info("No objects found in the dataset.")
                    else:
                        st.rerun()
                else:
                    img = current_obj["img"]
                    bbox = current_obj["bbox"]
                    obj_label = current_obj["label"]
                    image_path = current_obj["image_path"]
                    label_path = current_obj["label_path"]
                    st.session_state.global_object_count = current_obj["num_labels"]


                    x, y, w, h = bbox
                    center_x = x + w / 2
                    center_y = y + h / 2

                    mode = st.radio(
                        "Display Mode",
                        options=["Cropped Object", "Full Image with BBox"],
                        key=f"object_display_mode_{current_obj['global_index']}",
                        disabled=object_running
                    )

                    if mode == "Cropped Object":
                        display_img = img.crop((x, y, x + w, y + h))
                        caption = f"{os.path.basename(image_path)} | Object {current_obj['global_index']}: Label {obj_label} (Cropped)"
                    else:
                        full_img = img.copy()
                        draw = ImageDraw.Draw(full_img)
                        draw.rectangle((x, y, x + w, y + h), outline="red", width=2)
                        display_img = full_img
                        caption = f"{os.path.basename(image_path)} | Object {current_obj['global_index']}: Label {obj_label} (Full Image)"

                    col_img, col_ctrl = st.columns(2)
                    with col_img:
                        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                        st.image(display_img, caption=caption, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    with col_ctrl:
                        st.number_input(
                            f"Center X for object {current_obj['global_index']}",
                            min_value=0.0,
                            max_value=float(img.width),
                            value=center_x,
                            key=f"object_{current_obj['global_index']}_center_x",
                            step=1.0,
                            on_change=object_by_object_edit_callback,
                            disabled=object_running
                        )
                        st.number_input(
                            f"Center Y for object {current_obj['global_index']}",
                            min_value=0.0,
                            max_value=float(img.height),
                            value=center_y,
                            key=f"object_{current_obj['global_index']}_center_y",
                            step=1.0,
                            on_change=object_by_object_edit_callback,
                            disabled=object_running
                        )
                        st.number_input(
                            f"Width for object {current_obj['global_index']}",
                            min_value=1.0,
                            max_value=float(img.width),
                            value=w,
                            key=f"object_{current_obj['global_index']}_w",
                            step=1.0,
                            on_change=object_by_object_edit_callback,
                            disabled=object_running
                        )
                        st.number_input(
                            f"Height for object {current_obj['global_index']}",
                            min_value=1.0,
                            max_value=float(img.height),
                            value=h,
                            key=f"object_{current_obj['global_index']}_h",
                            step=1.0,
                            on_change=object_by_object_edit_callback,
                            disabled=object_running
                        )

                        col_input1, col_input2 = st.columns(2)
                        with col_input1:
                            num_labels = current_obj["num_labels"]
                            new_global_index = st.number_input(
                                f"Set Global Object Index (0-{num_labels-1})",
                                min_value=0,
                                value=st.session_state.global_object_index,
                                key="global_obj_index_input",
                                disabled=object_running
                            )
                            if new_global_index != st.session_state.global_object_index:
                                st.session_state.global_object_index = int(new_global_index)
                                st.rerun()

                        with col_input2:

                            try:
                                jump_frame = st.number_input(
                                    f"Jump to Frame Number (0-{st.session_state.max_images-1})",
                                    min_value=0,
                                    value=st.session_state.frame_index,
                                    max_value=st.session_state.max_images,
                                    key="jump_to_frame_input",
                                    on_change=jump_frame_object_by_object_callback,
                                    disabled=object_running
                                )
                            except:
                                pass

                            if st.session_state.object_by_object_jump_warning is None:     
                                if st.session_state.object_by_object_jump_valid:
                                    st.session_state.object_by_object_jump_valid = False                         
                                    st.rerun()
                            else: 
                                st.warning(st.session_state.object_by_object_jump_warning)
                                st.session_state.object_by_object_jump_warning = None  # Reset flag.

                        col_nav1, col_nav2, col_nav3 = st.columns(3)
                        with col_nav1:
                            if st.button("Previous Object", key="prev_global_obj", disabled=object_running):
                                if st.session_state.global_object_index - 1 < 0:
                                    st.session_state.global_object_index = current_obj["num_labels"] - 1
                                else:
                                    st.session_state.global_object_index -= 1
                                st.rerun()
                        with col_nav2:
                            if st.button("Next Object", key="next_global_obj", disabled=object_running):
                                if st.session_state.global_object_index >= current_obj["num_labels"]:
                                    st.session_state.global_object_index = 0
                                else:
                                    st.session_state.global_object_index += 1
                                st.rerun()
                        with col_nav3:
                            if st.button("Delete Object", key="delete_global_obj", disabled=object_running):
                                try:
                                    with open(label_path, "r") as f:
                                        lines = f.readlines()
                                    local_idx = current_obj["local_index"]
                                    if local_idx < len(lines):
                                        del lines[local_idx]
                                        with open(label_path, "w") as f:
                                            f.writelines(lines)
                                        st.success("Object deleted.")
                                    else:
                                        st.error("Local object index out of range in label file.")
                                except Exception as e:
                                    st.error(f"Error deleting object: {e}")
                                st.rerun()

                        # Reference selector (no default)
                    reference_indices = st.multiselect(
                        "Select reference object(s) for clustering",
                        options=list(range(st.session_state.global_object_count)),
                        default=[],
                        key="cluster_refs",
                        disabled=object_running
                    )

        with st.expander("Clustered Objects"):
            # 1) Compute cluster.csv path
            images_dir = st.session_state.paths["unverified_images_path"]
            cluster_csv_path = os.path.join(os.path.dirname(images_dir), "cluster.csv")

            # 2) Load existing predictions (if any), handling empty files
            if os.path.exists(cluster_csv_path):
                
                try:
                    # If the file is empty, this will raise EmptyDataError
                    df = pd.read_csv(cluster_csv_path, header=None)
                    preds = df[0].tolist()
                except pd.errors.EmptyDataError:
                    preds = []  # treat empty file as no predictions

                st.session_state["predicted_indices"] = preds
                if preds:
                    with st.spinner("Loading clustered objects…"):
                        to_remove = render_checkbox_grid(preds, thumb_width=150, cols=4)

                    if st.button("Remove Selected Objects from Cluster", key="cluster_remove_selected"):
                        remaining = [i for i in preds if i not in to_remove]
                        # Update session state and CSV
                        st.session_state.predicted_indices = remaining
                        import pandas as pd
                        cluster_csv_path = os.path.join(os.path.dirname(st.session_state.paths["unverified_images_path"]), "cluster.csv")
                        pd.DataFrame(remaining).to_csv(cluster_csv_path, index=False, header=False)
                        st.success(f"Removed {len(to_remove)} objects; {len(remaining)} remain.")
                        st.rerun()

                    if st.button("Delete Unchecked Objects in Cluster from Labels", key="cluster_delete_unchecked"):
                        indices = [i for i in st.session_state.predicted_indices.copy() if i not in to_remove]
                        file_map = {}
                        file_map = {}
                        for idx in indices:
                            obj = get_object_by_global_index(idx)
                            if obj is None:
                                continue
                            path = obj["label_path"]
                            file_map.setdefault(path, []).append(obj["local_index"])

                        # remove lines in each label file
                        for path, local_indices in file_map.items():
                            try:
                                with open(path, "r") as f:
                                    lines = f.readlines()
                                for li in sorted(set(local_indices), reverse=True):
                                    if 0 <= li < len(lines):
                                        del lines[li]
                                with open(path, "w") as f:
                                    f.writelines(lines)
                            except Exception as e:
                                st.error(f"Error updating {path}: {e}")
                        # clear cluster.csv and UI state
                        pd.DataFrame([]).to_csv(cluster_csv_path, index=False, header=False)
                        st.session_state.predicted_indices = []
                        st.success("Deleted all unchecked clustered objects from their label files and cleared cluster.csv.")
                        st.rerun()

                    if st.button("Delete All Clustered Objects From Labels", key="cluster_delete_all"):
                        indices = st.session_state.predicted_indices.copy()
                        file_map = {}
                        for idx in indices:
                            obj = get_object_by_global_index(idx)
                            if obj is None:
                                continue
                            path = obj["label_path"]
                            file_map.setdefault(path, []).append(obj["local_index"])
                        # remove lines in each label file
                        for path, local_indices in file_map.items():
                            try:
                                with open(path, "r") as f:
                                    lines = f.readlines()
                                for li in sorted(set(local_indices), reverse=True):
                                    if 0 <= li < len(lines):
                                        del lines[li]
                                with open(path, "w") as f:
                                    f.writelines(lines)
                            except Exception as e:
                                st.error(f"Error updating {path}: {e}")
                        # clear cluster.csv and UI state
                        pd.DataFrame([]).to_csv(cluster_csv_path, index=False, header=False)
                        st.session_state.predicted_indices = []
                        st.success("Deleted all clustered objects from their label files and cleared cluster.csv.")
                        st.rerun()

                else:
                    st.info("cluster.csv is present but empty.")
            else:
                st.info("No cluster.csv yet—run “Begin Clustering” to create it.")

        with st.expander("Cluster Object Settings"):

            st.subheader("Virtual Environment")
            path_navigator("venv_path", radio_button_prefix="cluster_")

            st.subheader("Script")
            path_navigator("cluster_script_path")
            python_code_editor("cluster_script_path")

        with st.expander("Cluster Objects"):
        
            st.session_state.cluster_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7, 0.1)
            
            # 2) Show tmux controls for running the clustering script
            c0, c1, c2 = st.columns([1,1,4], gap="small")
            with c0:
                output = check_gpu_status("cluster_check_gpu")
            with c1:
                gpu = st.selectbox(
                    "GPU", 
                    options=list(range(len(st.session_state.gpu_list))),
                    format_func=lambda x: f"GPU {x}",
                    key="cluster_gpu_select"
                )

            with c2:
                # Begin / Refresh / Clear / Kill buttons
                btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4, gap="small")
                with btn_col1:
                    if st.button("▶ Begin Clustering", key="cluster_begin"):
                        if reference_indices is not None:
                            # Save index list to CSV (no header, one index per line)
                            pd.DataFrame(reference_indices).to_csv(
                                cluster_csv_path,
                                index=False,
                                header=False
                            )

                        run_in_tmux(
                            session_key="cluster",
                            script_path=st.session_state.paths["cluster_script_path"],
                            venv_path=st.session_state.paths["venv_path"],
                            args={
                                "images_dir": st.session_state.paths["unverified_images_path"].replace(" ", "\\ "),
                                "cluster_csv": cluster_csv_path.replace(" ", "\\ "),
                                "threshold": st.session_state.cluster_threshold,
                                "gpu": st.session_state.cluster_gpu_select
                            }
                        )

                        time.sleep(3)
                        output = update_tmux_terminal("cluster")

                with btn_col2:
                    if st.button("🔄 Refresh", key="cluster_refresh"):
                        output = update_tmux_terminal("cluster")
                with btn_col3:
                    if st.button("🧹 Clear", key="cluster_clear"):
                        output = None
                with btn_col4:
                    if st.button("❌ Kill", key="cluster_kill"):
                        output = kill_tmux_session("cluster")

    elif action_option == "Frame by Frame Review":
            
        with st.expander("Settings"):
            
            st.write("### Images Path")
            st.write("The path to the images.")
            path_navigator("unverified_images_path", radio_button_prefix="frame_")

            st.write("### Label Names YAML Path")
            st.write("The path to the YAML file containing the label names. To edit in the window, add the changes and click the apply button")
            path_navigator("unverified_names_yaml_path", radio_button_prefix="frame_")
            
            st.write("### Image Scale")
            st.write("Scale the image to fit the screen. This is useful for large images.")
            st.number_input(
                "Image Scale", 
                value=st.session_state.unverified_image_scale,
                step=0.25,
                key="frame_unverified_image_scale_input",
                on_change=set_scale
            )

            if st.session_state.paths["prev_unverified_images_path"] != st.session_state.paths["unverified_images_path"] or st.session_state.paths["prev_unverified_names_yaml_path"] != st.session_state.paths["unverified_names_yaml_path"]:
                st.session_state.paths["prev_unverified_images_path"] = st.session_state.paths["unverified_images_path"]
                st.session_state.paths["prev_unverified_names_yaml_path"] = st.session_state.paths["unverified_names_yaml_path"]
                update_unverified_data_path()

                if st.session_state.max_images > 0:
                    update_unverified_frame()
                    
                st.rerun()
            
            yaml_editor("unverified_names_yaml_path")

        with st.expander("Subset Selection"):
            st.write("Select only a small subset of images to review or manually label.")
            if handle_image_list_update(prefix="subset_"):
                # --- CSV Path Selection ---
                st.subheader("Choose CSV Path for Subset")
                
                path_option = st.radio("Please Choose:", ["Default", "Custom"], key="frame_default_subset", label_visibility="collapsed")
                if path_option == "Default":
                    st.session_state.paths["unverified_subset_csv_path"] = os.path.join(st.session_state.paths["unverified_images_path"], "subset.csv")
                else:
                    path_navigator("unverified_subset_csv_path", radio_button_prefix="frame_")

                csv_file = st.session_state.paths["unverified_subset_csv_path"]
                if os.path.exists(csv_file):
                    # Reload the subset frames from the CSV file
                    st.session_state.subset_frames = load_subset_frames(csv_file)

                    st.subheader("Modify/View Subset")

                    # Add/Remove Frames
                    if st.session_state.max_images > 0:
                        c1, c2, c3 = st.columns([10, 10, 10])
                        with c1:
                            st.number_input(
                                "Add Frame Index",
                                min_value=0,
                                max_value=st.session_state.max_images - 1,
                                value=None,
                                step=1,
                                key="frame_subset_add_frame",
                                on_change=add_frame_callback,
                                args=("subset_add_frame",)
                            )

                        with c2:
                            st.selectbox("View Frames in Subset (Selection Does Nothing)", st.session_state.subset_frames, key="frame_subset_view_frames_in_subset")

                        with c3:
                            st.number_input(
                                "Remove Frame Index",
                                min_value=0,
                                max_value=st.session_state.max_images - 1,
                                value=None,
                                step=1,
                                key="frame_subset_remove_frame",
                                on_change=remove_frame_callback,
                                args=("subset_remove_frame",)
                            )
                    else:
                        st.warning("No images available.")

                    # --- Copy CSV to a New File ---
                    base, ext = os.path.splitext(csv_file)
                    default_copy_path = base + "_copy" + ext
                    new_save_path = st.text_input("Enter path for new CSV copy", value=default_copy_path)
                    if st.button("Copy CSV to new file"):
                        if new_save_path:
                            try:
                                st.session_state["skip_label_update"] = True
                                save_subset_csv(new_save_path, st.session_state.subset_frames)
                                st.success(f"Subset CSV copied to {new_save_path}")
                            except Exception as e:
                                st.error(f"Error copying file: {e}")
                        else:
                            st.error("Please enter a valid new file path.")
                
                    # ---  Bulk Operations ---
                    st.markdown("---")
                    st.subheader("Bulk Subset Operations")

                    if st.button("Clear Subset Selection", key="frame_clear_subset_selection", help="Clears all frames from the subset selection list."):
                        st.session_state.subset_frames = []
                        save_subset_csv(csv_file, st.session_state.subset_frames)
                        st.success("Cleared all frames from subset selection.")
                        st.rerun()

                    if st.button("Add ALL Labeled Frames", help="Adds every frame containing at least one label to the subset."):
                        st.session_state["skip_label_update"] = True
                        labeled = [i for i in range(st.session_state.max_images) if _frame_label_count(i)>0]
                        for i in labeled:
                            if i not in st.session_state.subset_frames:
                                st.session_state.subset_frames.append(i)
                        save_subset_csv(csv_file, st.session_state.subset_frames)
                        st.success(f"Added {len(labeled)} labeled frames.")
                
                    if st.button("Remove ALL Labeled Frames", help="Removes every frame containing labels from the subset."):
                        st.session_state["skip_label_update"] = True
                        before = set(st.session_state.subset_frames)
                        st.session_state.subset_frames = [i for i in st.session_state.subset_frames if _frame_label_count(i)==0]
                        save_subset_csv(csv_file, st.session_state.subset_frames)
                        removed = len(before) - len(st.session_state.subset_frames)
                        st.success(f"Removed {removed} frames.")

                    if st.button("Add ALL Unlabeled Frames", help="Adds every frame with no labels to the subset."):
                        st.session_state["skip_label_update"] = True
                        unlabeled = [i for i in range(st.session_state.max_images) if _frame_label_count(i)==0]
                        for i in unlabeled:
                            if i not in st.session_state.subset_frames:
                                st.session_state.subset_frames.append(i)
                        save_subset_csv(csv_file, st.session_state.subset_frames)
                        st.success(f"Added {len(unlabeled)} unlabeled frames.")
                    
                    if st.button("Remove ALL Unlabeled Frames", help="Removes every frame with no labels from the subset."):
                        st.session_state["skip_label_update"] = True
                        before = set(st.session_state.subset_frames)
                        st.session_state.subset_frames = [i for i in st.session_state.subset_frames if _frame_label_count(i)>0]
                        save_subset_csv(csv_file, st.session_state.subset_frames)
                        removed = len(before) - len(st.session_state.subset_frames)
                        st.success(f"Removed {removed} frames.")

                    if st.button("Invert Subset Selection", help="Selects all frames not currently in the subset and deselects those that are."):
                        st.session_state["skip_label_update"] = True
                        # all frame indices 0 … max_images-1
                        all_frames = list(range(st.session_state.max_images))
                        # keep only those not already in subset
                        new_subset = [i for i in all_frames if i not in st.session_state.subset_frames]
                        st.session_state.subset_frames = new_subset
                        # persist to CSV
                        save_subset_csv(csv_file, st.session_state.subset_frames)
                        st.success(f"Inverted subset: {len(new_subset)} frames selected.")
                        st.rerun()

                    if st.button("Delete ALL Labels in Subset", help="Deletes all label files for the currently selected subset frames."):
                        st.session_state["skip_label_update"] = True
                        for idx in st.session_state.subset_frames:
                            # construct label path
                            if st.session_state.image_pattern:
                                img = os.path.join(
                                    st.session_state.images_dir,
                                    st.session_state.image_pattern.format(idx)
                                )
                            else:
                                img = st.session_state.image_list[idx]
                            lbl = img.replace("/images/", "/labels/").rsplit(".",1)[0] + ".txt"
                            open(lbl, "w").close()
                        st.success("Cleared all label files for frames in subset.")

                    st.markdown("---")
                    # --- Save subset to directory ---
                    st.subheader("Save Subset to New Directory")
                    save_dir = path_navigator("subset_save_path", radio_button_prefix="frame_save_subset")
                    if st.button("Save images & labels here"):
                        st.session_state["skip_label_update"] = True
                        out_imgs = os.path.join(save_dir, "images")
                        out_lbls = os.path.join(save_dir, "labels")
                        os.makedirs(out_imgs, exist_ok=True, mode=0o777)
                        os.makedirs(out_lbls, exist_ok=True, mode=0o777)

                        for idx in st.session_state.subset_frames:
                            # resolve source image
                            if st.session_state.image_pattern:
                                src = os.path.join(
                                    st.session_state.images_dir,
                                    st.session_state.image_pattern.format(idx)
                                )
                            else:
                                src = st.session_state.image_list[idx]
                            # copy image
                            shutil.copy2(src, os.path.join(out_imgs, os.path.basename(src)))
                            # copy or touch label
                            lbl = src.replace("/images/", "/labels/").rsplit(".",1)[0] + ".txt"
                            dst_lbl = os.path.join(out_lbls, os.path.basename(lbl))
                            if os.path.exists(lbl):
                                shutil.copy2(lbl, dst_lbl)
                            else:
                                open(dst_lbl, "w").close()
                        st.success(f"Saved {len(st.session_state.subset_frames)} frames to `{save_dir}`")
                
                else:
                    st.info("No CSV found. Create or upload a CSV to begin using a subset.")

        with st.expander("Frame by Frame Label Review"):
            st.write( "Review the labels in a frame by frame sequence.")
            if handle_image_list_update(prefix="frame_by_frame_"):
                loading = st.session_state.get("detection_running", False)

                if st.session_state.max_images > 0:
                    if st.session_state.max_images > 1:
                        # --- Top Navigation (Prev / Next) ---
                        col_prev, _, col_next = st.columns([4, 5, 4])
                        with col_prev:
                            st.button("Prev Frame", key="top_prev_btn", on_click=prev_callback, disabled=loading)
                        with col_next:
                            st.button("Next Frame", key="top_next_btn", on_click=next_callback, disabled=loading)

                        col_copy_prev, _, col_copy_next = st.columns([4, 5, 4])
                        with col_copy_prev:
                            st.button("Copy Labels from Prev Slide", key="copy_prev_btn", on_click=copy_prev_labels, disabled=loading)
                        with col_copy_next:
                            st.button("Copy Labels from Next Slide", key="copy_next_btn", on_click=copy_next_labels, disabled=loading)

                    # --- Read and Display Current Frame ---
                    update_unverified_frame()

                    st.write(f"Current File Path: {st.session_state.image_path}")

                    # Annotate with detection()
                    det_container = st.container()
                    with det_container:
                        if not loading:
                            with st.spinner("Processing..."):
                                out = detection(**st.session_state.detection_config)
                        else:
                            out = detection(**st.session_state.detection_config)

                    st.session_state.out = out

                    # Update labels if changed
                    update_labels_from_detection()

                    c1, c2 = st.columns([10, 90])
                    with c1:
                        st.markdown(
                            """
                            <style>
                            .stCheckbox input[type="checkbox"] {
                                transform: scale(1.5);
                            }
                            .stCheckbox label {
                                font-size: 24px;
                                font-weight: bold;
                            }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )
                        use_subset_val = st.checkbox(
                            "Use Subset",
                            value=st.session_state.use_subset,
                            key="manual_label_subset_btn",
                            on_change=manual_label_subset_checkbox_callback,
                            disabled=not len(st.session_state.subset_frames) > 1
                        )
                        if st.session_state.use_subset_changed:
                            st.session_state.use_subset_changed = False
                            st.rerun()
       
                    with c2:
                        c12, c22, c32 = st.columns([10, 10, 10])
                        with c12:
                            st.number_input(
                                "Add Frame Index",
                                min_value=0,
                                max_value=st.session_state.max_images - 1,
                                value=None,
                                step=1,
                                key="frame_by_frame_add_frame",
                                on_change=add_frame_callback,
                                args=("frame_by_frame_add_frame",)
                            )

                        with c22:
                            st.selectbox("View Frames in Subset (Selection Does Nothing)", st.session_state.subset_frames, key="frame_by_frame_view_frames_in_subset")

                        with c32:
                            st.number_input(
                                "Remove Frame Index",
                                min_value=0,
                                max_value=st.session_state.max_images - 1,
                                value=None,
                                step=1,
                                key="frame_by_frame_remove_frame",
                                on_change=remove_frame_callback,
                                args=("frame_by_frame_remove_frame",)
                            )

                    if not len(st.session_state.subset_frames) > 1:
                            st.warning("Subset needs to be two or larger.")
                    
                    if st.session_state.max_images > 1:
                        # Additional navigation (jump, slider, second Prev/Next)
                        st.number_input(
                            "Current Frame",
                            min_value=0,
                            max_value=st.session_state.max_images-1,
                            value=st.session_state.frame_index,
                            step=10,
                            key="jump_page",
                            on_change=jump_frame_frame_by_frame_callback
                        )
                        col_prev, col_slider, col_next = st.columns([2, 10, 4])
                        with col_prev:
                            st.button("Prev Frame", key="prev_btn", on_click=prev_callback)
                        with col_slider:
                            st.slider(
                                f"Subset Index: {st.session_state.frame_index}  Frame Index: {st.session_state.actual_frame_index}"
                                if st.session_state.use_subset else f"Frame Index: {st.session_state.actual_frame_index}",
                                0,
                                st.session_state.max_images - 1 if not st.session_state.use_subset
                                else len(st.session_state.subset_frames) - 1,
                                st.session_state.frame_index,
                                key="slider_det",
                                on_change=frame_slider_frame_by_frame_callback,
                                label_visibility="collapsed"
                            )
                        with col_next:
                            st.button("Next Frame", key="next_btn", on_click=next_callback)
                else:
                    st.warning("Data Path is empty...")
        
            else:
                if st.button("Refresh", key="refresh_empty"):
                    update_unverified_data_path()
                    st.rerun()
        
        with st.expander("Zoomed‑in Bounding Box Regions"):
            # Safely pull values (defaults to empty or None)
            image    = st.session_state.get("image", None)
            bboxes   = st.session_state.get("bboxes_xyxy", [])
            labels   = st.session_state.get("labels", [])
            bbox_ids = st.session_state.get("bbox_ids", [])

            # If there's no image or no boxes, show a message
            if image is None or not bboxes:
                st.warning("No bounding boxes in current frame.")
            else:
                # Update frame if needed
                update_unverified_frame()

                # Render each crop + controls
                for i, (x, y, w, h) in enumerate(bboxes):
                    # Compute coordinates
                    x1, y1 = max(x, 0.0), max(y, 0.0)
                    x2, y2 = x1 + w,      y1 + h

                    # Sanity check
                    if x2 <= x1 or y2 <= y1:
                        st.session_state.labels.pop(i)
                        st.session_state.bboxes_xyxy.pop(i)
                        st.session_state.bbox_ids.pop(i)
                        st.rerun()

                    # Crop & caption
                    cropped = image.crop((x1, y1, x2, y2))
                    if i < len(labels):
                        idx = labels[i]
                        try:
                            label_name = st.session_state.label_list[idx]
                        except Exception:
                            label_name = f"Label {idx}"
                    else:
                        label_name = "Unknown"
                    caption = (
                        f"ID: {bbox_ids[i]}, "
                        f"Label: {label_name}, "
                        f"BBox: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
                    )

                    key_x = f"bbox_{bbox_ids[i]}_cx"
                    key_y = f"bbox_{bbox_ids[i]}_cy"
                    key_w = f"bbox_{bbox_ids[i]}_w"
                    key_h = f"bbox_{bbox_ids[i]}_h"

                    st.markdown(f"#### Edit Bounding Box {i} Parameters")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(cropped, caption=caption)

                    # Default center coords
                    center_x = x + w / 2
                    center_y = y + h / 2

                    with col2:
                        st.number_input(
                            f"Center X for bbox {i}",
                            min_value=0.0,
                            max_value=float(st.session_state.image_width),
                            value=center_x,
                            key=key_x,
                            step=1.0,
                            on_change=lambda i=i: zoom_edit_callback(i),
                        )
                        st.number_input(
                            f"Center Y for bbox {i}",
                            min_value=0.0,
                            max_value=float(st.session_state.image_height),
                            value=center_y,
                            key=key_y,
                            step=1.0,
                            on_change=lambda i=i: zoom_edit_callback(i),
                        )
                        st.number_input(
                            f"Width for bbox {i}",
                            min_value=1.0,
                            max_value=float(st.session_state.image_width),
                            value=w,
                            key=key_w,
                            step=1.0,
                            on_change=lambda i=i: zoom_edit_callback(i),
                        )
                        st.number_input(
                            f"Height for bbox {i}",
                            min_value=1.0,
                            max_value=float(st.session_state.image_height),
                            value=h,
                            key=key_h,
                            step=1.0,
                            on_change=lambda i=i: zoom_edit_callback(i),
                    )
                        
    elif action_option == "Move Directory":
        # SETTINGS
        with st.expander("Move Directory Settings"):
            st.subheader("Source Directory")
            path_navigator("move_src_path")
            st.subheader("Destination Directory")
            path_navigator("move_dest_path", must_exist=False)

            # Swap paths button
            if st.button("🔄 Swap Source and Destination", key="swap_move_paths_btn"):
                src = st.session_state.paths.get("move_src_path", "")
                dst = st.session_state.paths.get("move_dest_path", "")
                st.session_state.paths["move_src_path"], st.session_state.paths["move_dest_path"] = dst, src
                st.rerun()


        # VENV
        with st.expander("Virtual Environment Path"):
            st.write("Path to the virtual environment for the move script.")
            path_navigator("venv_path", radio_button_prefix="move_dir")

        # SCRIPT
        with st.expander("Move Script"):
            st.write("Your `move_dir.py` (must accept `--src_dir` and `--dst_dir`).")
            path_navigator("move_dir_script_path")
            python_code_editor("move_dir_script_path")

        # ACTIONS
        with st.expander("Execute Move"):
            c1, c2, c3, c4 = st.columns(4, gap="small")
            with c1:
                if st.button("▶ Begin Move", key="begin_move_dir_btn"):
                    run_in_tmux(
                        session_key="move_dir",
                        script_path=st.session_state.paths["move_dir_script_path"],
                        venv_path=st.session_state.paths["venv_path"],
                        args={
                            "src_dir": st.session_state.paths["move_src_path"].replace(" ", "\\ "),
                            "dst_dir": st.session_state.paths["move_dest_path"].replace(" ", "\\ "),
                        },
                        script_type="python"
                    )
                    time.sleep(3)
                    output = update_tmux_terminal("move_dir")

            with c2:
                if st.button("🔄 Refresh Terminal", key="refresh_move_dir_btn"):
                    output = update_tmux_terminal("move_dir")
            with c3:
                if st.button("🧹 Clear Output", key="clear_move_dir_btn"):
                    output = None
            with c4:
                if st.button("❌ Kill Session", key="kill_move_dir_btn"):
                    output = kill_tmux_session("move_dir")
          
    elif action_option == "Split YOLO Dataset into Objects / No Objects":
        with st.expander("Dataset Settings"):
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Dataset To Be Split")
                st.write("The path to the dataset to be split.")
                path_navigator("split_data_path")
            with c2:
                st.subheader("Save Path")
                st.write("The path to save to on the server.")
                save_path_option = st.radio("Choose save path option:", ["Default", "Custom"], key=f"split_save_radio", label_visibility="collapsed")
                key = "split_data_save_path"
                if save_path_option == "Default":
                    st.session_state.paths[key] = st.session_state.paths["split_data_path"]
                    st.write(f"**Current {' '.join(word.capitalize() for word in key.split('_'))}:** {st.session_state.paths[key]}")
                else:
                    path_navigator(key)

        with st.expander("Virtual Environment Path"):
            st.write("The path to the virtual environment to run the script in. This contains all python packages needed to run the script.")
            path_navigator("venv_path", radio_button_prefix="split_data")

        with st.expander("Script"):
            path_navigator("split_data_script_path")
            python_code_editor("split_data_script_path")

        with st.expander("Split Data"):
            st.write("Press 'Begin Splitting Data' to split the images in the dataset into a group of images with objects and those without.")
            output = None
            c1, c2, c3, c4 = st.columns(4, gap="small")

            with c1:
                if st.button("▶ Begin Splitting Data", key="begin_split_data_btn"):
                    run_in_tmux(
                        session_key="split_data", 
                        script_path=st.session_state.paths["split_data_script_path"], 
                        venv_path=st.session_state.paths["venv_path"],
                        args={
                            "data_path" : st.session_state.paths["split_data_path"].replace(" ", "\ "),
                            "save_path" : st.session_state.paths["split_data_save_path"].replace(" ", "\ "),
                        }
                    )
                    time.sleep(3)
                    output = update_tmux_terminal("split_data")

            with c2:
                if st.button("🔄 Refresh Terminal", key="check_split_data_btn"):
                    output = update_tmux_terminal("split_data")

            with c3:
                if st.button("🧹 Clear Output", key="split_data_clear_terminal_btn"):
                    output = None

            with c4:
                if st.button("❌ Kill Session", key="split_data_kill_tmux_session_btn"):
                    output = kill_tmux_session("split_data")
    
    elif action_option == "Combine YOLO Datasets":
        # Combine YOLO Datasets
        with st.expander("Dataset Settings"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("Dataset 1")
                st.write("The path to the first dataset.")
                path_navigator("combine_dataset_1_path")
            with c2:
                st.subheader("Dataset 2")
                st.write("The path to the second dataset.")
                path_navigator("combine_dataset_2_path")
            with c3:
                st.subheader("Save Path")
                st.write("The path to save to on the server.")
                path_navigator("combine_dataset_save_path")

        with st.expander("Virtual Environment Path"):
            path_navigator("venv_path", radio_button_prefix="combine_data")

        with st.expander("Script"):
            path_navigator("combine_dataset_script_path")
            python_code_editor("combine_dataset_script_path")

        with st.expander("Combine Data"):
            st.write("Press 'Begin Combining Data' to combine the two datasets into one.")
            output = None
            c1, c2, c3, c4 = st.columns(4, gap="small")

            with c1:
                if st.button("▶ Begin Combining Data", key="begin_combine_dataset_btn"):
                    run_in_tmux(
                        session_key="combine_dataset", 
                        script_path=st.session_state.paths["combine_dataset_script_path"], 
                        venv_path=st.session_state.paths["venv_path"],
                        args={
                            "dataset1" : st.session_state.paths["combine_dataset_1_path"].replace(" ", "\ "),
                            "dataset2" : st.session_state.paths["combine_dataset_2_path"].replace(" ", "\ "),
                            "dst_dir" : st.session_state.paths["combine_dataset_save_path"].replace(" ", "\ ")
                        }
                    )
                    time.sleep(3)
                    output = update_tmux_terminal("combine_dataset")

            with c2:
                if st.button("🔄 Refresh Terminal", key="check_combine_dataset_btn"):
                    output = update_tmux_terminal("combine_dataset")

            with c3:
                if st.button("🧹 Clear Output", key="combine_dataset_clear_terminal_btn"):
                    output = None

            with c4:
                if st.button("❌ Kill Session", key="combine_dataset_kill_tmux_session_btn"):
                    output = kill_tmux_session("combine_dataset")
    
    elif action_option == "Daset Statistics":
        with st.expander("Dataset Statistics"):
            st.subheader("Dataset Figures")
            st.write("The path to the formated dataset (with images and labels folders).")

            path_navigator("dataset_path")

            dataset_path = st.session_state.paths.get("dataset_path", "")
            if dataset_path:
                images_path = os.path.join(dataset_path, "images")
                labels_path = os.path.join(dataset_path, "labels")

                # Two columns for generating figures and clearing buttons
                c1, c2 = st.columns([1,1])
                with c1:
                    if st.button("Generate Figures", key="generate_figs_btn"):
                        st.session_state["show_figs"] = True
                with c2:
                    if st.button("Clear Figures", key="clear_figs_btn"):
                        st.session_state["show_figs"] = False

                # Show figs only if toggled on
                if st.session_state.get("show_figs", False):
                    class_counts, x_vals, y_vals, w_vals, h_vals = parse_label_data(labels_path)

                    st.markdown("### Sample Labeled Images")
                    display_images(images_path, labels_path)

                    st.markdown("### Bounding Box Location, Size, and Class Distribution")
                    st.write("These figures will help see where the bounding boxes are commonly located"
                    " on the images (0.0 y-axis is the top of the image), the size, and the distribution"
                    " of the classes (unlabeled are the number of images that do not have any labels).")
                    generate_label_plots(class_counts, x_vals, y_vals, w_vals, h_vals)

                else:
                    st.info("Figures are cleared. Click 'Generate Figures' to display them.")

    else:
        with st.expander("Data YAML"):
            st.write("The path to the data YAML file. This file contains the paths to the train, test, and validation datasets as well as the class names.")
            path_navigator("train_data_yaml_path")
            yaml_editor("train_data_yaml_path")
        
        with st.expander("Model YAML"):
            st.write("The path to the model YAML file. This file contains the model architecture and layers.")
            path_navigator("train_model_yaml_path")
            yaml_editor("train_model_yaml_path")
            
        with st.expander("Train YAML Path"):
            st.write("The path to the train YAML file. This file contains all model hyperparameters.")
            path_navigator("train_train_yaml_path")
            yaml_editor("train_train_yaml_path")

        with st.expander("Venv Path"):
            st.write("The path to the virtual environment to run the script in. This contains all python packages needed to run the script.")
            path_navigator("venv_path", radio_button_prefix="train")

        with st.expander("Script"):
            path_navigator("train_script_path")
            python_code_editor("train_script_path")

        with st.expander("Finetune Model"):
            st.write("Click the 'Begin Training' button to start the training process.")
            output = None
            c1, c2, c3, c4, c5 = st.columns(5, gap="small")

            with c1:
                output = check_gpu_status("train_check_gpu_status_button")

            with c2:
                if st.button("▶ Begin Training", key="begin_train_btn"):
                    output = run_in_tmux(
                        session_key="auto_label_trainer", 
                        script_path=st.session_state.paths["train_script_path"], 
                        venv_path=st.session_state.paths["venv_path"],
                        args={
                            "data_path": st.session_state.paths["train_data_yaml_path"].replace(" ", "\ "),
                            "model_path": st.session_state.paths["train_model_yaml_path"].replace(" ", "\ "),
                            "train_path" : st.session_state.paths["train_train_yaml_path"].replace(" ", "\ ")
                        }
                    )
                    time.sleep(3)
                    output = update_tmux_terminal("auto_label_trainer")

            with c3:
                if st.button("Check Training", key="check_train_btn"):
                    output = update_tmux_terminal("auto_label_trainer")

            with c4:
                if st.button("🧹 Clear Output", key="clear_terminal_btn"):
                    output = None

            with c5:
                if st.button("❌ Kill Session", key="kill_tmux_session_btn"):
                    output = kill_tmux_session("auto_label_trainer")

            terminal_output = st.empty()
            if output is not None:
                terminal_output.text(output)

    terminal_output = st.empty()
    if output is not None:
        display_terminal_output(output)

# ----------------------- Linux Terminal Tab -----------------------
with tabs[1]:
    # Initialize accumulated terminal output in session state.
    if "terminal_text" not in st.session_state:
        st.session_state.terminal_text = ""
    
    # Container for the text input (at the top)
    text_input_container = st.container()
    # Single placeholder for the accumulated output
    output_placeholder = st.empty()

    def run_command_and_accumulate(command):
        """
        Executes a shell command and appends its output to st.session_state.terminal_text,
        updating output_placeholder in real time in a single code block.
        """
        try:
            process = subprocess.Popen(
                command, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1
            )
            for line in process.stdout:
                st.session_state.terminal_text += line
                output_placeholder.code(st.session_state.terminal_text, language="bash")
            for line in process.stderr:
                st.session_state.terminal_text += line
                output_placeholder.code(st.session_state.terminal_text, language="bash")
        except Exception as e:
            st.session_state.terminal_text += f"Error executing command: {e}\n"
            output_placeholder.code(st.session_state.terminal_text, language="bash")

    def local_run_callback():
        command = st.session_state.command_input
        if command.strip():
            # Clear previous output on each new enter, then add the command prompt.
            st.session_state.terminal_text = f"$ {command}\n"
            output_placeholder.code(st.session_state.terminal_text, language="bash")
            run_command_and_accumulate(command)
        else:
            output_placeholder.warning("Please enter a valid command.")

    # Render the text input at the top.
    with text_input_container:
        st.text_input("Enter a Linux command:", "", key="command_input", on_change=local_run_callback)

    # Display the accumulated terminal output
    output_placeholder.code(st.session_state.terminal_text, language="bash")

# ----------------------- Unrestrict Tab -----------------------
with tabs[2]:
    st.subheader("Open Workspace")
    st.write("Pick a folder to keep with fully unrestricted (777) permissions on every rerun.")
    path_navigator("open_workspace", must_exist=True)

    
    ow = st.session_state.paths["open_workspace"]


    enable_option = st.radio(
        "Choose option:",
        ["Disable", "Enable"],
        key=f"enable_chmod_radio",
        label_visibility="collapsed",
    )

    if enable_option == "Enable":
        if ow:
            st.success(f"📂 Will keep `{ow}` at 777 perms on each rerun")
        else:
            st.info("No open workspace set yet.")

        ow = st.session_state.paths.get("open_workspace")
        if ow and os.path.exists(ow):
            # kill any previous session to avoid duplicates
            subprocess.call("tmux kill-session -t chmod_free_workspace 2>/dev/null", shell=True)
            # spawn a new background tmux session that does the chmod
            subprocess.call(
                f"tmux new-session -d -s chmod_free_workspace \"chmod -R 777 '{ow}'\"",
                shell=True
            )
    else:
        if ow and os.path.exists(ow):
            st.info(f"Currently disabled. Workspace is set to `{ow}`. Once enabled, permissions will be set to 777 on each rerun.")
        else:
            st.info("Currently disabled and no valid open workspace set yet.")