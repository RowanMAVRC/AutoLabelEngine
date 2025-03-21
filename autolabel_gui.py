import os
import re
import yaml
import streamlit as st
import subprocess
import glob
import zipfile
from PIL import Image
from streamlit_label_kit import detection
# from streamlit_img_label.manage import ImageManager, ImageDirManager
from streamlit_ace import st_ace
from pathlib import Path
import time
import numpy as np
from moviepy.editor import ImageSequenceClip, VideoClip, clips_array
from PIL import Image, ImageDraw, ImageFont
import shutil
import hashlib


## Functions
#-------------------------------------------------------------------------------------------------------------------------#

# List of keys to save and load from session_state
SELECTED_KEYS = [
    "auto_label_gpu",
    "copy_path_auto_label_script_path",
    "copy_path_train_script_path",
    "copy_path_unverified_names_yaml_path",
    "data_cfg",
    "frame_index",
    "gpu_list",
    "images_dir",
    "label_list",
    "paths",
    "python_codes",
    "yamls",
    "unverified_image_scale"
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
    try:
        state_to_save = {key: st.session_state[key] for key in SELECTED_KEYS if key in st.session_state}
        with open(default_yaml_path, "w") as f:
            yaml.dump(state_to_save, f)
    except Exception as e:
        st.error(f"Error saving session state: {e}")

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

def run_command(command):
    output = []
    terminal_output = st.empty()

    with subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
    ) as process:
        for line in process.stdout:
            output.append(line)
            # Replace carriage returns with newlines in the joined output.
            terminal_output.code("".join(output).replace('\r', '\n'), language="bash")
        for line in process.stderr:
            output.append(line)
            terminal_output.code("".join(output).replace('\r', '\n'), language="bash")
    
    return "".join(output).replace('\r', '\n')

def run_callback():
    command = st.session_state.command_input
    if command.strip():
        st.markdown(f"<div class='terminal'><strong>Running:</strong> {command}</div>", unsafe_allow_html=True)
        run_command(command)
    else:
        st.warning("Please enter a valid command.")

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
        type=["txt", "csv", "jpg", "png", "pdf", "py", "yaml", "zip", "mp4"],
        key=uploader_key
    )
    
    if uploaded_file is not None:
        # Ensure the save directory exists.
        os.makedirs(save_dir, exist_ok=True)
        
        # If the file is a ZIP archive, extract it.
        if uploaded_file.name.endswith(".zip"):
            temp_zip_path = os.path.join(save_dir, uploaded_file.name)
            with open(temp_zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
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

        # Reset the file uploader state
        if uploader_key in st.session_state:
            del st.session_state[uploader_key]  # Properly reset uploader key
    
        st.rerun()  # Refresh the UI

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

def run_in_tmux(session_key, py_file_path, venv_path, args=""):
    """
    Opens a new tmux session with the given session_key, activates the virtual environment from venv_path,
    runs the specified Python file with additional arguments, captures its terminal output, displays it in Streamlit,
    and then kills the session.

    If a tmux session with the same session_key already exists, it will be killed first.

    Args:
        session_key (str): Unique key used as the tmux session name.
        py_file_path (str): Path to the Python (.py) file to be executed.
        venv_path (str): Path to the virtual environment directory.
        args (str or dict): Additional command-line arguments to be appended to the Python command.
                            If a dictionary is provided, it will be converted to a string of command-line arguments.
    
    Returns:
        str or None: The decoded terminal output if successful; otherwise, None.
    """

    # If a session with the given key already exists, kill it.
    try:
        subprocess.check_call(f"tmux kill-session -t {session_key}", shell=True)
    except subprocess.CalledProcessError:
        # If there's no such session, ignore the error.
        pass

    # Check if the Python file exists
    if not os.path.exists(py_file_path):
        st.error(f"Python file not found: {py_file_path}")
        return None

    # Check if the virtual environment activation script exists
    activate_script = os.path.join(venv_path, "bin", "activate")
    if not os.path.exists(activate_script):
        st.error(f"Virtual environment activation script not found: {activate_script}")
        return None

    # If args is provided as a dictionary, convert it to a string of command-line arguments.
    if isinstance(args, dict):
        args = " ".join(f"--{key} {value}" for key, value in args.items())

    # Build the inner command that activates the virtual environment, runs the Python script,
    # and then executes bash to keep the tmux session open.
    inner_command = f"source {activate_script} && python {py_file_path} {args}; exec bash"
    
    # Build the complete tmux command using bash -c to handle quoting correctly.
    tmux_cmd = f'tmux new-session -d -s {session_key} "bash -c \'{inner_command}\'"'
    
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

def check_gpu_status(button_key):
    if st.button("Check GPU Status", key=button_key):
        try:
            # Run the gpustat command and capture its output
            output = subprocess.check_output(["gpustat"]).decode("utf-8")
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
    
    st.markdown("Edit YAML content")

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
    
    st.markdown("Edit Python Code")

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

def path_navigator(key, radio_button_prefix="", button_and_selectbox_display_size=[2, 25]):
    """
    A file/directory navigator that can operate in two modes:
    1) Default: Navigate through directories with a selectbox and a ".." button.
    2) Custom: Manually enter a path in a text input field.

    If a chosen path doesn't exist, you'll be prompted to either create it
    or go up directories until you find one that exists.
    """

    # Retrieve or set initial path in session state
    current_path = st.session_state.paths.get(key, "/")
    current_path = os.path.normpath(current_path)

    # Allow user to choose "Default" navigation or "Enter Path as Text" path
    save_path_option = st.radio("Choose save path option:", ["File Explorer", "Enter Path as Text"], key=f"{radio_button_prefix}_{key}_radio", label_visibility="collapsed")

    if save_path_option == "Enter Path as Text":
        # -- CUSTOM PATH MODE --
        # Now default to the current path in the text input
        custom_path = st.text_input(
            "Enter custom save path:",
            value=current_path,  # <--- Prefills with current path
            key=f"{radio_button_prefix}_{key}_custom_path_input",
            label_visibility="collapsed"
        )

        st.write(f"**Current {' '.join(word.capitalize() for word in key.split('_'))}:** {current_path}")

        if custom_path:
            custom_path = os.path.normpath(custom_path)
            if not os.path.exists(custom_path):
                # Path doesn't exist; ask user how to proceed
                st.warning(f"Path '{custom_path}' does not exist. Choose an option below:")
                create_col, up_col = st.columns(2)

                with create_col:
                    if st.button("Create this path", key=f"{radio_button_prefix}_{key}_create_custom"):
                        # Prompt user for a final directory or filename to create
                        new_name = st.text_input(
                            "Optionally enter a different name for the new path:",
                            value=custom_path,
                            key=f"{radio_button_prefix}_{key}_new_path_name"
                        )
                        if new_name:
                            try:
                                os.makedirs(new_name, exist_ok=True)
                                st.session_state.paths[key] = new_name
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to create directory: {e}")
                                return custom_path

                with up_col:
                    if st.button("Go up until path exists", key=f"{radio_button_prefix}_{key}_go_up_custom"):
                        temp_path = custom_path
                        while not os.path.exists(temp_path) and temp_path not in ("/", ""):
                            temp_path = os.path.dirname(temp_path)

                        if not os.path.exists(temp_path):
                            st.error("No valid parent directory found.")
                            return custom_path
                        else:
                            st.session_state.paths[key] = temp_path
                            st.rerun()

                return custom_path
            else:
                # Path exists, store in session and proceed
                st.session_state.paths[key] = custom_path
                return custom_path
        else:
            # If user hasn't typed a path yet, just return whatever was stored
            return st.session_state.paths.get(key, "/")

    else:
        # -- DEFAULT NAVIGATION MODE --

        if os.path.isfile(current_path):
            directory_to_list = os.path.dirname(current_path)
        else:
            directory_to_list = current_path

        col1, col2 = st.columns(button_and_selectbox_display_size, gap="small")

        # ".." Button to go up
        with col1:
            go_up_button_key = f"go_up_button_{radio_button_prefix}_{key}"
            if st.button("..", key=go_up_button_key):
                if os.path.isdir(current_path):
                    parent = os.path.dirname(current_path)
                else:
                    parent = os.path.dirname(os.path.dirname(current_path))
                parent = os.path.normpath(parent)
                st.session_state.paths[key] = parent
                st.rerun()

        # Attempt to list directory
        if not os.path.exists(directory_to_list):
            st.warning(f"Path '{directory_to_list}' does not exist. Choose an option below:")
            create_col, up_col = st.columns(2)

            with create_col:
                if st.button("Create this path", key=f"{radio_button_prefix}_{key}_create_default"):
                    # Ask for a final directory name to create
                    new_name = st.text_input(
                        "Optionally enter a different name for the new path:",
                        value=directory_to_list,
                        key=f"{radio_button_prefix}_{key}_new_default_path_name"
                    )
                    if new_name:
                        try:
                            os.makedirs(new_name, exist_ok=True)
                            st.session_state.paths[key] = new_name
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to create directory: {e}")
                            return current_path

            with up_col:
                if st.button("Go up until path exists", key=f"{radio_button_prefix}_{key}_go_up_default"):
                    temp_path = directory_to_list
                    while not os.path.exists(temp_path) and temp_path not in ("/", ""):
                        temp_path = os.path.dirname(temp_path)

                    if not os.path.exists(temp_path):
                        st.error("No valid parent directory found.")
                        return current_path
                    else:
                        st.session_state.paths[key] = temp_path
                        st.rerun()

            return current_path

        try:
            entries = os.listdir(directory_to_list)
        except Exception as e:
            st.error(f"Error reading directory: {e}")
            return current_path

        # Build the selectbox options
        options_list = []
        options_mapping = {}
        indent = "└"
        top_label = directory_to_list
        options_list.append(top_label)
        options_mapping[top_label] = None

        for entry in entries:
            full_path = os.path.join(directory_to_list, entry)
            full_path = os.path.normpath(full_path)
            label = f"{indent} {entry}"
            options_list.append(label)
            options_mapping[label] = full_path

        # Determine which item to highlight
        default_index = 0
        for i, lbl in enumerate(options_list):
            if lbl == top_label:
                continue
            mapped_path = options_mapping[lbl]
            if mapped_path and os.path.normpath(mapped_path) == os.path.normpath(current_path):
                default_index = i
                break

        widget_key = f"navigator_select_{radio_button_prefix}_{key}"

        def on_selectbox_change():
            selected_label = st.session_state[widget_key]
            new_path = options_mapping[selected_label]
            if new_path is not None:
                st.session_state.paths[key] = new_path

        with col2:
            st.selectbox(
                "Select a subdirectory or file:",
                options_list,
                index=default_index,
                key=widget_key,
                on_change=on_selectbox_change,
                label_visibility="collapsed"
            )

        st.write(f"**Current {' '.join(word.capitalize() for word in key.split('_'))}:** {current_path}")

        return current_path

def list_files(directory, extension):
    return [f for f in os.listdir(directory) if f.endswith(extension)]

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

def update_labels():
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

def update_unverified_frame():

    # Ensure frame index is feasible
    st.session_state.frame_index = st.session_state.max_images - 1 if st.session_state.frame_index < 0 else st.session_state.frame_index
    st.session_state.frame_index = 0 if st.session_state.frame_index > st.session_state.max_images - 1 else st.session_state.frame_index

    # Get Image Path
    images_dir = st.session_state.images_dir
    if st.session_state.image_pattern:
        image_path = os.path.join(images_dir, st.session_state.image_pattern.format(st.session_state.frame_index))
    elif "image_list" in st.session_state and st.session_state.image_list:
        try:
            image_path = st.session_state.image_list[st.session_state.frame_index]
        except IndexError:
            st.error("Frame index out of range for image list.")
            return
    else:
        st.error("No image naming pattern or image list set.")
        return

    # Open Image
    image = Image.open(image_path)
    image_width, image_height = image.size

    # Get Labels
    labels_dir = images_dir.replace("images", "labels")
    label_path = image_path.replace("images", "labels").replace("jpg", "txt").replace("png", "txt")

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

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
                bbox_xyxy = [x_center_abs - w_abs / 2, y_center_abs - h_abs / 2, w_abs, h_abs]
                bboxes_xyxy.append(bbox_xyxy)
                labels.append(cls)
    else:
        with open(label_path, "w") as f:
            f.write("")

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

    # Map out-of-range labels to a valid index and store the display-friendly mapping
    display_labels = known_labels.copy()  # Start with known labels
    unknown_label_map = {}

    updated_labels = []
    for label in labels:
        if label < len(known_labels):
            updated_labels.append(label)
        else:
            # Ensure a unique entry for each unknown label
            if label not in unknown_label_map:
                unknown_label_map[label] = len(display_labels)
                display_labels.append(f"Unknown: {label}")
            updated_labels.append(unknown_label_map[label])  # Use mapped index

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
        "item_editor": True,
        "item_editor_position": "right",
        "edit_description": False,
        "edit_meta": False,
        "item_selector": True,
        "item_selector_position": "right",
        "bbox_format": "XYWH",
        "bbox_show_info": True,
        "key": "detector"
    }

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
    if pattern_info is None:
        # Set a flag instead of immediately showing a warning.
        st.session_state.naming_pattern_warning = "Could not infer an image naming pattern or sequential numeric sequence found."
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

def zoom_edit_callback(i):
    # Try to get the old bounding box. If the index is out of range,
    # display a warning and exit the callback.
    try:
        old_bbox = st.session_state.bboxes_xyxy[i]
    except IndexError:
        st.warning(f"Bounding box index {i} is out of range. Skipping update.")
        return
    
    # Read the new values from session_state using the unique keys.
    new_center_x = st.session_state[f"bbox_{i}_center_x_input"]
    # Here we assume the UI is showing the flipped center y (0 at bottom).
    flipped_center_y = st.session_state[f"bbox_{i}_center_y_input"]
    image_height = st.session_state.image_height
    # Convert flipped center y to actual center y (standard: 0 at top).
    actual_center_y = image_height - flipped_center_y

    new_w = st.session_state[f"bbox_{i}_w_input"]
    new_h = st.session_state[f"bbox_{i}_h_input"]

    # Calculate new top-left x and y based on the actual center and new dimensions.
    new_x = new_center_x - new_w / 2
    new_y = actual_center_y - new_h / 2

    # Clamp the bounding box to the image boundaries.
    if new_x < 0:
        new_x = 0.0
    if new_y < 0:
        new_y = 0.0
    if new_x + new_w > st.session_state.image_width:
        new_x = st.session_state.image_width - new_w
    if new_y + new_h > image_height:
        new_y = image_height - new_h

    # If any parameter changed, update the bbox and write changes to the label file.
    if (new_x, new_y, new_w, new_h) != tuple(old_bbox):
        st.session_state.bboxes_xyxy[i] = [new_x, new_y, new_w, new_h]

        label_path = st.session_state.label_path
        image_width = st.session_state.image_width
        with open(label_path, "w") as f:
            for label, bbox in zip(st.session_state.labels, st.session_state.bboxes_xyxy):
                bx, by, bw, bh = bbox
                x_center_norm = (bx + bw / 2) / image_width
                y_center_norm = (by + bh / 2) / image_height
                width_norm = bw / image_width
                height_norm = bh / image_height
                f.write(f"{label} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")

        st.session_state["skip_label_update"] = True

def next_callback():

    st.session_state.frame_index += 1
    st.session_state["skip_label_update"] = True

def prev_callback():
    
    st.session_state.frame_index -= 1
    st.session_state["skip_label_update"] = True

def frame_slider_callback():
    # Get the new value from the slider (using the key "slider_det")
    new_frame_index = st.session_state.slider_det
    # Compare with the current frame_index stored in session_state
    if new_frame_index != st.session_state.frame_index:
        st.session_state.frame_index = new_frame_index
        st.session_state["skip_label_update"] = True

def jump_page_callback():
    # Retrieve the new value from the number input via its key "jump_page"
    new_frame_index = st.session_state.jump_page
    # Compare it with the current frame_index in session_state
    if st.session_state.frame_index != new_frame_index:
        st.session_state.frame_index = new_frame_index
        st.session_state["skip_label_update"] = True

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
    """
    Overlays a large text at the center of the image.
    The text is "Frame Number: {index}".
    
    Args:
        frame (numpy.array): The image as a numpy array.
        index (int): The frame index.
        
    Returns:
        numpy.array: The image with the overlaid text.
    """
    # Convert the frame (numpy array) to a PIL Image.
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    # Attempt to load a large TrueType font.
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 100)
    except IOError:
        font = ImageFont.load_default()
    
    # Create the overlay text.
    text = f"Frame Number: {index}"
    
    # Calculate text dimensions using draw.textbbox.
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate the center position of the image.
    width, height = img.size
    x = (width - text_width) / 2
    y = int((0.95 * height) - text_height)
    
    # Draw a text outline for better visibility.
    outline_range = 2  # Adjust for thicker outline if needed.
    for dx in range(-outline_range, outline_range + 1):
        for dy in range(-outline_range, outline_range + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill="black")
                
    # Draw the main text in white.
    draw.text((x, y), text, font=font, fill="white")
    
    return np.array(img)

def create_video_file(image_paths, fps, scale=1.0, output_path="temp_label_review_video.mp4"):
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
    final_clip.write_videofile(output_path, codec="libx264", audio=False, verbose=False, logger=None)
    
    return output_path

@st.cache_data(show_spinner=True)
def generating_mp4(image_paths, fps):
    return create_video_file(image_paths, fps)

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
    os.makedirs(labels_dir, exist_ok=True)
    
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

#--------------------------------------------------------------------------------------------------------------------------------#


##  Initialization
#--------------------------------------------------------------------------------------------------------------------------------#

if "session_running" not in st.session_state:
    st.session_state.session_running = True

    st.session_state.playback_active = False
    st.session_state.fps = 1
    st.session_state.video_index = 0
    st.session_state.include_labels = True
    st.session_state.video_image_scale = 1.0

    st.set_page_config(layout="wide")

    st.session_state.paths = {

        "venv_path" : "../envs/auto-label-engine/",

        "prev_unverified_images_path" : "example_data",
        "unverified_images_path" : "example_data",
        "prev_unverified_names_yaml_path" : "cfgs/gui/manual_labels/default.yaml",
        "unverified_names_yaml_path" : "cfgs/gui/manual_labels/default.yaml",

        "upload_save_path": "",

        "mp4_path" : "",
        "mp4_save_path" : "",
        "mp4_script_path" : "convert_mp4_2_png.py",

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

    }

    st.session_state.unverified_image_scale = 1.0

    load_session_state()
    
    update_unverified_data_path()

    gpu_info = subprocess.check_output("nvidia-smi -L", shell=True).decode("utf-8")
    st.session_state.gpu_list = [line.strip() for line in gpu_info.splitlines() if line.strip()]

#--------------------------------------------------------------------------------------------------------------------------------#

## Run each iteration
save_session_state()



# Define tabs
#--------------------------------------------------------------------------------------------------------------------------------#
tabs = st.tabs(["Generate Datasets", "Auto Label", "Manual Labeling", "Finetune Model", "Linux Terminal"])

# ----------------------- Generate Data Tab -----------------------
with tabs[0]:  
    output = None
    action_option = st.radio(
        "Choose save path option:", 
        [
            "Upload Data", 
            "Convert MP4 to PNGs", 
            "Rotate Image Dataset",
            "Split YOLO Dataset into Objects / No Objects", 
            "Combine YOLO Datasets"
        ],
        key=f"split_vs_combine_radio",
        label_visibility="collapsed"
    )

    if action_option == "Upload Data":
        with st.expander("Upload Data"):
            st.write("Save Path")
            path_navigator("upload_save_path")
            upload_to_dir(st.session_state.paths["upload_save_path"])

    elif action_option == "Convert MP4 to PNGs":
        with st.expander("Settings"):
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("MP4 Path")
                path_navigator(
                    "mp4_path", 
                    button_and_selectbox_display_size=[4,30]
                )
            
            with c2:
                st.subheader("Save Path")
                save_path_option = st.radio("Choose save path option:", ["Default", "Custom"], key=f"split_save_radio", label_visibility="collapsed")
                key = "mp4_save_path"
                if save_path_option == "Default":
                    st.session_state.paths[key] = st.session_state.paths["mp4_path"].replace(".mp4", "/images/")
                    st.write(f"**Current {' '.join(word.capitalize() for word in key.split('_'))}:** {st.session_state.paths[key]}")
                else:
                    path_navigator(
                        key,
                        button_and_selectbox_display_size=[4,30]
                    )

        with st.expander("Venv Path"):
            path_navigator("venv_path", radio_button_prefix="convert_mp4")

        with st.expander("Script"):
            path_navigator("mp4_script_path")
            python_code_editor("mp4_script_path")

        with st.expander("Convert MP4 to PNGs"):
            output = None
            c1, c2, c3, c4 = st.columns(4, gap="small")
            with c1:
                if st.button("Begin Converting", key="begin_converting_data_btn"):
                    run_in_tmux(
                        session_key="mp4_data", 
                        py_file_path=st.session_state.paths["mp4_script_path"], 
                        venv_path=st.session_state.paths["venv_path"],
                        args={
                            "video_path" : st.session_state.paths["mp4_path"],
                            "output_folder" : st.session_state.paths["mp4_save_path"],
                        }
                    )
                    time.sleep(3)
                    output = update_tmux_terminal("mp4")

            with c2:
                if st.button("Update Terminal Output", key="check_mp4_btn"):
                    output = update_tmux_terminal("mp4")

            with c3:
                if st.button("Clear Terminal Output", key="mp4_clear_terminal_btn"):
                    output = None

            with c4:
                if st.button("Kill TMUX Session", key="mp4_kill_tmux_session_btn"):
                    output = kill_tmux_session("mp4")

    elif action_option == "Rotate Image Dataset":
        with st.expander("Settings"):
            st.subheader("Image Path")
            path_navigator(
                "rotate_images_path", 
                button_and_selectbox_display_size=[4,30]
            )

        with st.expander("Venv Path"):
            path_navigator("venv_path", radio_button_prefix="rotate_images")

        with st.expander("Script"):
            path_navigator("rotate_images_script_path")
            python_code_editor("rotate_images_script_path")

        with st.expander("Rotate Images"):
            output = None
            c1, c2, c3, c4, c5 = st.columns(5, gap="small")

            with c1: 
                action_option = st.radio(
                    "Choose Rotation:", 
                    [
                        "CW", 
                        "CCW", 
                        "180",
                    ],
                    key=f"rotate_images_radio",
                    label_visibility="collapsed"
                )

            with c2:
                if st.button("Begin Rotating Images", key="begin_rotating_data_btn"):
                    run_in_tmux(
                        session_key="rotate_images", 
                        py_file_path=st.session_state.paths["rotate_images_script_path"], 
                        venv_path=st.session_state.paths["venv_path"],
                        args={
                            "directory" : st.session_state.paths["rotate_images_path"],
                            "rotation" : action_option,
                        }
                    )
                    time.sleep(3)
                    output = update_tmux_terminal("rotate_images")

            with c3:
                if st.button("Update Terminal Output", key="check_rotate_images_btn"):
                    output = update_tmux_terminal("rotate_images")

            with c4:
                if st.button("Clear Terminal Output", key="rotate_images_clear_terminal_btn"):
                    output = None

            with c5:
                if st.button("Kill TMUX Session", key="rotate_images_kill_tmux_session_btn"):
                    output = kill_tmux_session("rotate_images")

    elif action_option == "Split YOLO Dataset into Objects / No Objects":
        with st.expander("Dataset Settings"):
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Dataset To Be Split")
                path_navigator(
                    "split_data_path", 
                    button_and_selectbox_display_size=[4,30]
                )
            with c2:
                st.subheader("Save Path")
                save_path_option = st.radio("Choose save path option:", ["Default", "Custom"], key=f"split_save_radio", label_visibility="collapsed")
                key = "split_data_save_path"
                if save_path_option == "Default":
                    st.session_state.paths[key] = st.session_state.paths["split_data_path"]
                    st.write(f"**Current {' '.join(word.capitalize() for word in key.split('_'))}:** {st.session_state.paths[key]}")
                else:
                    path_navigator(
                        key,
                        button_and_selectbox_display_size=[4,30]
                    )

        with st.expander("Venv Path"):
            path_navigator("venv_path", radio_button_prefix="split_data")

        with st.expander("Script"):
            path_navigator("split_data_script_path")
            python_code_editor("split_data_script_path")

        with st.expander("Split Data"):
            output = None
            c1, c2, c3, c4 = st.columns(4, gap="small")

            with c1:
                if st.button("Begin Splitting Data", key="begin_split_data_btn"):
                    run_in_tmux(
                        session_key="split_data", 
                        py_file_path=st.session_state.paths["split_data_script_path"], 
                        venv_path=st.session_state.paths["venv_path"],
                        args={
                            "data_path" : st.session_state.paths["split_data_path"],
                            "save_path" : st.session_state.paths["split_data_save_path"],
                        }
                    )
                    time.sleep(3)
                    output = update_tmux_terminal("split_data")

            with c2:
                if st.button("Update Terminal Output", key="check_split_data_btn"):
                    output = update_tmux_terminal("split_data")

            with c3:
                if st.button("Clear Terminal Output", key="split_data_clear_terminal_btn"):
                    output = None

            with c4:
                if st.button("Kill TMUX Session", key="split_data_kill_tmux_session_btn"):
                    output = kill_tmux_session("split_data")
    else:
        # Combine YOLO Datasets
        with st.expander("Dataset Settings"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("Dataset 1")
                path_navigator(
                    "combine_dataset_1_path", 
                    button_and_selectbox_display_size=[4,30]
                )
            with c2:
                st.subheader("Dataset 2")
                path_navigator(
                    "combine_dataset_2_path", 
                    button_and_selectbox_display_size=[4,30]
                )
            with c3:
                st.subheader("Save Path")
                path_navigator(
                    "combine_dataset_save_path", 
                    button_and_selectbox_display_size=[4,30]
                )

        with st.expander("Script"):
            path_navigator("combine_dataset_script_path")
            python_code_editor("combine_dataset_script_path")

        with st.expander("Combine Data"):
            output = None
            c1, c2, c3, c4 = st.columns(4, gap="small")

            with c1:
                if st.button("Begin Combining Data", key="begin_combine_dataset_btn"):
                    run_in_tmux(
                        session_key="combine_dataset", 
                        py_file_path=st.session_state.paths["combine_dataset_script_path"], 
                        venv_path=st.session_state.paths["venv_path"],
                        args={
                            "dataset1" : st.session_state.paths["combine_dataset_1_path"],
                            "dataset2" : st.session_state.paths["combine_dataset_2_path"],
                            "dst_dir" : st.session_state.paths["combine_dataset_save_path"]
                        }
                    )
                    time.sleep(3)
                    output = update_tmux_terminal("combine_dataset")

            with c2:
                if st.button("Update Terminal Output", key="check_combine_dataset_btn"):
                    output = update_tmux_terminal("combine_dataset")

            with c3:
                if st.button("Clear Terminal Output", key="combine_dataset_clear_terminal_btn"):
                    output = None

            with c4:
                if st.button("Kill TMUX Session", key="combine_dataset_kill_tmux_session_btn"):
                    output = kill_tmux_session("combine_dataset")
    
    terminal_output = st.empty()
    if output is not None:
        display_terminal_output(output)

# ----------------------- Auto Label Tab -----------------------
with tabs[1]:
    with st.expander("Auto Label Settings"):
        st.subheader("Model Weights Path")
        path_navigator("auto_label_model_weight_path")

        st.subheader("Images Path")
        path_navigator("auto_label_data_path")

        st.subheader("Label Save Path")
        path_navigator("auto_label_save_path")

        st.subheader("Venv Path")
        path_navigator("venv_path", radio_button_prefix="auto_label")

    with st.expander("Script"):
        path_navigator("auto_label_script_path")
        python_code_editor("auto_label_script_path")

    with st.expander("Auto Label Data"):
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
            if st.button("Begin Auto Labeling Data", key="begin_auto_labeling_data_btn"):
                run_in_tmux(
                    session_key="auto_label_data", 
                    py_file_path=st.session_state.paths["auto_label_script_path"], 
                    venv_path=st.session_state.paths["venv_path"],
                    args={
                        "model_weights_path" : st.session_state.paths["auto_label_model_weight_path"],
                        "images_dir_path" : st.session_state.paths["auto_label_data_path"],
                        "labels_save_path" : st.session_state.paths["auto_label_save_path"],
                        "gpu_number": st.session_state.auto_label_gpu
                    }
                )
                time.sleep(3)
                output = update_tmux_terminal("auto_label_data")

        with c4:
            if st.button("Update Terminal Output", key="check_auto_labeling_data_btn"):
                output = update_tmux_terminal("auto_label_data")

        with c5:
            if st.button("Clear Terminal Output", key="auto_labeling_clear_terminal_btn"):
                output = None

        with c6:
            if st.button("Kill TMUX Session", key="auto_labeling_kill_tmux_session_btn"):
                output = kill_tmux_session("auto_label_data")

        terminal_output = st.empty()
        if output is not None:
            terminal_output.text(output)

# ----------------------- Manual Label Tab -----------------------
with tabs[2]:

    with st.expander("Settings"):
        st.write("Image Scale")
        image_scale = st.number_input(
            "Image Scale", 
            value=1.0, 
            step=0.25, 
            label_visibility="collapsed"
        )
        if float(image_scale) != st.session_state.unverified_image_scale:
            st.session_state.unverified_image_scale = image_scale
            st.session_state["skip_label_update"] = True
            st.rerun()

        st.write("Images Path")
        path_navigator("unverified_images_path", button_and_selectbox_display_size=[1,25])

        st.write("Label Names YAML Path")
        path_navigator("unverified_names_yaml_path", button_and_selectbox_display_size=[2,25])

        if st.session_state.paths["prev_unverified_images_path"] != st.session_state.paths["unverified_images_path"] or st.session_state.paths["prev_unverified_names_yaml_path"] != st.session_state.paths["unverified_names_yaml_path"]:
            st.session_state.paths["prev_unverified_images_path"] = st.session_state.paths["unverified_images_path"]
            st.session_state.paths["prev_unverified_names_yaml_path"] = st.session_state.paths["unverified_names_yaml_path"]
            update_unverified_data_path()

            if st.session_state.max_images > 0:
                update_unverified_frame()
                
            st.rerun()
        
        yaml_editor("unverified_names_yaml_path")

    with st.expander("Manual Label Review"):

        if st.session_state.get("no_images_warning"):
            st.warning(st.session_state.no_images_warning)

        elif st.session_state.get("naming_pattern_warning"):
            st.warning(st.session_state.naming_pattern_warning)
            option = st.radio(
                "Select an option to proceed:",
                options=["Rename files with a pattern (lose original file names for faster performance)", "Store list of all images (retain orignal file names but lose performance)"],
                key="naming_pattern_choice"
            )
            if option == "Rename files with a pattern (lose original file names for faster performance)":
                if st.button("Apply Rename", key="apply_rename"):
                    # Get the images directory from session state.
                    images_dir = st.session_state.paths["unverified_images_path"]
                    
                    # Call the safe renaming function.
                    new_pattern, total_images = safe_rename_images(images_dir)
                    
                    if new_pattern is not None:
                        # Update the session state.
                        st.session_state.image_pattern = new_pattern
                        st.session_state.start_index = 0
                        st.session_state.max_images = total_images
                        st.session_state.naming_pattern_warning = None
                        st.session_state.image_list = None
                        st.success("Session state updated after renaming.")
                        st.rerun()
            else:
                if st.button("Store Image List", key="store_image_list"):
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
                    st.success("Image list stored. Note: This may slow down performance for large datasets.")
                    st.rerun()

        else:
            if st.session_state.max_images > 0:
                
                # --- Top Navigation (Prev / Next) ---
                col_prev, _, col_next = st.columns([4, 5, 4])
                with col_prev:
                    st.button("Prev Frame", key="top_prev_btn", on_click=prev_callback)
                with col_next:
                    st.button("Next Frame", key="top_next_btn", on_click=next_callback)

                col_copy_prev, _, col_copy_next = st.columns([4, 5, 4])
                with col_copy_prev:
                    st.button("Copy Labels from Prev Slide", key="copy_prev_btn", on_click=copy_prev_labels)
                with col_copy_next:
                    st.button("Copy Labels from Next Slide", key="copy_next_btn", on_click=copy_next_labels)

                # Read from .txt file & build detection_config
                update_unverified_frame()

                # Display current frame file path
                st.write(f"Current File Path: {st.session_state.image_path}")

                # Let user annotate with detection()
                st.session_state.out = detection(**st.session_state.detection_config)

                # Update labels if changed in detection()
                update_labels()
                    
                # Additional navigation (jump, slider, second Prev/Next)
                st.number_input(
                    "Jump to Image", min_value=0, max_value=st.session_state.max_images-1,
                    value=st.session_state.frame_index, step=10, key="jump_page",
                    on_change=jump_page_callback
                )
                
                col_prev, col_slider, col_next = st.columns([2, 10, 4])
                with col_prev:
                    st.button("Prev Frame", key="prev_btn", on_click=prev_callback)
                with col_slider:
                    st.slider(
                        "Frame Index", 0, st.session_state.max_images - 1,
                        st.session_state.frame_index, key="slider_det",
                        on_change=frame_slider_callback
                    )
                with col_next:
                    st.button("Next Frame", key="next_btn", on_click=next_callback)
 
            else:
                st.warning("Data Path is empty...")

    with st.expander("Zoomed-in Bounding Box Regions"):
        if 'image' in st.session_state and 'bboxes_xyxy' in st.session_state:
            update_unverified_frame()

            bboxes = st.session_state.bboxes_xyxy
            labels = st.session_state.labels
            bbox_ids = st.session_state.bbox_ids
            if bboxes:
                for i, bbox in enumerate(bboxes):
                    # bbox is stored as [x, y, width, height] (XYWH)
                    x, y, w, h = bbox
                    x, y = max(x, 0.0) , max(y, 0.0)
                    x1, y1, x2, y2 = x, y, x + w, y + h

                    # Check for invalid bounding box coordinates before cropping
                    if x2 <= x1 or y2 <= y1:
                        st.session_state.labels.pop(i)
                        st.session_state.bboxes_xyxy.pop(i)
                        st.session_state.bbox_ids.pop(i)
                        st.rerun()

                    # Crop the image and prepare the caption
                    cropped = st.session_state.image.crop((x1, y1, x2, y2))
                    if "label_list" in st.session_state and i < len(labels):
                        try:
                            label_name = st.session_state.label_list[labels[i]]
                        except Exception:
                            label_name = f"Label {labels[i]}"
                    else:
                        label_name = f"Label {labels[i]}" if i < len(labels) else "Unknown"
                    caption = f"ID: {bbox_ids[i]}, Label: {label_name}, BBox: ({x1}, {y1}, {x2}, {y2})"

                    st.markdown(f"#### Edit Bounding Box {i} Parameters")

                    # Create two columns: left for the image, right for the number inputs.
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(cropped, caption=caption)

                    # Read the center values from session state if they exist; otherwise use default (x + w/2, y + h/2)
                    center_x = x + w / 2
                    center_y = y + h / 2

                    with col2:
                        new_center_x = st.number_input(
                            f"Center X for bbox {i}",
                            min_value=0.0,
                            max_value=float(st.session_state.image_width),
                            value=center_x,
                            key=f"bbox_{i}_center_x_input",
                            step=1.0,
                            on_change=lambda i=i: zoom_edit_callback(i)
                        )
                        new_center_y = st.number_input(
                            f"Center Y for bbox {i}",
                            min_value=0.0,
                            max_value=float(st.session_state.image_height),
                            value=st.session_state.image_height - center_y,
                            key=f"bbox_{i}_center_y_input",
                            step=1.0,
                            on_change=lambda i=i: zoom_edit_callback(i)
                        )
                        new_w = st.number_input(
                            f"Width for bbox {i}",
                            min_value=1.0,
                            max_value=float(st.session_state.image_width),
                            value=float(w),
                            key=f"bbox_{i}_w_input",
                            step=1.0,
                            on_change=lambda i=i: zoom_edit_callback(i)
                        )
                        new_h = st.number_input(
                            f"Height for bbox {i}",
                            min_value=1.0,
                            max_value=float(st.session_state.image_height),
                            value=float(h),
                            key=f"bbox_{i}_h_input",
                            step=1.0,
                            on_change=lambda i=i: zoom_edit_callback(i)
                        )

    with st.expander("Video Review"):
        c1, c2, c3 = st.columns([10,10,100])
        with c1:
            # Slider to adjust the playback speed (seconds per frame).
            st.session_state.fps = st.number_input(
                "FPS",
                min_value=1,
                max_value=10,
                value=int(st.session_state.fps),
                step=1
            )

        with c2:
            # Slider to adjust the image scale.
            st.session_state.video_image_scale = st.number_input(
                "Image Scale",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.video_image_scale,
                step=0.1
            )

        if st.button("Generate Video on Current Labels"):            

            if st.session_state.max_images > 0:
                if st.session_state.image_pattern:
                    image_list = [
                        os.path.join(st.session_state.images_dir, st.session_state.image_pattern.format(i))
                        for i in range(st.session_state.start_index, st.session_state.start_index + st.session_state.max_images)
                    ]
                elif "image_list" in st.session_state and st.session_state.image_list:
                    image_list = st.session_state.image_list
                else:
                    st.write("No images available to generate a video.")
                    image_list = []

                if image_list:
                    generating_mp4.clear() 
                    video_file_path = generating_mp4(image_list, st.session_state.fps)
                    st.video(video_file_path, autoplay=True, loop=True)
                else:
                    st.write("No images available to generate a video.")

            else:
                st.write("No images available to generate a video.")
                
# ----------------------- Train Status Tab -----------------------
with tabs[3]:
    with st.expander("Data YAML"):
        path_navigator("train_data_yaml_path")
        yaml_editor("train_data_yaml_path")
    
    with st.expander("Model YAML"):
        path_navigator("train_model_yaml_path")
        yaml_editor("train_model_yaml_path")
        
    with st.expander("Train YAML Path"):
        path_navigator("train_train_yaml_path")
        yaml_editor("train_train_yaml_path")

    with st.expander("Venv Path"):
        path_navigator("venv_path", radio_button_prefix="train")

    with st.expander("Script"):
        path_navigator("train_script_path")
        python_code_editor("train_script_path")

    with st.expander("Finetune Model"):
        output = None
        c1, c2, c3, c4, c5 = st.columns(5, gap="small")

        with c1:
            output = check_gpu_status("train_check_gpu_status_button")

        with c2:
            if st.button("Begin Training", key="begin_train_btn"):
                output = run_in_tmux(
                    session_key="auto_label_trainer", 
                    py_file_path=st.session_state.paths["train_script_path"], 
                    venv_path=st.session_state.paths["venv_path"],
                    args={
                        "data_path": st.session_state.paths["train_data_yaml_path"],
                        "model_path": st.session_state.paths["train_model_yaml_path"],
                        "train_path" : st.session_state.paths["train_train_yaml_path"]
                    }
                )
                time.sleep(3)
                output = update_tmux_terminal("auto_label_trainer")

        with c3:
            if st.button("Check Training", key="check_train_btn"):
                output = update_tmux_terminal("auto_label_trainer")

        with c4:
            if st.button("Clear Terminal Output", key="clear_terminal_btn"):
                output = None

        with c5:
            if st.button("Kill TMUX Session", key="kill_tmux_session_btn"):
                output = kill_tmux_session("auto_label_trainer")

        terminal_output = st.empty()
        if output is not None:
            terminal_output.text(output)

# ----------------------- Linux Terminal Tab -----------------------
with tabs[4]:
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
