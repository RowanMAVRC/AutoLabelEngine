import os
import yaml
import streamlit as st
import subprocess
import glob
import zipfile
from PIL import Image
from streamlit_label_kit import detection
from streamlit_ace import st_ace
from pathlib import Path
import time

## Functions
#-------------------------------------------------------------------------------------------------------------------------#

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
    if yaml_key not in st.session_state.yamls:
        st.session_state.yamls[yaml_key] = file_content

    ace_key = f"edited_content_{yaml_key}"
    
    # with col1:
    st.markdown("Edit YAML content")

    lines = file_content.splitlines()
    line_count = len(lines) if len(lines) > 0 else 1
    calculated_height = max(300, line_count * 19)

    edited_content = st_ace(
        value=file_content,
        language="yaml",
        theme="",
        height=calculated_height,
        font_size=17, 
        key=ace_key
    )

    # Auto-save if the edited content has changed compared to the last saved version
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

def path_navigator(key, radio_button_prefix="", button_and_selectbox_display_size=[1, 25]):
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

def update_labels():
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

    if current_bboxes != bboxes_xyxy or current_labels != labels:
        # Write normalized YOLO-format labels to file
        with open(label_path, "w") as f:
            for label, bbox in zip(current_labels, current_bboxes):

                x_min, y_min, width, height = bbox
                # Convert the absolute coordinates back to normalized YOLO format:
                # Calculate center coordinates normalized by image dimensions.
                x_center_norm = (x_min + width / 2) / image_width
                y_center_norm = (y_min + height / 2) / image_height
                # Normalize width and height.
                width_norm = width / image_width
                height_norm = height / image_height
                # Write the line in YOLO format: class x_center y_center width height
                f.write(f"{label} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")

def update_unverified_frame():
    
    image_dir = st.session_state.image_dir
    image_path = st.session_state.image_path_list[st.session_state.frame_index]
    
    image = Image.open(image_path)
    image_width, image_height = image.size

    labels_dir = image_dir.replace("images", "labels")
    label_path = image_path.replace("images", "labels").replace("jpg", "txt").replace("png", "txt")

    # Ensure label path exists
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # Read the YOLO-format labels (rows of: class x y w h normalized).
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

    st.session_state.image_path = image_path
    st.session_state.image = image
    st.session_state.labels_dir = labels_dir
    st.session_state.label_path = label_path
    st.session_state.image_width = image_width
    st.session_state.image_height = image_height
    st.session_state.bboxes_xyxy = bboxes_xyxy
    st.session_state.labels = labels
    st.session_state.bbox_ids = bbox_ids

    st.session_state["detection_config"] = {
        "image_path": st.session_state.image_path,
        "image_height": st.session_state.image_height,
        "image_width": st.session_state.image_width,
        "label_list": st.session_state.label_list,
        "bboxes": st.session_state.bboxes_xyxy,
        "labels": st.session_state.labels,
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
        "key": None
    }

def update_unverified_data_path():

    data_yaml_path = st.session_state.paths["unverified_data_yaml_path"]

    if Path(data_yaml_path).suffix.lower() in ['.yaml', '.yml']:
        pass
    
    with open(data_yaml_path, 'r') as file:
        data_cfg = yaml.safe_load(file)

    image_dir = data_cfg["path"]
    image_path_list = glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg"))
    image_path_list.sort()
    label_list=list(data_cfg["names"].values())

    st.session_state.data_cfg = data_cfg
    st.session_state.label_list = label_list
    st.session_state.image_dir = image_dir
    st.session_state.image_path_list = image_path_list
    st.session_state.frame_index = 0

#--------------------------------------------------------------------------------------------------------------------------------#


##  Initialization
#--------------------------------------------------------------------------------------------------------------------------------#

if "session_running" not in st.session_state:
    st.session_state.session_running = True

    st.set_page_config(layout="wide")

    st.session_state.paths = {

        "venv_path" : "/home/naddeok5/envs/auto-label-engine/",

        "unverified_data_yaml_path" : "/data/TGSSE/ALE/cfgs/verify/default.yaml",

        "upload_save_path": "/data/TGSSE",

        "mp4_path" : "/data/TGSSE",
        "mp4_save_path" : "/data/TGSSE/",
        "mp4_script_path" : "/home/naddeok5/AutoLabelEngine/convert_mp4_2_png.py",

        "split_data_path" : "/data/TGSSE/HololensCombined/random_subset_50/",
        "split_data_save_path" : "/data/TGSSE/HololensCombined/random_subset_50/",
        "split_data_script_path" : "/home/naddeok5/AutoLabelEngine/split_yolo_data_by_object.py",

        "auto_label_save_path" : "/data/TGSSE/HololensCombined/random_subset_50/labels/",
        "auto_label_model_weight_path" : "/data/TGSSE/weights/coco_2_ijcnn_vr_full_2_real_world_combination_2_hololens_finetune-v3.pt",
        "auto_label_data_path" :  "/data/TGSSE/HololensCombined/random_subset_50/images/",
        "auto_label_script_path" : "/home/naddeok5/AutoLabelEngine/inference.py",
     
        "combine_dataset_1_path": "/data/TGSSE/ALE/",
        "combine_dataset_2_path": "/data/TGSSE/ALE/",
        "combine_dataset_save_path": "/data/TGSSE/ALE/",
        "combine_dataset_script_path" : "/home/naddeok5/AutoLabelEngine/combine_yolo_dirs.py",

        "train_data_yaml_path": "/data/TGSSE/ALE/cfgs/yolo/data/default.yaml",
        "train_model_yaml_path": "/data/TGSSE/ALE/cfgs/yolo/model/default.yaml",
        "train_train_yaml_path": "/data/TGSSE/ALE/cfgs/yolo/train/default.yaml",
        "train_script_path" : "/home/naddeok5/AutoLabelEngine/train_yolo.py",

    }

    update_unverified_data_path()

    gpu_info = subprocess.check_output("nvidia-smi -L", shell=True).decode("utf-8")
    st.session_state.gpu_list = [line.strip() for line in gpu_info.splitlines() if line.strip()]
    
#--------------------------------------------------------------------------------------------------------------------------------#


## Define tabs
#--------------------------------------------------------------------------------------------------------------------------------#
tabs = st.tabs(["Auto Label", "Generate Datasets", "Manual Labeling", "Finetune Model", "Linux Terminal"])

# ----------------------- Auto Label Tab -----------------------
with tabs[0]:

    # Create an expander for the auto label settings (data, weights, and save_path)
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

        # Define a placeholder for terminal output so we can update or clear it.
        output = None
        c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")

        with c1:
            output = check_gpu_status("auto_label_check_gpu_status_button")

        with c2:
            try:
                # Get the list of available GPUs using nvidia-smi
                
                if st.session_state.gpu_list:
                    selected_gpu = st.selectbox("Select GPU", options=list(range(len(st.session_state.gpu_list))),
                                                  format_func=lambda x: f"GPU {x}", label_visibility="collapsed")
                    st.session_state.auto_label_gpu = int(selected_gpu)  # store as numeric value
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

# ----------------------- Generate Data Tab -----------------------
with tabs[1]:  
    action_option = st.radio(
        "Choose save path option:", 
        [
            "Upload Data", 
            "Convert MP4 to PNGs", 
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

        with st.expander("Script"):
            path_navigator("mp4_script_path")
            python_code_editor("mp4_script_path")

        with st.expander("Convert MP4 to PNGs"):

            # Define a placeholder for terminal output so we can update or clear it.
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

        with st.expander("Script"):
            path_navigator("split_data_script_path")
            python_code_editor("split_data_script_path")

        with st.expander("Split Data"):

            # Define a placeholder for terminal output so we can update or clear it.
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

            # Define a placeholder for terminal output so we can update or clear it.
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

# ----------------------- Manual Label Tab -----------------------
with tabs[2]:

    with st.expander("Settings"):
        key = "unverified_data_yaml_path"
        path_navigator(
            key, 
            button_and_selectbox_display_size=[1,25]
        )

        yaml_editor(key)

    with st.expander("Review"):
        # Generate Screen
        if len(st.session_state.image_path_list) > 0:
            update_unverified_frame()

            st.session_state.out = detection(
                **st.session_state.detection_config
            )

            # Check for label changes
            if st.session_state.out["key"] != 0:
                update_labels()

            # Navigation controls: Save labels before navigating away.
            frame_index = st.number_input(
                "Jump to Image", min_value=0, max_value=len(st.session_state.image_path_list)-1,
                value=st.session_state.frame_index, step=10, key="jump_page"
            )
            if st.session_state.frame_index != frame_index:
                st.session_state.frame_index = frame_index
                st.rerun()
            
            # Frame Index Prev/Slider/Next
            col_prev, col_slider, col_next = st.columns([1, 10, 2])
            with col_prev:
                if st.button("Prev", key="prev_btn"):
                    if st.session_state.frame_index > 0:
                        st.session_state.frame_index -= 1
                        st.rerun()
            with col_slider:
                frame_index = st.slider(
                    "Frame Index", 0, len(st.session_state.image_path_list) - 1,
                    st.session_state.frame_index, key="slider_det"
                )
                if frame_index != st.session_state.frame_index:
                    st.session_state.frame_index = frame_index
                    st.rerun()
            with col_next:
                if st.button("Next", key="next_btn"):
                    if st.session_state.frame_index < len(st.session_state.image_path_list) - 1:
                        st.session_state.frame_index += 1
                        st.rerun()

            update_unverified_frame()

        else:
            st.warning("Data Path is empty...")

    with st.expander("Zoomed-in Bounding Box Regions"):
        if 'image' in st.session_state and 'bboxes_xyxy' in st.session_state:
            bboxes = st.session_state.bboxes_xyxy
            labels = st.session_state.labels
            bbox_ids = st.session_state.bbox_ids
            if bboxes:
                for i, bbox in enumerate(bboxes):
                    # bbox is stored as [x, y, width, height] (XYWH)
                    x, y, w, h = bbox
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    cropped = st.session_state.image.crop((x1, y1, x2, y2))
                    # Lookup label name if available; fallback to the label index.
                    if "label_list" in st.session_state and i < len(labels):
                        try:
                            label_name = st.session_state.label_list[labels[i]]
                        except Exception:
                            label_name = f"Label {labels[i]}"
                    else:
                        label_name = f"Label {labels[i]}" if i < len(labels) else "Unknown"
                    caption = f"ID: {bbox_ids[i]}, Label: {label_name}, BBox: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
                    st.image(cropped, caption=caption)
            else:
                st.write("No bounding boxes detected.")
        else:
            st.write("Frame not available.")

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
    
        # Define a placeholder for terminal output so we can update or clear it.
        output = None
        c1, c2, c3, c4, c5, = st.columns(5, gap="small")

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
            # Stream stdout line-by-line and update the same placeholder.
            for line in process.stdout:
                st.session_state.terminal_text += line
                output_placeholder.code(st.session_state.terminal_text, language="bash")
            # Stream stderr line-by-line and update the same placeholder.
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

    # Display the accumulated terminal output in one code block below the input.
    output_placeholder.code(st.session_state.terminal_text, language="bash")

#--------------------------------------------------------------------------------------------------------------------------------#

