import os
import yaml
import streamlit as st
import subprocess
import glob
from PIL import Image
from streamlit_label_kit import detection
from streamlit_ace import st_ace
from pathlib import Path

def check_gpu_status(button_key):
    if st.button("Check GPU Status", key=button_key):
        try:
            # Run the gpustat command and capture its output
            output = subprocess.check_output(["gpustat"]).decode("utf-8")
            st.text(output)  # Display the raw output
        except Exception as e:
            st.error(f"Failed to run gpustat: {e}")

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


def path_navigator(key, button_and_selectbox_display_size=[1, 25]):
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
    save_path_option = st.radio("Choose save path option:", ["File Explorer", "Enter Path as Text"], key=f"{key}_radio", label_visibility="collapsed")

    if save_path_option == "Enter Path as Text":
        # -- CUSTOM PATH MODE --
        # Now default to the current path in the text input
        custom_path = st.text_input(
            "Enter custom save path:",
            value=current_path,  # <--- Prefills with current path
            key=f"{key}_custom_path_input",
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
                    if st.button("Create this path", key=f"{key}_create_custom"):
                        # Prompt user for a final directory or filename to create
                        new_name = st.text_input(
                            "Optionally enter a different name for the new path:",
                            value=custom_path,
                            key=f"{key}_new_path_name"
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
                    if st.button("Go up until path exists", key=f"{key}_go_up_custom"):
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
            go_up_button_key = f"go_up_button_{key}"
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
                if st.button("Create this path", key=f"{key}_create_default"):
                    # Ask for a final directory name to create
                    new_name = st.text_input(
                        "Optionally enter a different name for the new path:",
                        value=directory_to_list,
                        key=f"{key}_new_default_path_name"
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
                if st.button("Go up until path exists", key=f"{key}_go_up_default"):
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

        widget_key = f"navigator_select_{key}"

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

if "session_running" not in st.session_state:
    st.session_state.session_running = True

    st.set_page_config(layout="wide")

    st.session_state.paths = {
        "unverified_data_yaml_path" : "/data/TGSSE/ALE/cfgs/verify/default.yaml",

        "split_data_path" : "/data/TGSSE/HololensCombined/random_subset_50/",
        "split_data_save_path" : "/data/TGSSE/HololensCombined/random_subset_50/",

        "auto_label_save_path" : "/data/TGSSE/HololensCombined/random_subset_50/labels/",
        "auto_label_model_weight_path" : "/data/TGSSE/weights/coco_2_ijcnn_vr_full_2_real_world_combination_2_hololens_finetune-v3.pt",
        "auto_label_data_path" :  "/data/TGSSE/HololensCombined/random_subset_50/images/",
     
        "combine_dataset_1_path": "/data/TGSSE/ALE/",
        "combine_dataset_2_path": "/data/TGSSE/ALE/",
        "combine_dataset_save_path": "/data/TGSSE/ALE/",

        "train_data_yaml_path": "/data/TGSSE/ALE/cfgs/yolo/data/default.yaml",
        "train_model_yaml_path": "/data/TGSSE/ALE/cfgs/yolo/model/default.yaml",
        "train_train_yaml_path": "/data/TGSSE/ALE/cfgs/yolo/train/default.yaml"

    }

    update_unverified_data_path()
    
# Define tabs
tabs = st.tabs(["Auto Label", "Generate Datasets", "Manual Labeling", "Finetune Model"])

# ----------------------- Auto Label Tab -----------------------
with tabs[0]:

    # Create an expander for the auto label settings (data, weights, and save_path)
    with st.expander("Auto Label Settings"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.subheader("Model Weights Path")
            path_navigator(
                "auto_label_model_weight_path", 
                button_and_selectbox_display_size=[4,30]
            )
        
        with c2:
            st.subheader("Images Path")
            path_navigator(
                "auto_label_data_path", 
                button_and_selectbox_display_size=[4,30]
            )
            
        with c3:

            st.subheader("Save Path")
            path_navigator(
                "auto_label_save_path", 
                button_and_selectbox_display_size=[4,30]
            )

    with st.expander("Check GPU Status") :
        check_gpu_status("auto_label_check_gpu_status_button")

    # TODO
    # with st.expander("Auto Label"):
        # Auto Label Button
        # Function for inference

# ----------------------- Generate Data Tab -----------------------
with tabs[1]:  
    with st.expander("Split Dataset Settings"):
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
                
    with st.expander("Combine Datasets Settings"):
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
                
# ----------------------- Manual Label Tab -----------------------
with tabs[2]:

    with st.expander("Manual Label Settings"):
        key = "unverified_data_yaml_path"
        path_navigator(
            key, 
            button_and_selectbox_display_size=[1,25]
        )

        yaml_editor(key)

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
# ----------------------- Train Status Tab -----------------------
with tabs[3]: 

    with st.expander("Data YAML"):
        key = "train_data_yaml_path"
        path_navigator(
            key, 
            # button_and_selectbox_display_size=[4,30]
        )
        yaml_editor(key)
    
    with st.expander("Model YAML"):
        key = "train_model_yaml_path"
        path_navigator(
            key, 
            # button_and_selectbox_display_size=[4,30]
        )
        yaml_editor(key)
        
    with st.expander("Train YAML Path"):
        key = "train_train_yaml_path"
        path_navigator(
            key, 
            # button_and_selectbox_display_size=[4,30]
        )
        yaml_editor(key)
 

    with st.expander("Check GPU Status") :
        check_gpu_status("train_check_gpu_status_button")

    # TODO
    # with st.expander("Auto Label"):
        # Auto Label Button
        # Function for inference

