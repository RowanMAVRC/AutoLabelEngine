import os
import yaml
import streamlit as st
import subprocess
import glob
from PIL import Image
from streamlit_label_kit import detection


import os
import streamlit as st

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
    save_path_option = st.radio("Choose save path option:", ["File Explorer", "Enter Path as Text"], key=f"{key}_radio")

    if save_path_option == "Enter Path as Text":
        # -- CUSTOM PATH MODE --
        # Now default to the current path in the text input
        custom_path = st.text_input(
            "Enter custom save path:",
            value=current_path,  # <--- Prefills with current path
            key=f"{key}_custom_path_input",
            label_visibility="collapsed"
        )

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
                # Removed st.rerun() here because callbacks already trigger a re-run

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
    data_yaml_path = os.path.join(
        st.session_state.paths["unverified_data_yaml_path"]
    )

    with open(data_yaml_path, 'r') as file:
        data_cfg = yaml.safe_load(file)

    image_dir = os.path.join(data_cfg["path"], "images")
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
        "unverified_data_yaml_path" : "/data/TGSSE/ALE/cfgs/yolo/data/default.yaml",

        "split_data_path" : "/data/TGSSE/HololensCombined/random_subset_50/images/",

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
    update_unverified_frame()
    
# Define constants and load images
label_list = st.session_state.label_list
image_path_list = st.session_state.image_path_list
image_size = [st.session_state.image_width, st.session_state.image_height]
DEFAULT_HEIGHT = st.session_state.image_height
tabs = st.tabs(["Auto Label", "Manual Labeling" , "GPU Status"])

# ----------------------- Auto Label Tab -----------------------
with tabs[0]:

    with st.expander("Manual Label Settings"):
        path_navigator(
            "unverified_data_yaml_path", 
            button_and_selectbox_display_size=[1,25]
        )

    # Create an expander for the auto label settings (data, weights, and save_path)
    with st.expander("Auto Label Settings"):
        c1, c2, c3 = st.columns(3)
        
        # The path the labeled images will go to (unverified - user will verify)
        with c1:
            st.subheader("Save Path")
            path_navigator(
                "auto_label_save_path", 
                button_and_selectbox_display_size=[4,30]
            )
        
        # The trained weights to use for auto-labeling
        with c2:
            st.subheader("Model Weights")
            path_navigator(
                "auto_label_model_weight_path", 
                button_and_selectbox_display_size=[4,30]
            )
            
        
        # The data configs to auto-label from (default option or define the file yourself)
        with c3:
            st.subheader("Data YAML")
            path_navigator(
                "auto_label_data_path", 
                button_and_selectbox_display_size=[4,30]
            )
# ----------------------- Detection Tab -----------------------
with tabs[1]:

    # Generate Screen
    st.session_state.out = detection(
        **st.session_state.detection_config
    )

    # Check for label changes
    if st.session_state.out["key"] != 0:
        update_labels()

    # Navigation controls: Save labels before navigating away.
    frame_index = st.number_input(
        "Jump to Image", min_value=0, max_value=len(image_path_list)-1,
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
            if st.session_state.frame_index < len(image_path_list) - 1:
                st.session_state.frame_index += 1
                st.rerun()

    update_unverified_frame()
# ----------------------- GPU Status Tab -----------------------
with tabs[2]:  # Third tab
    st.header("GPU Status")

    st.write("Click the button below to check the GPU status on Lambda 2.")
    
    if st.button("Check GPU Status"):
        try:
            # Run the gpustat command and capture output
            output = subprocess.check_output(["gpustat"]).decode("utf-8")
            st.text(output)  # Display the raw output
        except Exception as e:
            st.error(f"Failed to run gpustat: {e}")

