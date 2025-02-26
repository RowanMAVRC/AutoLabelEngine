import os
import yaml
import streamlit as st
import subprocess
import glob
from PIL import Image
from streamlit_label_kit import detection, absolute_to_relative, convert_bbox_format

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

def update_frame():
    
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

def update_data_path():
    with open(st.session_state.data_path, 'r') as file:
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

    st.session_state.data_path = "cfgs/yolo/data/hololens_combined.yaml"

    update_data_path()
update_frame()

# Define constants and load images
label_list = st.session_state.label_list
image_path_list = st.session_state.image_path_list
image_size = [st.session_state.image_width, st.session_state.image_height]
DEFAULT_HEIGHT = st.session_state.image_height
DEFAULT_LINE_WIDTH = 1.0
tabs = st.tabs(["Auto Label", "Manual Labeling" , "GPU Status"])

# ----------------------- Auto Label Tab -----------------------
# TODO - cd into /data/TGSSE and create an ALE dir. Then create the cfgs/yolo/data dirs for a default yaml (can just choose the hololense one)
with tabs[0]:

    # Create an expander for the auto label settings (data, weights, and save_path)
    with st.expander("Settings"):
        c1, c2, c3 = st.columns(3)
        
        # The path the labeled images will go to (unverified - user will verify)
        with c1:
            st.subheader("Save Path")
            save_path_option = st.radio("Choose save path option:", ["Default", "Custom"])
            if save_path_option == "Default":
                _save_path = st.selectbox("Select default save path:", ["/data/TGSSE/ALE/unverified", "/data/TGSSE/ALE/testing"])
            else:
                _save_path = st.text_input("Enter custom save path:")
        
        # The trained weights to use for auto-labeling
        with c2:
            st.subheader("Model Weights")
            weights_option = st.radio("Choose weights option:", ["Default", "Upload"])
            if weights_option == "Default":
                default_weights = list_files("/data/TGSSE/weights", ".pt")
                _model_weights = st.selectbox("Select default weights:", default_weights)
            else:
                _model_weights = st.file_uploader("Upload model weights", type=["pt"])
        
        # The data configs to auto-label from (default option or define the file yourself)
        with c3:
            st.subheader("Data YAML")
            yaml_option = st.radio("Choose YAML option:", ["Default", "Custom"])
            if yaml_option == "Default":
                default_yamls = list_files("/data/TGSSE/ALE/cfgs/yolo/data", ".yaml")
                _data_yaml = st.selectbox("Select default YAML:", default_yamls)
            else:
                st.write("Define custom YAML settings:")
                _yaml_path = st.text_input("Dataset path:", "/data/TGSSE/HololensCombined/random_subset")
                _yaml_train = st.text_input("Train folder:", "images")
                _yaml_val = st.text_input("Validation folder:", "images")
                _yaml_test = st.text_input("Test folder:", "images")
                
                st.write("Define class names:")
                num_classes = st.number_input("Number of classes:", min_value=1, value=6)
                class_names = {}
                for i in range(num_classes):
                    class_name = st.text_input(f"Class {i}:", value=f"Class_{i}")
                    class_names[i] = class_name
                
                _data_yaml = {
                    "path": _yaml_path,
                    "train": _yaml_train,
                    "val": _yaml_val,
                    "test": _yaml_test,
                    "names": class_names
                }

        if yaml_option == "Custom":
            st.write("Generated YAML:")
            st.code(yaml.dump(_data_yaml, default_flow_style=False), language="yaml")

        st.write("Selected save path:", _save_path)
        st.write("Selected model weights:", _model_weights)

        if yaml_option == "Default":
            st.write("Selected/Generated data YAML:", _data_yaml)

        # Add a button to save the upload and create data YAML
        if st.button("Save Upload and Create Data YAML"):
            try:
                # Check if the save_path exists, if not, create it
                if not os.path.exists(_save_path):
                    os.makedirs(_save_path)
                    st.info(f"Created directory: {_save_path}")

                # Save the upload to the save_path
                if weights_option == "Upload" and _model_weights is not None:
                    upload_path = os.path.join(_save_path, _model_weights.name)
                    with open(upload_path, "wb") as f:
                        f.write(_model_weights.getbuffer())
                    st.success(f"Upload saved to: {upload_path}")

                # Create and save the data YAML
                if yaml_option == "Custom":
                    yaml_filename = "data_config.yaml"
                    yaml_path = os.path.join(_save_path, yaml_filename)
                    with open(yaml_path, "w") as f:
                        yaml.dump(_data_yaml, f, default_flow_style=False)
                    st.success(f"Data YAML saved to: {yaml_path}")

                st.success("Upload saved and Data YAML created successfully!")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

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

