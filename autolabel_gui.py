import os
import yaml
import streamlit as st
import subprocess
import glob
from PIL import Image
from streamlit_label_kit import detection, absolute_to_relative, convert_bbox_format

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
tabs = st.tabs(["Configure", "Manual Labeling" , "GPU Status"])

# ----------------------- Configure Tab -----------------------
# with tabs[0]:
    

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

