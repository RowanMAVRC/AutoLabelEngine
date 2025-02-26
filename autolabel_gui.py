
import os
import streamlit as st
import subprocess
import glob
from PIL import Image
from streamlit_label_kit import detection, absolute_to_relative, convert_bbox_format

from utils import load_yaml

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
    classes = []
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
                classes.append(cls)
    else:
        with open(label_path, "w") as f:
            f.write("")

    bbox_ids = ["bbox-" + str(i) for i in range(len(bboxes_xyxy))]

    st.session_state.image_path = image_path
    st.session_state.image = image
    st.session_state.image_width = image_width
    st.session_state.image_height = image_height
    st.session_state.bboxes_xyxy = bboxes_xyxy
    st.session_state.classes = classes
    st.session_state.bbox_ids = bbox_ids


def update_data_path(data_path):
    data_cfg = load_yaml(data_path)
    image_dir = os.path.join(data_cfg["path"], "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg"))
    image_path_list.sort()
    label_list=list(data_cfg["names"].values())

    st.session_state.data_path = data_path
    st.session_state.data_cfg = data_cfg
    st.session_state.label_list = label_list
    st.session_state.image_dir = image_dir
    st.session_state.image_path_list = image_path_list
    st.session_state.frame_index = 0



# Initialize
if "session_running" not in st.session_state:
    print("Initializing session")
    
    # Set running status
    st.session_state.session_running = True
    
    # Set Data
    update_data_path("cfgs/yolo/data/hololens_combined.yaml")

    # Set Tabs
    st.session_state.tabs = st.tabs(["Configure Manual Labeling Window", "Manual Labeling", "GPU Status"])

update_frame()
print(st.session_state.frame_index)

# ----------------------- Configure Tab -----------------------
with st.session_state.tabs[0]:
    
    with st.expander("Image & Inputs"):
        
        _bbox_show_label = st.toggle("Show Bounding Box Labels", True, key="bbox_show_label")
        
        c1, c2 = st.columns(2)
        with c1:
            _info_dict_help = st.toggle("Info Dict", help='e.g., [{"Confidence": 0.1, "testScore": 0.98}, {"Confidence": 0.2}]', key="info_dict_help")
        with c2:
            _meta_help = st.toggle("MetaData/Description", help='e.g., [["meta/description1", "meta1", "meta2"], ["meta/description2"]]', key="meta_help")
        _meta = [["meta/description1", "meta1", "meta2"], ["meta/description2"]] if _meta_help else []
        if _info_dict_help:
            _info_dict = [{"Confidence": 0.1, "testScore": 0.98}, {"Confidence": 0.2}]
            _bbox_show_info = st.toggle("Show Bounding Box Info", True, key="bbox_show_info")
        else:
            _info_dict = []
            _bbox_show_info = False
        
        c1, c2 = st.columns(2)
        with c1:
            _bbox_format = st.selectbox("Bounding Box Format", 
                                        ["XYWH", "XYXY", "CXYWH", "REL_XYWH", "REL_XYXY", "REL_CXYWH"],
                                        key="bbox_format")
    
    with st.expander("UI Size"):
        c1, c2, c3 = st.columns(3)
        with c1:
            _ui_size = st.selectbox("UI Size", ("small", "medium", "large"), key="ui_size")
    
        c1, c2, c3 = st.columns(3)
        with c1:
            _ui_left_size = st.selectbox("Left UI Size", (None, "small", "medium", "large", "custom"), key="ui_left_size")
        if _ui_left_size == "custom":
            with c2:
                _ui_left_size = st.number_input("Left Size (px)", min_value=0, value=198, key="left_size")
    
        c1, c2, c3 = st.columns(3)
        with c1:
            _ui_bottom_size = st.selectbox("Bottom UI Size", (None, "small", "medium", "large", "custom"), key="ui_bottom_size")
        if _ui_bottom_size == "custom":
            with c2:
                _ui_bottom_size = st.number_input("Bottom Size (px)", min_value=0, value=198, key="bottom_size")
    
        c1, c2, c3 = st.columns(3)
        with c1:
            _ui_right_size = st.selectbox("Right UI Size", (None, "small", "medium", "large", "custom"), key="ui_right_size")
        if _ui_right_size == "custom":
            with c2:
                _ui_right_size = st.number_input("Right Size (px)", min_value=0, value=34, key="right_size")
    
    with st.expander("UI Settings & Position"):
        c1, c2, c3 = st.columns(3)
        with c1:
            _comp_alignment = st.selectbox("Component Alignment", ("left", "center", "right"), key="comp_alignment")
        c1, c2, c3 = st.columns(3)
        with c1:
            _ui_position = st.selectbox("UI Position", ("left", "right"), key="ui_position")
        with c2:
            _line_width = st.number_input("Line Width", min_value=0.5, value=1.0, step=0.1, key="line_width")
        with c3:
            _read_only = st.toggle("Read-Only Mode", False, key="read_only")
    
        c1, c2, c3 = st.columns(3)
        with c1:
            _class_select_type = st.radio("Class Select Type", ("radio", "select"), key="class_select_type")
        with c2:
            _class_select_position = st.selectbox("Class Select Position", (None, "left", "right", "bottom"), key="class_select_position")
    
        c1, c2, c3 = st.columns(3)
        with c1:
            _item_editor = st.toggle("Enable Item Editor", True, key="item_editor")
        if _item_editor:
            with c2:
                _item_editor_position = st.selectbox("Item Editor Position", (None, "left", "right"), index=2, key="item_editor_position")
            with c3:
                _edit_description = st.toggle("Edit Description", key="edit_description")
                _edit_meta = st.toggle("Edit Meta Data", key="edit_meta")
        else:
            _item_editor_position = None
            _edit_description = False
            _edit_meta = False
    
        c1, c2, c3 = st.columns(3)
        with c1:
            _item_selector = st.toggle("Enable Item Selector", True, key="item_selector")
        if _item_selector:
            with c2:
                _item_selector_position = st.selectbox("Item Selector Position", (None, "left", "right"), index=2, key="item_selector_position")
        else:
            _item_selector_position = None
    
    with st.expander("API"):
       
        image_path_list = st.session_state.image_path_list
        frame_index = st.session_state.frame_index

    
        original_format = _bbox_format.replace("REL_", "")
        # _bbox_converted = [convert_bbox_format(bbox, "XYWH", original_format) for bbox in _bbox]
        # if "REL" in _bbox_format:
        #     _bbox_converted = [
        #         absolute_to_relative(bbox, _width, _height)
        #         for bbox in _bbox_converted
        #     ]
            
        # result_dict = {}
        # for img in batch_images:
        #     result_dict[img] = {"bboxes": st.session_state.bboxes_xyxy, "labels": st.session_state.classes}
        # st.session_state["result"] = result_dict.copy()
    
        function_args = [
            "\timage_path=image_path",
            f"label_list={st.session_state.label_list }",
            f"bboxes=st.session_state.bboxes_xyxy",
        ]
    
        
        function_args.append(f"bbox_ids={st.session_state.bbox_ids}")
        function_args.append(f"bbox_format='XYWH'")
        function_args.append(f"meta_data=False")
        function_args.append(f"image_height={st.session_state.image_height}")
        function_args.append(f"image_width={st.session_state.image_width}")
        if _ui_position != "left":
            function_args.append(f"ui_position={repr(_ui_position)}")
        if _class_select_position:
            function_args.append(f"class_select_position={repr(_class_select_position)}")
        if _item_editor_position:
            function_args.append(f"item_editor_position={repr(_item_editor_position)}")
        if _item_selector_position:
            function_args.append(f"item_selector_position={repr(_item_selector_position)}")
        if _class_select_type != "select":
            function_args.append(f"class_select_type={repr(_class_select_type)}")
        if _item_editor:
            function_args.append(f"item_editor={_item_editor}")
        if _item_selector:
            function_args.append(f"item_selector={_item_selector}")
        if _edit_description:
            function_args.append(f"edit_description={_edit_description}")
        if _edit_meta:
            function_args.append(f"edit_meta={_edit_meta}")
        if _ui_size != "small":
            function_args.append(f"ui_size={repr(_ui_size)}")
        if _ui_left_size:
            function_args.append(f"ui_left_size={repr(_ui_left_size)}")
        if _ui_bottom_size:
            function_args.append(f"ui_bottom_size={repr(_ui_bottom_size)}")
        if _ui_right_size:
            function_args.append(f"ui_right_size={repr(_ui_right_size)}")
        if _bbox_show_info:
            function_args.append(f"bbox_show_info={repr(_bbox_show_info)}")
        if _bbox_show_label:
            function_args.append(f"bbox_show_label={repr(_bbox_show_label)}")
        if _read_only:
            function_args.append(f"read_only={repr(_read_only)}")
        if _comp_alignment != "left":
            function_args.append(f"component_alignment={repr(_comp_alignment)}")
    
        # Use a dynamic key based on the current page so that each image reloads its labels.
        function_args.append(f"key='detection_component_{frame_index}'")
        final_function_call = "detection(\n" + ",\n\t".join(function_args) + "\n)"
    
        st.code(f"result = {final_function_call}", language="python")
    
        st.session_state["config"] = {
            "image_height": st.session_state.image_height,
            "image_width": st.session_state.image_width,
            "label_list": st.session_state.label_list ,
            "bbox_show_label": _bbox_show_label,
            "info_dict": _info_dict,
            "meta_data": _meta,
            "ui_size": _ui_size,
            "ui_left_size": _ui_left_size,
            "ui_bottom_size": _ui_bottom_size,
            "ui_right_size": _ui_right_size,
            "component_alignment": _comp_alignment,
            "ui_position": _ui_position,
            "line_width": _line_width,
            "read_only": _read_only,
            "class_select_type": _class_select_type,
            "class_select_position": _class_select_position,
            "item_editor": _item_editor,
            "item_editor_position": _item_editor_position,
            "edit_description": _edit_description,
            "edit_meta": _edit_meta,
            "item_selector": _item_selector,
            "item_selector_position": _item_selector_position,
            "bbox_format": _bbox_format,
            "bbox": st.session_state.bboxes_xyxy,
            "bbox_ids": st.session_state.bbox_ids,
            "bbox_show_info": _bbox_show_info,
        }

# ----------------------- Detection Tab -----------------------
with st.session_state.tabs[1]:
    frame_index = st.session_state.frame_index
    # batch_start = st.session_state.batch_start
    # if frame_index - batch_start < PAD or (batch_start + MAX_IMAGES - 1) - frame_index < PAD:
    #     new_batch_start = frame_index - MAX_IMAGES // 2
    #     new_batch_start = max(0, new_batch_start)
    #     new_batch_start = min(new_batch_start, len(image_path_list) - MAX_IMAGES)
    #     st.session_state.batch_start = new_batch_start
    #     batch_start = new_batch_start

    # batch_images = image_path_list[batch_start:batch_start+MAX_IMAGES]
    # _bbox = st.session_state["config"].get("bbox", [[0, 0, 200, 100], [10, 20, 100, 150]])
    # result_dict = {}
    # for img in batch_images:
    #     result_dict[img] = {"bboxes": _bbox, "labels": [0, 0]}
    # st.session_state["result"] = result_dict.copy()
    
    target_image_path = image_path_list[frame_index]
    config = st.session_state.get("config", {})

    # Determine the labels directory (assumed to be a sibling of the "images" folder)
    
    image_dir = st.session_state.image_dir
    labels_dir = os.path.join(os.path.dirname(image_dir), "labels")
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    
    target_base = os.path.splitext(os.path.basename(target_image_path))[0]
    label_file = os.path.join(labels_dir, target_base + ".txt")
    
    # Open the target image to determine its actual size.
    target_img = Image.open(target_image_path)
    img_w, img_h = target_img.size

    # Read the YOLO-format labels (rows of: class x y w h normalized).
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            lines = f.readlines()
        bboxes_from_file = []
        label_ids_from_file = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])
                if config.get("bbox_format", "XYWH") == "XYWH":
                    x_center_abs = x_center * img_w
                    y_center_abs = y_center * img_h
                    w_abs = w * img_w
                    h_abs = h * img_h
                    bbox = [x_center_abs - w_abs / 2, y_center_abs - h_abs / 2, w_abs, h_abs]
                else:
                    bbox = [x_center, y_center, w, h]
                bboxes_from_file.append(bbox)
                label_ids_from_file.append(cls)
    else:
        with open(label_file, "w") as f:
            f.write("")
        bboxes_from_file = []
        label_ids_from_file = []

    st.write("File Name:", os.path.basename(target_image_path))
    st.write("Label File Path:", label_file)
    
    st.text("Component")
    # Get the detection output and store it using a key that includes the current page.
    detection_out = detection(
        image_path=target_image_path,
        bboxes=bboxes_from_file,
        bbox_format=config.get("bbox_format", "XYWH"),
        bbox_ids=config.get("bbox_ids", None),
        labels=label_ids_from_file,
        info_dict=config.get("info_dict", []),
        meta_data=config.get("meta_data", []),
        label_list=st.session_state.label_list,
        line_width=config.get("line_width", 1.0),
        class_select_type=config.get("class_select_type", "select"),
        item_editor=config.get("item_editor", False),
        item_selector=config.get("item_selector", False),
        edit_meta=config.get("edit_meta", False),
        edit_description=config.get("edit_description", False),
        ui_position=config.get("ui_position", "left"),
        class_select_position=config.get("class_select_position", None),
        item_editor_position=config.get("item_editor_position", None),
        item_selector_position=config.get("item_selector_position", None),
        image_height=img_h,
        image_width=img_w,
        ui_size=config.get("ui_size", "small"),
        ui_left_size=config.get("ui_left_size", None),
        ui_bottom_size=config.get("ui_bottom_size", None),
        ui_right_size=config.get("ui_right_size", None),
        bbox_show_info=config.get("bbox_show_info", False),
        bbox_show_label=config.get("bbox_show_label", True),
        read_only=config.get("read_only", False),
        component_alignment=config.get("component_alignment", "left"),
        key=f"detection_component_{frame_index}",
    )
    st.session_state[f"detection_component_{frame_index}_out"] = detection_out

    # Function to save the current image's labels to its txt file.
    def save_current_labels():
        current_img_path = image_path_list[st.session_state.frame_index]
        target_base = os.path.splitext(os.path.basename(current_img_path))[0]
        labels_dir = os.path.join(os.path.dirname(image_dir), "labels")
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        label_file_path = os.path.join(labels_dir, target_base + ".txt")
        current_img = Image.open(current_img_path)
        curr_w, curr_h = current_img.size
        out = st.session_state.get(f"detection_component_{st.session_state.frame_index}_out", None)
        if out is not None:
            new_bboxes = out.get("bboxes", [])
            new_labels = out.get("labels", [])
            # Always write the file if the lengths match, even if empty
            if len(new_bboxes) == len(new_labels):
                with open(label_file_path, "w") as f:
                    for cls, bbox in zip(new_labels, new_bboxes):
                        x, y, w, h = bbox
                        x_center = (x + w / 2) / curr_w
                        y_center = (y + h / 2) / curr_h
                        w_norm = w / curr_w
                        h_norm = h / curr_h
                        f.write(f"{cls} {x_center} {y_center} {w_norm} {h_norm}\n")
    
    # Navigation controls: Save labels before navigating away.
    jump_page = st.number_input("Jump to Image", min_value=0, max_value=len(image_path_list)-1,
                                value=st.session_state.frame_index, step=10, key="jump_page")
    if jump_page != st.session_state.frame_index:
        save_current_labels()
        st.session_state.frame_index = jump_page
        st.rerun()
    
    col_prev, col_slider, col_next = st.columns([1, 10, 2])
    with col_prev:
        if st.button("Previous", key="prev_btn"):
            if st.session_state.frame_index > 0:
                save_current_labels()
                st.session_state.frame_index -= 1
                st.rerun()
    with col_slider:
        slider_page = st.slider("Page", 0, len(image_path_list) - 1,
                                st.session_state.frame_index, key="slider_det")
        if slider_page != st.session_state.frame_index:
            save_current_labels()
            st.session_state.frame_index = slider_page
            st.rerun()
    with col_next:
        if st.button("Next", key="next_btn"):
            if st.session_state.frame_index < len(image_path_list) - 1:
                save_current_labels()
                st.session_state.frame_index += 1
                st.rerun()

    # Re-read and display the current label file's content at the bottom.
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            label_file_content = f.read()
    else:
        label_file_content = ""
    st.markdown("**Current Label File Content:**")
    st.code(label_file_content, language="text")

# ----------------------- GPU Status Tab -----------------------
with st.session_state.tabs[2]:
    st.write("Click the button below to check the GPU status on Lambda 2.")
    
    if st.button("Check GPU Status"):
        try:
            output = subprocess.check_output(["gpustat"]).decode("utf-8")
            st.text(output)
        except Exception as e:
            st.error(f"Failed to run gpustat: {e}")
