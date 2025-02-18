#
# Streamlit components for general labeling tasks
#
# Copyright (c) 2024 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only
#

import streamlit as st
from glob import glob
from streamlit_label_kit import detection, absolute_to_relative, convert_bbox_format

def wide_space_default():
    st.set_page_config(layout="wide")

wide_space_default()

# Define constants and load images
label_list = ["deer", "human", "dog", "penguin", "flamingo", "teddy bear"]
image_path_list = glob("image/*.jpg")
image_size = [700, 467]
DEFAULT_HEIGHT = 512
DEFAULT_LINE_WIDTH = 1.0

# Ensure we have a page index in session state
if "num_page" not in st.session_state:
    st.session_state.num_page = 1

# Create two tabs: one for configuration, one for the detection component.
tabs = st.tabs(["Configure", "Detection"])

# ----------------------- Configure Tab -----------------------
with tabs[0]:
    # Use the current page to get the target image (used in the API expander)
    target_image_path = image_path_list[st.session_state.num_page]
    
    st.header("Configure Component")
    
    with st.expander("Image & Inputs"):
        c1, c2 = st.columns(2)
        with c1:
            _height = st.number_input("image_height (px)", min_value=0, value=DEFAULT_HEIGHT, key="height_input")
        with c2:
            _width = st.number_input("image_width (px)", min_value=0, value=DEFAULT_HEIGHT, key="width_input")
        
        _label_list = st.multiselect("Label List", options=label_list, default=label_list, key="label_list")
        _bbox_show_label = st.toggle("bbox_show_label", True, key="bbox_show_label")
        
        c1, c2 = st.columns(2)
        with c1:
            _info_dict_help = st.toggle("Info Dict", help='value = [{"Confidence": 0.1, "testScore": 0.98}, {"Confidence": 0.2}]', key="info_dict_help")
        with c2:
            _meta_help = st.toggle("MetaData/Description", help='value = [["meta/description1", "meta1", "meta2"], ["meta/description2"]]', key="meta_help")
        if _meta_help:
            _meta = [["meta/description1", "meta1", "meta2"], ["meta/description2"]]
        else:
            _meta = []
        
        if _info_dict_help:
            _info_dict = [{"Confidence": 0.1, "testScore": 0.98}, {"Confidence": 0.2}]
            _bbox_show_info = st.toggle("bbox_show_info", True, key="bbox_show_info")
        else:
            _info_dict = []
            _bbox_show_info = False
        
        c1, c2 = st.columns(2)
        with c1:
            _bbox_format = st.selectbox("bbox_format", 
                                        ["XYWH", "XYXY", "CXYWH", "REL_XYWH", "REL_XYXY", "REL_CXYWH"],
                                        key="bbox_format")
    
    with st.expander("Ui Size"):
        c1, c2, c3 = st.columns(3)
        with c1: 
            _ui_size = st.selectbox("ui_size", ("small", "medium", "large"), key="ui_size")
    
        c1, c2, c3 = st.columns(3)
        with c1: 
            _ui_left_size = st.selectbox("ui_left_size", (None, "small", "medium", "large", "custom"), key="ui_left_size")
        if _ui_left_size == "custom":
            with c2: 
                _ui_left_size = st.number_input("left_size (px)", min_value=0, value=198, key="left_size")
    
        c1, c2, c3 = st.columns(3)
        with c1: 
            _ui_bottom_size = st.selectbox("ui_bottom_size", (None, "small", "medium", "large", "custom"), key="ui_bottom_size")
        if _ui_bottom_size == "custom":
            with c2: 
                _ui_bottom_size = st.number_input("bottom_size (px)", min_value=0, value=198, key="bottom_size")
            
        c1, c2, c3 = st.columns(3)
        with c1: 
            _ui_right_size = st.selectbox("ui_right_size", (None, "small", "medium", "large", "custom"), key="ui_right_size")
        if _ui_right_size == "custom":
            with c2: 
                _ui_right_size = st.number_input("right_size (px)", min_value=0, value=34, key="right_size")
        
    with st.expander("UI Setting & Position"):
        c1, c2, c3 = st.columns(3)
        with c1: 
            _comp_alignment = st.selectbox("component_alignment", ("left", "center", "right"), key="comp_alignment")
        
        c1, c2, c3 = st.columns(3)
        with c1: 
            _ui_position = st.selectbox("ui_position", ("left", "right"), key="ui_position")
        with c2: 
            _line_width = st.number_input("line_width", min_value=0.5, value=DEFAULT_LINE_WIDTH, step=0.1, key="line_width")
        with c3: 
            _read_only = st.toggle("read_only", False, key="read_only")
        
        c1, c2, c3 = st.columns(3)
        with c1: 
            _class_select_type = st.radio("class_select_type", ("select", "radio"), key="class_select_type")
        with c2: 
            _class_select_position = st.selectbox("class_select_position", (None, "left", "right", "bottom"), key="class_select_position")
    
        c1, c2, c3 = st.columns(3)
        with c1: 
            _item_editor = st.toggle("item_editor", False, key="item_editor")
        if _item_editor:
            with c2: 
                _item_editor_position = st.selectbox("item_editor_position", (None, "left", "right"), key="item_editor_position")
            with c3:
                _edit_description = st.toggle("edit_description", key="edit_description")
                _edit_meta = st.toggle("edit_meta", key="edit_meta")
        else:
            _item_editor_position = None
            _edit_description = False
            _edit_meta = False
            
        c1, c2, c3 = st.columns(3)
        with c1: 
            _item_selector = st.toggle("item_selector", False, key="item_selector")
        if _item_selector:
            with c2: 
                _item_selector_position = st.selectbox("item_selector_position", (None, "left", "right"), key="item_selector_position")
        else:
            _item_selector_position = None
    
    with st.expander("API"):
        # Prepare the bounding boxes
        _bbox = [[0, 0, 200, 100], [10, 20, 100, 150]]
        _bbox_id = ["bbox-" + str(i) for i in range(len(_bbox))]
    
        original_format = _bbox_format.replace("REL_", "")
        _bbox = [convert_bbox_format(bbox, "XYWH", original_format) for bbox in _bbox]
        if "REL" in _bbox_format:
            _bbox = [
                absolute_to_relative(bbox, image_size[0], image_size[1])
                for bbox in _bbox
            ]
            
        # Build a result dictionary for each image
        result_dict = {}
        for img in image_path_list:
            result_dict[img] = {
                "bboxes": _bbox,
                "labels": [0, 0],
            }
        st.session_state["result"] = result_dict.copy()
    
        # Create the function call string
        function_args = [
            "\timage_path=image_path",
            f"label_list={_label_list}",
            f"bboxes=st.session_state['result'][target_image_path]['bboxes']",
        ]
    
        if _bbox_id:
            function_args.append(f"bbox_ids={_bbox_id}")
        if _bbox_format != "XYWH":
            function_args.append(f"bbox_format={_bbox_format}")
        if _info_dict_help:
            function_args.append(f"info_dict={_info_dict}")
        if _meta_help:
            function_args.append(f"meta_data={_meta}")
        if _height != DEFAULT_HEIGHT:
            function_args.append(f"image_height={_height}")
        if _width != DEFAULT_HEIGHT:
            function_args.append(f"image_width={_width}")
        if _line_width != DEFAULT_LINE_WIDTH:
            function_args.append(f"line_width={_line_width}")
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
    
        function_args.append("key=None")
        final_function_call = "detection(\n" + ",\n\t".join(function_args) + "\n)"
    
        st.code(f"result = {final_function_call}", language="python")
    
    # Save configuration settings into session_state for use in the Detection tab
    st.session_state["config"] = {
        "image_height": _height,
        "image_width": _width,
        "label_list": _label_list,
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
        "bbox": _bbox,
        "bbox_ids": _bbox_id,
        "bbox_show_info": _bbox_show_info,
    }

# ----------------------- Detection Tab -----------------------
with tabs[1]:
    st.header("Detection Component")
    # Get the current image based on the page number
    target_image_path = image_path_list[st.session_state.num_page]
    
    # Retrieve configuration settings (or use defaults if not set)
    config = st.session_state.get("config", {})
    
    st.text("Component")
    st.session_state.out = detection(
        image_path=target_image_path,
        bboxes=st.session_state["result"][target_image_path]["bboxes"],
        bbox_format=config.get("bbox_format", "XYWH"),
        bbox_ids=config.get("bbox_ids", None),
        labels=st.session_state["result"][target_image_path]["labels"],
        info_dict=config.get("info_dict", []),
        meta_data=config.get("meta_data", []),
        label_list=config.get("label_list", label_list),
        line_width=config.get("line_width", DEFAULT_LINE_WIDTH),
        class_select_type=config.get("class_select_type", "select"),
        item_editor=config.get("item_editor", False),
        item_selector=config.get("item_selector", False),
        edit_meta=config.get("edit_meta", False),
        edit_description=config.get("edit_description", False),
        ui_position=config.get("ui_position", "left"),
        class_select_position=config.get("class_select_position", None),
        item_editor_position=config.get("item_editor_position", None),
        item_selector_position=config.get("item_selector_position", None),
        image_height=config.get("image_height", DEFAULT_HEIGHT),
        image_width=config.get("image_width", DEFAULT_HEIGHT),
        ui_size=config.get("ui_size", "small"),
        ui_left_size=config.get("ui_left_size", None),
        ui_bottom_size=config.get("ui_bottom_size", None),
        ui_right_size=config.get("ui_right_size", None),
        bbox_show_info=config.get("bbox_show_info", False),
        bbox_show_label=config.get("bbox_show_label", True),
        read_only=config.get("read_only", False),
        component_alignment=config.get("component_alignment", "left"),
        key=None,
    )
    
    # Image navigation slider (syncs with session state)
    num_page = st.slider("page", 0, len(image_path_list) - 1, st.session_state.num_page, key="slider_det")
    col_left, col_middle, col_right = st.columns([1, 12, 1])
    
    with col_left:
        if st.button("Previous", key="prev_btn"):
            if st.session_state.num_page > 0:
                st.session_state.num_page -= 1
                st.rerun()
    
    with col_right:
        if st.button("Next", key="next_btn"):
            if st.session_state.num_page < len(image_path_list) - 1:
                st.session_state.num_page += 1
                st.rerun()
    
    st.text("Component Returns")
    st.write(st.session_state.out)
