#
# Streamlit components for general labeling tasks
#
# Copyright (c) 2024 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only
#

import streamlit as st
from glob import glob
from streamlit_label_kit import detection, annotation, segmentation, absolute_to_relative, convert_bbox_format

def wide_space_default():
    st.set_page_config(layout="wide")

wide_space_default()

mode = st.tabs(["Detection"])
label_list = ["deer", "human", "dog", "penguin", "flamingo", "teddy bear"]
image_path_list = glob("image/*.jpg")

image_size = [700, 467]
DEFAULT_HEIGHT = 512
DEFAULT_LINE_WIDTH = 1.0


with mode[0]:
    # Ensure num_page is tracked in session state
    if "num_page" not in st.session_state:
        st.session_state.num_page = 1  # Default to page 1

    # Set target image based on the current page
    target_image_path = image_path_list[st.session_state.num_page]
    
    st.text("Configure Component")
    with st.expander("Image & Inputs"):
        c1, c2, = st.columns(2)
        with c1: _height = st.number_input("image_height (px)", min_value=0, value=DEFAULT_HEIGHT)
        with c2: _width = st.number_input("image_width (px)", min_value=0, value=DEFAULT_HEIGHT)
        
        _label_list = st.multiselect("Lable List", options=label_list, default=label_list)
        _bbox_show_label = st.toggle("bbox_show_label", True)
        
        c1, c2 = st.columns(2)
        with c1: _info_dict_help = st.toggle("Info Dict", help='value = [{"Confidence": 0.1, "testScore": 0.98}, {"Confidence": 0.2}]')
        with c2: _meta_help = st.toggle("MetaData/Description", help='value = [["meta/description1", "meta1", "meta2"], ["meta/description2"]]')
        if _meta_help:
            _meta = [["meta/description1", "meta1", "meta2"], ["meta/description2"]]
        else:
            _meta = []
        
        if _info_dict_help:
            _info_dict = [{"Confidence": 0.1, "testScore": 0.98}, {"Confidence": 0.2}]
            _bbox_show_info = st.toggle("bbox_show_info", True)
        else:
            _info_dict = []
            _bbox_show_info = False
            
        c1, c2 = st.columns(2)
        with c1: _bbox_format = st.selectbox("bbox_format", ["XYWH", "XYXY", "CXYWH", "REL_XYWH", "REL_XYXY", "REL_CXYWH"])
       
        
    with st.expander("Ui Size"):
        c1, c2, c3 = st.columns(3)
        with c1: _ui_size = st.selectbox("ui_size", ("small", "medium", "large"))
    
        c1, c2, c3 = st.columns(3)
        with c1: _ui_left_size = st.selectbox("ui_left_size", (None, "small", "medium", "large", "custom"))
        if _ui_left_size == "custom":
            with c2: _ui_left_size = st.number_input("left_size (px)", min_value=0, value=198)
        
        
        c1, c2, c3 = st.columns(3)
        with c1: _ui_bottom_size = st.selectbox("ui_bottom_size", (None, "small", "medium", "large", "custom"))
        if _ui_bottom_size == "custom":
            with c2: _ui_bottom_size = st.number_input("bottom_size (px)", min_value=0, value=198)
            
        c1, c2, c3 = st.columns(3)
        with c1: _ui_right_size = st.selectbox("ui_right_size", (None, "small", "medium", "large", "custom"))
        if _ui_right_size == "custom":
            with c2: _ui_right_size = st.number_input("right_size (px)", min_value=0, value=34)
        
    with st.expander("UI Setting & Position"):  
        c1, c2, c3 = st.columns(3)
        with c1: 
            _comp_alignment = st.selectbox("component_alignment", ("left", "center", "right"), key="compAlign_det")        
        
        c1, c2, c3 = st.columns(3)
        with c1: _ui_position = st.selectbox("ui_position", ("left", "right"))
        with c2: _line_width = st.number_input("line_width", min_value=0.5, value=DEFAULT_LINE_WIDTH, step=0.1)
        with c3: 
            _read_only = st.toggle("read_only", False)
    
    
        c1, c2, c3 = st.columns(3)
        with c1: _class_select_type = st.radio("class_select_type", ("select", "radio"))
        with c2: _class_select_position = st.selectbox("class_select_position", (None, "left", "right", "bottom"))
    
        c1, c2, c3 = st.columns(3)
        with c1: _item_editor = st.toggle("item_editor", False)
        if _item_editor:
            with c2: _item_editor_position = st.selectbox("item_editor_position", (None, "left", "right"))
            with c3: 
                _edit_description = st.toggle("edit_description")
                _edit_meta = st.toggle("edit_meta")
        else:
            _item_editor_position = None
            _edit_description = False
            _edit_meta = False
            
        c1, c2, c3 = st.columns(3)
        with c1: _item_selector = st.toggle("item_selector", False)
        if _item_selector:
            with c2: _item_selector_position = st.selectbox("item_selector_position", (None, "left", "right"))
        else:
            _item_selector_position = None
    
    _bbox = [[0, 0, 200, 100], [10, 20, 100, 150]]
    result_dict = {}
    _bbox_id =  ["bbox-" + str(i) for i in range(len(_bbox))]

    
    original_format = _bbox_format.replace("REL_", "")
    _bbox = [convert_bbox_format(bbox, "XYWH", original_format) for bbox in _bbox]
    if "REL" in _bbox_format:
        _bbox = [
            absolute_to_relative(bbox, image_size[0], image_size[1])
            for bbox in _bbox
        ]
        
    for img in image_path_list:
        result_dict[img] = {
            "bboxes": _bbox,
            "labels": [0, 0],
        }
    st.session_state["result"] = result_dict.copy()   
    
    # API
    function_args = [
        "\timage_path=image_path",
        f"label_list={_label_list}",
        f"bboxes={st.session_state['result'][target_image_path]['bboxes']}",
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

    if _width != DEFAULT_HEIGHT:  # Assuming DEFAULT_WIDTH is the correct constant
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


    with st.expander("api"):
        st.code(f"result = {final_function_call}", language="python")
   
    st.text("Component")
    st.session_state.out = detection(
        image_path=target_image_path,
        bboxes=st.session_state["result"][target_image_path]["bboxes"],
        bbox_format=_bbox_format,
        bbox_ids=_bbox_id,
        labels=st.session_state["result"][target_image_path]["labels"],
        info_dict=_info_dict,
        meta_data=_meta,
        label_list=_label_list,
        line_width=_line_width,
        class_select_type=_class_select_type,
        item_editor=_item_editor,
        item_selector=_item_selector,
        edit_meta=_edit_meta,
        edit_description=_edit_description,
        ui_position=_ui_position,
        class_select_position=_class_select_position,
        item_editor_position=_item_editor_position,
        item_selector_position=_item_selector_position,
        image_height=_height,
        image_width=_width,
        ui_size=_ui_size,
        ui_left_size=_ui_left_size,
        ui_bottom_size=_ui_bottom_size,
        ui_right_size=_ui_right_size,
        bbox_show_info = _bbox_show_info,
        bbox_show_label = _bbox_show_label,
        read_only=_read_only,
        component_alignment=_comp_alignment,
    )
    
    # Display slider, syncing it with session state
    num_page = st.slider("page", 0, len(image_path_list) - 1, st.session_state.num_page, key="slider_det")

    # Navigation buttons
    c1, c2 = st.columns([1, 1])

    with c1:
        if st.button("Previous", key="prev_btn"):
            if st.session_state.num_page > 0:
                st.session_state.num_page -= 1
                st.rerun()

    with c2:
        if st.button("Next", key="next_btn"):
            if st.session_state.num_page < len(image_path_list) - 1:
                st.session_state.num_page += 1
                st.rerun()
    
            
    st.text("Component Returns")
    st.session_state.out
    
    
           
    st.text("Other Examples")
 
    with st.expander("two-synchronized Example"):
        
        with st.echo():
            if "result_dup" not in st.session_state:
                st.session_state.result_dup = []
                
            if "result_dup_out1" not in st.session_state:
                st.session_state.result_dup_out1 = {"key": "0", "bbox": []}
                
            if "result_dup_out2" not in st.session_state:
                st.session_state.result_dup_out2 = {"key": "0", "bbox": []}
                
            data = st.session_state.result_dup or []
                                    
            bboxes = [item['bboxes'] for item in data]
            bbox_ids = [item['bbox_ids'] for item in data]
            labels = [item['labels'] for item in data]
            meta_data = [item['meta_data'] for item in data]
            info_dict = [item['info_dict'] for item in data]
            
            c1, c2 = st.columns(2)
            with c1: 
                test_out1 = detection(
                    image_path=target_image_path,
                    bboxes=bboxes,
                    bbox_ids=bbox_ids,
                    bbox_format=st.session_state.out["bbox_format"],
                    labels=labels,
                    info_dict=info_dict,
                    meta_data=meta_data,
                    label_list=_label_list,
                    line_width=_line_width,
                    class_select_type=_class_select_type,
                    ui_position="left",
                    item_editor=True,
                    # item_selector=True,
                    edit_meta=True,
                    bbox_show_label=True,
                    key="detection_dup1"
                )
                test_out1
            
            with c2:
                test_out2 = detection(
                    image_path=target_image_path,
                    bboxes=bboxes,
                    bbox_ids=bbox_ids,
                    bbox_format=st.session_state.out["bbox_format"],
                    labels=labels,
                    info_dict=info_dict,
                    meta_data=meta_data,
                    label_list=_label_list,
                    line_width=_line_width,
                    class_select_type=_class_select_type,
                    ui_position="right",
                    # item_editor=True,
                    item_selector=True,
                    edit_meta=True,
                    bbox_show_label=True,
                    key="detection_dup2"
                )
                test_out2
            
            if (test_out1["key"] != st.session_state.result_dup_out1["key"] or test_out2["key"] != st.session_state.result_dup_out2["key"]):
                if test_out1["key"] != st.session_state.result_dup_out1["key"]:
                    st.session_state.result_dup_out1["key"] = test_out1["key"]
                    st.session_state.result_dup_out1["bbox"] = test_out1["bbox"]
                if test_out2["key"] != st.session_state.result_dup_out2["key"]:
                    st.session_state.result_dup_out2["key"] = test_out2["key"]
                    st.session_state.result_dup_out2["bbox"] = test_out2["bbox"]
                
                
                if st.session_state.result_dup_out2["key"] > st.session_state.result_dup_out1["key"]:  
                    st.session_state.result_dup = st.session_state.result_dup_out2["bbox"]
                else:
                    st.session_state.result_dup = st.session_state.result_dup_out1["bbox"]
                
                st.rerun()
            
    with st.expander("self-synchronized Example"):
        with st.echo():
            if "self_sync" not in st.session_state:
                st.session_state.self_sync = {"key": "", "bbox": []}
                
            data = st.session_state.self_sync["bbox"] or []
                                    
            bboxes = [item['bboxes'] for item in data]
            bbox_ids = [item['bbox_ids'] for item in data]
            labels = [item['labels'] for item in data]
            meta_data = [item['meta_data'] for item in data]
            info_dict = [item['info_dict'] for item in data]
            
            result = detection(
                image_path=target_image_path,
                bboxes=bboxes,
                bbox_ids=bbox_ids,
                bbox_format=st.session_state.out["bbox_format"],
                labels=labels,
                info_dict=info_dict,
                meta_data=meta_data,
                label_list=_label_list,
                key="self_sync_output"
            )
            
            if result["key"] != st.session_state.self_sync["key"]:
                st.session_state.self_sync["key"] = result["key"]
                st.session_state.self_sync["bbox"] = result["bbox"]
                
                st.rerun()   
            
        st.write('''
            #if you want to manipulate data do:\n
                st.session_state.self_sync["bbox"] = new_bbox_info\n
                st.rerun()
                    ''')

        st.session_state.self_sync 

        
