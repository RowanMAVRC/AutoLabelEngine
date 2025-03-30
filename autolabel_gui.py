# autolabel_gui.py
import streamlit as st
import subprocess
import time
import os
import glob
import pandas as pd

from modules.config_loader import load_config
from modules import file_utils, tmux_utils, image_utils, label_utils, navigator, code_editor

# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
if "session_running" not in st.session_state:
    st.session_state.session_running = True
    st.session_state.playback_active = False
    st.session_state.fps = 1
    st.session_state.video_index = 0
    st.session_state.include_labels = True
    st.session_state.video_image_scale = 1.0

    st.set_page_config(layout="wide")

    # Load configuration from cfgs/gui/paths/default.yaml
    st.session_state.paths = load_config("cfgs/gui/paths/default.yaml")

    st.session_state.use_subset = False
    st.session_state.subset_frames = []
    st.session_state.subset_index = 0
    st.session_state.automatic_generate_list = False

    # Load configuration into session_state (e.g., paths)
    file_utils.load_session_state(st.session_state, default_yaml_path="cfgs/gui/paths/default.yaml")

    try:
        gpu_info = subprocess.check_output("nvidia-smi -L", shell=True).decode("utf-8")
        st.session_state.gpu_list = [line.strip() for line in gpu_info.splitlines() if line.strip()]
    except Exception:
        st.session_state.gpu_list = []

# -----------------------------------------------------------------------------
# Define Tabs for the App
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "Generate Datasets",
    "Auto Label",
    "Manual Labeling",
    "Finetune Model",
    "Linux Terminal"
])

# -----------------------------------------------------------------------------
# Tab 0: Generate Datasets
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("Generate Datasets")
    action_option = st.radio(
        "Choose action:",
        ["Upload Data", "Convert MP4 to PNGs", "Rotate Image Dataset", "Split YOLO Dataset", "Combine YOLO Datasets"],
        key="action_option",
        label_visibility="visible"
    )
    if action_option == "Upload Data":
        with st.expander("Upload Data"):
            st.subheader("Save Path")
            navigator.path_navigator("upload_save_path")
            file_utils.upload_to_dir(st.session_state.paths["upload_save_path"], st)
    elif action_option == "Convert MP4 to PNGs":
        with st.expander("Convert MP4 to PNGs"):
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("MP4 Path")
                navigator.path_navigator("mp4_path", button_and_selectbox_display_size=[4, 30])
            with c2:
                st.subheader("Save Path")
                save_path_option = st.radio("Choose save path option:", ["Default", "Custom"],
                                            key="mp4_save_radio", label_visibility="collapsed")
                key = "mp4_save_path"
                if save_path_option == "Default":
                    st.session_state.paths[key] = st.session_state.paths["mp4_path"].replace(".mp4", "/images/")
                    st.write(f"**Current MP4 Save Path:** {st.session_state.paths[key]}")
                else:
                    navigator.path_navigator(key, button_and_selectbox_display_size=[4, 30])
            with st.expander("Venv Path"):
                navigator.path_navigator("venv_path", radio_button_prefix="convert_mp4")
            with st.expander("Script"):
                navigator.path_navigator("mp4_script_path")
                code_editor.python_code_editor("mp4_script_path", st)
            with st.expander("Run Conversion"):
                if st.button("Begin Converting", key="begin_converting_data_btn"):
                    tmux_utils.run_in_tmux(
                        session_key="mp4_data",
                        py_file_path=st.session_state.paths["mp4_script_path"],
                        venv_path=st.session_state.paths["venv_path"],
                        args={"video_path": st.session_state.paths["mp4_path"],
                              "output_folder": st.session_state.paths["mp4_save_path"]}
                    )
                    time.sleep(3)
                    output = tmux_utils.update_tmux_terminal("mp4")
                    st.code(output, language="bash")
    elif action_option == "Rotate Image Dataset":
        with st.expander("Rotate Image Dataset"):
            st.subheader("Image Path")
            navigator.path_navigator("rotate_images_path", button_and_selectbox_display_size=[4, 30])
            with st.expander("Venv Path"):
                navigator.path_navigator("venv_path", radio_button_prefix="rotate_images")
            with st.expander("Script"):
                navigator.path_navigator("rotate_images_script_path")
                code_editor.python_code_editor("rotate_images_script_path", st)
            with st.expander("Run Rotation"):
                if st.button("Begin Rotating Images", key="begin_rotating_data_btn"):
                    tmux_utils.run_in_tmux(
                        session_key="rotate_images",
                        py_file_path=st.session_state.paths["rotate_images_script_path"],
                        venv_path=st.session_state.paths["venv_path"],
                        args={"directory": st.session_state.paths["rotate_images_path"], "rotation": "CW"}
                    )
                    time.sleep(3)
                    output = tmux_utils.update_tmux_terminal("rotate_images")
                    st.code(output, language="bash")
    elif action_option == "Split YOLO Dataset":
        with st.expander("Split YOLO Dataset"):
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Dataset To Be Split")
                navigator.path_navigator("split_data_path", button_and_selectbox_display_size=[4, 30])
            with c2:
                st.subheader("Save Path")
                save_path_option = st.radio("Choose save path option:", ["Default", "Custom"],
                                            key="split_save_radio", label_visibility="collapsed")
                key = "split_data_save_path"
                if save_path_option == "Default":
                    st.session_state.paths[key] = st.session_state.paths["split_data_path"]
                    st.write(f"**Current Save Path:** {st.session_state.paths[key]}")
                else:
                    navigator.path_navigator(key, button_and_selectbox_display_size=[4, 30])
            with st.expander("Venv Path"):
                navigator.path_navigator("venv_path", radio_button_prefix="split_data")
            with st.expander("Script"):
                navigator.path_navigator("split_data_script_path")
                code_editor.python_code_editor("split_data_script_path", st)
            with st.expander("Run Split"):
                if st.button("Begin Splitting Data", key="begin_split_data_btn"):
                    tmux_utils.run_in_tmux(
                        session_key="split_data",
                        py_file_path=st.session_state.paths["split_data_script_path"],
                        venv_path=st.session_state.paths["venv_path"],
                        args={"data_path": st.session_state.paths["split_data_path"],
                              "save_path": st.session_state.paths["split_data_save_path"]}
                    )
                    time.sleep(3)
                    output = tmux_utils.update_tmux_terminal("split_data")
                    st.code(output, language="bash")
    elif action_option == "Combine YOLO Datasets":
        with st.expander("Combine YOLO Datasets"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("Dataset 1")
                navigator.path_navigator("combine_dataset_1_path", button_and_selectbox_display_size=[4, 30])
            with c2:
                st.subheader("Dataset 2")
                navigator.path_navigator("combine_dataset_2_path", button_and_selectbox_display_size=[4, 30])
            with c3:
                st.subheader("Save Path")
                navigator.path_navigator("combine_dataset_save_path", button_and_selectbox_display_size=[4, 30])
            with st.expander("Script"):
                navigator.path_navigator("combine_dataset_script_path")
                code_editor.python_code_editor("combine_dataset_script_path", st)
            with st.expander("Run Combine"):
                if st.button("Begin Combining Data", key="begin_combine_dataset_btn"):
                    tmux_utils.run_in_tmux(
                        session_key="combine_dataset",
                        py_file_path=st.session_state.paths["combine_dataset_script_path"],
                        venv_path=st.session_state.paths["venv_path"],
                        args={
                            "dataset1": st.session_state.paths["combine_dataset_1_path"],
                            "dataset2": st.session_state.paths["combine_dataset_2_path"],
                            "dst_dir": st.session_state.paths["combine_dataset_save_path"]
                        }
                    )
                    time.sleep(3)
                    output = tmux_utils.update_tmux_terminal("combine_dataset")
                    st.code(output, language="bash")

# -----------------------------------------------------------------------------
# Tab 1: Auto Label
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("Auto Label")
    with st.expander("Auto Label Settings"):
        st.subheader("Model Weights Path")
        navigator.path_navigator("auto_label_model_weight_path")
        st.subheader("Images Path")
        navigator.path_navigator("auto_label_data_path")
        st.subheader("Label Save Path")
        navigator.path_navigator("auto_label_save_path")
        st.subheader("Venv Path")
        navigator.path_navigator("venv_path", radio_button_prefix="auto_label")
    with st.expander("Script"):
        navigator.path_navigator("auto_label_script_path")
        code_editor.python_code_editor("auto_label_script_path", st)
    with st.expander("Run Auto Label"):
        c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
        with c1:
            output = tmux_utils.check_gpu_status("auto_label_check_gpu_status_button")
            if output:
                st.code(output, language="bash")
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
                tmux_utils.run_in_tmux(
                    session_key="auto_label_data",
                    py_file_path=st.session_state.paths["auto_label_script_path"],
                    venv_path=st.session_state.paths["venv_path"],
                    args={
                        "model_weights_path": st.session_state.paths["auto_label_model_weight_path"],
                        "images_dir_path": st.session_state.paths["auto_label_data_path"],
                        "labels_save_path": st.session_state.paths["auto_label_save_path"],
                        "gpu_number": st.session_state.auto_label_gpu
                    }
                )
                time.sleep(3)
                output = tmux_utils.update_tmux_terminal("auto_label_data")
                st.code(output, language="bash")
        with c4:
            if st.button("Update Terminal Output", key="check_auto_labeling_data_btn"):
                output = tmux_utils.update_tmux_terminal("auto_label_data")
                st.code(output, language="bash")
        with c5:
            if st.button("Clear Terminal Output", key="auto_labeling_clear_terminal_btn"):
                output = ""
                st.code(output, language="bash")
        with c6:
            if st.button("Kill TMUX Session", key="auto_labeling_kill_tmux_session_btn"):
                output = tmux_utils.kill_tmux_session("auto_label_data")
                st.code(output, language="bash")

# -----------------------------------------------------------------------------
# Tab 2: Manual Labeling
# -----------------------------------------------------------------------------
with tabs[2]:
    st.header("Manual Labeling")
    with st.expander("Settings"):
        st.subheader("Image Scale")
        image_scale = st.number_input("Image Scale", value=1.0, step=0.25, label_visibility="collapsed")
        if float(image_scale) != st.session_state.get("unverified_image_scale", 1.0):
            st.session_state.unverified_image_scale = image_scale
            st.session_state["skip_label_update"] = True
            st.rerun()
        st.subheader("Images Path")
        navigator.path_navigator("unverified_images_path", button_and_selectbox_display_size=[1, 25])
        st.subheader("Label Names YAML Path")
        navigator.path_navigator("unverified_names_yaml_path", button_and_selectbox_display_size=[2, 25])
    with st.expander("Subset Selection"):
        st.markdown("""
            <style>
            .stCheckbox input[type="checkbox"] {
                transform: scale(1.5);
            }
            .stCheckbox label {
                font-size: 24px;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)
        use_subset_val = st.checkbox("Use Subset", value=st.session_state.use_subset, key="subset_subset_btn")
        if not len(st.session_state.subset_frames) > 1:
            st.warning("Subset needs to be two or larger.")
        st.subheader("Choose CSV Path for Subset")
        navigator.path_navigator("unverified_subset_csv_path")
        csv_file = st.session_state.paths["unverified_subset_csv_path"]
        if os.path.exists(csv_file):
            st.session_state.subset_frames = file_utils.load_subset_frames(csv_file)
            subset_df = pd.DataFrame(st.session_state.subset_frames, columns=["Frame Index"])
            subset_df.insert(0, "Subset Index", range(1, len(subset_df)+1))
            st.write("Subset Indices:")
            st.write(subset_df)
            def add_frame_callback(key):
                add_val = st.session_state[key]
                if add_val not in st.session_state.subset_frames:
                    st.session_state.subset_frames.append(add_val)
                    file_utils.save_subset_frames(csv_file, st.session_state.subset_frames)
                    st.session_state["skip_label_update"] = True
            def remove_frame_callback():
                remove_val = st.session_state["remove_frame"]
                if remove_val in st.session_state.subset_frames:
                    st.session_state.subset_frames.remove(remove_val)
                    file_utils.save_subset_frames(csv_file, st.session_state.subset_frames)
                    st.session_state["skip_label_update"] = True
            if st.session_state.get("max_images", 0) > 0:
                c1, c2 = st.columns([10, 10])
                with c1:
                    st.number_input("Add Frame Index", min_value=0, max_value=st.session_state.max_images - 1,
                                    value=0, step=1, key="add_frame_1", on_change=add_frame_callback, args=("add_frame_1",))
                with c2:
                    st.number_input("Remove Frame Index", min_value=0, max_value=st.session_state.max_images - 1,
                                    value=0, step=1, key="remove_frame", on_change=remove_frame_callback)
            else:
                st.warning("No images available.")
            base, ext = os.path.splitext(csv_file)
            default_copy_path = base + "_copy" + ext
            new_save_path = st.text_input("Enter path for new CSV copy", value=default_copy_path)
            if st.button("Copy CSV to new file"):
                if new_save_path:
                    try:
                        file_utils.save_subset_frames(new_save_path, st.session_state.subset_frames)
                        st.success(f"Subset CSV copied to {new_save_path}")
                    except Exception as e:
                        st.error(f"Error copying file: {e}")
                else:
                    st.error("Please enter a valid new file path.")
        else:
            st.info("No CSV found. Create or upload a CSV to begin using a subset.")
    with st.expander("Frame by Frame Label Review"):
        if st.session_state.get("max_images", 0) > 0:
            if st.session_state.max_images > 1:
                col_prev, _, col_next = st.columns([4, 5, 4])
                with col_prev:
                    st.button("Prev Frame", key="top_prev_btn")
                with col_next:
                    st.button("Next Frame", key="top_next_btn")
                col_copy_prev, _, col_copy_next = st.columns([4, 5, 4])
                with col_copy_prev:
                    st.button("Copy Labels from Prev Slide", key="copy_prev_btn",
                              on_click=label_utils.copy_prev_labels, args=(st,))
                with col_copy_next:
                    st.button("Copy Labels from Next Slide", key="copy_next_btn",
                              on_click=label_utils.copy_next_labels, args=(st,))
            label_utils.update_unverified_frame(st)
            st.write(f"Current File Path: {st.session_state.image_path}")
            # For demonstration, we set an empty detection output.
            st.session_state.out = {}
            label_utils.update_labels_from_detection(st)
        else:
            st.warning("Data Path is empty...")

# -----------------------------------------------------------------------------
# Tab 3: Finetune Model
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("Finetune Model")
    with st.expander("Data YAML"):
        navigator.path_navigator("train_data_yaml_path")
        code_editor.yaml_editor("train_data_yaml_path", st)
    with st.expander("Model YAML"):
        navigator.path_navigator("train_model_yaml_path")
        code_editor.yaml_editor("train_model_yaml_path", st)
    with st.expander("Train YAML Path"):
        navigator.path_navigator("train_train_yaml_path")
        code_editor.yaml_editor("train_train_yaml_path", st)
    with st.expander("Venv Path"):
        navigator.path_navigator("venv_path", radio_button_prefix="train")
    with st.expander("Script"):
        navigator.path_navigator("train_script_path")
        code_editor.python_code_editor("train_script_path", st)
    with st.expander("Run Finetuning"):
        c1, c2, c3, c4, c5 = st.columns(5, gap="small")
        with c1:
            output = tmux_utils.check_gpu_status("train_check_gpu_status_button")
            if output:
                st.code(output, language="bash")
        with c2:
            if st.button("Begin Training", key="begin_train_btn"):
                tmux_utils.run_in_tmux(
                    session_key="auto_label_trainer",
                    py_file_path=st.session_state.paths["train_script_path"],
                    venv_path=st.session_state.paths["venv_path"],
                    args={
                        "data_path": st.session_state.paths["train_data_yaml_path"],
                        "model_path": st.session_state.paths["train_model_yaml_path"],
                        "train_path": st.session_state.paths["train_train_yaml_path"]
                    }
                )
                time.sleep(3)
                output = tmux_utils.update_tmux_terminal("auto_label_trainer")
                st.code(output, language="bash")
        with c3:
            if st.button("Check Training", key="check_train_btn"):
                output = tmux_utils.update_tmux_terminal("auto_label_trainer")
                st.code(output, language="bash")
        with c4:
            if st.button("Clear Terminal Output", key="clear_terminal_btn"):
                output = ""
                st.code(output, language="bash")
        with c5:
            if st.button("Kill TMUX Session", key="kill_tmux_session_btn"):
                output = tmux_utils.kill_tmux_session("auto_label_trainer")
                st.code(output, language="bash")

# -----------------------------------------------------------------------------
# Tab 4: Linux Terminal
# -----------------------------------------------------------------------------
with tabs[4]:
    st.header("Linux Terminal")
    if "terminal_text" not in st.session_state:
        st.session_state.terminal_text = ""
    output_placeholder = st.empty()
    def run_command_and_accumulate(command):
        try:
            process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
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
            st.session_state.terminal_text = f"$ {command}\n"
            output_placeholder.code(st.session_state.terminal_text, language="bash")
            run_command_and_accumulate(command)
        else:
            output_placeholder.warning("Please enter a valid command.")
    st.text_input("Enter a Linux command:", "", key="command_input", on_change=local_run_callback)
    output_placeholder.code(st.session_state.terminal_text, language="bash")


# -----------------------------------------------------------------------------
# Save settings to default
# -----------------------------------------------------------------------------
file_utils.save_session_state(st.session_state, default_yaml_path="cfgs/gui/paths/default.yaml")
