# modules/navigator.py
import os
import streamlit as st

def path_navigator(key, radio_button_prefix="", button_and_selectbox_display_size=[2, 25]):
    """
    A file/directory navigator that works in two modes:
    - "Enter Path as Text"
    - "File Explorer" (using a selectbox and a “..” button)
    """
    current_path = st.session_state.paths.get(key, "/")
    current_path = os.path.normpath(current_path)
    save_path_option = st.radio(
        "Choose save path option:", ["Enter Path as Text", "File Explorer"],
        key=f"{radio_button_prefix}_{key}_radio",
        label_visibility="collapsed"
    )
    if save_path_option == "Enter Path as Text":
        custom_path = st.text_input(
            "Enter custom save path:",
            value=current_path,
            key=f"{radio_button_prefix}_{key}_custom_path_input",
            label_visibility="collapsed"
        )
        st.write(f"**Current {' '.join(word.capitalize() for word in key.split('_'))}:** {current_path}")
        if custom_path:
            custom_path = os.path.normpath(custom_path)
            if not os.path.exists(custom_path):
                st.warning(f"Path '{custom_path}' does not exist. Choose an option below:")
                create_col, up_col = st.columns(2)
                with create_col:
                    if st.button("Create this path", key=f"{radio_button_prefix}_{key}_create_custom"):
                        new_name = st.text_input(
                            "Optionally enter a different name for the new path:",
                            value=custom_path,
                            key=f"{radio_button_prefix}_{key}_new_path_name"
                        )
                        if new_name:
                            try:
                                os.makedirs(new_name, exist_ok=True)
                                st.session_state.paths[key] = new_name
                                st.experimental_rerun()
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
                            st.experimental_rerun()
                return custom_path
            else:
                st.session_state.paths[key] = custom_path
                return custom_path
        else:
            return st.session_state.paths.get(key, "/")
    else:
        if os.path.isfile(current_path):
            directory_to_list = os.path.dirname(current_path)
        else:
            directory_to_list = current_path
        col1, col2 = st.columns(button_and_selectbox_display_size, gap="small")
        with col1:
            go_up_button_key = f"go_up_button_{radio_button_prefix}_{key}"
            if st.button("..", key=go_up_button_key):
                if os.path.isdir(current_path):
                    parent = os.path.dirname(current_path)
                else:
                    parent = os.path.dirname(os.path.dirname(current_path))
                parent = os.path.normpath(parent)
                st.session_state.paths[key] = parent
                st.experimental_rerun()
        if not os.path.exists(directory_to_list):
            st.warning(f"Path '{directory_to_list}' does not exist. Choose an option below:")
            create_col, up_col = st.columns(2)
            with create_col:
                if st.button("Create this path", key=f"{radio_button_prefix}_{key}_create_default"):
                    new_name = st.text_input(
                        "Optionally enter a different name for the new path:",
                        value=directory_to_list,
                        key=f"{radio_button_prefix}_{key}_new_default_path_name"
                    )
                    if new_name:
                        try:
                            os.makedirs(new_name, exist_ok=True)
                            st.session_state.paths[key] = new_name
                            st.experimental_rerun()
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
                        st.experimental_rerun()
            return current_path
        try:
            entries = os.listdir(directory_to_list)
        except Exception as e:
            st.error(f"Error reading directory: {e}")
            return current_path
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
