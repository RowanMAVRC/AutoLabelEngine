# modules/code_editor.py
import os
import yaml
import hashlib
import streamlit as st
from streamlit_ace import st_ace

def yaml_editor(yaml_key, st):
    """
    Displays a YAML editor using st_ace for a YAML file specified by session_state.paths.
    """
    if "paths" not in st.session_state or yaml_key not in st.session_state.paths:
        st.error(f"Path for key '{yaml_key}' not found in session_state.paths")
        return
    file_path = st.session_state.paths[yaml_key]
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return
    if "yamls" not in st.session_state:
        st.session_state.yamls = {}
    st.session_state.yamls[yaml_key] = file_content
    content_hash = hashlib.md5(file_content.encode('utf-8')).hexdigest()
    ace_key = f"edited_content_{yaml_key}_{content_hash}"
    st.subheader("Edit YAML Content")
    lines = file_content.splitlines()
    line_count = len(lines) if len(lines) > 0 else 1
    calculated_height = max(100, line_count * 20 + 25)
    edited_content = st_ace(
        value=file_content,
        language="yaml",
        theme="",
        height=calculated_height,
        font_size=17, 
        key=ace_key,
    )
    if edited_content != st.session_state.yamls[yaml_key]:
        try:
            parsed_yaml = yaml.safe_load(edited_content)
        except yaml.YAMLError as e:
            st.error(f"Invalid YAML format: {e}")
        else:
            try:
                with open(file_path, 'w') as file:
                    yaml.dump(parsed_yaml, file, default_flow_style=False, sort_keys=False)
                st.session_state.yamls[yaml_key] = edited_content
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error saving file: {e}")
    base, ext = os.path.splitext(file_path)
    default_copy_path = base + "_copy" + ext
    new_save_path = st.text_input("Enter new file path", key=f"copy_path_{yaml_key}", value=default_copy_path)
    if st.button("Copy YAML to new file", key=f"copy_button_{yaml_key}"):
        if new_save_path:
            st.session_state.paths[yaml_key] = new_save_path
            try:
                parsed_yaml = yaml.safe_load(edited_content)
            except yaml.YAMLError as e:
                st.error(f"Invalid YAML format, cannot copy: {e}")
            else:
                try:
                    with open(new_save_path, 'w') as new_file:
                        yaml.dump(parsed_yaml, new_file, default_flow_style=False, sort_keys=False)
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error copying file: {e}")
        else:
            st.error("Please enter a valid new file path")

def python_code_editor(code_key, st):
    """
    Displays a Python code editor using st_ace for a Python file specified by session_state.paths.
    """
    if "paths" not in st.session_state or code_key not in st.session_state.paths:
        st.error(f"Path for key '{code_key}' not found in session_state.paths")
        return
    file_path = st.session_state.paths[code_key]
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return
    if "python_codes" not in st.session_state:
        st.session_state.python_codes = {}
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
    if edited_content != st.session_state.python_codes[code_key]:
        try:
            compile(edited_content, file_path, 'exec')
        except SyntaxError as e:
            st.error(f"Invalid Python syntax: {e}")
        else:
            try:
                with open(file_path, 'w') as file:
                    file.write(edited_content)
                st.session_state.python_codes[code_key] = edited_content
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error saving file: {e}")
    base, ext = os.path.splitext(file_path)
    default_copy_path = base + "_copy" + ext
    new_save_path = st.text_input("Enter new file path", key=f"copy_path_{code_key}", value=default_copy_path)
    if st.button("Copy Python code to new file", key=f"copy_button_{code_key}"):
        if new_save_path:
            st.session_state.paths[code_key] = new_save_path
            try:
                compile(edited_content, new_save_path, 'exec')
            except SyntaxError as e:
                st.error(f"Invalid Python syntax, cannot copy: {e}")
            else:
                try:
                    with open(new_save_path, 'w') as new_file:
                        new_file.write(edited_content)
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error copying file: {e}")
        else:
            st.error("Please enter a valid new file path")
