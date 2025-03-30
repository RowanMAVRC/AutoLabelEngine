# modules/file_utils.py
import os
import glob
import yaml
import zipfile
import shutil
import re

# List of keys used for session state management (if needed)
SELECTED_KEYS = [
    "auto_label_gpu",
    "frame_index",
    "gpu_list",
    "paths",
    "unverified_image_scale",
    "subset_frames"
]

def load_session_state(session_state, default_yaml_path="cfgs/gui/paths/default.yaml"):
    """
    Loads the configuration keys from the YAML file into session_state.
    The configuration is stored in session_state["paths"].
    """
    if os.path.exists(default_yaml_path):
        try:
            with open(default_yaml_path, "r") as f:
                content = f.read().replace("\x00", "")
                saved_state = yaml.safe_load(content)
            if saved_state:
                session_state["paths"] = saved_state
        except Exception as e:
            raise RuntimeError(f"Error loading session state: {e}")
    else:
        save_session_state(session_state, default_yaml_path)

def save_session_state(session_state, default_yaml_path="cfgs/gui/paths/default.yaml"):
    """
    Saves the current session state's paths back to the YAML file.
    """
    try:
        if "paths" in session_state:
            with open(default_yaml_path, "w") as f:
                yaml.dump(session_state["paths"], f)
    except Exception as e:
        raise RuntimeError(f"Error saving session state: {e}")

def infer_image_pattern(images_dir, extensions=(".jpg", ".png")):
    """
    Infers a naming pattern for images in the given directory.
    Returns a tuple: (pattern, start_index, end_index) or None.
    """
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    if not files:
        return None

    patterns = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        match = re.search(r"^(.*?)(\d+)(\.[^.]+)$", filename)
        if match:
            prefix, number_str, ext = match.groups()
            num_length = len(number_str)
            pattern = f"{prefix}{{:0{num_length}d}}{ext}"
            num = int(number_str)
            patterns.setdefault(pattern, []).append(num)
        else:
            return None

    if not patterns:
        return None

    best_pattern, numbers = max(patterns.items(), key=lambda item: len(item[1]))
    if len(numbers) != len(files):
        return None

    numbers.sort()
    for i in range(len(numbers) - 1):
        if numbers[i] + 1 != numbers[i+1]:
            return None

    return best_pattern, numbers[0], numbers[-1]

def upload_to_dir(save_dir, st):
    """
    Allows a user to upload a file or ZIP archive and saves/extracts it.
    """
    uploader_key = "file_uploader"
    uploaded_file = st.file_uploader(
        "Upload a file or ZIP archive containing a directory",
        type=["txt", "csv", "jpg", "png", "pdf", "py", "yaml", "zip", "mp4"],
        key=uploader_key
    )
    if uploaded_file is not None:
        os.makedirs(save_dir, exist_ok=True)
        if uploaded_file.name.endswith(".zip"):
            temp_zip_path = os.path.join(save_dir, uploaded_file.name)
            with open(temp_zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(save_dir)
            os.remove(temp_zip_path)
            st.success(f"ZIP file extracted to: {save_dir}")
        else:
            file_path = os.path.join(save_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        if uploader_key in st.session_state:
            del st.session_state[uploader_key]
        st.experimental_rerun()

def safe_rename_images(images_dir, st):
    """
    Safely renames image files and corresponding label files.
    Returns (new_pattern, total_images).
    """
    extensions = [".jpg", ".jpeg", ".png"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    image_paths.sort()

    labels_dir = images_dir.replace("images", "labels")
    os.makedirs(labels_dir, exist_ok=True)

    total_steps = len(image_paths) * 2
    current_step = 0

    progress_bar = st.progress(0)
    progress_text = st.empty()

    temp_image_label_pairs = []
    label_temp_mapping = {}

    for i, orig_image_path in enumerate(image_paths):
        orig_dir, orig_image_name = os.path.split(orig_image_path)
        temp_image_name = "__tmp__image_{:04d}.jpg".format(i)
        temp_image_path = os.path.join(orig_dir, temp_image_name)
        os.rename(orig_image_path, temp_image_path)

        orig_label_name = os.path.splitext(orig_image_name)[0] + ".txt"
        orig_label_path = os.path.join(labels_dir, orig_label_name)
        temp_label_name = "__tmp__image_{:04d}.txt".format(i)
        temp_label_path = os.path.join(labels_dir, temp_label_name)

        if orig_label_name in label_temp_mapping:
            shutil.copy2(label_temp_mapping[orig_label_name], temp_label_path)
        elif os.path.exists(orig_label_path):
            os.rename(orig_label_path, temp_label_path)
            label_temp_mapping[orig_label_name] = temp_label_path
        else:
            with open(temp_label_path, "w") as f:
                pass

        temp_image_label_pairs.append((temp_image_path, temp_label_path, orig_image_name))
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        progress_text.text(f"Phase 1: Renaming images {i+1} of {len(image_paths)}")

    new_pattern_image = "image_{:04d}.jpg"
    new_pattern_label = "image_{:04d}.txt"

    for i, (temp_image_path, temp_label_path, orig_image_name) in enumerate(temp_image_label_pairs):
        final_image_name = new_pattern_image.format(i)
        final_image_path = os.path.join(os.path.dirname(temp_image_path), final_image_name)
        os.rename(temp_image_path, final_image_path)

        final_label_name = new_pattern_label.format(i)
        final_label_path = os.path.join(os.path.dirname(temp_label_path), final_label_name)
        os.rename(temp_label_path, final_label_path)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        progress_text.text(f"Phase 2: Renaming files {i+1} of {len(temp_image_label_pairs)}")

    progress_text.text("Renaming complete!")
    return new_pattern_image, len(image_paths)

def load_subset_frames(csv_path):
    """
    Reads frame indices from a CSV file and returns a sorted list.
    """
    frames = []
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    frames.append(int(line))
    return sorted(set(frames))

def save_subset_frames(csv_path, frames):
    """
    Writes sorted unique frame indices to a CSV file.
    """
    frames_sorted = sorted(set(frames))
    with open(csv_path, "w") as f:
        for frame in frames_sorted:
            f.write(f"{frame}\n")
