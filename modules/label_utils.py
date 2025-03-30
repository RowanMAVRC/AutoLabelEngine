# modules/label_utils.py
import os
import yaml
import hashlib
import shutil
from pathlib import Path
import streamlit as st
from modules.file_utils import load_subset_frames

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    area_box1 = w1 * h1
    area_box2 = w2 * h2
    union_area = area_box1 + area_box2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def are_bboxes_equal(current_bboxes, current_labels, bboxes_xyxy, labels_xyxy, threshold=0.9):
    if len(current_bboxes) != len(bboxes_xyxy) or len(current_labels) != len(labels_xyxy):
        return False
    unmatched = list(zip(bboxes_xyxy, labels_xyxy))
    for box, label in zip(current_bboxes, current_labels):
        found_match = False
        for i, (box2, label2) in enumerate(unmatched):
            if iou(box, box2) >= threshold and label == label2:
                found_match = True
                del unmatched[i]
                break
        if not found_match:
            return False
    return not bool(unmatched)

def update_labels_from_detection(st):
    """
    Updates label files if detected bounding boxes (from a detection module) differ.
    """
    if "out" not in st.session_state or "skip_label_update" not in st.session_state:
        st.session_state["skip_label_update"] = True
        return None
    elif st.session_state["skip_label_update"]:
        st.session_state["skip_label_update"] = False
        return None
    else:
        current_bboxes = []
        current_labels = []
        for bbox in st.session_state.out.get('bbox', []):
            current_bboxes.append(bbox.get('bboxes'))
            current_labels.append(bbox.get('labels'))
        label_path = st.session_state.label_path
        image_width = st.session_state.image_width
        image_height = st.session_state.image_height
        labels = st.session_state.labels
        bboxes_xyxy = st.session_state.bboxes_xyxy
        if (not are_bboxes_equal(current_bboxes, current_labels, bboxes_xyxy, labels, threshold=0.9)) and not st.session_state["skip_label_update"]:
            st.session_state["skip_label_update"] = False
            with open(label_path, "w") as f:
                for label, bbox in zip(current_labels, current_bboxes):
                    x_min, y_min, width, height = bbox
                    x_center_norm = (x_min + width / 2) / image_width
                    y_center_norm = (y_min + height / 2) / image_height
                    width_norm = width / image_width
                    height_norm = height / image_height
                    f.write(f"{label} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
            st.experimental_rerun()
        else:
            if st.session_state["skip_label_update"]:
                st.session_state["skip_label_update"] = False

def update_unverified_frame(st):
    """
    Updates the current image (frame) and its associated labels.
    """
    if st.session_state.frame_index < 0:
        st.session_state.frame_index = st.session_state.max_images - 1
    if st.session_state.frame_index >= st.session_state.max_images:
        st.session_state.frame_index = 0

    if st.session_state.use_subset:
        st.session_state.subset_frames = load_subset_frames(st.session_state.paths["unverified_subset_csv_path"])
        if not st.session_state.subset_frames:
            st.session_state.frame_index = 0
            st.session_state.max_images = 0
            st.error("Subset CSV is empty. No frames to load.")
            return
        if st.session_state.frame_index >= len(st.session_state.subset_frames):
            st.session_state.frame_index = len(st.session_state.subset_frames) - 1
        if st.session_state.frame_index < 0:
            st.session_state.frame_index = 0
        actual_frame_index = st.session_state.subset_frames[st.session_state.frame_index]
    else:
        actual_frame_index = st.session_state.frame_index

    st.session_state.actual_frame_index = actual_frame_index
    images_dir = st.session_state.images_dir
    if st.session_state.image_pattern:
        image_path = os.path.join(images_dir, st.session_state.image_pattern.format(actual_frame_index))
    elif "image_list" in st.session_state and st.session_state.image_list:
        try:
            image_path = st.session_state.image_list[actual_frame_index]
        except IndexError:
            st.error("Frame index out of range for image list.")
            return
    else:
        st.error("No image naming pattern or image list set.")
        return

    try:
        from PIL import Image
        image = Image.open(image_path)
    except Exception as e:
        st.error(f"Error opening image: {e}")
        return

    image_width, image_height = image.size
    labels_dir = images_dir.replace("images", "labels")
    label_path = (
        image_path
        .replace("images", "labels")
        .replace("jpg", "txt")
        .replace("png", "txt")
    )
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    bboxes_xyxy = []
    labels = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:5])
                except Exception:
                    continue
                x_center_abs = x_center * image_width
                y_center_abs = y_center * image_height
                w_abs = w * image_width
                h_abs = h * image_height
                bbox_xyxy = [x_center_abs - w_abs/2, y_center_abs - h_abs/2, w_abs, h_abs]
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

def copy_labels_from_slide(source_index, st):
    """
    Copies labels from a source image to the current image.
    """
    if st.session_state.image_pattern:
        src_image_path = os.path.join(st.session_state.images_dir, st.session_state.image_pattern.format(source_index))
    elif "image_list" in st.session_state and st.session_state.image_list:
        try:
            src_image_path = st.session_state.image_list[source_index]
        except IndexError:
            st.warning(f"Source index {source_index} is out of range for image list.")
            return
    else:
        st.error("No image naming pattern or image list set.")
        return

    src_label_path = src_image_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    if os.path.exists(src_label_path):
        with open(src_label_path, "r") as f:
            src_labels = f.read()
    else:
        st.warning(f"No labels found in slide {source_index}.")
        src_labels = ""
    if st.session_state.image_pattern:
        curr_image_path = os.path.join(st.session_state.images_dir, st.session_state.image_pattern.format(st.session_state.frame_index))
    elif "image_list" in st.session_state and st.session_state.image_list:
        try:
            curr_image_path = st.session_state.image_list[st.session_state.frame_index]
        except IndexError:
            st.error("Current frame index is out of range for image list.")
            return
    else:
        st.error("No image naming pattern or image list set.")
        return

    curr_label_path = curr_image_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    with open(curr_label_path, "w") as f:
        f.write(src_labels)
    st.session_state["skip_label_update"] = True

def copy_prev_labels(st):
    if st.session_state.frame_index > 0:
        source_index = st.session_state.frame_index - 1
    else:
        source_index = st.session_state.max_images - 1
    copy_labels_from_slide(source_index, st)

def copy_next_labels(st):
    if st.session_state.frame_index < st.session_state.start_index + st.session_state.max_images - 1:
        source_index = st.session_state.frame_index + 1
    else:
        source_index = 0
    copy_labels_from_slide(source_index, st)

def zoom_edit_callback(i, st):
    if st.session_state.get("active_edit_view") == "object":
        return
    try:
        old_bbox = st.session_state.bboxes_xyxy[i]
    except IndexError:
        st.warning(f"Bounding box index {i} is out of range. Skipping update.")
        return
    new_center_x = st.session_state.get(f"bbox_{i}_center_x_input", old_bbox[0] + old_bbox[2]/2)
    flipped_center_y = st.session_state.get(f"bbox_{i}_center_y_input", st.session_state.image_height - (old_bbox[1] + old_bbox[3]/2))
    image_height = st.session_state.image_height
    actual_center_y = image_height - flipped_center_y
    new_w = st.session_state.get(f"bbox_{i}_w_input", old_bbox[2])
    new_h = st.session_state.get(f"bbox_{i}_h_input", old_bbox[3])
    new_x = new_center_x - new_w/2
    new_y = actual_center_y - new_h/2
    if new_x < 0:
        new_x = 0.0
    if new_y < 0:
        new_y = 0.0
    if new_x + new_w > st.session_state.image_width:
        new_x = st.session_state.image_width - new_w
    if new_y + new_h > image_height:
        new_y = image_height - new_h
    if (new_x, new_y, new_w, new_h) != tuple(old_bbox):
        st.session_state.bboxes_xyxy[i] = [new_x, new_y, new_w, new_h]
        label_path = st.session_state.label_path
        image_width = st.session_state.image_width
        with open(label_path, "w") as f:
            for label, bbox in zip(st.session_state.labels, st.session_state.bboxes_xyxy):
                bx, by, bw, bh = bbox
                x_center_norm = (bx + bw/2) / image_width
                y_center_norm = (by + bh/2) / st.session_state.image_height
                width_norm = bw / image_width
                height_norm = bh / image_width
                f.write(f"{label} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        st.session_state["skip_label_update"] = True

def get_object_by_global_index(global_index, st):
    if st.session_state.get("image_pattern") is not None:
        image_list = [
            os.path.join(st.session_state.paths["unverified_images_path"],
                         st.session_state.image_pattern.format(i))
            for i in range(st.session_state.start_index, st.session_state.start_index + st.session_state.max_images)
        ]
    else:
        image_list = st.session_state.image_list if "image_list" in st.session_state else []
    count = 0
    from PIL import Image
    for image_path in image_list:
        try:
            img = Image.open(image_path)
        except Exception:
            continue
        label_file = image_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        bboxes = []
        labels = []
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls = int(parts[0])
                            x_center, y_center, w_norm, h_norm = map(float, parts[1:5])
                        except Exception:
                            continue
                        w_abs = w_norm * img.width
                        h_abs = h_norm * img.height
                        x_abs = (x_center * img.width) - (w_abs / 2)
                        y_abs = (y_center * img.height) - (h_abs / 2)
                        bboxes.append([x_abs, y_abs, w_abs, h_abs])
                        labels.append(cls)
        for local_index, bbox in enumerate(bboxes):
            if count == global_index:
                return {
                    "img": img,
                    "image_path": image_path,
                    "label_path": label_file,
                    "bbox": bbox,
                    "label": labels[local_index] if local_index < len(labels) else None,
                    "local_index": local_index,
                    "global_index": global_index
                }
            count += 1
    return None

def object_by_object_edit_callback(st):
    current_obj = get_object_by_global_index(st.session_state.global_object_index, st)
    if current_obj is None:
        st.warning("No object found to update.")
        return
    img = current_obj["img"]
    old_bbox = current_obj["bbox"]
    global_idx = current_obj["global_index"]
    new_center_x = st.session_state.get(f"object_{global_idx}_center_x", old_bbox[0] + old_bbox[2]/2)
    new_center_y = st.session_state.get(f"object_{global_idx}_center_y", old_bbox[1] + old_bbox[3]/2)
    new_w = st.session_state.get(f"object_{global_idx}_w", old_bbox[2])
    new_h = st.session_state.get(f"object_{global_idx}_h", old_bbox[3])
    new_x = new_center_x - new_w/2
    new_y = new_center_y - new_h/2
    if new_x < 0:
        new_x = 0.0
    if new_y < 0:
        new_y = 0.0
    if new_x + new_w > img.width:
        new_x = img.width - new_w
    if new_y + new_h > img.height:
        new_y = img.height - new_h
    new_bbox = [new_x, new_y, new_w, new_h]
    if new_bbox != old_bbox:
        label_file = current_obj["label_path"]
        try:
            with open(label_file, "r") as f:
                lines = f.readlines()
            local_idx = current_obj["local_index"]
            x_center_norm = (new_x + new_w/2) / img.width
            y_center_norm = (new_y + new_h/2) / img.height
            width_norm = new_w / img.width
            height_norm = new_h / img.height
            new_line = f"{current_obj['label']} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n"
            lines[local_idx] = new_line
            with open(label_file, "w") as f:
                f.writelines(lines)
            st.success("Object updated.")
        except Exception as e:
            st.error(f"Error updating label file: {e}")
    else:
        st.info("No changes detected.")
