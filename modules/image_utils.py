# modules/image_utils.py
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from moviepy.editor import ImageSequenceClip, VideoClip, clips_array
import streamlit as st

def add_labels(frame, image_path):
    """
    Overlays YOLO-format labels onto an image frame.
    """
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    base, _ = os.path.splitext(image_path)
    label_file = base + ".txt"
    label_file = label_file.replace("/images/", "/labels/")
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, x_center, y_center, w, h = parts[:5]
            try:
                x_center, y_center, w, h = map(float, (x_center, y_center, w, h))
            except Exception:
                continue
            img_w, img_h = pil_img.size
            box_w = w * img_w
            box_h = h * img_h
            top_left_x = (x_center * img_w) - (box_w / 2)
            top_left_y = (y_center * img_h) - (box_h / 2)
            bottom_right_x = top_left_x + box_w
            bottom_right_y = top_left_y + box_h
            draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], outline="red", width=2)
            draw.text((top_left_x, top_left_y), str(cls), fill="red")
    return np.array(pil_img)

def overlay_frame_text(frame, index):
    """
    Adds an overlay (e.g., frame number) to the frame.
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 100)
    except IOError:
        font = ImageFont.load_default()
    text = f"Frame Number: {index}"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    width, height = img.size
    x = (width - text_width) / 2
    y = int((0.95 * height) - text_height)
    outline_range = 2
    for dx in range(-outline_range, outline_range + 1):
        for dy in range(-outline_range, outline_range + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill="black")
    draw.text((x, y), text, font=font, fill="white")
    return np.array(img)

def create_video_file(image_paths, fps, scale=1.0, output_path="temp_label_review_video.mp4"):
    """
    Creates a composite video from image frames.
    """
    duration = len(image_paths) / fps
    clip_original = ImageSequenceClip(image_paths, fps=fps)
    def make_labeled_frame(t):
        index = int(t * fps)
        if index >= len(image_paths):
            index = len(image_paths) - 1
        current_path = image_paths[index]
        frame = np.array(Image.open(current_path))
        frame_with_labels = add_labels(frame, current_path)
        return frame_with_labels
    clip_labeled = VideoClip(make_labeled_frame, duration=duration).set_fps(fps)
    final_clip = clips_array([[clip_original, clip_labeled]])
    def add_overlay(get_frame, t):
        frame = get_frame(t)
        index = int(t * fps)
        return overlay_frame_text(frame, index)
    final_clip = final_clip.fl(add_overlay, apply_to=['mask', 'video'])
    final_clip.write_videofile(output_path, codec="libx264", audio=False, verbose=False, logger=None)
    return output_path

@st.cache_data(show_spinner=True)
def generating_mp4(image_paths, fps):
    return create_video_file(image_paths, fps)
