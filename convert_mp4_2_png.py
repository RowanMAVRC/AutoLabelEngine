import os
import cv2
import argparse
import shutil
from tqdm import tqdm

def extract_frames(video_path, output_folder):
    """
    Extracts frames from an MP4/MOV video and saves them as PNG images.
    
    Args:
        video_path (str): Path to the input video.
        output_folder (str): Path to the folder where images will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=frame_count, desc="Extracting frames", unit="frame")

    frame_index = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_index:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_index += 1
        progress_bar.update(1)

    progress_bar.close()
    video_capture.release()
    print(f"Saved images to: {output_folder}\n")

def move_video(video_file, base_copy_destination, base_video_path):
    """
    Moves the original video file to a new location while maintaining its relative
    directory structure.
    
    Args:
        video_file (str): Full path to the video file.
        base_copy_destination (str): Global base folder for the moved videos.
        base_video_path (str): The original base directory of the video files.
    """
    base_dir = os.path.abspath(base_video_path)
    rel_dir = os.path.relpath(os.path.dirname(video_file), base_dir)
    dest_dir = os.path.join(base_copy_destination, rel_dir)
    os.makedirs(dest_dir, exist_ok=True)
    destination_file = os.path.join(dest_dir, os.path.basename(video_file))
    try:
        shutil.move(video_file, destination_file)
        print(f"Moved video: {video_file} -> {destination_file}")
    except Exception as e:
        print(f"Error moving video {video_file}: {e}")

def process_single_video(video_file, base_video_path, output_base, copy_destination):
    """
    Processes a single video file from a directory.
    
    In directory mode, the output folder is built by taking the video file's 
    relative path (without extension) and appending an "images" subfolder to the global output base.
    After conversion, the video file is moved.
    
    Args:
        video_file (str): Full path to the video file.
        base_video_path (str): The base directory provided as input.
        output_base (str): Global base folder for saving image outputs.
        copy_destination (str): Global base folder where the original video is to be moved.
    """
    # This block is only used in directory mode.
    base_dir = os.path.abspath(base_video_path)
    rel_file = os.path.relpath(video_file, base_dir)
    rel_file_no_ext = os.path.splitext(rel_file)[0]
    output_folder = os.path.join(output_base, rel_file_no_ext, "images")
    
    print(f"Starting conversion for: {os.path.relpath(video_file, base_dir)}")
    extract_frames(video_file, output_folder)
    print(f"Completed conversion for: {os.path.relpath(video_file, base_dir)}\n")
    
    if copy_destination:
        move_video(video_file, copy_destination, base_video_path)

def process_videos_in_directory(video_path, output_base, copy_destination):
    """
    Recursively finds video files in a directory and processes them sequentially.
    
    Args:
        video_path (str): The base directory to scan for video files.
        output_base (str): Global base folder for saving image outputs.
        copy_destination (str): Global base folder where the original videos will be moved.
    """
    video_files = []
    for root, dirs, files in os.walk(video_path):
        for f in files:
            if f.lower().endswith(('.mp4', '.mov')):
                video_files.append(os.path.join(root, f))
    
    if not video_files:
        print("No video files found in the directory.")
        return

    for video_file in sorted(video_files):
        process_single_video(video_file, video_path, output_base, copy_destination)

def main():
    parser = argparse.ArgumentParser(
        description="Convert video(s) to frames and move the original video after conversion. " +
                    "For directories, the provided global output folder is used as the head and the file " +
                    "structure is preserved."
    )
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to a video file or a directory of video files.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Global base folder for saving outputs. " +
                             "For a single file, this folder is used directly.")
    parser.add_argument("--copy_destination", type=str, required=True,
                        help="Global base folder to move original videos after conversion.")
    args = parser.parse_args()

    video_path = args.video_path

    if os.path.isdir(video_path):
        process_videos_in_directory(video_path, args.output_folder, args.copy_destination)
    else:
        # Single file mode: use the global output folder as provided.
        print(f"Starting conversion for: {video_path}")
        extract_frames(video_path, args.output_folder)
        print(f"Completed conversion for: {video_path}")
        move_video(video_path, args.copy_destination, os.path.dirname(video_path))

if __name__ == "__main__":
    main()
