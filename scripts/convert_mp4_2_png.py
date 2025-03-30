import cv2
import os
import argparse
from tqdm import tqdm

def extract_frames(video_path, output_folder):
    """
    Extracts frames from an MP4 video and saves them as PNG images.

    Args:
        video_path (str): Path to the input MP4 video.
        output_folder (str): Path to the folder where images will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize tqdm progress bar
    progress_bar = tqdm(total=frame_count, desc="Extracting frames", unit="frame")

    frame_index = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Save the frame as a PNG file
        frame_filename = os.path.join(output_folder, f"frame_{frame_index:05d}.png")
        cv2.imwrite(frame_filename, frame)

        frame_index += 1
        progress_bar.update(1)

    progress_bar.close()
    video_capture.release()

    progress_bar.close()
    print()  
    print(f"Saved images to: {output_folder}")
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from an MP4 video and save them as PNG images."
    )
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input MP4 video.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder where images will be saved.")
    args = parser.parse_args()

    extract_frames(args.video_path, args.output_folder)

if __name__ == "__main__":
    main()
