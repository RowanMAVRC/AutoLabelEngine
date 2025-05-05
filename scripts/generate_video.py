
#!/usr/bin/env python3
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from moviepy.editor import ImageSequenceClip

# Unique color map
CLASS_COLORS = {}
def get_color(class_id: int):
    if class_id not in CLASS_COLORS:
        CLASS_COLORS[class_id] = tuple(random.randint(0, 255) for _ in range(3))
    return CLASS_COLORS[class_id]

def make_frame_by_frame_video(images_dir, labels_dir, out_path, fps):
    """Side-by-side original vs. labeled frames, with centered text below."""
    imgs = sorted(
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg','jpeg','.png'))
    )
    if not imgs:
        print(f"⚠️ No images found in {images_dir}")
        return

    frames = []
    for idx, img_path in enumerate(tqdm(imgs, desc="Frame‑by‑frame", unit="frame")):
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        labeled = img.copy()
        draw = ImageDraw.Draw(labeled)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(12, H//40))
        except IOError:
            font = ImageFont.load_default()

        txt_fname = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        txt_path = os.path.join(labels_dir, txt_fname)
        if os.path.exists(txt_path):
            for line in open(txt_path).read().splitlines():
                parts = line.split()
                if len(parts) < 5: continue
                cid, xc, yc, wn, hn = parts[:5]
                c = int(cid)
                xc, yc, wn, hn = map(float, (xc, yc, wn, hn))
                bw, bh = wn * W, hn * H
                x1 = xc*W - bw/2; y1 = yc*H - bh/2
                x2, y2 = x1 + bw, y1 + bh

                color = get_color(c)
                thickness = max(1, int(H/200))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
                txt = str(c)
                tb = draw.textbbox((0,0), txt, font=font)
                tw, th = tb[2]-tb[0], tb[3]-tb[1]
                draw.rectangle([x1, y1, x1+tw+4, y1+th+4], fill=color)
                draw.text((x1+2, y1+2), txt, fill=(255,255,255), font=font)

        fig = plt.figure(figsize=(12,8))
        gs = fig.add_gridspec(2,2, height_ratios=[8,1])
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,:])

        ax1.imshow(img); ax1.axis('off'); ax1.set_title("Original", pad=10)
        ax2.imshow(labeled); ax2.axis('off'); ax2.set_title("Labeled", pad=10)

        ax3.axis('off')
        ax3.text(0.5, 0.6, img_path, ha='center', va='center', wrap=True)
        ax3.text(0.5, 0.3, f"Frame {idx}", ha='center', va='center')

        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,
                            wspace=0.05, hspace=0.05)

        canvas = FigureCanvas(fig)
        canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frame = buf[..., :3]
        frames.append(frame)
        plt.close(fig)

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(out_path, codec="libx264", audio=False, logger="bar", threads=8)
    print(f"→ frame_by_frame video saved to {out_path}")

def make_object_by_object_video(images_dir, labels_dir, out_path, fps):
    """
    Zoom each object to fit within a 1920×1080 letterbox (min padding),
    with a larger bottom panel for text and increased font size.
    """
    TARGET_W, TARGET_H = 1920, 1080
    TEXT_H = 300  # more padding for text

    imgs = sorted(
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg','jpeg','.png'))
    )
    if not imgs:
        print(f"⚠️ No images found in {images_dir}")
        return

    frames = []
    for idx, img_path in enumerate(tqdm(imgs, desc="Object‑by‑object", unit="frame")):
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        txt_fname = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        txt_path = os.path.join(labels_dir, txt_fname)
        lines = open(txt_path).read().splitlines() if os.path.exists(txt_path) else []
        if not lines:
            continue

        for obj_i, line in enumerate(tqdm(lines, desc=f"Frame {idx} objects", leave=False)):
            parts = line.split()
            if len(parts) < 5:
                continue
            _, xc, yc, wn, hn = map(float, parts[:5])
            bw, bh = wn*W, hn*H
            x1 = xc*W - bw/2; y1 = yc*H - bh/2
            x2, y2 = x1 + bw, y1 + bh

            crop = img.crop((x1,y1,x2,y2))
            scale = min(TARGET_W / crop.width, TARGET_H / crop.height)
            new_w, new_h = int(crop.width * scale), int(crop.height * scale)
            resized = crop.resize((new_w, new_h), Image.LANCZOS)

            letterbox = Image.new("RGB", (TARGET_W, TARGET_H), (0, 0, 0))
            x_off = (TARGET_W - new_w) // 2
            y_off = (TARGET_H - new_h) // 2
            letterbox.paste(resized, (x_off, y_off))

            fig = plt.figure(figsize=(TARGET_W/100, (TARGET_H+TEXT_H)/100), dpi=100)
            gs = fig.add_gridspec(2, 1, height_ratios=[TARGET_H, TEXT_H])
            ax_img = fig.add_subplot(gs[0])
            ax_txt = fig.add_subplot(gs[1])

            ax_img.imshow(letterbox)
            ax_img.axis('off')

            ax_txt.axis('off')
            # larger font and extra padding
            ax_txt.text(0.5, 0.8, img_path, ha='center', va='center', wrap=True, fontsize=24)
            ax_txt.text(0.5, 0.55, f"Frame {idx}", ha='center', va='center', fontsize=24)
            ax_txt.text(0.5, 0.3, f"Object {obj_i+1}", ha='center', va='center', fontsize=24)

            fig.subplots_adjust(left=0, right=1, top=1, bottom=0.1)

            canvas = FigureCanvas(fig)
            canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            frame = buf[..., :3]
            frames.append(frame)
            plt.close(fig)

    if not frames:
        print("⚠️ No objects found – skipping object_by_object video.")
        return

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(out_path, codec="libx264", audio=False, logger="bar", threads=8)
    print(f"→ object_by_object video saved to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate review videos from a YOLO dataset")
    parser.add_argument("--input_path", required=True,
                        help="Folder containing `images/`+`labels/` or parent dir of many subfolders")
    parser.add_argument("--fps", type=float, default=5.0, help="Frames per second")
    parser.add_argument("--mode", choices=["Frame by Frame","Object by Object","Both"],
                        default="Both", help="Which videos to generate")
    args = parser.parse_args()

    root = args.input_path
    pairs = []

    # ─── RECURSIVELY find any dir under root with both `images/` and `labels/`
    for dirpath, dirnames, filenames in os.walk(root):
        if 'images' in dirnames and 'labels' in dirnames:
            imgs = os.path.join(dirpath, 'images')
            lbls = os.path.join(dirpath, 'labels')
            # use only the last component for video filenames
            name = os.path.basename(dirpath)
            pairs.append((name, imgs, lbls))

    if not pairs:
        print(f"⚠️ No image/label folder pairs found in {root}")
        exit(1)

    # ─── generate videos
    for name, imgs, lbls in tqdm(pairs, desc=f"Generating {args.mode}", unit="set"):
        out_dir = os.path.join(os.path.dirname(imgs), 'videos_with_labels')
        os.makedirs(out_dir, exist_ok=True)

        if args.mode in ("Frame by Frame","Both"):
            ff = os.path.join(out_dir, f"{name}_frame_by_frame.mp4")
            make_frame_by_frame_video(imgs, lbls, ff, args.fps)

        if args.mode in ("Object by Object","Both"):
            obo = os.path.join(out_dir, f"{name}_object_by_object.mp4")
            make_object_by_object_video(imgs, lbls, obo, args.fps)