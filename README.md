# Auto Label Engine 🚀  
_“Auto-Label → Review → Learn → Repeat”_

The **Auto Label Engine** is a Streamlit-based GUI that lets you upload or convert datasets, run YOLO-based auto-labelling, review / fix labels, then finetune the detector on the newly vetted data – all in one place.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/RowanMAVRC/AutoLabelEngine.git
cd AutoLabelEngine

# 2. (Optional) create the project virtual-env
bash scripts/setup_venv.sh                     # creates ./envs/auto-label-engine

# 3. Launch the service (always via the login hub)
bash run_login.sh                      # spawns per-user GUI sessions
```

`run_login.sh` does the following:

1. Activates the project venv (or prompts you to generate one).
2. Launches the Streamlit login page (`login.py`) on **localhost:8501** (and optional ngrok tunnel).
3. Prints a shareable portal URL so multiple reviewers can sign in and be routed to their own session.
4. After a user logs in, a new **tmux** session is created running `autolabel_gui.py` on its own port.
5. The new session URL is opened in a browser tab automatically using the server's IP. Set `AUTO_LABEL_BASE_URL` to override the host.
   Each session forwards the username to the GUI so any datasets you upload are prefixed with that name.

Always start the platform with `run_login.sh`. Directly calling `autolabel_gui.py` or `run_autolabel_gui.sh` skips per-user isolation, session routing, and the shareable review URL.

Once the login hub is up, share the printed URL (or the ngrok tunnel) with teammates. Each reviewer signs in with their credentials, receives a dedicated Streamlit workspace, and can collaborate on reviewing, pruning, and retraining datasets without stepping on each other’s sessions.

---

## Typical Workflow ⚙️

| Step | GUI Tab | What happens |
|------|---------|--------------|
| 1 | **Upload Data** | Drop raw images/videos or zip archives – files are unpacked & stored on the server. |
| 2 | **Convert Video → Frames** | One-click MP4/MOV → PNG frame extraction (background tmux). |
| 3 | **Auto Label** | YOLO weights infer on your dataset, writing YOLO-format `labels/*.txt` (GPU selectable). |
| 4 | **Frame by Frame / Object by Object** | Visually inspect each frame/object, add/move/delete boxes. |
| 5 | **Cluster Objects** | Find visually similar false-positives, bulk-delete in one shot. |
| 6 | **Finetune Model** | Train YOLO on the corrected labels directly from the GUI. |
| 7 | **Generate Labeled Video** | Produce side-by-side “before/after” MP4s for quick QA. |

Repeat steps 3-6 until the detector reaches your desired precision.

### Recommended Labeling Loop 🔁

1. Start with a reasonably trained model (even if it is a rough baseline) and run **Auto Label** to seed annotations across the dataset.
2. Use the review tooling – **Frame/Object Review**, **Cluster Objects**, bulk delete, and search – to strip bad predictions and keep only the high-quality labels.
3. Manually add a handful of trusted corrections where coverage is thin or classes are missing.
4. Create new train/val splits from the curated set, finetune via **Finetune Model**, and push the updated weights back into **Auto Label**.
5. Re-label the remaining or previously incorrect samples with the improved model and loop until performance stabilises.

---

## File Structure 🗂️

```text
AutoLabelEngine/
│
├─ run_login.sh                # login page entry-point
├─ run_autolabel_gui.sh        # original single-user launcher
├─ autolabel_gui.py            # Streamlit GUI
├─ cfgs/                        # YAML configs (GUI, YOLO data/model/train)
├─ scripts/                     # helper Python/Bash utilities
│   ├─ convert_mp4_2_png.py
│   ├─ inference.py
│   └─ train_yolo.py
├─ example_data/                # small sample dataset
└─ weights/                     # default YOLO weights
```

---

## Configuration

* **GUI state** is auto-saved to `cfgs/gui/session_state/default.yaml` every time you quit; edit it for persistent defaults.  
* **YOLO training** uses three YAMLs:  
  * `cfgs/yolo/data/*.yaml`   – dataset paths & class names  
  * `cfgs/yolo/model/*.yaml`  – model definition  
  * `cfgs/yolo/train/*.yaml`  – hyper-parameters / epochs / imgsz / batch  
  Edit them via the **Finetune Model** tab (live YAML editor).

---

## Requirements

* Linux / WSL 2  
* Python ≥ 3.8  
* CUDA 11+ & NVIDIA driver (for GPU inference/training)  
* `tmux`, `ffmpeg`, `gpustat` system packages  
* Python libs pinned in `requirements.txt` (installed by `setup_venv.sh`)

---

## Related Projects

Looking for a lightweight, pure annotation client? Check out the companion repository **LabelTools** – it is built on OpenCV for speed and is ideal for high-volume hand labeling. In contrast, **Auto Label Engine** focuses on the auto-label → review → retrain loop, multi-user coordination, and the tooling required to clean up and iterate on model-generated labels.

---

## License

Released under the MIT License – see `LICENSE` for details.

---

Happy auto-labelling! 🤖
