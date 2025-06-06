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

# 3. (Optional) start the login portal for multiple users
bash run_login_app.sh                  # launches a login page via ngrok
#     each user is redirected to their own ngrok URL after login

# 4. (Optional) CLI login with CPU pinning (no password)
python get_login.py                    # directly launches a personal GUI

# 5. Launch the full GUI & helpers without login
bash run_autolabel_engine.sh           # <— main entry-point
```

`run_autolabel_engine.sh` does the following:

1. Activates the project venv (or prompts you to generate one).  
2. Exports permissive `umask 000` so every file created by the GUI is world-writable (helpful on multi-user servers).  
3. Starts the Streamlit app (`autolabel_gui.py`) on **localhost:8501** and opens your browser.  
4. Spins up a **tmux** session so all long-running jobs stay alive even if you close the browser tab.

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

---

## File Structure 🗂️

```text
AutoLabelEngine/
│
├─ run_autolabel_engine.sh      # main launcher (see above)
├─ autolabel_gui.py             # Streamlit GUI
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

## License

Released under the MIT License – see `LICENSE` for details.

---

Happy auto-labelling! 🤖