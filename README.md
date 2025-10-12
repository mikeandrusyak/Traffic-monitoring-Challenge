# Traffic-monitoring-Challenge

## 🚀 Raspberry Pi Setup Guide

### 1️⃣ System preparation
```bash
sudo apt update && sudo apt upgrade -y
```

---

### 2️⃣ Install Python packages and camera support via APT
```bash
sudo apt install -y python3-pip python3-venv git \
    python3-picamera2 python3-prctl python3-pil python3-piexif python3-av \
    libcamera-apps
```

- `python3-picamera2` — official library for working with the v2 camera via libcamera
- `libcamera-apps` — `rpicam-hello`, `rpicam-still`, `rpicam-vid` utilities for testing  
- `python3-venv` — creating virtual environments
- `git` — for cloning the repository

---

### 3️⃣ Clone repository
```bash
cd ~
git clone https://github.com/<your_repo>.git Traffic-monitoring-Challenge
cd Traffic-monitoring-Challenge
```

---

### 4️⃣ Create and activate virtual environment
```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```
> The `--system-site-packages` option allows you to use system packages  
> installed via APT (in particular `picamera2` and `prctl`).

---

### 5️⃣ Upgrade pip and install Python dependencies
Edit `requirements.txt` so that it **does not contain lines** for `picamera2` and `python-prctl`.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 6️⃣ Verify installation
```bash
python - << 'PY'
import cv2, numpy as np
from picamera2 import Picamera2
print("OpenCV:", cv2.__version__)
print("Picamera2 ready:", bool(Picamera2.global_camera_info()))
PY
```

---

### 7️⃣ Test camera hardware
```bash
rpicam-hello -t 3000          
rpicam-hello --list-cameras  
```
