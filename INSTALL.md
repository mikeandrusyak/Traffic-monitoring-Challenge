# Installation Guide for Raspberry Pi

## Step 1: Install System Packages
First, install system packages using the provided script:

```bash
chmod +x install_system_packages.sh
./install_system_packages.sh
```

## Step 2: Create Virtual Environment
Create virtual environment that can access system packages:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

## Step 3: Install Pip Packages
Install remaining packages through pip:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 4: Verify Installation
Test that everything works:

```bash
python3 -c "
import cv2
import numpy as np
from picamera2 import Picamera2
import RPi.GPIO
print('All packages imported successfully!')
print('OpenCV version:', cv2.__version__)
print('NumPy version:', np.__version__)
"
```

## Why This Approach?

- **System packages via APT**: Raspberry Pi optimized versions
- **Python packages via pip**: Latest versions that work with system packages
- **--system-site-packages**: Allows virtual environment to access system packages

## Original Problem

The original requirements.txt contained:
- `arandr==0.1.11` - This is a GUI tool, not a Python package
- Many system packages that should be installed via APT
- Exact versions that may not be available on all platforms

This approach separates system dependencies from Python dependencies for better compatibility.