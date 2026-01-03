#!/bin/bash
set -e  # Exit on any error

echo "=== Installing system packages for Raspberry Pi traffic monitoring ==="

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Core system packages
echo "🔧 Installing core system packages..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-setuptools \
    git \
    cmake \
    build-essential

# Raspberry Pi specific packages
echo "🍓 Installing Raspberry Pi specific packages..."
sudo apt install -y \
    python3-picamera2 \
    python3-libcamera \
    python3-kms++ \
    python3-prctl \
    python3-pil \
    python3-piexif \
    python3-av \
    libcamera-apps \
    python3-gpiozero \
    python3-rpi.gpio

# OpenCV and computer vision
echo "👁️ Installing computer vision packages..."
sudo apt install -y \
    python3-opencv \
    python3-numpy \
    python3-scipy \
    python3-matplotlib

# Additional system packages that were in original requirements
echo "🌐 Installing additional packages..."
sudo apt install -y \
    python3-flask \
    python3-requests \
    python3-serial \
    python3-psutil

# GUI packages (if needed)
echo "🖥️ Installing GUI packages..."
sudo apt install -y \
    python3-pyqt5 \
    python3-tk

# Test critical packages
echo "✅ Testing critical package installation..."
python3 -c "
try:
    import cv2, numpy, picamera2, RPi.GPIO
    print('✅ All critical packages imported successfully')
    print(f'OpenCV: {cv2.__version__}')
    print(f'NumPy: {numpy.__version__}')
except ImportError as e:
    print(f'❌ Package import failed: {e}')
    print('Please check the installation and try again.')
    exit(1)
"

echo ""
echo "✅ System packages installation completed!"
echo "📋 Next steps:"
echo "1. Create virtual environment: python3 -m venv .venv --system-site-packages"
echo "2. Activate it: source .venv/bin/activate"
echo "3. Install pip packages: pip install -r requirements.txt"
echo "4. Fix simplejpeg: sudo apt remove python3-simplejpeg -y && pip install simplejpeg"