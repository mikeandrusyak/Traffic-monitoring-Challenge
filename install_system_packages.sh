#!/bin/bash
# Script to install system packages for Raspberry Pi traffic monitoring

echo "Installing system packages for Raspberry Pi..."

sudo apt update && sudo apt upgrade -y

# Core system packages
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-setuptools \
    git

# Raspberry Pi specific packages
sudo apt install -y \
    python3-picamera2 \
    python3-prctl \
    python3-pil \
    python3-piexif \
    python3-av \
    libcamera-apps \
    python3-gpiozero \
    python3-rpi.gpio

# OpenCV and computer vision
sudo apt install -y \
    python3-opencv \
    python3-numpy \
    python3-scipy \
    python3-matplotlib

# Additional system packages that were in original requirements
sudo apt install -y \
    python3-flask \
    python3-requests \
    python3-pil \
    python3-serial \
    python3-psutil

# GUI packages (if needed)
sudo apt install -y \
    python3-pyqt5 \
    python3-tk

echo "System packages installation completed!"
echo "Now create virtual environment with: python3 -m venv .venv --system-site-packages"