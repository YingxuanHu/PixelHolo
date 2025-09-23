"""Utility helpers for filesystem preparation."""

import os
import shutil

from .config import UPLOAD_FOLDER, VOICE_SAMPLES_DIR, INPUT_VIDEO_DIR, TEMP_DIR


def setup_upload_directories() -> None:
    """Ensure upload directories exist and are clean on startup."""
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)
    os.makedirs(INPUT_VIDEO_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs("icons", exist_ok=True)  # Ensure icons directory exists
