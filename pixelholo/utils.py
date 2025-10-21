"""Utility helpers for filesystem preparation."""

import shutil
from pathlib import Path

from . import config


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def setup_upload_directories() -> None:
    """Ensure runtime directories exist and are clean on startup."""
    _reset_directory(config.UPLOADS_DIR)
    config.VOICE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    config.INPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    _reset_directory(config.TEMP_DIR)
    _reset_directory(config.OUTPUTS_DIR)
    _reset_directory(config.CACHE_DIR)
