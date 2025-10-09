"""Configuration constants for the PixelHolo application."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Asset directories
ASSETS_DIR = PROJECT_ROOT / "assets"
ICON_DIR = ASSETS_DIR / "icons"
MODELS_DIR = ASSETS_DIR / "models"
WEIGHTS_DIR = ASSETS_DIR / "weights"

# Runtime directories
RUNTIME_DIR = PROJECT_ROOT / "runtime"
UPLOADS_DIR = RUNTIME_DIR / "uploads"
VOICE_SAMPLES_DIR = UPLOADS_DIR / "voice_samples"
INPUT_VIDEO_DIR = UPLOADS_DIR / "video"
OUTPUTS_DIR = RUNTIME_DIR / "outputs"
CACHE_DIR = RUNTIME_DIR / "cache"
TEMP_DIR = RUNTIME_DIR / "temp"

# Runtime files
EXTRACTED_AUDIO_NAME = "extracted_audio.wav"
ISOLATED_VOICE_NAME = "isolated_voice.wav"
OUTPUT_WAV_NAME = "generated_speech.wav"
FIRST_FRAME_NAME = "first_frame.jpg"
SYNCED_VIDEO_NAME = "synced_output.mp4"
PROCESSED_VIDEO_NAME = "processed_video.mp4"

EXTRACTED_AUDIO_PATH = VOICE_SAMPLES_DIR / EXTRACTED_AUDIO_NAME
ISOLATED_VOICE_PATH = VOICE_SAMPLES_DIR / ISOLATED_VOICE_NAME
OUTPUT_WAV_PATH = OUTPUTS_DIR / OUTPUT_WAV_NAME
FIRST_FRAME_PATH = OUTPUTS_DIR / FIRST_FRAME_NAME
SYNCED_VIDEO_PATH = OUTPUTS_DIR / SYNCED_VIDEO_NAME
PROCESSED_VIDEO_PATH = INPUT_VIDEO_DIR / PROCESSED_VIDEO_NAME

# Icon assets
MIC_ICON_PATH = ICON_DIR / "mic_on.png"
THINKING_ICON_PATH = ICON_DIR / "thinking.png"
INTERNET_ICON_PATH = ICON_DIR / "internet.png"
SPEECH_ICON_PATH = ICON_DIR / "speech-synthesis.png"
VIDEO_ICON_PATH = ICON_DIR / "video-generation.png"

# External services
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"

BASE_SYSTEM_PROMPT = (
    "Keep your responses quite short (aim for less than 7 seconds of verbal speech) but still "
    "friendly. You are a hologram clone of a human, so act human, don't act robotic. Additional "
    "details may or may not be provided for who you are."
)

DISPLAY_WINDOW_NAME = "PixelHolo Clone"

MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1 GB

# Web interface configuration (overridable via ENV)
FLASK_HOST = os.environ.get("PIXELHOLO_FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.environ.get("PIXELHOLO_FLASK_PORT", "5000"))
_fallback_env = os.environ.get("PIXELHOLO_FLASK_PORT_FALLBACKS")
if _fallback_env:
    FLASK_PORT_FALLBACKS = tuple(
        int(value.strip())
        for value in _fallback_env.split(",")
        if value.strip().isdigit()
    )
else:
    FLASK_PORT_FALLBACKS = (5001, 5002, 5003)

# Serial configuration
POSSIBLE_PORTS = ["/dev/ttyUSB0"]
BAUD = 9600
PAN_SPEED = 0x18
TILT_SPEED = 0x18

# Background removal tuning
BG_PROCESS_EVERY = 3  # run segmenter every N frames (>=1)
BG_MASK_EMA = 0.7     # temporal smoothing for masks (0..1)
BG_DOWNSCALE_MAX = 768  # compute masks at <= this max dimension
BG_MODEL_NAME = None    # e.g., "isnet-general-use", "u2netp"; None=auto
