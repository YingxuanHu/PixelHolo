"""Configuration constants for the PixelHolo application."""

import os

UPLOAD_FOLDER = "uploads"
VOICE_SAMPLES_DIR = os.path.join(UPLOAD_FOLDER, "voice_samples")
INPUT_VIDEO_DIR = os.path.join(UPLOAD_FOLDER, "video")
OUTPUT_WAV_PATH = "generated_speech.wav"
TEMP_DIR = "temp"
EXTRACTED_AUDIO_PATH = "extracted_audio.wav"
ISOLATED_VOICE_PATH = "isolated_voice.wav"

MIC_ICON_PATH = "icons/mic_on.png"
THINKING_ICON_PATH = "icons/thinking.png"

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi4:latest"

BASE_SYSTEM_PROMPT = (
    "Keep your responses quite short (aim for less than 7 seconds of verbal speech) but still "
    "friendly. You are a hologram clone of a human, so act human, don't act robotic. Additional "
    "details may or may not be provided for who you are."
)

DISPLAY_WINDOW_NAME = "PixelHolo Clone"

MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1 GB

POSSIBLE_PORTS = ["/dev/ttyUSB0"]
BAUD = 9600
PAN_SPEED = 0x18
TILT_SPEED = 0x18
