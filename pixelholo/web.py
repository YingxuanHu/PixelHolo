"""Flask web application components."""

import errno
import socket
from datetime import date
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from .audio_processing import extract_audio_from_video, isolate_voice_from_audio
from .config import (
    BASE_SYSTEM_PROMPT,
    EXTRACTED_AUDIO_PATH,
    FLASK_HOST,
    FLASK_PORT,
    FLASK_PORT_FALLBACKS,
    INPUT_VIDEO_DIR,
    ISOLATED_VOICE_PATH,
    MAX_CONTENT_LENGTH,
    UPLOADS_DIR,
)
from .state import AppState
from .utils import setup_upload_directories

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def create_app(state: AppState) -> Flask:
    app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
    app.config["UPLOAD_FOLDER"] = str(UPLOADS_DIR)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/upload_files", methods=["POST"])
    def upload_files_route():
        state.uploaded_voice_samples = []
        setup_upload_directories()

        video_file = request.files.get("input_video")
        if not video_file or not video_file.filename:
            return "Missing video file. Please upload a video.", 400

        filename = secure_filename(video_file.filename)
        filepath = INPUT_VIDEO_DIR / filename
        video_file.save(str(filepath))
        state.uploaded_input_video = filepath

        extracted_audio_path = EXTRACTED_AUDIO_PATH
        if extract_audio_from_video(filepath, extracted_audio_path):
            isolated_voice_path = ISOLATED_VOICE_PATH
            if isolate_voice_from_audio(extracted_audio_path, isolated_voice_path):
                state.uploaded_voice_samples = [str(isolated_voice_path)]
                print("✅ Voice extraction and isolation completed successfully!")
            else:
                state.uploaded_voice_samples = [str(extracted_audio_path)]
                print("⚠️ Voice isolation failed, using raw extracted audio")
        else:
            return "Failed to extract audio from video. Please ensure the video has an audio track.", 400

        today_date = date.today().strftime("%B %d, %Y")
        date_prompt = f"For your information, today's date is {today_date}."
        user_custom_prompt = request.form.get("system_prompt", "").strip()
        if user_custom_prompt:
            state.user_system_prompt = f"{BASE_SYSTEM_PROMPT} {date_prompt} {user_custom_prompt}"
        else:
            state.user_system_prompt = f"{BASE_SYSTEM_PROMPT} {date_prompt}"

        if not state.uploaded_voice_samples or not state.uploaded_input_video:
            return "Missing files or failed to process video. Please try uploading a different video file.", 400

        state.setup_complete = True
        return render_template("upload_success.html")

    return app


def run_flask_app(app: Flask) -> None:
    host = FLASK_HOST
    primary = FLASK_PORT
    candidates = [primary]
    for candidate in FLASK_PORT_FALLBACKS:
        if candidate not in candidates:
            candidates.append(candidate)

    def _can_bind(port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
            return True
        except OSError:
            return False

    for port in candidates:
        if not _can_bind(port):
            print(f"⚠️ Port {port} is in use. Trying next available port...")
            continue
        try:
            print(f"Starting web interface on http://{host}:{port}")
            app.run(host=host, port=port, debug=False, use_reloader=False)
            return
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                print(f"⚠️ Port {port} became busy. Trying next available port...")
                continue
            raise

    print("❌ Unable to start the web interface; all configured ports are busy.")
