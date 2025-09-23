"""Flask web application components."""

import os
from datetime import date
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from .audio_processing import extract_audio_from_video, isolate_voice_from_audio
from .config import (
    BASE_SYSTEM_PROMPT,
    EXTRACTED_AUDIO_PATH,
    INPUT_VIDEO_DIR,
    ISOLATED_VOICE_PATH,
    MAX_CONTENT_LENGTH,
    UPLOAD_FOLDER,
    VOICE_SAMPLES_DIR,
)
from .state import AppState
from .utils import setup_upload_directories

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def create_app(state: AppState) -> Flask:
    app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
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
        filepath = os.path.join(INPUT_VIDEO_DIR, filename)
        video_file.save(filepath)
        state.uploaded_input_video = filepath

        extracted_audio_path = os.path.join(VOICE_SAMPLES_DIR, EXTRACTED_AUDIO_PATH)
        if extract_audio_from_video(filepath, extracted_audio_path):
            isolated_voice_path = os.path.join(VOICE_SAMPLES_DIR, ISOLATED_VOICE_PATH)
            if isolate_voice_from_audio(extracted_audio_path, isolated_voice_path):
                state.uploaded_voice_samples = [isolated_voice_path]
                print("✅ Voice extraction and isolation completed successfully!")
            else:
                state.uploaded_voice_samples = [extracted_audio_path]
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
    print("Starting web interface on http://127.0.0.1:5000")
    app.run(debug=False, use_reloader=False)
