"""Flask web application components."""

import json
import time
import uuid
from datetime import date
from pathlib import Path

import torch
import soundfile as sf
from chatterbox.tts import ChatterboxTTS
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from lipsync import LipSync
from werkzeug.utils import secure_filename

from .ai import get_ollama_response
from .audio_processing import extract_audio_from_video, extract_first_frame, isolate_voice_from_audio
from .background import remove_background_from_video
from .config import (
    BASE_SYSTEM_PROMPT,
    CACHE_DIR,
    EXTRACTED_AUDIO_PATH,
    FIRST_FRAME_PATH,
    INPUT_VIDEO_DIR,
    ISOLATED_VOICE_PATH,
    MAX_CONTENT_LENGTH,
    OUTPUTS_DIR,
    OUTPUT_WAV_PATH,
    PROCESSED_VIDEO_PATH,
    SYNCED_VIDEO_PATH,
    TEMP_DIR,
    UPLOADS_DIR,
    WEIGHTS_DIR,
)
from .pipeline import PipelineManager
from .state import AppState
from .stt_handler import STTHandler
from .utils import setup_upload_directories

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

def _ensure_stt_handler(state: AppState) -> bool:
    """Initialize STT handler on-demand. Returns True if available."""
    if state.stt_handler is not None:
        return True

    stt_device = "cuda" if torch.cuda.is_available() else "cpu"
    stt_compute_type = "float16" if stt_device == "cuda" else "int8"
    print(f"🎤 Initializing STT handler on {stt_device.upper()}...")
    try:
        state.stt_handler = STTHandler(
            model_size="base",
            device=stt_device,
            compute_type=stt_compute_type,
        )
        state.stt_last_error = None
        print("✅ STT handler initialized")
        return True
    except Exception as exc:
        state.stt_handler = None
        state.stt_last_error = str(exc)
        print(f"⚠️ STT handler initialization failed: {exc}")
        return False


def _save_wav(path: str, wav_tensor: torch.Tensor, sample_rate: int) -> None:
    """Persist a waveform tensor to disk using soundfile."""
    waveform = wav_tensor.detach().cpu()
    if waveform.dim() == 2:  # (C, T) -> (T, C)
        waveform = waveform.transpose(0, 1)
    elif waveform.dim() > 2:
        waveform = waveform.squeeze()
    sf.write(path, waveform.numpy(), sample_rate, subtype="PCM_16")


def create_app(state: AppState) -> Flask:
    app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
    app.config["UPLOAD_FOLDER"] = str(UPLOADS_DIR)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload():
        """Accept video file upload, save to runtime folder, and trigger model initialization."""
        state.uploaded_voice_samples = []
        setup_upload_directories()

        video_file = request.files.get("video")
        if not video_file or not video_file.filename:
            return jsonify({"error": "Missing video file"}), 400

        filename = secure_filename(video_file.filename)
        filepath = INPUT_VIDEO_DIR / filename
        video_file.save(str(filepath))
        state.uploaded_input_video = str(filepath)

        # Extract and isolate voice from video
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
            return jsonify({"error": "Failed to extract audio from video"}), 400

        # Process video (background removal)
        print("🎬 Starting video background removal process...")
        processed_video_path = PROCESSED_VIDEO_PATH
        video_to_use = filepath
        if remove_background_from_video(filepath, processed_video_path):
            print("✅ Background removal successful. Using processed video.")
            video_to_use = processed_video_path
            state.processed_video_path = str(processed_video_path)
        else:
            print("⚠️ Background removal failed. Proceeding with the original video.")
            state.processed_video_path = None

        # Extract first frame for poster
        if not extract_first_frame(video_to_use, FIRST_FRAME_PATH):
            return jsonify({"error": "Failed to extract first frame"}), 400

        # Initialize models
        print("🚀 Initializing models...")

        # Initialize TTS model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading Chatterbox-TTS model on {device.upper()}...")
        try:
            tts = ChatterboxTTS.from_pretrained(device=device)
        except RuntimeError as exc:
            if device == "cuda":
                print(
                    f"⚠️ CUDA TTS load failed ({exc}). Falling back to CPU for Chatterbox-TTS.")
                device = "cpu"
                tts = ChatterboxTTS.from_pretrained(device=device)
            else:
                return jsonify({"error": f"Failed to load TTS model: {exc}"}), 500

        state.tts_model = tts

        # Pre-cache speaker conditionals so per-chunk generate() calls skip the
        # ~150 ms speaker-encoder pass and reuse one stable speaker embedding.
        state.voice_conditionals_ready = False
        if state.uploaded_voice_samples:
            try:
                print("🎤 Pre-caching TTS speaker conditionals...")
                tts.prepare_conditionals(state.uploaded_voice_samples[0])
                state.voice_conditionals_ready = True
                print("✅ TTS conditionals cached")
            except Exception as exc:
                print(f"⚠️ Failed to pre-cache TTS conditionals: {exc}")

        # Initialize LipSync model
        video_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎬 Loading LipSync model on {video_device.upper()}...")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            lip = LipSync(
                model="wav2lip",
                checkpoint_path=str(WEIGHTS_DIR / "wav2lip_gan.pth"),
                nosmooth=True,
                device=video_device,
                cache_dir=str(CACHE_DIR),
                img_size=96,
                save_cache=True,
            )
            state.lip_sync_model = lip
            print("✅ LipSync model loaded")
        except Exception as exc:
            return jsonify({"error": f"Failed to load LipSync model: {exc}"}), 500

        # Initialize STT handler
        _ensure_stt_handler(state)

        print("✅ All models initialized successfully")

        # Set up system prompt
        today_date = date.today().strftime("%B %d, %Y")
        date_prompt = f"For your information, today's date is {today_date}."
        state.user_system_prompt = f"{BASE_SYSTEM_PROMPT} {date_prompt}"

        # Get poster image URL
        poster_url = f"/static/{FIRST_FRAME_PATH.name}"

        return jsonify({
            "success": True,
            "poster_url": poster_url,
            "stt_available": state.stt_handler is not None,
            "stt_error": state.stt_last_error,
            "message": "Video uploaded and models initialized"
        })

    @app.route("/chat", methods=["POST"])
    def chat():
        """Start the streaming pipeline for video generation."""
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing text field"}), 400

        user_text = data["text"]
        print(f"💬 User said: {user_text}")

        # Check if models are initialized
        if state.tts_model is None or state.lip_sync_model is None:
            return jsonify({"error": "Models not initialized. Please upload a video first."}), 400

        if not state.uploaded_input_video:
            return jsonify({"error": "No video uploaded"}), 400

        if not state.uploaded_voice_samples:
            return jsonify({"error": "No voice samples available"}), 400

        # Generate unique pipeline ID
        pipeline_id = str(uuid.uuid4())

        # Determine which video to use
        video_to_use = str(state.processed_video_path) if state.processed_video_path else str(
            state.uploaded_input_video)
        voice_sample_path = str(state.uploaded_voice_samples[0])

        # Create and start pipeline
        pipeline = PipelineManager(
            tts_model=state.tts_model,
            lip_sync_model=state.lip_sync_model,
            video_path=video_to_use,
            voice_sample_path=voice_sample_path,
            system_prompt=state.user_system_prompt,
            conversation_history=state.conversation_history.copy(),
            voice_conditionals_ready=state.voice_conditionals_ready,
        )

        pipeline.start(user_text)

        # Store pipeline
        state.active_pipelines[pipeline_id] = pipeline

        # Update conversation history with user message
        state.conversation_history.append(
            {"role": "user", "content": user_text})

        return jsonify({
            "success": True,
            "pipeline_id": pipeline_id,
            "message": "Pipeline started"
        })

    @app.route("/stream_events", methods=["GET"])
    def stream_events():
        """Server-Sent Events stream that pushes chunk-ready notifications.

        Replaces the 500 ms /stream_status polling loop. Eliminates the average
        ~250 ms gap between when a chunk file is written and when the browser
        learns about it. The server-side loop checks the pipeline every 50 ms
        and flushes a `data:` frame the instant a chunk transitions to READY.
        """
        pipeline_id = request.args.get("pipeline_id")
        if not pipeline_id:
            return jsonify({"error": "Missing pipeline_id parameter"}), 400
        if pipeline_id not in state.active_pipelines:
            return jsonify({"error": "Pipeline not found"}), 404

        pipeline = state.active_pipelines[pipeline_id]

        def event_stream():
            # Drain READY chunks as fast as they arrive; sleep briefly when waiting
            # so we don't spin a CPU. Total tick latency is ~50 ms vs 500 ms polling.
            yield "retry: 2000\n\n"
            poll_interval = 0.05
            idle_yield_at = 0.5
            last_yield = time.monotonic()
            while True:
                result = pipeline.get_next_chunk()
                status = result.get("status")
                if status == "READY":
                    yield f"data: {json.dumps(result)}\n\n"
                    last_yield = time.monotonic()
                    continue
                if status in ("DONE", "ERROR"):
                    if status == "DONE":
                        full_response = pipeline.get_full_response()
                        if full_response:
                            state.conversation_history.append(
                                {"role": "assistant", "content": full_response})
                            if len(state.conversation_history) > 20:
                                state.conversation_history = state.conversation_history[-20:]
                    yield f"data: {json.dumps(result)}\n\n"
                    return
                # WAIT: emit a keepalive ping every ~0.5 s so proxies/browsers don't
                # close the connection, but otherwise stay quiet.
                now = time.monotonic()
                if now - last_yield >= idle_yield_at:
                    yield f"data: {json.dumps({'status': 'WAIT'})}\n\n"
                    last_yield = now
                time.sleep(poll_interval)

        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable proxy buffering (nginx)
            "Connection": "keep-alive",
        }
        return Response(event_stream(), mimetype="text/event-stream", headers=headers)

    @app.route("/stream_status", methods=["GET"])
    def stream_status():
        """Get the status of the streaming pipeline and next available chunk."""
        pipeline_id = request.args.get("pipeline_id")
        if not pipeline_id:
            return jsonify({"error": "Missing pipeline_id parameter"}), 400

        if pipeline_id not in state.active_pipelines:
            return jsonify({"error": "Pipeline not found"}), 404

        pipeline = state.active_pipelines[pipeline_id]
        result = pipeline.get_next_chunk()

        # If pipeline is done, update conversation history and clean up
        if result["status"] == "DONE":
            # Update conversation history with full assistant response
            full_response = pipeline.get_full_response()
            if full_response:
                state.conversation_history.append(
                    {"role": "assistant", "content": full_response})
                if len(state.conversation_history) > 20:
                    state.conversation_history = state.conversation_history[-20:]

            # Clean up pipeline after a delay (frontend might poll once more)
            # Could implement cleanup logic here if needed

        return jsonify(result)

    @app.route("/chat_audio", methods=["POST"])
    def chat_audio():
        """Accept audio file, transcribe it, and start the streaming pipeline."""
        # Ensure STT handler is available (lazy init)
        if not _ensure_stt_handler(state):
            return jsonify({
                "error": "STT is not available on the server. Install faster-whisper (and ensure ffmpeg is installed), then restart and re-upload.",
                "detail": state.stt_last_error,
            }), 500

        # Check if models are initialized
        if state.tts_model is None or state.lip_sync_model is None:
            return jsonify({"error": "Models not initialized. Please upload a video first."}), 400

        if not state.uploaded_input_video:
            return jsonify({"error": "No video uploaded"}), 400

        if not state.uploaded_voice_samples:
            return jsonify({"error": "No voice samples available"}), 400

        # Get audio file from request
        audio_file = request.files.get("audio")
        if not audio_file or not audio_file.filename:
            return jsonify({"error": "Missing audio file"}), 400

        # Save audio to temporary file
        temp_dir = TEMP_DIR
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file extension
        filename = secure_filename(audio_file.filename)
        file_ext = Path(filename).suffix.lower()
        if not file_ext:
            file_ext = ".webm"  # Default to webm for browser recordings
        
        # Create temp file
        temp_audio_path = temp_dir / f"audio_input_{uuid.uuid4()}{file_ext}"
        audio_file.save(str(temp_audio_path))
        
        try:
            # Transcribe audio
            print(f"🎤 Transcribing audio file: {temp_audio_path.name}")
            transcribed_text = state.stt_handler.transcribe(str(temp_audio_path))
            
            if not transcribed_text or not transcribed_text.strip():
                return jsonify({"error": "No speech detected in audio"}), 400
            
            print(f"💬 Transcribed text: {transcribed_text}")
            
            # Clean up temp file
            temp_audio_path.unlink()
            
            # Generate unique pipeline ID
            pipeline_id = str(uuid.uuid4())
            
            # Determine which video to use
            video_to_use = str(state.processed_video_path) if state.processed_video_path else str(
                state.uploaded_input_video)
            voice_sample_path = str(state.uploaded_voice_samples[0])
            
            # Create and start pipeline (reuse existing logic)
            pipeline = PipelineManager(
                tts_model=state.tts_model,
                lip_sync_model=state.lip_sync_model,
                video_path=video_to_use,
                voice_sample_path=voice_sample_path,
                system_prompt=state.user_system_prompt,
                conversation_history=state.conversation_history.copy(),
                voice_conditionals_ready=state.voice_conditionals_ready,
            )
            
            pipeline.start(transcribed_text)
            
            # Store pipeline
            state.active_pipelines[pipeline_id] = pipeline
            
            # Update conversation history with user message
            state.conversation_history.append({"role": "user", "content": transcribed_text})
            
            return jsonify({
                "success": True,
                "status": "processing",
                "transcription": transcribed_text,
                "pipeline_id": pipeline_id,
                "message": "Audio transcribed and pipeline started"
            })
            
        except Exception as exc:
            # Clean up temp file on error
            if temp_audio_path.exists():
                temp_audio_path.unlink()
            print(f"❌ Error processing audio: {exc}")
            return jsonify({"error": f"Failed to process audio: {str(exc)}"}), 500

    @app.route("/static/<filename>")
    def serve_static(filename):
        """Serve static files from the outputs directory."""
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = OUTPUTS_DIR / filename
        if not file_path.exists():
            return jsonify({"error": "File not found"}), 404
        return send_from_directory(str(OUTPUTS_DIR), filename)

    return app


def run_flask_app(app: Flask) -> None:
    print("Starting web interface on http://127.0.0.1:5000")
    app.run(debug=False, use_reloader=False)
