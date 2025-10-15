"""Main application orchestration for PixelHolo."""

import os
from pathlib import Path
import shutil
import threading
import time

import cv2
import torch
import soundfile as sf
from chatterbox.tts import ChatterboxTTS
from lipsync import LipSync

from . import config
from .ai import get_ollama_response
from .audio_processing import extract_first_frame
from .background import remove_background_from_video
from .playback import play_video_and_revert
from .speech import listen_for_speech
from .state import AppState, TrackingState
from .timing import time_operation, print_timing_summary, save_timing_report, configure_timing
from .tracking import (
    cleanup_tracking,
    initialize_camera,
    initialize_serial,
    start_face_tracking,
)
from .ui import overlay_icon
from .utils import setup_upload_directories
from .web import create_app, run_flask_app


def _save_wav(path: str, wav_tensor: torch.Tensor, sample_rate: int) -> None:
    """Persist a waveform tensor to disk using soundfile."""
    waveform = wav_tensor.detach().cpu()
    if waveform.dim() == 2:  # (C, T) -> (T, C)
        waveform = waveform.transpose(0, 1)
    elif waveform.dim() > 2:
        waveform = waveform.squeeze()
    sf.write(path, waveform.numpy(), sample_rate, subtype="PCM_16")


def _wait_for_setup(state: AppState) -> None:
    print("Please open your web browser to http://127.0.0.1:5000 to upload files.")
    while not state.setup_complete:
        time.sleep(1)


def _prepare_video_assets(state: AppState) -> Path:
    print("\n✅ Files uploaded! Initializing models...")

    processed_video_path = config.PROCESSED_VIDEO_PATH
    if state.uploaded_input_video is None:
        raise RuntimeError("Input video missing after upload phase.")

    video_to_use = Path(state.uploaded_input_video)

    print("🎬 Starting video background removal process...")
    with time_operation("Background Removal", verbose=True, track_memory=True):
        if remove_background_from_video(video_to_use, processed_video_path):
            print("✅ Background removal successful. Using processed video.")
            video_to_use = processed_video_path
        else:
            print("⚠️ Background removal failed. Proceeding with the original video.")

    return video_to_use


def _load_icons():
    mic_icon = cv2.imread(str(config.MIC_ICON_PATH), cv2.IMREAD_UNCHANGED)
    thinking_icon = cv2.imread(
        str(config.THINKING_ICON_PATH), cv2.IMREAD_UNCHANGED)
    internet_icon = cv2.imread(
        str(config.INTERNET_ICON_PATH), cv2.IMREAD_UNCHANGED)
    speech_icon = cv2.imread(
        str(config.SPEECH_ICON_PATH), cv2.IMREAD_UNCHANGED)
    video_icon = cv2.imread(str(config.VIDEO_ICON_PATH), cv2.IMREAD_UNCHANGED)

    icons = {
        "mic": mic_icon,
        "thinking": thinking_icon,
        "internet": internet_icon,
        "speech": speech_icon,
        "video": video_icon,
    }

    for name, icon in icons.items():
        if icon is None:
            raise FileNotFoundError(
                f"ERROR: Could not load {name} icon. Ensure the file exists in 'assets/icons'."
            )

    return icons


def _initialize_models(video_device: str) -> LipSync:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("🎬 Loading LipSync model...")
    with time_operation("LipSync Model Loading", verbose=True, track_memory=True):
        lip = LipSync(
            model="wav2lip",
            checkpoint_path=str(config.WEIGHTS_DIR / "wav2lip_gan.pth"),
            nosmooth=True,
            device=video_device,
            cache_dir=str(config.CACHE_DIR),
            img_size=96,
            save_cache=True,
        )
    print("\n✅ LipSync model loaded")
    print(f"LipSync using device: {video_device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(
            f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    return lip


def main(enable_lipsync: bool = True, enable_timing: bool = False) -> None:
    """Run the voice cloning and interactive session.

    Args:
        enable_lipsync: Enable lip-sync video generation (requires wav2lip model)
        enable_timing: Enable performance timing reports (shows execution time for each operation)
    """
    # Configure timing system
    configure_timing(enabled=enable_timing, verbose=enable_timing)

    print("🚀 Starting the Interactive Voice Clone with Ollama! 🚀")

    setup_upload_directories()
    state = AppState()
    tracking_state = TrackingState()

    app = create_app(state)
    flask_thread = threading.Thread(
        target=run_flask_app, args=(app,), daemon=True)
    flask_thread.start()

    _wait_for_setup(state)
    video_to_use = _prepare_video_assets(state)

    initialize_serial(tracking_state, config.POSSIBLE_PORTS, config.BAUD)
    camera_ready = initialize_camera(tracking_state)
    if tracking_state.ser and camera_ready:
        start_face_tracking(tracking_state)
        print("✔️ Face tracking started")

    video_device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"Using device: {video_device.upper()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading Chatterbox-TTS model on {device.upper()}...")
    try:
        tts = ChatterboxTTS.from_pretrained(device=device)
    except RuntimeError as exc:
        if device == "cuda":
            print(
                f"⚠️ CUDA TTS load failed ({exc}). Falling back to CPU for Chatterbox-TTS.")
            device = "cpu"
            with time_operation("Chatterbox-TTS Model Loading", verbose=True, track_memory=True):
                tts = ChatterboxTTS.from_pretrained(device=device)
        else:
            raise

    lip = _initialize_models(video_device) if enable_lipsync else None

    if not extract_first_frame(video_to_use, config.FIRST_FRAME_PATH):
        raise RuntimeError(
            f"Failed to extract first frame from {video_to_use}.")

    base_img = cv2.imread(str(config.FIRST_FRAME_PATH))
    if base_img is None:
        raise RuntimeError(
            f"Failed to load photo from {config.FIRST_FRAME_PATH}.")

    icons = _load_icons()

    display_enabled = bool(os.environ.get("DISPLAY"))
    if display_enabled:
        try:
            cv2.namedWindow(config.DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(config.DISPLAY_WINDOW_NAME, base_img)
            cv2.waitKey(1)
        except cv2.error as exc:
            display_enabled = False
            print(
                f"⚠️ GUI display unavailable ({exc}). Running in headless mode.")

    if display_enabled:
        print("\n" + "=" * 50)
        print("✅ Setup complete! The clone is ready.")
        print("Press the SPACEBAR in the clone's window to talk.")
        print("Press 'q' in the window to quit the program.")
        print("Make sure Ollama is running with the right model installed!")
        print("=" * 50 + "\n")
    else:
        print("\n" + "=" * 50)
        print("✅ Setup complete! Running in headless mode.")
        print("Type your prompts into this terminal. Enter 'quit' to exit.")
        print("=" * 50 + "\n")

    def _show_icon(icon_key: str) -> None:
        if not display_enabled:
            return
        frame = overlay_icon(base_img, icons[icon_key])
        cv2.imshow(config.DISPLAY_WINDOW_NAME, frame)
        cv2.waitKey(1)

    def _reset_display() -> None:
        if display_enabled:
            cv2.imshow(config.DISPLAY_WINDOW_NAME, base_img)
            cv2.waitKey(1)

    def _process_prompt(user_prompt: str) -> None:
        print(f"💬 You said: {user_prompt}")
        with time_operation("Ollama Response Generation", verbose=True, track_memory=True):
            text_to_speak = get_ollama_response(
                user_prompt,
                state.user_system_prompt,
                state.conversation_history,
                base_img if display_enabled else None,
                icons["internet"] if display_enabled else None,
            )

        if not text_to_speak:
            print("Ollama returned an empty response.")
            _reset_display()
            return

        print(f"🤖 Clone says: {text_to_speak}")

        state.conversation_history.append(
            {"role": "user", "content": user_prompt})
        state.conversation_history.append(
            {"role": "assistant", "content": text_to_speak})
        if len(state.conversation_history) > 20:
            state.conversation_history = state.conversation_history[-20:]

        if not state.uploaded_voice_samples:
            raise RuntimeError("Voice sample missing for TTS generation.")

        try:
            if display_enabled:
                _show_icon("speech")
            else:
                print("Generating speech with TTS...")
            with time_operation("TTS Speech Generation", verbose=True, track_memory=True):
                prompt_audio = Path(state.uploaded_voice_samples[0])
                wav = tts.generate(
                    text_to_speak, audio_prompt_path=str(prompt_audio))
                _save_wav(str(config.OUTPUT_WAV_PATH), wav, tts.sr)
        except Exception as tts_error:
            print(f"TTS error: {tts_error}")
            with time_operation("TTS Speech Generation (Retry)", verbose=True, track_memory=True):
                wav = tts.generate(
                    text_to_speak, audio_prompt_path=str(prompt_audio))
                _save_wav(str(config.OUTPUT_WAV_PATH), wav, tts.sr)

        if display_enabled:
            _show_icon("video")

        if enable_lipsync and lip is not None:
            try:
                print(
                    "🎬 Starting lip-sync generation (includes face detection + inference)...")
                with time_operation("Lip-sync Video Generation (Complete)", verbose=True, track_memory=True):
                    lip.sync(
                        str(video_to_use),
                        str(config.OUTPUT_WAV_PATH),
                        str(config.SYNCED_VIDEO_PATH),
                    )
                print("✅ Lip-sync video generation completed.")
                if display_enabled:
                    print("Playing lip-synced video...")
                    with time_operation("Video Playback", verbose=True, track_memory=False):
                        play_video_and_revert(
                            config.SYNCED_VIDEO_PATH, config.FIRST_FRAME_PATH)
                else:
                    print(
                        f"Lip-synced video saved to: {config.SYNCED_VIDEO_PATH}")
            except Exception as sync_error:
                print(f"❌ Lip-sync generation failed: {sync_error}")
                if display_enabled:
                    with time_operation("Video Playback (Fallback)", verbose=True, track_memory=False):
                        print(
                            "Falling back to original video with generated audio...")
                        play_video_and_revert(
                            video_to_use, config.FIRST_FRAME_PATH)
        else:
            if display_enabled:
                print("Playing original video with generated audio...")
                with time_operation("Video Playback", verbose=True, track_memory=False):
                    play_video_and_revert(
                        video_to_use, config.FIRST_FRAME_PATH)

        if not display_enabled:
            print(f"Generated speech saved to: {config.OUTPUT_WAV_PATH}")

        _reset_display()

    try:
        while True:
            if display_enabled:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    print("Shutting down...")
                    tracking_state.tracking_active = False
                    time.sleep(0.2)
                    break
                if key == ord(" "):
                    _show_icon("mic")

                    try:
                        user_prompt = listen_for_speech() or ""
                    except Exception as exc:
                        print(f"Microphone error: {exc}")
                        user_prompt = ""

                    if not user_prompt:
                        print("Could not understand you. Please try again.")
                        _reset_display()
                        continue

                    _process_prompt(user_prompt)
            else:
                try:
                    user_prompt = input(
                        "\nType your prompt (or 'quit' to exit): ").strip()
                except EOFError:
                    user_prompt = "quit"

                if user_prompt.lower() in {"quit", "q"}:
                    print("Shutting down...")
                    tracking_state.tracking_active = False
                    time.sleep(0.2)
                    break

                if not user_prompt:
                    print("No prompt captured. Please try again.")
                    continue

                _process_prompt(user_prompt)
    finally:
        print("Cleaning up...")
        cleanup_tracking(tracking_state)

        if display_enabled:
            cv2.destroyAllWindows()

        print("Cleaning up temporary files...")
        for path in (config.UPLOADS_DIR, config.TEMP_DIR):
            if path.exists():
                shutil.rmtree(path)

    # Print timing summary before exit (only if timing enabled)
    if enable_timing:
        print_timing_summary()
        save_timing_report("timing_report.txt")

    print("Program finished cleanly.")
