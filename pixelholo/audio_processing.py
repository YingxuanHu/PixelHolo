"""Audio and video processing helpers."""

import logging
import os
import subprocess
import wave

import cv2


def isolate_voice_from_audio(input_audio_path: str, output_audio_path: str) -> bool:
    """Extract and isolate voice from audio using audio-separator library."""
    try:
        from audio_separator.separator import Separator

        print("🎵 Isolating voice from audio using AI...")
        separator = Separator(
            log_level=logging.WARNING,
            log_formatter=None,
            model_file_dir="./models",
        )
        separator.separate(
            input_audio_path,
            output_dir=os.path.dirname(output_audio_path),
            stem_name="vocals",
            output_format="wav",
            clean_work_dir=True,
        )

        expected_output = os.path.join(os.path.dirname(output_audio_path), "vocals.wav")
        if os.path.exists(expected_output):
            os.rename(expected_output, output_audio_path)
            print(f"✅ Voice isolated successfully: {output_audio_path}")
            return True
        print("⚠️ Voice isolation did not produce the expected output file.")
        return False
    except Exception as exc:
        print(f"❌ Voice isolation failed: {exc}")
        return False


def extract_audio_from_video(video_path: str, output_audio_path: str) -> bool:
    """Extract audio from video file using ffmpeg."""
    try:
        print("🎬 Extracting audio from video...")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "22050",
            "-ac",
            "1",
            output_audio_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Audio extracted successfully to: {output_audio_path}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"❌ Audio extraction failed: {exc}")
        print("Make sure ffmpeg is installed and accessible from command line")
        return False
    except Exception as exc:
        print(f"❌ Unexpected error during audio extraction: {exc}")
        return False


def extract_first_frame(video_path: str, output_path: str) -> bool:
    """Extract the first frame from a video and save as image."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False

    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(output_path, frame)
        return True
    print("Error: Could not read first frame from video")
    return False


def get_video_duration(video_path: str) -> float:
    """Get the duration of a video in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration


def get_audio_duration(audio_path: str) -> float:
    """Get the duration of an audio file in seconds."""
    try:
        with wave.open(audio_path, "rb") as audio_file:
            frames = audio_file.getnframes()
            sample_rate = audio_file.getframerate()
            return frames / sample_rate
    except Exception:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    audio_path,
                ],
                capture_output=True,
                text=True,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0


def loop_video_to_match_audio(video_path: str, audio_duration: float, output_path: str) -> bool:
    """Loop video to match audio duration."""
    video_duration = get_video_duration(video_path)
    if video_duration == 0:
        return False

    loop_count = int(audio_duration / video_duration) + 1
    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        str(loop_count),
        "-i",
        video_path,
        "-t",
        str(audio_duration),
        "-c",
        "copy",
        output_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        print("Error: ffmpeg not available. Cannot loop video.")
        return False
