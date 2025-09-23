"""Video playback helpers."""

import sys

import cv2
import pygame

from pathlib import Path

from .config import DISPLAY_WINDOW_NAME, OUTPUT_WAV_PATH


def play_video_and_revert(video_path: Path | str, image_path: Path | str) -> None:
    """Play a video with synchronized audio using pygame, then revert to an image."""
    video_path = str(video_path)
    image_path = str(image_path)
    print(f"Playing video with robust, clock-synced audio: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Warning: Could not get video FPS. Defaulting to 30.")
        fps = 30

    pygame.init()
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(str(OUTPUT_WAV_PATH))
        pygame.mixer.music.play()
    except Exception as exc:
        print(f"Error initializing Pygame or loading audio: {exc}. Video will play without sound.")
        wait_ms = int(1000 / fps)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(DISPLAY_WINDOW_NAME, frame)
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break
        cap.release()
        if "pygame" in sys.modules and pygame.get_init():
            pygame.quit()
        base_img = cv2.imread(image_path)
        if base_img is not None:
            cv2.imshow(DISPLAY_WINDOW_NAME, base_img)
            cv2.waitKey(1)
        return

    playback_start_time = pygame.time.get_ticks()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        target_time_ms = pygame.time.get_ticks() - playback_start_time
        current_frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        expected_time_ms = (current_frame_number / fps) * 1000

        delay_ms = expected_time_ms - target_time_ms
        if delay_ms > 2:
            pygame.time.delay(int(delay_ms))
        elif delay_ms < -10:
            continue

        cv2.imshow(DISPLAY_WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if "pygame" in sys.modules and pygame.get_init():
        pygame.mixer.music.stop()
        pygame.quit()

    base_img = cv2.imread(image_path)
    if base_img is not None:
        cv2.imshow(DISPLAY_WINDOW_NAME, base_img)
        cv2.waitKey(1)
