"""Face tracking utilities using VISCA and MediaPipe."""

import threading
import time
from typing import Iterable

import cv2
import mediapipe as mp
import serial

from .config import PAN_SPEED, TILT_SPEED
from .state import TrackingState


def initialize_serial(state: TrackingState, possible_ports: Iterable[str], baud: int) -> None:
    """Try to connect to the first available serial port."""
    for port in possible_ports:
        try:
            state.ser = serial.Serial(port, baud, timeout=1)
            print(f"✔️ Successfully connected to serial port: {port}")
            return
        except Exception as exc:
            print(f"❌ Failed to connect to {port}: {exc}")
    print("❌ Could not establish serial connection. Face tracking will be disabled.")


def initialize_camera(state: TrackingState, camera_index: int = 0) -> bool:
    """Initialize camera capture and start frame acquisition thread."""
    try:
        state.camera_cap = cv2.VideoCapture(camera_index)
        if state.camera_cap.isOpened():
            print("✔️ Camera initialized for face tracking")

            def camera_thread():
                while True:
                    ret, frame = state.camera_cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue
                    with state.camera_lock:
                        state.latest_frame = frame
                    time.sleep(0.033)

            threading.Thread(target=camera_thread, daemon=True).start()
            return True
        print("❌ Could not initialize camera for face tracking")
        state.camera_cap = None
    except Exception as exc:
        print(f"❌ Camera initialization failed: {exc}")
        state.camera_cap = None
    return False


def send_visca(state: TrackingState, cmd_bytes):
    try:
        if state.ser and state.ser.is_open:
            state.ser.write(bytes(cmd_bytes + [0xFF]))
            time.sleep(0.01)
    except (serial.SerialException, OSError) as exc:
        print(f"Serial communication error: {exc}")


def pan_tilt_command(state: TrackingState, x_dir: int, y_dir: int):
    send_visca(state, [0x81, 0x01, 0x06, 0x01, PAN_SPEED, TILT_SPEED, x_dir, y_dir])


def stop_motion(state: TrackingState):
    pan_tilt_command(state, 0x03, 0x03)


def start_face_tracking(state: TrackingState):
    """Start background thread that keeps the face centered."""

    def tracking_loop():
        face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        frame_width = 640
        frame_height = 480
        center_x = frame_width // 2
        center_y = frame_height // 2
        tolerance = 40

        while state.tracking_active:
            if state.tracking_paused:
                time.sleep(0.1)
                continue

            with state.camera_lock:
                frame = None if state.latest_frame is None else state.latest_frame.copy()

            if frame is None:
                time.sleep(0.05)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face.process(rgb)

            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * frame_width)
                y = int(bbox.ymin * frame_height)
                w = int(bbox.width * frame_width)
                h = int(bbox.height * frame_height)

                face_center_x = x + w // 2
                face_center_y = y + h // 2
                dx = face_center_x - center_x
                dy = face_center_y - center_y

                x_dir = 0x03
                y_dir = 0x03

                if dx < -tolerance:
                    x_dir = 0x02
                elif dx > tolerance:
                    x_dir = 0x01

                if dy < -tolerance:
                    y_dir = 0x01
                elif dy > tolerance:
                    y_dir = 0x02

                if state.tracking_active and state.ser and state.ser.is_open:
                    if x_dir == 0x03 and y_dir == 0x03:
                        stop_motion(state)
                    else:
                        pan_tilt_command(state, x_dir, y_dir)
            else:
                if state.tracking_active and state.ser and state.ser.is_open:
                    stop_motion(state)

            time.sleep(0.05)

        print("Face tracking thread stopped cleanly")

    threading.Thread(target=tracking_loop, daemon=True).start()


def cleanup_tracking(state: TrackingState):
    """Stop threads and release resources."""
    state.tracking_active = False
    time.sleep(0.5)

    if state.ser and state.ser.is_open:
        try:
            stop_motion(state)
            time.sleep(0.1)
        except Exception as exc:
            print(f"Error stopping motion during cleanup: {exc}")
        finally:
            state.ser.close()
            print("Serial connection closed")

    if state.camera_cap:
        state.camera_cap.release()
        print("Camera released")
