"""Shared application state objects."""

from dataclasses import dataclass, field
from threading import Lock
from typing import List, Optional, Dict, Any


@dataclass
class AppState:
    """Holds shared state between the web layer and the runtime."""

    uploaded_voice_samples: List[str] = field(default_factory=list)
    uploaded_input_video: Optional[str] = None
    processed_video_path: Optional[str] = None  # Path to processed video (after background removal)
    setup_complete: bool = False
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    user_system_prompt: Optional[str] = None
    tts_model: Optional[Any] = None  # ChatterboxTTS model
    lip_sync_model: Optional[Any] = None  # LipSync model
    stt_handler: Optional[Any] = None  # STTHandler for speech-to-text
    stt_last_error: Optional[str] = None  # Last STT init/transcribe error (for UI/debugging)
    active_pipelines: Dict[str, Any] = field(default_factory=dict)  # Store active pipeline instances
    voice_conditionals_ready: bool = False  # True once tts.prepare_conditionals has been called for the current voice sample


@dataclass
class TrackingState:
    """Stores runtime tracking resources."""

    ser: Optional[Any] = None
    tracking_paused: bool = False
    camera_lock: Lock = field(default_factory=Lock)
    latest_frame: Optional[Any] = None  # OpenCV frame
    camera_cap: Optional[Any] = None
    tracking_active: bool = True
