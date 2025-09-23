"""Shared application state objects."""

from dataclasses import dataclass, field
from threading import Lock
from typing import List, Optional, Dict, Any


@dataclass
class AppState:
    """Holds shared state between the web layer and the runtime."""

    uploaded_voice_samples: List[str] = field(default_factory=list)
    uploaded_input_video: Optional[str] = None
    setup_complete: bool = False
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    user_system_prompt: Optional[str] = None


@dataclass
class TrackingState:
    """Stores runtime tracking resources."""

    ser: Optional[Any] = None
    tracking_paused: bool = False
    camera_lock: Lock = field(default_factory=Lock)
    latest_frame: Optional[Any] = None  # OpenCV frame
    camera_cap: Optional[Any] = None
    tracking_active: bool = True
