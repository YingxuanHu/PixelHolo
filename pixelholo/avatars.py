"""Persist and restore avatar bundles (video, voice sample, optional processed clip, poster)."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from . import config
from .session_settings import settings_payload
from .state import AppState
from .utils import setup_upload_directories

META_FILENAME = "meta.json"
VOICE_FILENAME = "voice.wav"
PROCESSED_FILENAME = "processed_video.mp4"
POSTER_FILENAME = "poster.jpg"
INPUT_PREFIX = "input_video"


def _validate_avatar_id(avatar_id: str) -> bool:
    if not avatar_id or len(avatar_id) > 128:
        return False
    return bool(re.fullmatch(r"[a-f0-9]{32}", avatar_id))


def avatar_dir(avatar_id: str) -> Path:
    if not _validate_avatar_id(avatar_id):
        raise ValueError("Invalid avatar id")
    return config.SAVED_AVATARS_DIR / avatar_id


def list_saved_avatars() -> List[dict[str, Any]]:
    """Return saved avatars newest first. Skips corrupt folders."""
    root = config.SAVED_AVATARS_DIR
    if not root.is_dir():
        return []

    entries: List[tuple[float, dict[str, Any]]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        meta_path = child / META_FILENAME
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        aid = meta.get("id") or child.name
        if not _validate_avatar_id(str(aid)):
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            mtime = 0.0
        entries.append(
            (
                mtime,
                {
                    "id": aid,
                    "name": meta.get("name") or aid[:8],
                    "created": meta.get("created"),
                    "poster_url": f"/avatars/{aid}/poster.jpg",
                },
            )
        )
    entries.sort(key=lambda x: x[0], reverse=True)
    return [e[1] for e in entries]


def save_avatar_from_state(state: AppState, display_name: str) -> dict[str, Any]:
    """Copy current session media into a new saved-avatar folder. Returns metadata dict."""
    name = (display_name or "").strip()
    if not name:
        raise ValueError("Name is required")
    if len(name) > 120:
        raise ValueError("Name is too long")

    if not state.uploaded_input_video:
        raise ValueError("No video loaded")
    if not state.uploaded_voice_samples:
        raise ValueError("No voice sample loaded")

    src_video = Path(state.uploaded_input_video)
    if not src_video.is_file():
        raise ValueError("Video file is missing")

    src_voice = Path(state.uploaded_voice_samples[0])
    if not src_voice.is_file():
        raise ValueError("Voice sample file is missing")

    if not config.FIRST_FRAME_PATH.is_file():
        raise ValueError("Poster frame is missing; finish setup first")

    config.SAVED_AVATARS_DIR.mkdir(parents=True, exist_ok=True)
    avatar_id = uuid.uuid4().hex
    out = config.SAVED_AVATARS_DIR / avatar_id
    out.mkdir(parents=False, exist_ok=False)

    ext = src_video.suffix or ".mp4"
    shutil.copy2(src_video, out / f"{INPUT_PREFIX}{ext}")
    shutil.copy2(src_voice, out / VOICE_FILENAME)
    has_processed = False
    if state.processed_video_path:
        proc = Path(state.processed_video_path)
        if proc.is_file():
            shutil.copy2(proc, out / PROCESSED_FILENAME)
            has_processed = True
    shutil.copy2(config.FIRST_FRAME_PATH, out / POSTER_FILENAME)

    created = datetime.now(timezone.utc).isoformat()
    meta = {
        "id": avatar_id,
        "name": name,
        "created": created,
        "input_ext": ext,
        "has_processed_video": has_processed,
        "session_prefs": settings_payload(state),
    }
    (out / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {"id": avatar_id, "name": name, "created": created, "poster_url": f"/avatars/{avatar_id}/poster.jpg"}


def delete_saved_avatar(avatar_id: str) -> None:
    path = avatar_dir(avatar_id)
    if not path.is_dir():
        raise FileNotFoundError("Avatar not found")
    shutil.rmtree(path)


def apply_saved_avatar_to_state(state: AppState, avatar_id: str) -> dict:
    """Reset runtime upload dirs, copy bundle from disk, and set ``AppState`` paths.

    Returns the parsed ``meta.json`` dict (for restoring session preferences).
    """
    root = avatar_dir(avatar_id)
    meta_path = root / META_FILENAME
    if not meta_path.is_file():
        raise FileNotFoundError("Avatar metadata missing")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FileNotFoundError("Invalid avatar metadata") from exc
    ext = meta.get("input_ext") or ".mp4"
    if not isinstance(ext, str) or not ext.startswith("."):
        ext = ".mp4"
    has_processed = bool(meta.get("has_processed_video"))

    src_input = root / f"{INPUT_PREFIX}{ext}"
    src_voice = root / VOICE_FILENAME
    src_poster = root / POSTER_FILENAME
    if not src_input.is_file() or not src_voice.is_file() or not src_poster.is_file():
        raise FileNotFoundError("Avatar files are incomplete")

    setup_upload_directories()

    dest_video = config.INPUT_VIDEO_DIR / f"avatar_input{ext}"
    shutil.copy2(src_input, dest_video)
    shutil.copy2(src_voice, config.ISOLATED_VOICE_PATH)
    shutil.copy2(src_poster, config.FIRST_FRAME_PATH)

    if has_processed:
        src_proc = root / PROCESSED_FILENAME
        if not src_proc.is_file():
            raise FileNotFoundError("Processed video missing for this avatar")
        shutil.copy2(src_proc, config.PROCESSED_VIDEO_PATH)
        state.processed_video_path = str(config.PROCESSED_VIDEO_PATH)
    else:
        state.processed_video_path = None

    state.uploaded_input_video = str(dest_video)
    state.uploaded_voice_samples = [str(config.ISOLATED_VOICE_PATH)]
    return meta
