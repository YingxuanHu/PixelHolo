"""Session-level LLM / TTS / display preferences (no re-upload required)."""

from __future__ import annotations

from typing import Any, Dict, List

from .state import AppState

VALID_EMOTIONS = frozenset({"angry", "happy", "sad", "scared", "disgust"})

EMOTION_LLM_HINTS: Dict[str, str] = {
    "angry": "angry, irritated, and sharp-toned",
    "happy": "happy, cheerful, and upbeat",
    "sad": "sad, melancholic, and subdued",
    "scared": "scared, anxious, and nervous",
    "disgust": "disgusted, repulsed, and disdainful",
}


def build_personality_addon(state: AppState) -> str:
    """Combine emotion, intensity, and extra instructions for the LLM system prompt."""
    parts: List[str] = []
    emotion = (state.avatar_emotion or "").strip().lower()
    intensity = max(0.0, min(1.0, float(state.avatar_emotion_intensity)))
    if emotion in VALID_EMOTIONS and intensity > 0:
        pct = int(round(intensity * 100))
        hint = EMOTION_LLM_HINTS[emotion]
        parts.append(
            f"Adopt an emotional tone that is {hint}. Apply this feeling at "
            f"approximately {pct}% intensity in your wording and delivery."
        )
    extra = (state.avatar_personality or "").strip()
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def merge_session_settings(state: AppState, body: Dict[str, Any]) -> List[str]:
    """Apply keys from ``body`` onto ``state``. Returns human-readable warnings."""
    warnings: List[str] = []
    if not isinstance(body, dict):
        return ["Body must be a JSON object"]

    if "llm_temperature" in body:
        state.llm_temperature = max(
            0.1, min(1.5, _to_float(body["llm_temperature"], state.llm_temperature))
        )

    if "avatar_emotion" in body:
        raw = body.get("avatar_emotion", "")
        if isinstance(raw, str):
            emotion = raw.strip().lower()
            state.avatar_emotion = emotion if emotion in VALID_EMOTIONS else ""
        else:
            warnings.append("avatar_emotion must be a string")

    if "avatar_emotion_intensity" in body:
        state.avatar_emotion_intensity = max(
            0.0, min(1.0, _to_float(body["avatar_emotion_intensity"], state.avatar_emotion_intensity))
        )

    if "avatar_personality" in body:
        raw = body.get("avatar_personality", "")
        if isinstance(raw, str):
            state.avatar_personality = raw.strip()[:2000]
        else:
            warnings.append("avatar_personality must be a string")

    for key, lo, hi in [
        ("tts_exaggeration", 0.25, 1.0),
        ("tts_temperature", 0.4, 1.2),
        ("tts_cfg_weight", 0.0, 1.0),
        ("tts_repetition_penalty", 1.0, 1.5),
        ("video_playback_rate", 0.6, 1.5),
        ("color_brightness", 0.5, 1.5),
        ("color_contrast", 0.5, 1.5),
        ("color_saturation", 0.0, 2.0),
    ]:
        if key not in body:
            continue
        cur = getattr(state, key)
        setattr(state, key, max(lo, min(hi, _to_float(body[key], cur))))

    return warnings


def default_settings() -> Dict[str, Any]:
    """Factory defaults for session tuning (matches ``AppState`` field defaults)."""
    return {
        "llm_temperature": 0.7,
        "avatar_emotion": "",
        "avatar_emotion_intensity": 0.5,
        "avatar_personality": "",
        "tts_exaggeration": 0.5,
        "tts_temperature": 0.8,
        "tts_cfg_weight": 0.5,
        "tts_repetition_penalty": 1.2,
        "video_playback_rate": 1.0,
        "color_brightness": 1.0,
        "color_contrast": 1.0,
        "color_saturation": 1.0,
    }


def settings_payload(state: AppState) -> Dict[str, Any]:
    return {
        "llm_temperature": state.llm_temperature,
        "avatar_emotion": state.avatar_emotion,
        "avatar_emotion_intensity": state.avatar_emotion_intensity,
        "avatar_personality": state.avatar_personality,
        "personality_composed": build_personality_addon(state),
        "tts_exaggeration": state.tts_exaggeration,
        "tts_temperature": state.tts_temperature,
        "tts_cfg_weight": state.tts_cfg_weight,
        "tts_repetition_penalty": state.tts_repetition_penalty,
        "video_playback_rate": state.video_playback_rate,
        "color_brightness": state.color_brightness,
        "color_contrast": state.color_contrast,
        "color_saturation": state.color_saturation,
    }


def apply_prefs_from_avatar_meta(state: AppState, meta: Dict[str, Any]) -> None:
    """Restore optional ``session_prefs`` / ``avatar_prefs`` from saved avatar metadata."""
    prefs = meta.get("session_prefs") or meta.get("avatar_prefs")
    if isinstance(prefs, dict):
        merge_session_settings(state, prefs)
