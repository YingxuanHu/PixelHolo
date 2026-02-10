"""Speech-to-Text handler using faster-whisper."""

import os
from pathlib import Path
from typing import Optional

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


class STTHandler:
    """Handles speech-to-text transcription using faster-whisper."""
    
    def __init__(self, model_size: str = "base", device: str = "cuda", compute_type: str = "float16"):
        """Initialize the STT handler with Whisper model.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: Device to run on ("cuda" or "cpu")
            compute_type: Computation type ("float16", "int8", "int8_float16", etc.)
        """
        if WhisperModel is None:
            raise ImportError(
                "faster-whisper is not installed. Install it with: pip install faster-whisper"
            )
        
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model: Optional[WhisperModel] = None
        self._is_loaded = False
    
    def _ensure_loaded(self) -> None:
        """Load the model if not already loaded."""
        if not self._is_loaded:
            print(f"🎤 Loading Whisper model ({self.model_size}) on {self.device}...")
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type
                )
                self._is_loaded = True
                print("✅ Whisper model loaded successfully")
            except Exception as e:
                # Fallback to CPU if CUDA fails
                if self.device == "cuda":
                    print(f"⚠️ CUDA load failed ({e}). Falling back to CPU.")
                    self.device = "cpu"
                    self.compute_type = "int8"  # Use int8 for CPU
                    self.model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8"
                    )
                    self._is_loaded = True
                    print("✅ Whisper model loaded on CPU")
                else:
                    raise
    
    def transcribe(self, audio_file_path: str | Path, language: Optional[str] = None) -> str:
        """Transcribe audio file to text.
        
        Args:
            audio_file_path: Path to audio file (supports .wav, .webm, .mp3, etc.)
            language: Optional language code (e.g., "en", "es"). Auto-detect if None.
        
        Returns:
            Transcribed text string.
        """
        self._ensure_loaded()
        
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        print(f"🎤 Transcribing audio: {audio_path.name}")
        
        try:
            # Transcribe with faster-whisper
            segments, info = self.model.transcribe(
                str(audio_path),
                language=language,
                beam_size=5,
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Combine all segments into a single text
            transcribed_text = " ".join([segment.text for segment in segments])
            transcribed_text = transcribed_text.strip()
            
            print(f"✅ Transcription complete: {transcribed_text[:100]}...")
            return transcribed_text
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            raise RuntimeError(f"Failed to transcribe audio: {e}")
    
    def is_available(self) -> bool:
        """Check if STT is available (model can be loaded)."""
        try:
            self._ensure_loaded()
            return True
        except Exception:
            return False
