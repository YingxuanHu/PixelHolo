"""Streaming pipeline for real-time video generation.

Latency strategy:

1. Adaptive ramp-up chunking. The first text chunk yielded to TTS is intentionally
   short (~5 words, broken at the earliest natural punctuation/conjunction) so the
   first audio+video clip can start playing in roughly one second. Subsequent chunks
   grow (10 -> 18 -> 24 words) so by the time chunk 1 finishes playing, the heavier
   later chunks are already rendered. The growth rate keeps each chunk's render
   time below the previous chunk's playback time, eliminating gaps.

2. Pipeline parallelism between TTS and LipSync. Instead of one worker doing
   TTS -> LipSync -> ffmpeg serially per chunk, we split into two threads connected
   by an intermediate audio queue. While the LipSync thread renders chunk N's video,
   the TTS thread is already producing chunk N+1's audio. On a single GPU this
   overlaps the GPU-bound and CPU/ffmpeg-bound stages without doubling VRAM usage.

3. Boundary hygiene for chunked TTS. Each generated waveform has leading/trailing
   silence trimmed (Chatterbox sometimes pads ~50-150 ms) and is generated with a
   fixed torch seed + cached speaker conditionals so prosody and timbre stay
   consistent across chunks. This makes the chunked output sound like one
   continuous sentence instead of stitched pieces.
"""

import json
import re
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

import librosa
import numpy as np
import requests
import soundfile as sf
import torch

from .ai import needs_internet_ml, remove_emojis, web_search
from .config import (
    BASE_SYSTEM_PROMPT,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    OUTPUTS_DIR,
)


TTS_SEED = 1234

CHUNK_WORD_TARGETS = [5, 10, 18, 24]

CHUNK_WORD_HARD_MAX = [9, 16, 26, 36]

CHUNK_WORD_MIN = [2, 5, 8, 10]

EARLY_BREAK_PUNCT = set(",;:—-")
HARD_BREAK_PUNCT = set(".!?")
CONJUNCTIONS = {
    "and", "but", "so", "because", "then", "or", "yet", "while", "when",
    "though", "although", "since", "however",
}


def _save_wav(path: str, wav_tensor: torch.Tensor, sample_rate: int) -> None:
    """Persist a waveform tensor to disk using soundfile."""
    waveform = wav_tensor.detach().cpu()
    if waveform.dim() == 2:  # (C, T) -> (T, C)
        waveform = waveform.transpose(0, 1)
    elif waveform.dim() > 2:
        waveform = waveform.squeeze()
    sf.write(path, waveform.numpy(), sample_rate, subtype="PCM_16")


def _trim_silence_tensor(wav_tensor: torch.Tensor, sample_rate: int,
                         top_db: float = 35.0,
                         pad_ms: float = 30.0) -> torch.Tensor:
    """Strip leading/trailing silence from a waveform, leaving a small pad.

    Chatterbox sometimes prepends/appends short silence to each clip; with N chunks
    that compounds into audible "stuttering" pauses at every chunk boundary. We trim
    aggressively (top_db=35 dB) and then re-add a tiny pad so the boundaries don't
    sound clipped.
    """
    waveform = wav_tensor.detach().cpu()
    if waveform.dim() == 2:
        # (C, T) -> use first channel for silence detection
        mono = waveform[0].numpy()
    else:
        mono = waveform.numpy()

    if mono.size == 0:
        return wav_tensor

    try:
        _, (start, end) = librosa.effects.trim(mono, top_db=top_db)
    except Exception:
        return wav_tensor

    if end <= start:
        return wav_tensor

    pad_samples = int(sample_rate * pad_ms / 1000.0)
    start = max(0, start - pad_samples)
    end = min(mono.size, end + pad_samples)

    if waveform.dim() == 2:
        trimmed = waveform[:, start:end]
    else:
        trimmed = waveform[start:end]
    return trimmed


def _strip_word_punct(word: str) -> str:
    """Return word lower-cased with punctuation stripped, for conjunction checks."""
    return re.sub(r"[^\w]", "", word).lower()


def _try_emit_chunk(buffer_words: list[str], chunk_idx: int) -> Optional[tuple[str, list[str]]]:
    """If buffer_words is ready to emit a chunk for the given chunk index, return
    (chunk_text, remaining_words). Otherwise return None.

    Sizing for chunk_idx 0..len(targets)-1 uses the ramp-up tables above; later
    chunks reuse the final entry. Three preference passes are tried in order:

      1. Earliest hard break (.!?) at >= min_words. This is the most natural place
         to split, so we take it even if it falls before the soft target.
      2. For early chunks (idx < 2): earliest soft break (, ; : - --) or
         conjunction boundary at >= target_words.
      3. Force-cut at hard_max once the buffer is long enough.
    """
    idx = min(chunk_idx, len(CHUNK_WORD_TARGETS) - 1)
    target = CHUNK_WORD_TARGETS[idx]
    hard_max = CHUNK_WORD_HARD_MAX[idx]
    min_words = CHUNK_WORD_MIN[idx]
    allow_soft_breaks = chunk_idx < 2

    n = len(buffer_words)
    if n == 0:
        return None

    search_end = min(n, hard_max)

    # Pass 1: prefer the earliest hard sentence break once we have min_words. We
    # actually only commit to this if *either* the hard break is past min_words
    # *and* there are enough trailing words to know we aren't truncating mid-stream.
    # We require buffer_words to contain at least one word past the candidate so
    # the LLM has clearly moved on past the punctuation.
    for i in range(min_words - 1, search_end):
        word = buffer_words[i]
        if word and word[-1] in HARD_BREAK_PUNCT and i + 1 < n:
            chunk_text = " ".join(buffer_words[: i + 1])
            return chunk_text, buffer_words[i + 1:]

    # Pass 2: at/after target, accept soft breaks or upcoming conjunctions.
    if allow_soft_breaks:
        for i in range(target - 1, search_end):
            word = buffer_words[i]
            if word and word[-1] in EARLY_BREAK_PUNCT and i + 1 < n:
                chunk_text = " ".join(buffer_words[: i + 1])
                return chunk_text, buffer_words[i + 1:]
            if i + 1 < n:
                next_clean = _strip_word_punct(buffer_words[i + 1])
                if next_clean in CONJUNCTIONS and i + 1 >= target - 1:
                    chunk_text = " ".join(buffer_words[: i + 1])
                    return chunk_text, buffer_words[i + 1:]

    # Pass 3: force-cut at hard_max regardless of punctuation.
    if n >= hard_max:
        chunk_text = " ".join(buffer_words[:hard_max])
        return chunk_text, buffer_words[hard_max:]

    return None


def _get_ollama_stream(
    prompt: str,
    system_prompt: Optional[str],
    conversation_history: Optional[list[dict[str, str]]] = None,
):
    """Stream LLM output and yield adaptive-size word chunks.

    Yields chunks aggressively short at the start so the first audio clip can be
    rendered quickly, then grows toward sentence-sized pieces as the avatar starts
    speaking. See module docstring for the reasoning.
    """
    search_context = ""
    if needs_internet_ml(prompt):
        print("🌐 Query requires internet search. Searching...")
        search_results = web_search(prompt)
        if search_results and "No results found" not in search_results:
            search_context = (
                f"\n\nCurrent search results for '{prompt}':\n{search_results}\n\n"
                "Please use this information to provide an accurate, up-to-date response."
            )
            print("✅ Search completed. Found relevant information.")
        else:
            print("❌ Search completed but no relevant results found.")

    current_system_prompt = system_prompt or BASE_SYSTEM_PROMPT
    full_prompt = current_system_prompt + "\n\n"

    if conversation_history:
        for message in conversation_history[-10:]:
            if message["role"] == "user":
                full_prompt += f"Human: {message['content']}\n"
            else:
                full_prompt += f"Assistant: {message['content']}\n"

    user_message = prompt + search_context
    full_prompt += f"Human: {user_message}\nAssistant: "

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": True,
        "options": {"temperature": 0.7, "num_predict": 196},
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            OLLAMA_API_URL, json=payload, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        word_buffer: list[str] = []
        partial_token = ""
        chunk_idx = 0

        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue

            chunk = data.get("response", "")
            done = bool(data.get("done", False))

            if chunk:
                # Stream tokens may end mid-word (no trailing whitespace). Buffer
                # the trailing partial token until whitespace arrives so we don't
                # split words across chunks.
                combined = partial_token + chunk
                # Split on whitespace, keeping order; the last item is partial if
                # combined doesn't end with whitespace.
                pieces = combined.split()
                if combined and not combined[-1].isspace():
                    partial_token = pieces[-1] if pieces else ""
                    pieces = pieces[:-1]
                else:
                    partial_token = ""
                word_buffer.extend(pieces)

                # Try to emit as many chunks as possible from the buffer
                while True:
                    emit = _try_emit_chunk(word_buffer, chunk_idx)
                    if emit is None:
                        break
                    chunk_text, word_buffer = emit
                    chunk_idx += 1
                    yield chunk_text

            if done:
                # Flush any leftover partial token + buffered words as a final chunk
                if partial_token:
                    word_buffer.append(partial_token)
                    partial_token = ""
                if word_buffer:
                    yield " ".join(word_buffer)
                    word_buffer = []
                return

        # Stream ended without an explicit done=true; flush remainder
        if partial_token:
            word_buffer.append(partial_token)
        if word_buffer:
            yield " ".join(word_buffer)

    except requests.exceptions.RequestException as exc:
        print(f"Error connecting to Ollama API: {exc}")
        yield "I seem to be having trouble thinking right now."
    except Exception as exc:
        print(f"Unexpected error in streaming: {exc}")
        yield "I seem to be having trouble thinking right now."


class PipelineManager:
    """Orchestrates LLM -> TTS -> LipSync as a 3-stage streaming pipeline.

    Stages run in parallel threads:

        LLM thread       -> text_queue  (one entry per adaptive chunk)
        TTS thread       -> audio_queue (one entry per generated WAV)
        LipSync thread   -> video_queue (one entry per rendered MP4)

    The TTS and LipSync stages overlap: while LipSync renders chunk N, TTS is
    already producing chunk N+1's audio. Order is preserved naturally because each
    stage is single-threaded and the queues are FIFO.
    """

    def __init__(
        self,
        tts_model,
        lip_sync_model,
        video_path: str,
        voice_sample_path: str,
        system_prompt: Optional[str],
        conversation_history: list[dict[str, str]],
        voice_conditionals_ready: bool = False,
    ):
        self.tts_model = tts_model
        self.lip_sync_model = lip_sync_model
        self.video_path = video_path
        self.voice_sample_path = voice_sample_path
        self.system_prompt = system_prompt
        self.conversation_history = conversation_history
        # When True, the caller has already invoked tts.prepare_conditionals() with
        # the voice sample; we can skip passing audio_prompt_path on every TTS call
        # (which otherwise re-runs speaker encoding ~150 ms per chunk).
        self.voice_conditionals_ready = voice_conditionals_ready

        self.text_queue: Queue = Queue()
        self.audio_queue: Queue = Queue()
        self.video_queue: Queue = Queue()

        self.llm_thread: Optional[threading.Thread] = None
        self.tts_thread: Optional[threading.Thread] = None
        self.lip_thread: Optional[threading.Thread] = None

        self.is_running = False
        self.is_complete = False
        self.error: Optional[str] = None
        self.chunk_counter = 0
        self.lock = threading.Lock()
        self.all_sentences: list[str] = []

    def start(self, user_text: str) -> None:
        with self.lock:
            if self.is_running:
                return
            self.is_running = True
            self.is_complete = False
            self.error = None
            self.chunk_counter = 0

            for q in (self.text_queue, self.audio_queue, self.video_queue):
                while not q.empty():
                    try:
                        q.get_nowait()
                    except Empty:
                        break

        self.llm_thread = threading.Thread(
            target=self._llm_worker, args=(user_text,), daemon=True)
        self.tts_thread = threading.Thread(
            target=self._tts_worker, daemon=True)
        self.lip_thread = threading.Thread(
            target=self._lipsync_worker, daemon=True)

        self.llm_thread.start()
        self.tts_thread.start()
        self.lip_thread.start()

    def _llm_worker(self, user_text: str) -> None:
        try:
            print(f"💬 User said: {user_text}")
            print("🧠 Getting streaming response from Ollama...")

            for sentence in _get_ollama_stream(
                user_text,
                self.system_prompt,
                self.conversation_history,
            ):
                if not self.is_running:
                    break
                sentence = remove_emojis(sentence).strip()
                if sentence:
                    print(f"📝 Chunk: {sentence}")
                    self.all_sentences.append(sentence)
                    self.text_queue.put(sentence)

            self.text_queue.put(None)
            print("✅ LLM streaming complete")
        except Exception as exc:
            print(f"❌ LLM worker error: {exc}")
            self.error = str(exc)
            self.is_running = False
            self.text_queue.put(None)

    def _generate_tts(self, text: str) -> torch.Tensor:
        """Generate one TTS waveform with consistent prosody across chunks.

        - Fixed torch seed so the diffusion sampler starts from the same noise
          distribution each call (prevents pitch/timbre drift between chunks).
        - Skip audio_prompt_path when conditionals were pre-cached at upload time;
          this saves ~150 ms per chunk and ensures the same speaker embedding is
          reused (otherwise it's re-encoded each call).
        """
        torch.manual_seed(TTS_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(TTS_SEED)

        if self.voice_conditionals_ready:
            return self.tts_model.generate(text)
        return self.tts_model.generate(
            text, audio_prompt_path=self.voice_sample_path)

    def _tts_worker(self) -> None:
        """Pull text chunks, synthesize audio, push (idx, audio_path) to lipsync."""
        try:
            while self.is_running:
                try:
                    text = self.text_queue.get(timeout=1.0)
                except Empty:
                    continue

                if text is None:
                    self.audio_queue.put(None)
                    print("✅ TTS worker complete")
                    break

                with self.lock:
                    chunk_num = self.chunk_counter
                    self.chunk_counter += 1

                print(
                    f"🎤 [chunk {chunk_num}] TTS: {text[:60]}{'...' if len(text) > 60 else ''}")
                wav = self._generate_tts(text)
                wav = _trim_silence_tensor(wav, self.tts_model.sr)

                audio_path = OUTPUTS_DIR / f"chunk_{chunk_num}_audio.wav"
                _save_wav(str(audio_path), wav, self.tts_model.sr)

                self.audio_queue.put((chunk_num, str(audio_path)))
        except Exception as exc:
            print(f"❌ TTS worker error: {exc}")
            self.error = str(exc)
            self.is_running = False
            self.audio_queue.put(None)

    def _lipsync_worker(self) -> None:
        """Pull audio chunks, render lip-synced video, expose to the frontend."""
        try:
            while self.is_running:
                try:
                    item = self.audio_queue.get(timeout=1.0)
                except Empty:
                    continue

                if item is None:
                    print("✅ LipSync worker complete")
                    with self.lock:
                        self.is_complete = True
                        self.is_running = False
                    break

                chunk_num, audio_path = item
                video_path = OUTPUTS_DIR / f"chunk_{chunk_num}_video.mp4"
                print(f"🎬 [chunk {chunk_num}] LipSync render...")

                self.lip_sync_model.sync(
                    self.video_path,
                    audio_path,
                    str(video_path),
                )

                self.video_queue.put(video_path.name)
                print(f"✅ [chunk {chunk_num}] ready: {video_path.name}")
        except Exception as exc:
            print(f"❌ LipSync worker error: {exc}")
            self.error = str(exc)
            with self.lock:
                self.is_running = False

    def get_next_chunk(self) -> dict:
        with self.lock:
            if self.error:
                return {"status": "ERROR", "error": self.error}

            if not self.is_running and self.is_complete:
                try:
                    video_filename = self.video_queue.get_nowait()
                    return {"status": "READY", "video_url": f"/static/{video_filename}"}
                except Empty:
                    return {"status": "DONE"}

            try:
                video_filename = self.video_queue.get_nowait()
                return {"status": "READY", "video_url": f"/static/{video_filename}"}
            except Empty:
                if self.is_running:
                    return {"status": "WAIT"}
                return {"status": "DONE"}

    def get_full_response(self) -> str:
        return " ".join(self.all_sentences)

    def stop(self) -> None:
        with self.lock:
            self.is_running = False
