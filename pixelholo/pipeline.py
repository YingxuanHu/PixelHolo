"""Streaming pipeline for real-time video generation."""

import json
import re
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

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


def _save_wav(path: str, wav_tensor: torch.Tensor, sample_rate: int) -> None:
    """Persist a waveform tensor to disk using soundfile."""
    waveform = wav_tensor.detach().cpu()
    if waveform.dim() == 2:  # (C, T) -> (T, C)
        waveform = waveform.transpose(0, 1)
    elif waveform.dim() > 2:
        waveform = waveform.squeeze()
    sf.write(path, waveform.numpy(), sample_rate, subtype="PCM_16")


def _split_into_word_chunks(text: str, words_per_chunk: int = 24) -> list[str]:
    """Split text into chunks of approximately equal word count."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_word_count = 0

    for word in words:
        current_chunk.append(word)
        current_word_count += 1

        # Check if we've reached the target word count
        # Also check for sentence endings to avoid cutting mid-sentence when possible
        if current_word_count >= words_per_chunk:
            # Check if the word ends with sentence punctuation
            if word and word[-1] in '.!?':
                # Complete chunk with sentence ending
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
            elif current_word_count >= words_per_chunk * 1.5:
                # Force split if we're getting too long
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0

    # Add any remaining words
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks if chunks else [text]


def _get_ollama_stream(
    prompt: str,
    system_prompt: Optional[str],
    conversation_history: Optional[list[dict[str, str]]] = None,
) -> str:
    """Get streaming response from Ollama and yield word-based chunks."""
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
        "stream": True,  # Enable streaming
        "options": {"temperature": 0.7, "num_predict": 96},
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            OLLAMA_API_URL, json=payload, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        accumulated_text = ""
        done_detected = False
        for line in response.iter_lines():
            if not line:
                continue

            try:
                json_data = line.decode('utf-8')
                # Ollama streaming format: each line is a JSON object
                if json_data.startswith('{'):
                    data = json.loads(json_data)
                    if "response" in data:
                        chunk = data["response"]
                        accumulated_text += chunk

                        # Check if we have complete word chunks
                        chunks = _split_into_word_chunks(
                            accumulated_text, words_per_chunk=12)
                        # Yield all but the last (incomplete) chunk
                        for chunk_text in chunks[:-1]:
                            yield chunk_text
                        # Keep the last one as it might be incomplete
                        accumulated_text = chunks[-1] if chunks else ""

                    # Check for done flag (check after processing response)
                    if "done" in data and data.get("done", False):
                        # Yield any remaining text
                        if accumulated_text.strip():
                            yield accumulated_text.strip()
                        done_detected = True
                        break
            except (ValueError, KeyError) as e:
                continue

        # Yield any remaining text only if we didn't already yield it (done wasn't detected)
        if not done_detected and accumulated_text.strip():
            yield accumulated_text.strip()

    except requests.exceptions.RequestException as exc:
        print(f"Error connecting to Ollama API: {exc}")
        yield "I seem to be having trouble thinking right now."
    except Exception as exc:
        print(f"Unexpected error in streaming: {exc}")
        yield "I seem to be having trouble thinking right now."


class PipelineManager:
    """Manages the streaming pipeline for video generation."""

    def __init__(
        self,
        tts_model,
        lip_sync_model,
        video_path: str,
        voice_sample_path: str,
        system_prompt: Optional[str],
        conversation_history: list[dict[str, str]],
    ):
        self.tts_model = tts_model
        self.lip_sync_model = lip_sync_model
        self.video_path = video_path
        self.voice_sample_path = voice_sample_path
        self.system_prompt = system_prompt
        self.conversation_history = conversation_history

        # Queues
        self.text_queue: Queue[str] = Queue()
        self.video_queue: Queue[str] = Queue()

        # Threads
        self.llm_thread: Optional[threading.Thread] = None
        self.generator_thread: Optional[threading.Thread] = None

        # State
        self.is_running = False
        self.is_complete = False
        self.error: Optional[str] = None
        self.chunk_counter = 0
        self.lock = threading.Lock()
        self.all_sentences = []  # Track all generated sentences for conversation history

    def start(self, user_text: str) -> None:
        """Start the pipeline with user input."""
        with self.lock:
            if self.is_running:
                return

            self.is_running = True
            self.is_complete = False
            self.error = None
            self.chunk_counter = 0

            # Clear queues
            while not self.text_queue.empty():
                try:
                    self.text_queue.get_nowait()
                except Empty:
                    break
            while not self.video_queue.empty():
                try:
                    self.video_queue.get_nowait()
                except Empty:
                    break

        # Start LLM thread
        self.llm_thread = threading.Thread(
            target=self._llm_worker,
            args=(user_text,),
            daemon=True
        )
        self.llm_thread.start()

        # Start generator thread
        self.generator_thread = threading.Thread(
            target=self._generator_worker,
            daemon=True
        )
        self.generator_thread.start()

    def _llm_worker(self, user_text: str) -> None:
        """Thread 1: Stream LLM responses and put sentences into text_queue."""
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
                    print(f"📝 Sentence: {sentence}")
                    self.all_sentences.append(sentence)
                    self.text_queue.put(sentence)

            # Signal end of text stream
            self.text_queue.put(None)  # Sentinel value
            print("✅ LLM streaming complete")

        except Exception as exc:
            print(f"❌ LLM worker error: {exc}")
            self.error = str(exc)
            self.is_running = False

    def _generator_worker(self) -> None:
        """Thread 2: Pull sentences, generate audio/video, put filenames into video_queue."""
        try:
            while self.is_running:
                try:
                    # Get sentence from queue (with timeout)
                    sentence = self.text_queue.get(timeout=1.0)

                    # Check for sentinel (end of stream)
                    if sentence is None:
                        print("✅ Generator worker complete")
                        with self.lock:
                            self.is_complete = True
                            self.is_running = False
                        break

                    # Generate audio for this sentence
                    print(f"🎤 Generating TTS for: {sentence[:50]}...")
                    wav = self.tts_model.generate(
                        sentence,
                        audio_prompt_path=self.voice_sample_path
                    )

                    # Save audio chunk
                    chunk_num = self.chunk_counter
                    self.chunk_counter += 1
                    audio_path = OUTPUTS_DIR / f"chunk_{chunk_num}_audio.wav"
                    _save_wav(str(audio_path), wav, self.tts_model.sr)

                    # Generate video chunk
                    print(f"🎬 Generating video chunk {chunk_num}...")
                    video_path = OUTPUTS_DIR / f"chunk_{chunk_num}_video.mp4"

                    self.lip_sync_model.sync(
                        self.video_path,
                        str(audio_path),
                        str(video_path),
                    )

                    # Put video filename into queue
                    video_filename = video_path.name
                    self.video_queue.put(video_filename)
                    print(f"✅ Chunk {chunk_num} ready: {video_filename}")

                except Empty:
                    # No sentence available yet, continue waiting
                    continue
                except Exception as exc:
                    print(f"❌ Generator worker error: {exc}")
                    self.error = str(exc)
                    with self.lock:
                        self.is_running = False
                    break

        except Exception as exc:
            print(f"❌ Generator worker fatal error: {exc}")
            self.error = str(exc)
            with self.lock:
                self.is_running = False

    def get_next_chunk(self) -> dict:
        """Get the next available video chunk or status."""
        with self.lock:
            if self.error:
                return {"status": "ERROR", "error": self.error}

            if not self.is_running and self.is_complete:
                # Check if there are any remaining chunks
                try:
                    video_filename = self.video_queue.get_nowait()
                    return {"status": "READY", "video_url": f"/static/{video_filename}"}
                except Empty:
                    return {"status": "DONE"}

            # Try to get a chunk
            try:
                video_filename = self.video_queue.get_nowait()
                return {"status": "READY", "video_url": f"/static/{video_filename}"}
            except Empty:
                if self.is_running:
                    return {"status": "WAIT"}
                else:
                    return {"status": "DONE"}

    def get_full_response(self) -> str:
        """Get the full accumulated response text."""
        return " ".join(self.all_sentences)

    def stop(self) -> None:
        """Stop the pipeline."""
        with self.lock:
            self.is_running = False
