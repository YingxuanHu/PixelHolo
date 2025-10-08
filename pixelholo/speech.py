"""Speech recognition helpers."""

from typing import Optional

import speech_recognition as sr


def listen_for_speech(timeout: int = 10) -> Optional[str]:
    """Capture audio from the microphone and convert it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        recognizer.pause_threshold = 1.0
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=timeout)
        except sr.WaitTimeoutError:
            print("No speech detected.")
            return None

    try:
        print("Recognizing...")
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio.")
    except sr.RequestError as exc:
        print(f"Could not request results from Google Speech Recognition service; {exc}")
    return None
