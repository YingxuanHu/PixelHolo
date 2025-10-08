#!/usr/bin/env python3
"""
Demo script showing how to use PixelHolo's timing system.
This script simulates the key operations and shows timing output.
"""

import time
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pixelholo.timing import (
    time_operation,
    print_performance_summary,
    save_performance_log,
)


def demo_model_loading():
    """Simulate model loading with timing."""
    print("🔄 Loading Chatterbox-TTS model...")

    with time_operation("tts_model_loading", device="cuda", model_size="large"):
        # Simulate model loading time
        time.sleep(2.5)
        print("✅ TTS model loaded")

    print("🔄 Loading LipSync model...")

    with time_operation("lipsync_model_loading", device="cuda", model_type="wav2lip"):
        # Simulate LipSync loading
        time.sleep(3.2)
        print("✅ LipSync model loaded")


def demo_tts_generation():
    """Simulate TTS generation with different text lengths."""
    test_texts = [
        "Hello, how are you?",
        "This is a longer text that should take more time to process and generate speech from.",
        "Short response.",
        "This is an even longer text that will definitely take more time to process through the TTS system and generate high-quality speech output.",
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"🔄 Generating speech {i}/4...")

        with time_operation(
            "tts_generation",
            text_length=len(text),
            iteration=i,
            text_preview=text[:30] + "...",
        ):
            # Simulate TTS generation (longer text = more time)
            time.sleep(0.5 + (len(text) * 0.01))
            print(f"✅ Speech generated for text {i}")


def demo_lipsync_processing():
    """Simulate lip-sync processing."""
    print("🔄 Processing lip-sync...")

    with time_operation(
        "lipsync_generation", video_duration=5.2, resolution="1920x1080", fps=30
    ):
        # Simulate lip-sync processing
        time.sleep(4.1)
        print("✅ Lip-sync processing completed")


def demo_background_removal():
    """Simulate background removal."""
    print("🔄 Removing background from video...")

    with time_operation(
        "background_removal",
        video_path="input_video.mp4",
        resolution="1920x1080",
        total_frames=150,
    ):
        # Simulate background removal
        time.sleep(6.8)
        print("✅ Background removal completed")


def demo_ai_operations():
    """Simulate AI operations."""
    queries = [
        "What's the weather like?",
        "Tell me about artificial intelligence",
        "How do I cook pasta?",
    ]

    for query in queries:
        print(f"🔄 Processing query: '{query}'")

        # Simulate internet search decision
        with time_operation("internet_detection", query_length=len(query)):
            needs_search = len(query) > 20  # Simple heuristic
            time.sleep(0.1)

        if needs_search:
            print("🌐 Query requires internet search...")
            with time_operation("web_search", query_length=len(query)):
                time.sleep(1.2)
                print("✅ Search completed")

        # Simulate Ollama response
        with time_operation("ollama_response", prompt_length=len(query)):
            time.sleep(0.8)
            print("✅ AI response generated")


def main():
    """Run the complete demo."""
    print("🚀 PixelHolo Timing System Demo")
    print("=" * 50)
    print("This demo shows how timing works in PixelHolo.")
    print("Each operation will be timed and measured.\n")

    # Demo different operations
    print("📦 MODEL LOADING PHASE")
    print("-" * 30)
    demo_model_loading()

    print("\n🎤 TTS GENERATION PHASE")
    print("-" * 30)
    demo_tts_generation()

    print("\n🎬 VIDEO PROCESSING PHASE")
    print("-" * 30)
    demo_lipsync_processing()
    demo_background_removal()

    print("\n🤖 AI OPERATIONS PHASE")
    print("-" * 30)
    demo_ai_operations()

    # Show performance summary
    print("\n" + "=" * 50)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 50)
    print_performance_summary()

    # Save log
    save_performance_log()

    print("\n✅ Demo completed!")
    print("📁 Check 'performance_log.json' for detailed data")
    print("🔍 Run 'python analyze_performance.py' for analysis")


if __name__ == "__main__":
    main()
