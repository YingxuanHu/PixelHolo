#!/usr/bin/env python3
"""Test script to demonstrate PixelHolo timing functionality."""

import time
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pixelholo.timing import (
    time_operation,
    print_performance_summary,
    save_performance_log,
    get_profiler,
)


def simulate_model_loading():
    """Simulate model loading time."""
    print("🔄 Simulating model loading...")
    time.sleep(2.5)  # Simulate 2.5 seconds
    print("✅ Model loaded")


def simulate_tts_generation(text: str):
    """Simulate TTS generation."""
    print(f"🔄 Generating speech for: '{text[:50]}...'")
    time.sleep(1.2)  # Simulate 1.2 seconds
    print("✅ Speech generated")


def simulate_lipsync():
    """Simulate lip-sync processing."""
    print("🔄 Processing lip-sync...")
    time.sleep(3.8)  # Simulate 3.8 seconds
    print("✅ Lip-sync completed")


def simulate_web_search(query: str):
    """Simulate web search."""
    print(f"🔄 Searching for: '{query}'")
    time.sleep(1.5)  # Simulate 1.5 seconds
    print("✅ Search completed")


def main():
    """Run timing tests."""
    print("🚀 Starting PixelHolo Performance Testing")
    print("=" * 50)

    # Test 1: Model Loading
    with time_operation("model_loading", model_type="chatterbox_tts"):
        simulate_model_loading()

    # Test 2: TTS Generation (multiple calls)
    test_texts = [
        "Hello, how are you today?",
        "This is a longer text that should take more time to process.",
        "Short text.",
    ]

    for i, text in enumerate(test_texts):
        with time_operation("tts_generation", text_length=len(text), iteration=i + 1):
            simulate_tts_generation(text)

    # Test 3: Lip-sync Processing
    with time_operation("lipsync_processing", video_duration=5.2):
        simulate_lipsync()

    # Test 4: Web Search
    search_queries = [
        "What's the weather today?",
        "Latest AI news",
        "Python programming tips",
    ]

    for query in search_queries:
        with time_operation("web_search", query_length=len(query)):
            simulate_web_search(query)

    # Test 5: Background Removal
    with time_operation("background_removal", video_resolution="1920x1080", fps=30):
        print("🔄 Removing background from video...")
        time.sleep(4.2)  # Simulate 4.2 seconds
        print("✅ Background removed")

    # Print performance summary
    print("\n" + "=" * 50)
    print("📊 PERFORMANCE ANALYSIS")
    print("=" * 50)

    print_performance_summary()
    save_performance_log()

    print("\n🎯 Key Performance Insights:")
    profiler = get_profiler()
    summary = profiler.get_summary()

    if "operation_stats" in summary:
        print("\n📈 Operation Breakdown:")
        for op, stats in summary["operation_stats"].items():
            print(f"  {op}:")
            print(f"    - Average time: {stats['avg_time']:.2f}s")
            print(f"    - Total calls: {stats['count']}")
            print(f"    - Total time: {stats['total_time']:.2f}s")

    print(f"\n💾 Performance log saved to: {profiler.log_file}")
    print("✅ Testing completed!")


if __name__ == "__main__":
    main()
