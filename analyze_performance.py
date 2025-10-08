#!/usr/bin/env python3
"""Performance analysis script for PixelHolo timing data."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np


def load_performance_data(log_file: Path) -> Dict[str, Any]:
    """Load performance data from JSON log file."""
    try:
        with open(log_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Performance log not found: {log_file}")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing performance log: {e}")
        return {}


def analyze_operation_times(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze operation timing statistics."""
    if not data or "timings" not in data:
        return {}

    timings = data["timings"]
    operation_stats = {}

    for timing in timings:
        op = timing["operation"]
        duration = timing["duration"]

        if op not in operation_stats:
            operation_stats[op] = {
                "durations": [],
                "memory_usage": [],
                "gpu_memory": [],
                "count": 0,
            }

        stats = operation_stats[op]
        stats["durations"].append(duration)
        stats["memory_usage"].append(timing["memory_after"])
        if timing.get("gpu_memory_after"):
            stats["gpu_memory"].append(timing["gpu_memory_after"])
        stats["count"] += 1

    # Calculate statistics
    for op, stats in operation_stats.items():
        durations = stats["durations"]
        stats["avg_time"] = np.mean(durations)
        stats["min_time"] = np.min(durations)
        stats["max_time"] = np.max(durations)
        stats["std_time"] = np.std(durations)
        stats["total_time"] = np.sum(durations)

        if stats["memory_usage"]:
            stats["avg_memory"] = np.mean(stats["memory_usage"])
            stats["max_memory"] = np.max(stats["memory_usage"])

        if stats["gpu_memory"]:
            stats["avg_gpu_memory"] = np.mean(stats["gpu_memory"])
            stats["max_gpu_memory"] = np.max(stats["gpu_memory"])

    return operation_stats


def print_detailed_analysis(operation_stats: Dict[str, Any]):
    """Print detailed performance analysis."""
    print("\n" + "=" * 80)
    print("📊 DETAILED PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Sort operations by total time
    sorted_ops = sorted(
        operation_stats.items(), key=lambda x: x[1]["total_time"], reverse=True
    )

    print(f"\n🔍 OPERATION RANKING (by total time):")
    for i, (op, stats) in enumerate(sorted_ops, 1):
        print(
            f"{i:2d}. {op:<20} | Total: {stats['total_time']:6.2f}s | "
            f"Avg: {stats['avg_time']:5.2f}s | Count: {stats['count']:2d}"
        )

    print(f"\n📈 DETAILED STATISTICS:")
    print("-" * 80)
    print(
        f"{'Operation':<20} {'Count':<6} {'Avg(s)':<8} {'Min(s)':<8} {'Max(s)':<8} {'Total(s)':<8} {'Std(s)':<8}"
    )
    print("-" * 80)

    for op, stats in sorted_ops:
        print(
            f"{op:<20} {stats['count']:<6} {stats['avg_time']:<8.3f} "
            f"{stats['min_time']:<8.3f} {stats['max_time']:<8.3f} "
            f"{stats['total_time']:<8.3f} {stats['std_time']:<8.3f}"
        )

    # Memory analysis
    print(f"\n💾 MEMORY USAGE ANALYSIS:")
    print("-" * 60)
    print(f"{'Operation':<20} {'Avg Memory(MB)':<15} {'Max Memory(MB)':<15}")
    print("-" * 60)

    for op, stats in sorted_ops:
        if "avg_memory" in stats:
            print(f"{op:<20} {stats['avg_memory']:<15.1f} {stats['max_memory']:<15.1f}")

    # GPU Memory analysis
    gpu_ops = [(op, stats) for op, stats in sorted_ops if "avg_gpu_memory" in stats]
    if gpu_ops:
        print(f"\n🎮 GPU MEMORY USAGE:")
        print("-" * 60)
        print(f"{'Operation':<20} {'Avg GPU(MB)':<15} {'Max GPU(MB)':<15}")
        print("-" * 60)

        for op, stats in gpu_ops:
            print(
                f"{op:<20} {stats['avg_gpu_memory']:<15.1f} {stats['max_gpu_memory']:<15.1f}"
            )


def identify_bottlenecks(operation_stats: Dict[str, Any]) -> List[str]:
    """Identify performance bottlenecks."""
    bottlenecks = []

    # Find operations with high average time
    high_avg_time = [
        (op, stats) for op, stats in operation_stats.items() if stats["avg_time"] > 3.0
    ]  # > 3 seconds average

    if high_avg_time:
        bottlenecks.append("🐌 High Average Time Operations:")
        for op, stats in sorted(
            high_avg_time, key=lambda x: x[1]["avg_time"], reverse=True
        ):
            bottlenecks.append(f"   - {op}: {stats['avg_time']:.2f}s average")

    # Find operations with high variability
    high_variability = [
        (op, stats)
        for op, stats in operation_stats.items()
        if stats["std_time"] > stats["avg_time"] * 0.5
    ]  # High coefficient of variation

    if high_variability:
        bottlenecks.append("\n📊 High Variability Operations:")
        for op, stats in sorted(
            high_variability, key=lambda x: x[1]["std_time"], reverse=True
        ):
            cv = stats["std_time"] / stats["avg_time"] if stats["avg_time"] > 0 else 0
            bottlenecks.append(f"   - {op}: {cv:.2f} coefficient of variation")

    # Find memory-intensive operations
    high_memory = [
        (op, stats)
        for op, stats in operation_stats.items()
        if "max_memory" in stats and stats["max_memory"] > 1000
    ]  # > 1GB

    if high_memory:
        bottlenecks.append("\n💾 High Memory Usage Operations:")
        for op, stats in sorted(
            high_memory, key=lambda x: x[1]["max_memory"], reverse=True
        ):
            bottlenecks.append(f"   - {op}: {stats['max_memory']:.1f}MB peak")

    return bottlenecks


def generate_recommendations(operation_stats: Dict[str, Any]) -> List[str]:
    """Generate optimization recommendations."""
    recommendations = []

    # Model loading optimization
    model_ops = [op for op in operation_stats.keys() if "model" in op.lower()]
    if model_ops:
        recommendations.append("🚀 Model Loading Optimization:")
        recommendations.append("   - Implement model caching to avoid reloading")
        recommendations.append("   - Use model quantization (INT8/FP16)")
        recommendations.append("   - Pre-load models in background threads")

    # TTS optimization
    tts_ops = [op for op in operation_stats.keys() if "tts" in op.lower()]
    if tts_ops:
        recommendations.append("\n🎤 TTS Optimization:")
        recommendations.append("   - Use faster TTS models (Tortoise, Bark)")
        recommendations.append("   - Implement audio caching for repeated phrases")
        recommendations.append("   - Use streaming TTS for long texts")

    # Video processing optimization
    video_ops = [
        op
        for op in operation_stats.keys()
        if any(x in op.lower() for x in ["lipsync", "background", "video"])
    ]
    if video_ops:
        recommendations.append("\n🎬 Video Processing Optimization:")
        recommendations.append(
            "   - Use frame sampling instead of processing every frame"
        )
        recommendations.append("   - Implement GPU parallel processing")
        recommendations.append("   - Use video streaming for real-time processing")
        recommendations.append("   - Cache processed video segments")

    # Web search optimization
    if "web_search" in operation_stats:
        recommendations.append("\n🌐 Web Search Optimization:")
        recommendations.append("   - Implement search result caching")
        recommendations.append("   - Use async HTTP requests")
        recommendations.append("   - Add search timeout limits")

    return recommendations


def main():
    """Main analysis function."""
    log_file = Path("performance_log.json")

    print("🔍 PixelHolo Performance Analysis")
    print("=" * 50)

    # Load data
    data = load_performance_data(log_file)
    if not data:
        print("❌ No performance data found. Run the application first.")
        return

    # Analyze operations
    operation_stats = analyze_operation_times(data)
    if not operation_stats:
        print("❌ No operation data found in log.")
        return

    # Print detailed analysis
    print_detailed_analysis(operation_stats)

    # Identify bottlenecks
    bottlenecks = identify_bottlenecks(operation_stats)
    if bottlenecks:
        print(f"\n⚠️  PERFORMANCE BOTTLENECKS:")
        for bottleneck in bottlenecks:
            print(bottleneck)

    # Generate recommendations
    recommendations = generate_recommendations(operation_stats)
    if recommendations:
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        for rec in recommendations:
            print(rec)

    print(f"\n✅ Analysis complete! Check the detailed breakdown above.")


if __name__ == "__main__":
    main()
