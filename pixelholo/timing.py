"""Performance timing and profiling utilities for PixelHolo."""

import time
import psutil
import gc
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class TimingResult:
    """Container for timing measurement results."""

    operation: str
    duration: float
    memory_before: float
    memory_after: float
    memory_peak: float
    gpu_memory_before: Optional[float] = None
    gpu_memory_after: Optional[float] = None
    gpu_memory_peak: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """Comprehensive performance profiler for PixelHolo operations."""

    def __init__(self, log_file: Optional[Path] = None):
        self.timings: List[TimingResult] = []
        self.log_file = log_file or Path("performance_log.json")
        self._session_start = time.time()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def get_gpu_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage in MB."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
        except ImportError:
            pass
        return None

    def get_gpu_memory_peak(self) -> Optional[float]:
        """Get peak GPU memory usage in MB."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / 1024 / 1024
        except ImportError:
            pass
        return None

    @contextmanager
    def time_operation(self, operation_name: str, **metadata):
        """Context manager to time an operation with memory tracking."""
        # Pre-operation measurements
        memory_before = self.get_memory_usage()
        gpu_memory_before = self.get_gpu_memory_usage()

        # Reset GPU memory peak if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

        start_time = time.time()

        try:
            yield
        finally:
            # Post-operation measurements
            end_time = time.time()
            duration = end_time - start_time

            memory_after = self.get_memory_usage()
            gpu_memory_after = self.get_gpu_memory_usage()
            gpu_memory_peak = self.get_gpu_memory_peak()

            # Create timing result
            result = TimingResult(
                operation=operation_name,
                duration=duration,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=memory_after,  # Simplified for now
                gpu_memory_before=gpu_memory_before,
                gpu_memory_after=gpu_memory_after,
                gpu_memory_peak=gpu_memory_peak,
                metadata=metadata,
            )

            self.timings.append(result)
            self._print_timing_result(result)

    def _print_timing_result(self, result: TimingResult):
        """Print formatted timing result."""
        print(f"\n⏱️  TIMING: {result.operation}")
        print(f"   Duration: {result.duration:.3f}s")
        print(f"   Memory: {result.memory_before:.1f}MB → {result.memory_after:.1f}MB")

        if result.gpu_memory_before is not None:
            print(
                f"   GPU Memory: {result.gpu_memory_before:.1f}MB → {result.gpu_memory_after:.1f}MB"
            )
            if result.gpu_memory_peak:
                print(f"   GPU Peak: {result.gpu_memory_peak:.1f}MB")

        if result.metadata:
            print(f"   Metadata: {result.metadata}")
        print()

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.timings:
            return {"message": "No timing data available"}

        total_time = sum(t.duration for t in self.timings)
        session_duration = time.time() - self._session_start

        # Group by operation type
        operation_stats = {}
        for timing in self.timings:
            op = timing.operation
            if op not in operation_stats:
                operation_stats[op] = {
                    "count": 0,
                    "total_time": 0,
                    "avg_time": 0,
                    "min_time": float("inf"),
                    "max_time": 0,
                    "total_memory": 0,
                }

            stats = operation_stats[op]
            stats["count"] += 1
            stats["total_time"] += timing.duration
            stats["min_time"] = min(stats["min_time"], timing.duration)
            stats["max_time"] = max(stats["max_time"], timing.duration)
            stats["total_memory"] += timing.memory_after

        # Calculate averages
        for op, stats in operation_stats.items():
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["avg_memory"] = stats["total_memory"] / stats["count"]

        return {
            "session_duration": session_duration,
            "total_operations": len(self.timings),
            "total_measured_time": total_time,
            "operation_stats": operation_stats,
            "memory_efficiency": (
                (total_time / session_duration) * 100 if session_duration > 0 else 0
            ),
        }

    def print_summary(self):
        """Print formatted performance summary."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 60)

        if "message" in summary:
            print(summary["message"])
            return

        print(f"Session Duration: {summary['session_duration']:.2f}s")
        print(f"Total Operations: {summary['total_operations']}")
        print(f"Total Measured Time: {summary['total_measured_time']:.2f}s")
        print(f"Memory Efficiency: {summary['memory_efficiency']:.1f}%")

        print("\n📈 OPERATION BREAKDOWN:")
        for op, stats in summary["operation_stats"].items():
            print(f"\n  {op}:")
            print(f"    Count: {stats['count']}")
            print(f"    Total Time: {stats['total_time']:.3f}s")
            print(f"    Average Time: {stats['avg_time']:.3f}s")
            print(f"    Min Time: {stats['min_time']:.3f}s")
            print(f"    Max Time: {stats['max_time']:.3f}s")
            print(f"    Avg Memory: {stats['avg_memory']:.1f}MB")

        print("\n" + "=" * 60)

    def save_log(self):
        """Save timing data to JSON file."""
        log_data = {
            "session_start": self._session_start,
            "session_duration": time.time() - self._session_start,
            "timings": [
                {
                    "operation": t.operation,
                    "duration": t.duration,
                    "memory_before": t.memory_before,
                    "memory_after": t.memory_after,
                    "memory_peak": t.memory_peak,
                    "gpu_memory_before": t.gpu_memory_before,
                    "gpu_memory_after": t.gpu_memory_after,
                    "gpu_memory_peak": t.gpu_memory_peak,
                    "metadata": t.metadata,
                }
                for t in self.timings
            ],
            "summary": self.get_summary(),
        }

        with open(self.log_file, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"📁 Performance log saved to: {self.log_file}")

    def clear_timings(self):
        """Clear all timing data."""
        self.timings.clear()
        self._session_start = time.time()
        print("🧹 Timing data cleared")


# Global profiler instance
profiler = PerformanceProfiler()


def time_operation(operation_name: str, **metadata):
    """Decorator/context manager for timing operations."""
    return profiler.time_operation(operation_name, **metadata)


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return profiler


def print_performance_summary():
    """Print the current performance summary."""
    profiler.print_summary()


def save_performance_log():
    """Save performance data to file."""
    profiler.save_log()


def clear_performance_data():
    """Clear all performance data."""
    profiler.clear_timings()
