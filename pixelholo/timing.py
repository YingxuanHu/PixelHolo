"""Performance timing and monitoring utilities."""

import time
import functools
from contextlib import contextmanager
from typing import Optional, Dict, List, Any, Callable
import psutil
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TimingStats:
    """Store and display timing statistics."""

    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.memory_usage: Dict[str, List[Dict[str, Any]]] = {}
        self.enabled = True
        self.verbose_output = True  # Controls whether timing prints during operations

    def record(
        self,
        operation: str,
        duration: float,
        cpu_percent: Optional[float] = None,
        memory_mb: Optional[float] = None,
        gpu_memory_mb: Optional[float] = None,
    ):
        """Record timing and resource usage for an operation."""
        if not self.enabled:
            return

        if operation not in self.timings:
            self.timings[operation] = []
            self.memory_usage[operation] = []

        self.timings[operation].append(duration)

        memory_info = {}
        if cpu_percent is not None:
            memory_info['cpu_percent'] = cpu_percent
        if memory_mb is not None:
            memory_info['memory_mb'] = memory_mb
        if gpu_memory_mb is not None:
            memory_info['gpu_memory_mb'] = gpu_memory_mb

        if memory_info:
            self.memory_usage[operation].append(memory_info)

    def get_stats(self, operation: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific operation."""
        if operation not in self.timings or not self.timings[operation]:
            return None

        times = self.timings[operation]
        stats = {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'last': times[-1],
        }

        # Add memory stats if available
        if operation in self.memory_usage and self.memory_usage[operation]:
            mem_records = self.memory_usage[operation]

            if 'cpu_percent' in mem_records[-1]:
                cpu_values = [r['cpu_percent']
                              for r in mem_records if 'cpu_percent' in r]
                if cpu_values:
                    stats['avg_cpu_percent'] = sum(
                        cpu_values) / len(cpu_values)

            if 'memory_mb' in mem_records[-1]:
                mem_values = [r['memory_mb']
                              for r in mem_records if 'memory_mb' in r]
                if mem_values:
                    stats['avg_memory_mb'] = sum(mem_values) / len(mem_values)

            if 'gpu_memory_mb' in mem_records[-1]:
                gpu_values = [r['gpu_memory_mb']
                              for r in mem_records if 'gpu_memory_mb' in r]
                if gpu_values:
                    stats['avg_gpu_memory_mb'] = sum(
                        gpu_values) / len(gpu_values)

        return stats

    def print_summary(self):
        """Print a summary of all timing statistics."""
        if not self.timings:
            print("\n📊 No timing data recorded yet.")
            return

        print("\n" + "=" * 80)
        print("📊 PERFORMANCE TIMING SUMMARY")
        print("=" * 80)

        # Sort operations by total time (descending)
        sorted_ops = sorted(
            self.timings.keys(),
            key=lambda op: sum(self.timings[op]),
            reverse=True
        )

        for operation in sorted_ops:
            stats = self.get_stats(operation)
            if not stats:
                continue

            print(f"\n🔹 {operation}")
            print(f"   Calls:      {stats['count']}")
            print(f"   Total time: {stats['total']:.3f}s")
            print(f"   Mean time:  {stats['mean']:.3f}s")
            print(f"   Min/Max:    {stats['min']:.3f}s / {stats['max']:.3f}s")
            print(f"   Last call:  {stats['last']:.3f}s")

            if 'avg_cpu_percent' in stats:
                print(f"   Avg CPU:    {stats['avg_cpu_percent']:.1f}%")
            if 'avg_memory_mb' in stats:
                print(f"   Avg Memory: {stats['avg_memory_mb']:.1f} MB")
            if 'avg_gpu_memory_mb' in stats:
                print(f"   Avg GPU Mem: {stats['avg_gpu_memory_mb']:.1f} MB")

        print("\n" + "=" * 80)
        total_time = sum(sum(times) for times in self.timings.values())
        print(f"Total tracked time: {total_time:.3f}s")
        print("=" * 80 + "\n")

    def save_to_file(self, filename: str = "timing_report.txt"):
        """Save timing statistics to a file."""
        with open(filename, 'w') as f:
            f.write("PERFORMANCE TIMING REPORT\n")
            f.write("=" * 80 + "\n\n")

            sorted_ops = sorted(
                self.timings.keys(),
                key=lambda op: sum(self.timings[op]),
                reverse=True
            )

            for operation in sorted_ops:
                stats = self.get_stats(operation)
                if not stats:
                    continue

                f.write(f"Operation: {operation}\n")
                f.write(f"  Calls:      {stats['count']}\n")
                f.write(f"  Total time: {stats['total']:.3f}s\n")
                f.write(f"  Mean time:  {stats['mean']:.3f}s\n")
                f.write(
                    f"  Min/Max:    {stats['min']:.3f}s / {stats['max']:.3f}s\n")
                f.write(f"  Last call:  {stats['last']:.3f}s\n")

                if 'avg_cpu_percent' in stats:
                    f.write(f"  Avg CPU:    {stats['avg_cpu_percent']:.1f}%\n")
                if 'avg_memory_mb' in stats:
                    f.write(f"  Avg Memory: {stats['avg_memory_mb']:.1f} MB\n")
                if 'avg_gpu_memory_mb' in stats:
                    f.write(
                        f"  Avg GPU Mem: {stats['avg_gpu_memory_mb']:.1f} MB\n")
                f.write("\n")

            total_time = sum(sum(times) for times in self.timings.values())
            f.write(f"\nTotal tracked time: {total_time:.3f}s\n")

        print(f"✅ Timing report saved to {filename}")

    def reset(self):
        """Clear all timing data."""
        self.timings.clear()
        self.memory_usage.clear()

    def disable(self):
        """Disable timing collection."""
        self.enabled = False

    def enable(self):
        """Enable timing collection."""
        self.enabled = True

    def set_verbose(self, verbose: bool):
        """Enable or disable verbose timing output during operations."""
        self.verbose_output = verbose


# Global timing stats instance
_global_stats = TimingStats()


def get_timing_stats() -> TimingStats:
    """Get the global timing stats instance."""
    return _global_stats


@contextmanager
def time_operation(
    operation_name: str,
    verbose: bool = True,
    track_memory: bool = True,
):
    """
    Context manager to time an operation and track resource usage.

    Usage:
        with time_operation("Loading model"):
            model = load_model()

    Args:
        operation_name: Name of the operation being timed
        verbose: If True, print timing info immediately (respects global verbose setting)
        track_memory: If True, track CPU and GPU memory usage
    """
    # Check if timing is enabled
    if not _global_stats.enabled:
        # If disabled, just yield without any timing overhead
        yield
        return

    # Respect global verbose setting
    verbose = verbose and _global_stats.verbose_output

    process = psutil.Process(os.getpid()) if track_memory else None

    # Get initial resource usage
    if track_memory and process:
        cpu_percent_start = process.cpu_percent()
        memory_start = process.memory_info().rss / 1024 / 1024  # MB

    gpu_memory_start = None
    if track_memory and TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_memory_start = torch.cuda.memory_allocated() / 1024 / 1024  # MB

    start_time = time.time()

    if verbose:
        print(f"\n⏱️  Starting: {operation_name}...")

    try:
        yield
    finally:
        duration = time.time() - start_time

        # Calculate resource usage
        cpu_percent = None
        memory_delta = None
        gpu_memory_delta = None

        if track_memory and process:
            cpu_percent = process.cpu_percent()
            memory_end = process.memory_info().rss / 1024 / 1024
            memory_delta = memory_end - memory_start

        if track_memory and gpu_memory_start is not None:
            gpu_memory_end = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_memory_delta = gpu_memory_end - gpu_memory_start

        # Record the timing
        _global_stats.record(
            operation_name,
            duration,
            cpu_percent=cpu_percent,
            memory_mb=memory_delta,
            gpu_memory_mb=gpu_memory_delta,
        )

        if verbose:
            # Build the timing message with all available metrics
            msg_parts = [
                f"⏱️  [{operation_name}] Completed in {duration:.3f}s"]

            if memory_delta is not None and abs(memory_delta) > 0.1:
                msg_parts.append(f"Memory: {memory_delta:+.1f} MB")
            if gpu_memory_delta is not None and abs(gpu_memory_delta) > 0.1:
                msg_parts.append(f"GPU: {gpu_memory_delta:+.1f} MB")
            if cpu_percent is not None and cpu_percent > 0:
                msg_parts.append(f"CPU: {cpu_percent:.1f}%")

            if len(msg_parts) > 1:
                # Multi-line format for detailed info
                print(f"\n{'─' * 70}")
                print(f"⏱️  {operation_name}: {duration:.3f}s")
                for part in msg_parts[1:]:
                    print(f"   └─ {part}")
                print(f"{'─' * 70}")
            else:
                # Single line for simple timing
                print(f"\n⏱️  {operation_name}: {duration:.3f}s")
                print(f"{'─' * 70}")


def time_function(operation_name: Optional[str] = None, verbose: bool = True, track_memory: bool = True):
    """
    Decorator to time a function and track resource usage.

    Usage:
        @time_function("Model inference")
        def run_model(data):
            return model(data)

    Args:
        operation_name: Name for this operation (defaults to function name)
        verbose: If True, print timing info immediately
        track_memory: If True, track CPU and GPU memory usage
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with time_operation(op_name, verbose=verbose, track_memory=track_memory):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def print_timing_summary():
    """Print a summary of all recorded timings."""
    if _global_stats.enabled:
        _global_stats.print_summary()


def save_timing_report(filename: str = "timing_report.txt"):
    """Save timing statistics to a file."""
    if _global_stats.enabled:
        _global_stats.save_to_file(filename)


def reset_timing_stats():
    """Clear all timing data."""
    _global_stats.reset()


def disable_timing():
    """Disable timing collection globally."""
    _global_stats.disable()


def enable_timing():
    """Enable timing collection globally."""
    _global_stats.enable()


def set_timing_verbose(verbose: bool):
    """
    Enable or disable verbose timing output.

    When False, timing is still collected but not printed during operations.
    Summary can still be printed at the end.
    """
    _global_stats.set_verbose(verbose)


def configure_timing(enabled: bool = True, verbose: bool = True):
    """
    Configure timing system globally.

    Args:
        enabled: If True, collect timing data. If False, disable all timing.
        verbose: If True, print timing after each operation. If False, only collect data.

    Examples:
        # Disable all timing
        configure_timing(enabled=False)

        # Enable timing but silent (show summary at end only)
        configure_timing(enabled=True, verbose=False)

        # Enable timing with verbose output (default)
        configure_timing(enabled=True, verbose=True)
    """
    if enabled:
        _global_stats.enable()
    else:
        _global_stats.disable()

    _global_stats.set_verbose(verbose)
