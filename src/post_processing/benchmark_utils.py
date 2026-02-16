#!/usr/bin/env python3
"""
Benchmark utilities for post-processing scripts.

Provides system information collection and timing statistics for scientific papers.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class SystemInfo:
    """System information for reproducibility in papers."""
    # OS info
    os_name: str = ""
    os_version: str = ""
    kernel_version: str = ""

    # CPU info
    cpu_model: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    cpu_freq_mhz: Optional[float] = None

    # Memory info
    ram_total_gb: float = 0.0

    # GPU info (if available)
    gpu_model: str = ""
    gpu_memory_gb: float = 0.0
    cuda_version: str = ""

    # Python info
    python_version: str = ""

    # Timestamp
    timestamp: str = ""


@dataclass
class BenchmarkStats:
    """Benchmark statistics for a post-processing run."""
    # Script info
    script_name: str = ""
    model_name: str = ""
    approach: str = ""  # single-pass, two-pass, etc.

    # Configuration
    chunk_size: int = 150
    concurrency: int = 2
    temperature: float = 0.0

    # File statistics
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    total_segments: int = 0

    # Timing
    start_time: str = ""
    end_time: str = ""
    total_duration_seconds: float = 0.0
    avg_time_per_file_seconds: float = 0.0
    avg_time_per_segment_seconds: float = 0.0

    # Per-file timing
    file_timings: dict = field(default_factory=dict)

    # System info
    system_info: Optional[SystemInfo] = None


def get_system_info() -> SystemInfo:
    """Collect system information for reproducibility."""
    info = SystemInfo()

    # Timestamp
    info.timestamp = datetime.now().isoformat()

    # OS info
    info.os_name = platform.system()
    info.os_version = platform.release()
    try:
        info.kernel_version = platform.version()
    except Exception:
        pass

    # Python version
    info.python_version = platform.python_version()

    # CPU info
    try:
        info.cpu_cores_logical = os.cpu_count() or 0

        # Try to get physical cores and model
        if platform.system() == "Linux":
            # Get CPU model
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.startswith("model name"):
                            info.cpu_model = line.split(":")[1].strip()
                            break
            except Exception:
                pass

            # Get physical cores
            try:
                result = subprocess.run(
                    ["lscpu"], capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.split("\n"):
                    if "Core(s) per socket" in line:
                        cores = int(line.split(":")[1].strip())
                    if "Socket(s)" in line:
                        sockets = int(line.split(":")[1].strip())
                info.cpu_cores_physical = cores * sockets
            except Exception:
                info.cpu_cores_physical = info.cpu_cores_logical // 2

            # Get CPU frequency
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.startswith("cpu MHz"):
                            info.cpu_freq_mhz = float(line.split(":")[1].strip())
                            break
            except Exception:
                pass

        elif platform.system() == "Darwin":  # macOS
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5
                )
                info.cpu_model = result.stdout.strip()
            except Exception:
                pass

    except Exception:
        pass

    # Memory info
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        mem_kb = int(line.split()[1])
                        info.ram_total_gb = round(mem_kb / (1024 * 1024), 1)
                        break
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5
            )
            info.ram_total_gb = round(int(result.stdout.strip()) / (1024**3), 1)
    except Exception:
        pass

    # GPU info (NVIDIA)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                info.gpu_model = parts[0].strip()
                info.gpu_memory_gb = round(float(parts[1].strip()) / 1024, 1)
    except Exception:
        pass

    # CUDA version
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    # Extract version like "12.1"
                    import re
                    match = re.search(r"release (\d+\.\d+)", line)
                    if match:
                        info.cuda_version = match.group(1)
                    break
    except Exception:
        pass

    return info


class BenchmarkTimer:
    """Context manager for timing benchmark runs."""

    def __init__(
        self,
        script_name: str,
        model_name: str,
        approach: str,
        chunk_size: int = 150,
        concurrency: int = 2,
        temperature: float = 0.0,
        output_dir: Optional[Path] = None,
    ):
        self.stats = BenchmarkStats(
            script_name=script_name,
            model_name=model_name,
            approach=approach,
            chunk_size=chunk_size,
            concurrency=concurrency,
            temperature=temperature,
        )
        self.output_dir = output_dir or Path("results/benchmarks")
        self._start_time: float = 0.0
        self._file_start_times: dict[str, float] = {}

    def __enter__(self) -> "BenchmarkTimer":
        self._start_time = time.time()
        self.stats.start_time = datetime.now().isoformat()
        self.stats.system_info = get_system_info()

        # Print system info
        print("=" * 60)
        print("SYSTEM INFORMATION")
        print("=" * 60)
        si = self.stats.system_info
        print(f"OS: {si.os_name} {si.os_version}")
        print(f"CPU: {si.cpu_model}")
        print(f"CPU Cores: {si.cpu_cores_physical} physical, {si.cpu_cores_logical} logical")
        if si.cpu_freq_mhz:
            print(f"CPU Frequency: {si.cpu_freq_mhz:.0f} MHz")
        print(f"RAM: {si.ram_total_gb} GB")
        if si.gpu_model:
            print(f"GPU: {si.gpu_model} ({si.gpu_memory_gb} GB)")
        if si.cuda_version:
            print(f"CUDA: {si.cuda_version}")
        print(f"Python: {si.python_version}")
        print("=" * 60)
        print()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stats.end_time = datetime.now().isoformat()
        self.stats.total_duration_seconds = time.time() - self._start_time

        # Calculate averages
        if self.stats.files_processed > 0:
            self.stats.avg_time_per_file_seconds = (
                self.stats.total_duration_seconds / self.stats.files_processed
            )
        if self.stats.total_segments > 0:
            self.stats.avg_time_per_segment_seconds = (
                self.stats.total_duration_seconds / self.stats.total_segments
            )

        # Print summary
        self._print_summary()

        # Save to JSON
        self._save_results()

        return False

    def start_file(self, filename: str):
        """Mark the start of processing a file."""
        self._file_start_times[filename] = time.time()

    def end_file(self, filename: str, segments: int, status: str):
        """Mark the end of processing a file."""
        if filename in self._file_start_times:
            duration = time.time() - self._file_start_times[filename]
            self.stats.file_timings[filename] = {
                "duration_seconds": round(duration, 2),
                "segments": segments,
                "status": status,
                "seconds_per_segment": round(duration / segments, 3) if segments > 0 else 0,
            }

            if status == "done":
                self.stats.files_processed += 1
                self.stats.total_segments += segments
            elif status == "skipped":
                self.stats.files_skipped += 1
            else:
                self.stats.files_failed += 1

    def _print_summary(self):
        """Print benchmark summary."""
        print()
        print("=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Script: {self.stats.script_name}")
        print(f"Model: {self.stats.model_name}")
        print(f"Approach: {self.stats.approach}")
        print(f"Chunk size: {self.stats.chunk_size}")
        print(f"Concurrency: {self.stats.concurrency}")
        print("-" * 60)
        print(f"Files processed: {self.stats.files_processed}")
        print(f"Files skipped: {self.stats.files_skipped}")
        print(f"Files failed: {self.stats.files_failed}")
        print(f"Total segments: {self.stats.total_segments}")
        print("-" * 60)
        print(f"Total duration: {self.stats.total_duration_seconds:.1f}s ({self.stats.total_duration_seconds/60:.1f} min)")
        if self.stats.files_processed > 0:
            print(f"Avg time per file: {self.stats.avg_time_per_file_seconds:.1f}s")
        if self.stats.total_segments > 0:
            print(f"Avg time per segment: {self.stats.avg_time_per_segment_seconds:.3f}s")
            print(f"Throughput: {self.stats.total_segments / self.stats.total_duration_seconds * 60:.1f} segments/min")
        print("=" * 60)

    def _save_results(self):
        """Save benchmark results to JSON file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{self.stats.script_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        # Convert to dict for JSON serialization
        data = asdict(self.stats)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nBenchmark saved to: {filepath}")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
