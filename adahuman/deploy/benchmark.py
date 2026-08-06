"""Latency, memory, and model-size measurement (RQ4).

These numbers describe one laptop CPU under one thread count. They are an
operational-cost comparison between two execution paths for the same model,
not a claim about production edge hardware -- no NPU, accelerator, mobile
device, or quantized build is involved.

Tail latency is reported alongside the median because a perception pipeline
misses frames on its slow requests, not its typical ones. A median that fits
the frame budget while p99 is three times over is a dropped-frame problem the
median cannot show.
"""

from __future__ import annotations

import dataclasses
import gc
import pathlib
import platform
import subprocess
import time
from typing import Any, Callable

import numpy as np


@dataclasses.dataclass
class LatencyReport:
    """Wall-clock per-inference latency over a fixed iteration count."""

    n_warmup: int
    n_measured: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    throughput_fps: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def measure_latency(
    run: Callable[[], Any], n_warmup: int, n_measured: int
) -> LatencyReport:
    """Time ``run`` repeatedly after a warmup period.

    Warmup is not optional: the first calls pay for lazy kernel selection,
    allocator growth, and cache population, and including them would report a
    startup cost as a steady-state one. Garbage collection is disabled during
    measurement so a collection pause is not attributed to inference.
    """
    for _ in range(n_warmup):
        run()

    samples: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(n_measured):
            start = time.perf_counter()
            run()
            samples.append((time.perf_counter() - start) * 1000.0)
    finally:
        if gc_was_enabled:
            gc.enable()

    array = np.array(samples)
    median = float(np.median(array))
    return LatencyReport(
        n_warmup=n_warmup,
        n_measured=n_measured,
        mean_ms=float(array.mean()),
        median_ms=median,
        p95_ms=float(np.percentile(array, 95)),
        p99_ms=float(np.percentile(array, 99)),
        min_ms=float(array.min()),
        max_ms=float(array.max()),
        std_ms=float(array.std()),
        throughput_fps=1000.0 / median if median > 0 else float("nan"),
    )


def measure_latency_paired(
    run_a: Callable[[], Any],
    run_b: Callable[[], Any],
    n_warmup: int,
    n_measured: int,
) -> tuple[LatencyReport, LatencyReport]:
    """Time two implementations by alternating between them.

    Running one path to completion and then the other confounds the path with
    everything that drifts over the intervening minutes: CPU temperature and
    frequency scaling, background processes, page-cache state. On a passively
    cooled machine the second path is measured on a hotter, slower CPU, and
    the difference is attributed to the wrong cause.

    Alternating A, B, A, B spreads any drift across both paths equally, so a
    remaining difference is attributable to the implementations. Interleaving
    costs nothing but ordering.
    """
    for _ in range(n_warmup):
        run_a()
        run_b()

    samples_a: list[float] = []
    samples_b: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(n_measured):
            start = time.perf_counter()
            run_a()
            samples_a.append((time.perf_counter() - start) * 1000.0)

            start = time.perf_counter()
            run_b()
            samples_b.append((time.perf_counter() - start) * 1000.0)
    finally:
        if gc_was_enabled:
            gc.enable()

    return _report(samples_a, n_warmup), _report(samples_b, n_warmup)


def _report(samples: list[float], n_warmup: int) -> LatencyReport:
    array = np.array(samples)
    median = float(np.median(array))
    return LatencyReport(
        n_warmup=n_warmup,
        n_measured=len(samples),
        mean_ms=float(array.mean()),
        median_ms=median,
        p95_ms=float(np.percentile(array, 95)),
        p99_ms=float(np.percentile(array, 99)),
        min_ms=float(array.min()),
        max_ms=float(array.max()),
        std_ms=float(array.std()),
        throughput_fps=1000.0 / median if median > 0 else float("nan"),
    )


def resident_memory_mb() -> float:
    """Current process resident set size, in MiB."""
    import psutil

    return psutil.Process().memory_info().rss / (1024 * 1024)


def measure_memory_delta(setup: Callable[[], Any]) -> tuple[Any, float]:
    """Resident-memory growth attributable to ``setup``.

    A coarse measure: RSS is shared with the rest of the process and the
    allocator does not return freed pages promptly. Reported as an indicative
    figure for comparing the two paths, not as an exact model footprint.
    """
    gc.collect()
    before = resident_memory_mb()
    result = setup()
    gc.collect()
    return result, resident_memory_mb() - before


def torch_model_size(model) -> dict[str, Any]:
    """Parameter and buffer footprint of a PyTorch model, in bytes."""
    parameters = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "parameter_bytes": parameters,
        "buffer_bytes": buffers,
        "total_bytes": parameters + buffers,
        "total_mb": (parameters + buffers) / (1024 * 1024),
    }


def file_size(path: pathlib.Path | str) -> dict[str, Any]:
    """On-disk size of a serialized artifact."""
    path = pathlib.Path(path)
    size = path.stat().st_size
    return {"path": str(path), "bytes": size, "mb": size / (1024 * 1024)}


def hardware_description() -> str:
    """A short, recordable identifier for the measurement machine.

    Written into the protocol so a latency figure is never quotable without the
    hardware it was measured on.
    """
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return f"{cpu} | {platform.system()} {platform.release()} | {platform.machine()}"
