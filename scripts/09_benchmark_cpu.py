#!/usr/bin/env python
"""RQ4, part two: latency, memory, and model size on the CPU deployment path.

Compares the PyTorch baseline against the exported ONNX artifact under
onnxruntime, both pinned to the same thread count, both fed the identical
sequence of pre-resized images. Reports median and tail latency, resident
memory growth, and model size.

The image sequence is cycled from the reference pool rather than a single
fixed frame: NMS cost depends on how many detections survive, so timing one
frame repeatedly would measure that frame's crowd level rather than the
model's. Both paths see the same sequence in the same order.

This is a deployment-format and CPU-runtime comparison. It is not a claim
about production edge accelerators -- no NPU, mobile device, quantization, or
pruning is involved.

Usage:
    scripts/09_benchmark_cpu.py
    scripts/09_benchmark_cpu.py --write-hardware
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from adahuman.config import DEFAULT_PROTOCOL, load_protocol  # noqa: E402
from adahuman.data.dataset import PoolDataset  # noqa: E402
from adahuman.deploy.benchmark import (  # noqa: E402
    file_size,
    hardware_description,
    measure_latency_paired,
    measure_memory_delta,
    torch_model_size,
)
from adahuman.deploy.export import make_session, resize_to_input, run_onnx  # noqa: E402
from adahuman.models.detector import load_detector  # noqa: E402
from adahuman.utils.run_log import RunLog  # noqa: E402
from adahuman.utils.seed import seed_everything  # noqa: E402

STAGE = "benchmark_cpu"
ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

#: Distinct frames cycled during timing, to average over per-frame NMS cost.
N_TIMING_IMAGES = 8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=pathlib.Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--write-hardware", action="store_true",
                        help="freeze deploy.hardware and deploy.runtime_version")
    args = parser.parse_args()

    protocol = load_protocol(STAGE, args.protocol, allow_unfrozen=True)
    seed_everything(protocol.get("seed"))
    log = RunLog(STAGE, args.protocol)
    log.pools_read("reference")

    import onnxruntime as ort

    onnx_path = ARTIFACTS / "detector_v1.onnx"
    if not onnx_path.is_file():
        raise SystemExit(f"{onnx_path} missing. Run scripts/08_export_onnx.py first.")
    log.input("onnx", onnx_path)

    input_size = tuple(protocol.get("model.input_size"))
    threads = protocol.get("deploy.intra_op_threads")
    warmup = protocol.get("deploy.warmup_iters")
    measured = protocol.get("deploy.measured_iters")

    hardware = hardware_description()
    print(f"hardware: {hardware}")
    print(f"threads:  {threads}   warmup: {warmup}   measured: {measured}\n")

    import contextlib
    import io

    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(protocol.get("data.annotations"))

    dataset = PoolDataset(protocol, "reference", coco)
    dataset.image_ids = dataset.image_ids[:N_TIMING_IMAGES]
    images = [resize_to_input(dataset[i][0], input_size) for i in range(len(dataset))]

    torch.set_num_threads(threads)

    model, torch_memory = measure_memory_delta(
        lambda: load_detector(protocol.get("model.weights_enum"))
    )
    session, onnx_memory = measure_memory_delta(
        lambda: make_session(onnx_path, threads)
    )

    # Each path gets its own counter, both starting at zero, so the two runs
    # see the same frames in the same order.
    def torch_runner():
        state = {"i": 0}

        def run():
            image = images[state["i"] % len(images)]
            state["i"] += 1
            with torch.no_grad():
                model([image])

        return run

    def onnx_runner():
        state = {"i": 0}

        def run():
            image = images[state["i"] % len(images)]
            state["i"] += 1
            run_onnx(session, image)

        return run

    print(f"timing both paths, interleaved ({measured} iterations each)")
    torch_latency, onnx_latency = measure_latency_paired(
        torch_runner(), onnx_runner(), warmup, measured
    )

    sizes = {"pytorch": torch_model_size(model), "onnx": file_size(onnx_path)}

    def row(label: str, a: float, b: float, unit: str = "ms") -> None:
        ratio = a / b if b else float("nan")
        print(f"  {label:16s}{a:10.2f}{b:12.2f}{ratio:10.2f}x   {unit}")

    print(f"\n  {'':16s}{'pytorch':>10s}{'onnxruntime':>12s}{'ratio':>10s}")
    row("median", torch_latency.median_ms, onnx_latency.median_ms)
    row("mean", torch_latency.mean_ms, onnx_latency.mean_ms)
    row("p95", torch_latency.p95_ms, onnx_latency.p95_ms)
    row("p99", torch_latency.p99_ms, onnx_latency.p99_ms)
    row("min", torch_latency.min_ms, onnx_latency.min_ms)
    row("max", torch_latency.max_ms, onnx_latency.max_ms)
    print(f"  {'throughput':16s}{torch_latency.throughput_fps:10.2f}"
          f"{onnx_latency.throughput_fps:12.2f}"
          f"{onnx_latency.throughput_fps / torch_latency.throughput_fps:10.2f}x   fps")
    print(f"  {'load memory':16s}{torch_memory:10.1f}{onnx_memory:12.1f}"
          f"{'':10s}   MiB")
    print(f"  {'model size':16s}{sizes['pytorch']['total_mb']:10.2f}"
          f"{sizes['onnx']['mb']:12.2f}{'':10s}   MB")

    results = {
        "hardware": hardware,
        "runtime": {"onnxruntime": ort.__version__, "torch": torch.__version__},
        "threads": threads,
        "n_timing_images": len(images),
        "latency": {
            "pytorch": torch_latency.as_dict(),
            "onnxruntime": onnx_latency.as_dict(),
        },
        "load_memory_mb": {"pytorch": torch_memory, "onnxruntime": onnx_memory},
        "sizes": sizes,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "rq4_benchmark.json"
    with out_path.open("w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")

    log.output("results", out_path)
    log.set("benchmark", results)

    if args.write_hardware:
        _freeze(args.protocol, {
            "  runtime_version:": ort.__version__,
            "  hardware:": hardware,
        })
        print(f"\nfroze deploy.hardware and deploy.runtime_version")

    print(f"\nresults: {out_path.relative_to(pathlib.Path.cwd())}")
    print(f"run log: {log.write().relative_to(pathlib.Path.cwd())}")
    return 0


def _freeze(path: pathlib.Path, replacements: dict[str, str]) -> None:
    lines = path.read_text().splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        for prefix, value in replacements.items():
            if line.startswith(prefix) and "PENDING" in line:
                head, _, tail = line.partition("PENDING")
                line = f"{head}{value}{tail}"
                break
        out.append(line)
    path.write_text("".join(out))


if __name__ == "__main__":
    raise SystemExit(main())
