#!/usr/bin/env python
"""RQ4, part one: export the detector to ONNX and measure conversion fidelity.

The whole detector is exported, including non-maximum suppression. Fidelity is
then measured on reference-pool images at the operating threshold: does the
converted graph produce the same person detections the PyTorch model does?

Uses the reference pool, never the held-out pool. Conversion fidelity is a
property of the model, not of the evaluation set, so there is no reason to
spend held-out data on it.

Usage:
    scripts/08_export_onnx.py
    scripts/08_export_onnx.py --probe-size 10     # quicker check
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from adahuman.config import DEFAULT_PROTOCOL, load_protocol  # noqa: E402
from adahuman.data.dataset import PoolDataset  # noqa: E402
from adahuman.deploy.benchmark import file_size, torch_model_size  # noqa: E402
from adahuman.deploy.export import (  # noqa: E402
    compare_fidelity,
    export_detector,
    make_session,
    resize_to_input,
)
from adahuman.models.detector import load_detector  # noqa: E402
from adahuman.utils.hashing import sha256_file, short  # noqa: E402
from adahuman.utils.run_log import RunLog  # noqa: E402
from adahuman.utils.seed import seed_everything  # noqa: E402

STAGE = "export_onnx"
ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=pathlib.Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--probe-size", type=int, default=None)
    args = parser.parse_args()

    protocol = load_protocol(STAGE, args.protocol)
    seed_everything(protocol.get("seed"))
    log = RunLog(STAGE, args.protocol)
    log.pools_read("reference")

    input_size = tuple(protocol.get("model.input_size"))
    opset = protocol.get("deploy.onnx_opset")
    threads = protocol.get("deploy.intra_op_threads")
    score_threshold = protocol.get("task.score_threshold")
    category_id = protocol.get("task.category_id")
    probe_size = args.probe_size or protocol.get("deploy.fidelity_probe_size")

    model = load_detector(protocol.get("model.weights_enum"))
    torch.set_num_threads(threads)

    print(f"exporting to ONNX opset {opset}, input {input_size}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    onnx_path = ARTIFACTS / "detector_v1.onnx"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        onnx_path, export_warnings = export_detector(
            model, onnx_path, input_size, opset
        )

    digest = sha256_file(onnx_path)
    sizes = {
        "onnx": file_size(onnx_path),
        "pytorch": torch_model_size(model),
    }
    print(f"  onnx      {sizes['onnx']['mb']:.2f} MB  sha256 {short(digest)}")
    print(f"  pytorch   {sizes['pytorch']['total_mb']:.2f} MB "
          f"({sizes['pytorch']['n_parameters']:,} parameters)")

    # Export warnings frequently include unsupported-operator notices and graph
    # rewrites. Recorded verbatim; the protocol requires disclosing them.
    unique_warnings = sorted(set(export_warnings))
    print(f"  warnings  {len(unique_warnings)} unique during export")
    for message in unique_warnings[:5]:
        print(f"    {message[:110]}")
    if len(unique_warnings) > 5:
        print(f"    ... and {len(unique_warnings) - 5} more (see run log)")

    print(f"\nmeasuring fidelity on {probe_size} reference images")
    import contextlib
    import io

    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(protocol.get("data.annotations"))

    dataset = PoolDataset(protocol, "reference", coco)
    dataset.image_ids = dataset.image_ids[:probe_size]
    images = [resize_to_input(dataset[i][0], input_size) for i in range(len(dataset))]

    session = make_session(onnx_path, threads)
    report = compare_fidelity(model, session, images, score_threshold, category_id)

    print(f"  images                  {report.n_images}")
    print(f"  detection-count mismatch {report.n_count_mismatches}")
    print(f"  max |dbox|              {report.max_abs_box_deviation:.3e} px")
    print(f"  max |dscore|            {report.max_abs_score_deviation:.3e}")
    print(f"  mean |dbox|             {report.mean_abs_box_deviation:.3e} px")
    print(f"  mean |dscore|           {report.mean_abs_score_deviation:.3e}")

    results = {
        "onnx": {
            "path": str(onnx_path.relative_to(ROOT)),
            "sha256": digest,
            "opset": opset,
            "input_size": list(input_size),
            "batch_size": 1,
            "includes_nms": True,
        },
        "sizes": sizes,
        "fidelity": report.as_dict(),
        "export_warnings": unique_warnings,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "rq4_export.json"
    with out_path.open("w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")

    log.output("onnx", onnx_path)
    log.output("results", out_path)
    log.set("fidelity", report.as_dict())
    log.set("export_warnings", unique_warnings)

    print(f"\nresults: {out_path.relative_to(pathlib.Path.cwd())}")
    print(f"run log: {log.write().relative_to(pathlib.Path.cwd())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
