#!/usr/bin/env python
"""RQ1a: clean person-detection performance on every frozen pool.

Establishes what the detector does before any attack. Every later claim is a
difference against these numbers, so this runs on all four pools -- including
the held-out one, which is why the protocol must be fully frozen first. The
freeze check in ``load_protocol`` enforces that; this script does not need to.

Usage:
    scripts/03_clean_baseline.py
    scripts/03_clean_baseline.py --pools reference --limit 20   # smoke test
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from adahuman.config import DEFAULT_PROTOCOL, load_protocol  # noqa: E402
from adahuman.data.dataset import PoolDataset, collate  # noqa: E402
from adahuman.eval.detection import (  # noqa: E402
    coco_person_map,
    operating_point,
    wilson_interval,
)
from adahuman.models.detector import load_detector, person_detections  # noqa: E402
from adahuman.utils.run_log import RunLog  # noqa: E402
from adahuman.utils.seed import seed_everything  # noqa: E402

STAGE = "clean_baseline"
RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def run_pool(model, coco, protocol, pool: str, limit: int | None, workers: int):
    """Run the detector over one pool and score it."""
    dataset = PoolDataset(protocol, pool, coco)
    if limit:
        dataset.image_ids = dataset.image_ids[:limit]

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate,
    )

    category_id = protocol.get("task.category_id")
    all_detections: list[tuple[torch.Tensor, torch.Tensor]] = []
    all_targets: list[dict] = []

    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images)
            for detection in person_detections(outputs, category_id):
                all_detections.append((detection.boxes, detection.scores))
            all_targets.extend(targets)

    score_threshold = protocol.get("task.score_threshold")
    iou_threshold = protocol.get("task.iou_threshold")
    point = operating_point(
        all_detections, all_targets, score_threshold, iou_threshold
    )

    result = {
        "pool": pool,
        "n_images": len(all_targets),
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "operating_point": point.as_dict(),
    }

    if point.n_ground_truth > 0:
        low, high = wilson_interval(point.true_positives, point.n_ground_truth)
        result["operating_point"]["recall_ci95"] = [low, high]
        result["coco_map"] = coco_person_map(
            coco, all_detections, all_targets, category_id
        )
    else:
        # The negative pool has no persons to recall. Average precision is
        # undefined there; the meaningful quantity is the false-positive rate.
        result["coco_map"] = None

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=pathlib.Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--pools", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None, help="images per pool")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    protocol = load_protocol(STAGE, args.protocol)
    seed_everything(protocol.get("seed"))
    log = RunLog(STAGE, args.protocol)

    from pycocotools.coco import COCO

    annotations = protocol.get("data.annotations")
    log.input("annotations", annotations)
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(annotations)

    model = load_detector(protocol.get("model.weights_enum"))
    torch.set_num_threads(protocol.get("deploy.intra_op_threads", 4))

    pools = args.pools or sorted(protocol.stage.pools)
    log.pools_read(*pools)

    results = {}
    for pool in pools:
        print(f"\n[{pool}]")
        result = run_pool(model, coco, protocol, pool, args.limit, args.workers)
        results[pool] = result

        point = result["operating_point"]
        print(f"  images              {result['n_images']}")
        print(f"  ground-truth people {point['n_ground_truth']}")
        print(f"  detections @{result['score_threshold']}      {point['n_detections']}")
        if point["n_ground_truth"]:
            ci = point.get("recall_ci95", [float('nan')] * 2)
            print(
                f"  recall              {point['recall']:.4f} "
                f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            )
            print(f"  precision           {point['precision']:.4f}")
            print(f"  missed-detection    {point['missed_detection_rate']:.4f}")
        print(f"  false pos / image   {point['false_positives_per_image']:.4f}")
        if result["coco_map"]:
            print(f"  AP@[.50:.95]        {result['coco_map']['AP@[.50:.95]']:.4f}")
            print(f"  AP@.50              {result['coco_map']['AP@.50']:.4f}")

    if args.limit:
        log.note(f"SMOKE TEST: limited to {args.limit} images per pool; not a result")
        out_path = RESULTS / "rq1_clean_baseline_smoke.json"
    else:
        out_path = RESULTS / "rq1_clean_baseline.json"

    RESULTS.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    log.output("results", out_path)

    path = log.write()
    print(f"\nresults: {out_path.relative_to(pathlib.Path.cwd())}")
    print(f"run log: {path.relative_to(pathlib.Path.cwd())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
