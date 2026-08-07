#!/usr/bin/env python
"""RQ1b, part two: score the frozen patch on the held-out pool.

This is the stage that spends the held-out set. It runs once, after the patch
is frozen, and its numbers are the reported attack result. Rerunning it after
adjusting anything would make those numbers in-sample; the protocol freeze
check exists to make that hard to do by accident.

Each held-out image is evaluated twice -- clean and patched -- so the attack
effect is a paired difference on identical images rather than a comparison
between two runs. Three quantities are reported:

* **recall on patched targets** -- what the attack achieved.
* **recall on unpatched targets** in the same frames -- the internal control.
  The patch is local, so these should barely move. If they collapse too, the
  effect is global degradation and must be described that way.
* **suppression rate** -- of the targets detected when clean, the fraction lost
  under attack. This is the paired quantity, and it is the honest headline: it
  cannot be inflated by targets the detector was already missing.

Also saves pooled features for both conditions, since this stage already
computes them and the monitor stage must score exactly these images.

Usage:
    scripts/05_eval_attack.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from adahuman.attack.patch import EOTParams, apply_patch  # noqa: E402
from adahuman.config import DEFAULT_PROTOCOL, load_protocol  # noqa: E402
from adahuman.data.dataset import PoolDataset, collate  # noqa: E402
from adahuman.eval.detection import (  # noqa: E402
    coco_person_map,
    grouped_recall,
    match_image,
    operating_point,
    wilson_interval,
)
from adahuman.eval.inference import detect_and_featurize  # noqa: E402
from adahuman.models.detector import load_detector  # noqa: E402
from adahuman.utils.run_log import RunLog  # noqa: E402
from adahuman.utils.seed import seed_everything  # noqa: E402

STAGE = "eval_attack"
ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

#: Offset applied to the protocol seed for evaluation-time EOT draws, so that
#: the transforms the patch is scored under are reproducible but not the same
#: sequence it was optimized against.
EVAL_SEED_OFFSET = 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=pathlib.Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--patch", type=pathlib.Path,
                        default=ARTIFACTS / "patch_v1.pt")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="images per pool; marks all outputs as a smoke test, not a result",
    )
    parser.add_argument(
        "--dev", action="store_true",
        help="score on attack_dev instead of the held-out pool; not a result",
    )
    parser.add_argument(
        "--untrained-patch", action="store_true",
        help="use a random patch: the occlusion control condition",
    )
    args = parser.parse_args()

    smoke = args.limit is not None
    stage = "eval_attack_dev" if args.dev else STAGE
    pools = ("attack_dev",) if args.dev else ("eval_untouched", "negative")

    protocol = load_protocol(stage, args.protocol, allow_unfrozen=smoke)
    seed = protocol.get("seed")
    seed_everything(seed)
    log = RunLog(stage, args.protocol)
    log.pools_read(*pools)
    if smoke:
        log.set("freeze_check_bypassed", True)
    if args.dev:
        log.note(
            "DEVELOPMENT run on attack_dev. Informs whether the attack is worth "
            "evaluating on held-out data; it is not the reported attack result."
        )

    if args.untrained_patch:
        # The occlusion control. A random patch of the same size, placed the
        # same way, suppresses some detections purely by covering the target --
        # a smoke run at n=24 already lost 13% of patched targets this way.
        # Without this condition, the trained patch's suppression rate would be
        # read against zero, when the honest comparison is against occlusion.
        size = protocol.get("attack.patch_size_px")
        patch = torch.rand(3, size, size, generator=torch.Generator().manual_seed(seed))
        log.note("CONTROL: untrained random patch; measures occlusion, not attack")
        print(f"patch {tuple(patch.shape)} UNTRAINED (occlusion control)")
    else:
        if not args.patch.is_file():
            raise SystemExit(
                f"patch not found at {args.patch}. Train it first "
                f"(notebooks/colab_train_patch.ipynb) and place the result there."
            )
        log.input("patch", args.patch)
        payload = torch.load(args.patch, map_location="cpu")
        patch = payload["pixels"]
        print(f"patch {tuple(patch.shape)} from {payload.get('steps')} steps")

    if smoke:
        log.note(f"SMOKE TEST: limited to {args.limit} images per pool; not a result")

    import contextlib
    import io

    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(protocol.get("data.annotations"))

    model = load_detector(protocol.get("model.weights_enum"))
    torch.set_num_threads(protocol.get("deploy.intra_op_threads", 4))

    eot = EOTParams.from_protocol(protocol)
    hook_path = protocol.get("monitor.feature_hook")
    category_id = protocol.get("task.category_id")
    score_threshold = protocol.get("task.score_threshold")
    iou_threshold = protocol.get("task.iou_threshold")
    scale_of_bbox = protocol.get("attack.patch_scale_of_bbox")
    min_area = protocol.get("attack.min_target_bbox_area")

    results: dict[str, object] = {}
    feature_store: dict[str, np.ndarray] = {}

    for pool in pools:
        print(f"\n[{pool}]")
        generator = torch.Generator().manual_seed(seed + EVAL_SEED_OFFSET)

        dataset = PoolDataset(protocol, pool, coco)
        if args.limit:
            dataset.image_ids = dataset.image_ids[: args.limit]
        loader = DataLoader(
            dataset, batch_size=8, shuffle=False,
            num_workers=args.workers, collate_fn=collate,
        )

        clean_dets, attacked_dets, all_targets = [], [], []
        clean_feats, attacked_feats = [], []

        for images, targets in loader:
            dets, feats = detect_and_featurize(model, images, category_id, hook_path)
            clean_dets.extend(dets)
            clean_feats.append(feats)

            patched_images = []
            for image, target in zip(images, targets):
                real = target["boxes"][~target["iscrowd"].bool()]
                patched_image, applied = apply_patch(
                    image, real, patch, eot, scale_of_bbox, min_area, generator
                )
                patched_images.append(patched_image)
                target["patched_gt"] = set(applied)
            all_targets.extend(targets)

            dets, feats = detect_and_featurize(
                model, patched_images, category_id, hook_path
            )
            attacked_dets.extend(dets)
            attacked_feats.append(feats)

        feature_store[f"{pool}_clean"] = torch.cat(clean_feats).numpy()
        feature_store[f"{pool}_attacked"] = torch.cat(attacked_feats).numpy()

        n_patched = sum(len(t["patched_gt"]) for t in all_targets)
        print(f"  images              {len(all_targets)}")
        print(f"  targets patched     {n_patched}")

        if n_patched == 0:
            # The negative pool has no person boxes, so nothing is patched and
            # the two conditions are identical by construction.
            results[pool] = {
                "n_images": len(all_targets),
                "n_patched_targets": 0,
                "note": "no person targets; clean and attacked conditions are identical",
                "clean": {"operating_point": operating_point(
                    clean_dets, all_targets, score_threshold, iou_threshold).as_dict()},
                "attacked": {"operating_point": operating_point(
                    attacked_dets, all_targets, score_threshold, iou_threshold).as_dict()},
            }
            fp_clean = results[pool]["clean"]["operating_point"]
            print(f"  false pos / image   {fp_clean['false_positives_per_image']:.4f}")
            continue

        clean_point = operating_point(
            clean_dets, all_targets, score_threshold, iou_threshold
        )
        attacked_point = operating_point(
            attacked_dets, all_targets, score_threshold, iou_threshold
        )
        clean_groups = grouped_recall(
            clean_dets, all_targets, score_threshold, iou_threshold
        )
        attacked_groups = grouped_recall(
            attacked_dets, all_targets, score_threshold, iou_threshold
        )
        suppression = _paired_suppression(
            clean_dets, attacked_dets, all_targets, score_threshold, iou_threshold
        )

        results[pool] = {
            "n_images": len(all_targets),
            "n_patched_targets": n_patched,
            "clean": {
                "operating_point": clean_point.as_dict(),
                "grouped_recall": clean_groups,
                "coco_map": coco_person_map(coco, clean_dets, all_targets, category_id),
            },
            "attacked": {
                "operating_point": attacked_point.as_dict(),
                "grouped_recall": attacked_groups,
                "coco_map": coco_person_map(
                    coco, attacked_dets, all_targets, category_id
                ),
            },
            "suppression": suppression,
        }

        print(f"\n  {'':22s}{'clean':>10s}{'attacked':>12s}{'delta':>10s}")
        _row("recall, patched", clean_groups["patched"]["recall"],
             attacked_groups["patched"]["recall"])
        _row("recall, unpatched", clean_groups["unpatched"]["recall"],
             attacked_groups["unpatched"]["recall"])
        _row("recall, overall", clean_point.recall, attacked_point.recall)
        if results[pool]["clean"]["coco_map"] and results[pool]["attacked"]["coco_map"]:
            _row("AP@.50",
                 results[pool]["clean"]["coco_map"]["AP@.50"],
                 results[pool]["attacked"]["coco_map"]["AP@.50"])
        print(
            f"\n  suppression rate    {suppression['rate']:.4f} "
            f"[{suppression['ci95'][0]:.4f}, {suppression['ci95'][1]:.4f}]  "
            f"({suppression['suppressed']}/{suppression['detected_when_clean']} "
            f"patched targets lost)"
        )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    suffix = (
        ("_dev" if args.dev else "")
        + ("_smoke" if smoke else "")
        + ("_control" if args.untrained_patch else "")
    )
    features_path = ARTIFACTS / f"features_eval{suffix}.npz"
    np.savez_compressed(features_path, **feature_store)

    out_path = RESULTS / f"rq1_attack{suffix}.json"
    with out_path.open("w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, default=_jsonable)
        handle.write("\n")

    log.output("results", out_path)
    log.output("features", features_path)
    print(f"\nresults: {out_path.relative_to(pathlib.Path.cwd())}")
    print(f"run log: {log.write().relative_to(pathlib.Path.cwd())}")
    return 0


def _row(label: str, clean: float, attacked: float) -> None:
    print(f"  {label:22s}{clean:10.4f}{attacked:12.4f}{attacked - clean:+10.4f}")


def _paired_suppression(clean_dets, attacked_dets, targets, score_thr, iou_thr):
    """Fraction of patched targets detected when clean but lost under attack.

    Paired per target, so targets the detector never found do not enter the
    denominator. Also counts the reverse -- targets found only under attack --
    which should be near zero and is reported because a non-trivial count
    would indicate the patch is perturbing matching rather than suppressing.
    """
    suppressed = detected_clean = revealed = 0

    for clean, attacked, target in zip(clean_dets, attacked_dets, targets):
        patched: set[int] = target.get("patched_gt", set())
        if not patched:
            continue
        _, _, _, matched_clean = match_image(
            clean[0], clean[1], target["boxes"], target["iscrowd"], score_thr, iou_thr
        )
        _, _, _, matched_attacked = match_image(
            attacked[0], attacked[1], target["boxes"], target["iscrowd"],
            score_thr, iou_thr,
        )
        for index in patched:
            was, now = index in matched_clean, index in matched_attacked
            detected_clean += int(was)
            suppressed += int(was and not now)
            revealed += int(now and not was)

    rate = suppressed / detected_clean if detected_clean else float("nan")
    low, high = wilson_interval(suppressed, detected_clean)
    return {
        "suppressed": suppressed,
        "detected_when_clean": detected_clean,
        "revealed_only_under_attack": revealed,
        "rate": rate,
        "ci95": [low, high],
    }


def _jsonable(value):
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)}")


if __name__ == "__main__":
    raise SystemExit(main())
