#!/usr/bin/env python
"""RQ3, part two: score the frozen monitor on the held-out pool.

Reports two separations, and the second is the one that matters:

1. **clean vs adversarial** -- can the monitor see the attack at all?
2. **ordinary shift vs adversarial** -- can it tell the attack apart from fog,
   motion blur, noise, overexposure, and JPEG artefacts?

A monitor that scores well on (1) and near chance on (2) is a novelty detector,
not an attack detector. In deployment it would alarm on every foggy morning,
operators would switch it off, and the reported AUROC would have been
meaningless. Both numbers are therefore reported together, and the conclusion
is written from the second.

Consumes the clean and attacked features saved by ``05_eval_attack.py`` so that
the monitor is scored on exactly the images whose detection results were
reported. Ordinary-shift features are computed here.

Usage:
    scripts/07_eval_monitor.py
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

from adahuman.config import DEFAULT_PROTOCOL, load_protocol  # noqa: E402
from adahuman.data.dataset import PoolDataset, collate  # noqa: E402
from adahuman.data.shift import apply_corruption  # noqa: E402
from adahuman.eval.inference import detect_and_featurize  # noqa: E402
from adahuman.eval.monitor import separation  # noqa: E402
from adahuman.models.detector import load_detector  # noqa: E402
from adahuman.monitor.mahalanobis import FeatureDistanceMonitor  # noqa: E402
from adahuman.utils.run_log import RunLog  # noqa: E402
from adahuman.utils.seed import seed_everything  # noqa: E402

STAGE = "eval_monitor"
ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

#: Offset for corruption RNG draws, distinct from the attack evaluation's.
SHIFT_SEED_OFFSET = 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=pathlib.Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="images of ordinary shift; consumes the smoke features from 05",
    )
    args = parser.parse_args()

    smoke = args.limit is not None
    protocol = load_protocol(STAGE, args.protocol, allow_unfrozen=smoke)
    seed = protocol.get("seed")
    seed_everything(seed)
    log = RunLog(STAGE, args.protocol)
    log.pools_read("eval_untouched", "negative")
    if smoke:
        log.set("freeze_check_bypassed", True)
        log.note(f"SMOKE TEST: {args.limit} images, smoke features; not a result")

    suffix = "_smoke" if smoke else ""
    monitor_path = ARTIFACTS / "monitor_v1.npz"
    features_path = ARTIFACTS / f"features_eval{suffix}.npz"
    for path in (monitor_path, features_path):
        if not path.is_file():
            raise SystemExit(
                f"{path} not found. Run 06_fit_monitor.py and 05_eval_attack.py first."
            )
    log.input("monitor", monitor_path)
    log.input("eval_features", features_path)

    monitor = FeatureDistanceMonitor.load(monitor_path)
    if monitor.threshold is None:
        raise SystemExit("monitor has no frozen threshold; rerun 06_fit_monitor.py")

    stored = np.load(features_path)
    clean = stored["eval_untouched_clean"]
    attacked = stored["eval_untouched_attacked"]
    print(f"clean {clean.shape}  attacked {attacked.shape}")

    shifted = _shifted_features(protocol, args.workers, args.limit)
    print(f"shifted {shifted['all'].shape}")

    clean_scores = monitor.score(clean)
    attacked_scores = monitor.score(attacked)
    shift_scores = {name: monitor.score(f) for name, f in shifted.items()}

    results = {
        "monitor": monitor.summary(),
        "score_medians": {
            "clean": float(np.median(clean_scores)),
            "attacked": float(np.median(attacked_scores)),
            "ordinary_shift": float(np.median(shift_scores["all"])),
        },
        "clean_vs_adversarial": separation(
            clean_scores, attacked_scores, monitor.threshold
        ),
        "ordinary_shift_vs_adversarial": separation(
            shift_scores["all"], attacked_scores, monitor.threshold
        ),
        "clean_vs_ordinary_shift": separation(
            clean_scores, shift_scores["all"], monitor.threshold
        ),
        "per_corruption_vs_adversarial": {
            name: separation(scores, attacked_scores, monitor.threshold)
            for name, scores in shift_scores.items()
            if name != "all"
        },
    }

    print(f"\n  threshold (frozen)  {monitor.threshold:.2f}")
    print(f"\n  {'comparison':32s}{'AUROC':>8s}{'AP':>8s}{'TPR@thr':>10s}{'FPR@thr':>10s}")
    for key in (
        "clean_vs_adversarial",
        "ordinary_shift_vs_adversarial",
        "clean_vs_ordinary_shift",
    ):
        row = results[key]
        print(
            f"  {key:32s}{row['auroc']:8.4f}{row['average_precision']:8.4f}"
            f"{row['tpr_at_threshold']:10.4f}{row['fpr_at_threshold']:10.4f}"
        )

    print(f"\n  per corruption vs adversarial:")
    for name, row in results["per_corruption_vs_adversarial"].items():
        print(f"    {name:24s}AUROC {row['auroc']:.4f}")

    verdict = _verdict(results)
    results["verdict"] = verdict
    print(f"\n  {verdict}")
    log.note(verdict)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"rq3_monitor_eval{suffix}.json"
    with out_path.open("w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")

    scores_path = ARTIFACTS / f"monitor_scores_eval{suffix}.npz"
    np.savez_compressed(
        scores_path, clean=clean_scores, attacked=attacked_scores, **shift_scores
    )

    log.output("results", out_path)
    log.output("scores", scores_path)
    print(f"\nresults: {out_path.relative_to(pathlib.Path.cwd())}")
    print(f"run log: {log.write().relative_to(pathlib.Path.cwd())}")
    return 0


def _shifted_features(
    protocol, workers: int, limit: int | None = None
) -> dict[str, np.ndarray]:
    """Features for each ordinary corruption, plus their union.

    Corruptions are applied only to held-out images, one corruption per image,
    cycling through the list so each is represented in roughly equal
    proportion without multiplying the number of forward passes by five.
    """
    import contextlib
    import io

    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(protocol.get("data.annotations"))

    model = load_detector(protocol.get("model.weights_enum"))
    torch.set_num_threads(protocol.get("deploy.intra_op_threads", 4))

    corruptions = protocol.get("ordinary_shift.corruptions")
    severity = protocol.get("ordinary_shift.severity")
    hook_path = protocol.get("monitor.feature_hook")
    category_id = protocol.get("task.category_id")

    dataset = PoolDataset(protocol, "eval_untouched", coco)
    if limit:
        dataset.image_ids = dataset.image_ids[:limit]
    loader = DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=workers, collate_fn=collate
    )

    generator = torch.Generator().manual_seed(protocol.get("seed") + SHIFT_SEED_OFFSET)
    buckets: dict[str, list[torch.Tensor]] = {name: [] for name in corruptions}
    index = 0

    print("applying ordinary shift")
    for images, _ in loader:
        names, corrupted = [], []
        for image in images:
            name = corruptions[index % len(corruptions)]
            index += 1
            names.append(name)
            corrupted.append(apply_corruption(image, name, generator, severity))

        _, features = detect_and_featurize(model, corrupted, category_id, hook_path)
        for name, row in zip(names, features):
            buckets[name].append(row)

    output = {
        name: torch.stack(rows).numpy() for name, rows in buckets.items() if rows
    }
    output["all"] = np.concatenate(list(output.values()), axis=0)
    return output


def _verdict(results: dict) -> str:
    """A plain-language reading of the numbers, fixed in advance.

    Written as code so the conclusion follows from thresholds chosen before the
    result was seen, rather than from prose composed afterwards.
    """
    against_shift = results["ordinary_shift_vs_adversarial"]["auroc"]
    against_clean = results["clean_vs_adversarial"]["auroc"]

    if against_shift >= 0.80:
        return (
            f"POSITIVE: separates adversarial from ordinary shift "
            f"(AUROC {against_shift:.3f})."
        )
    if against_shift >= 0.65:
        return (
            f"MIXED: some separation from ordinary shift (AUROC "
            f"{against_shift:.3f}), too weak to act on unaided."
        )
    if against_clean >= 0.80:
        return (
            f"NEGATIVE: flags unusual inputs (clean AUROC {against_clean:.3f}) "
            f"but does not distinguish adversarial from benign shift "
            f"(AUROC {against_shift:.3f}). This is novelty detection, not "
            f"attack detection."
        )
    return (
        f"NEGATIVE: no useful separation in either framing "
        f"(clean {against_clean:.3f}, shift {against_shift:.3f})."
    )


if __name__ == "__main__":
    raise SystemExit(main())
