#!/usr/bin/env python
"""RQ3, part one: fit the feature-distance monitor and freeze its threshold.

Reads the reference pool only. The mean and covariance describe clean,
in-distribution features; the operating threshold is the score below which
95% of those clean features fall. Nothing adversarial, shifted, or held-out
enters this stage, so the operating point is fixed without any knowledge of
what it will later be asked to detect.

Runs before the patch exists. That ordering is deliberate: a threshold chosen
after seeing attacked scores would be a threshold fitted to the answer.

Usage:
    scripts/06_fit_monitor.py
    scripts/06_fit_monitor.py --write-threshold
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
from adahuman.eval.inference import detect_and_featurize  # noqa: E402
from adahuman.eval.monitor import threshold_at_fpr  # noqa: E402
from adahuman.models.detector import load_detector  # noqa: E402
from adahuman.monitor.mahalanobis import FeatureDistanceMonitor  # noqa: E402
from adahuman.utils.run_log import RunLog  # noqa: E402
from adahuman.utils.seed import seed_everything  # noqa: E402

STAGE = "fit_monitor"
ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

#: Clean-data false-positive rate the frozen threshold targets. Matches
#: `monitor.threshold_rule` in the protocol.
TARGET_FPR = 0.05


def extract_features(model, protocol, coco, pool: str, workers: int) -> torch.Tensor:
    dataset = PoolDataset(protocol, pool, coco)
    loader = DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=workers, collate_fn=collate
    )
    hook_path = protocol.get("monitor.feature_hook")
    category_id = protocol.get("task.category_id")

    chunks = []
    for images, _ in loader:
        _, features = detect_and_featurize(model, images, category_id, hook_path)
        chunks.append(features)
    return torch.cat(chunks, dim=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=pathlib.Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--write-threshold", action="store_true")
    args = parser.parse_args()

    protocol = load_protocol(STAGE, args.protocol)
    seed_everything(protocol.get("seed"))
    log = RunLog(STAGE, args.protocol)
    log.pools_read("reference")

    import contextlib
    import io

    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(protocol.get("data.annotations"))

    model = load_detector(protocol.get("model.weights_enum"))
    torch.set_num_threads(protocol.get("deploy.intra_op_threads", 4))

    print("extracting clean reference features")
    features = extract_features(model, protocol, coco, "reference", args.workers)
    print(f"  {tuple(features.shape)}")

    expected_dim = protocol.get("monitor.feature_dim")
    if features.shape[1] != expected_dim:
        raise SystemExit(
            f"feature dimension {features.shape[1]} does not match the frozen "
            f"protocol value {expected_dim}."
        )

    # Split the reference pool: estimate the covariance on one part, calibrate
    # the threshold on the other. Calibrating in-sample reports the target rate
    # by construction and hides whether the score generalizes at all -- an
    # earlier version of this stage did exactly that and understated the true
    # false-positive rate by more than a factor of ten.
    calibration_fraction = protocol.get("monitor.calibration_fraction")
    n_components = protocol.get("monitor.pca_components")

    order = np.random.default_rng(protocol.get("seed")).permutation(len(features))
    n_calibration = int(round(len(features) * calibration_fraction))
    fit_index = order[n_calibration:]
    calibration_index = order[:n_calibration]

    fit_features = features[fit_index]
    calibration_features = features[calibration_index]

    monitor = FeatureDistanceMonitor.fit(fit_features, n_components=n_components)
    calibration_scores = monitor.score(calibration_features)
    monitor.threshold = threshold_at_fpr(calibration_scores, TARGET_FPR)

    fit_scores = monitor.score(fit_features)
    in_sample_fpr = float((fit_scores >= monitor.threshold).mean())
    calibration_fpr = float((calibration_scores >= monitor.threshold).mean())

    print(f"\n  PCA components     {monitor.n_components} "
          f"({monitor.explained_variance_ratio:.1%} of variance)")
    print(f"  shrinkage          {monitor.shrinkage:.4f}")
    print(f"  covariance fit on  {monitor.n_fit} clean reference images")
    print(f"  calibrated on      {len(calibration_features)} held-out reference images")
    print(f"  score median       {np.median(fit_scores):.2f}")
    print(f"  threshold @{TARGET_FPR:.0%} FPR  {monitor.threshold:.2f}")
    print(f"\n  FPR on fit split          {in_sample_fpr:.4f}  (in-sample)")
    print(f"  FPR on calibration split  {calibration_fpr:.4f}  "
          f"(target {TARGET_FPR}, by construction)")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    monitor_path = monitor.save(ARTIFACTS / "monitor_v1.npz")
    features_path = ARTIFACTS / "features_reference_clean.npz"
    np.savez_compressed(features_path, features=features.numpy())

    summary = {
        "monitor": monitor.summary(),
        "target_fpr": TARGET_FPR,
        "n_fit_split": int(len(fit_features)),
        "n_calibration_split": int(len(calibration_features)),
        "fpr_on_fit_split": in_sample_fpr,
        "fpr_on_calibration_split": calibration_fpr,
        "reference_score_quantiles": {
            str(q): float(np.quantile(fit_scores, q))
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
    }
    out_path = RESULTS / "rq3_monitor_fit.json"
    with out_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for key, path in (
        ("monitor", monitor_path),
        ("reference_features", features_path),
        ("results", out_path),
    ):
        log.output(key, path)
    log.set("monitor", summary)

    if args.write_threshold:
        _write_threshold(args.protocol, monitor.threshold)
        print(f"\nfroze monitor.threshold_value = {monitor.threshold:.6f}")

    print(f"\nresults: {out_path.relative_to(pathlib.Path.cwd())}")
    print(f"run log: {log.write().relative_to(pathlib.Path.cwd())}")
    return 0


def _write_threshold(path: pathlib.Path, threshold: float) -> None:
    text = path.read_text()
    if "threshold_value: PENDING" not in text:
        print("monitor.threshold_value already frozen; leaving it alone")
        return
    path.write_text(
        text.replace("threshold_value: PENDING", f"threshold_value: {threshold:.6f}", 1)
    )


if __name__ == "__main__":
    raise SystemExit(main())
