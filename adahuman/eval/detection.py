"""Person-detection metrics.

Two families of number are reported, because they answer different questions:

* **COCO mAP**, computed by ``pycocotools``, is the field-standard summary and
  is what makes the clean baseline comparable to published results for this
  checkpoint. It sweeps score and IoU thresholds.
* **Operating-point metrics** -- recall, precision, missed-detection rate --
  are computed at the single fixed threshold in the protocol. These are the
  numbers that describe what a deployed system would actually do, and they are
  what attack success is measured against. The matching is implemented here
  rather than extracted from ``pycocotools`` internals so that a reviewer can
  read exactly how a hit is defined.

Attack success is deliberately defined as the *drop in recall* under attack,
not as a raw post-attack score. A patch that suppresses detections in images
where the detector already failed has achieved nothing, and a difference metric
says so.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
from typing import Any, Sequence

import numpy as np
import torch


@dataclasses.dataclass
class OperatingPoint:
    """Counts and rates at one fixed (score, IoU) threshold."""

    true_positives: int
    false_positives: int
    false_negatives: int
    n_ground_truth: int
    n_detections: int
    n_images: int

    @property
    def recall(self) -> float:
        """Fraction of annotated persons that were detected."""
        if self.n_ground_truth == 0:
            return float("nan")
        return self.true_positives / self.n_ground_truth

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return float("nan")
        return self.true_positives / denominator

    @property
    def missed_detection_rate(self) -> float:
        """1 - recall. Reported separately because it is the quantity the
        person-suppression attack is trying to drive up."""
        return float("nan") if self.n_ground_truth == 0 else 1.0 - self.recall

    @property
    def false_positives_per_image(self) -> float:
        return self.false_positives / self.n_images if self.n_images else float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "n_ground_truth": self.n_ground_truth,
            "n_detections": self.n_detections,
            "n_images": self.n_images,
            "recall": self.recall,
            "precision": self.precision,
            "missed_detection_rate": self.missed_detection_rate,
            "false_positives_per_image": self.false_positives_per_image,
        }


def iou_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of xyxy boxes, shape ``(len(a), len(b))``."""
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.zeros((boxes_a.shape[0], boxes_b.shape[0]))

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (
        boxes_a[:, 3] - boxes_a[:, 1]
    ).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (
        boxes_b[:, 3] - boxes_b[:, 1]
    ).clamp(min=0)

    top_left = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    overlap = (bottom_right - top_left).clamp(min=0)
    intersection = overlap[..., 0] * overlap[..., 1]

    union = area_a[:, None] + area_b[None, :] - intersection
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def match_image(
    det_boxes: torch.Tensor,
    det_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_crowd: torch.Tensor,
    score_threshold: float,
    iou_threshold: float,
) -> tuple[int, int, int, set[int]]:
    """Greedily match detections to ground truth for one image.

    Detections are considered in descending score order; each takes the
    highest-IoU unmatched ground-truth box above ``iou_threshold``. This is the
    standard greedy assignment, and it is order-dependent by construction --
    which is why the score sort is explicit.

    A detection that matches no real box but overlaps a *crowd* region is
    discarded rather than counted as a false positive, following COCO
    semantics. Crowd boxes are never counted as recallable ground truth.

    Returns:
        ``(true_positives, false_positives, false_negatives, matched)`` where
        ``matched`` is the set of indices into the *non-crowd* ground-truth
        boxes that were detected. The caller needs those indices to split
        recall by group -- patched versus unpatched targets -- which is how
        attack success is separated from incidental degradation.
    """
    keep = det_scores >= score_threshold
    det_boxes = det_boxes[keep]
    det_scores = det_scores[keep]

    order = torch.argsort(det_scores, descending=True)
    det_boxes = det_boxes[order]

    is_crowd = gt_crowd.bool()
    real_gt = gt_boxes[~is_crowd]
    crowd_gt = gt_boxes[is_crowd]

    ious = iou_matrix(det_boxes, real_gt)
    crowd_ious = iou_matrix(det_boxes, crowd_gt)

    matched_gt: set[int] = set()
    true_positives = 0
    false_positives = 0

    for det_index in range(det_boxes.shape[0]):
        best_iou, best_gt = -1.0, -1
        for gt_index in range(real_gt.shape[0]):
            if gt_index in matched_gt:
                continue
            value = float(ious[det_index, gt_index])
            if value > best_iou:
                best_iou, best_gt = value, gt_index

        if best_gt >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_gt)
            true_positives += 1
            continue

        # Unmatched. Ignore it if it lands inside a crowd region.
        in_crowd = (
            crowd_gt.shape[0] > 0
            and float(crowd_ious[det_index].max()) >= iou_threshold
        )
        if not in_crowd:
            false_positives += 1

    false_negatives = int(real_gt.shape[0]) - true_positives
    return true_positives, false_positives, false_negatives, matched_gt


def operating_point(
    detections: Sequence[tuple[torch.Tensor, torch.Tensor]],
    targets: Sequence[dict[str, Any]],
    score_threshold: float,
    iou_threshold: float,
) -> OperatingPoint:
    """Aggregate :func:`match_image` over a pool."""
    totals = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "det": 0}

    for (boxes, scores), target in zip(detections, targets):
        gt_boxes = target["boxes"]
        gt_crowd = target["iscrowd"]
        tp, fp, fn, _ = match_image(
            boxes, scores, gt_boxes, gt_crowd, score_threshold, iou_threshold
        )
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["gt"] += int((~gt_crowd.bool()).sum())
        totals["det"] += int((scores >= score_threshold).sum())

    return OperatingPoint(
        true_positives=totals["tp"],
        false_positives=totals["fp"],
        false_negatives=totals["fn"],
        n_ground_truth=totals["gt"],
        n_detections=totals["det"],
        n_images=len(targets),
    )


def grouped_recall(
    detections: Sequence[tuple[torch.Tensor, torch.Tensor]],
    targets: Sequence[dict[str, Any]],
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, dict[str, Any]]:
    """Recall split by whether a target received a patch.

    This is the measurement that separates the attack's effect from everything
    else. Targets are partitioned into two groups:

    * **patched** -- boxes that actually received a patch instance. Recall here
      is what the attack drove down.
    * **unpatched** -- boxes in the same images that were too small to be
      patched. They are an internal control: the patch is local, so their
      recall should stay near the clean baseline. If it collapses too, the
      effect is not localized suppression and the result must be described
      differently.

    Requires ``target["patched_gt"]`` -- a set of indices into the non-crowd
    ground-truth boxes -- which the attack evaluation stage populates.
    """
    groups = {
        "patched": {"hit": 0, "total": 0},
        "unpatched": {"hit": 0, "total": 0},
    }

    for (boxes, scores), target in zip(detections, targets):
        patched: set[int] = target.get("patched_gt", set())
        _, _, _, matched = match_image(
            boxes,
            scores,
            target["boxes"],
            target["iscrowd"],
            score_threshold,
            iou_threshold,
        )
        n_real = int((~target["iscrowd"].bool()).sum())
        for index in range(n_real):
            key = "patched" if index in patched else "unpatched"
            groups[key]["total"] += 1
            groups[key]["hit"] += int(index in matched)

    result = {}
    for name, counts in groups.items():
        hit, total = counts["hit"], counts["total"]
        recall = hit / total if total else float("nan")
        low, high = wilson_interval(hit, total)
        result[name] = {
            "detected": hit,
            "n_targets": total,
            "recall": recall,
            "recall_ci95": [low, high],
        }
    return result


def coco_person_map(
    coco_gt: Any,
    detections: Sequence[tuple[torch.Tensor, torch.Tensor]],
    targets: Sequence[dict[str, Any]],
    category_id: int = 1,
) -> dict[str, float]:
    """COCO average precision for the person class over the given images.

    Returns the standard 12-entry COCOeval summary, keyed by name. Returns an
    empty dict when there are no detections at all, which happens under a
    fully successful suppression attack and must not crash the pipeline.
    """
    from pycocotools.cocoeval import COCOeval

    results = []
    for (boxes, scores), target in zip(detections, targets):
        for box, score in zip(boxes.tolist(), scores.tolist()):
            x1, y1, x2, y2 = box
            results.append(
                {
                    "image_id": target["image_id"],
                    "category_id": category_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )

    if not results:
        return {}

    image_ids = [target["image_id"] for target in targets]
    # pycocotools writes progress to stdout; suppressed so run output stays
    # readable. Nothing diagnostic is discarded -- failures still raise.
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.params.imgIds = image_ids
        evaluator.params.catIds = [category_id]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    names = [
        "AP@[.50:.95]", "AP@.50", "AP@.75",
        "AP_small", "AP_medium", "AP_large",
        "AR@1", "AR@10", "AR@100",
        "AR_small", "AR_medium", "AR_large",
    ]
    return {name: float(value) for name, value in zip(names, evaluator.stats)}


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Reported alongside recall so that differences between conditions are read
    against sampling error. The Wilson form is used rather than the normal
    approximation because it stays inside [0, 1] at the extreme rates a
    successful suppression attack produces.
    """
    if trials == 0:
        return (float("nan"), float("nan"))

    from scipy.stats import norm

    z = float(norm.ppf(1 - (1 - confidence) / 2))
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    margin = (
        z
        * np.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2))
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))
