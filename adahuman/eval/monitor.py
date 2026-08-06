"""Detection metrics for the runtime monitor.

Reported as separation between a *negative* condition (clean, in-distribution)
and a *positive* one (adversarial, or ordinarily shifted). Two framings matter
and are reported separately:

* **clean vs adversarial** -- can the monitor see the attack at all?
* **ordinary shift vs adversarial** -- can it tell the attack apart from
  weather, motion, and codec noise? This is the harder and more operationally
  meaningful question. A monitor that scores highly on the first and at chance
  on the second is a novelty detector, not an attack detector, and would
  produce an alarm on every foggy morning.

Threshold-free ranking metrics (AUROC, average precision) are reported
alongside the rate at the single frozen operating threshold, because a
deployed monitor runs at one threshold and its false-positive rate there is
what determines whether operators keep it switched on.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def separation(
    negative_scores: np.ndarray,
    positive_scores: np.ndarray,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Summarize how well scores separate two conditions.

    Args:
        negative_scores: Scores for the condition that should *not* alarm.
        positive_scores: Scores for the condition that should alarm.
        threshold: Frozen operating threshold, if one has been selected.

    Returns:
        AUROC, average precision, the equal-error rate, and -- when a threshold
        is supplied -- the true- and false-positive rates at it.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

    negative = np.asarray(negative_scores, dtype=np.float64)
    positive = np.asarray(positive_scores, dtype=np.float64)
    if negative.size == 0 or positive.size == 0:
        return {"error": "one condition is empty", "n_negative": int(negative.size),
                "n_positive": int(positive.size)}

    labels = np.concatenate([np.zeros(negative.size), np.ones(positive.size)])
    scores = np.concatenate([negative, positive])

    fpr, tpr, _ = roc_curve(labels, scores)
    equal_error = float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])

    result: dict[str, Any] = {
        "n_negative": int(negative.size),
        "n_positive": int(positive.size),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "equal_error_rate": equal_error,
        "negative_median": float(np.median(negative)),
        "positive_median": float(np.median(positive)),
        "tpr_at_fpr_5pct": float(np.interp(0.05, fpr, tpr)),
        "tpr_at_fpr_1pct": float(np.interp(0.01, fpr, tpr)),
    }

    if threshold is not None:
        result["threshold"] = float(threshold)
        result["tpr_at_threshold"] = float((positive >= threshold).mean())
        result["fpr_at_threshold"] = float((negative >= threshold).mean())
    return result


def threshold_at_fpr(clean_scores: np.ndarray, target_fpr: float) -> float:
    """Pick the score threshold giving ``target_fpr`` on clean data.

    Selected on the reference pool alone, so the operating point is fixed
    without reference to any attacked or held-out input. Uses the empirical
    quantile: the threshold above which ``target_fpr`` of clean scores fall.
    """
    if not 0 < target_fpr < 1:
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
    scores = np.asarray(clean_scores, dtype=np.float64)
    return float(np.quantile(scores, 1 - target_fpr))
