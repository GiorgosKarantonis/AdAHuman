"""One forward pass, both outputs.

Detections and monitor features come from the same forward pass rather than
two. Beyond the obvious cost saving, it guarantees that a reported detection
result and the monitor score for the same image describe the same evaluation of
the same pixels -- there is no second pass whose random transforms could have
differed.
"""

from __future__ import annotations

from typing import Sequence

import torch

from adahuman.models.detector import FeatureTap, person_detections


def detect_and_featurize(
    model,
    images: Sequence[torch.Tensor],
    category_id: int,
    hook_path: str,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Run the detector and tap its backbone features.

    Returns:
        ``(detections, features)`` where ``detections`` is one
        ``(boxes, scores)`` pair per image, filtered to ``category_id``, and
        ``features`` is ``(B, D)`` pooled backbone features.
    """
    with torch.no_grad(), FeatureTap(model, hook_path) as tap:
        outputs = model(list(images))
        features = tap.pooled().cpu()

    detections = [
        (detection.boxes, detection.scores)
        for detection in person_detections(outputs, category_id)
    ]
    return detections, features
