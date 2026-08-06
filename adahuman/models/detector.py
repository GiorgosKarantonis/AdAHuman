"""The frozen detector, and the feature tap the RQ3 monitor reads.

The artifact evaluates exactly one model: torchvision's
``ssdlite320_mobilenet_v3_large`` with public COCO weights. It is loaded here
and nowhere else, so that every stage provably uses the same network.

The monitor in RQ3 needs an image-level representation. Rather than train or
tune a feature extractor, it taps a tensor the detector already computes -- the
deepest backbone feature map, at output stride 32 -- and global-average-pools
it. That choice keeps the diagnostic cheap enough to be plausible at runtime
(it adds no forward pass) and keeps it free of any learned component of its
own, which matters because a learned detector would be a different claim.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import torch
import torchvision
from torch import nn

#: Dotted path, relative to the detection model, of the output-stride-32
#: backbone tensor. torchvision splits the SSDLite MobileNetV3 backbone into
#: ``features[0]`` (through stride 16, feeding the first detection head) and
#: ``features[1]`` (through stride 32). The latter is the deepest tensor before
#: the extra SSD layers begin.
FEATURE_HOOK_PATH = "backbone.features.1"


@dataclasses.dataclass
class Detection:
    """Post-NMS detections for one image, filtered to the person class."""

    boxes: torch.Tensor  # (N, 4) in xyxy, original image coordinates
    scores: torch.Tensor  # (N,)

    def above(self, threshold: float) -> "Detection":
        keep = self.scores >= threshold
        return Detection(self.boxes[keep], self.scores[keep])

    def __len__(self) -> int:
        return int(self.boxes.shape[0])


def load_detector(weights_enum: str | None = None) -> nn.Module:
    """Load the frozen detector in eval mode.

    Args:
        weights_enum: Dotted torchvision weights enum, e.g.
            ``"SSDLite320_MobileNet_V3_Large_Weights.COCO_V1"``. Defaults to
            that value. Passing anything else is a protocol change, not a
            configuration tweak.

    Returns:
        The model in ``eval()`` mode with gradients disabled on its parameters.
        Parameters stay frozen for every stage including patch optimization,
        where gradients flow to the patch and never to the network.
    """
    weights_enum = weights_enum or "SSDLite320_MobileNet_V3_Large_Weights.COCO_V1"
    enum_name, member = weights_enum.split(".")
    weights = getattr(getattr(torchvision.models.detection, enum_name), member)

    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=weights
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def resolve_module(model: nn.Module, dotted: str) -> nn.Module:
    """Resolve a dotted module path, tolerating numeric ``Sequential`` indices."""
    node: nn.Module = model
    for part in dotted.split("."):
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node


class FeatureTap:
    """Captures the backbone feature map during an ordinary detection forward.

    Used as a context manager so the hook is always removed, including on
    exceptions -- a leaked hook would silently pollute later stages.

    Example:
        >>> with FeatureTap(model) as tap:
        ...     detections = model(images)
        ...     features = tap.pooled()
    """

    def __init__(self, model: nn.Module, hook_path: str = FEATURE_HOOK_PATH):
        self.model = model
        self.hook_path = hook_path
        self._captured: torch.Tensor | None = None
        self._handle: Any = None

    def __enter__(self) -> "FeatureTap":
        module = resolve_module(self.model, self.hook_path)

        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            self._captured = output

        self._handle = module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @property
    def raw(self) -> torch.Tensor:
        """The captured (B, C, H, W) tensor from the most recent forward pass."""
        if self._captured is None:
            raise RuntimeError(
                "no feature tensor captured; run a forward pass inside the "
                "FeatureTap context before reading it."
            )
        return self._captured

    def pooled(self) -> torch.Tensor:
        """Global-average-pooled features, shape (B, C).

        Pooling over space is what makes this an *image-level* score. It
        discards where in the frame an anomaly occurs, which is a real
        limitation for localized patch attacks and is recorded as such in
        LIMITATIONS.md. Per-region monitoring is deliberately out of scope.
        """
        return self.raw.mean(dim=(2, 3))


def person_detections(
    outputs: Sequence[dict[str, torch.Tensor]], category_id: int = 1
) -> list[Detection]:
    """Filter raw torchvision detection outputs down to one class.

    torchvision returns COCO's 91-way category ids directly, so the person
    class needs no remapping.
    """
    results: list[Detection] = []
    for output in outputs:
        keep = output["labels"] == category_id
        results.append(
            Detection(boxes=output["boxes"][keep], scores=output["scores"][keep])
        )
    return results
