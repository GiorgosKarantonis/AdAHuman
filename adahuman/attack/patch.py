"""Person-suppression adversarial patch, after Thys et al. (CVPRW 2019).

Independently implemented from the published description. No employer or client
code, thresholds, or learned parameters are involved.

Two design choices are worth stating because they bound what the result means:

**The patch is applied in native image space, before the detector's own
resize.** A patch applied after the 320x320 downsample would be optimized
against pixels that no camera produces. Applying it beforehand means the patch
passes through the same interpolation a real one would.

**Every placement is drawn under expectation over transformation.** Scale,
rotation, brightness, contrast, perspective, and sensor noise are randomized at
every optimization step and again at evaluation. A patch that only works at one
pose is an artifact of the optimizer, not a finding, and EOT is what separates
the two. It also makes the reported attack strictly weaker than a pose-locked
attack would look -- which is the honest direction to err in.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.v2 import functional as TF


@dataclasses.dataclass
class EOTParams:
    """Ranges for expectation-over-transformation sampling."""

    scale_range: tuple[float, float]
    rotation_deg_range: tuple[float, float]
    brightness_range: tuple[float, float]
    contrast_range: tuple[float, float]
    perspective_distortion: float
    noise_std: float

    @classmethod
    def from_protocol(cls, protocol: Any) -> "EOTParams":
        eot = protocol.get("attack.eot")
        return cls(
            scale_range=tuple(eot["scale_range"]),
            rotation_deg_range=tuple(eot["rotation_deg_range"]),
            brightness_range=tuple(eot["brightness_range"]),
            contrast_range=tuple(eot["contrast_range"]),
            perspective_distortion=float(eot["perspective_distortion"]),
            noise_std=float(eot["noise_std"]),
        )


class AdversarialPatch(nn.Module):
    """A single patch image, optimized in unconstrained space.

    The patch is stored as unbounded logits and squashed through a sigmoid to
    produce pixels in [0, 1]. This keeps it in the valid image range by
    construction rather than by clamping after each step, which would zero the
    gradient wherever the patch is saturated.
    """

    def __init__(self, size_px: int, seed: int | None = None):
        super().__init__()
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        # Small initial logits keep the starting patch mid-grey rather than
        # saturated, so early gradients are informative.
        logits = torch.randn(3, size_px, size_px, generator=generator) * 0.1
        self.logits = nn.Parameter(logits)

    @property
    def pixels(self) -> torch.Tensor:
        """The patch as an image tensor in [0, 1], shape (3, S, S)."""
        return torch.sigmoid(self.logits)

    def total_variation(self) -> torch.Tensor:
        """Mean absolute difference between neighbouring pixels.

        Added to the loss to discourage per-pixel noise that a printer could
        not reproduce and a camera would not resolve.
        """
        pixels = self.pixels
        dx = (pixels[:, :, 1:] - pixels[:, :, :-1]).abs().mean()
        dy = (pixels[:, 1:, :] - pixels[:, :-1, :]).abs().mean()
        return dx + dy


def _sample(low: float, high: float, generator: torch.Generator | None) -> float:
    return float(torch.empty(1).uniform_(low, high, generator=generator))


def transform_patch(
    patch: torch.Tensor,
    side: int,
    eot: EOTParams,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize and randomly transform one patch instance.

    Args:
        patch: The patch in [0, 1], shape (3, S, S).
        side: Target side length in image pixels.
        eot: Sampling ranges.
        generator: RNG for reproducible transform draws.

    Returns:
        ``(pixels, alpha)`` both at ``(., side, side)``. ``alpha`` is 1 inside
        the patch and 0 where a rotation or perspective warp has moved the
        patch off its own canvas, so the caller composites only real patch
        pixels rather than the black fill those warps introduce.
    """
    resized = F.interpolate(
        patch.unsqueeze(0), size=(side, side), mode="bilinear", align_corners=False
    )
    alpha = torch.ones(1, 1, side, side, device=patch.device, dtype=patch.dtype)

    angle = _sample(*eot.rotation_deg_range, generator=generator)
    if angle:
        resized = TF.rotate(resized, angle, interpolation=TF.InterpolationMode.BILINEAR)
        alpha = TF.rotate(alpha, angle, interpolation=TF.InterpolationMode.BILINEAR)

    if eot.perspective_distortion > 0:
        start, end = _perspective_points(side, eot.perspective_distortion, generator)
        resized = TF.perspective(
            resized, start, end, interpolation=TF.InterpolationMode.BILINEAR
        )
        alpha = TF.perspective(
            alpha, start, end, interpolation=TF.InterpolationMode.BILINEAR
        )

    # Photometric jitter stands in for exposure and print/display response.
    brightness = _sample(*eot.brightness_range, generator=generator)
    contrast = _sample(*eot.contrast_range, generator=generator)
    resized = resized * brightness
    resized = (resized - resized.mean()) * contrast + resized.mean()

    if eot.noise_std > 0:
        # Drawn on CPU, then moved. Two reasons, and the second is the one that
        # matters: a torch.Generator is bound to a device type, so handing a CPU
        # generator to a CUDA allocation raises outright; and a CUDA generator is
        # a different RNG from a CPU one, so generating device-side would make a
        # GPU training run unreproducible on a CPU-only machine. The patch is
        # optimized on a Colab GPU while every other stage runs on CPU, so the
        # transform draws are kept device-independent. The tensor is small and
        # the transfer is not measurable against the forward pass.
        noise = torch.randn(resized.shape, generator=generator).to(
            device=resized.device, dtype=resized.dtype
        )
        resized = resized + noise * eot.noise_std

    return resized.squeeze(0).clamp(0, 1), alpha.squeeze(0).clamp(0, 1)


def _perspective_points(
    side: int, distortion: float, generator: torch.Generator | None
) -> tuple[list[list[int]], list[list[int]]]:
    """Corner correspondences for a random perspective warp."""
    limit = int(distortion * side)
    corners = [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]]
    warped = [
        [
            corner[0] + int(_sample(-limit, limit, generator)),
            corner[1] + int(_sample(-limit, limit, generator)),
        ]
        for corner in corners
    ]
    return corners, warped


def apply_patch(
    image: torch.Tensor,
    boxes: torch.Tensor,
    patch: torch.Tensor,
    eot: EOTParams,
    scale_of_bbox: float,
    min_bbox_area: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Composite transformed patch instances onto every eligible person box.

    The patch is centred on each box and sized so its area is
    ``scale_of_bbox`` of the box area, which keeps the attack's visual
    footprint proportional to the target rather than fixed in pixels.

    Callers should pass non-crowd boxes only, since the returned indices are
    positions in ``boxes`` and downstream grouping indexes the non-crowd
    ground truth.

    Returns:
        ``(patched_image, applied_indices)`` where ``applied_indices`` lists
        the boxes that actually received a patch -- boxes below the size
        threshold, or whose patch fell entirely outside the frame, are skipped.
        Compositing is a differentiable alpha blend, so gradients reach the
        patch through this function.
    """
    _, height, width = image.shape
    result = image
    applied: list[int] = []

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if box_area < min_bbox_area:
            continue

        jitter = _sample(*eot.scale_range, generator=generator)
        side = int(math.sqrt(box_area * scale_of_bbox) * jitter)
        if side < 8:
            # Below this the patch is a smudge; skip rather than report a
            # placement that carries no signal.
            continue

        pixels, alpha = transform_patch(patch, side, eot, generator)

        centre_x = int((x1 + x2) / 2)
        centre_y = int((y1 + y2) / 2)
        left = centre_x - side // 2
        top = centre_y - side // 2

        # Clip against the image border, cropping the patch to match.
        src_left = max(0, -left)
        src_top = max(0, -top)
        dst_left = max(0, left)
        dst_top = max(0, top)
        dst_right = min(width, left + side)
        dst_bottom = min(height, top + side)
        if dst_right <= dst_left or dst_bottom <= dst_top:
            continue

        visible_w = dst_right - dst_left
        visible_h = dst_bottom - dst_top
        pixels = pixels[:, src_top : src_top + visible_h, src_left : src_left + visible_w]
        alpha = alpha[:, src_top : src_top + visible_h, src_left : src_left + visible_w]

        canvas = torch.zeros_like(result)
        mask = torch.zeros(1, height, width, device=image.device, dtype=image.dtype)
        canvas[:, dst_top:dst_bottom, dst_left:dst_right] = pixels
        mask[:, dst_top:dst_bottom, dst_left:dst_right] = alpha

        result = result * (1 - mask) + canvas * mask
        applied.append(index)

    return result, applied


def person_scores(model: nn.Module, images: Sequence[torch.Tensor], category_id: int = 1):
    """Pre-NMS person-class scores for every anchor.

    The detector's ``forward`` returns post-NMS boxes, which is not a
    differentiable path to the patch. This reproduces the transform, backbone,
    and head so the attack can optimize the class scores directly -- the same
    quantity NMS later thresholds.

    Returns:
        ``(B, A)`` person probabilities, one per anchor.
    """
    from collections import OrderedDict

    image_list, _ = model.transform(list(images))
    features = model.backbone(image_list.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    head_outputs = model.head(list(features.values()))
    logits = head_outputs["cls_logits"]  # (B, A, num_classes)
    return logits.softmax(dim=-1)[..., category_id]


def suppression_loss(scores: torch.Tensor, top_k: int = 1) -> torch.Tensor:
    """Loss driving person detections below the decision threshold.

    Uses the mean of the ``top_k`` highest person scores per image. With
    ``top_k=1`` this is the max-objectness objective of Thys et al.: suppress
    the single most confident person. Larger ``top_k`` spreads pressure across
    several anchors, which matters in crowded frames where suppressing one
    detection leaves the rest untouched.
    """
    k = min(top_k, scores.shape[1])
    top = scores.topk(k, dim=1).values
    return top.mean()
