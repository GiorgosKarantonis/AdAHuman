"""Ordinary (non-malicious) distribution shift.

The RQ3 monitor is only interesting if it separates *adversarial* inputs from
*benign but unusual* ones. A detector of "anything that isn't pristine COCO" is
trivial to build and useless in deployment, where weather, motion, exposure,
and codec artefacts are constant. These corruptions are the control condition
that makes the question non-trivial.

Severity constants follow Hendrycks & Dietterich's ImageNet-C at severity 3.
Three of the five (``gaussian_noise``, ``brightness``, ``jpeg_compression``)
use the published constants and are faithful reimplementations. Two are not:

* ``motion_blur`` -- the reference implementation calls ImageMagick via Wand.
  This is an independent Gaussian-weighted line-kernel convolution at a
  comparable radius. Visually similar, not bit-identical.
* ``fog`` -- built on an independent diamond-square plasma fractal rather than
  the reference's variant.

Both deviations are recorded here and in LIMITATIONS.md rather than presented
as reproductions of ImageNet-C. Nothing in the artifact compares its numbers
against published ImageNet-C results, so the deviation costs no comparability;
it would matter if such a comparison were ever made.
"""

from __future__ import annotations

import io
import math
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

#: ImageNet-C severity 3 constants, by corruption name.
SEVERITY_3 = {
    "gaussian_noise": 0.18,
    "motion_blur": (15, 8),  # (radius, sigma)
    "brightness": 0.3,
    "fog": (2.5, 1.7),  # (intensity, wibble decay)
    "jpeg_compression": 15,  # quality
}


def gaussian_noise(
    image: torch.Tensor, generator: torch.Generator, severity: int = 3
) -> torch.Tensor:
    """Additive white Gaussian noise. Stands in for low-light sensor noise."""
    std = SEVERITY_3["gaussian_noise"]
    noise = torch.randn(image.shape, generator=generator) * std
    return (image + noise).clamp(0, 1)


def motion_blur(
    image: torch.Tensor, generator: torch.Generator, severity: int = 3
) -> torch.Tensor:
    """Directional blur at a random angle. Stands in for camera or subject motion."""
    radius, sigma = SEVERITY_3["motion_blur"]
    size = radius * 2 + 1
    angle = float(torch.empty(1).uniform_(-45, 45, generator=generator))

    # A line through the kernel centre, Gaussian-weighted by distance along it.
    kernel = torch.zeros(size, size)
    radians = math.radians(angle)
    for offset in range(-radius, radius + 1):
        x = int(round(radius + offset * math.cos(radians)))
        y = int(round(radius + offset * math.sin(radians)))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] += math.exp(-(offset**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    channels = image.shape[0]
    weight = kernel.expand(channels, 1, size, size)
    padded = F.pad(image.unsqueeze(0), (radius,) * 4, mode="reflect")
    blurred = F.conv2d(padded, weight, groups=channels)
    return blurred.squeeze(0).clamp(0, 1)


def brightness(
    image: torch.Tensor, generator: torch.Generator, severity: int = 3
) -> torch.Tensor:
    """Raise the HSV value channel. Stands in for overexposure."""
    amount = SEVERITY_3["brightness"]
    hue, saturation, value = _rgb_to_hsv(image)
    return _hsv_to_rgb(hue, saturation, (value + amount).clamp(0, 1))


def fog(
    image: torch.Tensor, generator: torch.Generator, severity: int = 3
) -> torch.Tensor:
    """Overlay a plasma-fractal haze. Stands in for fog, smoke, or a dirty lens."""
    intensity, decay = SEVERITY_3["fog"]
    _, height, width = image.shape

    size = 1 << max(height, width).bit_length()
    haze = _plasma_fractal(size, decay, generator)[:height, :width]

    peak = float(image.max())
    hazed = image + intensity * haze.unsqueeze(0)
    # Rescale so the haze brightens rather than simply saturating the frame.
    return (hazed * peak / (peak + intensity)).clamp(0, 1)


def jpeg_compression(
    image: torch.Tensor, generator: torch.Generator, severity: int = 3
) -> torch.Tensor:
    """Round-trip through JPEG. Stands in for codec artefacts on a video feed."""
    quality = SEVERITY_3["jpeg_compression"]
    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as handle:
        decoded = np.array(handle.convert("RGB"))

    return torch.from_numpy(decoded).permute(2, 0, 1).float().div(255.0)


CORRUPTIONS: dict[str, Callable[..., torch.Tensor]] = {
    "gaussian_noise": gaussian_noise,
    "motion_blur": motion_blur,
    "brightness": brightness,
    "fog": fog,
    "jpeg_compression": jpeg_compression,
}


def apply_corruption(
    image: torch.Tensor,
    name: str,
    generator: torch.Generator,
    severity: int = 3,
) -> torch.Tensor:
    """Apply one named corruption to a ``(3, H, W)`` float image in [0, 1]."""
    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption {name!r}; known: {sorted(CORRUPTIONS)}")
    if severity != 3:
        raise ValueError(
            f"only severity 3 is defined; the protocol fixes it. Got {severity}."
        )
    return CORRUPTIONS[name](image, generator, severity)


def _plasma_fractal(
    size: int, decay: float, generator: torch.Generator
) -> torch.Tensor:
    """Diamond-square plasma noise on a ``size x size`` grid, normalized to [0, 1].

    Independent implementation of the standard algorithm: repeatedly fill
    square then diamond midpoints with the mean of their neighbours plus
    decaying random displacement.
    """
    grid = torch.zeros(size + 1, size + 1)
    step = size
    amplitude = 1.0

    def jitter(shape: tuple[int, ...]) -> torch.Tensor:
        return (torch.rand(shape, generator=generator) - 0.5) * amplitude

    while step > 1:
        half = step // 2

        # Square step: centre of each cell from its four corners.
        corners = grid[0:size:step, 0:size:step]
        right = grid[step : size + 1 : step, 0:size:step]
        down = grid[0:size:step, step : size + 1 : step]
        diagonal = grid[step : size + 1 : step, step : size + 1 : step]
        centres = (corners + right + down + diagonal) / 4
        grid[half:size:step, half:size:step] = centres + jitter(centres.shape)

        # Diamond step: edge midpoints from their neighbours. Averaging the two
        # axis-aligned neighbours is a simplification at the grid border, where
        # a wrapped or clipped fourth neighbour would otherwise be needed.
        horizontal = (grid[0:size:step, 0:size:step] + grid[step : size + 1 : step, 0:size:step]) / 2
        grid[half:size:step, 0:size:step] = horizontal + jitter(horizontal.shape)

        vertical = (grid[0:size:step, 0:size:step] + grid[0:size:step, step : size + 1 : step]) / 2
        grid[0:size:step, half:size:step] = vertical + jitter(vertical.shape)

        step = half
        amplitude /= decay

    grid = grid[:size, :size]
    grid = grid - grid.min()
    peak = float(grid.max())
    return grid / peak if peak > 0 else grid


def _rgb_to_hsv(
    image: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RGB in [0, 1] to (hue in [0, 1), saturation, value)."""
    red, green, blue = image[0], image[1], image[2]
    value, _ = image.max(dim=0)
    minimum, _ = image.min(dim=0)
    chroma = value - minimum

    saturation = torch.where(value > 0, chroma / value.clamp(min=1e-8), torch.zeros_like(value))

    hue = torch.zeros_like(value)
    safe = chroma.clamp(min=1e-8)
    hue = torch.where(value == red, ((green - blue) / safe) % 6, hue)
    hue = torch.where(value == green, ((blue - red) / safe) + 2, hue)
    hue = torch.where(value == blue, ((red - green) / safe) + 4, hue)
    hue = torch.where(chroma <= 1e-8, torch.zeros_like(hue), hue / 6)
    return hue, saturation, value


def _hsv_to_rgb(
    hue: torch.Tensor, saturation: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Inverse of :func:`_rgb_to_hsv`."""
    sector = (hue * 6).floor()
    fraction = hue * 6 - sector

    p = value * (1 - saturation)
    q = value * (1 - saturation * fraction)
    t = value * (1 - saturation * (1 - fraction))

    sector = sector.long() % 6
    options = torch.stack(
        [
            torch.stack([value, t, p]),
            torch.stack([q, value, p]),
            torch.stack([p, value, t]),
            torch.stack([p, q, value]),
            torch.stack([t, p, value]),
            torch.stack([value, p, q]),
        ]
    )
    index = sector.unsqueeze(0).unsqueeze(0).expand(1, 3, *hue.shape)
    return options.gather(0, index).squeeze(0).clamp(0, 1)
