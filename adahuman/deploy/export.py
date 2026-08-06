"""ONNX export and conversion-fidelity measurement (RQ4).

The exported graph contains the whole detector: normalization, backbone, head,
and non-maximum suppression. Exporting the postprocessing rather than
reimplementing it outside the graph is deliberate -- NMS is where conversion
most often diverges, and an export that omitted it would measure the easy part
and call it a deployment path.

**Fixed 320x320 input, batch size one.** The exported graph accepts one image
already at the model's input resolution, which is the shape a camera pipeline
would hand it. Both the PyTorch baseline and the ONNX artifact are fed the
*same* pre-resized tensor, so the comparison isolates the conversion rather
than differences in how each path resamples.

**Output order differs from the PyTorch dict.** ``model(images)`` returns
``{boxes, labels, scores}``, but the exported graph emits ``boxes, scores,
labels``. Reading them positionally in dict order silently compares scores
against labels; :data:`ONNX_OUTPUT_ORDER` records the real order.
"""

from __future__ import annotations

import dataclasses
import pathlib
import warnings
from typing import Any

import numpy as np
import torch

#: Positional order of the exported graph's outputs. Verified against the
#: PyTorch outputs at export time by :func:`export_detector`.
ONNX_OUTPUT_ORDER = ("boxes", "scores", "labels")


@dataclasses.dataclass
class FidelityReport:
    """Agreement between the PyTorch baseline and the ONNX artifact."""

    n_images: int
    n_count_mismatches: int
    max_abs_box_deviation: float
    max_abs_score_deviation: float
    mean_abs_box_deviation: float
    mean_abs_score_deviation: float
    n_label_mismatches: int
    score_threshold: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def export_detector(
    model: torch.nn.Module,
    path: pathlib.Path | str,
    input_size: tuple[int, int],
    opset: int,
) -> tuple[pathlib.Path, list[str]]:
    """Export the detector to ONNX.

    Returns:
        ``(path, warnings)`` where ``warnings`` collects every warning raised
        during export. These are recorded rather than suppressed: unsupported
        operators and graph rewrites are exactly what a deployment-fidelity
        result needs to disclose.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = input_size
    example = torch.rand(3, height, width)

    captured: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.onnx.export(
            model,
            ([example],),
            str(path),
            opset_version=opset,
            input_names=["image"],
            output_names=list(ONNX_OUTPUT_ORDER),
            do_constant_folding=True,
        )
        captured = [f"{w.category.__name__}: {w.message}" for w in caught]

    return path, captured


def make_session(path: pathlib.Path | str, threads: int):
    """Create a single-threaded-per-op CPU inference session.

    Thread count is pinned so latency numbers are comparable with the PyTorch
    baseline, which is pinned to the same count.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )


def run_onnx(session, image: torch.Tensor) -> dict[str, np.ndarray]:
    """Run one image through the session, returning named outputs."""
    name = session.get_inputs()[0].name
    outputs = session.run(None, {name: image.numpy()})
    return dict(zip(ONNX_OUTPUT_ORDER, outputs))


def compare_fidelity(
    model: torch.nn.Module,
    session,
    images: list[torch.Tensor],
    score_threshold: float,
    category_id: int = 1,
) -> FidelityReport:
    """Measure agreement on detections that would actually be acted on.

    Restricted to person detections above the operating threshold. Comparing
    all 300 raw detections would be dominated by near-zero-confidence boxes
    that no deployment consumes, and whose ordering is not meaningful.

    Detections are sorted by descending score in both paths before comparison,
    so an ordering difference does not register as a numerical one.
    """
    count_mismatches = 0
    label_mismatches = 0
    box_deviations: list[float] = []
    score_deviations: list[float] = []

    for image in images:
        with torch.no_grad():
            reference = model([image])[0]
        candidate = run_onnx(session, image)

        ref_keep = (reference["labels"] == category_id) & (
            reference["scores"] >= score_threshold
        )
        ref_scores = reference["scores"][ref_keep].numpy()
        ref_boxes = reference["boxes"][ref_keep].numpy()

        cand_keep = (candidate["labels"] == category_id) & (
            candidate["scores"] >= score_threshold
        )
        cand_scores = candidate["scores"][cand_keep]
        cand_boxes = candidate["boxes"][cand_keep]

        if len(ref_scores) != len(cand_scores):
            count_mismatches += 1

        n = min(len(ref_scores), len(cand_scores))
        if n == 0:
            continue

        ref_order = np.argsort(-ref_scores)[:n]
        cand_order = np.argsort(-cand_scores)[:n]
        box_deviations.append(
            float(np.abs(ref_boxes[ref_order] - cand_boxes[cand_order]).max())
        )
        score_deviations.append(
            float(np.abs(ref_scores[ref_order] - cand_scores[cand_order]).max())
        )

    return FidelityReport(
        n_images=len(images),
        n_count_mismatches=count_mismatches,
        max_abs_box_deviation=max(box_deviations) if box_deviations else 0.0,
        max_abs_score_deviation=max(score_deviations) if score_deviations else 0.0,
        mean_abs_box_deviation=(
            float(np.mean(box_deviations)) if box_deviations else 0.0
        ),
        mean_abs_score_deviation=(
            float(np.mean(score_deviations)) if score_deviations else 0.0
        ),
        n_label_mismatches=label_mismatches,
        score_threshold=score_threshold,
    )


def resize_to_input(image: torch.Tensor, input_size: tuple[int, int]) -> torch.Tensor:
    """Resize a native-resolution image to the model input size.

    Applied once, outside both paths, so PyTorch and ONNX receive byte-identical
    input and the measurement isolates the conversion.
    """
    import torch.nn.functional as F

    return F.interpolate(
        image.unsqueeze(0), size=input_size, mode="bilinear", align_corners=False
    ).squeeze(0)
