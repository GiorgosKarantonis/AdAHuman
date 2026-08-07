"""Pool-backed COCO dataset.

Images are loaded by the ids recorded in the frozen manifests, never by
directory scan, so that what a stage evaluates is exactly what the manifest
says it evaluates.

Images are returned at native resolution as float tensors in [0, 1]. The
detector's own ``GeneralizedRCNNTransform`` performs the resize to 320x320 and
normalization, which keeps this artifact's preprocessing identical to
torchvision's reference pipeline rather than a reimplementation of it. Patch
application therefore happens in native image space, which is also where a
physical patch would live.
"""

from __future__ import annotations

import pathlib
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from adahuman.config import Protocol
from adahuman.data.pools import PERSON_CATEGORY_ID, load_manifest


class PoolDataset(Dataset):
    """COCO images and person annotations for one frozen pool.

    Each item is ``(image, target)`` where ``image`` is a float tensor
    ``(3, H, W)`` in [0, 1] and ``target`` carries person boxes in xyxy plus
    bookkeeping needed for evaluation.
    """

    def __init__(self, protocol: Protocol, pool: str, coco: Any):
        self.protocol = protocol
        self.pool = pool
        self.coco = coco
        self.images_dir = pathlib.Path(protocol.get("data.images_dir"))
        self.category_id = protocol.get("task.category_id", PERSON_CATEGORY_ID)
        # Entitlement is checked inside load_manifest; a stage that may not read
        # this pool cannot construct the dataset at all.
        self.image_ids: list[int] = load_manifest(protocol, pool)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image_id = self.image_ids[index]
        info = self.coco.loadImgs([image_id])[0]
        path = self.images_dir / info["file_name"]

        with Image.open(path) as handle:
            image = handle.convert("RGB")
            # bytearray copies into writable memory; torch refuses to wrap the
            # read-only buffer PIL hands back.
            array = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            tensor = array.view(image.size[1], image.size[0], 3)
        image_tensor = tensor.permute(2, 0, 1).float().div(255.0)

        boxes, crowd = self._person_boxes(image_id)
        target = {
            "image_id": image_id,
            "file_name": info["file_name"],
            "boxes": boxes,
            "iscrowd": crowd,
            "height": info["height"],
            "width": info["width"],
        }
        return image_tensor, target

    def _person_boxes(self, image_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Person boxes in xyxy, with the crowd flag preserved.

        Crowd regions are kept rather than dropped: COCO semantics treat a
        detection inside a crowd region as neither a hit nor a false positive,
        and discarding them here would inflate the false-positive count.
        """
        annotation_ids = self.coco.getAnnIds(
            imgIds=[image_id], catIds=[self.category_id], iscrowd=None
        )
        boxes: list[list[float]] = []
        crowd: list[int] = []
        for ann in self.coco.loadAnns(annotation_ids):
            x, y, width, height = ann["bbox"]
            boxes.append([x, y, x + width, y + height])
            crowd.append(int(ann.get("iscrowd", 0)))

        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(
                (0,), dtype=torch.int64
            )
        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(crowd, dtype=torch.int64),
        )


def collate(
    batch: Sequence[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    """Keep images as a list; torchvision detectors batch them internally.

    Images in a pool have different native resolutions, so stacking them here
    would require a resize that the model is about to perform anyway.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
