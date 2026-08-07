"""Frozen, disjoint evaluation pools over COCO val2017.

Pool membership is the foundation of every claim the artifact makes. It is
therefore derived deterministically from the protocol seed, written to explicit
manifests, and digested, so that a reviewer can confirm two things
independently: that the pools are disjoint, and that the held-out pool was
fixed before any result was produced.

Selection procedure, in full:

1. Take every image in ``instances_val2017.json``.
2. An image is *person-bearing* if it has at least one non-crowd person
   annotation whose bounding-box area is at least
   ``data.person_min_bbox_area``. It is a *negative* if it has no person
   annotation at all. Images with only small or crowd persons are eligible for
   neither pool, so the negative pool stays unambiguous.
3. Sort each group by image id ascending. This removes any dependence on JSON
   ordering.
4. Shuffle each group with ``random.Random(seed)``.
5. Slice sequentially: reference, then attack_dev, then eval_untouched from the
   person-bearing group; negative from the negative group.

Step 3 before step 4 is what makes the result reproducible from the seed alone.
"""

from __future__ import annotations

import json
import pathlib
import random
from typing import Any, Iterable

from adahuman.config import Protocol
from adahuman.utils.hashing import sha256_json

#: COCO category id for "person".
PERSON_CATEGORY_ID = 1

#: Order matters: pools are sliced sequentially from the shuffled pool, so
#: changing this order changes membership.
PERSON_POOL_ORDER = ("reference", "attack_dev", "eval_untouched")
NEGATIVE_POOL = "negative"


class PoolBuildError(RuntimeError):
    """Raised when the requested pools cannot be built from the annotations."""


def _load_annotations(path: pathlib.Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def partition_image_ids(
    annotations: dict[str, Any], min_bbox_area: float
) -> tuple[list[int], list[int]]:
    """Split all image ids into person-bearing and negative groups.

    Args:
        annotations: Parsed COCO instances JSON.
        min_bbox_area: Minimum person bounding-box area, in original-image
            square pixels, for an image to count as person-bearing.

    Returns:
        ``(person_bearing, negative)``, each sorted ascending by image id.
        Images holding only small or crowd persons appear in neither list.
    """
    has_any_person: set[int] = set()
    has_large_person: set[int] = set()

    for ann in annotations["annotations"]:
        if ann["category_id"] != PERSON_CATEGORY_ID:
            continue
        image_id = ann["image_id"]
        has_any_person.add(image_id)
        if ann.get("iscrowd", 0):
            continue
        _, _, width, height = ann["bbox"]
        if width * height >= min_bbox_area:
            has_large_person.add(image_id)

    all_ids = {image["id"] for image in annotations["images"]}
    person_bearing = sorted(has_large_person)
    negative = sorted(all_ids - has_any_person)
    return person_bearing, negative


def build_pools(protocol: Protocol, pool_names: Iterable[str]) -> dict[str, list[int]]:
    """Select image ids for the named pools.

    Slicing is always performed over the full :data:`PERSON_POOL_ORDER`, so a
    pool's membership never depends on which pools the caller happened to ask
    for.
    """
    requested = set(pool_names)
    seed = protocol.get("seed")
    annotations_path = pathlib.Path(protocol.get("data.annotations"))
    if not annotations_path.is_file():
        raise PoolBuildError(
            f"annotations not found at {annotations_path}. "
            f"Run scripts/00_fetch_coco.sh first."
        )

    annotations = _load_annotations(annotations_path)
    person_bearing, negative = partition_image_ids(
        annotations, protocol.get("data.person_min_bbox_area")
    )

    random.Random(seed).shuffle(person_bearing)
    random.Random(seed).shuffle(negative)

    pools: dict[str, list[int]] = {}
    cursor = 0
    for name in PERSON_POOL_ORDER:
        size = protocol.get(f"data.pools.{name}.size")
        chunk = person_bearing[cursor : cursor + size]
        if len(chunk) < size:
            raise PoolBuildError(
                f"pool {name!r} needs {size} person-bearing images but only "
                f"{len(chunk)} remain; {len(person_bearing)} were eligible in "
                f"total. Lower the pool sizes or the min_bbox_area threshold."
            )
        cursor += size
        if name in requested:
            pools[name] = sorted(chunk)

    if NEGATIVE_POOL in requested:
        size = protocol.get(f"data.pools.{NEGATIVE_POOL}.size")
        chunk = negative[:size]
        if len(chunk) < size:
            raise PoolBuildError(
                f"negative pool needs {size} person-free images but only "
                f"{len(chunk)} were eligible."
            )
        pools[NEGATIVE_POOL] = sorted(chunk)

    _assert_disjoint(pools)
    return pools


def _assert_disjoint(pools: dict[str, list[int]]) -> None:
    """Fail loudly if any image appears in two pools.

    Overlap between the development and held-out pools would silently
    invalidate every out-of-sample claim, so it is checked rather than assumed.
    """
    seen: dict[int, str] = {}
    for name, ids in pools.items():
        for image_id in ids:
            if image_id in seen:
                raise PoolBuildError(
                    f"image {image_id} appears in both {seen[image_id]!r} and "
                    f"{name!r}; pools must be disjoint."
                )
            seen[image_id] = name


def manifest_for(protocol: Protocol, name: str, image_ids: list[int]) -> dict[str, Any]:
    """Build the manifest record written to ``configs/manifests/``."""
    payload = {
        "pool": name,
        "size": len(image_ids),
        "seed": protocol.get("seed"),
        "source": protocol.get("data.source"),
        "person_min_bbox_area": protocol.get("data.person_min_bbox_area"),
        "purpose": protocol.get(f"data.pools.{name}.purpose").strip(),
        "image_ids": image_ids,
    }
    payload["sha256"] = sha256_json(payload["image_ids"])
    return payload


def write_manifest(
    protocol: Protocol, name: str, image_ids: list[int]
) -> pathlib.Path:
    """Write one pool manifest and return its path."""
    directory = pathlib.Path(protocol.get("data.manifest_dir"))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    with path.open("w") as handle:
        json.dump(manifest_for(protocol, name, image_ids), handle, indent=2)
        handle.write("\n")
    return path


def load_manifest(protocol: Protocol, name: str) -> list[int]:
    """Read a frozen pool manifest, verifying its digest.

    Every consumer loads pools through this function rather than recomputing
    them, so that a later change to the selection code cannot silently move the
    held-out pool out from under results already reported.
    """
    protocol.pool(name)  # entitlement check; raises if the stage may not read it.

    path = pathlib.Path(protocol.get("data.manifest_dir")) / f"{name}.json"
    if not path.is_file():
        raise PoolBuildError(
            f"manifest {path} missing. Run scripts/02_freeze_pools.py first."
        )
    with path.open() as handle:
        payload = json.load(handle)

    recorded = payload.get("sha256")
    actual = sha256_json(payload["image_ids"])
    if recorded != actual:
        raise PoolBuildError(
            f"manifest {path} is corrupt or was edited: recorded sha256 "
            f"{recorded} but contents digest to {actual}."
        )
    return payload["image_ids"]
