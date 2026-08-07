#!/usr/bin/env python
"""Fix pool membership and write the manifests.

This is the point of no return for the held-out set. After this script runs,
``configs/manifests/eval_untouched.json`` names the 500 images against which the
final numbers will be reported, and every later stage loads that file rather
than reselecting. Changing it afterwards would mean the reported metrics are no
longer out-of-sample, so the manifests are committed and their digests recorded.

Usage:
    scripts/02_freeze_pools.py            # report the selection
    scripts/02_freeze_pools.py --write    # write manifests and stamp frozen_on
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from adahuman.config import DEFAULT_PROTOCOL, load_protocol  # noqa: E402
from adahuman.data.pools import (  # noqa: E402
    NEGATIVE_POOL,
    PERSON_POOL_ORDER,
    build_pools,
    manifest_for,
    partition_image_ids,
    write_manifest,
)
from adahuman.utils.hashing import short  # noqa: E402
from adahuman.utils.run_log import RunLog  # noqa: E402
from adahuman.utils.seed import seed_everything  # noqa: E402

STAGE = "freeze_pools"
ALL_POOLS = PERSON_POOL_ORDER + (NEGATIVE_POOL,)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the manifests")
    parser.add_argument("--protocol", type=pathlib.Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    protocol = load_protocol(STAGE, args.protocol)
    seed_everything(protocol.get("seed"))
    log = RunLog(STAGE, args.protocol)
    log.input("annotations", protocol.get("data.annotations"))

    annotations_path = pathlib.Path(protocol.get("data.annotations"))
    with annotations_path.open() as handle:
        annotations = json.load(handle)

    min_area = protocol.get("data.person_min_bbox_area")
    person_bearing, negative = partition_image_ids(annotations, min_area)

    total = len(annotations["images"])
    needed = sum(protocol.get(f"data.pools.{p}.size") for p in PERSON_POOL_ORDER)

    print(f"source            {protocol.get('data.source')}  ({total} images)")
    print(f"min person bbox   {min_area} px^2")
    print(f"person-bearing    {len(person_bearing)} eligible, {needed} needed")
    print(f"person-free       {len(negative)} eligible")
    print()

    pools = build_pools(protocol, ALL_POOLS)

    headroom = len(person_bearing) - needed
    if headroom < 0:
        print(f"ERROR: short by {-headroom} person-bearing images")
        return 1

    summary = {}
    for name in ALL_POOLS:
        image_ids = pools[name]
        manifest = manifest_for(protocol, name, image_ids)
        summary[name] = {
            "size": len(image_ids),
            "sha256": manifest["sha256"],
            "first_id": image_ids[0],
            "last_id": image_ids[-1],
        }
        print(
            f"  {name:15s} n={len(image_ids):4d}  "
            f"sha256={short(manifest['sha256'])}  "
            f"ids {image_ids[0]}..{image_ids[-1]}"
        )

    # Disjointness is already asserted inside build_pools; restated here so the
    # run log carries positive evidence of the check rather than its absence.
    everything = [i for ids in pools.values() for i in ids]
    assert len(everything) == len(set(everything)), "pools overlap"
    print(f"\n  disjoint: yes ({len(everything)} distinct images across 4 pools)")
    print(f"  headroom: {headroom} eligible person-bearing images unused")

    log.set("pools", summary)
    log.set("eligible", {"person_bearing": len(person_bearing), "negative": len(negative)})

    if args.write:
        for name in ALL_POOLS:
            path = write_manifest(protocol, name, pools[name])
            log.output(f"manifest_{name}", path)
        today = _dt.date.today().isoformat()
        _stamp_frozen_on(args.protocol, today)
        # Report the value the protocol actually carries. _stamp_frozen_on is a
        # no-op once the field is set, so printing `today` unconditionally would
        # claim a re-stamp that did not happen -- on a re-run it would assert the
        # pools were frozen today when they were frozen days earlier.
        stamped = re.search(
            r"^frozen_on: (\S+)", args.protocol.read_text(), re.M
        )
        print(
            f"\nwrote {len(ALL_POOLS)} manifests; "
            f"frozen_on = {stamped.group(1) if stamped else 'unknown'}"
        )
    else:
        print("\n(dry run; pass --write to persist)")

    path = log.write()
    print(f"run log: {path.relative_to(pathlib.Path.cwd())}")
    return 0


def _stamp_frozen_on(path: pathlib.Path, today: str) -> None:
    text = path.read_text()
    if "frozen_on: PENDING" not in text:
        return  # already stamped; freezing is not repeated
    path.write_text(text.replace("frozen_on: PENDING", f"frozen_on: {today}", 1))


if __name__ == "__main__":
    raise SystemExit(main())
