#!/usr/bin/env python
"""RQ1b, part one: optimize the person-suppression patch.

Runs on the attack-development pool only. The held-out pool is not reachable
from this stage -- ``PoolDataset`` refuses to construct against it -- so no
amount of tuning here can contaminate the reported result.

Designed to run on a Colab GPU. Everything downstream consumes only the saved
patch tensor and its digest, so the training device never enters a locally
measured number.

Usage:
    scripts/04_train_patch.py --probe-timing        # measure steps/sec, no output
    scripts/04_train_patch.py --epochs 30           # full run
    scripts/04_train_patch.py --epochs 30 --write-steps
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from adahuman.attack.patch import (  # noqa: E402
    AdversarialPatch,
    EOTParams,
    apply_patch,
    person_scores,
    suppression_loss,
)
from adahuman.config import DEFAULT_PROTOCOL, load_protocol  # noqa: E402
from adahuman.data.dataset import PoolDataset, collate  # noqa: E402
from adahuman.models.detector import load_detector  # noqa: E402
from adahuman.utils.run_log import RunLog  # noqa: E402
from adahuman.utils.seed import seed_everything  # noqa: E402

STAGE = "train_patch"
ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=pathlib.Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--probe-timing", action="store_true",
                        help="run 5 steps, report throughput, write nothing")
    parser.add_argument("--write-steps", action="store_true",
                        help="freeze attack.steps in the protocol on completion")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    protocol = load_protocol(STAGE, args.protocol)
    seed = protocol.get("seed")
    seed_everything(seed)
    log = RunLog(STAGE, args.protocol)
    log.pools_read("attack_dev")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device}")
    if device.type == "cpu":
        log.note("trained on CPU; expect substantially fewer steps than on GPU")

    from pycocotools.coco import COCO
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(protocol.get("data.annotations"))

    model = load_detector(protocol.get("model.weights_enum")).to(device)
    model.eval()

    dataset = PoolDataset(protocol, "attack_dev", coco)
    loader = DataLoader(
        dataset,
        batch_size=protocol.get("attack.batch_size"),
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        drop_last=True,
    )

    patch = AdversarialPatch(protocol.get("attack.patch_size_px"), seed=seed).to(device)
    optimizer = torch.optim.Adam([patch.logits], lr=protocol.get("attack.learning_rate"))

    eot = EOTParams.from_protocol(protocol)
    scale_of_bbox = protocol.get("attack.patch_scale_of_bbox")
    min_area = protocol.get("attack.min_target_bbox_area")
    tv_weight = protocol.get("attack.tv_weight")
    category_id = protocol.get("task.category_id")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    epochs = 1 if args.probe_timing else args.epochs
    max_steps = 5 if args.probe_timing else None

    history = []
    step = 0
    started = time.time()

    for epoch in range(epochs):
        epoch_loss, epoch_score, epoch_applied, batches = 0.0, 0.0, 0, 0

        for images, targets in loader:
            patched = []
            applied_total = 0
            for image, target in zip(images, targets):
                # Non-crowd boxes only: a crowd region is not a target the
                # attack is trying to suppress, and patching one would train
                # the patch against annotations the metrics ignore.
                real = target["boxes"][~target["iscrowd"].bool()]
                out, applied = apply_patch(
                    image.to(device),
                    real,
                    patch.pixels,
                    eot,
                    scale_of_bbox,
                    min_area,
                    generator,
                )
                patched.append(out)
                applied_total += len(applied)

            if applied_total == 0:
                continue  # no eligible target in this batch; nothing to learn from

            scores = person_scores(model, patched, category_id)
            suppression = suppression_loss(scores)
            total_variation = patch.total_variation()
            loss = suppression + tv_weight * total_variation

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss)
            epoch_score += float(scores.max(dim=1).values.mean())
            epoch_applied += applied_total
            batches += 1
            step += 1

            if max_steps and step >= max_steps:
                break

        if batches:
            history.append(
                {
                    "epoch": epoch,
                    "loss": epoch_loss / batches,
                    "mean_max_person_score": epoch_score / batches,
                    "patches_applied": epoch_applied,
                }
            )
            print(
                f"  epoch {epoch:3d}  loss {epoch_loss / batches:.4f}  "
                f"max-person {epoch_score / batches:.4f}  "
                f"patches {epoch_applied}"
            )
        if max_steps and step >= max_steps:
            break

    elapsed = time.time() - started
    rate = step / elapsed if elapsed else 0.0
    print(f"\n{step} steps in {elapsed:.1f}s  ({rate:.2f} steps/s)")

    if args.probe_timing:
        per_epoch = len(dataset) // protocol.get("attack.batch_size")
        print(f"\nsteps per epoch: {per_epoch}")
        for candidate in (20, 30, 50):
            print(
                f"  {candidate:3d} epochs = {candidate * per_epoch:5d} steps "
                f"~ {candidate * per_epoch / rate / 60:6.1f} min"
            )
        log.set("timing_probe", {"steps": step, "seconds": elapsed, "steps_per_sec": rate})
        log.note("timing probe only; no patch written")
        print(f"run log: {log.write()}")
        return 0

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    patch_path = ARTIFACTS / "patch_v1.pt"
    torch.save(
        {
            "pixels": patch.pixels.detach().cpu(),
            "logits": patch.logits.detach().cpu(),
            "protocol_version": protocol.get("protocol_version"),
            "seed": seed,
            "steps": step,
            "epochs": epochs,
        },
        patch_path,
    )

    from torchvision.utils import save_image

    preview_path = ARTIFACTS / "patch_v1.png"
    save_image(patch.pixels.detach().cpu(), preview_path)

    history_path = ARTIFACTS / "patch_v1_training.json"
    with history_path.open("w") as handle:
        json.dump({"history": history, "steps": step, "seconds": elapsed}, handle, indent=2)
        handle.write("\n")

    for key, path in (
        ("patch", patch_path),
        ("preview", preview_path),
        ("history", history_path),
    ):
        log.output(key, path)
    log.set("training", {"steps": step, "epochs": epochs, "steps_per_sec": rate})

    if args.write_steps:
        _write_steps(args.protocol, step)
        print(f"froze attack.steps = {step}")

    print(f"patch:   {patch_path}")
    print(f"run log: {log.write()}")
    return 0


def _write_steps(path: pathlib.Path, steps: int) -> None:
    text = path.read_text()
    if "steps: PENDING" not in text:
        print("attack.steps already frozen; leaving it alone")
        return
    path.write_text(text.replace("steps: PENDING", f"steps: {steps}", 1))


if __name__ == "__main__":
    raise SystemExit(main())
