#!/usr/bin/env python
"""Resolve the environment-dependent protocol fields.

Several protocol values cannot be written by hand: the exact feature tensor
shape, the checkpoint digest, the resolved library versions. This script probes
the loaded model, reports them, and -- with ``--write`` -- fills them into
``configs/protocol_v1.yaml``, replacing the ``PENDING`` sentinels.

It reads no pool data and produces no measurement, so it is safe to run before
the protocol is frozen. That is the point: it is what makes freezing possible.

Usage:
    scripts/01_probe_model.py            # report only
    scripts/01_probe_model.py --write    # report and freeze the fields
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from adahuman.config import DEFAULT_PROTOCOL, load_protocol  # noqa: E402
from adahuman.models.detector import (  # noqa: E402
    FEATURE_HOOK_PATH,
    FeatureTap,
    load_detector,
    resolve_module,
)
from adahuman.utils.hashing import sha256_file, short  # noqa: E402
from adahuman.utils.run_log import RunLog  # noqa: E402
from adahuman.utils.seed import seed_everything  # noqa: E402

STAGE = "probe_model"


def checkpoint_path(weights_enum: str) -> pathlib.Path | None:
    """Locate the cached torchvision checkpoint so it can be digested."""
    cache = pathlib.Path(torch.hub.get_dir()) / "checkpoints"
    if not cache.is_dir():
        return None
    candidates = sorted(cache.glob("ssdlite320_mobilenet_v3_large*.pth"))
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="freeze the probed values into the protocol YAML",
    )
    parser.add_argument("--protocol", type=pathlib.Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    protocol = load_protocol(STAGE, args.protocol)
    seed_everything(protocol.get("seed"))
    log = RunLog(STAGE, args.protocol)

    weights_enum = protocol.get("model.weights_enum")
    print(f"loading {protocol.get('model.name')} [{weights_enum}]")
    model = load_detector(weights_enum)

    # A single synthetic forward pass at the protocol input size. Synthetic
    # input is deliberate: this probe must not touch pool data.
    height, width = protocol.get("model.input_size")
    probe_batch = [torch.rand(3, height, width)]

    with torch.no_grad(), FeatureTap(model, FEATURE_HOOK_PATH) as tap:
        model(probe_batch)
        raw = tap.raw
        pooled = tap.pooled()

    unpooled_shape = list(raw.shape[1:])  # (C, H, W), batch dim dropped
    feature_dim = int(pooled.shape[1])

    module = resolve_module(model, FEATURE_HOOK_PATH)
    parameters = sum(p.numel() for p in model.parameters())

    ckpt = checkpoint_path(weights_enum)
    weights_sha256 = sha256_file(ckpt) if ckpt else None

    print()
    print(f"  feature hook       {FEATURE_HOOK_PATH}  ({type(module).__name__})")
    print(f"  unpooled shape     {tuple(unpooled_shape)}  (C, H, W)")
    print(f"  pooled dimension   {feature_dim}")
    print(f"  model parameters   {parameters:,}")
    print(f"  torch              {torch.__version__}")
    print(f"  checkpoint         {ckpt.name if ckpt else 'not found in cache'}")
    print(f"  checkpoint sha256  {short(weights_sha256, 16) if weights_sha256 else '-'}")

    stride = height / unpooled_shape[1]
    print(f"  output stride      {stride:g}")
    if stride != 32:
        log.note(f"unexpected output stride {stride:g}; expected 32")
        print(
            f"  WARNING: expected output stride 32 at {FEATURE_HOOK_PATH}, "
            f"got {stride:g}. Verify the hook path before freezing."
        )

    log.set(
        "probe",
        {
            "feature_hook": FEATURE_HOOK_PATH,
            "feature_module_type": type(module).__name__,
            "feature_unpooled_shape": unpooled_shape,
            "feature_dim": feature_dim,
            "output_stride": stride,
            "model_parameters": parameters,
            "weights_sha256": weights_sha256,
            "checkpoint_file": ckpt.name if ckpt else None,
        },
    )

    if args.write:
        _freeze(
            args.protocol,
            {
                "  weights_sha256:": weights_sha256 or "unavailable",
                "  torch_version:": torch.__version__,
                "  torchvision_version:": __import__("torchvision").__version__,
                "  feature_hook:": FEATURE_HOOK_PATH,
                "  feature_unpooled_shape:": str(unpooled_shape),
                "  feature_dim:": str(feature_dim),
            },
        )
        print(f"\nfroze 6 fields in {args.protocol.name}")
        log.note("wrote probed values into the protocol")

    path = log.write()
    print(f"run log: {path.relative_to(pathlib.Path.cwd())}")
    return 0


def _freeze(path: pathlib.Path, replacements: dict[str, str]) -> None:
    """Replace ``PENDING`` on specific keys, leaving the rest of the file alone.

    Line-oriented on purpose: a YAML round-trip would discard the comments,
    and those comments are what explain the protocol to a reviewer.
    """
    lines = path.read_text().splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        for prefix, value in replacements.items():
            if line.startswith(prefix) and "PENDING" in line:
                head, _, tail = line.partition("PENDING")
                line = f"{head}{value}{tail}"
                break
        out.append(line)
    path.write_text("".join(out))


if __name__ == "__main__":
    raise SystemExit(main())
