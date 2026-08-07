"""Dated, hashed run records.

Every stage writes one JSON record describing what it did: when, on what
hardware, with which library versions, against which protocol, reading which
pools, producing which outputs and their digests.

These records are the difference between "here is a number" and "here is a
number a third party can situate and check". They are committed to the
repository alongside the results they describe.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import platform
import subprocess
from typing import Any

from adahuman.utils.hashing import sha256_file

RUN_LOG_DIR = pathlib.Path(__file__).resolve().parents[2] / "logs"


def _git_commit() -> str | None:
    """Current commit, or None outside a git checkout.

    Recorded so a result can be traced to the exact code that produced it.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=pathlib.Path(__file__).resolve().parents[2],
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip() or None


#: Paths whose state determines what the code does. Everything else in the
#: working tree is output: logs/, results/ and artifacts/ are written by the
#: stages themselves, and a stage that inspected the whole tree would report
#: itself dirty because an earlier stage had left a log behind.
SOURCE_PATHS = (
    "adahuman",
    "scripts",
    "configs",
    "notebooks",
    "requirements.txt",
    "requirements.lock",
)


def _git_status_source() -> list[str] | None:
    """Porcelain status lines for source paths only.

    Scoped deliberately. The question this answers is whether the code that
    produced a result differs from the commit recorded alongside it, so it
    covers source, configuration and the frozen protocol, and ignores the
    directories stages write into. An unscoped check reports dirty as soon as
    any stage has run, which makes the flag carry no information.

    Untracked files inside these paths do count: an uncommitted module changes
    behaviour just as an edited one does.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", *SOURCE_PATHS],
            capture_output=True,
            text=True,
            check=True,
            cwd=pathlib.Path(__file__).resolve().parents[2],
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return [line for line in out.stdout.splitlines() if line.strip()]


def _git_dirty() -> bool | None:
    """Whether the source differs from the recorded commit.

    A result produced from modified source is not reproducible from the commit
    alone, so the fact is recorded rather than hidden.
    """
    lines = _git_status_source()
    return None if lines is None else bool(lines)


def environment() -> dict[str, Any]:
    """Library and platform versions, captured at run time."""
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    for module_name, key in (
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("numpy", "numpy"),
        ("onnxruntime", "onnxruntime"),
        ("sklearn", "scikit_learn"),
    ):
        try:
            module = __import__(module_name)
            env[key] = getattr(module, "__version__", "unknown")
        except ImportError:
            env[key] = None

    try:
        import torch

        env["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["torch_cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return env


class RunLog:
    """Accumulates a stage's provenance record, then writes it once."""

    def __init__(self, stage: str, protocol_path: pathlib.Path | str):
        self.stage = stage
        self.started_at = _dt.datetime.now(_dt.timezone.utc)
        self.record: dict[str, Any] = {
            "stage": stage,
            "started_at_utc": self.started_at.isoformat(),
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            # The offending entries, so a dirty run says which files diverged
            # instead of leaving it to be reconstructed afterwards.
            "git_dirty_paths": _git_status_source() or [],
            "protocol": {
                "path": str(protocol_path),
                "sha256": sha256_file(protocol_path),
            },
            "environment": environment(),
            "pools_read": [],
            "inputs": {},
            "outputs": {},
            "notes": [],
        }

    def pools_read(self, *names: str) -> None:
        self.record["pools_read"].extend(names)

    def input(self, key: str, path: pathlib.Path | str) -> None:
        """Record an input file and its digest."""
        path = pathlib.Path(path)
        self.record["inputs"][key] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }

    def output(self, key: str, path: pathlib.Path | str) -> None:
        """Record an output file and its digest."""
        path = pathlib.Path(path)
        self.record["outputs"][key] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }

    def note(self, message: str) -> None:
        """Attach a free-text observation.

        Used for anything a reader would want disclosed: a failed run, a
        numerical warning, an unsupported operator during export, a deviation
        from the plan.
        """
        self.record["notes"].append(message)

    def set(self, key: str, value: Any) -> None:
        self.record[key] = value

    def write(self, directory: pathlib.Path | str = RUN_LOG_DIR) -> pathlib.Path:
        """Write the record and return its path."""
        finished = _dt.datetime.now(_dt.timezone.utc)
        self.record["finished_at_utc"] = finished.isoformat()
        self.record["duration_seconds"] = round(
            (finished - self.started_at).total_seconds(), 3
        )

        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.strftime("%Y%m%dT%H%M%SZ")
        path = directory / f"{stamp}_{self.stage}.json"
        with path.open("w") as handle:
            json.dump(self.record, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path
