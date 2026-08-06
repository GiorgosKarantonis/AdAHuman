"""Deterministic seeding.

Reproducibility is one of the artifact's claims, so seeding is centralized and
the seed used is recorded in every run log rather than left implicit.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and torch RNGs.

    Args:
        seed: The protocol seed.
        deterministic: Request deterministic algorithms from torch. This is
            slower and some ops have no deterministic implementation; when that
            happens torch raises, which is preferable to silently varying
            results between runs.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
