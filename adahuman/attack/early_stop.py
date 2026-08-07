"""Plateau detection for patch optimization.

Fixing the epoch count in advance has two failure modes, and the first training
run hit both. Too few epochs and the patch is under-trained -- 30 epochs left
the mean maximum person score at 0.71, still descending. Too many and the run
wastes GPU time after the objective stops moving. Neither is knowable before
the run, so the criterion should be a property of the curve, not a guess.

There is a third problem a fixed count does not even address: the *last* epoch
is not the best one. In that run the final score was 0.7137 while epoch 27 had
reached 0.7000. Saving whatever the patch happened to be when the loop exited
discards the better patch the run already found.

Both are decided on the ``attack_dev`` training curve, with the held-out pool
untouched, so this is selection the pool discipline permits. Choosing an epoch
count after seeing held-out results would not be.

Smoothing matters here. Epoch-to-epoch scores in the observed run varied by
about +/-0.005 from expectation-over-transformation sampling alone, while the
genuine improvement late in training was around 0.0035 per epoch. A criterion
reading raw epoch values would fire on noise, and picking the single best raw
epoch would checkpoint a lucky sample rather than a better patch. Both
decisions therefore use a trailing mean.
"""

from __future__ import annotations

import dataclasses
import statistics
from typing import Any


@dataclasses.dataclass
class StopDecision:
    """What the stopper concluded after one epoch."""

    should_stop: bool
    is_best: bool
    smoothed: float | None
    best_smoothed: float | None
    epochs_without_improvement: int
    reason: str | None = None


class PlateauStopper:
    """Stops when a minimized metric stops improving.

    Args:
        patience: Epochs to allow without improvement before stopping.
        min_delta: How much the smoothed metric must fall to count as an
            improvement. Should exceed the noise floor of the smoothed value,
            or the stopper will keep resetting on sampling noise.
        window: Trailing epochs averaged before comparison. A window of 1
            disables smoothing.

    The stopper reports ``is_best`` so the caller can checkpoint the patch at
    the best point rather than the last one.
    """

    def __init__(self, patience: int = 30, min_delta: float = 0.003, window: int = 5):
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")

        self.patience = patience
        self.min_delta = min_delta
        self.window = window

        self.history: list[float] = []
        self.best_smoothed: float | None = None
        self.best_epoch: int | None = None
        self.epochs_without_improvement = 0
        self.stopped_early = False

    def update(self, value: float) -> StopDecision:
        """Record one epoch's metric and decide whether to continue.

        Args:
            value: The metric for this epoch. Lower is better.
        """
        self.history.append(value)
        epoch = len(self.history) - 1

        # Until a full window exists the trailing mean is computed over fewer
        # points and is correspondingly noisier, so no stopping decision is
        # taken yet. The first window is still eligible to be the best, since
        # otherwise a run that converges immediately would checkpoint nothing.
        smoothed = statistics.fmean(self.history[-self.window :])
        warming_up = len(self.history) < self.window

        if self.best_smoothed is None or smoothed < self.best_smoothed - self.min_delta:
            self.best_smoothed = smoothed
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return StopDecision(False, True, smoothed, self.best_smoothed, 0)

        # Track the best value even when the improvement is too small to reset
        # patience, so a long slow drift is not mistaken for a new best later.
        is_best = smoothed < self.best_smoothed
        if is_best:
            self.best_smoothed = smoothed
            self.best_epoch = epoch

        if warming_up:
            return StopDecision(False, is_best, smoothed, self.best_smoothed, 0)

        self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
            self.stopped_early = True
            return StopDecision(
                True,
                is_best,
                smoothed,
                self.best_smoothed,
                self.epochs_without_improvement,
                reason=(
                    f"no improvement greater than {self.min_delta} in the "
                    f"{self.window}-epoch trailing mean for {self.patience} epochs"
                ),
            )

        return StopDecision(
            False, is_best, smoothed, self.best_smoothed,
            self.epochs_without_improvement,
        )

    def summary(self) -> dict[str, Any]:
        """Record written into the run log and the training history file."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "window": self.window,
            "epochs_run": len(self.history),
            "best_epoch": self.best_epoch,
            "best_smoothed": self.best_smoothed,
            "stopped_early": self.stopped_early,
            "epochs_without_improvement": self.epochs_without_improvement,
        }
