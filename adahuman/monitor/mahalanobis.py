"""Feature-distance runtime monitor (RQ3).

A Mahalanobis-style distance from an input's pooled backbone features to the
clean in-distribution feature statistics. Clean-room implementation from public
literature (Lee et al., NeurIPS 2018); no employer or client metric, threshold,
or feature representation is reused.

The hypothesis under test is narrow: *some* adversarial or patch-affected
inputs may sit far enough from the clean feature distribution to be flagged at
runtime without an unacceptable false-positive rate. It is a hypothesis, not a
defence. Published work establishes both that distance scores catch some
out-of-distribution and adversarial inputs, and that detecting adversarial
examples can be nearly as hard as classifying them correctly under an adaptive
attacker. This monitor is evaluated only against a non-adaptive attacker, which
is the weaker and easier condition.

Two properties of the fit are load-bearing, and the first is easy to get wrong.

**The threshold is calibrated out of sample.** An in-sample quantile hits its
target false-positive rate *by construction* and says nothing about
deployment. A first version of this monitor selected its threshold on the same
images it estimated the covariance from, reported the 5% it was guaranteed to
report, and produced a **60% false-positive rate on held-out clean images from
the same pool**. The threshold is now calibrated on a reference split held out
from the covariance fit, which brings held-out FPR to within sampling error of
the 5% target.

**Features are projected by PCA before the covariance is estimated.** Fitting a
480x480 covariance from a few hundred images leaves roughly one sample per
dimension; the estimate is near-singular and its inverse is dominated by noise
in the smallest eigendirections. Measured on reference data, this mattered less
than the calibration bug -- out-of-sample calibration alone brings the error
rate back in line at any dimension tested -- but the projection is retained
because a near-singular precision matrix is a fragile thing to carry into an
evaluation whose whole point is whether the score generalizes.

The component count is fixed a priori by a stated rule (smallest power of two
retaining at least 95% of variance, which selects 64) rather than searched
against any outcome.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import numpy as np
import torch


@dataclasses.dataclass
class FeatureDistanceMonitor:
    """Fitted clean-feature statistics and the distance they induce."""

    feature_mean: np.ndarray  # (D,), PCA centring
    components: np.ndarray  # (k, D), PCA basis
    explained_variance_ratio: float
    mean: np.ndarray  # (k,), mean in the projected space
    precision: np.ndarray  # (k, k), inverted shrunk covariance
    shrinkage: float
    n_fit: int
    feature_dim: int
    n_components: int
    threshold: float | None = None

    @classmethod
    def fit(
        cls,
        features: torch.Tensor | np.ndarray,
        n_components: int = 64,
    ) -> "FeatureDistanceMonitor":
        """Estimate the projection, mean, and shrunk precision from clean features.

        Args:
            features: ``(N, D)`` pooled features from clean in-distribution
                images. Must come from the reference pool only.
            n_components: PCA dimension. Fixed a priori rather than searched,
                so it is not a knob tuned against the outcome.
        """
        from sklearn.covariance import LedoitWolf
        from sklearn.decomposition import PCA

        array = _as_array(features)
        if array.ndim != 2:
            raise ValueError(f"expected (N, D) features, got {array.shape}")
        if n_components >= array.shape[0]:
            raise ValueError(
                f"n_components={n_components} needs more than {n_components} "
                f"fit samples; got {array.shape[0]}"
            )

        pca = PCA(n_components=n_components, svd_solver="full").fit(array)
        projected = pca.transform(array)
        estimator = LedoitWolf().fit(projected)

        return cls(
            feature_mean=pca.mean_,
            components=pca.components_,
            explained_variance_ratio=float(pca.explained_variance_ratio_.sum()),
            mean=estimator.location_,
            precision=estimator.precision_,
            shrinkage=float(estimator.shrinkage_),
            n_fit=int(array.shape[0]),
            feature_dim=int(array.shape[1]),
            n_components=int(n_components),
        )

    def project(self, features: torch.Tensor | np.ndarray) -> np.ndarray:
        """Map raw pooled features into the fitted PCA subspace."""
        array = _as_array(features)
        if array.shape[1] != self.feature_dim:
            raise ValueError(
                f"monitor was fit on {self.feature_dim} dimensions, "
                f"got {array.shape[1]}"
            )
        return (array - self.feature_mean) @ self.components.T

    def score(self, features: torch.Tensor | np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance for each row. Higher is more anomalous."""
        centred = self.project(features) - self.mean
        # einsum rather than a full (N, k) @ (k, k) @ (k, N) product, which
        # would materialize an N x N matrix to read only its diagonal.
        return np.einsum("nk,kl,nl->n", centred, self.precision, centred)

    def flag(self, features: torch.Tensor | np.ndarray) -> np.ndarray:
        """Boolean anomaly flags at the frozen threshold."""
        if self.threshold is None:
            raise RuntimeError(
                "threshold not set; fit it on the reference and development "
                "pools before scoring held-out data."
            )
        return self.score(features) >= self.threshold

    def save(self, path: pathlib.Path | str) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            feature_mean=self.feature_mean,
            components=self.components,
            explained_variance_ratio=self.explained_variance_ratio,
            mean=self.mean,
            precision=self.precision,
            shrinkage=self.shrinkage,
            n_fit=self.n_fit,
            feature_dim=self.feature_dim,
            n_components=self.n_components,
            threshold=np.nan if self.threshold is None else self.threshold,
        )
        return path

    @classmethod
    def load(cls, path: pathlib.Path | str) -> "FeatureDistanceMonitor":
        data = np.load(path)
        threshold = float(data["threshold"])
        return cls(
            feature_mean=data["feature_mean"],
            components=data["components"],
            explained_variance_ratio=float(data["explained_variance_ratio"]),
            mean=data["mean"],
            precision=data["precision"],
            shrinkage=float(data["shrinkage"]),
            n_fit=int(data["n_fit"]),
            feature_dim=int(data["feature_dim"]),
            n_components=int(data["n_components"]),
            threshold=None if np.isnan(threshold) else threshold,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "n_components": self.n_components,
            "explained_variance_ratio": self.explained_variance_ratio,
            "n_fit": self.n_fit,
            "shrinkage": self.shrinkage,
            "threshold": self.threshold,
        }


def _as_array(features: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(features, torch.Tensor):
        return features.detach().cpu().double().numpy()
    return np.asarray(features, dtype=np.float64)
