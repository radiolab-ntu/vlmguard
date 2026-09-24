"""Equation (6): singular-value-weighted projection energy on unlabeled data."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


@dataclass(frozen=True)
class Subspace:
    mean: np.ndarray
    vectors: np.ndarray
    singular_values: np.ndarray
    layer: int
    k: int


def fit_subspace(features: np.ndarray, max_k: int, layer: int = 0) -> Subspace:
    """Fit one centered SVD on the unlabeled training features only."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2:
        raise ValueError("features must have shape (samples, dimensions), both >= 2")
    if not np.isfinite(x).all():
        raise ValueError("features contain NaN or infinity")
    if not 1 <= max_k <= min(x.shape) - 1:
        raise ValueError("max_k must be between 1 and min(samples, dimensions) - 1")
    mean = x.mean(axis=0)
    centered = x - mean
    gram = centered.T @ centered
    eigvals, vectors = eigh(gram, subset_by_index=(gram.shape[0] - max_k, gram.shape[0] - 1))
    eigvals = np.maximum(eigvals[::-1], 0.0)
    vectors = vectors[:, ::-1]
    if eigvals[0] <= 0:
        raise ValueError("unlabeled features have zero variance")
    return Subspace(mean, vectors, np.sqrt(eigvals), layer, max_k)


def score(features: np.ndarray, subspace: Subspace, k: int | None = None, weighted: bool = False) -> np.ndarray:
    """Return projection energy, optionally weighted by singular values."""
    count = subspace.k if k is None else k
    if not 1 <= count <= subspace.k:
        raise ValueError("k is outside the fitted subspace")
    x = np.asarray(features, dtype=np.float64)
    projection = (x - subspace.mean) @ subspace.vectors[:, :count]
    energy = projection**2
    if weighted:
        energy = energy * subspace.singular_values[None, :count]
    return np.mean(energy, axis=1)


def select_subspace(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    layers: list[int],
    min_k: int,
    max_k: int,
    weighted: bool = False,
) -> tuple[Subspace, float]:
    """Fit on unlabeled train; use validation AUROC only to choose layer and k."""
    train = np.asarray(train_features)
    val = np.asarray(validation_features)
    labels = np.asarray(validation_labels)
    if train.ndim != 3 or val.ndim != 3 or train.shape[1:] != val.shape[1:]:
        raise ValueError("train and validation features need matching (N, layers, dimensions)")
    if len(labels) != len(val) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("validation labels must match features and contain both classes")
    if not layers or any(layer < 0 or layer >= train.shape[1] for layer in layers):
        raise ValueError("layer selection is empty or outside feature dimensions")
    if min_k < 1 or max_k < min_k:
        raise ValueError("require 1 <= min_k <= max_k")
    best: tuple[Subspace, float] | None = None
    for layer in layers:
        fitted = fit_subspace(train[:, layer], max_k, layer)
        for k in range(min_k, max_k + 1):
            candidate = Subspace(fitted.mean, fitted.vectors, fitted.singular_values, layer, k)
            auroc = float(roc_auc_score(labels, score(val[:, layer], candidate, weighted=weighted)))
            if best is None or auroc > best[1]:
                best = (candidate, auroc)
    assert best is not None
    return best


def select_threshold(
    train_scores: np.ndarray,
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    steps: int = 100,
) -> tuple[float, float, float]:
    """Choose a threshold by validation balanced accuracy, from train quantiles."""
    if steps < 2:
        raise ValueError("steps must be at least 2")
    quantiles = np.linspace(0, 1, steps + 2)[1:-1]
    thresholds = np.unique(np.quantile(train_scores, quantiles))
    if len(thresholds) == 0:
        raise ValueError("cannot select a threshold from empty training scores")
    best = max(
        ((float(t), float(balanced_accuracy_score(validation_labels, validation_scores > t))) for t in thresholds),
        key=lambda item: item[1],
    )
    fraction_positive = float(np.mean(train_scores > best[0]))
    if fraction_positive in (0.0, 1.0):
        raise ValueError("threshold produced a single pseudo-label class")
    return best[0], best[1], fraction_positive
