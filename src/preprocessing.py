"""
Preprocessing — all centering/normalization modes for V4 centering study.

Modes
-----
center        subtract per-feature mean  (X·1_n = 0)  ← paper states this
standardize   subtract mean + divide by std  (z-score)
pca_whiten    center + PCA + unit variance per component  ← γ→0 limit in paper
l2_sample     L2-normalize each sample vector
center_l2     center features, then L2-normalize each sample
frobenius     divide whole matrix by its Frobenius norm
robust        subtract median, divide by IQR  (robust z-score)
minmax        scale each feature to [0, 1]
row_center    subtract per-sample mean  (row-wise centering)
double_center subtract row means + col means + add grand mean
false/none    no preprocessing  (raw features)
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler,
    normalize,
)


# ── Label encoding ──────────────────────────────────────────────────────────

def encode_labels(y_raw) -> np.ndarray:
    return LabelEncoder().fit_transform(np.asarray(y_raw).ravel())


# ── Individual centering functions ──────────────────────────────────────────

def mode_center(X: np.ndarray) -> np.ndarray:
    """① Subtract per-feature mean  →  X·1_n = 0  (paper's stated assumption)."""
    return X - X.mean(axis=0)


def mode_standardize(X: np.ndarray) -> np.ndarray:
    """② Subtract mean + divide by std  (z-score / StandardScaler)."""
    return StandardScaler().fit_transform(X)


def mode_pca_whiten(X: np.ndarray) -> np.ndarray:
    """③ Center + PCA + unit variance per component.

    Equivalent to FDKM with γ→0 (paper Section 5 limiting case).
    Dimensionality preserved (all components used).
    """
    X_c   = X - X.mean(axis=0)
    n_comp = min(X_c.shape[0] - 1, X_c.shape[1])
    pca   = PCA(n_components=n_comp, whiten=True, random_state=0)
    return pca.fit_transform(X_c)


def mode_l2_sample(X: np.ndarray) -> np.ndarray:
    """④ L2-normalize each sample vector  ||x_i||_2 = 1."""
    return normalize(X, norm="l2", axis=1)


def mode_center_l2(X: np.ndarray) -> np.ndarray:
    """⑤ Center features first, then L2-normalize each sample.

    Common in text/image clustering — removes feature-scale bias
    then projects each sample onto the unit sphere.
    """
    X_c = X - X.mean(axis=0)
    return normalize(X_c, norm="l2", axis=1)


def mode_frobenius(X: np.ndarray) -> np.ndarray:
    """⑥ Divide by Frobenius norm  →  ||X||_F = 1.

    Global rescaling; preserves all relative ratios.
    """
    fro = np.linalg.norm(X, "fro")
    return X / fro if fro > 0 else X.copy()


def mode_robust(X: np.ndarray) -> np.ndarray:
    """⑦ Subtract median, divide by IQR  (robust z-score)."""
    return RobustScaler().fit_transform(X)


def mode_minmax(X: np.ndarray) -> np.ndarray:
    """⑧ Scale each feature to [0, 1]."""
    return MinMaxScaler().fit_transform(X)


def mode_row_center(X: np.ndarray) -> np.ndarray:
    """⑨ Subtract per-sample mean (row centering).

    Centers across features for each sample independently.
    """
    return X - X.mean(axis=1, keepdims=True)


def mode_double_center(X: np.ndarray) -> np.ndarray:
    """⑩ Double centering: subtract row + col means, add grand mean.

    Makes both row means and column means zero simultaneously.
    Used in metric learning and MDS.
    """
    row_m   = X.mean(axis=1, keepdims=True)
    col_m   = X.mean(axis=0, keepdims=True)
    grand_m = X.mean()
    return X - row_m - col_m + grand_m


def mode_none(X: np.ndarray) -> np.ndarray:
    """⑪ No preprocessing (raw features)."""
    return X.copy()


# ── Dispatch ────────────────────────────────────────────────────────────────

_MODES = {
    "center":        mode_center,
    "standardize":   mode_standardize,
    "pca_whiten":    mode_pca_whiten,
    "l2_sample":     mode_l2_sample,
    "center_l2":     mode_center_l2,
    "frobenius":     mode_frobenius,
    "robust":        mode_robust,
    "minmax":        mode_minmax,
    "row_center":    mode_row_center,
    "double_center": mode_double_center,
    "false":         mode_none,
    "none":          mode_none,
    True:            mode_center,
    False:           mode_none,
}

ALL_MODES = [
    "center", "standardize", "pca_whiten",
    "l2_sample", "center_l2", "frobenius",
    "robust", "minmax", "row_center", "double_center",
    "false",
]


def apply_normalization(X: np.ndarray, mode) -> np.ndarray:
    fn = _MODES.get(mode, mode_none)
    return fn(X).astype(np.float64)


# ── Other helpers ────────────────────────────────────────────────────────────

def make_random_init(n_samples: int, n_clusters: int, seed: int) -> np.ndarray:
    """Random cluster assignment ensuring every cluster has ≥ 1 member."""
    rng    = np.random.default_rng(seed)
    labels = np.empty(n_samples, dtype=int)
    order  = rng.permutation(n_samples)
    labels[order[:n_samples if n_clusters > n_samples else n_clusters]] = (
        np.arange(min(n_clusters, n_samples))
    )
    if n_samples > n_clusters:
        labels[order[n_clusters:]] = rng.integers(
            0, n_clusters, size=n_samples - n_clusters
        )
    return labels


def stratified_sample(
    X: np.ndarray, y: np.ndarray, max_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or max_samples >= len(y):
        return X, y
    from sklearn.model_selection import train_test_split
    n_classes = len(np.unique(y))
    size = max(max_samples, n_classes)
    try:
        idx, _ = train_test_split(
            np.arange(len(y)), train_size=size, stratify=y, random_state=seed
        )
    except ValueError:
        idx = np.random.default_rng(seed).choice(len(y), size=size, replace=False)
    return X[np.sort(idx)], y[np.sort(idx)]
