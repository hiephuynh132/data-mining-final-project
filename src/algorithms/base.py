"""
Shared base class and helpers for all V3 clustering algorithms.

V3 design principles (best of V1 + V2):
  - No singleton skip    (follows paper exactly — V2 approach)
  - Tolerance threshold  (only move if gain > tol — V1 stability)
  - Objective tracking   (store history — V1 feature)
  - accept init_labels   (shared init from runner — V1 fair comparison)
  - center_data=False    (normalization done OUTSIDE by runner — cleaner)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class BaseAlgorithm(ABC):
    """
    Interface: algorithms accept pre-normalized X and optional init_labels.
    Returns labels + stores runtime, objective, objective_history.
    """

    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-6):
        self.n_clusters = n_clusters
        self.max_iter   = max_iter
        self.tol        = tol

        self.labels_:            Optional[np.ndarray] = None
        self.objective_:         Optional[float]      = None
        self.objective_history_: List[float]          = []
        self.n_iter_:            int                  = 0
        self.iteration_times_:   List[float]          = []

    # ── subclasses implement ────────────────────────────────────────────────

    @property
    def uses_gamma(self) -> bool:
        return False

    @abstractmethod
    def _fit(self, X: np.ndarray, gamma: Optional[float],
             init_labels: Optional[np.ndarray]) -> np.ndarray:
        """Return cluster labels (int array shape (n,))."""

    # ── public API ──────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, gamma: Optional[float] = None,
            init_labels: Optional[np.ndarray] = None) -> tuple[np.ndarray, float]:
        """
        Fit and return (labels, elapsed_seconds).

        init_labels : optional pre-computed starting assignment.
                      If None, each algorithm falls back to its own strategy.
        """
        t0     = time.perf_counter()
        labels = self._fit(X, gamma=gamma, init_labels=init_labels)
        self.labels_ = labels
        return labels, time.perf_counter() - t0

    # ── shared helpers ──────────────────────────────────────────────────────

    @staticmethod
    def solve_G(X: np.ndarray, gamma: float) -> np.ndarray:
        """G = (X^TX + γI_d)^{-1} X^T  shape (d, n). Efficient for d << n."""
        n, d = X.shape
        A    = X.T @ X + gamma * np.eye(d)      # (d, d)
        return np.linalg.solve(A, X.T)           # (d, n)

    @staticmethod
    def compute_L(labels: np.ndarray, nk: np.ndarray, n: int, c: int) -> np.ndarray:
        """L = Y (Y^T Y)^{-1/2}  shape (n, c). Zero for empty clusters."""
        L = np.zeros((n, c))
        for k in range(c):
            if nk[k] > 0:
                L[labels == k, k] = 1.0 / np.sqrt(nk[k])
        return L
