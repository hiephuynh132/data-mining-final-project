"""
DisKmeans — batch kernel K-means.

V3 improvements over V1 + V2:
  • Accepts shared init_labels (V1 fair comparison)
  • Symmetrizes K = (K+K^T)/2 (V1 robustness)
  • Vectorized batch update (both V1 and V2)
  • Tracks objective history (V1)
  • Convergence: labels_stable OR Δobj ≤ tol (V1)
"""

import logging
import time
from typing import List, Optional

import numpy as np

from .base import BaseAlgorithm

logger = logging.getLogger(__name__)


class DisKmeans(BaseAlgorithm):

    @property
    def uses_gamma(self) -> bool:
        return True

    def _fit(self, X: np.ndarray, gamma: float = 1.0,
             init_labels: Optional[np.ndarray] = None) -> np.ndarray:
        n, d = X.shape
        c    = self.n_clusters

        # ── Build kernel K = X(X^TX+γI)^{-1}X^T ─────────────────────────
        G    = self.solve_G(X, gamma)      # (d, n)
        K    = X @ G                       # (n, n)
        K    = (K + K.T) * 0.5            # symmetrize (V1)
        Kd   = np.diag(K)

        # ── Initialize ───────────────────────────────────────────────────
        if init_labels is not None:
            labels = init_labels.copy().astype(int)
        else:
            from src.preprocessing import make_random_init
            labels = make_random_init(n, c, seed=42)

        logger.debug(f"DisKmeans n={n} d={d} c={c} γ={gamma:.2e}")

        # ── Compute initial objective ─────────────────────────────────────
        def objective_fn(lbl):
            obj = 0.0
            for k in range(c):
                mask = lbl == k
                nk   = mask.sum()
                if nk > 0:
                    obj += K[np.ix_(mask, mask)].sum() / nk
            return obj

        objective  = objective_fn(labels)
        obj_hist   = [objective]
        iter_times = []

        # ── Main loop ─────────────────────────────────────────────────────
        for iteration in range(self.max_iter):
            t0 = time.perf_counter()
            old_labels = labels.copy()
            prev_obj   = objective

            nk   = np.bincount(labels, minlength=c).astype(float)
            Y    = (labels[:, None] == np.arange(c)).astype(float)  # (n, c)
            ksum = K @ Y                                             # (n, c)
            wsum = (Y * ksum).sum(axis=0)                           # (c,)

            # distance: d[i,k] = K[i,i] - 2*ksum[i,k]/nk[k] + wsum[k]/nk[k]^2
            with np.errstate(invalid="ignore", divide="ignore"):
                scores = (
                    Kd[:, None]
                    - 2.0 * ksum / np.where(nk > 0, nk, 1.0)[None, :]
                    + wsum[None, :] / np.where(nk > 0, nk ** 2, 1.0)[None, :]
                )
            scores[:, nk == 0] = np.inf   # never assign to empty cluster
            labels = np.argmin(scores, axis=1).astype(int)

            # Repair if any cluster became empty (V1 strategy)
            labels = _repair_empty(labels, c)

            objective = objective_fn(labels)
            obj_hist.append(objective)
            self.n_iter_ = iteration + 1
            iter_times.append(time.perf_counter() - t0)

            n_changed = int((labels != old_labels).sum())
            delta_obj = abs(objective - prev_obj)
            logger.debug(f"  iter {iteration+1}: changed={n_changed} Δobj={delta_obj:.2e}")

            if labels_equal(labels, old_labels) or delta_obj <= self.tol:
                logger.debug(f"  converged at iter {iteration+1}")
                break

        self.objective_         = objective
        self.objective_history_ = obj_hist
        self.iteration_times_   = iter_times
        return labels


def labels_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))


def _repair_empty(labels: np.ndarray, c: int) -> np.ndarray:
    """Move one point from the largest cluster to any empty cluster."""
    labels  = labels.copy()
    counts  = np.bincount(labels, minlength=c)
    for empty in np.flatnonzero(counts == 0):
        donors  = np.flatnonzero(counts > 1)
        if len(donors) == 0:
            break
        donor   = int(donors[np.argmax(counts[donors])])
        idx     = int(np.flatnonzero(labels == donor)[0])
        labels[idx]    = int(empty)
        counts[donor] -= 1
        counts[empty]  = 1
    return labels
