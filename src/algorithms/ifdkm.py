"""
Iterative Fast Discriminative K-means (IFDKM) — Algorithm 2 in the paper.

V3 improvements:
  • Accepts shared init_labels (V1 fair comparison)
  • NO singleton skip (follows paper exactly — V2)
  • Tolerance threshold for row updates (V1 stability)
  • Tracks both linear and DisKmeans objective (V1)
  • Convergence: labels stable OR Δobj ≤ tol (V1)

Score (eq. 37, paper): after removing i from cluster m:
    s_ik = (dot_k + M[i,k]) / √(n_k+1) - dot_k / √n_k
  where dot_k = y_k^T m_k, m_k = M[:,k], M = X·W.
  Move i to best_k if s[best_k] - s[m] > tol.
"""

import logging
import time
from typing import List, Optional

import numpy as np

from .base import BaseAlgorithm
from .diskmeans import _repair_empty

logger = logging.getLogger(__name__)


class IFDKM(BaseAlgorithm):

    @property
    def uses_gamma(self) -> bool:
        return True

    # ── Helper: objective for FDKM/DisKmeans (same objective) ────────────
    def _diskm_obj(self, X: np.ndarray, labels: np.ndarray,
                   G: np.ndarray, n: int, d: int, c: int) -> float:
        A  = np.zeros((c, d))
        nk = np.zeros(c)
        for i in range(n):
            k = labels[i]; A[k] += X[i]; nk[k] += 1
        B  = G @ np.eye(n)  # not efficient; use accumulation instead
        # Efficient: B[k] = G @ y_k = sum of G[:,j] for j in Ck
        B = np.zeros((c, d))
        for i in range(n):
            B[labels[i]] += G[:, i]
        return float(sum(
            float(A[k] @ B[k] / nk[k]) for k in range(c) if nk[k] > 0
        ))

    def _fit(self, X: np.ndarray, gamma: float = 1.0,
             init_labels: Optional[np.ndarray] = None) -> np.ndarray:
        n, d = X.shape
        c    = self.n_clusters

        # ── Precompute G ─────────────────────────────────────────────────
        G = self.solve_G(X, gamma)   # (d, n)

        # ── Initialize ───────────────────────────────────────────────────
        if init_labels is not None:
            labels = init_labels.copy().astype(int)
        else:
            from src.preprocessing import make_random_init
            labels = make_random_init(n, c, seed=42)
        labels = _repair_empty(labels, c)

        logger.debug(f"IFDKM n={n} d={d} c={c} γ={gamma:.2e}")

        nk = np.bincount(labels, minlength=c).astype(float)

        # ── Initial W, M, dot ─────────────────────────────────────────────
        L  = self.compute_L(labels, nk, n, c)   # (n, c)
        W  = G @ L                               # (d, c)
        M  = X @ W                               # (n, c)

        dot = np.zeros(c)
        for k in range(c):
            if nk[k] > 0:
                dot[k] = float(M[labels == k, k].sum())

        # ── DisKmeans objective for tracking ──────────────────────────────
        def diskm_obj_fast():
            A = np.zeros((c, d)); B = np.zeros((c, d)); nk2 = np.zeros(c)
            for i in range(n): k=labels[i]; A[k]+=X[i]; B[k]+=G[:,i]; nk2[k]+=1
            return float(sum(float(A[k]@B[k]/nk2[k]) for k in range(c) if nk2[k]>0))

        objective  = diskm_obj_fast()
        obj_hist   = [objective]
        iter_times = []

        # ── Main loop ─────────────────────────────────────────────────────
        for iteration in range(self.max_iter):
            t0         = time.perf_counter()
            old_labels = labels.copy()
            prev_obj   = objective

            # ── Inner: update Y row by row (W, M fixed) ──────────────────
            for i in range(n):
                m   = int(labels[i])
                mi  = M[i]            # (c,) row i of M

                # Remove i from m
                dot[m]  -= mi[m]
                nk[m]   -= 1

                # Score for each k: s_ik = (dot_k + M[i,k])/sqrt(nk+1) - dot_k/sqrt(nk)
                safe_nk = np.where(nk > 0, nk, 1.0)
                new_t   = (dot + mi) / np.sqrt(nk + 1.0)         # (c,)
                old_t   = np.where(nk > 0, dot / np.sqrt(safe_nk), 0.0)
                scores  = new_t - old_t

                best_k  = int(np.argmax(scores))
                if best_k != m and (scores[best_k] - scores[m]) <= self.tol:
                    best_k = m

                dot[best_k] += mi[best_k]
                nk[best_k]  += 1
                labels[i]    = best_k

            # ── Update W, M after full pass ───────────────────────────────
            L   = self.compute_L(labels, nk, n, c)
            W   = G @ L
            M   = X @ W
            dot = np.zeros(c)
            for k in range(c):
                if nk[k] > 0:
                    dot[k] = float(M[labels == k, k].sum())

            objective = diskm_obj_fast()
            obj_hist.append(objective)
            self.n_iter_ = iteration + 1
            iter_times.append(time.perf_counter() - t0)

            n_changed = int((labels != old_labels).sum())
            delta_obj = abs(objective - prev_obj)
            logger.debug(f"  iter {iteration+1}: changed={n_changed} Δobj={delta_obj:.2e}")

            if n_changed == 0 or delta_obj <= self.tol:
                logger.debug(f"  IFDKM converged at iter {iteration+1}")
                break

        self.objective_         = objective
        self.objective_history_ = obj_hist
        self.iteration_times_   = iter_times
        self.W_                 = W
        self.M_                 = M
        return labels
