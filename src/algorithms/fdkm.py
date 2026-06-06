"""
Fast Discriminative K-means (FDKM) — Algorithm 1 in the paper.

V3 improvements (best of V1 + V2):
  • Accepts shared init_labels (V1 fair comparison)
  • NO singleton skip  → follows paper exactly (V2)
  • Tolerance threshold: only move if Δobjective > tol (V1 stability)
  • Incremental A/B updates (V2 efficiency — O(cd) per point, not O(cd²))
  • Objective tracking (V1)
  • Convergence: labels stable OR Δobj ≤ tol (V1)

Score formula (eq. 20, paper):
  After removing i from cluster m, for each candidate k:
    s_ik = (AB_k + 2*(A_k·g_i) + K[i,i]) / (n_k+1) - AB_k/n_k

  where A_k = Σ x_j, B_k = G y_k = (X^TX+γI)^{-1} A_k, AB_k = A_k·B_k.
  K[i,i] = x_i · g_i.

  Move i to best_k if   s[best_k] - s[m] > tol   (Δ-total-objective > tol).
  This is exactly V1's   candidate_objective > current_objective + tol.
"""

import logging
import time
from typing import List, Optional

import numpy as np

from .base import BaseAlgorithm
from .diskmeans import _repair_empty

logger = logging.getLogger(__name__)


class FDKM(BaseAlgorithm):

    @property
    def uses_gamma(self) -> bool:
        return True

    def _fit(self, X: np.ndarray, gamma: float = 1.0,
             init_labels: Optional[np.ndarray] = None) -> np.ndarray:
        n, d = X.shape
        c    = self.n_clusters

        # ── Precompute G and per-sample K[i,i] ──────────────────────────
        G     = self.solve_G(X, gamma)              # (d, n)
        Kdiag = np.einsum("ij,ji->i", X, G)        # (n,)  K[i,i] = x_i·g_i

        # ── Initialize ───────────────────────────────────────────────────
        if init_labels is not None:
            labels = init_labels.copy().astype(int)
        else:
            from src.preprocessing import make_random_init
            labels = make_random_init(n, c, seed=42)
        labels = _repair_empty(labels, c)

        logger.debug(f"FDKM n={n} d={d} c={c} γ={gamma:.2e}")

        # ── Build accumulators ───────────────────────────────────────────
        A  = np.zeros((c, d))       # A[k] = Σ_{j∈Ck} x_j
        B  = np.zeros((c, d))       # B[k] = Σ_{j∈Ck} G[:,j]
        nk = np.zeros(c)
        for i in range(n):
            k = labels[i]; A[k] += X[i]; B[k] += G[:, i]; nk[k] += 1
        AB = np.einsum("kd,kd->k", A, B)  # AB[k] = A[k]·B[k]

        # ── Compute initial objective ─────────────────────────────────────
        def obj_from_AB():
            return float(np.where(nk > 0, AB / nk, 0.0).sum())

        objective  = obj_from_AB()
        obj_hist   = [objective]
        iter_times = []

        # ── Main loop ─────────────────────────────────────────────────────
        for iteration in range(self.max_iter):
            t0         = time.perf_counter()
            prev_obj   = objective
            old_labels = labels.copy()

            for i in range(n):
                m  = int(labels[i])
                xi = X[i]
                gi = G[:, i]

                # ── Remove i from cluster m ────────────────────────────
                A[m]  -= xi; B[m]  -= gi; nk[m] -= 1; AB[m] = float(A[m] @ B[m])

                # ── Scores for all clusters k ──────────────────────────
                # s_ik = (AB_k + 2*cross_k + K[i,i]) / (nk+1) - AB_k/nk
                cross   = A @ gi                                         # (c,)
                old_obj = np.divide(AB, nk, where=nk > 0, out=np.zeros(c))
                scores  = (AB + 2.0 * cross + Kdiag[i]) / (nk + 1.0) - old_obj

                # ── Tolerance: only move if Δ > tol (V1 stability) ────
                best_k = int(np.argmax(scores))
                # scores[best_k] - scores[m] > tol  ≡  Δ total obj > tol
                if best_k != m and (scores[best_k] - scores[m]) <= self.tol:
                    best_k = m   # keep in m

                # ── Add i to winning cluster ───────────────────────────
                A[best_k]  += xi; B[best_k]  += gi
                nk[best_k] += 1;  AB[best_k]  = float(A[best_k] @ B[best_k])
                labels[i]   = best_k

            objective = obj_from_AB()
            obj_hist.append(objective)
            self.n_iter_ = iteration + 1
            iter_times.append(time.perf_counter() - t0)

            n_changed = int((labels != old_labels).sum())
            delta_obj = abs(objective - prev_obj)
            logger.debug(f"  iter {iteration+1}: changed={n_changed} Δobj={delta_obj:.2e}")

            if n_changed == 0 or delta_obj <= self.tol:
                logger.debug(f"  FDKM converged at iter {iteration+1}")
                break

        self.objective_         = objective
        self.objective_history_ = obj_hist
        self.iteration_times_   = iter_times
        return labels
