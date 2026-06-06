"""
IFDFD = IFDKM → FDKM sequential combination.

V3: shared init passes through IFDKM, FDKM starts from IFDKM's result.
Objective history = ifdkm_history + fdkm_history[1:].
"""

import logging
from typing import Optional

import numpy as np

from .base import BaseAlgorithm
from .fdkm  import FDKM
from .ifdkm import IFDKM

logger = logging.getLogger(__name__)


class IFDFD(BaseAlgorithm):

    @property
    def uses_gamma(self) -> bool:
        return True

    def _fit(self, X: np.ndarray, gamma: float = 1.0,
             init_labels: Optional[np.ndarray] = None) -> np.ndarray:
        n, d = X.shape
        c    = self.n_clusters
        logger.debug(f"IFDFD n={n} d={d} c={c} γ={gamma:.2e}")

        # ── Stage 1: IFDKM ────────────────────────────────────────────────
        logger.debug("  [IFDFD] Stage 1 – IFDKM")
        ifdkm        = IFDKM(n_clusters=c, max_iter=self.max_iter, tol=self.tol)
        ifdkm_labels, _ = ifdkm.fit(X, gamma=gamma, init_labels=init_labels)

        # ── Stage 2: FDKM warm-started from IFDKM result ─────────────────
        logger.debug("  [IFDFD] Stage 2 – FDKM (warm-start from IFDKM)")
        fdkm         = FDKM(n_clusters=c, max_iter=self.max_iter, tol=self.tol)
        fdkm_labels, _ = fdkm.fit(X, gamma=gamma, init_labels=ifdkm_labels)

        # ── Combine history ───────────────────────────────────────────────
        combined_hist = (ifdkm.objective_history_ +
                         fdkm.objective_history_[1:])
        self.n_iter_            = ifdkm.n_iter_ + fdkm.n_iter_
        self.objective_         = fdkm.objective_
        self.objective_history_ = combined_hist
        self.iteration_times_   = ifdkm.iteration_times_ + fdkm.iteration_times_
        self.ifdkm_model_       = ifdkm
        self.fdkm_model_        = fdkm

        return fdkm_labels
