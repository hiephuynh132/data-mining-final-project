"""
Standard K-means (sklearn wrapper).

V3 choice: keep sklearn KMeans with k-means++ (gives 85.41±0% on BC center-only,
matching the paper exactly). K-means gets its OWN initialization (independent from
the shared random init used by kernel algorithms).
"""

import logging
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from .base import BaseAlgorithm

logger = logging.getLogger(__name__)


class KMeansAlgo(BaseAlgorithm):

    @property
    def uses_gamma(self) -> bool:
        return False

    def _fit(self, X: np.ndarray, gamma=None,
             init_labels: Optional[np.ndarray] = None) -> np.ndarray:
        # KMeans ignores init_labels and uses k-means++ for consistency with paper
        n, d = X.shape
        seed = getattr(self, "_seed", 42)
        logger.debug(f"KMeans n={n} d={d} c={self.n_clusters} seed={seed}")

        km = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=1,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=seed,
        )
        km.fit(X)
        logger.debug(f"  converged in {km.n_iter_} iters, inertia={km.inertia_:.4f}")

        self.n_iter_            = km.n_iter_
        self.objective_         = float(-km.inertia_)   # negative inertia as "objective"
        self.objective_history_ = [float(-km.inertia_)]
        return km.labels_.astype(int)
