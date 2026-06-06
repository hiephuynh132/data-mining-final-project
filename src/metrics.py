"""ACC (Hungarian), NMI, ARI — evaluation metrics."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Clustering accuracy via optimal Hungarian label matching."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return float(cm[row_ind, col_ind].sum() / len(y_true))


def clustering_nmi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic"))


def clustering_ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(adjusted_rand_score(y_true, y_pred))


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "acc": clustering_accuracy(y_true, y_pred),
        "nmi": clustering_nmi(y_true, y_pred),
        "ari": clustering_ari(y_true, y_pred),
    }
