"""K-means clustering + Hungarian-aligned accuracy (Stage 2)."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from .config import K_IDENTITIES, RANDOM_SEED


def cluster_accuracy(y_true, y_pred) -> float:
    """Best-match accuracy after optimal cluster-to-label assignment.

    WHY Hungarian: k-means assigns arbitrary integer labels, so we need
    the permutation of cluster IDs that maximizes diagonal agreement with
    the true labels.
    """
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    cm = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
    for i, lt in enumerate(labels_true):
        for j, lp in enumerate(labels_pred):
            cm[i, j] = np.sum((y_true == lt) & (y_pred == lp))
    row, col = linear_sum_assignment(-cm)
    return cm[row, col].sum() / len(y_true)


def run_kmeans(X, y, k: int = K_IDENTITIES, seed: int = RANDOM_SEED):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    pred = km.fit_predict(X)
    return pred, cluster_accuracy(y, pred), km
