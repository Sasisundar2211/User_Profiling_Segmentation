"""K-Means clustering module for segmenting user profiles."""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
from sklearn.cluster import KMeans


def cluster_users(
    X: Any,
    k: int = 4,
    save_model: bool = False,
    model_path: str = "models/clustering_model.pkl",
) -> np.ndarray:
    """Executes K-Means clustering algorithm on preprocessed feature matrix.

    Args:
        X: Preprocessed numerical feature matrix.
        k: Number of clusters requested (default 4).
        save_model: If True, attempts to save trained model to model_path.
        model_path: Output filepath for model persistence.

    Returns:
        np.ndarray: Array of cluster assignment labels.
    """
    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("Feature matrix X cannot be empty.")

    effective_k = min(k, n_samples)
    if effective_k < 1:
        effective_k = 1

    kmeans = KMeans(n_clusters=effective_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    if save_model:
        try:
            folder = os.path.dirname(model_path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            joblib.dump(kmeans, model_path)
        except Exception:
            # Serverless compatibility check for read-only filesystem
            pass

    return labels
