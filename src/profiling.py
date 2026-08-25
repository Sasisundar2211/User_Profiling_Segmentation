"""Profiling module to aggregate segment characteristics."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def profile_segments(df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Calculates summary statistics (mean values for numerical features) for each user segment.

    Args:
        df: Input DataFrame containing original and engineered features.
        labels: Cluster assignment labels corresponding to DataFrame rows.

    Returns:
        Dict mapping segment ID to dictionary of feature averages.
    """
    if len(df) != len(labels):
        raise ValueError(f"Length mismatch: df length ({len(df)}) != labels length ({len(labels)})")

    df_profile = df.copy()
    df_profile["segment"] = labels

    numeric_df = df_profile.select_dtypes(include=[np.number])

    profiles_raw = numeric_df.groupby("segment").mean().to_dict(orient="index")

    profiles: Dict[str, Dict[str, Any]] = {}
    for segment_id, values in profiles_raw.items():
        profiles[str(segment_id)] = {
            feature: float(round(val, 4)) if isinstance(val, (float, int, np.floating, np.integer)) else val
            for feature, val in values.items()
        }

    return profiles
