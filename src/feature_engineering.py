"""Feature engineering pipeline for user behavioral data."""

from __future__ import annotations

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers synthetic behavioral features for segmentation modeling.

    Features generated:
    - engagement_score: Average time spent online across weekday and weekend.
    - ad_responsiveness: Product of Click-Through Rate (CTR) and Conversion Rate.

    Args:
        df: Input DataFrame containing user behavior metrics.

    Returns:
        pd.DataFrame: Copy of input DataFrame enriched with engineered features.
    """
    if df.empty:
        raise ValueError("Cannot perform feature engineering on an empty DataFrame.")

    df_engineered = df.copy()

    if (
        "Time Spent Online (hrs/weekday)" in df_engineered.columns
        and "Time Spent Online (hrs/weekend)" in df_engineered.columns
    ):
        df_engineered["engagement_score"] = (
            df_engineered["Time Spent Online (hrs/weekday)"]
            + df_engineered["Time Spent Online (hrs/weekend)"]
        ) / 2.0
    else:
        df_engineered["engagement_score"] = 0.0

    if (
        "Click-Through Rates (CTR)" in df_engineered.columns
        and "Conversion Rates" in df_engineered.columns
    ):
        df_engineered["ad_responsiveness"] = (
            df_engineered["Click-Through Rates (CTR)"]
            * df_engineered["Conversion Rates"]
        )
    else:
        df_engineered["ad_responsiveness"] = 0.0

    return df_engineered
