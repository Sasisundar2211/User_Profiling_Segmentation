"""Data preprocessing and feature scaling utilities for clustering pipeline."""

from __future__ import annotations

from typing import Any, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_FEATURES = [
    "Time Spent Online (hrs/weekday)",
    "Time Spent Online (hrs/weekend)",
    "Click-Through Rates (CTR)",
    "Conversion Rates",
    "Ad Interaction Time (sec)",
    "engagement_score",
    "ad_responsiveness",
]

CATEGORICAL_FEATURES = [
    "Age",
    "Gender",
    "Income Level",
    "Education Level",
    "Device Usage",
]


def preprocess_data(
    df: pd.DataFrame,
) -> Tuple[Any, ColumnTransformer]:
    """Applies Standard Scaling to numeric features and One-Hot Encoding to categorical features.

    Args:
        df: Input DataFrame with features.

    Returns:
        Tuple containing the transformed feature matrix (X) and fitted ColumnTransformer.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty. Preprocessing aborted.")

    existing_num = [col for col in NUMERICAL_FEATURES if col in df.columns]
    existing_cat = [col for col in CATEGORICAL_FEATURES if col in df.columns]

    if not existing_num and not existing_cat:
        raise ValueError("No matching numerical or categorical features found in input DataFrame.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), existing_num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), existing_cat),
        ],
        remainder="drop",
    )

    X_transformed = preprocessor.fit_transform(df)
    return X_transformed, preprocessor
