"""Unit tests for machine learning pipeline modules."""

from __future__ import annotations

import pandas as pd
import pytest
import numpy as np

from src.feature_engineering import engineer_features
from src.data_preprocessing import preprocess_data
from src.clustering import cluster_users
from src.profiling import profile_segments


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Age": "25-34", "Gender": "Female", "Income Level": "High",
            "Education Level": "Graduate", "Device Usage": "Mobile",
            "Time Spent Online (hrs/weekday)": 4.0,
            "Time Spent Online (hrs/weekend)": 6.0,
            "Click-Through Rates (CTR)": 0.10,
            "Conversion Rates": 0.05,
            "Ad Interaction Time (sec)": 20,
        },
        {
            "Age": "35-44", "Gender": "Male", "Income Level": "Medium",
            "Education Level": "Undergraduate", "Device Usage": "Desktop",
            "Time Spent Online (hrs/weekday)": 2.0,
            "Time Spent Online (hrs/weekend)": 4.0,
            "Click-Through Rates (CTR)": 0.04,
            "Conversion Rates": 0.02,
            "Ad Interaction Time (sec)": 10,
        },
    ])


def test_engineer_features(sample_raw_dataframe: pd.DataFrame):
    df_engineered = engineer_features(sample_raw_dataframe)
    assert "engagement_score" in df_engineered.columns
    assert "ad_responsiveness" in df_engineered.columns
    assert df_engineered.loc[0, "engagement_score"] == 5.0
    assert pytest.approx(df_engineered.loc[0, "ad_responsiveness"]) == 0.0050


def test_engineer_features_empty():
    with pytest.raises(ValueError, match="Cannot perform feature engineering on an empty DataFrame"):
        engineer_features(pd.DataFrame())


def test_preprocess_data(sample_raw_dataframe: pd.DataFrame):
    df_engineered = engineer_features(sample_raw_dataframe)
    X, preprocessor = preprocess_data(df_engineered)
    assert X.shape[0] == 2
    assert X.shape[1] > 0


def test_preprocess_data_empty():
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        preprocess_data(pd.DataFrame())


def test_cluster_users(sample_raw_dataframe: pd.DataFrame):
    df_engineered = engineer_features(sample_raw_dataframe)
    X, _ = preprocess_data(df_engineered)
    labels = cluster_users(X, k=2)
    assert len(labels) == 2
    assert isinstance(labels, np.ndarray)


def test_profile_segments(sample_raw_dataframe: pd.DataFrame):
    df_engineered = engineer_features(sample_raw_dataframe)
    labels = np.array([0, 1])
    profiles = profile_segments(df_engineered, labels)
    assert "0" in profiles
    assert "1" in profiles
    assert "engagement_score" in profiles["0"]
