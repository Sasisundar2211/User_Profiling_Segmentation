import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(df):
    numerical_cols = [
        'Time Spent Online (hrs/weekday)',
        'Time Spent Online (hrs/weekend)',
        'Click-Through Rates (CTR)',
        'Conversion Rates',
        'Ad Interaction Time (sec)',
        'engagement_score',
        'ad_responsiveness'
    ]

    categorical_cols = [
        'Age',               # ← categorical (e.g., '25-34')
        'Gender',
        'Income Level',      # ← also likely a string like 'High'
        'Education Level',
        'Device Usage'
    ]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    X = preprocessor.fit_transform(df)
    return X, preprocessor
