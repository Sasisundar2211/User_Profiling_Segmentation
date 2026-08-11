import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.constants import NUMERICAL_COLS, CATEGORICAL_COLS

def preprocess_data(df):
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUMERICAL_COLS),
        ('cat', OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS)
    ])

    X = preprocessor.fit_transform(df)
    return X, preprocessor
