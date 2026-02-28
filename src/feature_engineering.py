from src.constants import (
    TIME_WEEKDAY_COL,
    TIME_WEEKEND_COL,
    CTR_COL,
    CONVERSION_COL,
    ENGAGEMENT_SCORE_COL,
    AD_RESPONSIVENESS_COL,
)

def engineer_features(df):
    df[ENGAGEMENT_SCORE_COL] = (
        df[TIME_WEEKDAY_COL] + df[TIME_WEEKEND_COL]
    ) / 2
    df[AD_RESPONSIVENESS_COL] = df[CTR_COL] * df[CONVERSION_COL]
    return df
