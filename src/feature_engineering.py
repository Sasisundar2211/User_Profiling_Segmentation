def engineer_features(df):
    df['engagement_score'] = (
        df['Time Spent Online (hrs/weekday)'] + df['Time Spent Online (hrs/weekend)']
    ) / 2
    df['ad_responsiveness'] = df['Click-Through Rates (CTR)'] * df['Conversion Rates']
    return df
