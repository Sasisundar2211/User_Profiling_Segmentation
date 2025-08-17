
def profile_segments(df, labels):
    df['segment'] = labels
    profiles = df.groupby('segment').mean(numeric_only=True).to_dict(orient='index')
    return profiles
