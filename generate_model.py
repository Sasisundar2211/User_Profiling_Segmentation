
import pandas as pd
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.clustering import cluster_users
from src.profiling import profile_segments
import os

# Load dataset
df = pd.read_csv("data/user_profiles_for_ads.csv")

# Feature engineering
df = engineer_features(df)

# Preprocess features
X, _ = preprocess_data(df)

# Make sure models folder exists
os.makedirs("models", exist_ok=True)

# Perform clustering and save model
labels = cluster_users(X, k=4)

# Assign segments and build profiles
profile_segments(df, labels)

# Save segmented output
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/segmented_users.csv", index=False)

print("✅ clustering_model.pkl saved to 'models/'")
print("✅ Segmented users saved to 'outputs/segmented_users.csv'")
