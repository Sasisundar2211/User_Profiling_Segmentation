
from sklearn.cluster import KMeans
import joblib

def cluster_users(X, k=4):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    joblib.dump(kmeans, 'models/clustering_model.pkl')
    return labels
