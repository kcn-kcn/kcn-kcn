import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

class HybridFraudDetector:
    """Hybrid model combining K-Means and DBSCAN"""

    def __init__(self, kmeans_n_clusters=5, dbscan_eps=0.5, dbscan_min_samples=10):
        self.kmeans_n_clusters = kmeans_n_clusters
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.kmeans = None
        self.dbscan = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.cluster_profiles = {}

    def train(self, X, feature_names):
        """Train both models"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train K-Means
        self.kmeans = KMeans(n_clusters=self.kmeans_n_clusters, random_state=42, n_init=10)
        kmeans_labels = self.kmeans.fit_predict(X_scaled)

        # Train DBSCAN
        self.dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        dbscan_labels = self.dbscan.fit_predict(X_scaled)

        # Calculate metrics
        metrics = {}
        metrics['kmeans_silhouette'] = silhouette_score(X_scaled, kmeans_labels)
        metrics['kmeans_db_index'] = davies_bouldin_score(X_scaled, kmeans_labels)

        if len(set(dbscan_labels)) > 1:
            metrics['dbscan_silhouette'] = silhouette_score(X_scaled, dbscan_labels)
            metrics['dbscan_db_index'] = davies_bouldin_score(X_scaled, dbscan_labels)
        else:
            metrics['dbscan_silhouette'] = 0
            metrics['dbscan_db_index'] = float('inf')

        metrics['n_clusters_dbscan'] = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        metrics['n_noise'] = list(dbscan_labels).count(-1)
        metrics['noise_pct'] = (metrics['n_noise'] / len(dbscan_labels)) * 100

        # Create cluster profiles
        for cluster in range(self.kmeans_n_clusters):
            mask = kmeans_labels == cluster
            if mask.sum() > 0:
                self.cluster_profiles[f'KMeans_Cluster_{cluster}'] = {
                    'size': mask.sum(),
                    'avg_amount': X[mask, 0].mean() if X.shape[1] > 0 else 0,
                    'avg_distance': X[mask, 1].mean() if X.shape[1] > 1 else 0
                }

        noise_mask = dbscan_labels == -1
        self.cluster_profiles['DBSCAN_Noise'] = {
            'size': noise_mask.sum(),
            'pct': (noise_mask.sum() / len(dbscan_labels)) * 100
        }

        self.is_trained = True
        return metrics, kmeans_labels, dbscan_labels

    def predict(self, X):
        """Predict fraud risk for new data"""
        X_scaled = self.scaler.transform(X)

        kmeans_pred = self.kmeans.predict(X_scaled)
        dbscan_pred = self.dbscan.fit_predict(X_scaled)

        # Calculate hybrid risk score
        risk_scores = []
        for i in range(len(X_scaled)):
            # Base risk from DBSCAN (noise = high risk)
            dbscan_risk = 0.8 if dbscan_pred[i] == -1 else 0.2

            # K-Means risk (based on cluster distance)
            distances = self.kmeans.transform(X_scaled[i].reshape(1, -1))
            min_distance = distances.min()
            kmeans_risk = min(1.0, min_distance / 5)  # Normalize distance to risk

            # Hybrid risk (weighted)
            hybrid_risk = 0.6 * dbscan_risk + 0.4 * kmeans_risk
            risk_scores.append(hybrid_risk)

        return kmeans_pred, dbscan_pred, np.array(risk_scores)
