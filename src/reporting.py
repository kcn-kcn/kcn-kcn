import json
import numpy as np

class ReportGenerator:
    """Generates and saves fraud detection reports"""

    @staticmethod
    def generate_report(df, predictions, risk_scores, metrics, timestamp):
        """Generate comprehensive report"""
        report = {
            'report_metadata': {
                'generated_at': timestamp,
                'system_version': 'Hybrid Fraud Detection v1.0',
                'total_transactions': len(df)
            },
            'summary_statistics': {
                'total_analyzed': len(predictions),
                'high_risk_transactions': int((risk_scores > 0.6).sum()),
                'medium_risk_transactions': int(((risk_scores > 0.3) & (risk_scores <= 0.6)).sum()),
                'low_risk_transactions': int((risk_scores <= 0.3).sum()),
                'dbscan_noise_points': int((predictions == -1).sum()),
                'fraud_alert_rate': f"{(risk_scores > 0.6).mean() * 100:.2f}%"
            },
            'model_performance': {
                'kmeans_silhouette': metrics.get('kmeans_silhouette', 0),
                'kmeans_davies_bouldin': metrics.get('kmeans_db_index', 0),
                'dbscan_silhouette': metrics.get('dbscan_silhouette', 0),
                'dbscan_davies_bouldin': metrics.get('dbscan_db_index', 0),
                'noise_percentage': metrics.get('noise_pct', 0)
            },
            'high_risk_transactions': []
        }

        # Add high risk transactions
        high_risk_indices = np.where(risk_scores > 0.6)[0][:100]  # Limit to 100
        for idx in high_risk_indices:
            if idx < len(df):
                report['high_risk_transactions'].append({
                    'transaction_id': df.iloc[idx].get('transaction_id', idx),
                    'amount': float(df.iloc[idx].get('transaction_amount', 0)),
                    'risk_score': float(risk_scores[idx]),
                    'dbscan_cluster': int(predictions[idx]) if idx < len(predictions) else -1
                })

        return report

    @staticmethod
    def save_report(report, filename):
        """Save report to file"""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        return filename
