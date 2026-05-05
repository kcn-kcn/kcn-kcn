import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .config import CORPORATE_COLORS

def create_dashboard(df, predictions, risk_scores, metrics):
    """Create comprehensive dashboard visualizations"""

    # Create subplots (using only supported types)
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=('Risk Score Distribution', 'Fraud Risk Level', 'DBSCAN Noise Points',
                        'K-Means Clusters', 'Transaction Amount by Risk', 'Risk by Hour',
                        'Model Performance Comparison', 'Transaction Type Risk', 'Summary Stats'),
        specs=[[{'type': 'histogram'}, {'type': 'pie'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'box'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}, {'type': 'scatter'}]]
    )

    # 1. Risk Score Distribution
    fig.add_trace(
        go.Histogram(x=risk_scores, nbinsx=30, name='Risk Scores',
                     marker_color=CORPORATE_COLORS['primary']),
        row=1, col=1
    )

    # 2. Fraud Risk Level Pie Chart
    risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']
    risk_counts = [
        (risk_scores <= 0.3).sum(),
        ((risk_scores > 0.3) & (risk_scores <= 0.6)).sum(),
        (risk_scores > 0.6).sum()
    ]
    fig.add_trace(
        go.Pie(labels=risk_levels, values=risk_counts,
               marker_colors=[CORPORATE_COLORS['success'], CORPORATE_COLORS['warning'], CORPORATE_COLORS['danger']]),
        row=1, col=2
    )

    # 3. DBSCAN Noise Points
    noise_count = (predictions == -1).sum()
    non_noise_count = len(predictions) - noise_count
    fig.add_trace(
        go.Bar(x=['Normal Points', 'Noise (Suspicious)'],
               y=[non_noise_count, noise_count],
               marker_color=[CORPORATE_COLORS['success'], CORPORATE_COLORS['danger']]),
        row=1, col=3
    )

    # 4. K-Means Cluster Distribution
    unique_clusters, cluster_counts = np.unique(predictions[predictions != -1], return_counts=True)
    if len(unique_clusters) > 0:
        fig.add_trace(
            go.Bar(x=[f'Cluster {c}' for c in unique_clusters[:10]], y=cluster_counts[:10],
                   marker_color=CORPORATE_COLORS['secondary']),
            row=2, col=1
        )

    # 5. Transaction Amount vs Risk
    if 'transaction_amount' in df.columns:
        sample_size = min(5000, len(df))
        sample_idx = np.random.choice(len(df), sample_size, replace=False)
        fig.add_trace(
            go.Scatter(x=df.iloc[sample_idx]['transaction_amount'],
                       y=risk_scores[sample_idx],
                       mode='markers',
                       marker=dict(size=5, color=risk_scores[sample_idx],
                                  colorscale='RdYlGn_r', showscale=True),
                       name='Transactions'),
            row=2, col=2
        )

    # 6. Risk by Hour
    if 'hour' in df.columns:
        hour_risk = []
        for hour in range(24):
            mask = df['hour'] == hour
            if mask.sum() > 0:
                hour_risk.append(risk_scores[mask].mean())
            else:
                hour_risk.append(0)
        fig.add_trace(
            go.Bar(x=list(range(24)), y=hour_risk,
                   marker_color=CORPORATE_COLORS['accent']),
            row=2, col=3
        )

    # 7. Model Performance Comparison
    fig.add_trace(
        go.Bar(x=['Silhouette', 'Davies-Bouldin'],
               y=[metrics.get('kmeans_silhouette', 0), metrics.get('kmeans_db_index', 0)],
               name='K-Means', marker_color=CORPORATE_COLORS['primary']),
        row=3, col=1
    )
    fig.add_trace(
        go.Bar(x=['Silhouette', 'Davies-Bouldin'],
               y=[metrics.get('dbscan_silhouette', 0), metrics.get('dbscan_db_index', 0)],
               name='DBSCAN', marker_color=CORPORATE_COLORS['accent']),
        row=3, col=1
    )

    # 8. Transaction Type Risk
    if 'transaction_type' in df.columns:
        type_risk = {}
        for ttype in df['transaction_type'].unique():
            mask = df['transaction_type'] == ttype
            if mask.sum() > 0:
                type_risk[ttype] = risk_scores[mask].mean()
        type_risk_sorted = dict(sorted(type_risk.items(), key=lambda x: x[1], reverse=True))
        fig.add_trace(
            go.Bar(x=list(type_risk_sorted.values())[:10],
                   y=list(type_risk_sorted.keys())[:10],
                   orientation='h',
                   marker_color=CORPORATE_COLORS['warning']),
            row=3, col=2
        )

    # 9. Summary Stats (Risk Score Distribution)
    fig.add_trace(
        go.Box(y=risk_scores, name='Risk Score Distribution',
               marker_color=CORPORATE_COLORS['primary']),
        row=3, col=3
    )

    fig.update_layout(
        title_text="Hybrid Fraud Detection Dashboard",
        showlegend=True,
        height=1000,
        template='plotly_white'
    )

    return fig


def create_explanation_chart(risk_scores, predictions, df):
    """Create explanation of fraud detection"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fraud Detection Analysis & Explanations', fontsize=16, fontweight='bold')

    # 1. Risk Score Interpretation
    axes[0, 0].hist(risk_scores, bins=30, color=CORPORATE_COLORS['primary'],
                    edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=0.3, color=CORPORATE_COLORS['warning'], linestyle='--',
                       linewidth=2, label='Medium Risk Threshold')
    axes[0, 0].axvline(x=0.6, color=CORPORATE_COLORS['danger'], linestyle='--',
                       linewidth=2, label='High Risk Threshold')
    axes[0, 0].set_xlabel('Risk Score')
    axes[0, 0].set_ylabel('Number of Transactions')
    axes[0, 0].set_title('Risk Score Distribution with Thresholds')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. DBSCAN Noise Explanation
    noise_mask = predictions == -1
    noise_risks = risk_scores[noise_mask] if noise_mask.sum() > 0 else [0]
    non_noise_risks = risk_scores[~noise_mask] if (~noise_mask).sum() > 0 else [0]

    axes[0, 1].boxplot([non_noise_risks, noise_risks],
                        labels=['Normal Clusters', 'DBSCAN Noise Points'],
                        patch_artist=True,
                        boxprops=dict(facecolor=CORPORATE_COLORS['light']),
                        medianprops=dict(color=CORPORATE_COLORS['primary'], linewidth=2))
    axes[0, 1].set_ylabel('Risk Score')
    axes[0, 1].set_title('DBSCAN: Noise Points vs Normal Clusters')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Feature Importance (based on risk correlation)
    if 'transaction_amount' in df.columns:
        risk_corr = {}
        risk_corr['Amount'] = np.corrcoef(df['transaction_amount'].fillna(0), risk_scores)[0, 1] if len(df) > 1 else 0
        if 'distance_from_home' in df.columns:
            risk_corr['Distance'] = np.corrcoef(df['distance_from_home'].fillna(0), risk_scores)[0, 1] if len(df) > 1 else 0
        if 'hour' in df.columns:
            risk_corr['Hour'] = np.corrcoef(df['hour'].fillna(0), risk_scores)[0, 1] if len(df) > 1 else 0

        risk_corr_abs = {k: abs(v) for k, v in risk_corr.items()}
        risk_corr_sorted = dict(sorted(risk_corr_abs.items(), key=lambda x: x[1]))

        axes[1, 0].barh(list(risk_corr_sorted.keys()), list(risk_corr_sorted.values()),
                        color=CORPORATE_COLORS['accent'])
        axes[1, 0].set_xlabel('Correlation with Risk Score')
        axes[1, 0].set_title('Feature Impact on Fraud Detection')
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Transaction Amount Analysis
    if 'transaction_amount' in df.columns:
        high_risk_amounts = df.loc[risk_scores > 0.6, 'transaction_amount'] if (risk_scores > 0.6).sum() > 0 else [0]
        medium_risk_amounts = df.loc[(risk_scores > 0.3) & (risk_scores <= 0.6), 'transaction_amount'] if ((risk_scores > 0.3) & (risk_scores <= 0.6)).sum() > 0 else [0]
        low_risk_amounts = df.loc[risk_scores <= 0.3, 'transaction_amount'] if (risk_scores <= 0.3).sum() > 0 else [0]

        axes[1, 1].hist([low_risk_amounts, medium_risk_amounts, high_risk_amounts],
                         bins=30, label=['Low Risk', 'Medium Risk', 'High Risk'],
                         color=[CORPORATE_COLORS['success'], CORPORATE_COLORS['warning'],
                                CORPORATE_COLORS['danger']], alpha=0.6)
        axes[1, 1].set_xlabel('Transaction Amount')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Transaction Amount Distribution by Risk Level')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xscale('log')

    plt.tight_layout()
    return fig


def create_performance_dashboard(metrics):
    """Create performance metrics dashboard without gauge type"""

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('K-Means Performance', 'DBSCAN Performance',
                        'Cluster Quality Comparison', 'Noise Detection Rate'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'indicator'}]]
    )

    # K-Means Metrics Bar Chart
    fig.add_trace(
        go.Bar(x=['Silhouette Score', 'Davies-Bouldin'],
               y=[metrics.get('kmeans_silhouette', 0), metrics.get('kmeans_db_index', 0)],
               name='K-Means',
               marker_color=CORPORATE_COLORS['primary']),
        row=1, col=1
    )

    # DBSCAN Metrics Bar Chart
    fig.add_trace(
        go.Bar(x=['Silhouette Score', 'Davies-Bouldin'],
               y=[metrics.get('dbscan_silhouette', 0), metrics.get('dbscan_db_index', 0)],
               name='DBSCAN',
               marker_color=CORPORATE_COLORS['accent']),
        row=1, col=2
    )

    # Comparison Bar Chart
    fig.add_trace(
        go.Bar(x=['K-Means Silhouette', 'DBSCAN Silhouette'],
               y=[metrics.get('kmeans_silhouette', 0), metrics.get('dbscan_silhouette', 0)],
               name='Silhouette Score',
               marker_color=CORPORATE_COLORS['primary']),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=['K-Means DBI', 'DBSCAN DBI'],
               y=[metrics.get('kmeans_db_index', 0), metrics.get('dbscan_db_index', 0)],
               name='Davies-Bouldin Index',
               marker_color=CORPORATE_COLORS['danger']),
        row=2, col=1
    )

    # Noise Detection Rate (using gauge from go.Indicator)
    noise_pct = metrics.get('noise_pct', 0)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=noise_pct,
            title={"text": "Suspicious Transaction Rate (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': CORPORATE_COLORS['warning']},
                'steps': [
                    {'range': [0, 5], 'color': CORPORATE_COLORS['success']},
                    {'range': [5, 15], 'color': CORPORATE_COLORS['warning']},
                    {'range': [15, 100], 'color': CORPORATE_COLORS['danger']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': noise_pct
                }
            }
        ),
        row=2, col=2
    )

    fig.update_layout(height=600, showlegend=True, template='plotly_white')

    return fig
