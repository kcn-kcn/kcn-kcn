import streamlit as st
import numpy as np
import base64
from datetime import datetime
import warnings

from src.config import CORPORATE_COLORS, apply_custom_css
from src.security import SecurityManager
from src.data_processing import DataProcessor
from src.models import HybridFraudDetector
from src.reporting import ReportGenerator
from src.visualizations import (
    create_dashboard,
    create_explanation_chart,
    create_performance_dashboard
)

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Hybrid Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply styling
apply_custom_css()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""

    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'security' not in st.session_state:
        st.session_state.security = SecurityManager()

    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>🔒 Hybrid Fraud Detection System</h1>
        <p>K-Means + DBSCAN | Zimbabwe Banking Sector</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown(f"<h3 style='color:{CORPORATE_COLORS['primary']}'>Navigation</h3>",
                    unsafe_allow_html=True)

        if not st.session_state.authenticated:
            option = st.radio("Select Option", ["Login", "About"])

            if option == "Login":
                st.markdown(f"<h3 style='color:{CORPORATE_COLORS['primary']}'>Login</h3>",
                            unsafe_allow_html=True)
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")

                if st.button("Login", use_container_width=True):
                    if st.session_state.login_attempts >= 3:
                        st.error("Maximum login attempts exceeded. Please try again later.")
                        st.stop()

                    if st.session_state.security.authenticate(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success(f"Welcome, {username}!")
                        st.rerun()
                    else:
                        st.session_state.login_attempts += 1
                        remaining = 3 - st.session_state.login_attempts
                        st.error(f"Invalid credentials. {remaining} attempts remaining.")

            else:
                st.markdown(f"""
                <div class="card">
                    <h4>System Overview</h4>
                    <p>This Hybrid Fraud Detection System combines:</p>
                    <ul>
                        <li><strong>K-Means Clustering</strong> - Pattern recognition and transaction archetypes</li>
                        <li><strong>DBSCAN</strong> - Anomaly detection and noise identification</li>
                    </ul>
                    <p><strong>Features:</strong></p>
                    <ul>
                        <li>CSV file processing</li>
                        <li>Real-time fraud detection</li>
                        <li>Comprehensive reporting</li>
                        <li>Performance metrics</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown(f"<p>Welcome, <strong>{st.session_state.username}</strong>!</p>",
                        unsafe_allow_html=True)

            if st.button("📤 Upload CSV", use_container_width=True):
                st.session_state.data_loaded = False
                st.session_state.analysis_complete = False

            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.login_attempts = 0
                st.rerun()

    # Main content area
    if st.session_state.authenticated:

        # File Upload Section
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📁 Data Upload")

        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

        if uploaded_file is not None:
            # Validate file
            is_valid, message = DataProcessor.validate_file(uploaded_file)

            if not is_valid:
                st.error(f"❌ {message}")
            else:
                st.success(f"✅ {message}")

                # Load data
                df, error = DataProcessor.load_data(uploaded_file)

                if error:
                    st.error(f"Error loading file: {error}")
                else:
                    st.session_state.df = df
                    st.session_state.data_loaded = True

                    # Display data preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)

                    st.info(f"📊 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

                    # Preprocess data
                    with st.spinner("Preprocessing data..."):
                        df_processed, feature_cols, label_encoders = DataProcessor.preprocess_data(df)
                        st.session_state.df_processed = df_processed
                        st.session_state.feature_cols = feature_cols
                        st.session_state.label_encoders = label_encoders

                    st.success("✅ Data preprocessing completed!")

                    # Parameter Configuration
                    st.subheader("⚙️ Model Configuration")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        kmeans_clusters = st.slider("K-Means Clusters", 3, 10, 5)

                    with col2:
                        dbscan_eps = st.slider("DBSCAN Epsilon (eps)", 0.1, 2.0, 0.5, 0.05)

                    with col3:
                        dbscan_min_samples = st.slider("DBSCAN Min Samples", 5, 30, 10)

                    # Run Analysis Button
                    if st.button("🔍 Detect Fraud", type="primary", use_container_width=True):
                        with st.spinner("Analyzing transactions..."):
                            # Prepare features
                            X = st.session_state.df_processed[st.session_state.feature_cols].values

                            # Initialize and train model
                            detector = HybridFraudDetector(
                                kmeans_n_clusters=kmeans_clusters,
                                dbscan_eps=dbscan_eps,
                                dbscan_min_samples=dbscan_min_samples
                            )

                            # Train and predict
                            metrics, kmeans_labels, dbscan_labels = detector.train(X, st.session_state.feature_cols)

                            # Get predictions
                            kmeans_pred, dbscan_pred, risk_scores = detector.predict(X)

                            # Store results
                            st.session_state.detector = detector
                            st.session_state.metrics = metrics
                            st.session_state.kmeans_labels = kmeans_labels
                            st.session_state.dbscan_labels = dbscan_labels
                            st.session_state.risk_scores = risk_scores
                            st.session_state.analysis_complete = True

                            st.success("✅ Analysis complete!")

        st.markdown("</div>", unsafe_allow_html=True)

        # Results Display Section
        if st.session_state.get('analysis_complete', False):
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📊 Fraud Detection Results")

            # Summary Statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>High Risk</h3>
                    <h2>{(st.session_state.risk_scores > 0.6).sum()}</h2>
                    <p>Transactions</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Medium Risk</h3>
                    <h2>{((st.session_state.risk_scores > 0.3) & (st.session_state.risk_scores <= 0.6)).sum()}</h2>
                    <p>Transactions</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Low Risk</h3>
                    <h2>{(st.session_state.risk_scores <= 0.3).sum()}</h2>
                    <p>Transactions</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                noise_pct = st.session_state.metrics.get('noise_pct', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>DBSCAN Noise</h3>
                    <h2>{st.session_state.metrics.get('n_noise', 0)}</h2>
                    <p>({noise_pct:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)

            # Fraud Alerts
            high_risk_indices = np.where(st.session_state.risk_scores > 0.6)[0]
            if len(high_risk_indices) > 0:
                st.markdown(f"""
                <div class="fraud-alert">
                    <h3>⚠️ FRAUD ALERTS</h3>
                    <p>{len(high_risk_indices)} transactions flagged as high-risk fraud suspects!</p>
                </div>
                """, unsafe_allow_html=True)

                # Show high risk transactions
                with st.expander("View High-Risk Transactions"):
                    high_risk_df = st.session_state.df.iloc[high_risk_indices].copy()
                    high_risk_df['risk_score'] = st.session_state.risk_scores[high_risk_indices]
                    high_risk_df['dbscan_cluster'] = st.session_state.dbscan_labels[high_risk_indices]
                    st.dataframe(high_risk_df.head(20), use_container_width=True)

            # Visualizations
            st.subheader("📈 Visual Analytics")

            # Main Dashboard
            st.plotly_chart(create_dashboard(
                st.session_state.df,
                st.session_state.dbscan_labels,
                st.session_state.risk_scores,
                st.session_state.metrics
            ), use_container_width=True)

            # Explanation Charts
            st.subheader("🔍 Fraud Detection Explanations")
            fig_explanation = create_explanation_chart(
                st.session_state.risk_scores,
                st.session_state.dbscan_labels,
                st.session_state.df
            )
            st.pyplot(fig_explanation)

            # Performance Metrics
            st.subheader("📊 Model Performance Metrics")
            st.plotly_chart(create_performance_dashboard(st.session_state.metrics),
                           use_container_width=True)

            # Explanation of Fraud Detection Logic
            st.subheader("📝 Why Transactions Are Flagged")

            st.markdown(f"""
            <div class="card">
                <h4>Detection Methodology</h4>
                <p>The system uses a <strong>hybrid approach</strong> combining K-Means and DBSCAN:</p>

                <h5>1. K-Means Clustering</h5>
                <ul>
                    <li><strong>Silhouette Score:</strong> {st.session_state.metrics.get('kmeans_silhouette', 0):.4f}
                        {'(Good separation)' if st.session_state.metrics.get('kmeans_silhouette', 0) > 0.5 else '(Moderate separation)'}</li>
                    <li><strong>Davies-Bouldin Index:</strong> {st.session_state.metrics.get('kmeans_db_index', 0):.4f}
                        {'(Low similarity - good)' if st.session_state.metrics.get('kmeans_db_index', 0) < 0.7 else '(High similarity - poor)'}</li>
                    <li>Groups transactions into behavioral archetypes</li>
                    <li>Identifies transactions that deviate from their expected cluster</li>
                </ul>

                <h5>2. DBSCAN (Density-Based Clustering)</h5>
                <ul>
                    <li><strong>Silhouette Score:</strong> {st.session_state.metrics.get('dbscan_silhouette', 0):.4f}</li>
                    <li><strong>Noise Points Detected:</strong> {st.session_state.metrics.get('n_noise', 0)} ({st.session_state.metrics.get('noise_pct', 0):.1f}%)</li>
                    <li>Identifies outliers that don't belong to any dense cluster</li>
                    <li>Flags these noise points as potential fraud suspects</li>
                </ul>

                <h5>3. Hybrid Risk Scoring</h5>
                <ul>
                    <li>K-Means Risk: Based on distance to nearest cluster center (40% weight)</li>
                    <li>DBSCAN Risk: Noise points automatically high risk (60% weight)</li>
                    <li><strong>High Risk Threshold:</strong> > 0.6</li>
                    <li><strong>Medium Risk Threshold:</strong> 0.3 - 0.6</li>
                    <li><strong>Low Risk Threshold:</strong> {'<'} 0.3</li>
                </ul>

                <h5>Key Fraud Indicators</h5>
                <ul>
                    <li><strong>Unusual Amounts:</strong> Transactions significantly different from typical patterns</li>
                    <li><strong>Distance Anomalies:</strong> Transactions far from home location</li>
                    <li><strong>Time Irregularities:</strong> Transactions at unusual hours</li>
                    <li><strong>Type Mismatches:</strong> Transaction types inconsistent with user profile</li>
                    <li><strong>Isolation:</strong> Transactions identified as noise by DBSCAN</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Report Generation
            if st.button("📄 Generate Forensic Report", type="primary", use_container_width=True):
                with st.spinner("Generating report..."):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report = ReportGenerator.generate_report(
                        st.session_state.df,
                        st.session_state.dbscan_labels,
                        st.session_state.risk_scores,
                        st.session_state.metrics,
                        timestamp
                    )

                    # Save report
                    filename = f"fraud_report_{timestamp}.json"
                    ReportGenerator.save_report(report, filename)

                    # Provide download link
                    with open(filename, 'r') as f:
                        report_json = f.read()

                    b64 = base64.b64encode(report_json.encode()).decode()
                    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">📥 Download Report (JSON)</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    st.success(f"Report saved as {filename}")

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Not authenticated - show login prompt
        st.info("👋 Please login from the sidebar to access the fraud detection system.")

if __name__ == "__main__":
    main()
