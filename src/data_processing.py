import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    """Handles data loading, validation, and preprocessing"""

    @staticmethod
    def validate_file(uploaded_file):
        """Validate file format"""
        if uploaded_file is None:
            return False, "No file uploaded"

        if not uploaded_file.name.endswith('.csv'):
            return False, "Incorrect format! Please upload a CSV file only."

        return True, "File validated successfully"

    @staticmethod
    def load_data(uploaded_file):
        """Load CSV data"""
        try:
            df = pd.read_csv(uploaded_file)
            return df, None
        except Exception as e:
            return None, str(e)

    @staticmethod
    def preprocess_data(df):
        """Preprocess data for clustering"""
        df_processed = df.copy()

        # Handle missing values
        df_processed = df_processed.dropna()

        # Create time-based features
        if 'time' in df_processed.columns:
            # Convert time string to hour
            if df_processed['time'].dtype == 'object':
                try:
                    df_processed['hour'] = pd.to_datetime(df_processed['time'], format='%H:%M').dt.hour
                except:
                    df_processed['hour'] = df_processed['time']
            else:
                df_processed['hour'] = df_processed['time']
            df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['hour']/24)
            df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['hour']/24)

        # Log transform amount
        if 'transaction_amount' in df_processed.columns:
            df_processed['log_amount'] = np.log1p(df_processed['transaction_amount'])

        # Encode categorical variables
        categorical_cols = ['transaction_type', 'merchant_category', 'location']
        label_encoders = {}
        for col in categorical_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le

        # Select features for clustering
        feature_cols = []
        if 'log_amount' in df_processed.columns:
            feature_cols.append('log_amount')
        if 'distance_from_home' in df_processed.columns:
            feature_cols.append('distance_from_home')
        if 'hour_sin' in df_processed.columns:
            feature_cols.append('hour_sin')
            feature_cols.append('hour_cos')
        if 'transaction_type_encoded' in df_processed.columns:
            feature_cols.append('transaction_type_encoded')
        if 'merchant_category_encoded' in df_processed.columns:
            feature_cols.append('merchant_category_encoded')
        if 'location_encoded' in df_processed.columns:
            feature_cols.append('location_encoded')

        return df_processed, feature_cols, label_encoders
