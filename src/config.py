import streamlit as st

# Corporate colors
CORPORATE_COLORS = {
    'primary': '#1a237e',      # Dark Blue
    'secondary': '#0d47a1',    # Blue
    'accent': '#00bcd4',       # Cyan
    'success': '#4caf50',      # Green
    'warning': '#ff9800',       # Orange
    'danger': '#f44336',        # Red
    'dark': '#263238',          # Dark Grey
    'light': '#eceff1',         # Light Grey
    'white': '#ffffff',
    'fraud': '#d32f2f',         # Dark Red
    'normal': '#388e3c'         # Dark Green
}

def apply_custom_css():
    """Apply custom CSS for styling"""
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {CORPORATE_COLORS['light']};
        }}
        .main-header {{
            background-color: {CORPORATE_COLORS['primary']};
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }}
        .card {{
            background-color: {CORPORATE_COLORS['white']};
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }}
        .metric-card {{
            background: linear-gradient(135deg, {CORPORATE_COLORS['primary']}, {CORPORATE_COLORS['secondary']});
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
        }}
        .fraud-alert {{
            background-color: {CORPORATE_COLORS['danger']};
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }}
        .normal-alert {{
            background-color: {CORPORATE_COLORS['success']};
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }}
    </style>
    """, unsafe_allow_html=True)
