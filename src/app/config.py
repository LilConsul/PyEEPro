import streamlit as st
from pathlib import Path


def setup_page_custom_css():
    st.markdown(
        """
    <style>
        h1 {
            font-size: 2.5rem; 
            color: #1E88E5; 
        }
        h2 {
            font-size: 1.5rem; 
            color: #424242;
        }
        .card {
            padding: 20px; 
            border-radius: 5px; 
            margin-bottom: 20px;
            background-color: #f5f5f5; 
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-card {
            background-color: #ffffff; 
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); 
            padding: 15px; 
            text-align: center;
        }
        .metric-value {
            font-size: 1.8rem; 
            font-weight: bold; 
            color: #1E88E5;
        }
        .metric-label {
            font-size: 0.9rem; 
            color: #616161;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


class Settings:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "smart-meters-in-london"
    DEBUG = True

    
setting = Settings()
