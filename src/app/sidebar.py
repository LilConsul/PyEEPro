import streamlit as st
from datetime import datetime, timedelta


def render_sidebar():
    with st.sidebar:
        st.header("Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your energy consumption data (CSV)", type=["csv"]
        )

        st.markdown("---")
        st.header("Filters")

        date_range = st.date_input(
            "Select Date Range", [datetime.now() - timedelta(days=30), datetime.now()]
        )

        consumer_type = st.multiselect(
            "Consumer Type",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )

        tariff_type = st.selectbox("Tariff Type", ["All", "Standard", "Economy-7"])

        filters = {
            "date_range": date_range,
            "consumer_type": consumer_type,
            "tariff_type": tariff_type,
        }

    return uploaded_file, filters
