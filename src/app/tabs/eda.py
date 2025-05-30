import streamlit as st
import polars as pl
from data import storage


def render_eda_tab(filters):
    st.header("📊 Exploratory Data Analysis")

    time_based_tab, household_tab, weather_tab = st.tabs(
        ["📈 Time-based trends", "📊 Household behavior", "📉 Weather impact"]
    )

    with time_based_tab:
        st.subheader("Time-based Energy Consumption Patterns")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Hourly Patterns")
            hourly_data = storage.get_hourly_patterns(
                years=filters.get("years", None),
            )
            st.dataframe(hourly_data)

        with col2:
            st.markdown("### Daily Patterns")

        with st.expander("Weekday vs Weekend Comparison"):
            st.markdown("### Weekday/Weekend Comparison")

        with st.expander("Seasonal Variations"):
            st.markdown("### Seasonal Variations")
