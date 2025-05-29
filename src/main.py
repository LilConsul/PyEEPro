import streamlit as st
from app.config import setup_page_custom_css
from app.sidebar import render_sidebar
from app.tabs.eda import render_eda_tab
from app.tabs.not_found import render_not_found_tab

from data.dummy_data import generate_demo_data


def main():
    st.set_page_config(
        page_title="PyEEPro - Smart Meters Energy Consumption Analysis",
        page_icon="⚡",
        layout="wide",
    )
    st.markdown(
        "<h1>⚡ Smart Meters Energy Consumption Analysis</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("""
        This app analyzes time-based patterns in energy consumption data from Smart Meters in London.
        Upload your data and explore hourly, daily, weekly, and seasonal trends.
    """)
    setup_page_custom_css()

    uploaded_file, filters = render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Exploratory Data Analysis",
            "🧹 Data Cleaning & Feature Engineering",
            "💡 Interesting Findings",
            "📈 Visualizations",
        ]
    )

    with tab1:
        render_eda_tab(generate_demo_data())
    with tab2:
        render_not_found_tab()
    with tab3:
        render_not_found_tab()
    with tab4:
        render_not_found_tab()


if __name__ == "__main__":
    main()
