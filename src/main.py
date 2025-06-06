import streamlit as st
from app.sidebar import render_sidebar
from app.tabs import render_household_tab, render_weather_tab, render_time_based_tab, render_ml_tab
from scripts import handle_dataset_availability


def setup_app():
    """Initialize the Streamlit app with base configuration"""
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
        Use the sidebar to select filters and explore the data interactively.
    """)


def initialize_session_state():
    """Initialize all session state variables needed for the application"""
    if "dataset_status" not in st.session_state:
        st.session_state.dataset_status = {
            "available": False,
            "last_check_time": 0,
            "check_count": 0,
            "message": "",
            "initialization_complete": False,
        }

    if "first_run" not in st.session_state:
        st.session_state.first_run = True


def render_app_content():
    """Render the main application content"""
    render_sidebar()

    # st.header("📊 Exploratory Data Analysis")
    time_based_tab, household_tab, weather_tab, ml_tab = st.tabs(
        ["📈 Time-based trends", "📊 Household behavior", "📉 Weather impact", "🤖 Machine Learning"]
    )

    with time_based_tab:
        render_time_based_tab()

    with household_tab:
        render_household_tab()

    with weather_tab:
        render_weather_tab()

    with ml_tab:
        render_ml_tab()


def main():
    """Main application entry point with improved logical flow"""
    # Initialize application UI
    setup_app()

    initialize_session_state()

    if handle_dataset_availability():
        render_app_content()
    else:
        st.stop()


if __name__ == "__main__":
    main()
