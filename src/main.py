import streamlit as st
from app.config import setup_page_custom_css
from app.sidebar import render_sidebar
from app.tabs.eda import render_eda_tab
from app.tabs.not_found import render_not_found_tab

from data.dummy_data import generate_demo_data
from scripts import handle_dataset_availability


def setup_app():
    """Initialize the Streamlit app with base configuration"""
    st.set_page_config(
        page_title="PyEEPro - Smart Meters Energy Consumption Analysis",
        page_icon="⚡",
        layout="wide",
    )

    setup_page_custom_css()

    st.markdown(
        "<h1>⚡ Smart Meters Energy Consumption Analysis</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("""
        This app analyzes time-based patterns in energy consumption data from Smart Meters in London.
        Upload your data and explore hourly, daily, weekly, and seasonal trends.
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
    # Get user input from sidebar
    uploaded_file, filters = render_sidebar()

    # Use dummy data for now (regardless of whether real data exists)
    data = generate_demo_data()

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Exploratory Data Analysis",
            "🧹 Data Cleaning & Feature Engineering",
            "💡 Interesting Findings",
            "📈 Visualizations",
        ]
    )

    with tab1:
        render_eda_tab(data)
    with tab2:
        render_not_found_tab()
    with tab3:
        render_not_found_tab()
    with tab4:
        render_not_found_tab()


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
