import time
from datetime import datetime

import streamlit as st
from app.config import setup_page_custom_css
from app.sidebar import render_sidebar
from app.tabs.eda import render_eda_tab
from app.tabs.not_found import render_not_found_tab

from data.dummy_data import generate_demo_data
import threading


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
            "extraction_in_progress": False,
            "extraction_progress": 0.0,
            "extraction_status": "",
        }


def handle_dataset_availability():
    """Display appropriate messages based on dataset availability"""
    dataset_status_container = st.empty()
    status = st.session_state.dataset_status

    if status["available"]:
        dataset_status_container.success(
            "✅ Dataset is available and ready for analysis"
        )
        return True
    else:
        last_check = (
            datetime.fromtimestamp(status["last_check_time"]).strftime("%H:%M:%S")
            if status["last_check_time"] > 0
            else "Not checked yet"
        )

        # Check if extraction is in progress
        if status.get("extraction_in_progress", False):
            with st.spinner("Extracting dataset..."):
                progress_placeholder = st.empty()
                progress_bar = progress_placeholder.progress(status["extraction_progress"])
                status_text = st.empty().text(status["extraction_status"])
                
                # The progress will be updated by the callback
                # Add a check button to force refresh if needed
                if st.button("Force refresh"):
                    st.rerun()
                    
                return False

        dataset_status_container.error(
            f"**Dataset Not Found**: {status['message']}\n\n"
            f"To fix this:\n"
            f"1. [Download](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london) the Smart Meters in London dataset from Kaggle\n"
            f"2. Place the ZIP file at **'data/smart-meters-in-london.zip'**\n\n"
            f"*Status: Last check at {last_check} (Attempt #{status['check_count']})*",
            icon="❌",
        )

        check_button = st.button("Check for dataset now")
        if check_button:
            from scripts.dataset_exists import check_dataset_exists
            
            # Set extraction in progress first
            st.session_state.dataset_status["extraction_in_progress"] = True
            st.session_state.dataset_status["extraction_progress"] = 0.0
            st.session_state.dataset_status["extraction_status"] = "Starting extraction..."
            
            # Create placeholder elements for progress display
            progress_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0.0)
            status_text = st.empty()
            status_text.text("Preparing to extract dataset...")
            
            # Define a callback that will be called from the extraction thread
            def update_progress(progress, status_text_msg):
                try:
                    # Update session state
                    st.session_state.dataset_status["extraction_progress"] = progress
                    st.session_state.dataset_status["extraction_status"] = status_text_msg
                    # Update UI elements
                    progress_bar.progress(progress)
                    status_text.text(status_text_msg)
                except Exception as e:
                    # In case of UI update errors
                    print(f"Error updating UI: {e}")
            
            # Start extraction in a separate thread to avoid blocking the UI
            def extraction_thread():
                try:
                    dataset_available, message = check_dataset_exists(update_progress)
                    
                    # Update status
                    st.session_state.dataset_status["available"] = dataset_available
                    st.session_state.dataset_status["message"] = message
                    st.session_state.dataset_status["last_check_time"] = time.time()
                    st.session_state.dataset_status["check_count"] += 1
                finally:
                    st.session_state.dataset_status["extraction_in_progress"] = False
                    st.rerun()

            thread = threading.Thread(target=extraction_thread)
            thread.daemon = True
            thread.start()
            
            # Show a message while extraction starts
            st.info("Starting extraction process, please wait...")
            st.rerun()

        return False


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

    # Set up session state
    initialize_session_state()

    # Handle dataset availability messaging
    if handle_dataset_availability():
        # Dataset is available, render the app
        render_app_content()
    else:
        # Stop execution if dataset is unavailable
        st.stop()


if __name__ == "__main__":
    main()
