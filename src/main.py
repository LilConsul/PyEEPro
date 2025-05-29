import time
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from app.config import setup_page_custom_css
from app.sidebar import render_sidebar
from app.tabs.eda import render_eda_tab
from app.tabs.not_found import render_not_found_tab

from data.dummy_data import generate_demo_data


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
        }
    
    # Force check dataset on first run
    if "first_run" not in st.session_state:
        st.session_state.first_run = True


def handle_dataset_availability():
    """Display appropriate messages based on dataset availability"""
    from scripts.dataset_exists import check_dataset_exists
    
    # Force check on first run
    if st.session_state.get("first_run", True):
        st.session_state.first_run = False
        
        # Create an expandable section for details
        with st.expander("Dataset Initialization", expanded=True):
            st.info("Checking for dataset...")
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            def update_progress(progress, message):
                progress_bar.progress(progress)
                status_text.text(message)
            
            # Check for dataset with progress reporting
            dataset_available, message = check_dataset_exists(update_progress)
            
            # Update status
            st.session_state.dataset_status["available"] = dataset_available
            st.session_state.dataset_status["message"] = message
            st.session_state.dataset_status["last_check_time"] = time.time()
            st.session_state.dataset_status["check_count"] += 1
            
            # Display final status
            if dataset_available:
                st.success(f"Dataset check complete: {message}")
            else:
                st.error(f"Dataset check failed: {message}")
    
    # Continue with normal display
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

        # Get project root to show absolute paths - correct the path to point to data at project root
        project_root = Path(__file__).resolve().parent.parent  # Go up two levels
        data_dir = project_root / "data"
        zip_path = data_dir / "smart-meters-in-london.zip"

        dataset_status_container.error(
            f"**Dataset Not Found**: {status['message']}\n\n"
            f"To fix this:\n"
            f"1. [Download](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london) the Smart Meters in London dataset from Kaggle\n"
            f"2. Place the ZIP file at **{zip_path}**\n\n"
            f"*Status: Last check at {last_check} (Attempt #{status['check_count']})*",
            icon="❌",
        )

        # Add data directory info
        if data_dir.exists():
            st.info(f"Data directory exists at {data_dir}")
            files = list(data_dir.glob("*.zip"))
            if files:
                st.info(f"Found ZIP files in data directory: {', '.join(f.name for f in files)}")
        else:
            st.warning(f"Data directory doesn't exist at {data_dir}")
            st.info(f"Creating data directory at {data_dir}")
            data_dir.mkdir(exist_ok=True, parents=True)

        check_button = st.button("Check for dataset now")
        if check_button:
            # Create UI elements for progress display
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            # Define a simpler callback function for progress updates
            def update_progress(progress, message):
                progress_bar.progress(progress)
                status_text.text(message)
            
            # Check for dataset with progress reporting
            dataset_available, message = check_dataset_exists(update_progress)
            
            # Update status
            status["available"] = dataset_available
            status["message"] = message
            status["last_check_time"] = time.time()
            status["check_count"] += 1
            
            # Show final status before refresh
            if dataset_available:
                st.success(f"Dataset check complete: {message}")
            else:
                st.error(f"Dataset check failed: {message}")
            
            # Refresh page after extraction completes
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
