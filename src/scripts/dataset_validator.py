import time
from datetime import datetime
from pathlib import Path
import streamlit as st

from .dataset_exists import check_dataset_exists


def handle_dataset_availability():
    """Display appropriate messages based on dataset availability"""

    if st.session_state.get(
        "first_run", True
    ) or not st.session_state.dataset_status.get("available", False):
        st.session_state.first_run = False

        if not st.session_state.dataset_status.get("initialization_complete", False):
            init_container = st.container()

            with init_container:
                st.info("Checking for dataset...")
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)

                dataset_available, message = check_dataset_exists(update_progress)

                st.session_state.dataset_status["available"] = dataset_available
                st.session_state.dataset_status["message"] = message
                st.session_state.dataset_status["last_check_time"] = time.time()
                st.session_state.dataset_status["check_count"] += 1

                if dataset_available:
                    st.success(f"Dataset check complete: {message}")
                    st.session_state.dataset_status["initialization_complete"] = True
                    st.rerun()
                else:
                    st.error(f"Dataset check failed: {message}")

    dataset_status_container = st.empty()
    status = st.session_state.dataset_status

    if status["available"]:
        return True
    last_check = (
        datetime.fromtimestamp(status["last_check_time"]).strftime("%H:%M:%S")
        if status["last_check_time"] > 0
        else "Not checked yet"
    )

    project_root = Path(__file__).resolve().parent.parent
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
            st.info(
                f"Found ZIP files in data directory: {', '.join(f.name for f in files)}"
            )
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

        if dataset_available:
            st.success(f"Dataset check complete: {message}")
            st.session_state.dataset_status["initialization_complete"] = True
            # Add a small delay to allow user to see success message
            time.sleep(1)
        else:
            st.error(f"Dataset check failed: {message}")

        st.rerun()

    return False
