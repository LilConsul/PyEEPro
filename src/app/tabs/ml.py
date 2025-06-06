import streamlit as st
import plotly.graph_objects as go
import random
import numpy as np


# Lazy import function to prevent PyTorch loading during Streamlit's module scanning
def import_ml_modules():
    """Lazily import ML modules to prevent conflicts with Streamlit's module scanning"""
    from ml.mail_ml import AutoencoderPipeline

    return AutoencoderPipeline


def plot_random_example(processed_data, reconstructed_data, anomaly_threshold=0.1):
    """
    Plot a random example from the processed and reconstructed data,
    highlighting anomalies with a yellow background where data differs significantly.

    Args:
        processed_data: Original processed data
        reconstructed_data: Reconstructed data from the autoencoder
        anomaly_threshold: Threshold for marking differences as anomalies
    """
    # Get a random index
    if "current_example_index" not in st.session_state:
        st.session_state["current_example_index"] = random.randint(
            0, len(processed_data) - 1
        )

    idx = st.session_state["current_example_index"]

    # Create a combined plot of original and reconstructed data
    fig = go.Figure()

    # Get the data for this example
    original = processed_data[idx].flatten()
    reconstructed = reconstructed_data[idx].flatten()

    # Calculate the differences
    differences = np.abs(original - reconstructed)
    anomaly_mask = differences > anomaly_threshold

    # Add yellow rectangles for anomaly regions
    for i in range(len(anomaly_mask)):
        if anomaly_mask[i]:
            fig.add_vrect(
                x0=i - 0.5,
                x1=i + 0.5,
                fillcolor="yellow",
                opacity=0.3,
                layer="below",
                line_width=0,
            )

    # Add the original data trace
    fig.add_trace(
        go.Scatter(
            y=original,
            mode="lines",
            name="Original Data",
            line=dict(color="blue"),
        )
    )

    # Add the reconstructed data trace
    fig.add_trace(
        go.Scatter(
            y=reconstructed,
            mode="lines",
            name="Reconstructed Data",
            line=dict(color="red"),
        )
    )

    # Create time labels for x-axis (48 half-hour intervals in a day)
    time_labels = []
    for i in range(48):
        hour = i // 2
        minute = (i % 2) * 30
        time_labels.append(f"{hour:02d}:{minute:02d}")

    # Update layout
    fig.update_layout(
        title=f"Autoencoder Reconstruction (Example #{idx})",
        xaxis_title="Time of day",
        yaxis_title="Energy Consumption (kWh)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(original))),
            ticktext=time_labels[:len(original)] if len(original) <= 48 else time_labels
        )
    )

    return fig


def render_model_training_section():
    """Render the model training and configuration section"""
    st.subheader("🧠 Model Training")

    # Model Settings Section
    st.subheader("Model Settings")

    # Create two columns for config and summary
    param_col, summary_col = st.columns(2)

    # Parameters column
    with param_col:
        # ACORN group selection
        acorn_groups = ["Adversity", "Comfortable", "Affluent"]
        acorn_group = st.selectbox(
            "Select ACORN Group",
            options=acorn_groups,
            index=1,  # Default to "Comfortable" (index 1)
            key="ml_acorn_groups",
        )

        # Year selection
        available_years = list(range(2011, 2015))
        selected_years_ml = st.multiselect(
            "Select Years",
            available_years,
            default=list(range(2011, 2013)),
            key="ml_years",
        )

        # For pipeline processing (which expects a tuple range)
        if isinstance(selected_years_ml, list) and len(selected_years_ml) >= 2:
            years_for_pipeline = (min(selected_years_ml), max(selected_years_ml))
        elif isinstance(selected_years_ml, list) and len(selected_years_ml) == 1:
            years_for_pipeline = (selected_years_ml[0], selected_years_ml[0])
        else:
            years_for_pipeline = (2011, 2012)  # Default
            selected_years_ml = list(range(2011, 2013))  # Default for display

        # Advanced settings in an expandable section
        with st.expander("Advanced Settings"):
            # Model parameters
            encoding_dim = st.slider(
                "Encoding dimension",
                min_value=1,
                max_value=10,
                value=2,
                key="ml_encoding_dim",
            )

            hidden_dim = st.slider(
                "Hidden dimension",
                min_value=4,
                max_value=64,
                value=8,
                key="ml_hidden_dim",
            )

            epochs = st.slider(
                "Training epochs",
                min_value=5,
                max_value=100,
                value=20,
                key="ml_epochs",
            )

    # Summary column
    with summary_col:
        st.markdown("### Configuration Summary")

        # Format the selected years for display
        years_display = (
            ", ".join(map(str, sorted(selected_years_ml)))
            if selected_years_ml
            else "None selected"
        )

        st.markdown(
            f"""
        - **Acorn Group:** {acorn_group}
        - **Years:** {years_display}
        - **Model:** Conditional Autoencoder
        - **Target:** Energy consumption anomalies
        - **Encoding Dimension:** {encoding_dim}
        - **Hidden Dimension:** {hidden_dim}
        - **Training Epochs:** {epochs}
        """
        )

    # Initialize the model or retrieve from session state
    if "ml_pipeline" not in st.session_state:
        st.session_state["ml_pipeline"] = None
        st.session_state["processed_data"] = None
        st.session_state["reconstructed_data"] = None

    # Train or run model button
    if st.button("Train/Run Model", type="primary"):
        with st.spinner("Running autoencoder pipeline..."):
            try:
                # Import the pipeline class
                AutoencoderPipeline = import_ml_modules()

                # Initialize the pipeline
                pipeline = AutoencoderPipeline(
                    acorn_group=acorn_group,
                    selected_years=years_for_pipeline,  # Use the tuple format for the pipeline
                    encoding_dim=st.session_state.get("ml_encoding_dim", 2),
                    hidden_dim=st.session_state.get("ml_hidden_dim", 8),
                    auto_resource_adjustment=True,
                    epochs=st.session_state.get("ml_epochs", 20),
                )

                # Run the pipeline
                processed_data, reconstructed_data = pipeline.run_pipeline(
                    force_retrain=False
                )

                # Store results in session state
                st.session_state["ml_pipeline"] = pipeline
                st.session_state["processed_data"] = processed_data
                st.session_state["reconstructed_data"] = reconstructed_data

                # Generate a random example index
                st.session_state["current_example_index"] = random.randint(
                    0, len(processed_data) - 1
                )

                st.success("Model training/loading completed successfully!")
            except Exception as e:
                st.error(f"Error running the autoencoder pipeline: {str(e)}")
                st.exception(e)

    # Results section - only show if we have results
    if (
        st.session_state.get("processed_data") is not None
        and st.session_state.get("reconstructed_data") is not None
    ):
        st.subheader("Model Results")

        processed_data = st.session_state["processed_data"]
        reconstructed_data = st.session_state["reconstructed_data"]

        # Add anomaly threshold slider
        anomaly_threshold = st.slider(
            "Anomaly Detection Threshold",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Set the threshold for marking anomalies. Higher values will highlight only larger differences.",
        )

        # Display the random example plot
        fig = plot_random_example(processed_data, reconstructed_data, anomaly_threshold)
        st.plotly_chart(fig, use_container_width=True)

        # Add a button to show another random example
        if st.button("Show Another Random Example"):
            # Generate a new random index
            st.session_state["current_example_index"] = random.randint(
                0, len(processed_data) - 1
            )
            # Force a rerun to update the visualization
            st.rerun()


def render_ml_tab():
    """Main function to render the ML analysis tab"""
    st.header("🤖 Machine Learning Analysis")

    st.markdown(
        """
    This tab will provide machine learning tools for analyzing energy consumption patterns,
    detecting anomalies, and forecasting future consumption.
    """
    )

    # Create tabs for different ML functions

    render_model_training_section()
