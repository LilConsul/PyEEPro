import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
import sys
import os
import importlib

# Lazy import function to prevent PyTorch loading during Streamlit's module scanning
def import_ml_modules():
    """Lazily import ML modules to prevent conflicts with Streamlit's module scanning"""
    from ml.mail_ml import AutoencoderPipeline
    return AutoencoderPipeline


def plot_random_example(processed_data, reconstructed_data):
    """
    Plot a random example from the processed and reconstructed data

    Args:
        processed_data: Original processed data
        reconstructed_data: Reconstructed data from the autoencoder
    """
    # Get a random index
    if "current_example_index" not in st.session_state:
        st.session_state["current_example_index"] = random.randint(0, len(processed_data) - 1)

    idx = st.session_state["current_example_index"]

    # Create a combined plot of original and reconstructed data
    fig = go.Figure()

    # Add the original data trace
    fig.add_trace(go.Scatter(
        y=processed_data[idx].flatten(),
        mode='lines',
        name='Original Data',
        line=dict(color='blue')
    ))

    # Add the reconstructed data trace
    fig.add_trace(go.Scatter(
        y=reconstructed_data[idx].flatten(),
        mode='lines',
        name='Reconstructed Data',
        line=dict(color='red')
    ))

    # Update layout
    fig.update_layout(
        title=f"Autoencoder Reconstruction (Example #{idx})",
        xaxis_title="Time",
        yaxis_title="Energy Consumption",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def render_model_training_section():
    """Render the model training and configuration section"""
    st.subheader("🧠 Model Training")

    # Get values from sidebar if available
    if "filters" in st.session_state and st.session_state.filters:
        acorn_group = st.session_state.filters.get("ml_acorn_groups", "Comfortable")
        selected_years = st.session_state.filters.get("ml_years", [2011, 2012])
    else:
        acorn_group = "Comfortable"
        selected_years = (2011, 2012)

    # If selected_years is a list, convert to tuple
    if isinstance(selected_years, list) and len(selected_years) >= 2:
        selected_years = (min(selected_years), max(selected_years))
    elif isinstance(selected_years, list) and len(selected_years) == 1:
        selected_years = (selected_years[0], selected_years[0])

    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox(
            "Model type:",
            options=["Autoencoder"],
            index=0,
            key="ml_model_type",
        )

        # Advanced settings expander
        with st.expander("Advanced Settings"):
            encoding_dim = st.slider(
                "Encoding dimension",
                min_value=1,
                max_value=10,
                value=2,
                key="ml_encoding_dim"
            )
            hidden_dim = st.slider(
                "Hidden dimension",
                min_value=4,
                max_value=64,
                value=8,
                key="ml_hidden_dim"
            )
            epochs = st.slider(
                "Training epochs",
                min_value=5,
                max_value=100,
                value=20,
                key="ml_epochs"
            )

    with col2:
        target_variable = st.selectbox(
            "Target variable:",
            options=["energy_consumption"],
            index=0,
            key="ml_target_variable",
        )

        # Display the configuration summary
        st.markdown(f"""
        **Configuration:**
        - Acorn Group: {acorn_group}
        - Years: {selected_years}
        - Model: {model_type}
        - Encoding Dimension: {st.session_state.get("ml_encoding_dim", 2)}
        """)

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
                    selected_years=selected_years,
                    encoding_dim=st.session_state.get("ml_encoding_dim", 2),
                    hidden_dim=st.session_state.get("ml_hidden_dim", 8),
                    auto_resource_adjustment=True,
                    epochs=st.session_state.get("ml_epochs", 20),
                )

                # Run the pipeline
                processed_data, reconstructed_data = pipeline.run_pipeline(force_retrain=False)

                # Store results in session state
                st.session_state["ml_pipeline"] = pipeline
                st.session_state["processed_data"] = processed_data
                st.session_state["reconstructed_data"] = reconstructed_data

                # Generate a random example index
                st.session_state["current_example_index"] = random.randint(0, len(processed_data) - 1)

                st.success("Model training/loading completed successfully!")
            except Exception as e:
                st.error(f"Error running the autoencoder pipeline: {str(e)}")
                st.exception(e)

    # Results section - only show if we have results
    if st.session_state.get("processed_data") is not None and st.session_state.get("reconstructed_data") is not None:
        st.subheader("Model Results")

        processed_data = st.session_state["processed_data"]
        reconstructed_data = st.session_state["reconstructed_data"]

        # Display the random example plot
        fig = plot_random_example(processed_data, reconstructed_data)
        st.plotly_chart(fig, use_container_width=True)

        # Add a button to show another random example
        if st.button("Show Another Random Example"):
            # Generate a new random index
            st.session_state["current_example_index"] = random.randint(0, len(processed_data) - 1)
            # Force a rerun to update the visualization
            st.rerun()


def render_model_evaluation_section():
    """Render the model evaluation and results section"""
    st.subheader("📊 Model Evaluation")

    st.info("This is a mock version of the ML evaluation section. Implementation coming soon.")

    st.selectbox(
        "Select trained model:",
        options=["[Mock] Model 1", "[Mock] Model 2", "[Mock] Model 3"],
        index=0,
        key="ml_evaluation_model",
    )

    # Mock metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mock Metric 1", "N/A")
    with col2:
        st.metric("Mock Metric 2", "N/A")
    with col3:
        st.metric("Mock Metric 3", "N/A")

    # Mock chart placeholder
    st.text("Visualization would appear here")
    fig = go.Figure()
    fig.update_layout(
        title="[Mock] Model Evaluation Plot",
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        height=300,
    )
    fig.add_annotation(
        text="Model evaluation visualizations will appear here",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)


def render_anomaly_detection_section():
    """Render the anomaly detection section"""
    st.subheader("🔎 Anomaly Detection")

    st.info("This is a mock version of the Anomaly Detection section. Implementation coming soon.")

    col1, col2 = st.columns(2)
    with col1:
        st.selectbox(
            "Detection method:",
            options=["[Mock] Method 1", "[Mock] Method 2"],
            index=0,
            key="ml_anomaly_method",
        )

    with col2:
        st.slider(
            "Sensitivity threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            disabled=True,
        )

    # Mock chart placeholder
    st.text("Anomaly detection visualization would appear here")
    fig = go.Figure()
    fig.update_layout(
        title="[Mock] Anomaly Detection Plot",
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        height=300,
    )
    fig.add_annotation(
        text="Anomaly detection visualizations will appear here",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)


def render_forecasting_section():
    """Render the energy consumption forecasting section"""
    st.subheader("📈 Energy Consumption Forecasting")

    st.info("This is a mock version of the Forecasting section. Implementation coming soon.")

    col1, col2 = st.columns(2)
    with col1:
        st.selectbox(
            "Forecast horizon:",
            options=["[Mock] Horizon 1", "[Mock] Horizon 2"],
            index=0,
            key="ml_forecast_horizon",
        )

    with col2:
        st.selectbox(
            "Forecast model:",
            options=["[Mock] Model 1", "[Mock] Model 2"],
            index=0,
            key="ml_forecast_model",
        )

    # Mock chart placeholder
    st.text("Forecasting visualization would appear here")
    fig = go.Figure()
    fig.update_layout(
        title="[Mock] Forecasting Plot",
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        height=300,
    )
    fig.add_annotation(
        text="Forecasting visualizations will appear here",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)


def render_ml_tab():
    """Main function to render the ML analysis tab"""
    st.header("🤖 Machine Learning Analysis")

    st.markdown("""
    This tab will provide machine learning tools for analyzing energy consumption patterns,
    detecting anomalies, and forecasting future consumption.
    
    **Note: This is a mock/placeholder implementation. Full ML functionality coming soon.**
    """)

    st.warning("""
    ### Development Status
    
    The Machine Learning tab is currently under development. The interfaces shown are
    mockups to demonstrate the planned functionality. Check back later for the full implementation.
    """)

    # Create tabs for different ML functions
    train_tab, evaluate_tab, anomaly_tab, forecast_tab = st.tabs([
        "🧠 Train Models", "📊 Evaluate Models",
        "🔎 Anomaly Detection", "📈 Forecasting"
    ])

    with train_tab:
        render_model_training_section()

    with evaluate_tab:
        render_model_evaluation_section()

    with anomaly_tab:
        render_anomaly_detection_section()

    with forecast_tab:
        render_forecasting_section()
