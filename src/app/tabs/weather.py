import streamlit as st
from data import storage
from app.utils import create_line_plot, render_years, create_bar_chart
import plotly.express as px
import polars as pl


def render_temperature_energy_plot(temperature_data):
    """
    Render visualization for temperature vs energy consumption patterns
    """
    col1, col2 = st.columns(2)

    with col1:
        metric = st.selectbox(
            "Select energy metric:",
            options=[
                "energy_mean",
                "energy_median",
                "energy_sum",
                "energy_std",
                "energy_max",
                "energy_min",
            ],
            format_func=lambda x: x.replace("energy_", "").capitalize(),
            index=0,
            key="temp_energy_metric",
        )

    with col2:
        chart_type = st.radio(
            "Chart type:",
            options=["Line chart", "Scatter plot"],
            horizontal=True,
            key="temp_energy_chart_type",
        )

    # Convert polars dataframe to pandas for plotly compatibility
    pandas_df = temperature_data.to_pandas()

    if chart_type == "Line chart":
        fig = px.line(
            pandas_df,
            x="avg_temperature",
            y=metric,
            color="month_name" if "month_name" in pandas_df.columns else None,
            markers=True,
            title=f"Temperature vs {metric.replace('energy_', '').capitalize()} Energy Consumption",
            labels={
                "avg_temperature": "Average Temperature (°C)",
                metric: f"{metric.replace('energy_', '').capitalize()} Energy (kWh)"
            }
        )
    else:  # Scatter plot
        fig = px.scatter(
            pandas_df,
            x="avg_temperature",
            y=metric,
            color="month_name" if "month_name" in pandas_df.columns else None,
            size="energy_count" if "energy_count" in pandas_df.columns else None,
            hover_data=["temp_bin"] if "temp_bin" in pandas_df.columns else None,
            title=f"Temperature vs {metric.replace('energy_', '').capitalize()} Energy Consumption",
            labels={
                "avg_temperature": "Average Temperature (°C)",
                metric: f"{metric.replace('energy_', '').capitalize()} Energy (kWh)"
            },
            opacity=0.7
        )

    fig.update_layout(
        xaxis_title="Average Temperature (°C)",
        yaxis_title=f"{metric.replace('energy_', '').capitalize()} Energy Consumption (kWh)",
        legend_title="Month" if "month_name" in pandas_df.columns else None
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Temperature Bins"):
        if "temp_bin" in pandas_df.columns:
            bin_summary = temperature_data.group_by("temp_bin").agg([
                pl.count().alias("count"),
                pl.min("avg_temperature").alias("min_temp"),
                pl.max("avg_temperature").alias("max_temp"),
                pl.mean("avg_temperature").alias("avg_temp"),
            ])
            st.dataframe(bin_summary, use_container_width=True)


def render_temperature_hourly_plot(hourly_temp_data):
    """
    Render visualization for hourly temperature impact on energy consumption
    """
    col1, col2 = st.columns(2)

    # Determine available metrics from the data columns
    available_metrics = []
    if "energy" in hourly_temp_data.columns:
        available_metrics.append("energy")
    elif "energy_mean" in hourly_temp_data.columns:
        available_metrics.append("energy_mean")

    if "energy_median" in hourly_temp_data.columns:
        available_metrics.append("energy_median")
    if "avg_humidity" in hourly_temp_data.columns:
        available_metrics.append("avg_humidity")

    with col1:
        metric = st.selectbox(
            "Select energy metric:",
            options=available_metrics,
            format_func=lambda x: x.replace("energy_", "").replace("avg_", "").capitalize(),
            index=0,
            key="temp_hourly_metric"
        )

    with col2:
        chart_type = st.radio(
            "Chart type:",
            options=["Heatmap", "Line chart", "3D View"],
            horizontal=True,
            key="temp_hourly_chart_type",
        )

    # Convert polars dataframe to pandas for plotly compatibility
    pandas_df = hourly_temp_data.to_pandas()

    energy_col = metric  # Use the selected metric

    if chart_type == "Heatmap":
        # For heatmap we need to pivot the data
        if "temp_bin" in pandas_df.columns:
            # Create a pivot table with hour as rows and temp_bin as columns
            pivot_df = pandas_df.pivot_table(
                values=energy_col,
                index="hour",
                columns="temp_bin",
                aggfunc="mean"
            ).reset_index()

            # Create heatmap using px.imshow
            fig = px.imshow(
                pivot_df.set_index("hour"),
                labels=dict(x="Temperature Bin", y="Hour of Day", color=f"{energy_col.replace('energy_', '').capitalize()} Consumption"),
                x=pivot_df.columns[1:],  # Temperature bins
                y=pivot_df["hour"],      # Hours
                color_continuous_scale="Viridis",
                title=f"Hourly {energy_col.replace('energy_', '').replace('avg_', '').capitalize()} by Temperature Bin (Heatmap)"
            )
            fig.update_layout(
                xaxis_title="Temperature Bin",
                yaxis_title="Hour of Day",
                yaxis=dict(tickmode="linear", tick0=0, dtick=1)
            )
        else:
            st.error("Temperature bin data is not available for heatmap visualization")
            return

    elif chart_type == "Line chart":
        if "temp_bin" in pandas_df.columns:
            fig = px.line(
                pandas_df,
                x="hour",
                y=energy_col,
                color="temp_bin",
                markers=True,
                title=f"Hourly {energy_col.replace('energy_', '').replace('avg_', '').capitalize()} by Temperature Bin",
                labels={
                    "hour": "Hour of Day",
                    energy_col: f"{energy_col.replace('energy_', '').replace('avg_', '').capitalize()}",
                    "temp_bin": "Temperature Bin"
                }
            )
            fig.update_layout(
                xaxis=dict(tickmode="linear", tick0=0, dtick=1, range=[0, 23]),
                yaxis_title=f"{energy_col.replace('energy_', '').replace('avg_', '').capitalize()}",
                legend_title="Temperature Bin"
            )
        else:
            st.error("Temperature bin data is not available for line chart visualization")
            return

    elif chart_type == "3D View":
        if "temp_bin" in pandas_df.columns and "avg_temperature" in pandas_df.columns:
            # Create a 3D scatter plot
            fig = px.scatter_3d(
                pandas_df,
                x="hour",
                y="avg_temperature",
                z=energy_col,
                color="temp_bin",
                title=f"3D View of Hour, Temperature, and {energy_col.replace('energy_', '').replace('avg_', '').capitalize()}",
                labels={
                    "hour": "Hour of Day",
                    "avg_temperature": "Temperature (°C)",
                    energy_col: f"{energy_col.replace('energy_', '').replace('avg_', '').capitalize()}"
                }
            )
        else:
            st.error("Required data is not available for 3D visualization")
            return

    st.plotly_chart(fig, use_container_width=True)

    # Additional information about the datasets
    with st.expander("Hourly Temperature Impact Details", expanded=False):
        st.markdown("""
        ### Understanding the Hourly Temperature Impact
        
        This analysis shows how temperature affects energy consumption across different hours of the day:
        
        - **Temperature Bins**: Data grouped into temperature ranges
        - **Hourly Patterns**: Reveals how temperature sensitivity varies throughout the day
        - **Peak Hours**: Identify when temperature has the strongest impact on consumption
        """)


def render_weather_tab():
    """
    Main function to render the weather impact analysis tab
    """
    filters = st.session_state.get("filters", {})
    selected_years = filters.get("years", None)
    selected_months = filters.get("months", None)
    selected_temp_bins = filters.get("temp_bins", None)

    st.subheader("🌡️ Weather Impact Analysis")

    # Create subtabs for daily and hourly analysis
    daily_tab, hourly_tab = st.tabs(["📆 Daily Temperature Impact", "⏰ Hourly Temperature Impact"])

    with daily_tab:
        st.markdown(f"### Temperature vs Energy Consumption | Year {render_years()}")

        with st.spinner("Loading temperature-energy patterns..."):
            temp_energy_data = storage.get_temperature_energy_patterns(
                years=selected_years
            )

            # Apply month filters if selected
            if selected_months and len(selected_months) > 0 and "month" in temp_energy_data.columns:
                temp_energy_data = temp_energy_data.filter(pl.col("month").is_in(selected_months))

            # Apply temperature bin filters if selected
            if selected_temp_bins and len(selected_temp_bins) > 0 and "temp_bin" in temp_energy_data.columns:
                temp_energy_data = temp_energy_data.filter(pl.col("temp_bin").is_in(selected_temp_bins))

        render_temperature_energy_plot(temp_energy_data)

        with st.expander("View Temperature-Energy Data", expanded=False):
            st.dataframe(temp_energy_data, use_container_width=True)

    with hourly_tab:
        st.markdown("### Hourly Temperature Impact")

        with st.spinner("Loading hourly temperature patterns..."):
            temp_hourly_data = storage.get_temperature_hourly_patterns()

            # Apply temperature bin filters if selected
            if selected_temp_bins and len(selected_temp_bins) > 0 and "temp_bin" in temp_hourly_data.columns:
                temp_hourly_data = temp_hourly_data.filter(pl.col("temp_bin").is_in(selected_temp_bins))

        render_temperature_hourly_plot(temp_hourly_data)

        with st.expander("View Hourly Temperature Data", expanded=False):
            st.dataframe(temp_hourly_data, use_container_width=True)
