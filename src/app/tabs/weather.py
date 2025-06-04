import streamlit as st
from data import storage
from app.utils import create_line_plot, render_years, create_bar_chart
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import pandas as pd


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
            options=["Line chart", "Box plot", "Month comparison"],
            horizontal=True,
            key="temp_energy_chart_type",
        )

    # Convert polars dataframe to pandas for plotly compatibility
    pandas_df = temperature_data.to_pandas()

    # Define temperature bin order
    temp_bin_order = [
        "Below 0°C",
        "0-5°C",
        "5-10°C",
        "10-15°C",
        "15-20°C",
        "20-25°C",
        "Above 25°C",
    ]

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
                metric: f"{metric.replace('energy_', '').capitalize()} Energy (kWh)",
            },
        )
    elif chart_type == "Box plot":
        # Use the custom temperature bin order
        available_temp_bins = [
            tb for tb in temp_bin_order if tb in pandas_df["temp_bin"].unique()
        ]

        fig = px.box(
            pandas_df,
            x="temp_bin",
            y=metric,
            color="month_name" if "month_name" in pandas_df.columns else None,
            title=f"Temperature Bin vs {metric.replace('energy_', '').capitalize()} Energy Consumption",
            labels={
                "temp_bin": "Temperature Bin",
                metric: f"{metric.replace('energy_', '').capitalize()} Energy (kWh)",
                "month_name": "Month",
            },
            category_orders={"temp_bin": available_temp_bins},
        )
    elif chart_type == "Month comparison":
        if "month_name" in pandas_df.columns:
            # Create a grouped bar chart by month
            month_order = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]

            # Filter and sort the dataframe to include only available months
            available_months = pandas_df["month_name"].unique()
            ordered_months = [m for m in month_order if m in available_months]

            fig = px.bar(
                pandas_df,
                x="month_name",
                y=metric,
                color="temp_bin",
                barmode="group",
                title=f"Monthly {metric.replace('energy_', '').capitalize()} Energy by Temperature Bin",
                labels={
                    "month_name": "Month",
                    metric: f"{metric.replace('energy_', '').capitalize()} Energy (kWh)",
                    "temp_bin": "Temperature Bin",
                },
                category_orders={
                    "month_name": ordered_months,
                    "temp_bin": [
                        tb
                        for tb in temp_bin_order
                        if tb in pandas_df["temp_bin"].unique()
                    ],
                },
            )
        else:
            st.error("Month data is not available for monthly comparison")
            return

    fig.update_layout(
        xaxis_title="Average Temperature (°C)"
        if chart_type == "Line chart"
        else ("Temperature Bin" if chart_type == "Box plot" else "Month"),
        yaxis_title=f"{metric.replace('energy_', '').capitalize()} Energy Consumption (kWh)",
        legend_title="Month"
        if "month_name" in pandas_df.columns and chart_type != "Month comparison"
        else "Temperature Bin"
        if chart_type == "Month comparison"
        else None,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add temperature distribution analysis
    with st.expander("Temperature Distribution Analysis"):
        # Temperature distribution histogram
        temp_hist = px.histogram(
            pandas_df,
            x="avg_temperature",
            nbins=20,
            title="Temperature Distribution",
            labels={"avg_temperature": "Average Temperature (°C)"},
            color_discrete_sequence=["skyblue"],
        )
        st.plotly_chart(temp_hist, use_container_width=True)

    with st.expander("View Temperature Bins"):
        if "temp_bin" in pandas_df.columns:
            bin_summary = temperature_data.group_by("temp_bin").agg(
                [
                    pl.count().alias("count"),
                    pl.min("avg_temperature").alias("min_temp"),
                    pl.max("avg_temperature").alias("max_temp"),
                    pl.mean("avg_temperature").alias("avg_temp"),
                ]
            )

            # Sort the bin_summary by the defined temperature bin order
            available_temp_bins = [
                tb for tb in temp_bin_order if tb in bin_summary["temp_bin"].to_list()
            ]
            bin_summary = bin_summary.filter(
                pl.col("temp_bin").is_in(available_temp_bins)
            )

            # Convert to pandas for plotting
            bin_summary_pd = bin_summary.to_pandas()
            bin_summary_pd["temp_bin"] = pd.Categorical(
                bin_summary_pd["temp_bin"], categories=available_temp_bins, ordered=True
            )
            bin_summary_pd = bin_summary_pd.sort_values("temp_bin")

            # Convert back to polars
            bin_summary = pl.from_pandas(bin_summary_pd)

            st.dataframe(bin_summary, use_container_width=True)


def render_temperature_hourly_plot(hourly_temp_data):
    """
    Render visualization for hourly temperature impact on energy consumption
    """
    # Always use energy as the metric (removing choice)
    energy_col = "energy"

    # Check if energy column is available
    if energy_col not in hourly_temp_data.columns:
        st.error("Energy data is not available in the hourly temperature data.")
        return

    chart_type = st.radio(
        "Chart type:",
        options=["Heatmap", "Line chart", "3D View"],
        horizontal=True,
        key="temp_hourly_chart_type",
    )

    # Convert polars dataframe to pandas for plotly compatibility
    pandas_df = hourly_temp_data.to_pandas()

    # Define temperature bin order
    temp_bin_order = [
        "Below 0°C",
        "0-5°C",
        "5-10°C",
        "10-15°C",
        "15-20°C",
        "20-25°C",
        "Above 25°C",
    ]

    # Get available temp bins in order
    if "temp_bin" in pandas_df.columns:
        available_temp_bins = [
            tb for tb in temp_bin_order if tb in pandas_df["temp_bin"].unique()
        ]

    if chart_type == "Heatmap":
        if "temp_bin" in pandas_df.columns:
            # Create a pivot table with hour as rows and temp_bin as columns
            pivot_df = pandas_df.pivot_table(
                values=energy_col, index="hour", columns="temp_bin", aggfunc="mean"
            ).reset_index()

            # Create heatmap using px.imshow
            fig = px.imshow(
                pivot_df.set_index("hour"),
                labels=dict(x="Temperature Bin", y="Hour of Day", color="Energy"),
                x=[
                    col for col in pivot_df.columns if col != "hour"
                ],  # Temperature bins
                y=pivot_df["hour"],  # Hours
                color_continuous_scale="Viridis",
                title=f"Hourly Energy Consumption by Temperature Bin",
            )
            fig.update_layout(
                xaxis_title="Temperature Bin",
                yaxis_title="Hour of Day",
                yaxis=dict(tickmode="linear", tick0=0, dtick=1),
            )

            # Sort x-axis categories based on temperature bin order
            if available_temp_bins:
                fig.update_xaxes(
                    categoryorder="array", categoryarray=available_temp_bins
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
                title=f"Hourly Energy Consumption by Temperature Bin",
                labels={
                    "hour": "Hour of Day",
                    energy_col: "Energy Consumption (kWh)",
                    "temp_bin": "Temperature Bin",
                },
                category_orders={"temp_bin": available_temp_bins}
                if available_temp_bins
                else None,
            )
            fig.update_layout(
                xaxis=dict(tickmode="linear", tick0=0, dtick=1, range=[0, 23]),
                yaxis_title="Energy Consumption (kWh)",
                legend_title="Temperature Bin",
            )
        else:
            st.error(
                "Temperature bin data is not available for line chart visualization"
            )
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
                title=f"3D View of Hour, Temperature, and Energy Consumption",
                labels={
                    "hour": "Hour of Day",
                    "avg_temperature": "Temperature (°C)",
                    energy_col: "Energy Consumption (kWh)",
                },
                opacity=0.7,
                category_orders={"temp_bin": available_temp_bins}
                if available_temp_bins
                else None,
            )
            # Improve 3D layout
            fig.update_layout(
                scene=dict(
                    xaxis_title="Hour of Day",
                    yaxis_title="Temperature (°C)",
                    zaxis_title="Energy Consumption (kWh)",
                )
            )
        else:
            st.error("Required data is not available for 3D visualization")
            return

    st.plotly_chart(fig, use_container_width=True)

    st.success("""
    ### ⏰ Hourly Temperature Impact Insights

    **What you're seeing in the charts:**
    * The heatmap shows how energy consumption varies by hour and temperature bin
    * The line chart displays hourly consumption patterns for different temperature ranges
    * The 3D view combines hour, temperature, and consumption in a spatial visualization

    **Key observations:**
    * **Evening peak intensity**: The highest energy consumption occurs between 18:00-19:00 (6-7 PM), with below-freezing temperatures showing consumption spikes up to 0.90 kWh, compared to only 0.45-0.50 kWh during the same hours in the 15-20°C range
    * **Morning peak patterns**: A secondary consumption peak appears between 7:00-9:00 AM, with cold temperatures (<5°C) showing consumption of 0.45-0.50 kWh versus 0.33-0.35 kWh in moderate temperatures
    * **Overnight efficiency**: Between 2:00-4:00 AM, energy consumption reaches its lowest point (0.20-0.30 kWh) and shows minimal temperature sensitivity
    * **Temperature threshold effect**: Below 5°C, each degree drop increases consumption by approximately 0.02-0.03 kWh per hour, with the effect amplified during peak hours
    """)

    # Temperature impact summary stats
    with st.expander("Temperature Impact Statistics", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Calculate peak hours
            if "hour" in pandas_df.columns and energy_col in pandas_df.columns:
                # Fix FutureWarning by explicitly setting observed=False
                peak_hours = (
                    pandas_df.groupby("hour", observed=False)[energy_col]
                    .mean()
                    .reset_index()
                )
                peak_hour = peak_hours.loc[peak_hours[energy_col].idxmax()]["hour"]

                st.metric(
                    label="Peak Energy Hour",
                    value=f"{int(peak_hour)}:00",
                    delta=None,
                    help="Hour of the day with highest average energy consumption",
                )

                if "avg_temperature" in pandas_df.columns:
                    avg_temp = pandas_df["avg_temperature"].mean()
                    st.metric(
                        label="Average Temperature",
                        value=f"{avg_temp:.1f} °C",
                        delta=None,
                    )

        with col2:
            if "avg_humidity" in pandas_df.columns and "energy" in pandas_df.columns:
                # Calculate correlation between humidity and energy
                humidity_corr = pandas_df[["energy", "avg_humidity"]].corr().iloc[0, 1]
                st.metric(
                    label="Humidity-Energy Correlation",
                    value=f"{humidity_corr:.2f}",
                    delta=None,
                    help="Correlation between humidity and energy consumption",
                )

            if "temp_bin" in pandas_df.columns:
                # Count unique temperature bins
                st.metric(
                    label="Temperature Ranges",
                    value=f"{pandas_df['temp_bin'].nunique()}",
                    delta=None,
                    help="Number of different temperature ranges in the data",
                )


def render_weather_tab():
    """
    Main function to render the weather impact analysis tab
    """
    filters = st.session_state.get("filters", {})
    selected_years = filters.get("years", None)
    selected_months = filters.get("months", None)
    selected_temp_bins = filters.get("temp_bins", None)

    st.subheader("🌡️ Weather Impact Analysis")

    st.markdown("""
    Analyze how weather conditions affect energy consumption across different time periods.
    Use the filters in the sidebar to narrow down specific time periods or temperature ranges.
    """)

    # Create subtabs for daily and hourly analysis
    daily_tab, hourly_tab = st.tabs(
        ["📆 Daily Temperature Impact", "⏰ Hourly Temperature Impact"]
    )

    with daily_tab:
        st.markdown(f"### Temperature vs Energy Consumption | Year {render_years()}")

        with st.spinner("Loading temperature-energy patterns..."):
            temp_energy_data = storage.get_temperature_energy_patterns(
                years=selected_years
            )

            # Apply month filters if selected
            if (
                selected_months
                and len(selected_months) > 0
                and "month" in temp_energy_data.columns
            ):
                temp_energy_data = temp_energy_data.filter(
                    pl.col("month").is_in(selected_months)
                )

            # Apply temperature bin filters if selected
            if (
                selected_temp_bins
                and len(selected_temp_bins) > 0
                and "temp_bin" in temp_energy_data.columns
            ):
                temp_energy_data = temp_energy_data.filter(
                    pl.col("temp_bin").is_in(selected_temp_bins)
                )

        render_temperature_energy_plot(temp_energy_data)
        st.success("""
        ### 🌡️ Daily Temperature Impact Insights

        **What you're seeing in the charts:**
        * The line chart shows how average energy consumption varies with temperature
        * The box plot displays the consumption distribution within each temperature range
        * The month comparison reveals seasonal patterns across different temperature bins

        **Key observations:**
        * **U-shaped consumption curve**: Energy usage is highest at temperature extremes (below 0°C) and lowest in the 15-20°C range, creating a distinctive U-shaped pattern
        * **Cold temperature sensitivity**: Below 0°C, energy consumption increases dramatically to 0.46-0.60 kWh (mean), nearly double the consumption in the optimal 15-20°C range (0.25-0.30 kWh)
        * **Seasonal transitions**: The months of November and March show particularly volatile consumption patterns as households transition between heating and non-heating periods
        * **Monthly variation**: Winter months (December-February) consistently show 20-30% higher energy consumption than summer months (June-August) across all temperature bins
        * **Temperature bin distribution**: The 5-10°C and 10-15°C ranges contain the highest number of observations, representing the most common temperature conditions in this climate region
        """)

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

