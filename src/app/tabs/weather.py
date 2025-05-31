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
    elif chart_type == "Box plot":
        fig = px.box(
            pandas_df,
            x="temp_bin",
            y=metric,
            color="month_name" if "month_name" in pandas_df.columns else None,
            title=f"Temperature Bin vs {metric.replace('energy_', '').capitalize()} Energy Consumption",
            labels={
                "temp_bin": "Temperature Bin",
                metric: f"{metric.replace('energy_', '').capitalize()} Energy (kWh)",
                "month_name": "Month"
            },
            category_orders={"temp_bin": sorted(pandas_df["temp_bin"].unique())}
        )
    elif chart_type == "Month comparison":
        if "month_name" in pandas_df.columns:
            # Create a grouped bar chart by month
            month_order = ["January", "February", "March", "April", "May", "June",
                         "July", "August", "September", "October", "November", "December"]

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
                    "temp_bin": "Temperature Bin"
                },
                category_orders={"month_name": ordered_months}
            )
        else:
            st.error("Month data is not available for monthly comparison")
            return

    fig.update_layout(
        xaxis_title="Average Temperature (°C)" if chart_type == "Line chart" else ("Temperature Bin" if chart_type == "Box plot" else "Month"),
        yaxis_title=f"{metric.replace('energy_', '').capitalize()} Energy Consumption (kWh)",
        legend_title="Month" if "month_name" in pandas_df.columns and chart_type != "Month comparison" else "Temperature Bin" if chart_type == "Month comparison" else None
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add temperature distribution analysis
    with st.expander("Temperature Distribution Analysis"):
        col1, col2 = st.columns(2)

        with col1:
            # Temperature distribution histogram
            temp_hist = px.histogram(
                pandas_df,
                x="avg_temperature",
                nbins=20,
                title="Temperature Distribution",
                labels={"avg_temperature": "Average Temperature (°C)"},
                color_discrete_sequence=['skyblue']
            )
            st.plotly_chart(temp_hist, use_container_width=True)

        with col2:
            # Energy metric vs Temperature correlation
            if "avg_temperature" in pandas_df.columns:
                corr = pandas_df[[metric, "avg_temperature"]].corr().iloc[0,1]
                st.metric(
                    label=f"Correlation: Temperature vs {metric.replace('energy_', '').capitalize()}",
                    value=f"{corr:.2f}",
                    delta=None,
                    help="Pearson correlation coefficient between temperature and energy metric"
                )

                # Add more insights based on correlation value
                if corr > 0.7:
                    st.info("Strong positive correlation: Energy consumption increases significantly with temperature")
                elif corr > 0.3:
                    st.info("Moderate positive correlation: Energy consumption tends to increase with temperature")
                elif corr > -0.3:
                    st.info("Weak correlation: Temperature has limited impact on energy consumption")
                elif corr > -0.7:
                    st.info("Moderate negative correlation: Energy consumption tends to decrease with temperature")
                else:
                    st.info("Strong negative correlation: Energy consumption decreases significantly with temperature")

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

    # Determine available metrics from the actual data columns
    available_metrics = []
    if "energy" in hourly_temp_data.columns:
        available_metrics.append("energy")
    if "avg_humidity" in hourly_temp_data.columns:
        available_metrics.append("avg_humidity")
    if "count" in hourly_temp_data.columns:
        available_metrics.append("count")

    if not available_metrics:
        st.error("No energy or humidity metrics available in the hourly temperature data.")
        return

    with col1:
        metric = st.selectbox(
            "Select metric:",
            options=available_metrics,
            format_func=lambda x: x.replace("energy", "Energy").replace("avg_", "").replace("count", "Data Count").capitalize(),
            index=0,
            key="temp_hourly_metric"
        )

    with col2:
        chart_type = st.radio(
            "Chart type:",
            options=["Heatmap", "Line chart", "3D View", "Calendar view"],
            horizontal=True,
            key="temp_hourly_chart_type",
        )

    # Convert polars dataframe to pandas for plotly compatibility
    pandas_df = hourly_temp_data.to_pandas()

    energy_col = metric  # Use the selected metric

    if chart_type == "Heatmap":
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
                labels=dict(x="Temperature Bin", y="Hour of Day", color=f"{energy_col.replace('energy', 'Energy').replace('avg_', '').capitalize()}"),
                x=pivot_df.columns[1:],  # Temperature bins
                y=pivot_df["hour"],      # Hours
                color_continuous_scale="Viridis",
                title=f"Hourly {energy_col.replace('energy', 'Energy').replace('avg_', '').replace('count', 'Data Count').capitalize()} by Temperature Bin"
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
                title=f"Hourly {energy_col.replace('energy', 'Energy').replace('avg_', '').replace('count', 'Data Count').capitalize()} by Temperature Bin",
                labels={
                    "hour": "Hour of Day",
                    energy_col: f"{energy_col.replace('energy', 'Energy').replace('avg_', '').replace('count', 'Data Count').capitalize()}",
                    "temp_bin": "Temperature Bin"
                }
            )
            fig.update_layout(
                xaxis=dict(tickmode="linear", tick0=0, dtick=1, range=[0, 23]),
                yaxis_title=f"{energy_col.replace('energy', 'Energy').replace('avg_', '').replace('count', 'Data Count').capitalize()}",
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
                title=f"3D View of Hour, Temperature, and {energy_col.replace('energy', 'Energy').replace('avg_', '').replace('count', 'Data Count').capitalize()}",
                labels={
                    "hour": "Hour of Day",
                    "avg_temperature": "Temperature (°C)",
                    energy_col: f"{energy_col.replace('energy', 'Energy').replace('avg_', '').replace('count', 'Data Count').capitalize()}"
                },
                opacity=0.7
            )
            # Improve 3D layout
            fig.update_layout(
                scene=dict(
                    xaxis_title="Hour of Day",
                    yaxis_title="Temperature (°C)",
                    zaxis_title=f"{energy_col.replace('energy', 'Energy').replace('avg_', '').replace('count', 'Data Count').capitalize()}",
                )
            )
        else:
            st.error("Required data is not available for 3D visualization")
            return

    elif chart_type == "Calendar view":
        st.warning("This visualization requires day-of-year data which may not be available. Showing hour patterns instead.")

        # Create a circular heatmap (clock view)
        if "hour" in pandas_df.columns:
            hourly_avg = pandas_df.groupby("hour")[energy_col].mean().reset_index()

            # Create a radial bar chart
            fig = go.Figure()

            fig.add_trace(go.Barpolar(
                r=hourly_avg[energy_col].tolist(),
                theta=[(h * 15) for h in hourly_avg["hour"]],  # Convert to degrees (15° per hour)
                width=14,  # Width of each bar in degrees
                marker_color=hourly_avg[energy_col],
                marker_colorscale="Viridis",
                hoverinfo="text",
                hovertext=[f"Hour {h}: {v:.2f}" for h, v in zip(hourly_avg["hour"], hourly_avg[energy_col])],
            ))

            # Update layout
            fig.update_layout(
                title=f"24-Hour Clock View of {energy_col.replace('energy', 'Energy').replace('avg_', '').replace('count', 'Data Count').capitalize()}",
                polar=dict(
                    radialaxis=dict(showticklabels=False, ticks=""),
                    angularaxis=dict(
                        tickvals=[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165,
                                  180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345],
                        ticktext=["12 AM", "1 AM", "2 AM", "3 AM", "4 AM", "5 AM", "6 AM", "7 AM", "8 AM", "9 AM", "10 AM", "11 AM",
                                  "12 PM", "1 PM", "2 PM", "3 PM", "4 PM", "5 PM", "6 PM", "7 PM", "8 PM", "9 PM", "10 PM", "11 PM"],
                    )
                )
            )
        else:
            st.error("Hour data is not available for calendar view")
            return

    st.plotly_chart(fig, use_container_width=True)

    # New component: Temperature impact summary stats
    with st.expander("Temperature Impact Statistics", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Calculate peak hours
            if "hour" in pandas_df.columns and energy_col in pandas_df.columns:
                peak_hours = pandas_df.groupby("hour")[energy_col].mean().reset_index()
                peak_hour = peak_hours.loc[peak_hours[energy_col].idxmax()]["hour"]

                st.metric(
                    label="Peak Energy Hour",
                    value=f"{int(peak_hour)}:00",
                    delta=None,
                    help="Hour of the day with highest average energy consumption"
                )

                if "avg_temperature" in pandas_df.columns:
                    avg_temp = pandas_df["avg_temperature"].mean()
                    st.metric(
                        label="Average Temperature",
                        value=f"{avg_temp:.1f} °C",
                        delta=None
                    )

        with col2:
            if "avg_humidity" in pandas_df.columns and "energy" in pandas_df.columns:
                # Calculate correlation between humidity and energy
                humidity_corr = pandas_df[["energy", "avg_humidity"]].corr().iloc[0,1]
                st.metric(
                    label="Humidity-Energy Correlation",
                    value=f"{humidity_corr:.2f}",
                    delta=None,
                    help="Correlation between humidity and energy consumption"
                )

            if "temp_bin" in pandas_df.columns:
                # Count unique temperature bins
                st.metric(
                    label="Temperature Ranges",
                    value=f"{pandas_df['temp_bin'].nunique()}",
                    delta=None,
                    help="Number of different temperature ranges in the data"
                )

    # Additional information about the datasets
    with st.expander("Hourly Temperature Impact Details", expanded=False):
        st.markdown("""
        ### Understanding the Hourly Temperature Impact
        
        This analysis shows how temperature affects energy consumption across different hours of the day:
        
        - **Temperature Bins**: Data grouped into temperature ranges
        - **Hourly Patterns**: Reveals how temperature sensitivity varies throughout the day
        - **Peak Hours**: Identify when temperature has the strongest impact on consumption
        - **Humidity Factor**: Analyze the relationship between humidity and energy use
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

    st.markdown("""
    Analyze how weather conditions affect energy consumption across different time periods.
    Use the filters in the sidebar to narrow down specific time periods or temperature ranges.
    """)

    # Create subtabs for daily and hourly analysis
    daily_tab, hourly_tab = st.tabs([
        "📆 Daily Temperature Impact",
        "⏰ Hourly Temperature Impact"
    ])

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

