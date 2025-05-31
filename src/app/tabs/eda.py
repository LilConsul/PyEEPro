import streamlit as st
import polars as pl
from data import storage

import plotly.express as px


def render_hourly_plot(hourly_data):
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
            ],
            format_func=lambda x: x.replace("energy_", "").capitalize(),
            index=0,
        )
    with col2:
        display_type = st.radio(
            "Display view:", options=["Aggregated", "By Year"], horizontal=True
        )

    # Define the unit for energy measurements
    energy_unit = "kWh"

    if display_type == "Aggregated":
        hourly_agg = (
            hourly_data.group_by("hour")
            .agg(**{f"{metric}": pl.col(metric).mean()})
            .sort("hour")
        )

        df = hourly_agg.to_pandas()

        fig = px.line(
            df,
            x="hour",
            y=metric,
            labels={
                "hour": "Hour of Day",
                metric: f"{metric.replace('energy_', '').capitalize()} Energy Consumption ({energy_unit})",
            },
            title=f"Hourly {metric.replace('energy_', '').capitalize()} Energy Consumption ({energy_unit})",
            markers=True,
        )

    else:  # By Year
        df = (
            hourly_data.select(["year", "hour", metric])
            .sort(["year", "hour"])
            .to_pandas()
        )

        fig = px.line(
            df,
            x="hour",
            y=metric,
            color="year",
            labels={
                "hour": "Hour of Day",
                metric: f"{metric.replace('energy_', '').capitalize()} Energy Consumption ({energy_unit})",
                "year": "Year",
            },
            title=f"Hourly {metric.replace('energy_', '').capitalize()} Energy Consumption by Year ({energy_unit})",
            markers=True,
        )

    fig.update_layout(
        xaxis=dict(tickmode="linear", tick0=0, dtick=1, range=[0, 23]),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_daily_plot(daily_data):
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
            ],
            format_func=lambda x: x.replace("energy_", "").capitalize(),
            index=0,
            key="daily_metric",
        )
    with col2:
        display_type = st.radio(
            "Display view:",
            options=["Aggregated", "By Year"],
            horizontal=True,
            key="daily_display_type",
        )

    # Define the unit for energy measurements
    energy_unit = "kWh"

    if display_type == "Aggregated":
        daily_agg = (
            daily_data.group_by("weekday", "weekday_name")
            .agg(**{f"{metric}": pl.col(metric).mean()})
            .sort("weekday")
        )

        df = daily_agg.to_pandas()

        fig = px.line(
            df,
            x="weekday_name",
            y=metric,
            labels={
                "weekday_name": "Day of Week",
                metric: f"{metric.replace('energy_', '').capitalize()} Energy Consumption ({energy_unit})",
            },
            title=f"Daily {metric.replace('energy_', '').capitalize()} Energy Consumption ({energy_unit})",
            markers=True,
        )

    else:  # By Year
        df = (
            daily_data.select(["year", "weekday", "weekday_name", metric])
            .sort(["year", "weekday"])
            .to_pandas()
        )

        fig = px.line(
            df,
            x="weekday_name",
            y=metric,
            color="year",
            labels={
                "weekday_name": "Day of Week",
                metric: f"{metric.replace('energy_', '').capitalize()} Energy Consumption ({energy_unit})",
                "year": "Year",
            },
            title=f"Daily {metric.replace('energy_', '').capitalize()} Energy Consumption by Year ({energy_unit})",
            markers=True,
        )

    fig.update_layout(
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_weekly_plot(weekly_data):
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
            ],
            format_func=lambda x: x.replace("energy_", "").capitalize(),
            index=0,
            key="weekly_metric",
        )
    with col2:
        display_type = st.radio(
            "Display view:",
            options=["Aggregated", "By Year"],
            horizontal=True,
            key="weekly_display_type",
        )

    # Define the unit for energy measurements
    energy_unit = "kWh"

    if display_type == "Aggregated":
        weekly_agg = (
            weekly_data.group_by("week")
            .agg(**{f"{metric}": pl.col(metric).mean()})
            .sort("week")
        )

        df = weekly_agg.to_pandas()

        fig = px.line(
            df,
            x="week",
            y=metric,
            labels={
                "week": "Week of Year",
                metric: f"{metric.replace('energy_', '').capitalize()} Energy Consumption ({energy_unit})",
            },
            title=f"Weekly {metric.replace('energy_', '').capitalize()} Energy Consumption ({energy_unit})",
            markers=True,
        )

    else:  # By Year
        df = (
            weekly_data.select(["year", "week", metric])
            .sort(["year", "week"])
            .to_pandas()
        )

        fig = px.line(
            df,
            x="week",
            y=metric,
            color="year",
            labels={
                "week": "Week of Year",
                metric: f"{metric.replace('energy_', '').capitalize()} Energy Consumption ({energy_unit})",
                "year": "Year",
            },
            title=f"Weekly {metric.replace('energy_', '').capitalize()} Energy Consumption by Year ({energy_unit})",
            markers=True,
        )

    fig.update_layout(
        xaxis=dict(tickmode="linear", tick0=1, dtick=4, range=[1, 53]),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_years() -> str:
    """
    Render the selected years from the session state.

    Returns:
        A string representation of selected years or "All" if none are selected.
    """
    return (
        str(st.session_state.get("filters", {}).get("years", "")).strip("[]") or "All"
    )


def render_eda_tab():
    filters = st.session_state.get("filters", {})
    st.header("📊 Exploratory Data Analysis")
    time_based_tab, household_tab, weather_tab = st.tabs(
        ["📈 Time-based trends", "📊 Household behavior", "📉 Weather impact"]
    )

    with time_based_tab:
        st.subheader("Time-based Energy Consumption Patterns")

        # Hourly Patterns Section
        st.markdown(f"### Hourly Patterns | Year {render_years()}")
        with st.spinner("Loading hourly patterns..."):
            hourly_data = storage.get_hourly_patterns(
                years=filters.get("years", None),
            )
        render_hourly_plot(hourly_data)
        with st.expander("View Dataframe", expanded=False):
            st.dataframe(hourly_data)
        st.divider()

        # Daily Patterns Section
        st.markdown(f"### Daily Patterns | Year {render_years()}")
        with st.spinner("Loading daily patterns..."):
            daily_data = storage.get_daily_patterns(
                years=filters.get("years", None),
            )
        render_daily_plot(daily_data)
        with st.expander("View Dataframe", expanded=False):
            st.dataframe(daily_data)
        st.divider()

        # Weekly Patterns Section
        st.markdown(f"### Weekly Patterns | Year {render_years()}")
        with st.spinner("Loading weekly patterns..."):
            weekly_data = storage.get_weekly_patterns(
                years=filters.get("years", None),
            )
        render_weekly_plot(weekly_data)
        with st.expander("View Dataframe", expanded=False):
            st.dataframe(weekly_data)
        st.divider()
