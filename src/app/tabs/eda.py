import streamlit as st
from data import storage
from app.utils import create_energy_plot, render_years


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

    fig = create_energy_plot(
        data=hourly_data,
        metric=metric,
        display_type=display_type,
        x_field="hour",
        x_label="Hour of Day",
        time_period="Hourly",
        sort_by="hour",
        extra_layout_options={
            "xaxis": dict(tickmode="linear", tick0=0, dtick=1, range=[0, 23])
        },
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

    # Print available columns to help debug
    available_cols = daily_data.columns

    x_field = None
    sort_field = None
    group_fields = None

    # Choose appropriate x field based on what's available
    if "weekday_name" in available_cols:
        x_field = "weekday_name"
    elif "day_of_week" in available_cols:
        x_field = "day_of_week"
    else:
        # Fallback to integer weekday if no name column exists
        for col in ["weekday", "day"]:
            if col in available_cols:
                x_field = col
                break

    if not x_field:
        st.error("No valid day of week column found in the data")
        return

    fig = create_energy_plot(
        data=daily_data,
        metric=metric,
        display_type=display_type,
        x_field=x_field,
        x_label="Day of Week",
        time_period="Daily",
        sort_by="weekday",
        group_fields=["weekday", "weekday_name"]
        if display_type == "Aggregated"
        else None,
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

    fig = create_energy_plot(
        data=weekly_data,
        metric=metric,
        display_type=display_type,
        x_field="week",
        x_label="Week of Year",
        time_period="Weekly",
        sort_by=["year", "week"] if display_type == "By Year" else "week",
        extra_layout_options={
            "xaxis": dict(tickmode="linear", tick0=1, dtick=4, range=[1, 53])
        },
    )

    st.plotly_chart(fig, use_container_width=True)


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
