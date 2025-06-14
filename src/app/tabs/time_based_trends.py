import streamlit as st
from app.utils import create_line_plot, render_years, create_bar_chart
from data import storage


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

    fig = create_line_plot(
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
    
    st.success("""
    ### 🕒 Hourly Consumption Analysis
    
    **Key observations:**
    * **Night hours (0-5)**: Low energy consumption during sleeping hours
    * **Morning peak (7-8)**: Increased usage during morning routines
    * **Daytime (9-15)**: Relatively stable consumption
    * **Evening peak (16-20)**: Highest consumption when people return home, turning on lights and appliances, having dinner
    * **Late night drop (21-23)**: Consumption tapers off as people go to bed
    These patterns typically reflect residential consumption cycles, influenced by work schedules and daylight availability.
    """)


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

    # Choose appropriate x field based on what's available
    # Yehor made some refactoring, now I do this :)
    if "weekday_name" in available_cols:
        x_field = "weekday_name"
    elif "day_of_week" in available_cols:
        x_field = "day_of_week"
    else:
        for col in ["weekday", "day"]:
            if col in available_cols:
                x_field = col
                break

    if not x_field:
        st.error("No valid day of week column found in the data")
        return

    fig = create_line_plot(
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
    
    st.success("""
    ### 📅 Daily Consumption Patterns
    
    **Key observations:**
    * **Weekday vs Weekend**: Notice the typical rise on weekends, more people stay at home using appliances
    * **Monday transitions**: Usually shows a ramp-down effect from weekend mode to work mode. Huge consumption drop
    * **Friday wind-up**: Shows higher consumption as people prepare for the weekend
    """)


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

    fig = create_line_plot(
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
    
    st.success("""
    ### 📊 Weekly Consumption Trends
    
    **Key observations:**
    * **Winter Effect**: Notice how weeks in winter months (December-February) shows higher consumption.
    * **Holiday periods**: Weeks with holidays often show distinctive patterns with spikes
    * **Lowest Consumption**: Lowest consumption of all weeks is usually seen in summer months (June-August) when people are on vacation and staying away from home. Also during summer, days are longer, so less energy is used for lighting.
    """)


def render_seasonal_plot(seasonal_data):
    col1, col2, col3 = st.columns(3)
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
            key="seasonal_metric",
        )
    with col2:
        display_type = st.radio(
            "Display view:",
            options=["Line Chart", "Bar Chart"],
            horizontal=True,
            key="seasonal_display_type",
        )
    with col3:
        group_option = st.selectbox(
            "Group by:",
            options=["Aggregated", "By Year"],
            index=1 if display_type == "Bar Chart" else 0,
            key="seasonal_group_option",
        )

    if group_option == "Aggregated":
        x_field = "season"
        color_field = "year"
    else:  # By Year
        x_field = "year"
        color_field = "season"

    if display_type == "Line Chart":
        fig = create_line_plot(
            data=seasonal_data,
            metric=metric,
            display_type=group_option,
            x_field="week",
            x_label="Week of Season",
            time_period="Seasonal",
            sort_by=["season", "week"],
            color_field="season",
            group_fields=["season", "week"],
            separate_years=True,
            extra_layout_options={
                "xaxis": dict(tickmode="linear", tick0=1, dtick=1, range=[1, 13])
            },
        )

    else:  # Bar Chart option
        fig = create_bar_chart(
            data=seasonal_data,
            metric=metric,
            group_by_fields=["year", "season"],
            x_field=x_field,
            color_field=color_field,
            title=f"Seasonal Energy Consumption by {x_field.capitalize()}",
            extra_options={
                "barmode": "group",
                "xaxis": dict(tickmode="linear"),
            },
        )

    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    ### 🌦️ Seasonal Energy Patterns
    
    **Key observations:**
    * **Winter**: Shows higher consumption in cold London winters due to heating needs and much shorter daylight hours
    * **Summer**: Expectedly shows the lowest consumption, as days are longer and people are often away on vacation with grills, pools, and outdoor activities :)
    * **Shoulder seasons**: Spring/Fall often show most moderate consumption
    * **Year-over-year**: Comparing years we can see how consumption patterns doesn't change much.
    """)


def render_weekday_vs_weekend_plot(weekday_weekend_data):
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
        key="weekday_weekend_metric",
    )

    # Since this is a horizontal bar chart (orientation="h"),
    # we need to ensure x-axis ticks (which display the values) are visible
    fig = create_bar_chart(
        data=weekday_weekend_data,
        metric=metric,
        group_by_fields=["year", "is_weekend"],
        x_field="year",
        color_field="is_weekend",
        title="Weekday vs Weekend Energy Consumption by Year",
        orientation="h",
        extra_options={"barmode": "group", "yaxis": dict(tickmode="linear", dtick=1)},
    )

    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    ### 🏠 Weekday vs Weekend Analysis
    
    **Key observations:**
    * **What is Weekday vs Weekend?**: This analysis takes into account that weekends are sometimes can be Monday-Friday, since holidays can fall on any day of the week. 
    * **Occupancy patterns**: Weekends typically show higher energy consumption as people are home more often, using appliances, cooking, and engaging in leisure activities.
    * **Year-over-year changes**: Changes in weekend/weekday ratio may indicate lifestyle or usage shifts. Interestingly enough, that 2011 shows the most consumption overall. 
    """)


def render_time_based_tab():
    filters = st.session_state.get("filters", {})
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

    # Seasonal Patterns Section
    st.markdown(f"### Seasonal Patterns | Year {render_years()}")
    with st.spinner("Loading seasonal patterns..."):
        seasonal_data = storage.get_seasonal_patterns(
            years=filters.get("years", None),
        )
    render_seasonal_plot(seasonal_data)
    with st.expander("View Dataframe", expanded=False):
        st.dataframe(seasonal_data)
    st.divider()

    # Weekday vs Weekend Patterns Section
    st.markdown(f"### Weekday vs Weekend Patterns | Year {render_years()}")
    with st.spinner("Loading weekday vs weekend patterns..."):
        weekday_weekend_data = storage.get_weekday_vs_weekend_patterns(
            years=filters.get("years", None),
        )
    render_weekday_vs_weekend_plot(weekday_weekend_data)
    with st.expander("View Dataframe", expanded=False):
        st.dataframe(weekday_weekend_data)
