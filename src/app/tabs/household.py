import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import polars as pl
from data import storage


def render_tariff_comparison(household_data):
    st.subheader("Tariff Type Comparison")

    col1, col2 = st.columns(2)

    with col1:
        metric = st.selectbox(
            "Select energy metric:",
            options=["energy_mean", "energy_median", "energy_sum", "energy_max"],
            format_func=lambda x: x.replace("energy_", "").capitalize() + " (kWh)",
            index=0,
            key="tariff_metric",
        )

    with col2:
        view_type = st.radio(
            "View by:",
            options=["ACORN Group", "Overall"],
            horizontal=True,
            key="tariff_view",
        )

    if view_type == "ACORN Group":
        df_pd = household_data.to_pandas()
        fig = px.bar(
            df_pd,
            x="Acorn_grouped",
            y=metric,
            color="stdorToU",
            barmode="group",
            color_discrete_map={"Std": "#1f77b4", "ToU": "#ff7f0e"},
            labels={
                "stdorToU": "Tariff Type",
                "Acorn_grouped": "ACORN Group",
                metric: metric.replace("energy_", "").capitalize() + " (kWh)",
            },
            title=f"Energy Consumption by ACORN Group and Tariff Type (kWh)",
        )
    else:
        summary_df = household_data.group_by("stdorToU").agg(
            [
                pl.mean(metric).alias(metric),
                pl.sum("household_count").alias("household_count"),
                pl.sum("days_count").alias("days_count"),
            ]
        )

        df_pd = summary_df.to_pandas()
        fig = px.bar(
            df_pd,
            x="stdorToU",
            y=metric,
            color="stdorToU",
            color_discrete_map={"Std": "#1f77b4", "ToU": "#ff7f0e"},
            labels={
                "stdorToU": "Tariff Type",
                metric: metric.replace("energy_", "").capitalize() + " (kWh)",
            },
            title=f"Overall Energy Consumption by Tariff Type (kWh)",
            text_auto=".2f",
        )

        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>"
            + f"{metric.replace('energy_', '').capitalize()} (kWh): %{{y:.3f}}<br>"
            + "Households: %{customdata[0]}<br>"
            + "Days measured: %{customdata[1]}",
            customdata=df_pd[["household_count", "days_count"]],
        )

    fig.update_layout(
        height=500,
        legend_title_text="Tariff Type",
        xaxis_title="ACORN Group" if view_type == "ACORN Group" else "Tariff Type",
        yaxis_title=metric.replace("energy_", "").capitalize() + " (kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success("""
    ### 🔌 Tariff Type Comparison Insights
    
    **Key observations:**
    * **Standard vs. Time-of-Use**: ToU customers generally show lower consumption compared to standard tariff users
    * **Affluent groups**: Show the highest consumption differential between tariff types, with Std tariff users consuming significantly more
    * **Behavior change**: ToU tariffs appear to encourage more efficient consumption patterns across all demographic groups
    """)


def render_acorn_analysis(household_data):
    st.subheader("ACORN Group Analysis")

    col1, col2 = st.columns(2)

    with col1:
        metric = st.selectbox(
            "Select energy metric:",
            options=["energy_mean", "energy_median", "energy_sum", "energy_max"],
            format_func=lambda x: x.replace("energy_", "").capitalize() + " (kWh)",
            index=0,
            key="acorn_metric",
        )

    with col2:
        chart_type = st.radio(
            "Chart type:",
            options=["Bar Chart", "Sunburst"],
            horizontal=True,
            key="acorn_chart_type",
        )

    if chart_type == "Bar Chart":
        acorn_summary = (
            household_data.group_by("Acorn_grouped")
            .agg(
                [
                    pl.mean(metric).alias(metric),
                    pl.sum("household_count").alias("household_count"),
                ]
            )
            .sort(metric, descending=True)
        )

        df_pd = acorn_summary.to_pandas()
        fig = px.bar(
            df_pd,
            x="Acorn_grouped",
            y=metric,
            color="Acorn_grouped",
            labels={
                "Acorn_grouped": "ACORN Group",
                metric: metric.replace("energy_", "").capitalize() + " (kWh)",
            },
            title=f"Energy Consumption by ACORN Group (kWh)",
            text="household_count",
        )

        fig.update_traces(texttemplate="%{text} households", textposition="outside")

    else:  # Sunburst
        df_pd = household_data.to_pandas()
        fig = px.sunburst(
            df_pd,
            path=["Acorn_grouped", "stdorToU"],
            values="household_count",
            color=metric,
            color_continuous_scale="Viridis",
            labels={
                "Acorn_grouped": "ACORN Group",
                "stdorToU": "Tariff Type",
                "household_count": "Number of Households",
                metric: metric.replace("energy_", "").capitalize() + " (kWh)",
            },
            title=f"Energy Consumption Distribution across ACORN Groups and Tariff Types (kWh)",
        )

    fig.update_layout(
        height=600,
        xaxis_title="ACORN Group" if chart_type == "Bar Chart" else None,
        yaxis_title=metric.replace("energy_", "").capitalize() + " (kWh)"
        if chart_type == "Bar Chart"
        else None,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success("""
    ### 👥 ACORN Group Analysis Insights
    
    **Key observations:**
    * **Affluent segments**: Show consistently higher energy consumption than other groups, while being the biggest group in terms of household count
    * **Adversity segments**: Groups K-Q (Adversity) show much lower consumption, while being top-2 in terms of household count. This means that in average, these consumers shows much lower energy consumption than the Affluent groups
    """)


def render_consumption_comparison(household_data):
    st.subheader("High vs. Low Consumers")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        metric = st.selectbox(
            "Select energy metric:",
            options=["energy_mean", "energy_median", "energy_max"],
            format_func=lambda x: x.replace("energy_", "").capitalize() + " (kWh)",
            index=0,
            key="consumption_metric",
        )

    with col2:
        n_groups = st.slider(
            "Number of quantile groups:",
            min_value=2,
            max_value=5,
            value=3,
            key="consumption_groups",
        )

    with col3:
        display_mode = st.radio(
            "Display mode:",
            options=["Combined", "Separate Metrics"],
            horizontal=True,
            key="display_mode",
        )

    df = household_data.sort(metric).to_pandas()

    # Use proper quantile-based grouping instead of equal-sized divisions
    # Default for 2 groups
    quantiles = [0, 0.5, 1.0]
    labels = ["Low Consumers", "High Consumers"]

    if n_groups == 3:
        quantiles = [0, 0.33, 0.67, 1.0]
        labels = ["Low Consumers", "Medium Consumers", "High Consumers"]
    elif n_groups == 4:
        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        labels = [
            "Low Consumers",
            "Medium-Low Consumers",
            "Medium-High Consumers",
            "High Consumers",
        ]
    elif n_groups == 5:
        quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = [
            "Very Low Consumers",
            "Low Consumers",
            "Medium Consumers",
            "High Consumers",
            "Very High Consumers",
        ]

    df["quantile_group"] = pd.qcut(df[metric], q=quantiles, labels=labels)

    summary_stats = (
        df.groupby("quantile_group")
        .agg(
            {
                "energy_mean": "mean",
                "energy_median": "mean",
                "energy_max": "mean",
                "energy_sum": "mean",
                "household_count": "sum",
                "stdorToU": lambda x: (x == "ToU").mean() * 100,  # Percentage of ToU
            }
        )
        .reset_index()
    )

    summary_stats["order"] = summary_stats["quantile_group"].map(
        {label: i for i, label in enumerate(labels)}
    )
    summary_stats = summary_stats.sort_values(by="order")
    summary_stats["stdorToU"] = summary_stats["stdorToU"].round(1).astype(str) + "% ToU"

    # Create visualization of consumer groups
    if display_mode == "Combined":
        fig1 = go.Figure()

        fig1.add_trace(
            go.Bar(
                x=summary_stats["quantile_group"],
                y=summary_stats["household_count"],
                name=metric.replace("energy_", "").capitalize() + " (kWh)",
                text=summary_stats["household_count"].astype(str)
                + " households<br>"
                + summary_stats["stdorToU"],
                textposition="auto",
            )
        )

        fig1.update_layout(
            title="Energy Consumption Metrics by Consumer Group (kWh)",
            xaxis_title="Consumer Group",
            yaxis_title="Amount of Households",
            legend_title="Metric",
            barmode="group",
            height=500,
        )
    else:
        fig1 = px.bar(
            summary_stats,
            x="quantile_group",
            y=["energy_mean", "energy_median"],
            barmode="group",
            labels={
                "quantile_group": "Consumer Group",
                "value": "Energy Consumption (kWh)",
                "variable": "Metric",
            },
            title="Mean and Median Energy Consumption by Consumer Group",
            text_auto=".3f",
        )

        for i, row in summary_stats.iterrows():
            fig1.add_annotation(
                x=row["quantile_group"],
                y=row["energy_mean"] / 2,
                text=f"{int(row['household_count'])} households<br>{row['stdorToU']}",
                showarrow=False,
                font=dict(size=12, color="white"),
                bgcolor="rgba(20,20,20, 0.8)",
                borderwidth=4,
            )

        fig1.update_layout(height=500)

    st.plotly_chart(fig1, use_container_width=True)

    st.success("""
    ### ⚡ Consumption Group Insights
    
    **What you're seeing in the chart:**
    * The consumers are divided into quantile groups based on their energy consumption
    * Each bar shows the average consumption metrics for households in that group
    * The annotations show how many households are in each group and what percentage use Time-of-Use tariffs
    
    **Key observations:**
    * **ToU adoption pattern**: Lower consumers tend to have higher Time-of-Use tariff adoption (possibly more cost-conscious consumers)
    * **Distribution skew**: The data shows that most households in low to medium consumption ranges and fewer very high consumers
    """)

    # Distribution of ACORN groups within each consumer group
    acorn_dist = (
        df.groupby(["quantile_group", "Acorn_grouped"]).size().reset_index(name="count")
    )

    # Calculate percentages within each consumption group
    total_by_group = acorn_dist.groupby("quantile_group")["count"].transform("sum")
    acorn_dist["percentage"] = (acorn_dist["count"] / total_by_group * 100).round(1)

    # Ensure groups are displayed in the correct order
    acorn_dist["order"] = acorn_dist["quantile_group"].map(
        {label: i for i, label in enumerate(labels)}
    )
    acorn_dist = acorn_dist.sort_values(
        by=["order", "percentage"], ascending=[True, False]
    )

    fig2 = px.bar(
        acorn_dist,
        x="quantile_group",
        y="percentage",
        color="Acorn_grouped",
        labels={
            "quantile_group": "Consumer Group",
            "percentage": "Percentage",
            "Acorn_grouped": "ACORN Group",
            "count": "Count",
        },
        title="ACORN Group Distribution by Consumer Group",
        text="percentage",
    )

    fig2.update_traces(texttemplate="%{text}%", textposition="inside")

    fig2.update_layout(
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.success("""
    ### 📊 ACORN Distribution Within Consumer Groups
    
    **What you're seeing in the chart:**
    * This chart shows the socioeconomic makeup of each consumption group
    * The percentages represent what portion of each consumer group belongs to a specific ACORN category
    
    **Key observations:**
    * **Economic correlation**: There's a clear socioeconomic gradient in energy consumption, with affluent households dominating high consumption groups. It is logically expected that the affluent households consume more energy, since they live in bigger houses, have more appliances, etc.
    * **Adversity in low consumption**: Groups K-Q (lower income and financial adversity) make up the majority of low consumers
    * **Mixed middle**: Medium consumer groups show a more balanced mix of socioeconomic backgrounds. This suggests that middle-income households have more diverse energy consumption patterns, possibly due to varying household sizes and lifestyles.
    """)


def render_household_distribution(household_data):
    st.subheader("Household Distribution")

    col1, col2 = st.columns(2)

    with col1:
        viz_type = st.radio(
            "Visualization type:",
            options=["Bubble Chart", "Scatter Plot"],
            horizontal=True,
            key="household_viz_type",
        )

    # Convert to pandas for plotting with Plotly
    df_pd = household_data.to_pandas()

    # Create figure based on selected visualization type
    if viz_type == "Bubble Chart":
        fig = px.scatter(
            df_pd,
            x="energy_mean",
            y="energy_max",
            size="household_count",
            color="stdorToU",
            hover_name="Acorn_grouped",
            color_discrete_map={"Std": "#1f77b4", "ToU": "#ff7f0e"},
            labels={
                "energy_mean": "Mean Energy Consumption (kWh)",
                "energy_max": "Maximum Energy Consumption (kWh)",
                "household_count": "Number of Households",
                "stdorToU": "Tariff Type",
            },
            title="Household Distribution by Energy Consumption (kWh)",
        )
    else:
        fig = px.scatter(
            df_pd,
            x="energy_median",
            y="energy_mean",
            color="Acorn_grouped",
            symbol="stdorToU",
            size="household_count",
            hover_data=["days_count"],
            labels={
                "energy_median": "Median Energy Consumption (kWh)",
                "energy_mean": "Mean Energy Consumption (kWh)",
                "Acorn_grouped": "ACORN Group",
                "stdorToU": "Tariff Type",
                "household_count": "Number of Households",
                "days_count": "Days Measured",
            },
            title="Energy Consumption Patterns across Household Groups (kWh)",
        )

    fig.update_layout(
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success("""
    ### 🔍 Household Distribution Insights
    
    **What you're seeing in the chart:**
    * Each point represents a group of households with the same ACORN group and tariff type
    * The position shows their mean and median/maximum energy consumption
    * The size of each bubble indicates how many households are in that group
    * Colors differentiate between tariff types or ACORN groups
    
    **Key observations:**
    * **Correlation patterns**: There's a strong positive correlation between mean and median/maximum consumption metrics
    * **Tariff clustering**: Time-of-Use (ToU) households tend to cluster in lower consumption regions compared to Standard tariff households
    * **ACORN segregation**: Clear separation of ACORN groups, with Affluent (A-E) consistently in higher consumption regions
    * **Outlier behavior**: Some household groups show unusually high maximum consumption despite moderate mean values, indicating occasional high usage spikes
    """)


def render_household_tab():
    st.header("📊 Household Energy Consumption Analysis")

    with st.spinner("Loading household data..."):
        household_data = storage.get_household_patterns()

    # Apply filters
    filters = st.session_state.get("filters", {})

    if filters.get("tariff_type"):
        household_data = household_data.filter(
            pl.col("stdorToU").is_in(filters["tariff_type"])
        )

    if filters.get("acorn"):
        household_data = household_data.filter(pl.col("Acorn").is_in(filters["acorn"]))

    if household_data.height == 0:
        st.warning(
            "No data available with the current filters. Please adjust your filters."
        )
        return

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Households", household_data.select(pl.sum("household_count")).item()
        )
    with col2:
        st.metric(
            "Tariff Types", household_data.select(pl.col("stdorToU").n_unique()).item()
        )
    with col3:
        st.metric(
            "ACORN Groups",
            household_data.select(pl.col("Acorn_grouped").n_unique()).item(),
        )
    with col4:
        avg_energy = household_data.select(pl.mean("energy_mean")).item()
        st.metric("Avg. Energy Consumption", f"{avg_energy:.3f} kWh")

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Tariff Comparison",
            "🏘️ ACORN Analysis",
            "⚡ Consumption Groups",
            "🔍 Household Distribution",
        ]
    )
    
    with tab1:
        render_tariff_comparison(household_data)

    with tab2:
        render_acorn_analysis(household_data)

    with tab3:
        render_consumption_comparison(household_data)

    with tab4:
        render_household_distribution(household_data)
    
    # Show raw data in an expander
    with st.expander("View Raw Data", expanded=False):
        st.dataframe(household_data)
