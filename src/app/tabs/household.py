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
            key="tariff_metric"
        )
    
    with col2:
        view_type = st.radio(
            "View by:",
            options=["ACORN Group", "Overall"],
            horizontal=True,
            key="tariff_view"
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
                metric: metric.replace("energy_", "").capitalize() + " (kWh)"
            },
            title=f"Energy Consumption by ACORN Group and Tariff Type (kWh)"
        )
    else:
        summary_df = household_data.group_by("stdorToU").agg([
            pl.mean(metric).alias(metric),
            pl.sum("household_count").alias("household_count"),
            pl.sum("days_count").alias("days_count")
        ])
        
        df_pd = summary_df.to_pandas()
        fig = px.bar(
            df_pd,
            x="stdorToU",
            y=metric,
            color="stdorToU",
            color_discrete_map={"Std": "#1f77b4", "ToU": "#ff7f0e"},
            labels={
                "stdorToU": "Tariff Type",
                metric: metric.replace("energy_", "").capitalize() + " (kWh)"
            },
            title=f"Overall Energy Consumption by Tariff Type (kWh)",
            text_auto='.2f'
        )
        
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>" +
                        f"{metric.replace('energy_', '').capitalize()} (kWh): %{{y:.3f}}<br>" +
                        "Households: %{customdata[0]}<br>" +
                        "Days measured: %{customdata[1]}",
            customdata=df_pd[["household_count", "days_count"]]
        )
    
    fig.update_layout(
        height=500,
        legend_title_text="Tariff Type",
        xaxis_title="ACORN Group" if view_type == "ACORN Group" else "Tariff Type",
        yaxis_title=metric.replace("energy_", "").capitalize() + " (kWh)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_acorn_analysis(household_data):
    st.subheader("ACORN Group Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        metric = st.selectbox(
            "Select energy metric:",
            options=["energy_mean", "energy_median", "energy_sum", "energy_max"],
            format_func=lambda x: x.replace("energy_", "").capitalize() + " (kWh)",
            index=0,
            key="acorn_metric"
        )
    
    with col2:
        chart_type = st.radio(
            "Chart type:",
            options=["Bar Chart", "Sunburst"],
            horizontal=True,
            key="acorn_chart_type"
        )
    
    if chart_type == "Bar Chart":
        acorn_summary = household_data.group_by("Acorn_grouped").agg([
            pl.mean(metric).alias(metric),
            pl.sum("household_count").alias("household_count")
        ]).sort(metric, descending=True)
        
        df_pd = acorn_summary.to_pandas()
        fig = px.bar(
            df_pd,
            x="Acorn_grouped",
            y=metric,
            color="Acorn_grouped",
            labels={
                "Acorn_grouped": "ACORN Group",
                metric: metric.replace("energy_", "").capitalize() + " (kWh)"
            },
            title=f"Energy Consumption by ACORN Group (kWh)",
            text="household_count"
        )
        
        fig.update_traces(
            texttemplate="%{text} households",
            textposition="outside"
        )
        
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
                metric: metric.replace("energy_", "").capitalize() + " (kWh)"
            },
            title=f"Energy Consumption Distribution across ACORN Groups and Tariff Types (kWh)"
        )
    
    fig.update_layout(
        height=600,
        xaxis_title="ACORN Group" if chart_type == "Bar Chart" else None,
        yaxis_title=metric.replace("energy_", "").capitalize() + " (kWh)" if chart_type == "Bar Chart" else None
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_consumption_comparison(household_data):
    st.subheader("High vs. Low Consumers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        metric = st.selectbox(
            "Select energy metric:",
            options=["energy_mean", "energy_median", "energy_max"],
            format_func=lambda x: x.replace("energy_", "").capitalize() + " (kWh)",
            index=0,
            key="consumption_metric"
        )
    
    with col2:
        n_groups = st.slider("Number of quantile groups:", min_value=2, max_value=5, value=3, key="consumption_groups")
    
    # Using pandas for this section as quantile grouping is more straightforward
    df = household_data.sort(metric).to_pandas()
    group_size = len(df) // n_groups
    remainder = len(df) % n_groups
    
    consumption_groups = []
    start_idx = 0
    
    for i in range(n_groups):
        end_idx = start_idx + group_size + (1 if i < remainder else 0)
        group_df = df.iloc[start_idx:end_idx].copy()
        group_name = f"Q{i+1}" if n_groups <= 3 else f"Q{i+1}/{n_groups}"
        
        if i == 0:
            group_label = "Low Consumers"
        elif i == n_groups - 1:
            group_label = "High Consumers"
        else:
            group_label = f"Medium Consumers{' ' + str(i) if n_groups > 3 else ''}"
            
        group_df["consumption_group"] = group_label
        group_df["quantile"] = group_name
        consumption_groups.append(group_df)
        start_idx = end_idx
    
    consumption_df = pd.concat(consumption_groups)
    
    # Create summary statistics for each consumption group
    summary_stats = consumption_df.groupby("consumption_group").agg({
        "energy_mean": "mean",
        "energy_median": "mean",
        "energy_max": "mean",
        "energy_sum": "mean",
        "household_count": "sum",
        "stdorToU": lambda x: (x == "ToU").mean() * 100  # Percentage of ToU
    }).reset_index()
    
    summary_stats = summary_stats.sort_values(by="energy_mean")
    summary_stats["stdorToU"] = summary_stats["stdorToU"].round(1).astype(str) + "% ToU"
    
    # Create visualization of consumer groups
    fig1 = go.Figure()
    
    metrics = ["energy_mean", "energy_median", "energy_max"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for i, met in enumerate(metrics):
        fig1.add_trace(go.Bar(
            x=summary_stats["consumption_group"],
            y=summary_stats[met],
            name=met.replace("energy_", "").capitalize() + " (kWh)",
            marker_color=colors[i],
            text=summary_stats["household_count"].astype(str) + " households<br>" + summary_stats["stdorToU"],
            textposition="auto"
        ))
    
    fig1.update_layout(
        title="Energy Consumption Metrics by Consumer Group (kWh)",
        xaxis_title="Consumer Group",
        yaxis_title="Energy Value (kWh)",
        legend_title="Metric",
        barmode="group",
        height=500
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Distribution of ACORN groups within each consumer group
    acorn_dist = consumption_df.groupby(["consumption_group", "Acorn_grouped"]).size().reset_index(name="count")
    
    # Calculate percentages within each consumption group
    total_by_group = acorn_dist.groupby("consumption_group")["count"].transform("sum")
    acorn_dist["percentage"] = (acorn_dist["count"] / total_by_group * 100).round(1)
    
    fig2 = px.bar(
        acorn_dist,
        x="consumption_group",
        y="percentage",
        color="Acorn_grouped",
        labels={
            "consumption_group": "Consumer Group",
            "percentage": "Percentage",
            "Acorn_grouped": "ACORN Group",
            "count": "Count"
        },
        title="ACORN Group Distribution by Consumer Group",
        text="percentage"
    )
    
    fig2.update_traces(
        texttemplate="%{text}%",
        textposition="inside"
    )
    
    fig2.update_layout(
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig2, use_container_width=True)


def render_household_distribution(household_data):
    st.subheader("Household Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        viz_type = st.radio(
            "Visualization type:",
            options=["Bubble Chart", "Scatter Plot"],
            horizontal=True,
            key="household_viz_type"
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
                "stdorToU": "Tariff Type"
            },
            title="Household Distribution by Energy Consumption (kWh)"
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
                "days_count": "Days Measured"
            },
            title="Energy Consumption Patterns across Household Groups (kWh)"
        )
    
    fig.update_layout(
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


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
        household_data = household_data.filter(
            pl.col("Acorn").is_in(filters["acorn"])
        )
    
    if household_data.height == 0:
        st.warning("No data available with the current filters. Please adjust your filters.")
        return
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Households", household_data.select(pl.sum("household_count")).item())
    with col2:
        st.metric("Tariff Types", household_data.select(pl.col("stdorToU").n_unique()).item())
    with col3:
        st.metric("ACORN Groups", household_data.select(pl.col("Acorn_grouped").n_unique()).item())
    with col4:
        avg_energy = household_data.select(pl.mean("energy_mean")).item()
        st.metric("Avg. Energy Consumption", f"{avg_energy:.3f} kWh")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Tariff Comparison",
        "🏘️ ACORN Analysis",
        "⚡ Consumption Groups",
        "🔍 Household Distribution"
    ])
    
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
