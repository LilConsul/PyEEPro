import polars as pl
import streamlit as st
import plotly.express as px


def create_energy_plot(
    data,
    metric,
    display_type,
    x_field,
    x_label,
    time_period="",
    sort_by=None,
    group_fields=None,
    extra_layout_options=None,
    energy_unit="kWh",
):
    """
    Generic function to create energy consumption plots

    Args:
        data: Polars DataFrame with the data
        metric: Energy metric column to plot
        display_type: "Aggregated" or "By Year"
        x_field: Field to use for x-axis
        x_label: Label for x-axis
        time_period: Time period description (Hourly, Daily, Weekly)
        sort_by: Field(s) to sort by
        group_fields: Fields to group by for aggregation
        extra_layout_options: Additional plotly layout options
        energy_unit: Unit of energy measurement (default is "kWh")
    """
    metric_display = metric.replace("energy_", "").capitalize()

    if sort_by is not None:
        # For list of sort columns
        if isinstance(sort_by, list):
            valid_sort_cols = [col for col in sort_by if col in data.columns]
            sort_by = valid_sort_cols if valid_sort_cols else None
        # For single column
        elif sort_by not in data.columns:
            sort_by = None

    # Validate group_fields columns exist in the data
    if group_fields:
        group_fields = [col for col in group_fields if col in data.columns]
        if not group_fields:  # If all specified group fields are invalid
            group_fields = [x_field] if x_field in data.columns else None

    # Setup plot configuration based on display type
    if display_type == "Aggregated":
        if group_fields is None:
            group_fields = [x_field] if x_field in data.columns else []

        if not group_fields:
            st.error(f"No valid grouping fields found. Expected {x_field} in the data.")
            return px.line()

        agg_data = data.group_by(*group_fields).agg(
            **{f"{metric}": pl.col(metric).mean()}
        )

        if sort_by:
            agg_data = agg_data.sort(sort_by)

        df = agg_data.to_pandas()

        fig = px.line(
            df,
            x=x_field,
            y=metric,
            labels={
                x_field: x_label,
                metric: f"{metric_display} Energy Consumption ({energy_unit})",
            },
            title=f"{time_period} {metric_display} Energy Consumption ({energy_unit})",
            markers=True,
        )
    else:  # By Year
        # Select necessary columns and sort
        select_fields = ["year", x_field, metric]
        if sort_by:
            select_fields.append(sort_by)

        if isinstance(x_field, list) or group_fields:
            select_fields = list(
                set(
                    ["year"]
                    + (group_fields or [])
                    + ([x_field] if isinstance(x_field, str) else x_field)
                    + [metric]
                )
            )

        # Filter out columns that don't need to be shown
        select_fields = [col for col in select_fields if col in data.columns]

        df = data.select(select_fields)

        if sort_by:
            try:
                df = df.sort(sort_by)
            except Exception as e:
                st.warning(f"{df=} Error sorting data: {e}")

        df = df.to_pandas()

        fig = px.line(
            df,
            x=x_field,
            y=metric,
            color="year",
            labels={
                x_field: x_label,
                metric: f"{metric_display} Energy Consumption ({energy_unit})",
                "year": "Year",
            },
            title=f"{time_period} {metric_display} Energy Consumption by Year ({energy_unit})",
            markers=True,
        )

    # Apply basic layout
    layout_options = {"hovermode": "x unified"}

    if extra_layout_options:
        layout_options.update(extra_layout_options)

    fig.update_layout(**layout_options)

    return fig
