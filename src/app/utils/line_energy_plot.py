import streamlit as st
import polars as pl
import plotly.express as px


def validate_columns(data, columns, default=None):
    """
    Validate that columns exist in the dataframe.

    Args:
        data: Polars DataFrame
        columns: Single column name or list of column names to validate
        default: Default value to return if columns are invalid

    Returns:
        List of valid columns or default value
    """
    if columns is None:
        return default

    if isinstance(columns, str):
        return [columns] if columns in data.columns else default

    valid_cols = [col for col in columns if col in data.columns]
    return valid_cols if valid_cols else default


def prepare_aggregated_data(data, x_field, metric, group_fields, sort_by):
    """
    Prepare data for aggregated display type.

    Args:
        data: Polars DataFrame
        x_field: Field for x-axis
        metric: Energy metric to plot
        group_fields: Fields to group by
        sort_by: Fields to sort by

    Returns:
        Pandas DataFrame with aggregated data
    """
    if group_fields is None:
        group_fields = [x_field] if x_field in data.columns else []

    if not group_fields:
        st.error(f"No valid grouping fields found. Expected {x_field} in the data.")
        return None

    agg_data = data.group_by(*group_fields).agg(**{f"{metric}": pl.col(metric).mean()})

    if sort_by:
        agg_data = agg_data.sort(sort_by)

    return agg_data.to_pandas()


def prepare_by_year_data(data, x_field, metric, group_fields, sort_by, color_field):
    """
    Prepare data for by-year display type.

    Args:
        data: Polars DataFrame
        x_field: Field for x-axis
        metric: Energy metric to plot
        group_fields: Fields to group by
        sort_by: Fields to sort by
        color_field: Field for color differentiation

    Returns:
        Pandas DataFrame with filtered data
    """
    select_fields = ["year", x_field, metric]
    if isinstance(x_field, list) or group_fields:
        select_fields = list(
            set(
                ["year"]
                + (group_fields or [])
                + ([x_field] if isinstance(x_field, str) else x_field)
                + [metric]
                + (sort_by if isinstance(sort_by, list) else [sort_by])
                + ([color_field] if color_field else [])
            )
        )

    select_fields = [col for col in select_fields if col in data.columns]
    df = data.select(select_fields)

    if sort_by:
        try:
            df = df.sort(sort_by)
        except Exception as e:
            st.warning(f"Error sorting data: {e}")

    return df.to_pandas()


def create_aggregated_plot(
    df,
    x_field,
    metric,
    metric_display,
    time_period,
    energy_unit,
    color_field,
    x_label=None,
):
    """
    Create a plot for aggregated data.

    Args:
        df: Pandas DataFrame with aggregated data
        x_field: Field for x-axis
        metric: Energy metric to plot
        metric_display: Display name for the metric
        time_period: Time period description
        energy_unit: Unit of energy measurement
        color_field: Field for color differentiation
        x_label: Custom label for x-axis (overrides default)

    Returns:
        Plotly figure
    """
    # Use custom x_label if provided, otherwise create a default label
    x_axis_label = (
        x_label
        if x_label
        else (x_field.capitalize() if isinstance(x_field, str) else "Value")
    )

    plot_kwargs = {
        "x": x_field,
        "y": metric,
        "markers": True,
        "labels": {
            x_field: x_axis_label,
            metric: f"{metric_display} Energy Consumption ({energy_unit})",
        },
        "title": f"{time_period} {metric_display} Energy Consumption ({energy_unit})",
    }

    # Use color_field if provided and exists in dataframe
    if color_field and color_field in df.columns:
        plot_kwargs["color"] = color_field
        plot_kwargs["labels"][color_field] = color_field.capitalize()

    return px.line(df, **plot_kwargs)


def create_by_year_plot(
    df,
    x_field,
    metric,
    metric_display,
    time_period,
    energy_unit,
    color_field,
    separate_years,
    x_label=None,
):
    """
    Create a plot for by-year data.

    Args:
        df: Pandas DataFrame with filtered data
        x_field: Field for x-axis
        metric: Energy metric to plot
        metric_display: Display name for the metric
        time_period: Time period description
        energy_unit: Unit of energy measurement
        color_field: Field for color differentiation
        separate_years: Whether to create facet plots by year
        x_label: Custom label for x-axis (overrides default)

    Returns:
        Plotly figure
    """
    years = sorted(df["year"].unique())
    title = f"{time_period} {metric_display} Energy Consumption by Year ({energy_unit})"

    # Use custom x_label if provided
    x_axis_label = (
        x_label
        if x_label
        else (x_field.capitalize() if isinstance(x_field, str) else "Value")
    )

    labels = {
        x_field: x_axis_label,
        metric: f"{metric_display} Energy Consumption ({energy_unit})",
        "year": "Year",
    }
    if color_field:
        labels[color_field] = color_field.capitalize()

    # Calculate y-axis range to fit the data
    y_min = max(0, df[metric].min() * 0.9)
    y_max = df[metric].max() * 1.1
    if y_max - y_min < 0.1:
        y_padding = 0.05
        y_min = max(0, df[metric].min() - y_padding)
        y_max = df[metric].max() + y_padding

    # Create the appropriate plot type
    if separate_years:
        num_cols = 2
        num_rows = (len(years) + num_cols - 1) // num_cols
        plot_height = max(500, 300 * num_rows)

        fig = px.line(
            df,
            x=x_field,
            y=metric,
            color=color_field if color_field else None,
            facet_col="year",
            facet_col_wrap=num_cols,
            category_orders={"year": years},
            labels=labels,
            title=title,
            markers=True,
            height=plot_height,
        )

        # Enhance facet plot appearance
        fig.for_each_annotation(
            lambda a: a.update(text=f"<b>Year {a.text.split('=')[-1]}</b>")
        )

        fig.update_layout(
            margin=dict(t=80, b=40, l=40, r=20),
            plot_bgcolor="white",
            legend_title_text="",
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5
            ),
        )

        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0,0,0,0.1)",
            range=[y_min, y_max],
        )

        # Adjust x-axis tick angle if needed
        fig.update_xaxes(
            tickangle=45
            if isinstance(x_field, str) and len(df[x_field].unique()) > 6
            else 0
        )
    else:
        # Create regular line plot with year as color
        fig = px.line(
            df,
            x=x_field,
            y=metric,
            color="year",
            category_orders={"year": years},
            labels=labels,
            title=title,
            markers=True,
        )

        # Apply y-axis range
        fig.update_yaxes(range=[y_min, y_max])

    return fig


def apply_layout_options(fig, extra_options=None):
    """
    Apply layout options to the plot.

    Args:
        fig: Plotly figure
        extra_options: Additional layout options

    Returns:
        Plotly figure with updated layout
    """

    layout_options = {"hovermode": "x unified"}

    if extra_options:
        layout_options.update(extra_options)

    fig.update_layout(**layout_options)

    return fig


def create_line_plot(
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
    color_field=None,
    separate_years=False,
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
        color_field: Field to use for color differentiation (for Aggregated view)
        separate_years: If True, create a facet plot with separate subplots by year
    """
    # Derive display name for the metric
    metric_display = metric.replace("energy_", "").capitalize()

    # Validate sort columns
    sort_by = validate_columns(data, sort_by)

    # Validate group fields
    group_fields = validate_columns(
        data, group_fields, default=[x_field] if x_field in data.columns else None
    )

    if display_type == "Aggregated":
        df = prepare_aggregated_data(data, x_field, metric, group_fields, sort_by)
        if df is None:
            return px.line()  # Return empty plot if data preparation failed
        fig = create_aggregated_plot(
            df,
            x_field,
            metric,
            metric_display,
            time_period,
            energy_unit,
            color_field,
            x_label,
        )
    else:  # By Year
        df = prepare_by_year_data(
            data, x_field, metric, group_fields, sort_by, color_field
        )
        fig = create_by_year_plot(
            df,
            x_field,
            metric,
            metric_display,
            time_period,
            energy_unit,
            color_field,
            separate_years,
            x_label,
        )

    return apply_layout_options(fig, extra_layout_options)
