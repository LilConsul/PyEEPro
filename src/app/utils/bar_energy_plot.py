import plotly.express as px
import polars as pl


def create_bar_chart(
    data,
    metric=None,
    group_by_fields=None,
    x_field=None,
    y_field=None,
    y_display=None,
    color_field=None,
    x_label=None,
    y_label=None,
    title=None,
    orientation="v",
    extra_options=None,
    energy_unit="kWh",
    show_grid=True,
):
    """
    Create a bar chart using Plotly Express with internal data preparation.

    Args:
        data: Pandas DataFrame or Polars DataFrame with data
        metric: Energy metric to display (e.g., "energy_mean")
        group_by_fields: List of fields to group by before aggregation
        x_field: Field for x-axis
        y_field: Field for y-axis (height of bars), only needed if metric is None
        y_display: Display name for y-axis (optional)
        color_field: Field for color differentiation (optional)
        x_label: Custom label for x-axis (optional)
        y_label: Custom label for y-axis (optional)
        title: Chart title (optional)
        orientation: 'v' for vertical, 'h' for horizontal bars
        extra_options: Additional layout options (optional)
        energy_unit: Unit of energy measurement (default is "kWh")
        show_grid: Whether to display grid lines (default is True)

    Returns:
        Plotly figure
    """
    # Convert to pandas if it's a polars dataframe
    if isinstance(data, pl.DataFrame):
        if metric and group_by_fields:
            df = (
                data.group_by(group_by_fields)
                .agg(**{f"{metric}_avg": pl.col(metric).mean()})
                .sort(group_by_fields)
                .to_pandas()
            )
            if y_field is None:
                y_field = f"{metric}_avg"
        else:
            df = data.to_pandas()
    else:
        df = data

    if metric and not y_display:
        y_display = metric.replace("energy_", "").capitalize()

    x_axis_label = (
        x_label
        if x_label
        else (x_field.capitalize() if isinstance(x_field, str) else "Value")
    )
    y_axis_label = (
        y_label
        if y_label
        else (
            y_display
            if y_display
            else (y_field.capitalize() if isinstance(y_field, str) else "Value")
        )
    )

    plot_kwargs = {
        "x": x_field if orientation == "v" else y_field,
        "y": y_field if orientation == "v" else x_field,
        "labels": {
            x_field: x_axis_label,
            y_field: y_axis_label,
        },
        "title": title + f" ({energy_unit})"
        if title
        else f"{y_axis_label} by {x_axis_label} ({energy_unit})",
        "orientation": orientation,
    }

    if color_field and color_field in df.columns:
        plot_kwargs["color"] = color_field
        plot_kwargs["labels"][color_field] = color_field.capitalize()

    fig = px.bar(df, **plot_kwargs)

    if show_grid:
        # For vertical charts (v), show horizontal grid lines
        # For horizontal charts (h), show vertical grid lines
        grid_config = {
            "xaxis": {
                "showgrid": orientation == "h",
                "gridwidth": 1,
                "gridcolor": "lightgray",
            },
            "yaxis": {
                "showgrid": orientation == "v",
                "gridwidth": 1,
                "gridcolor": "lightgray",
            },
        }
        
        if orientation == "h":
            grid_config["xaxis"].update({
                "tickformat": ".1f",
                "ticksuffix": " " + energy_unit,
            })
        
        fig.update_layout(**grid_config)

    if extra_options:
        fig.update_layout(**extra_options)

    return fig
