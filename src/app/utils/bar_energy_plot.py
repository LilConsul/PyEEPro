import plotly.express as px


def create_bar_chart(
    df,
    x_field,
    y_field,
    y_display=None,
    color_field=None,
    x_label=None,
    y_label=None,
    title=None,
    orientation="v",
    extra_options=None,
    energy_unit="kWh",
):
    """
    Create a bar chart using Plotly Express.

    Args:
        df: Pandas DataFrame with data
        x_field: Field for x-axis
        y_field: Field for y-axis (height of bars)
        y_display: Display name for y-axis (optional)
        color_field: Field for color differentiation (optional)
        x_label: Custom label for x-axis (optional)
        y_label: Custom label for y-axis (optional)
        title: Chart title (optional)
        orientation: 'v' for vertical, 'h' for horizontal bars
        extra_options: Additional layout options (optional)
        energy_unit: Unit of energy measurement (default is "kWh")

    Returns:
        Plotly figure
    """
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
        or f"{y_axis_label} by {x_axis_label} ({energy_unit})",
        "orientation": orientation,
    }
    if color_field and color_field in df.columns:
        plot_kwargs["color"] = color_field
        plot_kwargs["labels"][color_field] = color_field.capitalize()
    fig = px.bar(df, **plot_kwargs)
    if extra_options:
        fig.update_layout(**extra_options)
    return fig

