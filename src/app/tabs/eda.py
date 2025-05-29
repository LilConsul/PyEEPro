import streamlit as st
import plotly.express as px


def render_eda_tab(data):
    st.header("📊 Exploratory Data Analysis")

    with st.container():
        st.write(
            "Explore time-based patterns in energy consumption data from Smart Meters."
        )

    st.subheader("🔑 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Avg Daily Consumption", value=f"{data['consumption'].mean():.2f} kWh"
        )

    with col2:
        st.metric(
            label="Maximum Consumption",
            value=f"{data['consumption'].max():.2f} kWh",
            delta="High usage",
        )

    with col3:
        peak_hour = (
            data.groupby(data["timestamp"].dt.hour)["consumption"].mean().idxmax()
        )
        st.metric(label="Peak Hour", value=f"{peak_hour}:00")

    with col4:
        daily_avg = data.groupby(data["timestamp"].dt.date)["consumption"].mean()
        max_day = daily_avg.idxmax()
        st.metric(label="Highest Consumption Day", value=f"{max_day}")

    st.divider()

    st.subheader("⏰ Hourly Consumption Pattern")

    with st.container():
        hourly_data = (
            data.groupby(data["timestamp"].dt.hour)["consumption"].mean().reset_index()
        )
        fig_hourly = px.line(
            hourly_data,
            x="timestamp",
            y="consumption",
            title="Average Hourly Consumption Pattern",
            labels={"timestamp": "Hour of Day", "consumption": "Consumption (kWh)"},
            line_shape="spline",
        )
        fig_hourly.update_traces(line_color="#1E88E5", line_width=3)
        fig_hourly.update_layout(
            plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="lightgrey"),
            yaxis=dict(showgrid=True, gridcolor="lightgrey"),
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

    with st.expander("📊 Detailed Statistics"):
        st.dataframe(data.describe())
