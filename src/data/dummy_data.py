import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data
def generate_demo_data():
    date_rng = pd.date_range(start="2022-01-01", end="2022-12-31", freq="H")

    df = pd.DataFrame(date_rng, columns=["timestamp"])

    season_component = 10 * np.sin(np.pi * df.index / (24 * 30.5 * 3))

    hour_of_day = df["timestamp"].dt.hour
    daily_component = 5 * np.sin(np.pi * (hour_of_day - 9) / 12)
    daily_component[hour_of_day < 6] = -5

    day_of_week = df["timestamp"].dt.dayofweek
    weekend_mask = day_of_week >= 5
    weekly_component = np.zeros(len(df))
    weekly_component[weekend_mask] = 3

    # Combine components
    df["consumption"] = (
        20
        + season_component
        + daily_component
        + weekly_component
        + np.random.normal(0, 3, len(df))
    )
    df["consumption"] = df["consumption"].clip(lower=0)

    df["household_type"] = np.random.choice(["High", "Medium", "Low"], size=len(df))
    df["tariff"] = np.random.choice(["Standard", "Economy-7"], size=len(df))

    df["temperature"] = (
        15
        + 10 * np.sin(np.pi * df.index / (24 * 30.5 * 6))
        + np.random.normal(0, 2, len(df))
    )

    return df


def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return generate_demo_data()
