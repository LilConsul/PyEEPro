import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.header("Filters")
        current_filters = st.session_state.get("filters_turned_on", {})
        if is_yearly := current_filters.get("years", False):
            available_years = list(range(2011, 2015))
            selected_years = st.multiselect(
                "Select Years", available_years, default=available_years
            )

        if is_consumers := current_filters.get("consumers", False):
            available_consumers = ["High", "Medium", "Low"]
            selected_consumers = st.multiselect(
                "Select Consumer Types",
                available_consumers,
                default=available_consumers,
            )

        filters = {
            "years": selected_years if is_yearly else None,
            "consumer_type": selected_consumers if is_consumers else None,
        }

    return filters