import streamlit as st


def render_years() -> str:
    """
    Render the selected years from the session state.

    Returns:
        A string representation of selected years or "All" if none are selected.
    """
    return (
        str(st.session_state.get("filters", {}).get("years", "")).strip("[]") or "All"
    )
