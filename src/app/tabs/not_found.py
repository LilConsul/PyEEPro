import streamlit as st


def render_not_found_tab():
    st.header("🔍 Not Found")

    with st.container():
        st.write("This page is currently under development.")

        st.divider()

        st.info(
            "This appears to be a placeholder for a future tab. Check back later for updates!"
        )
