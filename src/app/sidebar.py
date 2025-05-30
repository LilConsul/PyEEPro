import streamlit as st
from data import storage


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

        if st.session_state.get("filters") is None:
            st.session_state["filters"] = {}
        st.session_state.filters = {
            "years": selected_years if is_yearly else None,
            "consumer_type": selected_consumers if is_consumers else None,
        }

        st.markdown("---")
        st.header("Cache Management")
        cached_files = storage.get_cached_files()

        if cached_files:
            # st.info("Note: Deleting cache files will not delete cache from RAM.")
            st.text(f"{len(cached_files)} files in cache")

            # Display files with direct delete buttons
            for file in cached_files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(
                        f"<div style='padding-top: 8px;'>{file}</div>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button("🗑️", key=f"delete_{file}", help=f"Delete {file}"):
                        storage.remove_cache(file)
                        st.success(f"Removed {file}")
                        st.rerun()

            st.markdown("---")

            # Initialize session state for confirmation
            if "show_clear_all_form" not in st.session_state:
                st.session_state.show_clear_all_form = False

            # Button to trigger the confirmation form
            if not st.session_state.show_clear_all_form:
                if st.button("Delete All Cache Files", type="primary"):
                    st.session_state.show_clear_all_form = True
                    st.rerun()

            # Show confirmation form only when requested
            if st.session_state.show_clear_all_form:
                with st.form(key="clear_cache_form"):
                    st.write("Clear all cache files")
                    st.warning(
                        "This will remove all cached files. This action cannot be undone."
                    )

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        cancel_button = st.form_submit_button(
                            "Cancel", use_container_width=True
                        )
                    with col2:
                        confirm_button = st.form_submit_button(
                            "Clear All", type="primary", use_container_width=True
                        )

                    if confirm_button:
                        storage.remove_all_caches()
                        st.success("All caches cleared successfully!")
                        st.session_state.show_clear_all_form = False
                        st.rerun()
                    if cancel_button:
                        st.session_state.show_clear_all_form = False
                        st.rerun()
        else:
            st.info("No files in cache")