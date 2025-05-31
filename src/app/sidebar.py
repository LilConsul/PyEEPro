import streamlit as st
from data import storage


def get_acorn_categories():
    with st.spinner("Loading ACORN categories..."):
        acorn = sorted(storage.get_household_patterns().get_column("Acorn").unique())
        return acorn


def get_temperature_bins():
    with st.spinner("Loading temperature bins..."):
        temp_bins = sorted(storage.get_temperature_energy_patterns().get_column("temp_bin").unique())
        return temp_bins


def render_sidebar():
    with st.sidebar:
        tab1, tab2 = st.tabs(["Filters", "Cache Management"])
        with tab1:
            st.header("Filters")
            st.warning(
                "Due to streamlit limitations, change tabs of filters to the same tab of data manually. For example, if you are viewing Time-Based Trends , change filters to the Time-Based tab of filters."
            )
            subtab1, subtab2, subtab3 = st.tabs(["Time-Based", "Household", "Weather"])
            with subtab1:
                available_years = list(range(2011, 2015))
                selected_years = st.multiselect(
                    "Select Years", available_years, default=available_years
                )
            with subtab2:
                tariff_types = {
                    "Std": "Standard",
                    "ToU": "Economy-7",
                }
                selected_tariff = st.multiselect(
                    "Select Tariff Type",
                    options=list(tariff_types.keys()),
                    default=list(tariff_types.keys()),
                    format_func=lambda x: tariff_types[x],
                )
                acorn_categories = get_acorn_categories()
                selected_acorn = st.multiselect(
                    "Select ACORN Categories",
                    options=acorn_categories,
                    default=acorn_categories,
                )
            with subtab3:
                st.subheader("Weather Filters")

                # Month filter
                all_months = list(range(1, 13))
                month_names = {
                    1: "January", 2: "February", 3: "March", 4: "April",
                    5: "May", 6: "June", 7: "July", 8: "August",
                    9: "September", 10: "October", 11: "November", 12: "December"
                }

                selected_months = st.multiselect(
                    "Select Months",
                    options=all_months,
                    default=all_months,
                    format_func=lambda x: month_names[x]
                )

                # Temperature bin filter
                temp_bins = get_temperature_bins()
                selected_temp_bins = st.multiselect(
                    "Select Temperature Ranges",
                    options=temp_bins,
                    default=temp_bins
                )

                st.info("Temperature bins represent ranges of temperatures for easier analysis.")

            if st.session_state.get("filters") is None:
                st.session_state["filters"] = {}
            st.session_state.filters = {
                "years": selected_years,
                "tariff_type": selected_tariff,
                "acorn": selected_acorn,
                "months": selected_months,
                "temp_bins": selected_temp_bins,
            }

        with tab2:
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
