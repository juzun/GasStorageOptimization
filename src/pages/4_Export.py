from utils import (
    check_session_initialization,
    check_for_uploaded_prices,
    check_for_initialized_storages,
    check_for_solved_storages,
    collect_storages,
    total_export_to_xlsx,
    export_to_xlsx,
    authentication_check,
)
import streamlit as st

st.set_page_config(layout="wide")

authentication_check()
check_session_initialization()

st.title("Export")
st.write("---")

check_for_uploaded_prices()
check_for_initialized_storages()
check_for_solved_storages()


# Total export
# ---------------------------------------------------
collect_storages()

col1, col2 = st.columns((3, 2))
col1.write("Total daily operations")
daily_format = {key: "{:.0f}" for key in st.session_state.total_daily_export.columns}
daily_format["Stav %"] = "{:.2%}"
col1.dataframe(
    st.session_state.total_daily_export.style.format(daily_format),
    use_container_width=True,
)

col2.write("Total monthly operations")
monthly_format = {
    key: "{:.0f}" for key in st.session_state.total_monthly_export.columns
}
monthly_format["Stav %"] = "{:.2%}"
col2.dataframe(
    st.session_state.total_monthly_export.style.format(monthly_format),
    use_container_width=True,
)

_, last_col = st.columns([3, 1])
with last_col:
    st.download_button(
        label="Download total export",
        data=total_export_to_xlsx(),
        file_name="total_export.xlsx",
        mime="application/vnd.ms-excel",
    )


# Export of storages
# ---------------------------------------------------
for tab, storage in zip(
    st.tabs(st.session_state.storages.keys()), st.session_state.storages.values()
):
    with tab:
        st.subheader(f"{storage.name} storage")

        if storage.solved:
            col1, col2 = st.columns((3, 2))
            col1.write("Daily operations")
            daily_format = {key: "{:.0f}" for key in storage.daily_export.columns}
            daily_format["Stav %"] = "{:.2%}"
            col1.dataframe(
                storage.daily_export[
                    ["Rok", "M", "W/I", "Stav", "Stav %", "Max C"]
                ].style.format(daily_format),
                use_container_width=True,
            )

            col2.write("Monthly operations")
            monthly_format = {key: "{:.0f}" for key in storage.monthly_export.columns}
            monthly_format["Stav %"] = "{:.2%}"
            col2.dataframe(
                storage.monthly_export[
                    ["Rok", "M", "W/I", "Stav", "Stav %"]
                ].style.format(monthly_format),
                use_container_width=True,
            )

            _, last_col = st.columns([4, 1])
            with last_col:
                st.download_button(
                    label="Download export",
                    data=export_to_xlsx(storage),
                    file_name=f"{storage.name}.xlsx",
                    mime="application/vnd.ms-excel",
                )
        else:
            st.info("Not optimized yet.")
