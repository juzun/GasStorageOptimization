from utils import (
    check_session_initialization,
    check_for_uploaded_prices,
    check_for_initialized_storages,
    solve_all_button,
    solve_button,
    authentication_check,
)
import streamlit as st

st.set_page_config(layout="wide")

authentication_check()
check_session_initialization()

st.title("Optimize")
st.write("---")

check_for_uploaded_prices()
check_for_initialized_storages()

st.button("Solve all", on_click=solve_all_button)

for tab, storage in zip(
    st.tabs(st.session_state.storages.keys()), st.session_state.storages.values()
):
    with tab:
        st.button(f"Solve {storage.name}", on_click=solve_button, args=(storage,))

        if storage.solved:
            st.text(
                f"Storage {storage.name} optimized.\n" f"Objective: {storage.objective}"
            )
        else:
            st.info("Not optimized yet.")
