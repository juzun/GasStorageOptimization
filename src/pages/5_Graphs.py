from utils import (
    check_session_initialization,
    check_for_uploaded_prices,
    check_for_initialized_storages,
    check_for_solved_storages,
    collect_storages,
    get_graph,
    authentication_check,
)
import streamlit as st

st.set_page_config(layout="wide")

authentication_check()
check_session_initialization()

st.title("Graphs")
st.write("---")

check_for_uploaded_prices()
check_for_initialized_storages()
check_for_solved_storages()


# Total graph
# ---------------------------------------------------
collect_storages()

st.plotly_chart(st.session_state.total_graph, use_container_width=True)

_, last_col = st.columns([3, 1])
with last_col:
    st.download_button(
        label="Download total graph",
        data=st.session_state.total_graph.to_html(),
        file_name="total_graph.html",
        mime="text/html",
    )


# Graphs
# ---------------------------------------------------
for tab, storage in zip(
    st.tabs(st.session_state.storages.keys()), st.session_state.storages.values()
):
    with tab:
        if storage.solved:
            st.plotly_chart(get_graph(storage), use_container_width=True)
            _, last_col = st.columns([4, 1])
            with last_col:
                st.download_button(
                    label="Download graph",
                    data=get_graph(storage).to_html(),
                    file_name=f"{storage.name}.html",
                    mime="text/html",
                )
        else:
            st.info("Not optimized yet.")
