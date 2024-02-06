from utils import (
    check_session_initialization,
    check_for_uploaded_prices,
    initialize_storage,
    check_for_duplicate_storage_names,
    authentication_check,
)
import streamlit as st
import datetime as dt
from typing import Optional

st.set_page_config(layout="wide")

authentication_check()
check_session_initialization()

st.title("Initialize storages")
st.write("---")

check_for_uploaded_prices()

for tab, storage_name in zip(
    st.tabs(st.session_state.storages_json.keys()),
    st.session_state.storages_json.keys(),
):
    with tab:
        with st.form(f"storage_{storage_name}_form"):
            st.info(f"Specify {storage_name} storage parameters.")
            new_storage_name = st.text_input(
                "Enter unique storage name: ", value=storage_name
            )
            date_start = st.date_input(
                label="Enter start date:",
                key=f"date_start_{storage_name}",
                value=dt.datetime.strptime(
                    st.session_state.storages_json[storage_name]["TimePeriods"][0][
                        "StartDate"
                    ],
                    "%Y-%m-%d",
                ).date(),
            )
            date_end = st.date_input(
                label="Enter end date:",
                key=f"date_end_{storage_name}",
                value=dt.datetime.strptime(
                    st.session_state.storages_json[storage_name]["TimePeriods"][0][
                        "EndDate"
                    ],
                    "%Y-%m-%d",
                ).date(),
            )
            initial_state = st.number_input(
                "Enter initial storage state:", step=1, key=f"init_state_{storage_name}"
            )
            empty_on_end_date = st.checkbox("Empty storage on last date.")
            optimization_time_limit: Optional[int] = st.number_input(
                value=900,
                label="Enter optimization time limit in seconds. "
                "This entry is optional, minimum value is 900 seconds, "
                "otherwise default value 3600 seconds will be used.",
                step=1,  # type: ignore
            )
            # typing ignored, for return type of st.number_input is int (because of value=900 and step=1)
            submitted = st.form_submit_button("Submit")
            if submitted:
                if isinstance(date_start, dt.date) and isinstance(date_end, dt.date):
                    if (
                        optimization_time_limit is not None
                        and optimization_time_limit < 900
                    ):
                        optimization_time_limit = None
                    initialize_storage(
                        storage_json=st.session_state.storages_json[storage_name],
                        date_start=date_start,
                        date_end=date_end,
                        initial_state=int(initial_state),
                        empty_on_end_date=empty_on_end_date,
                        storage_name=check_for_duplicate_storage_names(
                            new_storage_name
                        ),
                        optimization_time_limit=optimization_time_limit,
                    )

keys_to_delete = []
for name, storage in st.session_state.storages.items():
    col1, col2, col3, col4, col5 = st.columns((1, 2, 3, 2, 1))
    col1.write(f"{name}")
    message = "Optimized" if storage.solved else "Not optimized"
    col2.write(message)
    col3.write(f"{storage.date_start} - {storage.date_end}")
    col4.write(f"initial state: {storage.z0} MWh")
    if col5.button("Delete", key=f"del{name}"):
        keys_to_delete.append(name)
for name in keys_to_delete:
    del st.session_state.storages[name]
