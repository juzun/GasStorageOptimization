from gas_storage_optim.gs_optim import GasStorage
from gas_storage_optim.gs_optim_tools import collect
# from authentication.msal_auth import streamlit_authenticate

# from config import ALLOWED_IDS
from typing import Tuple
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as po
import io
import json


def authentication_process():
    # streamlit_authenticate(ALLOWED_IDS)
    pass


def authentication_check():
    # if "login_token" in st.session_state and "authenticated" in st.session_state:
    #     if st.session_state.login_token and st.session_state.authenticated:
    #         st.write(f'User: {st.session_state.login_token["account"]["username"]}')
    #     else:
    #         st.info("Please go to 'Main' page and log in.")
    #         st.stop()
    # else:
    #     st.info("Please go to 'Main' page and log in.")
    #     st.stop()
    pass


def initialize_storage(
    storage_json: dict,
    date_start: dt.date,
    date_end: dt.date,
    initial_state: int,
    empty_on_end_date: bool,
    storage_name: str,
    optimization_time_limit: int,
) -> None:
    """
    Initialize storage using data from .json file and user input.
    """
    date_start, date_end = correct_date_range(storage_json, date_start, date_end)
    if (date_start < st.session_state.prices["date"][0].date()) or (
        date_end > st.session_state.prices["date"].iloc[-1].date()
    ):
        st.error("Invalid input storage dates.")
        st.info(
            "Input dates for storage are out of prices dates range. "
            "Please change start and end dates of your storage so they are inside "
            "the range of prices dates or add new file with prices."
        )
        st.stop()
    storage = GasStorage(storage_name, date_start, date_end)
    storage.load_prices(st.session_state.prices)
    for period in storage_json["TimePeriods"]:
        period_start_date = dt.datetime.strptime(period["StartDate"], "%Y-%m-%d").date()
        period_end_date = dt.datetime.strptime(period["EndDate"], "%Y-%m-%d").date()
        if period_end_date >= date_start:
            storage.load_attribute(
                "wgv", period["WGV"], period_start_date, period_end_date
            )
            storage.load_attribute(
                "wr", period["WithdrawalRate"], period_start_date, period_end_date
            )
            storage.load_attribute(
                "ir", period["InjectionRate"], period_start_date, period_end_date
            )
    storage.load_attribute(
        "inj_curve",
        np.array(storage_json["InjectionCurve"]) / 100,
        date_start,
        date_end,
    )
    storage.load_attribute(
        "wit_curve",
        np.array(storage_json["WithdrawalCurve"]) / 100,
        date_start,
        date_end,
    )
    storage.set_initial_state(initial_state)
    storage.set_injection_season(storage_json["InjectionSeason"])
    storage.set_state_to_date(
        {int(key): val for key, val in storage_json["StatesToDate"].items()}
    )
    dates_to_empty_storage = []
    for date in storage_json["DatesToEmptyStorage"]:
        dates_to_empty_storage.append(dt.datetime.strptime(date, "%Y-%m-%d").date())
    if empty_on_end_date:
        dates_to_empty_storage.append(date_end)
    storage.set_dates_to_empty_storage(dates_to_empty_storage)

    if optimization_time_limit is not None:
        storage.set_optimization_time_limit(optimization_time_limit)
    else:
        storage.set_optimization_time_limit(storage_json["DefaultTimeLimit"])

    storage.create_model()
    st.session_state.storages[storage.name] = storage

    st.success(
        f"{storage_json['GasStorageName']} storage named {storage_name} initialized."
    )


def correct_date_range(
    storage_json: dict, date_start: dt.date, date_end: dt.date
) -> Tuple[dt.date, dt.date]:
    """
    Check whether user input dates are valid and change it eventualy.
    """
    min_period_start_date, max_period_end_date = dt.date.max, dt.date.min
    for period in storage_json["TimePeriods"]:
        period_start_date = dt.datetime.strptime(period["StartDate"], "%Y-%m-%d").date()
        period_end_date = dt.datetime.strptime(period["EndDate"], "%Y-%m-%d").date()
        min_period_start_date = (
            period_start_date
            if period_start_date <= min_period_start_date
            else min_period_start_date
        )
        max_period_end_date = (
            period_end_date
            if period_end_date >= max_period_end_date
            else max_period_end_date
        )

    if date_start > date_end:
        st.error("Invalid input dates.")
        st.info("Start date must be before end date.")
        st.stop()
    elif date_end < min_period_start_date or date_start > max_period_end_date:
        st.error("Invalid input dates.")
        st.info("Entered dates are out of the storage dates range.")
        st.stop()
    elif (
        date_start < min_period_start_date
        and min_period_start_date <= date_end <= max_period_end_date
    ):
        return min_period_start_date, date_end
    elif (
        date_end > max_period_end_date
        and min_period_start_date <= date_start <= max_period_end_date
    ):
        return date_start, max_period_end_date
    elif date_end > max_period_end_date and date_start < min_period_start_date:
        return min_period_start_date, max_period_end_date
    else:
        return date_start, date_end


def get_graph(storage) -> po.Figure:
    """
    Return the graph object from storage object.
    """
    storage.create_graph()
    return storage.fig


def export_to_xlsx(storage) -> io.BytesIO:
    """
    Return the excel export of optimized model.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, mode="w", engine="xlsxwriter") as writer:
        storage.daily_export[["Rok", "M", "W/I", "Stav", "Stav %", "Max C"]].to_excel(
            writer, sheet_name="data_daily", index=True, index_label="Datum"
        )
        storage.monthly_export[["Rok", "M", "W/I", "Stav", "Stav %"]].to_excel(
            writer, sheet_name="data_monthly", index=False
        )
        percent_format = writer.book.add_format({"num_format": "0%"})
        writer.sheets["data_daily"].set_column(5, 5, None, percent_format)
        writer.sheets["data_daily"].set_column(
            0, 0, 10
        )  # set width of first column to 10
        writer.sheets["data_monthly"].set_column(4, 4, None, percent_format)
    return buffer


def total_export_to_xlsx() -> io.BytesIO:
    """
    Return the excel export of all optimized models.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, mode="w", engine="xlsxwriter") as writer:
        st.session_state.total_daily_export.to_excel(
            writer, sheet_name="data_daily", index=True, index_label="Datum"
        )
        st.session_state.total_monthly_export.to_excel(
            writer, sheet_name="data_monthly", index=False
        )
        writer.sheets["data_daily"].set_column(0, 0, 10)
    return buffer


def import_prices(uploaded_file) -> pd.DataFrame:
    """
    Import prices from excel file.
    """
    try:
        excel_file = pd.read_excel(
            uploaded_file, parse_dates=["date"], usecols=["date", "price"]
        )
    except Exception:
        st.error("Invalid file format.")
        st.info("Please upload Excel file with dates and prices.")
        st.stop()
    return excel_file


def reset_session_state() -> None:
    """
    Set or reset variables in the Streamlit session state.
    """
    st.session_state.session_initialized = True
    st.session_state.prices = None
    st.session_state.storages = {}
    st.session_state.storages_json = {}
    st.session_state.solved = False
    st.session_state.uploaded_file = None
    load_storages_from_json()

    st.success("Session initialized.")


def load_storages_from_json() -> None:
    """
    Load the storages from json file and save them to Streamlit session state.
    """
    try:
        with open("src/data/storages.json", "r") as file:
            # with open("app/src/data/storages.json", "r") as file:  # for Docker build
            storages_json = json.load(file)
        for i in storages_json:
            st.session_state.storages_json[i["GasStorageName"]] = i
    except Exception:
        st.error("Couldn't load storage.json file.")
        st.stop()

    check_for_duplicate_json_storage_names()


def check_for_duplicate_json_storage_names() -> None:
    """
    Check for duplicate storage names from json file.
    """
    if len(st.session_state.storages_json.keys()) == len(
        (st.session_state.storages_json.keys())
    ):
        pass
    else:
        st.error("Storage names must be unique.")
        st.info(
            "Go to definition of storages in 'src/data' "
            "and change storage names so they are unique."
        )
        st.stop()


def check_for_duplicate_storage_names(new_storage_name):
    if new_storage_name in st.session_state.storages.keys():
        new_storage_name = new_storage_name + "_another"
        new_storage_name = check_for_duplicate_storage_names(new_storage_name)
    return new_storage_name


def check_session_initialization() -> None:
    """
    Check whether session was initialized.
    """
    if "session_initialized" not in st.session_state:
        reset_session_state()
    elif "session_initialized" in st.session_state:
        if st.session_state.session_initialized:
            pass
        else:
            reset_session_state()


def check_for_uploaded_prices() -> None:
    """
    Check whether prices were uploaded.
    """
    if st.session_state.prices is not None:
        pass
    else:
        st.error("No prices available.")
        st.info("Go to 'Prices' page and upload a file with prices.")
        st.stop()


def check_for_initialized_storages() -> None:
    """
    Check whether there are some initialized storages.
    """
    if st.session_state.storages:
        pass
    else:
        st.error("No storage initialized.")
        st.info("Go to 'Storages' page and initialize a storage.")
        st.stop()


def check_for_solved_storages() -> None:
    """
    Check whether there is a solved storage.
    """
    for storage in st.session_state.storages.values():
        if storage.solved:
            break
    else:
        st.error("No results to export.")
        st.info("Go to 'Optimize' page and run the optimization.")
        st.stop()


def solve_button(storage) -> None:
    """
    Run optimization for specified storage.
    """
    storage.solve_model(solver_name="scip", stream_solver=True)
    if not storage.solved:
        st.error(
            f"Couldn't find any feasible solution for storage '{storage.name}'. "
            f"Termination condition: '{storage.termination_condition}'."
        )
        st.info("Please check the input data of storage.")


def solve_all_button() -> None:
    """
    Run optimization for all storages.
    """
    for storage in st.session_state.storages.values():
        storage.solve_model(solver_name="scip", stream_solver=True)
        if not storage.solved:
            st.error(
                f"Couldn't find any feasible solution for storage '{storage.name}'. "
                f"Termination condition: '{storage.termination_condition}'."
            )
            st.info("Please check the input data of storage.")


def collect_storages() -> None:
    """
    Collect export from all solved storages.
    """
    solved_storages = []
    for storage in st.session_state.storages.values():
        if storage.solved:
            solved_storages.append(storage)
    (
        st.session_state.total_graph,
        st.session_state.total_daily_export,
        st.session_state.total_monthly_export,
    ) = collect(solved_storages)
