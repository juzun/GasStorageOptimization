from gas_tank.gs_optim import GasStorage
from gas_tank.gs_optim_tools import *
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as po
import io
import json


def initialize_storage():

    with open('zasobnik.json', 'r') as file:
        json_data = json.load(file)

    date_start = dt.date(2023, 12, 1)
    date_end = dt.date(2026, 3, 31)
    initial_state = 2078317

    storage = GasStorage(json_data['GasStorageName'], date_start, date_end)

    storage.load_prices(st.session_state.prices)
    
    for period in json_data['TimePeriods']:
        period_start_date = dt.datetime.strptime(period['StartDate'], '%Y-%m-%d').date()
        period_end_date = dt.datetime.strptime(period['EndDate'], '%Y-%m-%d').date()
        if period_end_date >= date_start:
            storage.load_attribute('wgv', period['WGV'], period_start_date, period_end_date)
            storage.load_attribute('wr', period['WithdrawalRate'], period_start_date, period_end_date)
            storage.load_attribute('ir', period['InjectionRate'], period_start_date, period_end_date)

    storage.load_attribute('inj_curve', np.array(json_data['InjectionCurve'])/100, date_start, date_end)
    storage.load_attribute('wit_curve', np.array(json_data['WithdrawalCurve'])/100, date_start, date_end)

    storage.set_initial_state(initial_state)
    storage.set_injection_season(json_data['InjectionSeason'])
    storage.set_state_to_date({int(key): val for key, val in json_data['StatesToDate'].items()})

    dates_to_empty_storage = []
    for date in json_data['DatesToEmptyStorage']:
        dates_to_empty_storage.append(dt.datetime.strptime(date, '%Y-%m-%d').date())
    dates_to_empty_storage.append(date_end)
    storage.set_dates_to_empty_storage(dates_to_empty_storage)

    storage.create_model()

    return storage

def initialize_storage1(imported_prices):
    date_start = dt.date(2024, 4, 1)
    date_end = dt.date(2025, 3, 31)
    storage = GasStorage('RWE', date_start, date_end)

    storage.load_prices(imported_prices)
    storage.load_attribute('wgv', 217079, date_start, date_end)
    storage.load_attribute('wr', 3635, date_start, date_end)
    storage.load_attribute('ir', 2181, date_start, date_end)
    inj_curve = np.array([[0,81,84,88,92,96], 
                        [81,84,88,92,96,100], 
                        [100,96.682,92.258,87.834,83.41,78.956]])/100
    storage.load_attribute('inj_curve', inj_curve, date_start, date_end)
    wit_curve = np.array([[0,4,8,12,16,20,22], 
                        [4,8,12,16,20,22,100], 
                        [49.854,58.971,68.089,77.206,84.044,95.441,100]])/100
    storage.load_attribute('wit_curve', wit_curve, date_start, date_end)

    storage.set_initial_state(0)
    storage.set_dates_to_empty_storage([date_end])
    storage.set_injection_season([4,5,6,7,8,9])
    storage.set_state_to_date({5: 0.05, 7: 0.3, 9: 0.6, 11: 0.9})

    storage.create_model()

    return storage

def initialize_storage2(imported_prices):
    date_start = dt.date(2024, 4, 1)
    date_end = dt.date(2025, 3, 31)
    storage = GasStorage('MGS', date_start, date_end)

    storage.load_prices(imported_prices)
    storage.load_attribute('wgv', 217079, date_start, date_end)
    storage.load_attribute('wr', 3635, date_start, date_end)
    storage.load_attribute('ir', 2181, date_start, date_end)
    inj_curve = np.array([[0,81,84,88,92,96], 
                        [81,84,88,92,96,100], 
                        [100,96.682,92.258,87.834,83.41,78.956]])/100
    storage.load_attribute('inj_curve', inj_curve, date_start, date_end)
    wit_curve = np.array([[0,4,8,12,16,20,22], 
                        [4,8,12,16,20,22,100], 
                        [49.854,58.971,68.089,77.206,84.044,95.441,100]])/100
    storage.load_attribute('wit_curve', wit_curve, date_start, date_end)

    storage.set_initial_state(0)
    storage.set_dates_to_empty_storage([date_end])
    storage.set_injection_season([4,5,6,7,8,9])
    storage.set_state_to_date({5: 0.05, 7: 0.3, 9: 0.6, 11: 0.9})

    storage.create_model()

    return storage


def get_graph(storage) -> po.Figure:
    storage.create_graph()
    return storage.fig

def export_to_xlsx(storage) -> io.BytesIO:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, mode='w', engine='xlsxwriter') as writer:
        storage.daily_export[['Rok','M','W/I','Stav','Stav %','Max C']].to_excel(writer, sheet_name='data_daily', index=True, index_label='Datum')
        storage.monthly_export[['Rok','M','W/I','Stav','Stav %']].to_excel(writer, sheet_name='data_monthly', index=False)
        percent_format = writer.book.add_format({"num_format": "0%"})
        writer.sheets['data_daily'].set_column(5, 5, None, percent_format)
        writer.sheets['data_daily'].set_column(0, 0, 10) # set width of first column to 10
        writer.sheets['data_monthly'].set_column(4, 4, None, percent_format)
    return buffer

def total_export_to_xlsx() -> io.BytesIO:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, mode='w', engine='xlsxwriter') as writer:
        st.session_state.total_daily_export.to_excel(writer, sheet_name='data_daily', index=True, index_label='Datum')
        st.session_state.total_monthly_export.to_excel(writer, sheet_name='data_monthly', index=False)
        writer.sheets['data_daily'].set_column(0, 0, 10)
    return buffer

def import_prices(uploaded_file) -> pd.DataFrame:
    return pd.read_excel(uploaded_file, parse_dates=['date'], usecols=['date', 'price'])

def update_storage_labels() -> None:
    st.session_state.storage_labels = [storage.id for storage in st.session_state.storages]

def reset_session_state() -> None:
    st.session_state.session_initialized = True
    st.session_state.prices = None
    st.session_state.storages = []
    st.session_state.storage_labels = []
    st.session_state.solved = False
    st.session_state.uploaded_file = None

    st.info('Session initialized.')

def check_session_initialization() -> None:
    if 'session_initialized' not in st.session_state:
        reset_session_state()
    elif 'session_initialized' in st.session_state:
        if st.session_state.session_initialized:
            pass
        else:
            reset_session_state()

def check_for_uploaded_prices() -> None:
    if st.session_state.prices is not None:
        pass
    else:
        st.error('No prices available.')
        st.info("Go to 'Prices' page and upload a file with prices.")
        st.stop()

def check_for_initialized_storages() -> None:
    if st.session_state.storages:
        pass
    else:
        st.error('No storage initialized.')
        st.info("Go to 'Storages' page and initialize a storage.")
        st.stop()

def check_for_solved_storages() -> None:
    for storage in st.session_state.storages:
        if storage.solved:
            break
    else:
        st.error('No results to export.')
        st.info("Go to 'Optimize' page and run the optimization.")
        st.stop()

def solve_button(storage) -> None:
    storage.solve_model(solver_name='scip', stream_solver=True)

def solve_all_button() -> None:
    for storage in st.session_state.storages:
        storage.solve_model(solver_name='scip', stream_solver=True)

def collect_storages():
    solved_storages = []
    for storage in st.session_state.storages:
        if storage.solved:
            solved_storages.append(storage)
    st.session_state.total_graph, st.session_state.total_daily_export, st.session_state.total_monthly_export = collect(solved_storages)