from gas_tank.gs_optim import GasStorage
import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import plotly.graph_objects as po
import io


def initialize_storage(imported_prices):
    date_start = dt.date(2024, 4, 1)
    date_end = dt.date(2025, 3, 31)
    storage = GasStorage('zásobníček', date_start, date_end)

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

def export_to_xlsx(storage) -> None:
    buffer = io.BytesIO()
    with pd.ExcelWriter('excel.xlsx', mode='w', engine='xlsxwriter') as writer:
        storage.daily_export.to_excel(writer, sheet_name='data_daily', index=True, index_label='Datum')
        storage.monthly_export.to_excel(writer, sheet_name='data_monthly', index=False)
        percent_format = writer.book.add_format({"num_format": "0%"})
        writer.sheets['data_daily'].set_column(5, 5, None, percent_format)
        writer.sheets['data_daily'].set_column(0, 0, 10)
        writer.sheets['data_monthly'].set_column(4, 4, None, percent_format)
    return buffer

def total_export_to_xlsx(path: Path = Path('total_export')) -> None:
    path = path.parent / (path.name + '.xlsx')
    GasStorage.collect_all_storages()
    with pd.ExcelWriter(path, mode='w', engine='xlsxwriter') as writer:
        GasStorage._total_daily_export[
            ['Rok','M','W/I',*[f'{self.name} W/I' for self in GasStorage._instances],'Stav %','Stav',*[f'{self.name} Stav' for self in GasStorage._instances],
                'Max C',*[f'{self.name} Max C' for self in GasStorage._instances]]
        ].to_excel(writer, sheet_name='data_daily', index=True, index_label='Datum')
        GasStorage._total_monthly_export[['Rok','M','W/I','Stav %','Stav']].to_excel(writer, sheet_name='data_monthly', index=False)
        writer.sheets['data_daily'].set_column(0, 0, 10)
        print(f'Results exported to {path}.xlsx')

def total_graph(show_fig: bool = True, path: Path = None) -> None:
    GasStorage.create_total_graph(show_fig)
    if path is not None:
        path = path.parent / (path.name + '.html')
        GasStorage._fig.write_html(path)

def import_prices(uploaded_file):
    return pd.read_excel(uploaded_file, parse_dates=['date'], usecols=['date', 'price'])

def reset_session_state():
    st.session_state.session_initialized = True
    st.session_state.prices = None
    st.session_state.storages = []
    st.session_state.solved = False
    st.session_state.uploaded_file = None

    st.info('Session initialized.')

def check_session_initialization():
    if 'session_initialized' not in st.session_state:
        st.session_state.session_initialized = False
        st.error('No session initialized.')
        st.info("Go to 'Main' page and initialize the session.")
        st.stop()
    elif 'session_initialized' in st.session_state:
        if st.session_state.session_initialized:
            pass
        else:
            st.error('No session initialized.')
            st.info("Go to 'Main' page and initialize the session.")
            st.stop()

def check_for_uploaded_prices():
    if st.session_state.prices is not None:
        pass
    else:
        st.error('No prices available.')
        st.info("Go to 'Prices' page and upload a file with prices.")
        st.stop()

def check_for_initialized_storages():
    if st.session_state.storages:
        pass
    else:
        st.error('No storage initialized.')
        st.info("Go to 'Storages' page and initialize a storage.")
        st.stop()

def check_for_solved_storages():
    if st.session_state.solved:
        pass
    else:
        st.error('No results to export.')
        st.info("Go to 'Optimize' page and run the optimization.")
        st.stop()

def solve_button(storage):
    for storage in st.session_state.storages:
        storage.solve_model(solver_name='cplex', stream_solver=True)
    st.session_state.solved = True
