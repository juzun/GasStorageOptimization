from gs_optim import GasStorage
import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path

def initialize_storage():
    date_start = dt.date(2024, 4, 1)
    date_end = dt.date(2025, 3, 31)
    storage = GasStorage('zásobníček', date_start, date_end)

    storage.load_prices('prices.xlsx')
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


def graph(storage, show_fig: bool = True, path: Path = None):
    storage.create_graph(show_fig=False)
    if path is not None:
        path = path.parent / (path.name + '.html')
        storage.fig.write_html(path)

def export_to_pdf(storage, path: Path = None):
    if path is None:
        path = f'{storage.id}_export.xlsx'
    else:
        path = path.parent / (path.name + '.xlsx')
    with pd.ExcelWriter(path, mode='w', engine='xlsxwriter') as writer:
        storage.daily_export[['Rok','M','W/I','Stav','Stav %','Max C']].to_excel(writer, sheet_name='data_daily', index=True, index_label='Datum')
        storage.monthly_export[['Rok','M','W/I','Stav','Stav %']].to_excel(writer, sheet_name='data_monthly', index=False)
        percent_format = writer.book.add_format({"num_format": "0%"})
        writer.sheets['data_daily'].set_column(5, 5, None, percent_format)
        writer.sheets['data_daily'].set_column(0, 0, 10)
        writer.sheets['data_monthly'].set_column(4, 4, None, percent_format)
        print(f'Results exported to {storage.id}_export.xlsx')

def total_export_to_xlsx(path: Path = Path('total_export')):
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

def total_graph(show_fig: bool = True, path: Path = None):
    GasStorage.create_total_graph(show_fig=True)
    if path is not None:
        path = path.parent / (path.name + '.html')
        GasStorage._fig.write_html(path)