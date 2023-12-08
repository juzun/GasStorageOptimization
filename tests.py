import unittest
import datetime as dt
from src.gas_storage_optim.gs_optim import GasStorage
import numpy as np
import pandas as pd

# python3 -m unittest tests

class TestCalculations(unittest.TestCase):

    def setUp(self):        
        date_start = dt.date(2022, 1, 1)
        date_end = dt.date(2022, 1, 31)
        self.storage = GasStorage('test_storage', date_start, date_end)
        prices = {date_start: 1}
        prices = pd.DataFrame(list(prices.items()), columns=['date','price'])
        prices = pd.DataFrame(zip(list(pd.to_datetime(prices['date'])), list(prices['price'])), columns=['date','price'])
        self.storage.load_prices(prices)
        self.storage.load_attribute('wgv', 100, date_start, date_end)
        self.storage.load_attribute('wr', 10, date_start, date_end)
        self.storage.load_attribute('ir', 10, date_start, date_end)
        inj_curve = np.array([[0,50], [50,100], [50,100]])/100
        self.storage.load_attribute('inj_curve', inj_curve, date_start, date_end)
        wit_curve = np.array([[0,50], [50,100], [50,100]])/100
        self.storage.load_attribute('wit_curve', wit_curve, date_start, date_end)
        self.storage.set_initial_state(100)
        self.storage.create_model()
    
    def test_model_exists(self):
        self.assertIsNotNone(self.storage.mdl, 'Model object should exist.')
    
    def test_attributes_table_exists(self):
        self.assertIsNotNone(self.storage.attr, 'Attributes table should exist.')
    
    def test_objective_exists(self):
        self.assertIn('objective', dir(self.storage.mdl), 'Objective should exist.')
    
    def test_objective_value(self):
        self.storage.solve_model(solver_name='scip', stream_solver=False)
        self.assertEqual(self.storage.objective, 100)

if __name__ == '__main__':
    unittest.main()