import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objects as po
import pyomo.environ as pyo
from pyomo.contrib import appsi
from typing import List, Dict, Union, Literal
import psutil


class GasStorage():
    _instances: list = []
    _dates: List[dt.date] = []
    _total_data: pd.DataFrame = None
    _total_operations: Dict[dt.date, float] = {}
    _total_max_operations: Dict[dt.date, float] = {}
    _total_gs_state: Dict[dt.date, float] = {}
    _total_wgv: Dict[dt.date, float] = {}

    def __init__(self, name: str, date_start: dt.date, date_end: dt.date) -> None:
        GasStorage._instances.append(self)
        
        self.name: str = name
        self.id: str = name + '_' + dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
        self.date_start: dt.date = date_start
        self.date_end: dt.date = date_end
        self.attr: pd.DataFrame = None
        self.model_vals: pd.DataFrame = None
        self.__initialize_df()
        
        self.z0: int = 0
        self.empty_storage: bool = False
        self.empty_on_dates: List[dt.date] = []
        self.bsd_state_to_date: Dict[int, float] = {}
        self.injection_season: List[int] = []
        self.injection_idx: List[int] = []
        self.withdrawal_idx: List[int] = []
        self.inj_curve_daily: Dict[dt.date, float] = {}
        self.wit_curve_daily: Dict[dt.date, float] = {}
        self.mdl: pyo.ConcreteModel = None
        self.slvr: Union[appsi.solvers.Cplex, appsi.solvers.Highs] = None
        self.objective: float = None
        self.results: Union[appsi.solvers.highs.HighsResults, appsi.solvers.cplex.CplexResults] = None
        self.solved: bool = False

        self.best_objective_bound: float = None
        self.best_feasible: float = None
        self.gap: float = None
    
    @property
    def dates(self) -> List[dt.date]:
        return [self.date_start + dt.timedelta(days=i) for i in range(0,(self.date_end-self.date_start).days+1)]
    
    @property
    def curve_value_type(self) -> List[str]:
        return ['lower', 'upper', 'portion']

    @property
    def attr_lbls(self) -> List[str]:
        return ['prices', 'wgv', 'ir', 'wr', 'inj_curve', 'wit_curve', 'm_const', 'bsd_wit_curve']
    
    @property
    def delta(self) -> dt.timedelta:
        return dt.timedelta(days=1)

    def __initialize_df(self) -> None:
        self.attr = pd.DataFrame(index=pd.DatetimeIndex(self.dates))
        self.attr['yyyy-mm-dd'] = self.attr.index.date
        self.attr['year'] = self.attr.index.year
        self.attr['month'] = self.attr.index.month
    
    def __transform_curve(self, curve_type: str) -> None:
        curve_dict = self.get_dict_from_column(curve_type)
        for i in self.dates:
            if curve_type == 'inj_curve':
                for j in range(len(self.injection_idx)):
                    for l in range(3):
                        self.inj_curve_daily[i, self.injection_idx[j], self.curve_value_type[l]] = curve_dict[i][l,j]
            elif curve_type == 'wit_curve':
                for k in range(len(self.withdrawal_idx)):
                    for l in range(3):
                        self.wit_curve_daily[i, self.withdrawal_idx[k], self.curve_value_type[l]] = curve_dict[i][l,k]
            else:
                raise Exception(f"Only following two types of curves are allowed: inj_curve, wit_curve.")

    def load_prices(self, path: str) -> pd.DataFrame:        
        self.prices_monthly = pd.read_excel(path, parse_dates=['date'], usecols=['date', 'price'])
        self.prices_monthly = self.prices_monthly[self.prices_monthly['date'] >= pd.to_datetime(self.date_start.replace(day=1))]
        self.prices_monthly['year'] = self.prices_monthly['date'].dt.year
        self.prices_monthly['month'] = self.prices_monthly['date'].dt.month
        self.attr['prices'] = (pd.merge(self.attr, self.prices_monthly, on=['year', 'month'])
                               .sort_values(['year', 'month'])['price'].values)

    def set_initial_state(self, z0:int) -> None:
        self.z0=z0

    def set_dates_to_empty_storage(self, empty_on_dates: List[dt.date]) -> None:
        self.empty_on_dates = empty_on_dates
        self.empty_storage = True if self.empty_on_dates else False
    
    def load_attribute(self, attr_name: str, value: Union[np.array, int], date_from: dt.date, date_to: dt.date) -> None:
        self.check_attribute_lbl(attr_name)
        if attr_name not in self.attr:
            self.attr[attr_name] = None
        selected_rows = (self.attr.index >= pd.to_datetime(date_from)) & (self.attr.index <= pd.to_datetime(date_to))
        self.attr.loc[selected_rows, attr_name] = pd.Series([value]*selected_rows.sum(), index=self.attr.index[selected_rows])
        
        if attr_name == 'wgv':
            self.load_attribute('m_const', value + 100000, date_from, date_to)        
        if attr_name == 'inj_curve':
            self.injection_idx = np.arange(1,self.attr['inj_curve'].iloc[0].shape[1]+1,1)
            self.__transform_curve(attr_name)
        if attr_name == 'wit_curve':
            self.withdrawal_idx = np.arange(1,self.attr['wit_curve'].iloc[0].shape[1]+1,1)
            self.__transform_curve(attr_name)

    def reset_attribute(self, attr_name: str) -> None:
        self.check_attribute_lbl(attr_name)
        self.attr[attr_name] = None

    def check_attribute_lbl(self, attr_name: str) -> None:
        if attr_name not in self.attr_lbls:
            raise Exception(f"Only following attributes are allowed: {', '.join(item for item in self.attr_lbls)}.")
    
    def remove_attribute(self, attr_name: str) -> None:
        if attr_name in self.attr:
            self.attr = self.attr.drop(columns=[attr_name])
    
    def set_state_to_date(self, bsd_state_to_date: Dict[int, float]) -> None:
        self.bsd_state_to_date = bsd_state_to_date
    
    def set_injection_season(self, injection_season: List[int]) -> None:
        self.injection_season = injection_season

    def get_dict_from_column(self, col_name: str) -> Dict[dt.date, Union[np.array, int]]:
        self.check_attribute_lbl(col_name)
        return pd.Series(self.attr[col_name], index=self.attr['yyyy-mm-dd']).to_dict()    

    def __mdl_initialize_sets(self) -> None:
        self.mdl.i = pyo.Set(initialize=self.dates)
        self.mdl.j = pyo.Set(initialize=self.injection_idx)
        self.mdl.k = pyo.Set(initialize=self.withdrawal_idx)
        self.mdl.curve_value_type = pyo.Set(initialize=self.curve_value_type)
        self.mdl.bsd_months = pyo.Set(initialize=list(self.bsd_state_to_date.keys()))
    
    def __mdl_initialize_params(self) -> None:
        self.mdl.p = pyo.Param(self.mdl.i, initialize=self.get_dict_from_column('prices'))
        self.mdl.wgv = pyo.Param(self.mdl.i, initialize=self.get_dict_from_column('wgv'))
        self.mdl.ir = pyo.Param(self.mdl.i, initialize=self.get_dict_from_column('ir'))
        self.mdl.wr = pyo.Param(self.mdl.i, initialize=self.get_dict_from_column('wr'))
        self.mdl.m_const = pyo.Param(self.mdl.i, initialize=self.get_dict_from_column('m_const'))
        self.mdl.tab_inj = pyo.Param(self.mdl.i, self.mdl.j, self.mdl.curve_value_type, initialize=self.inj_curve_daily)
        self.mdl.tab_wit = pyo.Param(self.mdl.i, self.mdl.k, self.mdl.curve_value_type, initialize=self.wit_curve_daily)
        self.mdl.bsd_state_to_date = pyo.Param(self.mdl.bsd_months, initialize=self.bsd_state_to_date)

    def __mdl_initialize_vars(self) -> None:
        self.mdl.x = pyo.Var(self.mdl.i, domain=pyo.NonNegativeIntegers, initialize=0, name='x')
        self.mdl.y = pyo.Var(self.mdl.i, domain=pyo.NonNegativeIntegers, initialize=0, name='y')
        self.mdl.z = pyo.Var(self.mdl.i, domain=pyo.NonNegativeIntegers, initialize=0, name='z')

        self.mdl.t_inj = pyo.Var(self.mdl.i, self.mdl.j, domain=pyo.Binary, initialize=0, name='t_inj')
        self.mdl.l_inj = pyo.Var(self.mdl.i, self.mdl.j, domain=pyo.Binary, initialize=0, name='l_inj')
        self.mdl.u_inj = pyo.Var(self.mdl.i, self.mdl.j, domain=pyo.Binary, initialize=0, name='u_inj')

        self.mdl.t_wit = pyo.Var(self.mdl.i, self.mdl.k, domain=pyo.Binary, initialize=0, name='t_wit')
        self.mdl.l_wit = pyo.Var(self.mdl.i, self.mdl.k, domain=pyo.Binary, initialize=0, name='l_wit')
        self.mdl.u_wit = pyo.Var(self.mdl.i, self.mdl.k, domain=pyo.Binary, initialize=0, name='u_wit')

    def __mdl_def_constraints(self) -> None:
        self.mdl.constr_balance = pyo.Constraint(expr = sum(self.mdl.y[i] for i in self.mdl.i) <= self.z0 + sum(self.mdl.x[i] for i in self.mdl.i))

        self.mdl.constr_empty_storage = pyo.ConstraintList()
        if self.empty_storage:
            for date in self.empty_on_dates:
                if (date >= self.date_start) and (date <= self.date_end):
                    self.mdl.constr_empty_storage.add(self.mdl.z[date] == 0)

        self.mdl.constr_capacity = pyo.ConstraintList()
        for i in self.mdl.i:
            self.mdl.constr_capacity.add(self.mdl.z[i] <= self.mdl.wgv[i])

        self.mdl.constr_gs = pyo.ConstraintList()
        for i in self.mdl.i:
            if i == self.date_start:
                self.mdl.constr_gs.add(self.mdl.z[i] == self.z0 + self.mdl.x[i] - self.mdl.y[i])
                continue
            self.mdl.constr_gs.add(self.mdl.z[i] == self.mdl.z[i-self.delta] + self.mdl.x[i] - self.mdl.y[i])

        self.mdl.constr_season = pyo.ConstraintList()
        for i in self.mdl.i:
            if i.month in self.injection_season:
                self.mdl.constr_season.add(self.mdl.y[i] == 0)
            else:
                self.mdl.constr_season.add(self.mdl.x[i] == 0)

        self.mdl.constr_state_to_date = pyo.ConstraintList()
        for i in self.mdl.i:
            for p in self.mdl.bsd_months:
                if i.month == p and i.day == 1:
                    self.mdl.constr_state_to_date.add(self.mdl.z[i] >= self.mdl.bsd_state_to_date[p]*self.mdl.wgv[i])

        self.mdl.constr_inj_low = pyo.ConstraintList()
        for i in self.mdl.i:
            for j in self.mdl.j:
                self.mdl.constr_inj_low.add(self.mdl.tab_inj[(i,j,'lower')]*self.mdl.wgv[i] <= self.mdl.z[i] + self.mdl.m_const[i]*(1-self.mdl.l_inj[i,j]))
                self.mdl.constr_inj_low.add(self.mdl.tab_inj[(i,j,'lower')]*self.mdl.wgv[i] >= self.mdl.z[i] - self.mdl.m_const[i]*self.mdl.l_inj[i,j])
        self.mdl.constr_inj_upp = pyo.ConstraintList()
        for i in self.mdl.i:
            for j in self.mdl.j:
                self.mdl.constr_inj_upp.add(self.mdl.tab_inj[(i,j,'upper')]*self.mdl.wgv[i] >= self.mdl.z[i] - self.mdl.m_const[i]*(1-self.mdl.u_inj[i,j]))
                self.mdl.constr_inj_upp.add(self.mdl.tab_inj[(i,j,'upper')]*self.mdl.wgv[i] <= self.mdl.z[i] + self.mdl.m_const[i]*self.mdl.u_inj[i,j])

        self.mdl.constr_inj_t = pyo.ConstraintList()
        for i in self.mdl.i:
            self.mdl.constr_inj_t.add(sum(self.mdl.t_inj[i,j] for j in self.mdl.j) == 1)
            for j in self.mdl.j:
                self.mdl.constr_inj_t.add(self.mdl.u_inj[i,j] + self.mdl.l_inj[i,j] - 2*self.mdl.t_inj[i,j] >= 0)
                self.mdl.constr_inj_t.add(self.mdl.u_inj[i,j] + self.mdl.l_inj[i,j] - 2*self.mdl.t_inj[i,j] <= 1)
        self.mdl.constr_inj = pyo.ConstraintList()
        for i in self.mdl.i:
            self.mdl.constr_inj.add(self.mdl.x[i] <= self.mdl.ir[i]*sum(self.mdl.tab_inj[(i,j,'portion')]*self.mdl.t_inj[i,j] for j in self.mdl.j))

        self.mdl.constr_wit_low = pyo.ConstraintList()
        for i in self.mdl.i:
            for k in self.mdl.k:
                self.mdl.constr_wit_low.add(self.mdl.tab_wit[(i,k,'lower')]*self.mdl.wgv[i] <= self.mdl.z[i] + self.mdl.m_const[i]*(1-self.mdl.l_wit[i,k]))
                self.mdl.constr_wit_low.add(self.mdl.tab_wit[(i,k,'lower')]*self.mdl.wgv[i] >= self.mdl.z[i] - self.mdl.m_const[i]*self.mdl.l_wit[i,k])
        self.mdl.constr_wit_upp = pyo.ConstraintList()
        for i in self.mdl.i:
            for k in self.mdl.k:
                self.mdl.constr_wit_upp.add(self.mdl.tab_wit[(i,k,'upper')]*self.mdl.wgv[i] >= self.mdl.z[i] - self.mdl.m_const[i]*(1-self.mdl.u_wit[i,k]))
                self.mdl.constr_wit_upp.add(self.mdl.tab_wit[(i,k,'upper')]*self.mdl.wgv[i] <= self.mdl.z[i] + self.mdl.m_const[i]*self.mdl.u_wit[i,k])

        self.mdl.constr_wit_t = pyo.ConstraintList()
        for i in self.mdl.i:
            self.mdl.constr_wit_t.add(sum(self.mdl.t_wit[i,k] for k in self.mdl.k) == 1)
            for k in self.mdl.k:
                self.mdl.constr_wit_t.add(self.mdl.u_wit[i,k] + self.mdl.l_wit[i,k] - 2*self.mdl.t_wit[i,k] >= 0)
                self.mdl.constr_wit_t.add(self.mdl.u_wit[i,k] + self.mdl.l_wit[i,k] - 2*self.mdl.t_wit[i,k] <= 1)
        self.mdl.constr_wit = pyo.ConstraintList()
        for i in self.mdl.i:
            self.mdl.constr_wit.add(self.mdl.y[i] <= self.mdl.wr[i]*sum(self.mdl.tab_wit[(i,k,'portion')]*self.mdl.t_wit[i,k] for k in self.mdl.k))
    
    def create_model(self) -> None:
        self.mdl = pyo.ConcreteModel(name='OptimusGas')
        self.__mdl_initialize_sets()
        self.__mdl_initialize_params()
        self.__mdl_initialize_vars()
        self.mdl.objective = pyo.Objective(
            expr=(sum(self.mdl.y[i]*self.mdl.p[i] for i in self.mdl.i) - sum(self.mdl.x[i]*self.mdl.p[i] for i in self.mdl.i)), 
            sense=pyo.maximize
        )
        self.__mdl_def_constraints()

    def solve_model(
            self, solver_name: Literal['cplex','highs','scip'], time_limit: int = 3600, gap: float = None, 
            stream_solver: bool = True, presolve_highs: Literal['off','choose','on'] = 'choose', presolve_scip: int = None
        ) -> None:
        if not self.mdl:
            self.create_model()
        self.solver_name = solver_name
        if self.solver_name == 'scip':
            self.slvr = pyo.SolverFactory('scip')
            self.slvr.options['lp/threads'] = psutil.cpu_count(logical=True)
            self.slvr.options['limits/time'] = time_limit
            if gap is not None:
                self.slvr.options['limits/gap'] = gap
            if presolve_scip is not None:
                self.slvr.options['presolving/maxrounds'] = presolve_scip

            self.results = self.slvr.solve(self.mdl, tee=stream_solver)
            self.termination_condition = self.results.solver.termination_condition
            if (self.termination_condition == pyo.TerminationCondition.optimal) or (self.results.solver.primal_bound is not None):
                self.__extract_values_from_model()
                self.objective = self.mdl.objective()
                self.best_feasible_objective = self.results.solver.primal_bound
                self.best_objective_bound = self.results.solver.dual_bound
                self.gap = (self.best_objective_bound - self.best_feasible_objective) / self.best_feasible_objective
                self.solved=True
                print('\nTermination condition: ', self.termination_condition)
                print('Solver status: ', self.results.solver.status)
                print('Solver message: ', self.results.solver.message)
                print('Best feasible objective: ', self.best_feasible_objective)
                print('Best objective bound: ', self.best_objective_bound)
                print('Gap: ', self.gap)
                print(f'Objective: {self.objective}\n')
            else:
                raise Exception(f"Couldn't find any feasible solution.\nTermination condition: {self.termination_condition}")
        else:
            if self.solver_name == 'cplex':
                self.slvr = appsi.solvers.Cplex()
                self.slvr.cplex_options = {'threads': psutil.cpu_count(logical=True)}
            elif self.solver_name == 'highs':
                self.slvr = appsi.solvers.Highs()
                self.slvr.highs_options = {'threads': psutil.cpu_count(logical=True), 'presolve': presolve_highs}
            else:
                raise Exception(f"Only two following solvers are available: cplex, highs.")

            self.slvr.config.time_limit = time_limit
            if gap is not None:
                self.slvr.config.mip_gap = gap
            self.slvr.config.stream_solver = stream_solver
            self.slvr.config.load_solution = False

            self.results = self.slvr.solve(self.mdl)
            self.termination_condition = self.results.termination_condition            
            if (self.termination_condition == appsi.base.TerminationCondition.optimal) or (self.results.best_feasible_objective is not None):
                self.results.solution_loader.load_vars()
            if (self.termination_condition == appsi.base.TerminationCondition.optimal) or (self.results.best_feasible_objective is not None):
                self.__extract_values_from_model()
                self.objective = self.mdl.objective()
                self.best_feasible_objective = self.results.best_feasible_objective
                self.best_objective_bound = self.results.best_objective_bound
                self.gap = (self.best_objective_bound - self.best_feasible_objective) / self.best_feasible_objective
                self.solved = True
                print('\nTermination condition: ', self.termination_condition)
                print('Best feasible objective: ', self.best_feasible_objective)
                print('Best objective bound: ', self.best_objective_bound)
                print('Gap: ', self.gap)
                print(f'Objective: {self.objective}\n')
            else:
                raise Exception(f"Couldn't find any feasible solution.\nTermination condition: {self.termination_condition}")
        
    def __extract_values_from_model(self) -> None:
        self.mdl.compute_statistics()
        self.statistics = self.mdl.statistics

        self.res_injection = self.mdl.x.extract_values()
        self.res_withdrawal = self.mdl.y.extract_values()
        self.res_gs_state = self.mdl.z.extract_values()
        self.res_operations = {key: self.res_injection[key] - self.res_withdrawal[key] for key in self.dates}
        
        ir = self.mdl.ir.extract_values()
        wr = self.mdl.wr.extract_values()
        t_inj = self.mdl.t_inj.extract_values()
        t_wit = self.mdl.t_wit.extract_values()
        self.max_operations = {}
        for i in self.dates:
            if i.month in self.injection_season:
                self.max_operations[i] = ir[i]*sum(self.inj_curve_daily[(i,j,'portion')]*t_inj[i,j] for j in list(self.mdl.j))
            else:
                self.max_operations[i] = -wr[i]*sum(self.wit_curve_daily[(i,k,'portion')]*t_wit[i,k] for k in list(self.mdl.k))
        
        self.daily_export = pd.DataFrame(
            list(zip(
                list(self.attr['year']),list(self.attr['month']),list(self.res_operations.values()),
                list(self.res_gs_state.values()),list(self.max_operations.values()),list(self.attr['wgv']))),
            index=self.dates,
            columns=['Rok','M','W/I','Stav','Max C', 'WGV'])
        daily_export_agg = self.daily_export.groupby(['Rok','M']).agg(
            w_i=('W/I','sum'), year=('Rok', 'min'), month=('M', 'min'), wgv=('WGV', 'min')
        )
        self.daily_export['Stav %'] = self.daily_export['Stav']/self.daily_export['WGV']

        gs_state = []
        for i, val in enumerate(daily_export_agg.w_i.values):
            if i == 0:
                gs_state.append(self.z0 + val)
                continue
            gs_state.append(gs_state[i-1] + val)
        self.monthly_export = pd.DataFrame(    
            list(zip(
                daily_export_agg.year.values, daily_export_agg.month.values, daily_export_agg.w_i.values, gs_state, daily_export_agg.wgv
            )),
            columns=['Rok','M','W/I','Stav','WGV']
        )
        self.monthly_export['Stav %'] = self.monthly_export['Stav']/self.monthly_export['WGV']
    
    def create_graph(self, show_fig: bool = True) -> None:
        self.fig = po.Figure()
        self.fig.add_trace(po.Scatter(x=self.dates, y=list(self.max_operations.values()), name='Max. operations', line_color='#ffa600', mode='lines'))
        self.fig.add_trace(po.Scatter(x=self.dates, y=list(self.res_operations.values()), name='Operations', fill='tozeroy', line_color='#74d576', mode='lines'))
        self.fig.add_trace(po.Scatter(x=self.dates, y=list(self.res_gs_state.values()), name='GS state', fill='tozeroy', line_color='#34dbeb', yaxis = 'y2'))
        self.fig.update_layout(
            title = f'{self.name} gas storage optimization<br><sup>Solver: {self.solver_name}</sup>',
            xaxis_title = 'Date',
            yaxis = dict(
                title = 'Operations [MWh/day]'),
            yaxis2 = dict(
                title = "GS state [MWh]",
                side = 'right',
                overlaying = 'y',
                titlefont = dict(color='#34dbeb'),
                tickfont = dict(color='#34dbeb')),
            legend = dict(
                orientation = "v",
                x = 1.06,
                xanchor = 'left',
                y = 1)
        )
        self.fig.update_xaxes(fixedrange=False)
        self.fig.update_yaxes(zeroline=True, zerolinewidth=3, zerolinecolor='grey')
        if show_fig:
            self.fig.show()

    @classmethod
    def collect_all_storages(cls) -> None:
        if not cls._instances:
            raise Exception('No objects initialized yet.')
        for self in cls._instances:
            if not self.solved:
                raise Exception("One of the objects wasn't solved yet")
        date_min = min(cls._instances[0].dates)
        date_max = max(cls._instances[0].dates)
        for self in cls._instances:
            if min(self.dates) < date_min:
                date_min = min(self.dates)
            if max(self.dates) > date_max:
                date_max = max(self.dates)

        cls._dates = [date_min + dt.timedelta(days=i) for i in range(0,(date_max-date_min).days+1)]
        cls._total_operations = {key: 0 for key in cls._dates}
        cls._total_max_operations = {key: 0 for key in cls._dates}
        cls._total_gs_state = {key: 0 for key in cls._dates}
        cls._total_wgv = {key: 0 for key in cls._dates}
        for self in cls._instances:
            wgv_dict = self.get_dict_from_column('wgv')
            for d in cls._dates:
                if d in self.res_operations.keys():
                    cls._total_operations[d] += self.res_operations[d]
                if d in self.max_operations.keys():
                    cls._total_max_operations[d] += self.max_operations[d]
                if d in self.res_gs_state.keys():
                    cls._total_gs_state[d] += self.res_gs_state[d]
                if d in wgv_dict.keys():
                    cls._total_wgv[d] += wgv_dict[d]
        cls._total_data = pd.DataFrame(
            list(zip(
                list(cls._total_operations.values()),list(cls._total_gs_state.values()),
                list(cls._total_max_operations.values()),list(cls._total_wgv.values()))),
            index=pd.DatetimeIndex(cls._dates),
            columns=['W/I','Stav','Max C', 'WGV'])
        cls._total_data['yyyy-mm-dd'] = cls._total_data.index.date
        cls._total_data['Rok'] = cls._total_data.index.year
        cls._total_data['M'] = cls._total_data.index.month
        cls._total_data['Stav %'] = cls._total_data['Stav']/cls._total_data['WGV']

        cls._total_daily_export = pd.DataFrame(
            cls._total_data[['Rok','M','W/I','Stav', 'Stav %','Max C', 'WGV']],
            index=cls._dates)
        total_daily_export_agg = cls._total_daily_export.groupby(['Rok','M']).agg(
            w_i=('W/I','sum'), year=('Rok', 'min'), month=('M', 'min'), wgv=('WGV', 'min')
        )
        for self in cls._instances:
            cls._total_daily_export[f'{self.name} Stav'] = self.daily_export['Stav']
            cls._total_daily_export[f'{self.name} W/I'] = self.daily_export['W/I']
            cls._total_daily_export[f'{self.name} Max C'] = self.daily_export['Max C']

        gs_state_monthly = []
        z0 = 0
        for self in cls._instances:
            if min(self.dates) == min(cls._dates):
                z0 += self.z0
        for i, val in enumerate(total_daily_export_agg.w_i.values):
            if i == 0:
                gs_state_monthly.append(z0 + val)
                continue
            gs_state_monthly.append(gs_state_monthly[i-1] + val)

        cls._total_monthly_export = pd.DataFrame(    
            list(zip(
                total_daily_export_agg.year.values, total_daily_export_agg.month.values, 
                total_daily_export_agg.w_i.values, gs_state_monthly, total_daily_export_agg.wgv
            )),
            columns=['Rok','M','W/I','Stav','WGV']
        )
        cls._total_monthly_export['Stav %'] = cls._total_monthly_export['Stav']/cls._total_monthly_export['WGV']

    @classmethod
    def create_total_graph(cls, show_fig: bool = False):
        cls._fig = po.Figure()
        cls._fig.add_trace(po.Scatter(x=cls._dates, y=list(cls._total_max_operations.values()), name='Max. operations', line_color='#ffa600', mode='lines'))
        cls._fig.add_trace(po.Scatter(x=cls._dates, y=list(cls._total_operations.values()), name='Operations', fill='tozeroy', line_color='#74d576', mode='lines'))
        cls._fig.add_trace(po.Scatter(x=cls._dates, y=list(cls._total_gs_state.values()), name='GS state', fill='tozeroy', line_color='#34dbeb', yaxis = 'y2'))
        cls._fig.update_layout(
            title = f'Total gas storage optimization',
            xaxis_title = 'Date',
            yaxis = dict(
                title = 'Operations [MWh/day]'),
            yaxis2 = dict(
                title = "GS state [MWh]",
                side = 'right',
                overlaying = 'y',
                titlefont = dict(color='#34dbeb'),
                tickfont = dict(color='#34dbeb')),
            legend = dict(
                orientation = "v",
                x = 1.06,
                xanchor = 'left',
                y = 1)
        )
        cls._fig.update_xaxes(fixedrange=False)
        cls._fig.update_yaxes(zeroline=True, zerolinewidth=3, zerolinecolor='grey')
        if show_fig:
            cls._fig.show()