import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objects as po
import pyomo.environ as pyo
from pyomo.contrib import appsi
from typing import List, Dict, Union, Literal, Tuple, Optional
import psutil


class GasStorage:
    def __init__(self, name: str, date_start: dt.date, date_end: dt.date) -> None:
        self.name: str = name
        self.id: str = name + "_" + dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        self.date_start: dt.date = date_start
        self.date_end: dt.date = date_end
        self.optimization_time_limit: Optional[int] = None
        self.attr: pd.DataFrame = pd.DataFrame()
        self.model_vals: Optional[pd.DataFrame] = None
        self.__initialize_df()
        self.z0: int = 0
        self.empty_storage: bool = False
        self.empty_on_dates: List[dt.date] = []
        self.bsd_state_to_date: Dict[int, float] = {}
        self.injection_season: List[int] = []
        self.injection_idx: List[int] = []
        self.withdrawal_idx: List[int] = []
        self.inj_curve_daily: Dict[Tuple[dt.date, int, str], float] = {}
        self.wit_curve_daily: Dict[Tuple[dt.date, int, str], float] = {}
        self.mdl: Optional[pyo.ConcreteModel] = None
        self.slvr: Optional[Union[appsi.solvers.Cplex, appsi.solvers.Highs]] = None
        self.objective: Optional[float] = None
        self.results: Optional[
            Union[appsi.solvers.highs.HighsResults, appsi.solvers.cplex.CplexResults]
        ] = None
        self.solved: bool = False

        self.best_objective_bound: Optional[float] = None
        self.best_feasible: Optional[float] = None
        self.gap: Optional[float] = None

    @property
    def dates(self) -> List[dt.date]:
        """
        Maximum range of dates used in optimization.
        """
        return [
            self.date_start + dt.timedelta(days=i)
            for i in range(0, (self.date_end - self.date_start).days + 1)
        ]

    @property
    def curve_value_type(self) -> List[str]:
        """
        Row values of injection/withdrawal curves.
        """
        return ["lower", "upper", "portion"]

    @property
    def attr_lbls(self) -> List[str]:
        """
        List of available attribute labels.
        """
        return [
            "prices",
            "wgv",
            "ir",
            "wr",
            "inj_curve",
            "wit_curve",
            "m_const",
            "bsd_wit_curve",
        ]

    @property
    def delta(self) -> dt.timedelta:
        return dt.timedelta(days=1)

    def __initialize_df(self) -> None:
        """
        Initialize dataframe with all dates as index and create separate
        columns for year, month and yyyy-mm-dd date format.
        """
        self.attr = pd.DataFrame(index=pd.DatetimeIndex(self.dates))
        self.attr["yyyy-mm-dd"] = self.attr.index.date
        self.attr["year"] = self.attr.index.year
        self.attr["month"] = self.attr.index.month

    def __transform_curve(self, curve_type: str) -> None:
        """
        When loading attributes, injection and withdrawal curves
        (array attributes) need to be transformed in different way
        than other single-value attributes. These array attributes
        are stored in ..._daily dictionary and ready to use in model.
        """
        curve_dict = self.get_dict_from_column(curve_type)
        if isinstance(curve_dict, dict):
            for i in self.dates:
                if curve_type == "inj_curve":
                    for j in range(len(self.injection_idx)):
                        for curve_value_type in range(3):
                            self.inj_curve_daily[
                                i,
                                self.injection_idx[j],
                                self.curve_value_type[curve_value_type],
                            ] = curve_dict[i][curve_value_type, j]
                elif curve_type == "wit_curve":
                    for k in range(len(self.withdrawal_idx)):
                        for curve_value_type in range(3):
                            self.wit_curve_daily[
                                i,
                                self.withdrawal_idx[k],
                                self.curve_value_type[curve_value_type],
                            ] = curve_dict[i][curve_value_type, k]
                else:
                    raise Exception(
                        "Only following two types of curves are allowed: "
                        "inj_curve, wit_curve."
                    )
        else:
            raise Exception(
                "Injection and withdrawal curves must be of type np.ndarray "
                "with shape (3,)"
            )

    def load_prices(self, imported_prices: pd.DataFrame) -> None:
        """
        Based on dataframe with imported prices, column of daily prices is
        added to attributes dataframe self.attr.
        """
        self.prices_monthly = imported_prices
        self.prices_monthly = self.prices_monthly[
            self.prices_monthly["date"]
            >= pd.to_datetime(self.date_start.replace(day=1))
        ]
        self.prices_monthly["year"] = self.prices_monthly["date"].dt.year
        self.prices_monthly["month"] = self.prices_monthly["date"].dt.month
        self.attr["prices"] = (
            pd.merge(self.attr, self.prices_monthly, on=["year", "month"])
            .sort_values(["year", "month"])["price"]
            .values
        )

    def set_initial_state(self, z0: int) -> None:
        self.z0 = z0

    def set_state_to_date(self, bsd_state_to_date: Dict[int, float]) -> None:
        self.bsd_state_to_date = bsd_state_to_date

    def set_injection_season(self, injection_season: List[int]) -> None:
        self.injection_season = injection_season

    def set_dates_to_empty_storage(self, empty_on_dates: List[dt.date]) -> None:
        """
        Setting desired dates to empty the storage from input.
        """
        self.empty_on_dates = empty_on_dates
        self.empty_storage = True if self.empty_on_dates else False

    def set_optimization_time_limit(self, time_limit: int) -> None:
        if time_limit is not None:
            self.optimization_time_limit = int(time_limit)
        else:
            self.optimization_time_limit = 3600

    def load_attribute(
        self,
        attr_name: str,
        value: Union[np.ndarray, int],
        date_from: dt.date,
        date_to: dt.date,
    ) -> None:
        """
        Loading all types of attributes (parameters of storage) into
        attributes dataframe self.attr. Single-valued attributes are
        stored straight to dataframe, array-valued (inj/wit curves)
        are transformed for the use of model.
        """
        self.check_attribute_lbl(attr_name)
        if attr_name not in self.attr:
            self.attr[attr_name] = None
        selected_rows = (self.attr.index >= pd.to_datetime(date_from)) & (
            self.attr.index <= pd.to_datetime(date_to)
        )
        self.attr.loc[selected_rows, attr_name] = pd.Series(
            [value] * selected_rows.sum(), index=self.attr.index[selected_rows]
        )

        if attr_name == "wgv":
            self.load_attribute("m_const", value + 100000, date_from, date_to)
        elif attr_name == "inj_curve":
            self.injection_idx = list(
                np.arange(1, self.attr["inj_curve"].iloc[0].shape[1] + 1, 1)
            )
            self.__transform_curve(attr_name)
        elif attr_name == "wit_curve":
            self.withdrawal_idx = list(
                np.arange(1, self.attr["wit_curve"].iloc[0].shape[1] + 1, 1)
            )
            self.__transform_curve(attr_name)

    def reset_attribute(self, attr_name: str) -> None:
        """
        Reset given attribute in attributes dataframe to column with None
        values.
        """
        self.check_attribute_lbl(attr_name)
        self.attr[attr_name] = None

    def remove_attribute(self, attr_name: str) -> None:
        """
        Remove given attribute from attributes dataframe.
        """
        if attr_name in self.attr:
            self.attr = self.attr.drop(columns=[attr_name])

    def check_attribute_lbl(self, attr_name: str) -> None:
        """
        Check, whether attribute name is valid.
        """
        if attr_name not in self.attr_lbls:
            raise Exception(
                f"Only following attributes are allowed: "
                f"{', '.join(item for item in self.attr_lbls)}."
            )

    def get_dict_from_column(self, col_name: str) -> dict:
        """
        Return a column from attributes dataframe as dictionary
        for the use of model.
        """
        self.check_attribute_lbl(col_name)
        return pd.Series(self.attr[col_name], index=self.attr["yyyy-mm-dd"]).to_dict()

    def __mdl_initialize_sets(self) -> None:
        """
        Initialize sets of model.
        """
        assert self.mdl is not None
        self.mdl.i = pyo.Set(initialize=self.dates)
        self.mdl.j = pyo.Set(initialize=self.injection_idx)
        self.mdl.k = pyo.Set(initialize=self.withdrawal_idx)
        self.mdl.curve_value_type = pyo.Set(initialize=self.curve_value_type)
        self.mdl.bsd_months = pyo.Set(initialize=list(self.bsd_state_to_date.keys()))

    def __mdl_initialize_params(self) -> None:
        """
        Initialize parameters of model.
        """
        assert self.mdl is not None
        self.mdl.p = pyo.Param(
            self.mdl.i, initialize=self.get_dict_from_column("prices")
        )
        self.mdl.wgv = pyo.Param(
            self.mdl.i, initialize=self.get_dict_from_column("wgv")
        )
        self.mdl.ir = pyo.Param(self.mdl.i, initialize=self.get_dict_from_column("ir"))
        self.mdl.wr = pyo.Param(self.mdl.i, initialize=self.get_dict_from_column("wr"))
        self.mdl.m_const = pyo.Param(
            self.mdl.i, initialize=self.get_dict_from_column("m_const")
        )
        self.mdl.tab_inj = pyo.Param(
            self.mdl.i,
            self.mdl.j,
            self.mdl.curve_value_type,
            initialize=self.inj_curve_daily,
        )
        self.mdl.tab_wit = pyo.Param(
            self.mdl.i,
            self.mdl.k,
            self.mdl.curve_value_type,
            initialize=self.wit_curve_daily,
        )
        self.mdl.bsd_state_to_date = pyo.Param(
            self.mdl.bsd_months, initialize=self.bsd_state_to_date
        )

    def __mdl_initialize_vars(self) -> None:
        """
        Initialize variables of model.
        """
        assert self.mdl is not None
        self.mdl.x = pyo.Var(
            self.mdl.i, domain=pyo.NonNegativeIntegers, initialize=0, name="x"
        )
        self.mdl.y = pyo.Var(
            self.mdl.i, domain=pyo.NonNegativeIntegers, initialize=0, name="y"
        )
        self.mdl.z = pyo.Var(
            self.mdl.i, domain=pyo.NonNegativeIntegers, initialize=0, name="z"
        )

        self.mdl.t_inj = pyo.Var(
            self.mdl.i, self.mdl.j, domain=pyo.Binary, initialize=0, name="t_inj"
        )
        self.mdl.l_inj = pyo.Var(
            self.mdl.i, self.mdl.j, domain=pyo.Binary, initialize=0, name="l_inj"
        )
        self.mdl.u_inj = pyo.Var(
            self.mdl.i, self.mdl.j, domain=pyo.Binary, initialize=0, name="u_inj"
        )

        self.mdl.t_wit = pyo.Var(
            self.mdl.i, self.mdl.k, domain=pyo.Binary, initialize=0, name="t_wit"
        )
        self.mdl.l_wit = pyo.Var(
            self.mdl.i, self.mdl.k, domain=pyo.Binary, initialize=0, name="l_wit"
        )
        self.mdl.u_wit = pyo.Var(
            self.mdl.i, self.mdl.k, domain=pyo.Binary, initialize=0, name="u_wit"
        )

    def __mdl_def_constraints(self) -> None:
        """
        Define constraints of model.
        """
        assert self.mdl is not None
        self.mdl.constr_balance = pyo.Constraint(
            expr=sum(self.mdl.y[i] for i in self.mdl.i)
            <= self.z0 + sum(self.mdl.x[i] for i in self.mdl.i)
        )

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
                self.mdl.constr_gs.add(
                    self.mdl.z[i] == self.z0 + self.mdl.x[i] - self.mdl.y[i]
                )
                continue
            self.mdl.constr_gs.add(
                self.mdl.z[i]
                == self.mdl.z[i - self.delta] + self.mdl.x[i] - self.mdl.y[i]
            )

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
                    self.mdl.constr_state_to_date.add(
                        self.mdl.z[i] >= self.mdl.bsd_state_to_date[p] * self.mdl.wgv[i]
                    )

        self.mdl.constr_inj_low = pyo.ConstraintList()
        for i in self.mdl.i:
            for j in self.mdl.j:
                self.mdl.constr_inj_low.add(
                    self.mdl.tab_inj[(i, j, "lower")] * self.mdl.wgv[i]
                    <= self.mdl.z[i] + self.mdl.m_const[i] * (1 - self.mdl.l_inj[i, j])
                )
                self.mdl.constr_inj_low.add(
                    self.mdl.tab_inj[(i, j, "lower")] * self.mdl.wgv[i]
                    >= self.mdl.z[i] - self.mdl.m_const[i] * self.mdl.l_inj[i, j]
                )
        self.mdl.constr_inj_upp = pyo.ConstraintList()
        for i in self.mdl.i:
            for j in self.mdl.j:
                self.mdl.constr_inj_upp.add(
                    self.mdl.tab_inj[(i, j, "upper")] * self.mdl.wgv[i]
                    >= self.mdl.z[i] - self.mdl.m_const[i] * (1 - self.mdl.u_inj[i, j])
                )
                self.mdl.constr_inj_upp.add(
                    self.mdl.tab_inj[(i, j, "upper")] * self.mdl.wgv[i]
                    <= self.mdl.z[i] + self.mdl.m_const[i] * self.mdl.u_inj[i, j]
                )

        self.mdl.constr_inj_t = pyo.ConstraintList()
        for i in self.mdl.i:
            self.mdl.constr_inj_t.add(
                sum(self.mdl.t_inj[i, j] for j in self.mdl.j) == 1
            )
            for j in self.mdl.j:
                self.mdl.constr_inj_t.add(
                    self.mdl.u_inj[i, j]
                    + self.mdl.l_inj[i, j]
                    - 2 * self.mdl.t_inj[i, j]
                    >= 0
                )
                self.mdl.constr_inj_t.add(
                    self.mdl.u_inj[i, j]
                    + self.mdl.l_inj[i, j]
                    - 2 * self.mdl.t_inj[i, j]
                    <= 1
                )
        self.mdl.constr_inj = pyo.ConstraintList()
        for i in self.mdl.i:
            self.mdl.constr_inj.add(
                self.mdl.x[i]
                <= self.mdl.ir[i]
                * sum(
                    self.mdl.tab_inj[(i, j, "portion")] * self.mdl.t_inj[i, j]
                    for j in self.mdl.j
                )
            )

        self.mdl.constr_wit_low = pyo.ConstraintList()
        for i in self.mdl.i:
            for k in self.mdl.k:
                self.mdl.constr_wit_low.add(
                    self.mdl.tab_wit[(i, k, "lower")] * self.mdl.wgv[i]
                    <= self.mdl.z[i] + self.mdl.m_const[i] * (1 - self.mdl.l_wit[i, k])
                )
                self.mdl.constr_wit_low.add(
                    self.mdl.tab_wit[(i, k, "lower")] * self.mdl.wgv[i]
                    >= self.mdl.z[i] - self.mdl.m_const[i] * self.mdl.l_wit[i, k]
                )
        self.mdl.constr_wit_upp = pyo.ConstraintList()
        for i in self.mdl.i:
            for k in self.mdl.k:
                self.mdl.constr_wit_upp.add(
                    self.mdl.tab_wit[(i, k, "upper")] * self.mdl.wgv[i]
                    >= self.mdl.z[i] - self.mdl.m_const[i] * (1 - self.mdl.u_wit[i, k])
                )
                self.mdl.constr_wit_upp.add(
                    self.mdl.tab_wit[(i, k, "upper")] * self.mdl.wgv[i]
                    <= self.mdl.z[i] + self.mdl.m_const[i] * self.mdl.u_wit[i, k]
                )

        self.mdl.constr_wit_t = pyo.ConstraintList()
        for i in self.mdl.i:
            self.mdl.constr_wit_t.add(
                sum(self.mdl.t_wit[i, k] for k in self.mdl.k) == 1
            )
            for k in self.mdl.k:
                self.mdl.constr_wit_t.add(
                    self.mdl.u_wit[i, k]
                    + self.mdl.l_wit[i, k]
                    - 2 * self.mdl.t_wit[i, k]
                    >= 0
                )
                self.mdl.constr_wit_t.add(
                    self.mdl.u_wit[i, k]
                    + self.mdl.l_wit[i, k]
                    - 2 * self.mdl.t_wit[i, k]
                    <= 1
                )
        self.mdl.constr_wit = pyo.ConstraintList()
        for i in self.mdl.i:
            self.mdl.constr_wit.add(
                self.mdl.y[i]
                <= self.mdl.wr[i]
                * sum(
                    self.mdl.tab_wit[(i, k, "portion")] * self.mdl.t_wit[i, k]
                    for k in self.mdl.k
                )
            )

    def create_model(self) -> None:
        """
        Create pyomo model.
        """
        self.mdl = pyo.ConcreteModel(name="OptimusGas")
        self.__mdl_initialize_sets()
        self.__mdl_initialize_params()
        self.__mdl_initialize_vars()
        self.mdl.objective = pyo.Objective(
            expr=(
                sum(self.mdl.y[i] * self.mdl.p[i] for i in self.mdl.i)
                - sum(self.mdl.x[i] * self.mdl.p[i] for i in self.mdl.i)
            ),
            sense=pyo.maximize,
        )
        self.__mdl_def_constraints()

    def solve_model(
        self,
        solver_name: Literal["cplex", "highs", "scip"],
        gap: Optional[float] = None,
        stream_solver: bool = True,
        presolve_highs: Literal["off", "choose", "on"] = "choose",
        presolve_scip: Optional[int] = None,
    ) -> None:
        """
        Solve pyomo model.
        """
        if self.mdl is None:
            self.create_model()
        self.solver_name = solver_name
        if self.solver_name == "scip":
            self.slvr = pyo.SolverFactory("scip")
            self.slvr.options["lp/threads"] = psutil.cpu_count(logical=True)
            self.slvr.options["limits/time"] = self.optimization_time_limit
            if gap is not None:
                self.slvr.options["limits/gap"] = gap
            if presolve_scip is not None:
                self.slvr.options["presolving/maxrounds"] = presolve_scip

            self.results = self.slvr.solve(self.mdl, tee=stream_solver)
            self.termination_condition = self.results.solver.termination_condition
            if (self.results.solver.status == pyo.SolverStatus.ok) and (
                (self.termination_condition == pyo.TerminationCondition.optimal)
                or (
                    (
                        self.termination_condition
                        == pyo.TerminationCondition.maxTimeLimit
                    )
                    and (self.results.solver.primal_bound is not None)
                )
            ):
                self.__extract_values_from_model()
                if self.mdl is not None:
                    self.objective = self.mdl.objective()
                else:
                    self.objective = None
                self.best_feasible_objective = self.results.solver.primal_bound
                self.best_objective_bound = self.results.solver.dual_bound
                if self.best_feasible_objective > 0:
                    self.gap = (
                        self.best_objective_bound - self.best_feasible_objective
                    ) / self.best_feasible_objective
                self.solved = True
                print("\nTermination condition: ", self.termination_condition)
                print("Solver status: ", self.results.solver.status)
                print("Solver message: ", self.results.solver.message)
                print("Best feasible objective: ", self.best_feasible_objective)
                print("Best objective bound: ", self.best_objective_bound)
                print("Gap: ", self.gap)
                print(f"Objective: {self.objective}\n")
            # else:
            #     raise Exception(
            #         f"Couldn't find any feasible solution.\nTermination "
            #         f"condition: {self.termination_condition}"
            #     )
        else:
            if self.solver_name == "cplex":
                self.slvr = appsi.solvers.Cplex()
                self.slvr.cplex_options = {"threads": psutil.cpu_count(logical=True)}
            elif self.solver_name == "highs":
                self.slvr = appsi.solvers.Highs()
                self.slvr.highs_options = {
                    "threads": psutil.cpu_count(logical=True),
                    "presolve": presolve_highs,
                }
            else:
                raise Exception(
                    "Only two following solvers are available: cplex, highs."
                )

            self.slvr.config.time_limit = self.optimization_time_limit
            if gap is not None:
                self.slvr.config.mip_gap = gap
            self.slvr.config.stream_solver = stream_solver
            self.slvr.config.load_solution = False

            self.results = self.slvr.solve(self.mdl)
            self.termination_condition = self.results.termination_condition
            if (self.results.solver.status == pyo.SolverStatus.ok) and (
                (self.termination_condition == pyo.TerminationCondition.optimal)
                or (
                    (
                        self.termination_condition
                        == pyo.TerminationCondition.maxTimeLimit
                    )
                    and (self.results.solver.primal_bound is not None)
                )
            ):
                # This line has to be in standalone if-statement, otherwise the code
                # after this line becomes unaccessible.
                self.results.solution_loader.load_vars()
            if (self.results.solver.status == pyo.SolverStatus.ok) and (
                (self.termination_condition == pyo.TerminationCondition.optimal)
                or (
                    (
                        self.termination_condition
                        == pyo.TerminationCondition.maxTimeLimit
                    )
                    and (self.results.solver.primal_bound is not None)
                )
            ):
                self.__extract_values_from_model()
                if self.mdl is not None:
                    self.objective = self.mdl.objective()
                else:
                    self.objective = None
                self.best_feasible_objective = self.results.best_feasible_objective
                self.best_objective_bound = self.results.best_objective_bound
                self.gap = (
                    self.best_objective_bound - self.best_feasible_objective
                ) / self.best_feasible_objective
                self.solved = True
                print("\nTermination condition: ", self.termination_condition)
                print("Best feasible objective: ", self.best_feasible_objective)
                print("Best objective bound: ", self.best_objective_bound)
                print("Gap: ", self.gap)
                print(f"Objective: {self.objective}\n")
            # else:
            #     raise Exception(
            #         f"Couldn't find any feasible solution.\nTermination "
            #         f"condition: {self.termination_condition}"
            #     )

    def __extract_values_from_model(self) -> None:
        """
        Extract values from variables of a solved model to instance variables.
        Also create monthly and daily export dataframe of these values.
        """
        assert self.mdl is not None
        self.mdl.compute_statistics()
        self.statistics = self.mdl.statistics

        self.res_injection = self.mdl.x.extract_values()
        self.res_withdrawal = self.mdl.y.extract_values()
        self.res_gs_state = self.mdl.z.extract_values()
        self.res_operations = {
            key: self.res_injection[key] - self.res_withdrawal[key]
            for key in self.dates
        }

        ir = self.mdl.ir.extract_values()
        wr = self.mdl.wr.extract_values()
        t_inj = self.mdl.t_inj.extract_values()
        t_wit = self.mdl.t_wit.extract_values()
        self.max_operations = {}
        for i in self.dates:
            if i.month in self.injection_season:
                self.max_operations[i] = ir[i] * sum(
                    self.inj_curve_daily[(i, j, "portion")] * t_inj[i, j]
                    for j in list(self.mdl.j)
                )
            else:
                self.max_operations[i] = -wr[i] * sum(
                    self.wit_curve_daily[(i, k, "portion")] * t_wit[i, k]
                    for k in list(self.mdl.k)
                )

        self.daily_export = pd.DataFrame(
            list(
                zip(
                    list(self.attr["year"]),
                    list(self.attr["month"]),
                    list(self.res_operations.values()),
                    list(self.res_gs_state.values()),
                    list(self.max_operations.values()),
                    list(self.attr["wgv"]),
                )
            ),
            index=self.dates,
            columns=["Rok", "M", "W/I", "Stav", "Max C", "WGV"],
        )
        self.daily_export["Stav %"] = (
            self.daily_export["Stav"] / self.daily_export["WGV"]
        )

        daily_export_agg = self.daily_export.groupby(["Rok", "M"]).agg(
            w_i=("W/I", "sum"),
            year=("Rok", "min"),
            month=("M", "min"),
            wgv=("WGV", "min"),
        )
        gs_state = []
        for j, val in enumerate(daily_export_agg.w_i.values):
            if j == 0:
                gs_state.append(self.z0 + val)
                continue
            gs_state.append(gs_state[j - 1] + val)
        self.monthly_export = pd.DataFrame(
            list(
                zip(
                    daily_export_agg.year.values,
                    daily_export_agg.month.values,
                    daily_export_agg.w_i.values,
                    gs_state,
                    daily_export_agg.wgv,
                )
            ),
            columns=["Rok", "M", "W/I", "Stav", "WGV"],
        )
        self.monthly_export["Stav %"] = (
            self.monthly_export["Stav"] / self.monthly_export["WGV"]
        )

    def create_graph(self) -> None:
        """
        Create graph of storage optimization.
        """
        self.fig = po.Figure()
        self.fig.add_trace(
            po.Scatter(
                x=self.dates,
                y=list(self.max_operations.values()),
                name="Max. operations",
                line_color="#ffa600",
                mode="lines",
            )
        )
        self.fig.add_trace(
            po.Scatter(
                x=self.dates,
                y=list(self.res_operations.values()),
                name="Operations",
                fill="tozeroy",
                line_color="#74d576",
                mode="lines",
            )
        )
        self.fig.add_trace(
            po.Scatter(
                x=self.dates,
                y=list(self.res_gs_state.values()),
                name="GS state",
                fill="tozeroy",
                line_color="#34dbeb",
                yaxis="y2",
            )
        )
        self.fig.update_layout(
            title=(
                f"{self.name} gas storage optimization<br><sup>Solver: "
                f"{self.solver_name}</sup>"
            ),
            xaxis_title="Date",
            yaxis=dict(title="Operations [MWh/day]"),
            yaxis2=dict(
                title="GS state [MWh]",
                side="right",
                overlaying="y",
                titlefont=dict(color="#34dbeb"),
                tickfont=dict(color="#34dbeb"),
            ),
            legend=dict(orientation="v", x=1.06, xanchor="left", y=1),
        )
        self.fig.update_xaxes(fixedrange=False)
        self.fig.update_yaxes(zeroline=True, zerolinewidth=3, zerolinecolor="grey")
