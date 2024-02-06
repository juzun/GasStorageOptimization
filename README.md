# Introduction
You are looking at Streamlit application. This application allows user to optimize operations over a gas storage (injection and withdrawal).

User can choose from predefined storage templates. Some storage parameters are then changeable during the application run, mainly dates to start and end the computation and the initial state of storage.

Correctly defined storage can then be optimized. After optimization (which duration might vary depending on size of the storage and its constraints) applicaiton produces downloadable export in form of graph and Excel table.


# Getting started
To run the application type command `poetry run streamlit run src/Main.py`. Note that this application is dockerized.

To use the application, user must login with his ADEEQP account, which has to be enlisted in specific Azure user's group (for more information contact <jakub.zapletal@eon.com>).


# Developers
To change or add gas storage templates you need to push changes to file `storages.json` which stores all of the templates. New pipeline release is then needed.


# Mathematical model

## Sets
$i \ldots \text{dates}$

$j \ldots \text{injection index}$

$k \ldots \text{withdrawal index}$

$curve\_value\_type \ldots \text{withdrawal index}$

$bsd\_months \ldots \text{withdrawal index}$

$empty\_on\_dates \ldots \text{dates when gas storage is to be emptied}$

$injection\_season \ldots \text{dates for injection}$

## Parameters
$p_i \ldots \text{prices}$

$wgv_i \ldots \text{working gas volume}$

$ir_i \ldots \text{injection rate}$

$wr_i \ldots \text{withdrawal rate}$

$M_i \ldots \text{linearization constant}$

$tab\_inj_{i, j, curve\_value\_type} \ldots \text{injection table}$

$tab\_wit_{i, j, curve\_value\_type} \ldots \text{withdrawal table}$

$bsd\_state\_to\_date_{bsd\_months} \ldots \text{BSD conditions for state of gas storage in given months}$

$z_0 \ldots \text{initial state}$


## Variables
### Decision variables
$x_i \ldots \text{injection operations}$

$y_i \ldots \text{withdrawal operations}$

### Additional variables
$z_i \ldots \text{gas storage state}$

### Supporting binary variables (variables used for linearizaiton)
$t\_inj_{i,j} \ldots \text{supporting binary variable for injection operaitons}$

$l\_inj_{i,j} \ldots \text{supporting lower bound binary variable for injection operaitons}$

$u\_inj_{i,j} \ldots \text{supporting upper bound binary variable for injection operaitons}$

$t\_wit_{i,j} \ldots \text{supporting binary variable for withdrawal operaitons}$

$l\_wit_{i,j} \ldots \text{supporting lower bound binary variable for withdrawal operaitons}$

$u\_wit_{i,j} \ldots \text{supporting upper bound binary variable for withdrawal operaitons}$


## Objective function
$$\max{\sum_i{y_i\cdot p_i - x_i\cdot p_i}}$$


## Constraints
- <b>Balance constraint</b>

$$\sum_i{y_i}\leq z_0 + \sum_i{x_i}$$

- <b> Continuity constraint </b>

$$z_i = z_0 + x_i - y_i,\quad i=1$$
$$z_i = z_{i-1} + x_i - y_i,\quad i\geq 2$$

- <b>Capacity constraint</b>

$$z_i \leq wgv_i$$

- <b>Empty storage on given date</b>

$$z_{empty\_on\_dates} = 0$$

- <b>Injection season constraint</b>

$$y_i = 0, i\in injection\_season $$
$$x_i = 0, i\notin injection\_season $$

- <b>State to date</b>

$$z_i \geq bsd\_ state\_to\_date_{bsd\_months}\cdot wgv_i$$


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