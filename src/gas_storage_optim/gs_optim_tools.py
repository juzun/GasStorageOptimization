import pandas as pd
import datetime as dt
import plotly.graph_objects as po


def collect(solved_storages: list) -> tuple:
    date_min = min(solved_storages[0].dates)
    date_max = max(solved_storages[0].dates)
    for self in solved_storages:
        if min(self.dates) < date_min:
            date_min = min(self.dates)
        if max(self.dates) > date_max:
            date_max = max(self.dates)

    dates = [
        date_min + dt.timedelta(days=i)
        for i in range(0, (date_max - date_min).days + 1)
    ]
    total_operations = {key: 0 for key in dates}
    total_max_operations = {key: 0 for key in dates}
    total_gs_state = {key: 0 for key in dates}
    total_wgv = {key: 0 for key in dates}
    for self in solved_storages:
        wgv_dict = self.get_dict_from_column("wgv")
        for d in dates:
            if d in self.res_operations.keys():
                total_operations[d] += self.res_operations[d]
            if d in self.max_operations.keys():
                total_max_operations[d] += self.max_operations[d]
            if d in self.res_gs_state.keys():
                total_gs_state[d] += self.res_gs_state[d]
            if d in wgv_dict.keys():
                total_wgv[d] += wgv_dict[d]
    total_data = pd.DataFrame(
        list(
            zip(
                list(total_operations.values()),
                list(total_gs_state.values()),
                list(total_max_operations.values()),
                list(total_wgv.values()),
            )
        ),
        index=pd.DatetimeIndex(dates),
        columns=["W/I", "Stav", "Max C", "WGV"],
    )
    total_data["yyyy-mm-dd"] = total_data.index.date
    total_data["Rok"] = total_data.index.year
    total_data["M"] = total_data.index.month
    total_data["Stav %"] = total_data["Stav"] / total_data["WGV"]

    total_daily_export = pd.DataFrame(
        total_data[["Rok", "M", "W/I", "Stav", "Stav %", "Max C", "WGV"]], index=dates
    )
    total_daily_export_agg = total_daily_export.groupby(["Rok", "M"]).agg(
        w_i=("W/I", "sum"), year=("Rok", "min"), month=("M", "min"), wgv=("WGV", "min")
    )
    for self in solved_storages:
        total_daily_export[f"Stav {self.name}"] = self.daily_export["Stav"]
        total_daily_export[f"W/I {self.name}"] = self.daily_export["W/I"]
        total_daily_export[f"Max C {self.name}"] = self.daily_export["Max C"]

    gs_state_monthly = []
    z0 = 0
    for self in solved_storages:
        if min(self.dates) == min(dates):
            z0 += self.z0
    for i, val in enumerate(total_daily_export_agg.w_i.values):
        if i == 0:
            gs_state_monthly.append(z0 + val)
            continue
        gs_state_monthly.append(gs_state_monthly[i - 1] + val)

    total_monthly_export = pd.DataFrame(
        list(
            zip(
                total_daily_export_agg.year.values,
                total_daily_export_agg.month.values,
                total_daily_export_agg.w_i.values,
                gs_state_monthly,
                total_daily_export_agg.wgv,
            )
        ),
        columns=["Rok", "M", "W/I", "Stav", "WGV"],
    )
    total_monthly_export["Stav %"] = (
        total_monthly_export["Stav"] / total_monthly_export["WGV"]
    )

    fig = po.Figure()
    fig.add_trace(
        po.Scatter(
            x=dates,
            y=list(total_max_operations.values()),
            name="Max. operations",
            line_color="#ffa600",
            mode="lines",
        )
    )
    fig.add_trace(
        po.Scatter(
            x=dates,
            y=list(total_operations.values()),
            name="Operations",
            fill="tozeroy",
            line_color="#74d576",
            mode="lines",
        )
    )
    fig.add_trace(
        po.Scatter(
            x=dates,
            y=list(total_gs_state.values()),
            name="GS state",
            fill="tozeroy",
            line_color="#34dbeb",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Total gas storage optimization",
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
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(zeroline=True, zerolinewidth=3, zerolinecolor="grey")

    total_monthly_export = total_monthly_export[["Rok", "M", "W/I", "Stav", "Stav %"]]
    total_daily_export = total_daily_export[
        [
            "Rok",
            "M",
            "W/I",
            *[f"W/I {self.name}" for self in solved_storages],
            "Stav",
            "Stav %",
            *[f"Stav {self.name}" for self in solved_storages],
            "Max C",
            *[f"Max C {self.name}" for self in solved_storages],
        ]
    ]

    return (fig, total_daily_export, total_monthly_export)
