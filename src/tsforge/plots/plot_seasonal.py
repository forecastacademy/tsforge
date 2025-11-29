import pandas as pd
import numpy as np
from typing import Union, List, Optional, Literal
from tsforge.plots.style import _apply_tsforge_style, PALETTE, HIGHLIGHT


def plot_seasonal(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    ids: Union[None, int, str, List[str]] = None,
    group_col: Union[str, List[str], None] = None,
    agg: str = "sum",                 # aggregation if group_col is used
    seasonal_agg: str = "mean",       # aggregation within seasonal cycle
    freq: str = "M",                  # "M","Q","W","D"
    mode: str = "overlay",            # overlay | facet | dropdown
    kind: Literal["line", "box"] = "line",
    engine: str = "plotly",           # plotly | matplotlib
):
    """
    Seasonal subseries plot: collapse time into a cycle (e.g., months in a year)
    and show either lines (yearly trajectories) or boxplots (distribution across years).

    Parameters
    ----------
    ids : None | int | str | list
        - None: sample up to 6 series.
        - int: sample that many series.
        - str: single series.
        - list[str]: explicit list of series.
    group_col : str or list, optional
        Aggregate series before plotting (e.g., roll SKUs up to category).
    agg : str, default 'sum'
        Aggregation if grouping by group_col.
    seasonal_agg : str, default 'mean'
        Aggregation for collapsing multiple observations within (id, year, cycle).
    freq : {"M","Q","W","D"}
        Seasonal cycle for x-axis.
    mode : {"overlay","facet","dropdown"}
        Overlay = all selected series on one chart.
        Facet = subplot per series.
        Dropdown = dropdown to select a series.
    kind : {"line","box"}, default "line"
        Line = subseries lines per year.
        Box = boxplots by seasonal position, pooled across years.
    engine : {"plotly","matplotlib"}
        Plotting backend.
    """

    # ---- prep
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # ---- group_col aggregation
    if group_col:
        if isinstance(group_col, str):
            group_keys = [group_col, date_col]
            df = df.groupby(group_keys, observed=True)[value_col].agg(agg).reset_index()
            id_col = group_col
        else:
            group_keys = list(group_col) + [date_col]
            df = df.groupby(group_keys, observed=True)[value_col].agg(agg).reset_index()
            df["_group_id"] = df[list(group_col)].astype(str).agg("|".join, axis=1)
            id_col = "_group_id"

    # ---- normalize ids
    unique_ids = df[id_col].dropna().unique().tolist()
    if ids is None:
        n_series = min(6, len(unique_ids))
        ids = pd.Series(unique_ids).sample(n_series, random_state=42).tolist()
    elif isinstance(ids, int):
        n_series = min(ids, len(unique_ids))
        ids = pd.Series(unique_ids).sample(n_series, random_state=42).tolist()
    elif isinstance(ids, str):
        ids = [ids]; n_series = 1
    else:
        ids = list(ids); n_series = len(ids)

    df_sub = df[df[id_col].isin(ids)].copy()

    # ---- seasonal cycle extraction
    df_sub["year"] = df_sub[date_col].dt.year
    f = freq.upper()
    if f.startswith("M"):
        df_sub["seasonal_x"] = df_sub[date_col].dt.month
        x_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    elif f.startswith("Q"):
        df_sub["seasonal_x"] = df_sub[date_col].dt.quarter
        x_labels = ["Q1","Q2","Q3","Q4"]
    elif f.startswith("W"):
        df_sub["seasonal_x"] = df_sub[date_col].dt.isocalendar().week.astype(int)
        x_labels = None
    elif f.startswith("D"):
        df_sub["seasonal_x"] = df_sub[date_col].dt.dayofyear
        x_labels = None
    else:
        raise ValueError(f"Unsupported freq '{freq}'. Use 'M','Q','W','D'.")

    # ---- collapse to one point per (id, year, seasonal_x)
    df_sub = (
        df_sub
        .groupby([id_col, "year", "seasonal_x"], observed=True, as_index=False)[value_col]
        .agg(seasonal_agg)
    )

    # =========================
    # KIND: BOX
    # =========================
    if kind == "box":
        if engine == "plotly":
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            box_trace_style = dict(pointpos=0, jitter=0.2,
                                   marker=dict(opacity=0.6, size=5))

            if mode == "overlay":
                fig = px.box(df_sub, x="seasonal_x", y=value_col,
                             points="all", color_discrete_sequence=[HIGHLIGHT])
                fig.update_traces(**box_trace_style)
                fig.update_layout(
                    title=dict(text=f"Seasonal Box Plot (freq={freq})", y=0.98),
                    legend=dict(yanchor="top", y=0.92),
                    xaxis_title=f"Cycle position ({freq})",
                    yaxis_title=value_col,
                )
                if x_labels:
                    fig.update_xaxes(tickmode="array",
                                     tickvals=list(range(1, len(x_labels)+1)),
                                     ticktext=x_labels)
                return _apply_tsforge_style(fig, engine="plotly", context="seasonal")

            elif mode == "facet":
                fig = make_subplots(rows=n_series, cols=1,
                                    subplot_titles=[str(uid) for uid in ids])
                for r, uid in enumerate(ids, 1):
                    sub = df_sub[df_sub[id_col] == uid]
                    box = px.box(sub, x="seasonal_x", y=value_col,
                                 points="all", color_discrete_sequence=[HIGHLIGHT])
                    box.update_traces(**box_trace_style)
                    for trace in box.data:
                        fig.add_trace(trace, row=r, col=1)
                fig.update_layout(height=300*n_series,
                                  title=f"Seasonal Box Facet (freq={freq})")
                return _apply_tsforge_style(fig, engine="plotly", context="seasonal")

            elif mode == "dropdown":
                fig = go.Figure()
                traces_per_series = None
                for i, uid in enumerate(ids):
                    sub = df_sub[df_sub[id_col] == uid]
                    box = px.box(sub, x="seasonal_x", y=value_col,
                                 points="all", color_discrete_sequence=[HIGHLIGHT])
                    box.update_traces(**box_trace_style)
                    if traces_per_series is None:
                        traces_per_series = len(box.data)
                    for trace in box.data:
                        trace.visible = (i == 0)
                        trace.name = f"{uid} box"
                        fig.add_trace(trace)
                # dropdown buttons
                buttons = []
                for i, uid in enumerate(ids):
                    visible = [False] * (traces_per_series * len(ids))
                    for j in range(traces_per_series):
                        visible[i*traces_per_series + j] = True
                    buttons.append(dict(label=str(uid),
                                        method="update",
                                        args=[{"visible": visible}]))
                fig.update_layout(
                    updatemenus=[{
                        "buttons": buttons,
                        "direction": "down",
                        "x": 1.05, "y": 1.15,
                        "xanchor": "left", "yanchor": "top"
                    }],
                    title=dict(text=f"Seasonal Box Dropdown (freq={freq})", y=0.98),
                    legend=dict(yanchor="top", y=0.92),
                )
                if x_labels:
                    fig.update_xaxes(tickmode="array",
                                     tickvals=list(range(1, len(x_labels)+1)),
                                     ticktext=x_labels)
                return _apply_tsforge_style(fig, engine="plotly", context="seasonal")

        elif engine == "matplotlib":
            import matplotlib.pyplot as plt
            if mode != "overlay":
                raise ValueError("Matplotlib seasonal box only supports overlay mode.")
            fig, ax = plt.subplots(figsize=(10, 6))
            grouped = [df_sub[df_sub["seasonal_x"] == s][value_col].dropna()
                       for s in sorted(df_sub["seasonal_x"].unique())]
            ax.boxplot(grouped,
                       tick_labels=x_labels or sorted(df_sub["seasonal_x"].unique()))
            ax.set_title(f"Seasonal Box Plot (freq={freq})", fontsize=14, pad=20)
            ax.set_xlabel(f"Cycle position ({freq})")
            ax.set_ylabel(value_col)
            plt.tight_layout()
            return _apply_tsforge_style(fig, engine="matplotlib", context="seasonal")

        # =========================
    # KIND: LINE
    # =========================
    if kind == "line":
        if engine == "plotly":
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            if mode == "overlay":
                fig = go.Figure()
                for i, (uid, sub) in enumerate(df_sub.groupby(id_col, observed=True)):
                    for j, (year, g) in enumerate(sub.groupby("year", observed=True)):
                        g = g.sort_values("seasonal_x")
                        fig.add_trace(go.Scatter(
                            x=g["seasonal_x"], y=g[value_col],
                            mode="lines+markers",
                            connectgaps=False,
                            name=f"{uid} - {year}" if n_series > 1 else str(year),
                            line=dict(color=PALETTE[j % len(PALETTE)], width=2),
                            opacity=0.9
                        ))
                fig.update_layout(
                    title=dict(text=f"Seasonal Subseries Plot (freq={freq})", y=0.98),
                    legend=dict(yanchor="top", y=0.92),
                    xaxis_title=f"Cycle position ({freq})",
                    yaxis_title=value_col,
                    xaxis=dict(
                        tickmode="array",
                        tickvals=list(range(1, len(x_labels)+1)) if x_labels else None,
                        ticktext=x_labels if x_labels else None
                    )
                )
                return _apply_tsforge_style(fig, engine="plotly", context="seasonal")

            elif mode == "facet":
                fig = make_subplots(rows=n_series, cols=1, shared_xaxes=True,
                                    subplot_titles=[str(uid) for uid in ids])
                for r, uid in enumerate(ids, 1):
                    sub = df_sub[df_sub[id_col] == uid]
                    for j, (year, g) in enumerate(sub.groupby("year", observed=True)):
                        g = g.sort_values("seasonal_x")
                        fig.add_trace(go.Scatter(
                            x=g["seasonal_x"], y=g[value_col],
                            mode="lines+markers",
                            connectgaps=False,
                            name=f"{uid} - {year}",
                            line=dict(color=PALETTE[j % len(PALETTE)], width=2),
                            opacity=0.9,
                            showlegend=(r == 1)
                        ), row=r, col=1)
                fig.update_layout(
                    height=300*n_series,
                    title=dict(text=f"Seasonal Subseries Facet (freq={freq})", y=0.98),
                    legend=dict(yanchor="top", y=0.92)
                )
                return _apply_tsforge_style(fig, engine="plotly", context="seasonal")

            elif mode == "dropdown":
                fig = go.Figure()
                for i, uid in enumerate(ids):
                    sub = df_sub[df_sub[id_col] == uid]
                    for j, (year, g) in enumerate(sub.groupby("year", observed=True)):
                        g = g.sort_values("seasonal_x")
                        fig.add_trace(go.Scatter(
                            x=g["seasonal_x"], y=g[value_col],
                            mode="lines+markers",
                            connectgaps=False,
                            name=f"{uid} - {year}",
                            line=dict(color=PALETTE[j % len(PALETTE)], width=2),
                            opacity=0.9,
                            visible=(i == 0)
                        ))
                # dropdown by series
                buttons = []
                for i, uid in enumerate(ids):
                    visible = []
                    for uid2 in ids:
                        years = df_sub[df_sub[id_col] == uid2]["year"].unique()
                        for _ in years:
                            visible.append(uid2 == uid)
                    buttons.append(dict(label=str(uid),
                                        method="update",
                                        args=[{"visible": visible}]))
                fig.update_layout(
                    updatemenus=[{
                        "buttons": buttons,
                        "direction": "down",
                        "x": 1.05, "y": 1.15,
                        "xanchor": "left", "yanchor": "top"
                    }],
                    title=dict(text=f"Seasonal Subseries Dropdown (freq={freq})", y=0.98),
                    legend=dict(yanchor="top", y=0.92),
                    xaxis=dict(
                        tickmode="array",
                        tickvals=list(range(1, len(x_labels)+1)) if x_labels else None,
                        ticktext=x_labels if x_labels else None
                    )
                )
                return _apply_tsforge_style(fig, engine="plotly", context="seasonal")

        elif engine == "matplotlib":
            import matplotlib.pyplot as plt

            if mode == "overlay":
                fig, ax = plt.subplots(figsize=(10, 6))
                for i, (uid, sub) in enumerate(df_sub.groupby(id_col, observed=True)):
                    for j, (year, g) in enumerate(sub.groupby("year", observed=True)):
                        g = g.sort_values("seasonal_x")
                        ax.plot(g["seasonal_x"], g[value_col],
                                label=f"{uid} - {year}" if n_series > 1 else str(year),
                                color=PALETTE[j % len(PALETTE)], lw=2, alpha=0.9)
                ax.set_title(f"Seasonal Subseries Plot (freq={freq})", fontsize=14, pad=20)
                ax.set_xlabel(f"Cycle position ({freq})")
                ax.set_ylabel(value_col)
                if x_labels:
                    ax.set_xticks(range(1, len(x_labels)+1))
                    ax.set_xticklabels(x_labels)
                ax.legend()
                plt.tight_layout()
                return _apply_tsforge_style(fig, engine="matplotlib", context="seasonal")

            elif mode == "facet":
                fig, axes = plt.subplots(n_series, 1, figsize=(10, 4*n_series), sharex=True)
                axes = np.atleast_1d(axes)
                for ax, uid in zip(axes, ids):
                    sub = df_sub[df_sub[id_col] == uid]
                    for j, (year, g) in enumerate(sub.groupby("year", observed=True)):
                        g = g.sort_values("seasonal_x")
                        ax.plot(g["seasonal_x"], g[value_col],
                                label=f"{uid} - {year}",
                                color=PALETTE[j % len(PALETTE)], lw=2, alpha=0.9)
                    ax.set_title(f"Series {uid}")
                    ax.legend()
                axes[-1].set_xlabel(f"Cycle position ({freq})")
                plt.tight_layout()
                return _apply_tsforge_style(fig, engine="matplotlib", context="seasonal")

    raise ValueError("Invalid combination of kind, mode, and engine.")
