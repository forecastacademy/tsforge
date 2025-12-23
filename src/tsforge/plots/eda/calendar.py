# tsforge/plots/eda/calendar.py
"""Calendar heatmap visualization."""
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .._styling import apply_theme


def plot_calendar_heatmap(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str = "y",
    ids: str | list[str] | int | None = None,
    group_col: str | None = None,
    agg: str = "sum",
    mode: str = "facet",
    wrap: int = 3,
):
    """
    Simple calendar heatmap: Day of month (x) vs Month (y),
    normalized 0-1 per series/group for pattern spotting.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if group_col:
        group_keys = [group_col, date_col]
        df = df.groupby(group_keys, observed=True)[value_col].agg(agg).reset_index()
        id_col = group_col

    unique_ids = df[id_col].dropna().unique().tolist()

    if ids is None:
        ids = pd.Series(unique_ids).sample(min(3, len(unique_ids)), random_state=42).tolist()
    elif isinstance(ids, int):
        ids = pd.Series(unique_ids).sample(min(ids, len(unique_ids)), random_state=42).tolist()
    elif isinstance(ids, str):
        ids = [ids]
    else:
        ids = list(ids)

    df = df.groupby([id_col, date_col], observed=True)[value_col].agg(agg).reset_index()
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    df[value_col] = df.groupby(id_col)[value_col].transform(lambda x: x / x.max() if x.max() > 0 else x)

    if mode == "facet":
        return _plot_facet(df, ids, id_col, value_col, wrap)
    elif mode == "dropdown":
        return _plot_dropdown(df, ids, id_col, value_col, month_names)
    else:
        raise ValueError("mode must be 'facet' or 'dropdown'")


def _plot_facet(df, ids, id_col, value_col, wrap):
    n = len(ids)
    rows = int(np.ceil(n / wrap))
    cols = min(wrap, n)

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[str(uid) for uid in ids], vertical_spacing=0.2)
    fig.update_layout(height=rows * 400, width=cols * 600)

    for i, uid in enumerate(ids):
        sub = df[df[id_col] == uid]
        heat = go.Heatmap(
            x=sub["day"], y=sub["month"], z=sub[value_col],
            zmin=0, zmax=1, colorscale="YlGnBu",
            colorbar=dict(title="Normalized Sales"),
            showscale=(i == 0)
        )
        r = i // wrap + 1
        c = i % wrap + 1
        fig.add_trace(heat, row=r, col=c)

    fig.update_xaxes(showgrid=True, gridcolor="lightgrey", dtick=1, title="Day of Month")
    fig.update_yaxes(
        showgrid=True, gridcolor="lightgrey", dtick=1,
        tickmode="array", tickvals=list(range(1, 13)),
        ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        title="Month"
    )
    return apply_theme(fig, "fa")


def _plot_dropdown(df, ids, id_col, value_col, month_names):
    fig = go.Figure()
    for i, uid in enumerate(ids):
        sub = df[df[id_col] == uid]
        heat = go.Heatmap(
            x=sub["day"], y=sub["month"], z=sub[value_col],
            colorscale="YlGnBu", zmin=0, zmax=1,
            visible=(i == 0), name=str(uid)
        )
        fig.add_trace(heat)

    buttons = []
    for i, uid in enumerate(ids):
        visible = [False] * len(ids)
        visible[i] = True
        buttons.append(dict(label=str(uid), method="update",
                            args=[{"visible": visible}, {"title": f"Calendar Heatmap: {uid}"}]))

    fig.update_layout(
        updatemenus=[{"buttons": buttons, "direction": "down",
                      "x": 1.05, "y": 1.15, "xanchor": "left", "yanchor": "top"}]
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgrey", dtick=1, title="Day of Month")
    fig.update_yaxes(showgrid=True, gridcolor="lightgrey", dtick=1,
                     tickmode="array", tickvals=list(range(1, 13)), ticktext=month_names, title="Month")
    return apply_theme(fig, "fa")
