# tsforge/plots/charts/seasonal.py
"""Seasonal subseries visualization."""
from __future__ import annotations

import pandas as pd
import numpy as np
from math import ceil
from typing import Union, List, Optional, Literal

from .._styling import PALETTE, HIGHLIGHT, apply_theme, apply_legend
from .._preprocessing import aggregate_by_group, select_ids
from .._layout import finalize_figure


def plot_seasonal(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    ids: Union[None, int, str, List[str]] = None,
    group_col: Union[str, List[str], None] = None,
    agg: str = "sum",
    seasonal_agg: str = "mean",
    freq: str = "M",
    mode: Literal["overlay", "facet", "dropdown"] = "overlay",
    kind: Literal["line", "box"] = "line",
    normalize: bool = False,
    show_mean: bool = False,
    wrap: Optional[int] = None,
    theme: str = "fa",
    style: Optional[dict] = None,
    engine: str = "plotly",
):
    """
    Seasonal subseries visualization.

    Supports monthly, quarterly, weekly, or daily cycles with
    overlay, facet, and dropdown layouts.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if engine != "plotly":
        raise NotImplementedError("Only Plotly engine is supported.")

    # Preprocessing
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df, id_col = aggregate_by_group(df, group_col, date_col, value_col, agg, id_col)
    ids = select_ids(df, id_col, ids, 6)
    df = df[df[id_col].isin(ids)].copy()

    # Extract cycle components
    df["year"] = df[date_col].dt.year
    f = freq.upper()
    if f == "M":
        df["seasonal_x"] = df[date_col].dt.month
        x_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    elif f == "Q":
        df["seasonal_x"] = df[date_col].dt.quarter
        x_labels = ["Q1", "Q2", "Q3", "Q4"]
    elif f == "W":
        df["seasonal_x"] = df[date_col].dt.isocalendar().week.astype(int)
        x_labels = None
    elif f == "D":
        df["seasonal_x"] = df[date_col].dt.dayofyear
        x_labels = None
    else:
        raise ValueError("freq must be one of M, Q, W, D")

    # Normalization
    y_axis_label = value_col
    if normalize:
        yearly_mean = df.groupby([id_col, "year"])[value_col].transform("mean")
        df[value_col] = df[value_col] / yearly_mean
        y_axis_label = f"{value_col} (normalized)"

    # Collapse to seasonal aggregates
    df = (
        df.groupby([id_col, "year", "seasonal_x"], observed=True)
        [value_col].agg(seasonal_agg).reset_index()
    )

    # Mean seasonal profile
    mean_df = None
    if show_mean and kind == "line":
        mean_df = (
            df.groupby([id_col, "seasonal_x"], observed=True)[value_col]
            .mean().reset_index()
        )

    # Build plot based on kind and mode
    if kind == "box":
        fig = _plot_box(df, ids, id_col, value_col, x_labels, mode, wrap)
    else:
        fig = _plot_line(df, ids, id_col, value_col, x_labels, mode, wrap,
                         show_mean, mean_df)

    fig.update_yaxes(title_text=y_axis_label)

    return finalize_figure(fig, theme, style)


def _plot_box(df, ids, id_col, value_col, x_labels, mode, wrap):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if mode == "overlay":
        fig = go.Figure()
        for uid in ids:
            sub = df[df[id_col] == uid]
            fig.add_trace(go.Box(
                x=sub["seasonal_x"], y=sub[value_col],
                name=str(uid), boxpoints="all", jitter=0.2, pointpos=0,
                marker=dict(opacity=0.6, size=4, color=HIGHLIGHT),
            ))
    elif mode == "facet":
        n = len(ids)
        cols = wrap or 1
        rows = ceil(n / cols)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[str(i) for i in ids])
        r, c = 1, 1
        for uid in ids:
            sub = df[df[id_col] == uid]
            fig.add_trace(go.Box(
                x=sub["seasonal_x"], y=sub[value_col], name=str(uid),
                boxpoints="all", jitter=0.2, showlegend=False,
                marker=dict(opacity=0.6, size=4, color=HIGHLIGHT),
            ), row=r, col=c)
            c += 1
            if c > cols:
                r += 1
                c = 1
    else:  # dropdown
        fig = go.Figure()
        for i, uid in enumerate(ids):
            sub = df[df[id_col] == uid]
            fig.add_trace(go.Box(
                x=sub["seasonal_x"], y=sub[value_col], name=str(uid),
                boxpoints="all", jitter=0.2, visible=(i == 0), showlegend=False,
                marker=dict(opacity=0.6, size=4, color=HIGHLIGHT),
            ))
        buttons = [
            dict(label=str(uid), method="update",
                 args=[{"visible": [j == i for j in range(len(ids))]}])
            for i, uid in enumerate(ids)
        ]
        fig.update_layout(updatemenus=[{
            "buttons": buttons, "direction": "down",
            "x": 1.0, "y": 1.15, "xanchor": "right", "yanchor": "top",
        }])

    if x_labels:
        fig.update_xaxes(tickmode="array", tickvals=list(range(1, len(x_labels) + 1)), ticktext=x_labels)
    return fig


def _plot_line(df, ids, id_col, value_col, x_labels, mode, wrap, show_mean, mean_df):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    opacity = 0.35 if show_mean else 0.9
    show_year_legend = not show_mean

    if mode == "overlay":
        fig = go.Figure()
        for uid in ids:
            sub = df[df[id_col] == uid]
            for j, (yr, g) in enumerate(sub.groupby("year")):
                g = g.sort_values("seasonal_x")
                is_first = (uid == ids[0])
                fig.add_trace(go.Scatter(
                    x=g["seasonal_x"], y=g[value_col], mode="lines+markers",
                    name=str(yr), legendgroup=str(yr),
                    line=dict(color=PALETTE[j % len(PALETTE)], width=2),
                    opacity=opacity, showlegend=show_year_legend and is_first,
                ))
            if show_mean and mean_df is not None:
                m = mean_df[mean_df[id_col] == uid].sort_values("seasonal_x")
                fig.add_trace(go.Scatter(
                    x=m["seasonal_x"], y=m[value_col], mode="lines",
                    name="Mean" if len(ids) == 1 else f"{uid} mean",
                    line=dict(color="black", width=4, dash="dash"),
                ))
    elif mode == "facet":
        n = len(ids)
        cols = wrap or 1
        rows = ceil(n / cols)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[str(i) for i in ids], shared_xaxes=True)
        years_in_legend = set()
        r, c = 1, 1
        for uid in ids:
            sub = df[df[id_col] == uid]
            for j, (yr, g) in enumerate(sub.groupby("year")):
                g = g.sort_values("seasonal_x")
                add_legend = show_year_legend and (yr not in years_in_legend)
                if add_legend:
                    years_in_legend.add(yr)
                fig.add_trace(go.Scatter(
                    x=g["seasonal_x"], y=g[value_col], mode="lines+markers",
                    name=str(yr), legendgroup=str(yr),
                    line=dict(color=PALETTE[j % len(PALETTE)], width=2),
                    opacity=opacity, showlegend=add_legend,
                ), row=r, col=c)
            if show_mean and mean_df is not None:
                m = mean_df[mean_df[id_col] == uid].sort_values("seasonal_x")
                show_mean_legend = (r == 1 and c == 1)
                fig.add_trace(go.Scatter(
                    x=m["seasonal_x"], y=m[value_col], mode="lines",
                    name="Mean", legendgroup="mean",
                    line=dict(color="black", width=4, dash="dash"),
                    showlegend=show_mean_legend,
                ), row=r, col=c)
            c += 1
            if c > cols:
                r += 1
                c = 1
    else:  # dropdown
        fig = go.Figure()
        trace_counts = []
        unique_years = sorted(df["year"].unique())
        year_to_idx = {yr: i for i, yr in enumerate(unique_years)}
        for idx, uid in enumerate(ids):
            sub = df[df[id_col] == uid]
            local_count = 0
            for j, (yr, g) in enumerate(sub.groupby("year")):
                g = g.sort_values("seasonal_x")
                color_idx = year_to_idx.get(yr, j)
                is_first = (idx == 0)
                fig.add_trace(go.Scatter(
                    x=g["seasonal_x"], y=g[value_col], mode="lines+markers",
                    name=str(yr), legendgroup=str(yr),
                    line=dict(color=PALETTE[color_idx % len(PALETTE)], width=2),
                    opacity=opacity, visible=(idx == 0),
                    showlegend=show_year_legend and is_first,
                ))
                local_count += 1
            if show_mean and mean_df is not None:
                m = mean_df[mean_df[id_col] == uid].sort_values("seasonal_x")
                fig.add_trace(go.Scatter(
                    x=m["seasonal_x"], y=m[value_col], mode="lines",
                    name="Mean", legendgroup="mean",
                    line=dict(color="black", width=4, dash="dash"),
                    visible=(idx == 0), showlegend=(idx == 0),
                ))
                local_count += 1
            trace_counts.append(local_count)

        buttons = []
        total = sum(trace_counts)
        start = 0
        for i, uid in enumerate(ids):
            visible = [False] * total
            for k in range(start, start + trace_counts[i]):
                visible[k] = True
            buttons.append(dict(label=str(uid), method="update", args=[{"visible": visible}]))
            start += trace_counts[i]
        fig.update_layout(updatemenus=[{
            "buttons": buttons, "direction": "down",
            "x": 1.0, "y": 1.15, "xanchor": "right", "yanchor": "top",
        }])

    if x_labels:
        fig.update_xaxes(tickmode="array", tickvals=list(range(1, len(x_labels) + 1)), ticktext=x_labels)
    return fig
