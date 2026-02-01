# tsforge/plots/charts/seasonal.py
"""Seasonal subseries visualization."""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Literal

import plotly.graph_objects as go

from .._styling import PALETTE, HIGHLIGHT
from .._preprocessing import aggregate_by_group, select_ids
from .._layout import finalize_figure
from .._display import render_by_mode


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

    Parameters
    ----------
    df : pd.DataFrame
        Input data with id, date, and value columns.
    id_col : str
        Column identifying each time series.
    date_col : str
        Column containing dates/timestamps.
    value_col : str
        Column containing values.
    ids : None, int, str, or list, optional
        Specific series to plot. If int, plots first N series.
    group_col : str or list, optional
        Column(s) to group by before plotting.
    agg : str, default "sum"
        Aggregation function for grouping.
    seasonal_agg : str, default "mean"
        Aggregation within seasonal periods.
    freq : str, default "M"
        Seasonal frequency: "M" (monthly), "Q" (quarterly), "W" (weekly), "D" (daily).
    mode : {"overlay", "facet", "dropdown"}, default "overlay"
        Display mode for multiple series.
    kind : {"line", "box"}, default "line"
        Chart type.
    normalize : bool, default False
        Normalize values by yearly mean.
    show_mean : bool, default False
        Show mean seasonal profile (line kind only).
    wrap : int, optional
        Columns for facet mode (default: 1).
    theme : str, default "fa"
        Theme name.
    style : dict, optional
        Style overrides.
    engine : str, default "plotly"
        Plotting engine (only "plotly" supported).

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
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

    # Build year-to-color mapping for consistent colors
    unique_years = sorted(df["year"].unique())
    year_colors = {yr: PALETTE[i % len(PALETTE)] for i, yr in enumerate(unique_years)}

    # Build traces for each series
    traces_by_id = {}
    for i, uid in enumerate(ids):
        sub = df[df[id_col] == uid]
        is_first = (i == 0)

        if kind == "box":
            traces_by_id[uid] = _build_box_traces(sub, uid, value_col)
        else:
            mean_sub = mean_df[mean_df[id_col] == uid] if mean_df is not None else None
            traces_by_id[uid] = _build_line_traces(
                sub, uid, value_col, mean_sub, show_mean,
                year_colors, is_first,
            )

    # Render using unified display module
    fig = render_by_mode(
        traces_by_id,
        mode=mode,
        wrap=wrap or 1,
        theme=theme,
        style=style,
        finalize=False,
    )

    # Apply x-axis labels
    if x_labels:
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(1, len(x_labels) + 1)),
            ticktext=x_labels,
        )

    fig.update_yaxes(title_text=y_axis_label)

    return finalize_figure(fig, theme, style)


def _build_box_traces(
    sub: pd.DataFrame,
    uid: str,
    value_col: str,
) -> List[go.BaseTraceType]:
    """Build box plot traces for a single series."""
    return [go.Box(
        x=sub["seasonal_x"],
        y=sub[value_col],
        name=str(uid),
        boxpoints="all",
        jitter=0.2,
        pointpos=0,
        marker=dict(opacity=0.6, size=4, color=HIGHLIGHT),
    )]


def _build_line_traces(
    sub: pd.DataFrame,
    uid: str,
    value_col: str,
    mean_sub: Optional[pd.DataFrame],
    show_mean: bool,
    year_colors: dict,
    is_first: bool,
) -> List[go.BaseTraceType]:
    """Build line traces for a single series (one trace per year + optional mean)."""
    traces = []
    opacity = 0.35 if show_mean else 0.9
    show_year_legend = not show_mean

    for yr, g in sub.groupby("year"):
        g = g.sort_values("seasonal_x")
        traces.append(go.Scatter(
            x=g["seasonal_x"],
            y=g[value_col],
            mode="lines+markers",
            name=str(yr),
            legendgroup=str(yr),
            line=dict(color=year_colors.get(yr, PALETTE[0]), width=2),
            opacity=opacity,
            showlegend=show_year_legend and is_first,
        ))

    if show_mean and mean_sub is not None and len(mean_sub) > 0:
        m = mean_sub.sort_values("seasonal_x")
        traces.append(go.Scatter(
            x=m["seasonal_x"],
            y=m[value_col],
            mode="lines",
            name="Mean",
            legendgroup="mean",
            line=dict(color="black", width=4, dash="dash"),
            showlegend=is_first,
        ))

    return traces
