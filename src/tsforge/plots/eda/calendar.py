# tsforge/plots/eda/calendar.py
"""Calendar heatmap visualization."""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Literal

import plotly.graph_objects as go

from .._styling import apply_theme
from .._layout import finalize_figure
from .._display import render_by_mode


def plot_calendar_heatmap(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str = "y",
    ids: Union[str, List[str], int, None] = None,
    group_col: Optional[str] = None,
    agg: str = "sum",
    mode: Literal["facet", "dropdown"] = "facet",
    wrap: int = 3,
    theme: str = "fa",
    style: Optional[dict] = None,
):
    """
    Calendar heatmap: Day of month (x) vs Month (y).

    Values are normalized 0-1 per series for pattern spotting.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with id, date, and value columns.
    id_col : str
        Column identifying each time series.
    date_col : str
        Column containing dates/timestamps.
    value_col : str, default "y"
        Column containing values.
    ids : str, list, int, or None, optional
        Specific series to plot. If int, samples N random series.
    group_col : str, optional
        Column to group by (replaces id_col).
    agg : str, default "sum"
        Aggregation function.
    mode : {"facet", "dropdown"}, default "facet"
        Display mode for multiple series.
    wrap : int, default 3
        Columns for facet mode.
    theme : str, default "fa"
        Theme name.
    style : dict, optional
        Style overrides.

    Returns
    -------
    go.Figure
        Plotly figure object.
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

    # Normalize values per series
    df[value_col] = df.groupby(id_col)[value_col].transform(
        lambda x: x / x.max() if x.max() > 0 else x
    )

    # Build traces for each series
    traces_by_id = {}
    for i, uid in enumerate(ids):
        traces_by_id[uid] = _build_heatmap_traces(
            df[df[id_col] == uid], uid, value_col, show_colorbar=(i == 0)
        )

    # Render using unified display module
    fig = render_by_mode(
        traces_by_id,
        mode=mode,
        wrap=wrap,
        row_height=400,
        vertical_spacing=0.2,
        theme=theme,
        style=style,
        finalize=False,
    )

    # Apply axis formatting
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig.update_xaxes(
        showgrid=True,
        gridcolor="lightgrey",
        dtick=1,
        title="Day of Month",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="lightgrey",
        dtick=1,
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=month_names,
        title="Month",
    )

    # Set width for facet mode
    if mode == "facet":
        n_cols = min(wrap, len(ids))
        fig.update_layout(width=n_cols * 600)

    return finalize_figure(fig, theme, style)


def _build_heatmap_traces(
    sub: pd.DataFrame,
    uid: str,
    value_col: str,
    show_colorbar: bool,
) -> List[go.BaseTraceType]:
    """Build heatmap trace for a single series."""
    return [go.Heatmap(
        x=sub["day"],
        y=sub["month"],
        z=sub[value_col],
        zmin=0,
        zmax=1,
        colorscale="YlGnBu",
        colorbar=dict(title="Normalized") if show_colorbar else None,
        showscale=show_colorbar,
        name=str(uid),
    )]
