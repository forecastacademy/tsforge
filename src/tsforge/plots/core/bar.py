# tsforge/plots/core/bar.py
"""Bar chart visualization for tsforge.

Supports grouped bars, stacked bars, horizontal/vertical orientation,
and overlay/facet/dropdown modes.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Literal, Dict

import plotly.graph_objects as go

from .._styling import PALETTE, apply_theme
from .._layout import finalize_figure
from .._display import render_by_mode


def plot_bar(
    df: pd.DataFrame,
    *,
    # Core parameters (consistent with plot_timeseries/plot_distribution)
    id_col: str,
    value_col: str,
    group_col: Optional[str] = None,
    agg: str = "mean",
    ids: Union[None, int, str, List[str]] = None,
    max_ids: int = 20,
    # Bar-specific options
    color_col: Optional[str] = None,
    orientation: Literal["v", "h"] = "v",
    barmode: Literal["group", "stack", "relative"] = "group",
    sort_by: Optional[str] = None,
    sort_ascending: bool = True,
    top_n: Optional[int] = None,
    show_values: bool = False,
    value_format: str = ".2f",
    # Threshold/reference lines
    thresholds: Optional[List[float]] = None,
    threshold_labels: Optional[List[str]] = None,
    threshold_color: str = "#e74c3c",
    # Layout
    mode: Literal["overlay", "facet", "dropdown"] = "overlay",
    wrap: int = 2,
    row_height: int = 350,
    col_width: int = 400,
    vertical_spacing: float = 0.15,
    # Styling
    colors: Optional[Union[str, Dict[str, str]]] = None,
    theme: str = "fa",
    style: Optional[dict] = None,
) -> go.Figure:
    """
    Bar chart visualization with flexible options.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    id_col : str
        Column for categories (x-axis for vertical, y-axis for horizontal).
    value_col : str
        Column for values (y-axis for vertical, x-axis for horizontal).
    group_col : str, optional
        Column to aggregate by before plotting. If set, data is grouped by this
        column and id_col is replaced by group_col values. Works like plot_timeseries.
    agg : str, default "mean"
        Aggregation function ("mean", "sum", "count", "median").
    ids : None, int, str, or list, optional
        Specific categories to include. If int, includes first N.
    max_ids : int, default 20
        Maximum categories to show if ids not specified.
    color_col : str, optional
        Column for creating grouped/stacked bars (multiple bar series).
    orientation : {"v", "h"}, default "v"
        Vertical or horizontal bars.
    barmode : {"group", "stack", "relative"}, default "group"
        How to arrange multiple bar traces when color_col is set.
    sort_by : str, optional
        Column to sort by. Defaults to value_col descending for horizontal.
    sort_ascending : bool, default True
        Sort direction.
    top_n : int, optional
        Only show top N categories (by value).
    show_values : bool, default False
        Show value labels on bars.
    value_format : str, default ".2f"
        Format string for value labels.
    thresholds : list of float, optional
        Reference lines to add.
    threshold_labels : list of str, optional
        Labels for threshold lines.
    threshold_color : str, default "#e74c3c"
        Color for threshold lines.
    mode : {"overlay", "facet", "dropdown"}, default "overlay"
        Display mode when color_col creates multiple series.
    wrap : int, default 2
        Columns for facet mode.
    row_height : int, default 350
        Pixel height per facet row.
    col_width : int, default 400
        Pixel width per facet column.
    vertical_spacing : float, default 0.15
        Spacing between facet rows (0-1 fraction).
    colors : str or dict, optional
        Color(s) for bars. Dict maps color_col values to colors.
    theme : str, default "fa"
        Theme name.
    style : dict, optional
        Style overrides.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> # Simple bar chart (auto-aggregates by id_col)
    >>> plot_bar(df, id_col='category', value_col='sales')

    >>> # Aggregate by a different column
    >>> plot_bar(df, id_col='unique_id', value_col='sales', group_col='category')

    >>> # Horizontal bar chart sorted by value
    >>> plot_bar(df, id_col='product', value_col='revenue', orientation='h')

    >>> # Grouped bar chart with colors
    >>> plot_bar(df, id_col='month', value_col='sales', color_col='region')
    """
    df = df.copy()

    # Handle group_col aggregation (like plot_timeseries)
    effective_id_col = id_col
    if group_col is not None:
        # Aggregate by group_col, replacing id_col
        agg_df = df.groupby(group_col)[value_col].agg(agg).reset_index()
        effective_id_col = group_col
    elif color_col is not None:
        # Group by both id_col and color_col for colored/stacked bars
        agg_df = df.groupby([id_col, color_col])[value_col].agg(agg).reset_index()
    else:
        # Check if we need to aggregate
        if df[id_col].duplicated().any():
            agg_df = df.groupby(id_col)[value_col].agg(agg).reset_index()
        else:
            agg_df = df[[id_col, value_col]].copy()

    # Default sort for horizontal bars
    if sort_by is None and orientation == "h":
        sort_by = value_col
        sort_ascending = False

    # Sort if requested
    if sort_by is not None:
        if color_col is not None:
            # Sort by total per category
            cat_totals = agg_df.groupby(effective_id_col)[value_col].sum().sort_values(ascending=sort_ascending)
            cat_order = cat_totals.index.tolist()
            agg_df[effective_id_col] = pd.Categorical(agg_df[effective_id_col], categories=cat_order, ordered=True)
            agg_df = agg_df.sort_values(effective_id_col)
        else:
            agg_df = agg_df.sort_values(sort_by, ascending=sort_ascending)

    # Select categories
    unique_cats = agg_df[effective_id_col].unique().tolist()
    if ids is not None:
        if isinstance(ids, int):
            unique_cats = unique_cats[:ids]
        elif isinstance(ids, str):
            unique_cats = [ids]
        else:
            unique_cats = [c for c in ids if c in unique_cats]
    elif top_n is not None:
        unique_cats = unique_cats[:top_n]
    else:
        unique_cats = unique_cats[:max_ids]

    agg_df = agg_df[agg_df[effective_id_col].isin(unique_cats)]

    # Normalize colors
    if isinstance(colors, str):
        color_map = {effective_id_col: colors}
    else:
        color_map = colors or {}

    # Build traces
    if color_col is None:
        # Single bar series
        color = color_map.get(effective_id_col, PALETTE[0])
        traces_by_id = {"data": _build_bar_traces(
            agg_df, effective_id_col, value_col, orientation, show_values, value_format, color, value_col
        )}
    else:
        # Multiple bar series by color_col
        unique_groups = agg_df[color_col].dropna().unique().tolist()

        traces_by_id = {}
        for i, gval in enumerate(unique_groups):
            sub = agg_df[agg_df[color_col] == gval]
            c = color_map.get(gval, PALETTE[i % len(PALETTE)])
            traces_by_id[str(gval)] = _build_bar_traces(
                sub, effective_id_col, value_col, orientation, show_values, value_format, c, str(gval)
            )

    # Use render_by_mode if multiple series, otherwise simple figure
    if len(traces_by_id) == 1 and "data" in traces_by_id:
        fig = go.Figure(data=traces_by_id["data"])
        fig = apply_theme(fig, theme)
        if style:
            fig = finalize_figure(fig, theme, style)
    else:
        fig = render_by_mode(
            traces_by_id,
            mode=mode,
            wrap=wrap,
            row_height=row_height,
            col_width=col_width,
            vertical_spacing=vertical_spacing,
            theme=theme,
            style=style,
        )
        fig.update_layout(barmode=barmode)

    # Add threshold lines
    if thresholds:
        for i, thresh in enumerate(thresholds):
            label = threshold_labels[i] if threshold_labels and i < len(threshold_labels) else None
            if orientation == "v":
                fig.add_hline(
                    y=thresh, line_dash="dot", line_color=threshold_color, line_width=2,
                    annotation_text=label, annotation_position="right" if label else None
                )
            else:
                fig.add_vline(
                    x=thresh, line_dash="dot", line_color=threshold_color, line_width=2,
                    annotation_text=label, annotation_position="top" if label else None
                )

    # Update axis labels
    id_label = effective_id_col.replace("_", " ").title()
    value_label = value_col.replace("_", " ").title()
    if orientation == "v":
        fig.update_xaxes(title_text=id_label)
        fig.update_yaxes(title_text=value_label)
    else:
        fig.update_xaxes(title_text=value_label)
        fig.update_yaxes(title_text=id_label)

    return fig


def _build_bar_traces(
    df: pd.DataFrame,
    id_col: str,
    value_col: str,
    orientation: str,
    show_values: bool,
    value_format: str,
    color: str,
    name: str,
) -> List[go.BaseTraceType]:
    """Build bar trace for a single series."""
    text = df[value_col].apply(lambda v: f"{v:{value_format}}") if show_values else None
    textposition = "outside" if show_values else None

    if orientation == "v":
        return [go.Bar(
            x=df[id_col],
            y=df[value_col],
            name=name,
            marker=dict(color=color),
            text=text,
            textposition=textposition,
            hovertemplate=f"<b>%{{x}}</b><br>{value_col}: %{{y:{value_format}}}<extra></extra>",
        )]
    else:
        return [go.Bar(
            x=df[value_col],
            y=df[id_col],
            name=name,
            marker=dict(color=color),
            orientation="h",
            text=text,
            textposition=textposition,
            hovertemplate=f"<b>%{{y}}</b><br>{value_col}: %{{x:{value_format}}}<extra></extra>",
        )]


def plot_category_counts(
    df: pd.DataFrame,
    category_col: str,
    *,
    weight_col: Optional[str] = None,
    order: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    show_pct: bool = True,
    orientation: Literal["v", "h"] = "h",
    theme: str = "fa",
    style: Optional[dict] = None,
) -> go.Figure:
    """
    Plot category distribution as bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    category_col : str
        Column containing category labels.
    weight_col : str, optional
        Column for weighted counts.
    order : list of str, optional
        Category order.
    colors : dict, optional
        Color mapping {category: color}.
    show_pct : bool, default True
        Show percentage labels.
    orientation : {"v", "h"}, default "h"
        Bar orientation.
    theme : str, default "fa"
        Theme name.
    style : dict, optional
        Style overrides.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    counts = df[category_col].value_counts()
    total = len(df)

    if order is None:
        order = counts.index.tolist()

    if colors is None:
        colors = {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(order)}

    values = [counts.get(cat, 0) for cat in order]
    bar_colors = [colors.get(cat, PALETTE[0]) for cat in order]

    if show_pct:
        text = [f"{v:,} ({100*v/total:.1f}%)" for v in values]
    else:
        text = [f"{v:,}" for v in values]

    fig = go.Figure()

    if orientation == "h":
        fig.add_trace(go.Bar(
            x=values,
            y=order,
            orientation="h",
            marker=dict(color=bar_colors),
            text=text,
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
        ))
        fig.update_xaxes(title_text="Count")
        fig.update_yaxes(autorange="reversed")
    else:
        fig.add_trace(go.Bar(
            x=order,
            y=values,
            marker=dict(color=bar_colors),
            text=text,
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
        ))
        fig.update_yaxes(title_text="Count")

    return finalize_figure(fig, theme, style)
