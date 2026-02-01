# tsforge/plots/core/distribution.py
"""
Unified distribution visualization for tsforge.

Supports two modes:
1. Time series value distributions (id_col, date_col, value_col)
2. Metric distributions across series (columns parameter)
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .._display import render_by_mode
from .._preprocessing import aggregate_by_group, select_ids
from .._styling import PALETTE

# Metric configuration for auto-defaults
METRIC_CONFIG = {
    # Structure metrics (blue)
    "trend": {
        "color": "#4A90A4",
        "threshold": 0.5,
        "label": "Strong Trend",
        "xlim": (0, 1),
        "description": "Trend strength measures how much variance is explained by a linear trend (0-1). Values ≥{threshold} indicate significant trend that should be modeled explicitly.",
        "action": "Consider trend/drift models",
    },
    "seasonal_strength": {
        "color": "#4A90A4",
        "threshold": 0.5,
        "label": "Strong Seasonality",
        "xlim": (0, 1),
        "description": "Seasonal strength measures recurring pattern intensity (0-1). Values ≥{threshold} indicate seasonality worth capturing in your model.",
        "action": "Use seasonal decomposition",
    },
    "x_acf1": {
        "color": "#4A90A4",
        "threshold": 0.6,
        "label": "Strong Persistence",
        "xlim": (-0.5, 1),
        "description": "First-order autocorrelation measures how much each value depends on the previous. Values ≥{threshold} indicate strong persistence.",
        "action": "AR/ARIMA models appropriate",
    },
    "structure_score": {
        "color": "#4A90A4",
        "threshold": 0.5,
        "label": "High Structure",
        "xlim": (0, 1),
        "description": "Combined measure of predictable patterns (trend + seasonality). Values ≥{threshold} indicate learnable structure.",
        "action": "Statistical models will help",
    },
    # Chaos metrics (orange) - invert_pct shows % BELOW threshold for consistency
    "entropy": {
        "color": "#D4775D",
        "threshold": 0.8,
        "label": "Chaotic",
        "label_inverted": "Predictable",
        "invert_pct": True,
        "xlim": (0, 1),
        "description": "Spectral entropy measures randomness/unpredictability (0-1). Values ≥{threshold} indicate high noise that limits forecastability.",
        "action": "Focus on prediction intervals",
    },
    "adi": {
        "color": "#D4775D",
        "threshold": 1.32,
        "label": "Intermittent",
        "label_inverted": "Regular",
        "invert_pct": True,
        "clip_quantile": 0.95,
        "description": "Average Demand Interval = mean periods between non-zero demands. Values ≥{threshold} indicate sparse/intermittent demand patterns.",
        "action": "Use Croston or SBA methods",
    },
    "cv2": {
        "color": "#D4775D",
        "threshold": 0.49,
        "label": "High Variability",
        "label_inverted": "Stable",
        "invert_pct": True,
        "clip_quantile": 0.95,
        "description": "Squared coefficient of variation of non-zero demand sizes. Values ≥{threshold} indicate erratic demand magnitudes.",
        "action": "Widen prediction intervals",
    },
    "lumpiness": {
        "color": "#D4775D",
        "threshold": 1.0,
        "label": "Lumpy",
        "label_inverted": "Smooth",
        "invert_pct": True,
        "clip_quantile": 0.95,
        "description": "Variance of tiled variances, measuring demand burstiness. Values ≥{threshold} indicate lumpy/bursty patterns.",
        "action": "Aggregate to smooth lumpiness",
    },
    "chaos_score": {
        "color": "#D4775D",
        "threshold": 0.5,
        "label": "High Chaos",
        "label_inverted": "Forecastable",
        "invert_pct": True,
        "xlim": (0, 1),
        "description": "Combined measure of unpredictability (entropy + intermittency). Values ≥{threshold} indicate difficult-to-forecast series.",
        "action": "Lower accuracy expectations",
    },
}


def plot_distribution(
    df: pd.DataFrame,
    *,
    # Time series mode parameters
    id_col: Optional[str] = None,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    ids: Union[None, int, str, List[str]] = None,
    max_ids: int = 6,
    group_col: Union[str, List[str], None] = None,
    agg: str = "sum",
    freq: Optional[str] = None,
    # Metric mode parameters
    columns: Union[str, List[str], None] = None,
    color_col: Union[str, List[str], None] = None,
    use_metric_defaults: bool = False,
    # Facet/dropdown control (metric mode with group_col)
    facet_by: Optional[Literal["columns", "group"]] = None,
    dropdown_by: Optional[Literal["columns", "group"]] = None,
    # Common parameters
    kind: Literal["histogram", "density", "box", "violin"] = "histogram",
    mode: Literal["overlay", "facet", "dropdown"] = "overlay",
    bins: int = 30,
    log_scale: bool = False,
    exclude_zeros: bool = False,
    # Threshold/reference lines
    thresholds: Optional[Union[float, Dict[str, float]]] = None,
    threshold_labels: Optional[Union[str, Dict[str, str]]] = None,
    threshold_color: str = "#e74c3c",
    show_threshold_pct: bool = False,
    show_threshold_legend: bool = False,
    # Statistics annotations
    show_kde: bool = False,
    show_median: bool = False,
    show_mean: bool = False,
    show_stats: bool = False,
    # Data transformation
    clip_quantile: Optional[Union[float, Dict[str, float]]] = None,
    # Layout
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
    Unified distribution visualization.

    Supports two modes:
    1. **Time series mode**: Distribution of values for specific series (use id_col, date_col, value_col)
    2. **Metric mode**: Distribution of metrics across all series (use columns parameter)

    Parameters
    ----------
    df : pd.DataFrame
        Input data.

    Time Series Mode
    ----------------
    id_col : str, optional
        Column identifying each time series.
    date_col : str, optional
        Column containing dates/timestamps.
    value_col : str, optional
        Column containing values to plot distribution of.
    ids : None, int, str, or list, optional
        Specific series to plot. If int, plots first N series.
    max_ids : int, default 6
        Maximum number of series to plot if ids not specified.
    group_col : str or list, optional
        Column(s) to group by. Behavior depends on mode:
        - Time series mode: aggregates data by group_col before plotting
        - Metric mode: use with facet_by/dropdown_by to control layout
    agg : str, default "sum"
        Aggregation function for grouping.
    freq : str, optional
        Resample to this frequency (e.g., "W", "M").

    Metric Mode
    -----------
    columns : str or list, optional
        Metric column(s) to plot distributions for. When provided, switches to metric mode.
    color_col : str or list, optional
        Column(s) for overlaid histograms within each facet (metric mode only).
        Creates overlaid histograms per color_col value.
    use_metric_defaults : bool, default False
        Auto-detect thresholds, colors, and clipping for known metrics.
    facet_by : {"columns", "group"}, optional
        What to create subplots for. Requires group_col to be set.
        - "columns": one subplot per metric column
        - "group": one subplot per group_col value
    dropdown_by : {"columns", "group"}, optional
        What to put in dropdown selector. Requires group_col to be set.
        - "columns": dropdown switches between metric columns
        - "group": dropdown switches between group_col values

    Common Parameters
    -----------------
    kind : {"histogram", "density", "box", "violin"}, default "histogram"
        Type of distribution visualization.
    mode : {"overlay", "facet", "dropdown"}, default "overlay"
        Display mode for multiple distributions.
    bins : int, default 30
        Number of bins for histogram/density.
    log_scale : bool, default False
        Use log scale for values.
    exclude_zeros : bool, default False
        Exclude zero values from distribution.
    thresholds : float or dict, optional
        Reference threshold lines. Dict maps column/id to threshold value.
    threshold_labels : str or dict, optional
        Labels for threshold lines.
    threshold_color : str, default "#e74c3c"
        Color for threshold lines.
    show_threshold_pct : bool, default False
        Show percentage of values above threshold.
    show_kde : bool, default False
        Overlay KDE curve on histogram.
    show_median : bool, default False
        Show median line with annotation.
    show_mean : bool, default False
        Show mean line with annotation.
    show_stats : bool, default False
        Show stats box annotation.
    clip_quantile : float or dict, optional
        Clip extreme values at this quantile.
    wrap : int, default 2
        Columns for facet mode.
    colors : str or dict, optional
        Colors for distributions.
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
    >>> # Time series mode: distribution of sales values
    >>> plot_distribution(
    ...     weekly_df,
    ...     id_col='unique_id', date_col='ds', value_col='y',
    ...     ids=3,
    ...     mode='facet',
    ... )

    >>> # Metric mode: distribution of diagnostic metrics
    >>> plot_distribution(
    ...     diagnostics,
    ...     columns=['trend', 'seasonal_strength', 'entropy', 'adi'],
    ...     use_metric_defaults=True,
    ...     show_kde=True,
    ...     show_median=True,
    ...     show_threshold_pct=True,
    ... )
    """
    # Determine mode based on parameters
    if columns is not None:
        return _plot_metric_distribution(
            df=df,
            columns=columns if isinstance(columns, list) else [columns],
            group_col=group_col,
            color_col=color_col,
            use_metric_defaults=use_metric_defaults,
            facet_by=facet_by,
            dropdown_by=dropdown_by,
            kind=kind,
            mode=mode,
            bins=bins,
            thresholds=thresholds,
            threshold_labels=threshold_labels,
            threshold_color=threshold_color,
            show_threshold_pct=show_threshold_pct,
            show_threshold_legend=show_threshold_legend,
            show_kde=show_kde,
            show_median=show_median,
            show_mean=show_mean,
            clip_quantile=clip_quantile,
            wrap=wrap,
            row_height=row_height,
            col_width=col_width,
            vertical_spacing=vertical_spacing,
            colors=colors,
            theme=theme,
            style=style,
        )
    elif id_col is not None and date_col is not None and value_col is not None:
        return _plot_timeseries_distribution(
            df=df,
            id_col=id_col,
            date_col=date_col,
            value_col=value_col,
            ids=ids,
            max_ids=max_ids,
            group_col=group_col,
            agg=agg,
            kind=kind,
            mode=mode,
            bins=bins,
            log_scale=log_scale,
            exclude_zeros=exclude_zeros,
            show_stats=show_stats,
            wrap=wrap,
            theme=theme,
            style=style,
        )
    else:
        raise ValueError(
            "Must provide either 'columns' (metric mode) or "
            "'id_col', 'date_col', 'value_col' (time series mode)"
        )


def _plot_metric_distribution(
    df: pd.DataFrame,
    columns: List[str],
    group_col: Union[str, List[str], None],
    color_col: Union[str, List[str], None],
    use_metric_defaults: bool,
    facet_by: Optional[Literal["columns", "group"]],
    dropdown_by: Optional[Literal["columns", "group"]],
    kind: str,
    mode: str,
    bins: int,
    thresholds: Optional[Union[float, Dict[str, float]]],
    threshold_labels: Optional[Union[str, Dict[str, str]]],
    threshold_color: str,
    show_threshold_pct: bool,
    show_threshold_legend: bool,
    show_kde: bool,
    show_median: bool,
    show_mean: bool,
    clip_quantile: Optional[Union[float, Dict[str, float]]],
    wrap: int,
    row_height: int,
    col_width: int,
    vertical_spacing: float,
    colors: Optional[Union[str, Dict[str, str]]],
    theme: str,
    style: Optional[dict],
) -> go.Figure:
    """Plot distribution of metrics across series (one subplot per metric)."""

    # Filter to available columns
    available = [c for c in columns if c in df.columns]
    if not available:
        raise ValueError(f"None of the columns {columns} found in DataFrame")
    columns = available

    # Resolve auto-defaults if enabled (only when not grouping)
    if use_metric_defaults and color_col is None and group_col is None:
        if thresholds is None:
            thresholds = {c: METRIC_CONFIG[c]["threshold"] for c in columns if c in METRIC_CONFIG}
        if threshold_labels is None:
            threshold_labels = {c: METRIC_CONFIG[c]["label"] for c in columns if c in METRIC_CONFIG}
        if colors is None:
            colors = {c: METRIC_CONFIG[c]["color"] for c in columns if c in METRIC_CONFIG}
        if clip_quantile is None:
            clip_quantile = {
                c: METRIC_CONFIG[c].get("clip_quantile")
                for c in columns
                if c in METRIC_CONFIG and "clip_quantile" in METRIC_CONFIG[c]
            }

    # Normalize parameters to dicts
    if isinstance(thresholds, (int, float)):
        thresholds = {c: float(thresholds) for c in columns}
    if isinstance(threshold_labels, str):
        threshold_labels = {c: threshold_labels for c in columns}
    if isinstance(colors, str):
        colors = {c: colors for c in columns}
    if isinstance(clip_quantile, (int, float)):
        clip_quantile = {c: float(clip_quantile) for c in columns}

    thresholds = thresholds or {}
    threshold_labels = threshold_labels or {}
    colors = colors or {}
    clip_quantile = clip_quantile or {}

    # Handle group_col: automatically use group for facet/dropdown selection
    if group_col is not None:
        # Auto-configure: group_col controls facet/dropdown, columns are always shown as subplots
        if mode == "dropdown":
            # Dropdown selects group, each selection shows all column subplots
            return _plot_metric_distribution_facet_dropdown(
                df=df,
                columns=columns,
                group_col=group_col,
                facet_by="columns",
                dropdown_by="group",
                kind=kind,
                bins=bins,
                thresholds=thresholds,
                threshold_labels=threshold_labels,
                threshold_color=threshold_color,
                show_threshold_pct=show_threshold_pct,
                show_threshold_legend=show_threshold_legend,
                show_kde=show_kde,
                show_median=show_median,
                clip_quantile=clip_quantile,
                wrap=wrap,
                row_height=row_height,
                col_width=col_width,
                vertical_spacing=vertical_spacing,
                colors=colors,
                theme=theme,
                style=style,
            )
        elif mode == "facet":
            # Grid: rows=groups, cols=columns
            # Use user's wrap when single column, otherwise use number of columns
            effective_wrap = wrap if len(columns) == 1 else len(columns)
            return _plot_metric_distribution_facet_dropdown(
                df=df,
                columns=columns,
                group_col=group_col,
                facet_by="group",
                dropdown_by=None,
                kind=kind,
                bins=bins,
                thresholds=thresholds,
                threshold_labels=threshold_labels,
                threshold_color=threshold_color,
                show_threshold_pct=show_threshold_pct,
                show_threshold_legend=show_threshold_legend,
                show_kde=show_kde,
                show_median=show_median,
                clip_quantile=clip_quantile,
                wrap=effective_wrap,
                row_height=row_height,
                col_width=col_width,
                vertical_spacing=vertical_spacing,
                colors=colors,
                theme=theme,
                style=style,
            )

    # Handle color grouping (overlaid histograms per group)
    if color_col is not None:
        return _plot_metric_distribution_grouped(
            df=df,
            columns=columns,
            color_col=color_col,
            kind=kind,
            mode=mode,
            bins=bins,
            thresholds=thresholds,
            threshold_labels=threshold_labels,
            threshold_color=threshold_color,
            show_threshold_pct=show_threshold_pct,
            show_threshold_legend=show_threshold_legend,
            clip_quantile=clip_quantile,
            wrap=wrap,
            row_height=row_height,
            col_width=col_width,
            vertical_spacing=vertical_spacing,
            theme=theme,
            style=style,
        )

    # Prepare data with clipping
    plot_data = {}
    for col in columns:
        data = df[col].dropna()
        if col in clip_quantile and clip_quantile[col]:
            cap = data.quantile(clip_quantile[col])
            data = data.clip(upper=cap)
        plot_data[col] = data

    # Build traces for each column (metric)
    traces_by_id: Dict[str, List[go.BaseTraceType]] = {}

    for i, col in enumerate(columns):
        data = plot_data[col]
        color = colors.get(col, PALETTE[i % len(PALETTE)])
        traces = []

        # Histogram
        traces.append(
            go.Histogram(
                x=data,
                nbinsx=bins,
                marker=dict(color=color, opacity=0.7),
                name=col,
            )
        )

        # KDE overlay
        if show_kde and len(data) > 1:
            hist, bin_edges = np.histogram(data, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # Scale KDE to match histogram
            hist_counts, _ = np.histogram(data, bins=bins)
            scale = hist_counts.max() / (hist.max() + 1e-10)
            traces.append(
                go.Scatter(
                    x=bin_centers,
                    y=hist * scale,
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"{col} (KDE)",
                    showlegend=False,
                )
            )

        traces_by_id[col] = traces

    # Render using unified display module
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

    # Add threshold and median lines (only works well for facet mode)
    if mode == "facet":
        for i, col in enumerate(columns):
            data = plot_data[col]
            color = colors.get(col, PALETTE[i % len(PALETTE)])

            # Median line
            if show_median:
                median_val = data.median()
                fig.add_vline(
                    x=median_val,
                    line_dash="dash",
                    line_color=color,
                    line_width=1.5,
                    row=(i // wrap) + 1,
                    col=(i % wrap) + 1,
                    annotation_text=f"Median: {median_val:.2f}",
                    annotation_position="top",
                    annotation_font_size=10,
                    annotation_font_color=color,
                )

            # Mean line
            if show_mean:
                mean_val = data.mean()
                fig.add_vline(
                    x=mean_val,
                    line_dash="dot",
                    line_color="gray",
                    line_width=1.5,
                    row=(i // wrap) + 1,
                    col=(i % wrap) + 1,
                )

            # Threshold line with textbox annotation
            if col in thresholds:
                thresh = thresholds[col]
                label = threshold_labels.get(col, "")
                pct_above = (data > thresh).mean() * 100
                row_idx = (i // wrap) + 1
                col_idx = (i % wrap) + 1

                # Check if this metric should show inverted percentage
                metric_config = METRIC_CONFIG.get(col, {})
                if metric_config.get("invert_pct", False):
                    pct_display = 100 - pct_above
                    label_display = metric_config.get("label_inverted", label)
                else:
                    pct_display = pct_above
                    label_display = label

                fig.add_vline(
                    x=thresh,
                    line_dash="solid",
                    line_color=threshold_color,
                    line_width=2,
                    row=row_idx,
                    col=col_idx,
                )

                # Build annotation text - simple format for threshold line
                if show_threshold_pct:
                    # Simplified: just "X% label" next to the line
                    ann_text = (
                        f"<b>{pct_display:.0f}%</b> {label_display}"
                        if label_display
                        else f"<b>{pct_display:.0f}%</b>"
                    )
                else:
                    ann_text = f"<b>{label}</b>" if label else ""

                if ann_text:
                    # Calculate axis references
                    axis_idx = (row_idx - 1) * wrap + col_idx
                    xaxis_ref = f"x{axis_idx}" if axis_idx > 1 else "x"
                    yaxis_ref = f"y{axis_idx}" if axis_idx > 1 else "y"

                    fig.add_annotation(
                        x=thresh,
                        y=0.85,
                        xref=xaxis_ref,
                        yref=f"{yaxis_ref} domain",
                        text=ann_text,
                        showarrow=False,
                        font=dict(color=threshold_color, size=10),
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor=threshold_color,
                        borderwidth=1,
                        borderpad=3,
                    )

    # Add threshold legend box above each subplot if requested
    if show_threshold_legend and mode == "facet" and thresholds:
        for i, col in enumerate(columns):
            if col in thresholds:
                thresh = thresholds[col]
                label = threshold_labels.get(col, col.replace("_", " ").title())
                row_idx = (i // wrap) + 1
                col_idx = (i % wrap) + 1
                # Calculate axis reference for this subplot
                axis_idx = (row_idx - 1) * wrap + col_idx
                xaxis_ref = f"x{axis_idx}" if axis_idx > 1 else "x"
                yaxis_ref = f"y{axis_idx}" if axis_idx > 1 else "y"

                # Use inverted label for chaos metrics
                metric_config = METRIC_CONFIG.get(col, {})
                if metric_config.get("invert_pct", False):
                    display_label = metric_config.get("label_inverted", label)
                    symbol = "≤"
                else:
                    display_label = label
                    symbol = "≥"

                fig.add_annotation(
                    x=0.5,
                    y=1.15,
                    xref=f"{xaxis_ref} domain",
                    yref=f"{yaxis_ref} domain",
                    xanchor="center",
                    yanchor="bottom",
                    text=f"<b>{display_label}</b> {symbol} <span style='color:{threshold_color}'><b>{thresh}</b></span>",
                    showarrow=False,
                    font=dict(size=10, color="#333"),
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#ddd",
                    borderwidth=1,
                    borderpad=4,
                )
        fig.update_layout(margin=dict(t=100))

    # For overlay mode, add lines to single plot
    elif mode == "overlay":
        # Add threshold lines for each metric (they'll stack on same plot)
        for i, col in enumerate(columns):
            data = plot_data[col]
            color = colors.get(col, PALETTE[i % len(PALETTE)])

            if col in thresholds:
                thresh = thresholds[col]
                fig.add_vline(
                    x=thresh,
                    line_dash="solid",
                    line_color=color,
                    line_width=2,
                    annotation_text=col if show_threshold_pct else None,
                    annotation_position="top",
                )

    return fig


def _plot_metric_distribution_facet_dropdown(
    df: pd.DataFrame,
    columns: List[str],
    group_col: Union[str, List[str]],
    facet_by: Optional[Literal["columns", "group"]],
    dropdown_by: Optional[Literal["columns", "group"]],
    kind: str,
    bins: int,
    thresholds: Dict[str, float],
    threshold_labels: Dict[str, str],
    threshold_color: str,
    show_threshold_pct: bool,
    show_threshold_legend: bool,
    show_kde: bool,
    show_median: bool,
    clip_quantile: Dict[str, float],
    wrap: int,
    row_height: int,
    col_width: int,
    vertical_spacing: float,
    colors: Dict[str, str],
    theme: str,
    style: Optional[dict],
) -> go.Figure:
    """
    Plot distribution with flexible facet and/or dropdown control.

    Supports any combination of:
    - facet_by="columns" or "group" (or None for single plot)
    - dropdown_by="columns" or "group" (or None for no dropdown)
    """
    from math import ceil

    from .._layout import finalize_figure

    # Handle list of group columns
    effective_group_col = group_col
    if isinstance(group_col, list):
        df = df.copy()
        df["_group"] = df[group_col].astype(str).agg(" | ".join, axis=1)
        effective_group_col = "_group"

    groups = sorted(df[effective_group_col].dropna().unique())

    # Determine facet and dropdown items based on settings
    if facet_by == "columns":
        facet_items = columns
        get_facet_title = lambda item: item.replace("_", " ").title()
    elif facet_by == "group":
        facet_items = groups
        get_facet_title = lambda item: str(item)
    else:
        facet_items = [None]  # Single plot, no faceting
        get_facet_title = lambda item: ""

    if dropdown_by == "columns":
        dropdown_items = columns
        get_dropdown_label = lambda item: item.replace("_", " ").title()
    elif dropdown_by == "group":
        dropdown_items = groups
        get_dropdown_label = lambda item: str(item)
    else:
        dropdown_items = [None]  # No dropdown
        get_dropdown_label = lambda item: ""

    n_facets = len(facet_items) if facet_items[0] is not None else 1
    cols = min(wrap, n_facets)
    rows = ceil(n_facets / cols)

    # Create subplots
    subplot_titles = (
        [get_facet_title(item) for item in facet_items] if facet_items[0] is not None else None
    )
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=vertical_spacing,
    )

    # Track trace indices for each dropdown item
    trace_indices_by_dropdown = {item: [] for item in dropdown_items}

    # Add traces
    for dropdown_idx, dropdown_item in enumerate(dropdown_items):
        visible = dropdown_idx == 0  # First dropdown item visible by default

        for facet_idx, facet_item in enumerate(facet_items):
            row = (facet_idx // cols) + 1 if facet_items[0] is not None else 1
            col_pos = (facet_idx % cols) + 1 if facet_items[0] is not None else 1

            # Determine which column and group to use
            if facet_by == "columns" and dropdown_by == "group":
                col_name = facet_item
                group_val = dropdown_item
            elif facet_by == "group" and dropdown_by == "columns":
                col_name = dropdown_item
                group_val = facet_item
            elif facet_by == "columns" and dropdown_by is None:
                col_name = facet_item
                group_val = None  # Use all data
            elif facet_by == "group" and dropdown_by is None:
                col_name = columns[0] if len(columns) == 1 else None
                group_val = facet_item
            elif facet_by is None and dropdown_by == "columns":
                col_name = dropdown_item
                group_val = None  # Use all data
            elif facet_by is None and dropdown_by == "group":
                col_name = columns[0] if len(columns) == 1 else None
                group_val = dropdown_item
            else:
                continue  # Invalid combination

            if col_name is None:
                continue

            # Filter data
            if group_val is not None:
                subset_df = df[df[effective_group_col] == group_val]
            else:
                subset_df = df

            data = subset_df[col_name].dropna()

            # Apply clipping
            if col_name in clip_quantile and clip_quantile[col_name]:
                if len(data) > 0:
                    cap = data.quantile(clip_quantile[col_name])
                    data = data.clip(upper=cap)

            if len(data) == 0:
                continue

            # Determine color
            if facet_by == "columns":
                color = colors.get(col_name, PALETTE[facet_idx % len(PALETTE)])
            elif dropdown_by == "columns":
                color = colors.get(col_name, PALETTE[dropdown_idx % len(PALETTE)])
            else:
                color = PALETTE[facet_idx % len(PALETTE)]

            # Add histogram
            trace = go.Histogram(
                x=data,
                nbinsx=bins,
                marker=dict(color=color, opacity=0.7),
                name=col_name,
                showlegend=False,
                visible=visible if dropdown_items[0] is not None else True,
            )
            fig.add_trace(trace, row=row, col=col_pos)
            if dropdown_item is not None:
                trace_indices_by_dropdown[dropdown_item].append(len(fig.data) - 1)

            # Add KDE if requested
            if show_kde and len(data) > 1:
                hist, bin_edges = np.histogram(data, bins=bins, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                hist_counts, _ = np.histogram(data, bins=bins)
                scale = hist_counts.max() / (hist.max() + 1e-10)

                kde_trace = go.Scatter(
                    x=bin_centers,
                    y=hist * scale,
                    mode="lines",
                    line=dict(color=color, width=2),
                    showlegend=False,
                    visible=visible if dropdown_items[0] is not None else True,
                )
                fig.add_trace(kde_trace, row=row, col=col_pos)
                if dropdown_item is not None:
                    trace_indices_by_dropdown[dropdown_item].append(len(fig.data) - 1)

    # Build dropdown buttons (only if we have dropdown)
    layout_updates = {
        "height": row_height * rows,
        "margin": dict(t=80 if dropdown_items[0] is not None else 50),
    }
    if col_width:
        layout_updates["width"] = col_width * cols

    if dropdown_items[0] is not None:
        buttons = []
        n_traces = len(fig.data)

        for dropdown_item in dropdown_items:
            visibility = [False] * n_traces
            for idx in trace_indices_by_dropdown[dropdown_item]:
                visibility[idx] = True

            buttons.append(
                dict(
                    label=get_dropdown_label(dropdown_item),
                    method="update",
                    args=[{"visible": visibility}],
                )
            )

        layout_updates["updatemenus"] = [
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "y": 1.15,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "white",
                "bordercolor": "#ccc",
                "font": {"size": 12},
            }
        ]
        layout_updates["margin"] = dict(t=100)

    fig.update_layout(**layout_updates)

    # Add threshold lines and annotations (for faceted columns)
    if facet_by == "columns":
        # Compute threshold annotations per dropdown selection
        dropdown_annotations = {item: [] for item in dropdown_items}

        for facet_idx, col in enumerate(columns):
            if col in thresholds:
                thresh = thresholds[col]
                label = threshold_labels.get(col, "")
                row_idx = (facet_idx // cols) + 1
                col_pos = (facet_idx % cols) + 1

                # Check if this metric should show inverted percentage
                metric_config = METRIC_CONFIG.get(col, {})
                should_invert = metric_config.get("invert_pct", False)
                label_inverted = metric_config.get("label_inverted", label)

                # Add vline (always visible)
                fig.add_vline(
                    x=thresh,
                    line_dash="solid",
                    line_color=threshold_color,
                    line_width=2,
                    row=row_idx,
                    col=col_pos,
                )

                # Calculate axis references
                axis_idx = (row_idx - 1) * cols + col_pos
                xaxis_ref = f"x{axis_idx}" if axis_idx > 1 else "x"
                yaxis_ref = f"y{axis_idx}" if axis_idx > 1 else "y"

                # Build annotation for each dropdown item (group)
                for dropdown_item in dropdown_items:
                    if dropdown_by == "group" and dropdown_item is not None:
                        subset_df = df[df[effective_group_col] == dropdown_item]
                    else:
                        subset_df = df

                    data = subset_df[col].dropna()
                    pct_above = (data > thresh).mean() * 100 if len(data) > 0 else 0

                    # Apply inversion for chaos metrics
                    if should_invert:
                        pct_display = 100 - pct_above
                        label_display = label_inverted
                    else:
                        pct_display = pct_above
                        label_display = label

                    # Simplified annotation: just "X% label"
                    if show_threshold_pct:
                        ann_text = (
                            f"<b>{pct_display:.0f}%</b> {label_display}"
                            if label_display
                            else f"<b>{pct_display:.0f}%</b>"
                        )
                    else:
                        ann_text = f"<b>{label}</b>" if label else ""

                    if ann_text:
                        dropdown_annotations[dropdown_item].append(
                            dict(
                                x=thresh,
                                y=0.85,
                                xref=xaxis_ref,
                                yref=f"{yaxis_ref} domain",
                                text=ann_text,
                                showarrow=False,
                                font=dict(color=threshold_color, size=10),
                                bgcolor="rgba(255,255,255,0.9)",
                                bordercolor=threshold_color,
                                borderwidth=1,
                                borderpad=3,
                            )
                        )

        # Preserve existing subplot title annotations
        subplot_title_annotations = list(fig.layout.annotations) if fig.layout.annotations else []

        # Update dropdown buttons to include annotations
        if dropdown_items[0] is not None and "updatemenus" in layout_updates:
            updated_buttons = []
            for dropdown_item in dropdown_items:
                visibility = [False] * n_traces
                for idx in trace_indices_by_dropdown[dropdown_item]:
                    visibility[idx] = True

                combined_annotations = (
                    subplot_title_annotations + dropdown_annotations[dropdown_item]
                )

                updated_buttons.append(
                    dict(
                        label=get_dropdown_label(dropdown_item),
                        method="update",
                        args=[{"visible": visibility}, {"annotations": combined_annotations}],
                    )
                )

            layout_updates["updatemenus"][0]["buttons"] = updated_buttons

            # Set initial annotations
            first_dropdown = dropdown_items[0]
            initial_annotations = subplot_title_annotations + dropdown_annotations[first_dropdown]
            fig.update_layout(annotations=initial_annotations)
        else:
            # No dropdown - just add all annotations
            all_annotations = []
            for anns in dropdown_annotations.values():
                all_annotations.extend(anns)
            fig.update_layout(annotations=subplot_title_annotations + all_annotations)

    fig.update_layout(**layout_updates)

    # Add threshold lines and annotations when faceting by group (single metric, multiple groups)
    if facet_by == "group" and len(columns) == 1 and thresholds:
        col = columns[0]
        if col in thresholds:
            thresh = thresholds[col]
            label = threshold_labels.get(col, "")

            # Check if this metric should show inverted percentage
            metric_config = METRIC_CONFIG.get(col, {})
            should_invert = metric_config.get("invert_pct", False)
            label_inverted = metric_config.get("label_inverted", label)

            for facet_idx, group_val in enumerate(groups):
                row_idx = (facet_idx // cols) + 1
                col_pos = (facet_idx % cols) + 1

                # Add vline
                fig.add_vline(
                    x=thresh,
                    line_dash="solid",
                    line_color=threshold_color,
                    line_width=2,
                    row=row_idx,
                    col=col_pos,
                )

                if show_threshold_pct:
                    # Calculate percentage for this group
                    group_data = df[df[effective_group_col] == group_val][col].dropna()
                    pct_above = (group_data > thresh).mean() * 100 if len(group_data) > 0 else 0

                    # Apply inversion for chaos metrics
                    if should_invert:
                        pct_display = 100 - pct_above
                        label_display = label_inverted
                    else:
                        pct_display = pct_above
                        label_display = label

                    ann_text = (
                        f"<b>{pct_display:.0f}%</b> {label_display}"
                        if label_display
                        else f"<b>{pct_display:.0f}%</b>"
                    )

                    # Calculate axis references
                    axis_idx = (row_idx - 1) * cols + col_pos
                    xaxis_ref = f"x{axis_idx}" if axis_idx > 1 else "x"
                    yaxis_ref = f"y{axis_idx}" if axis_idx > 1 else "y"

                    fig.add_annotation(
                        x=thresh,
                        y=0.85,
                        xref=xaxis_ref,
                        yref=f"{yaxis_ref} domain",
                        text=ann_text,
                        showarrow=False,
                        font=dict(color=threshold_color, size=10),
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor=threshold_color,
                        borderwidth=1,
                        borderpad=3,
                    )

    # Add threshold legend box above each subplot if requested
    if show_threshold_legend and facet_by == "columns" and thresholds:
        for i, col in enumerate(columns):
            if col in thresholds:
                thresh = thresholds[col]
                label = threshold_labels.get(col, col.replace("_", " ").title())
                row_idx = (i // wrap) + 1
                col_idx = (i % wrap) + 1
                # Calculate axis reference for this subplot
                axis_idx = (row_idx - 1) * wrap + col_idx
                xaxis_ref = f"x{axis_idx}" if axis_idx > 1 else "x"
                yaxis_ref = f"y{axis_idx}" if axis_idx > 1 else "y"

                # Use inverted label for chaos metrics
                metric_config = METRIC_CONFIG.get(col, {})
                if metric_config.get("invert_pct", False):
                    display_label = metric_config.get("label_inverted", label)
                    symbol = "≤"
                else:
                    display_label = label
                    symbol = "≥"

                fig.add_annotation(
                    x=0.5,
                    y=1.15,
                    xref=f"{xaxis_ref} domain",
                    yref=f"{yaxis_ref} domain",
                    xanchor="center",
                    yanchor="bottom",
                    text=f"<b>{display_label}</b> {symbol} <span style='color:{threshold_color}'><b>{thresh}</b></span>",
                    showarrow=False,
                    font=dict(size=10, color="#333"),
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#ddd",
                    borderwidth=1,
                    borderpad=4,
                )
        fig.update_layout(margin=dict(t=100))

    return finalize_figure(fig, theme=theme, style=style)


def _plot_metric_distribution_grouped(
    df: pd.DataFrame,
    columns: List[str],
    color_col: Union[str, List[str]],
    kind: str,
    mode: str,
    bins: int,
    thresholds: Dict[str, float],
    threshold_labels: Dict[str, str],
    threshold_color: str,
    show_threshold_pct: bool,
    show_threshold_legend: bool,
    clip_quantile: Dict[str, float],
    wrap: int,
    row_height: int,
    col_width: int,
    vertical_spacing: float,
    theme: str,
    style: Optional[dict],
) -> go.Figure:
    """Plot distribution of metrics grouped by color_col (overlaid histograms per group)."""

    # Handle list of color columns by creating a combined column
    effective_color_col = color_col
    if isinstance(color_col, list):
        df = df.copy()
        df["_color_group"] = df[color_col].astype(str).agg(" | ".join, axis=1)
        effective_color_col = "_color_group"

    groups = sorted(df[effective_color_col].dropna().unique())

    # Build traces_by_id: each metric gets a list of traces (one per group)
    traces_by_id: Dict[str, List[go.BaseTraceType]] = {}

    for i, col in enumerate(columns):
        traces = []

        for j, group in enumerate(groups):
            group_data = df[df[effective_color_col] == group][col].dropna()

            # Apply clipping if configured
            if col in clip_quantile and clip_quantile[col]:
                cap = group_data.quantile(clip_quantile[col])
                group_data = group_data.clip(upper=cap)

            if len(group_data) == 0:
                continue

            color = PALETTE[j % len(PALETTE)]

            traces.append(
                go.Histogram(
                    x=group_data,
                    nbinsx=bins,
                    marker=dict(color=color, opacity=0.6),
                    name=str(group),
                    legendgroup=str(group),
                )
            )

        traces_by_id[col] = traces

    # Render using unified display module
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

    # Set barmode for overlaid histograms
    fig.update_layout(barmode="overlay")

    # Add threshold lines and annotations for facet mode
    if mode == "facet":
        for i, col in enumerate(columns):
            if col in thresholds:
                thresh = thresholds[col]
                label = threshold_labels.get(col, "")
                row_idx = (i // wrap) + 1
                col_idx = (i % wrap) + 1

                # Check if this metric should show inverted percentage
                metric_config = METRIC_CONFIG.get(col, {})
                should_invert = metric_config.get("invert_pct", False)
                label_inverted = metric_config.get("label_inverted", label)

                fig.add_vline(
                    x=thresh,
                    line_dash="solid",
                    line_color=threshold_color,
                    line_width=2,
                    row=row_idx,
                    col=col_idx,
                )

                # Calculate percentage above threshold (across all groups)
                data = df[col].dropna()
                pct_above = (data > thresh).mean() * 100 if len(data) > 0 else 0

                # Apply inversion for chaos metrics
                if should_invert:
                    pct_display = 100 - pct_above
                    label_display = label_inverted
                else:
                    pct_display = pct_above
                    label_display = label

                # Simplified annotation: just "X% label"
                if show_threshold_pct:
                    ann_text = (
                        f"<b>{pct_display:.0f}%</b> {label_display}"
                        if label_display
                        else f"<b>{pct_display:.0f}%</b>"
                    )
                else:
                    ann_text = f"<b>{label}</b>" if label else ""

                if ann_text:
                    # Calculate axis references
                    axis_idx = (row_idx - 1) * wrap + col_idx
                    xaxis_ref = f"x{axis_idx}" if axis_idx > 1 else "x"
                    yaxis_ref = f"y{axis_idx}" if axis_idx > 1 else "y"

                    fig.add_annotation(
                        x=thresh,
                        y=0.85,
                        xref=xaxis_ref,
                        yref=f"{yaxis_ref} domain",
                        text=ann_text,
                        showarrow=False,
                        font=dict(color=threshold_color, size=10),
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor=threshold_color,
                        borderwidth=1,
                        borderpad=3,
                    )

    # Add threshold legend box above each subplot if requested
    if show_threshold_legend and mode == "facet" and thresholds:
        for i, col in enumerate(columns):
            if col in thresholds:
                thresh = thresholds[col]
                label = threshold_labels.get(col, col.replace("_", " ").title())
                row_idx = (i // wrap) + 1
                col_idx = (i % wrap) + 1
                # Calculate axis reference for this subplot
                axis_idx = (row_idx - 1) * wrap + col_idx
                xaxis_ref = f"x{axis_idx}" if axis_idx > 1 else "x"
                yaxis_ref = f"y{axis_idx}" if axis_idx > 1 else "y"

                # Use inverted label for chaos metrics
                metric_config = METRIC_CONFIG.get(col, {})
                if metric_config.get("invert_pct", False):
                    display_label = metric_config.get("label_inverted", label)
                    symbol = "≤"
                else:
                    display_label = label
                    symbol = "≥"

                fig.add_annotation(
                    x=0.5,
                    y=1.15,
                    xref=f"{xaxis_ref} domain",
                    yref=f"{yaxis_ref} domain",
                    xanchor="center",
                    yanchor="bottom",
                    text=f"<b>{display_label}</b> {symbol} <span style='color:{threshold_color}'><b>{thresh}</b></span>",
                    showarrow=False,
                    font=dict(size=10, color="#333"),
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#ddd",
                    borderwidth=1,
                    borderpad=4,
                )
        fig.update_layout(margin=dict(t=100))

    return fig


def _plot_timeseries_distribution(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    ids: Union[None, int, str, List[str]],
    max_ids: int,
    group_col: Union[str, List[str], None],
    agg: str,
    kind: str,
    mode: str,
    bins: int,
    log_scale: bool,
    exclude_zeros: bool,
    show_stats: bool,
    wrap: int,
    theme: str,
    style: Optional[dict],
) -> go.Figure:
    """Plot distribution of time series values."""

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df, id_col = aggregate_by_group(df, group_col, date_col, value_col, agg, id_col)
    ids_list = select_ids(df, id_col, ids, max_ids)
    df = df[df[id_col].isin(ids_list)].copy()

    if exclude_zeros:
        df = df[df[value_col] > 0].copy()

    plot_col = value_col
    if log_scale:
        df["_log_value"] = np.log1p(df[value_col].clip(lower=0))
        plot_col = "_log_value"

    # Compute stats for annotations
    stats = {}
    for uid in ids_list:
        sub = df[df[id_col] == uid][value_col]
        stats[uid] = {
            "mean": sub.mean(),
            "median": sub.median(),
            "std": sub.std(),
            "min": sub.min(),
            "max": sub.max(),
            "n": len(sub),
        }

    # Build traces for each series
    traces_by_id = {}
    for i, uid in enumerate(ids_list):
        sub = df[df[id_col] == uid]
        color = PALETTE[i % len(PALETTE)]

        if kind == "histogram":
            traces_by_id[uid] = [
                go.Histogram(
                    x=sub[plot_col],
                    name=str(uid),
                    opacity=0.7,
                    nbinsx=bins,
                    marker=dict(color=color),
                )
            ]
        elif kind == "density":
            data = sub[plot_col].dropna()
            if len(data) >= 2:
                hist, bin_edges = np.histogram(data, bins=bins, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                traces_by_id[uid] = [
                    go.Scatter(
                        x=bin_centers,
                        y=hist,
                        mode="lines",
                        name=str(uid),
                        line=dict(color=color, width=2),
                        fill="tozeroy",
                        opacity=0.6,
                    )
                ]
            else:
                traces_by_id[uid] = []
        elif kind == "box":
            traces_by_id[uid] = [
                go.Box(
                    y=sub[plot_col],
                    name=str(uid),
                    marker=dict(color=color),
                    boxpoints="outliers",
                )
            ]
        elif kind == "violin":
            traces_by_id[uid] = [
                go.Violin(
                    y=sub[plot_col],
                    name=str(uid),
                    line=dict(color=color),
                    box_visible=True,
                    meanline_visible=True,
                )
            ]
        else:
            raise ValueError("kind must be one of: histogram, density, box, violin")

    # Render using unified display module
    fig = render_by_mode(
        traces_by_id,
        mode=mode,
        wrap=wrap,
        theme=theme,
        style=style,
    )

    # Update axis labels
    x_label = f"log({value_col})" if log_scale else value_col
    if kind in ["histogram", "density"]:
        fig.update_xaxes(title_text=x_label)
        fig.update_yaxes(title_text="Frequency" if kind == "histogram" else "Density")
    else:
        fig.update_yaxes(title_text=x_label)

    return fig
