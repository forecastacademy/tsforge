# tsforge/plots/core/scatter.py
"""Scatter plot visualization for tsforge.

Supports quadrant analysis, threshold lines, 2D histograms,
and overlay/facet/dropdown modes.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Literal, Dict, Tuple

import plotly.graph_objects as go

from .._styling import PALETTE, hex_to_rgba, apply_theme, resolve_color
from .._layout import finalize_figure
from .._display import render_by_mode
from .._preprocessing import select_ids, resolve_group_col, clip_by_quantile


# Default quadrant colors
QUADRANT_COLORS = {
    "top_right": "#e74c3c",     # Red - typically "danger"
    "top_left": "#f39c12",      # Orange
    "bottom_right": "#27ae60",  # Green - typically "good"
    "bottom_left": "#3498db",   # Blue
}

# Metric threshold defaults
METRIC_THRESHOLDS = {
    "trend": 0.5,
    "seasonal_strength": 0.5,
    "x_acf1": 0.6,
    "entropy": 0.8,
    "adi": 1.32,
    "cv2": 0.49,
    "lumpiness": 1.0,
}

# Default quadrant labels for common metric interactions
INTERACTION_LABELS = {
    ("trend", "adi"): {
        "top_right": "DANGER\nTrend unreliable",
        "top_left": "Intermittent\n(weak trend OK)",
        "bottom_right": "Real Trend\n(trust it)",
        "bottom_left": "Smooth\n(no trend)",
    },
    ("seasonal_strength", "adi"): {
        "top_right": "DANGER\nSeasonality unreliable",
        "top_left": "Intermittent\n(weak seasonal OK)",
        "bottom_right": "Real Seasonality\n(trust it)",
        "bottom_left": "Smooth\n(no seasonality)",
    },
    ("trend", "entropy"): {
        "top_right": "Noisy Trend\n(don't trust)",
        "top_left": "Pure Chaos\n(no pattern)",
        "bottom_right": "Clean Trend\n(trust it)",
        "bottom_left": "Stable\n(low chaos, low trend)",
    },
    ("seasonal_strength", "entropy"): {
        "top_right": "Noisy Seasonal\n(don't trust)",
        "top_left": "Pure Chaos\n(no pattern)",
        "bottom_right": "Clean Seasonal\n(trust it)",
        "bottom_left": "Stable\n(low chaos, weak seasonal)",
    },
    ("cv2", "adi"): {
        "top_right": "Lumpy",
        "top_left": "Intermittent",
        "bottom_right": "Erratic",
        "bottom_left": "Smooth",
    },
}


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    kind: Literal["scatter", "density"] = "scatter",
    # Grouping & selection (consistent with timeseries/distribution/bar)
    color_col: Optional[str] = None,
    size: Optional[str] = None,
    ids: Union[None, int, str, List[str]] = None,
    max_ids: Optional[int] = None,
    id_col: Optional[str] = None,
    group_col: Optional[Union[str, List[str]]] = None,
    agg: str = "mean",
    hover_data: Optional[List[str]] = None,
    # Thresholds and quadrants
    x_threshold: Optional[float] = None,
    y_threshold: Optional[float] = None,
    quadrant_labels: Optional[Dict[str, str]] = None,
    quadrant_colors: Optional[Dict[str, str]] = None,
    quadrant_label_style: Literal["badge", "watermark"] = "badge",
    show_quadrant_pcts: bool = True,
    show_threshold_annotation: bool = False,
    use_metric_defaults: bool = False,
    # Data processing
    clip_quantile: Optional[float] = None,
    # Axis range control
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    # Figure size
    height: Optional[int] = None,
    width: Optional[int] = None,
    # Styling
    marker_size: int = 8,
    opacity: float = 0.7,
    colors: Optional[Union[str, Dict[str, str]]] = None,
    nbins: int = 40,
    colorscale: str = "Blues",
    # Layout (consistent with timeseries/distribution/bar)
    mode: Literal["overlay", "facet", "dropdown"] = "overlay",
    wrap: int = 2,
    row_height: int = 400,
    col_width: int = 500,
    vertical_spacing: float = 0.12,
    theme: str = "fa",
    style: Optional[dict] = None,
) -> go.Figure:
    """
    Scatter plot with optional quadrant analysis.

    Supports both scatter points and 2D density heatmaps for large datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x : str
        Column for x-axis.
    y : str
        Column for y-axis.
    kind : {"scatter", "density"}, default "scatter"
        Visualization type:
        - "scatter": Individual points (best for small datasets)
        - "density": 2D histogram heatmap (best for large datasets)
    color_col : str, optional
        Column to color points by (scatter only). Creates one trace per
        unique value, enabling per-group legend toggle.
    size : str, optional
        Column to size points by (scatter only).
    ids : None, int, str, or list, optional
        If color_col specified, which color values to include.
        If int, includes first N values.
    max_ids : int, optional
        Maximum color_col values to show if ids not specified.
        Default: no limit (show all unique values).
    id_col : str, optional
        Column containing point identifiers for hover.
    group_col : str or list, optional
        Column(s) to aggregate by before plotting. If set, data is grouped
        and x/y values are aggregated per group. Also controls
        facet/dropdown selection: in "dropdown" mode, each group value
        becomes a dropdown item; in "facet" mode, each gets its own panel.
        color_col traces are overlaid within each group panel.
    agg : str, default "mean"
        Aggregation function when group_col is set.
    hover_data : list of str, optional
        Additional columns to show in hover.
    x_threshold : float, optional
        Vertical threshold line. Auto-detected if use_metric_defaults=True.
    y_threshold : float, optional
        Horizontal threshold line. Auto-detected if use_metric_defaults=True.
    quadrant_labels : dict, optional
        Labels for quadrants. Keys: "top_left", "top_right", "bottom_left",
        "bottom_right". Auto-detected for known metric pairs if
        use_metric_defaults=True.
    quadrant_colors : dict, optional
        Colors for quadrant labels (badge style only).
    quadrant_label_style : {"badge", "watermark"}, default "badge"
        Label rendering style:
        - "badge": Small boxed labels with border and optional percentages.
          Best for diagnostic quadrant plots where labels teach what each zone means.
        - "watermark": Large faint text without box or percentages.
          Best for maps where data density tells the story and labels just orient.
    show_quadrant_pcts : bool, default True
        Show percentage of points in each quadrant (badge style only).
    show_threshold_annotation : bool, default False
        Show threshold values as annotation at bottom.
    use_metric_defaults : bool, default False
        Auto-detect thresholds and quadrant labels for known metrics.
    clip_quantile : float, optional
        Clip data to this quantile range (e.g., 0.95 clips to 5-95%).
    x_range : tuple of (float, float), optional
        Explicit (min, max) range for x-axis. Applied after clip_quantile.
    y_range : tuple of (float, float), optional
        Explicit (min, max) range for y-axis. Applied after clip_quantile.
    height : int, optional
        Override figure height in pixels.
    width : int, optional
        Override figure width in pixels.
    marker_size : int, default 8
        Base marker size (scatter only).
    opacity : float, default 0.7
        Marker opacity.
    colors : str or dict, optional
        Color(s) for points. Dict maps color_col values to hex colors.
        String applies a single color to all points.
    nbins : int, default 40
        Number of bins for density plot.
    colorscale : str, default "Blues"
        Colorscale for density plot.
    mode : {"overlay", "facet", "dropdown"}, default "overlay"
        Display mode when color_col creates multiple series.
    wrap : int, default 2
        Columns for facet mode.
    row_height : int, default 400
        Pixel height per facet row.
    col_width : int, default 500
        Pixel width per facet column.
    vertical_spacing : float, default 0.12
        Spacing between facet rows (0-1 fraction).
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
    >>> # Basic scatter plot
    >>> plot_scatter(df, 'trend', 'entropy')

    >>> # Colored by category
    >>> plot_scatter(df, 'chaos_score', 'structure_score',
    ...             color_col='forecastability',
    ...             colors={'Stable': '#2ecc71', 'Messy': '#e74c3c'})

    >>> # Density plot with auto-detected quadrants
    >>> plot_scatter(df, 'trend', 'adi', kind='density',
    ...             use_metric_defaults=True)

    >>> # Forecastability map with watermark labels
    >>> plot_scatter(df, 'chaos_score', 'structure_score',
    ...             color_col='forecastability',
    ...             x_threshold=0.5, y_threshold=0.5,
    ...             quadrant_labels={'top_left': 'STABLE', ...},
    ...             quadrant_label_style='watermark',
    ...             x_range=(0, 1), y_range=(0, 1))

    >>> # Aggregated scatter (mean metric per group)
    >>> plot_scatter(df, 'trend', 'entropy',
    ...             group_col='department', agg='mean')
    """
    df = df.copy()

    # Handle group_col aggregation
    # Only aggregate when group_col is used for data reduction (overlay mode).
    # In dropdown/facet mode, group_col slices data into panels — no aggregation.
    if group_col is not None:
        if isinstance(group_col, str):
            group_keys = [group_col]
        else:
            group_keys = list(group_col)

        if mode == "overlay":
            agg_cols = [x, y]
            if color_col and color_col not in group_keys:
                group_keys_with_color = group_keys + [color_col]
                df = df.groupby(group_keys_with_color, observed=True)[agg_cols].agg(agg).reset_index()
            else:
                df = df.groupby(group_keys, observed=True)[agg_cols].agg(agg).reset_index()

        # Use first group_col as id_col for hover if not already set
        if id_col is None:
            id_col = group_keys[0]

    # Auto-detect thresholds and labels if use_metric_defaults
    if use_metric_defaults:
        if x_threshold is None and x in METRIC_THRESHOLDS:
            x_threshold = METRIC_THRESHOLDS[x]
        if y_threshold is None and y in METRIC_THRESHOLDS:
            y_threshold = METRIC_THRESHOLDS[y]
        if quadrant_labels is None:
            quadrant_labels = INTERACTION_LABELS.get((x, y), _get_generic_quadrant_labels(x, y))

    # Clip data if requested
    if clip_quantile is not None:
        df = clip_by_quantile(df, [x, y], clip_quantile)

    # Build figure based on kind
    if kind == "density":
        fig = go.Figure()
        fig.add_trace(go.Histogram2d(
            x=df[x],
            y=df[y],
            nbinsx=nbins,
            nbinsy=nbins,
            colorscale=colorscale,
            colorbar=dict(title="Count", len=0.8),
        ))
        fig = apply_theme(fig, theme)
    else:
        # Scatter plot
        hover_cols = [x, y]
        if id_col:
            hover_cols.insert(0, id_col)
        if hover_data:
            hover_cols.extend(hover_data)

        # Resolve color values and filter using select_ids
        color_vals = None
        if color_col is not None:
            all_color_vals = df[color_col].dropna().unique().tolist()
            color_vals = select_ids(
                df, color_col, ids, max_ids or len(all_color_vals),
            )

        def _color_traces(subset: pd.DataFrame) -> List[go.BaseTraceType]:
            """Build colored traces for a single panel (group or full data).

            Draws largest groups first (bottom layer) so smaller groups
            remain visible on top.
            """
            if color_vals is None:
                c = resolve_color(colors, "data", 0)
                return _build_scatter_traces(
                    subset, x, y, size, marker_size, opacity,
                    c, "data", hover_cols,
                )
            # Sort: largest group first (drawn on bottom) → smallest last (on top)
            sorted_vals = sorted(
                color_vals,
                key=lambda v: len(subset[subset[color_col] == v]),
                reverse=True,
            )
            # Maintain original color_vals index for consistent color assignment
            val_to_idx = {v: i for i, v in enumerate(color_vals)}
            traces = []
            for cval in sorted_vals:
                sub = subset[subset[color_col] == cval]
                if sub.empty:
                    continue
                i = val_to_idx[cval]
                c = resolve_color(colors, cval, i)
                traces.extend(_build_scatter_traces(
                    sub, x, y, size, marker_size, opacity,
                    c, str(cval), hover_cols,
                ))
            return traces

        # group_col drives dropdown/facet; color_col is overlay within each panel
        if group_col is not None and mode in ("dropdown", "facet"):
            df, effective_group_col = resolve_group_col(df, group_col)

            groups = sorted(df[effective_group_col].dropna().unique())
            traces_by_id = {}
            for gval in groups:
                group_df = df[df[effective_group_col] == gval]
                traces_by_id[str(gval)] = _color_traces(group_df)

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
        else:
            # No group_col or overlay mode: color_col keys traces_by_id
            if color_vals is None:
                c = resolve_color(colors, "data", 0)
                traces_by_id = {"data": _build_scatter_traces(
                    df, x, y, size, marker_size, opacity,
                    c, "data", hover_cols,
                )}
            else:
                # Sort: largest group first (bottom) → smallest last (top)
                sorted_vals = sorted(
                    color_vals,
                    key=lambda v: len(df[df[color_col] == v]),
                    reverse=True,
                )
                val_to_idx = {v: i for i, v in enumerate(color_vals)}
                traces_by_id = {}
                for cval in sorted_vals:
                    sub = df[df[color_col] == cval]
                    i = val_to_idx[cval]
                    c = resolve_color(colors, cval, i)
                    traces_by_id[str(cval)] = _build_scatter_traces(
                        sub, x, y, size, marker_size, opacity,
                        c, str(cval), hover_cols,
                    )

            if len(traces_by_id) == 1 and "data" in traces_by_id:
                fig = go.Figure(data=traces_by_id["data"])
                fig = apply_theme(fig, theme)
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

    # Add threshold lines (lighter for watermark style)
    if quadrant_label_style == "watermark":
        _thr_color, _thr_width, _thr_opacity = "gray", 1, 1.0
    else:
        _thr_color, _thr_width, _thr_opacity = "#2c3e50", 2.5, 0.8

    if x_threshold is not None:
        fig.add_vline(x=x_threshold, line_dash="dash", line_color=_thr_color, line_width=_thr_width, opacity=_thr_opacity)
    if y_threshold is not None:
        fig.add_hline(y=y_threshold, line_dash="dash", line_color=_thr_color, line_width=_thr_width, opacity=_thr_opacity)

    # Add quadrant labels
    if quadrant_labels and x_threshold is not None and y_threshold is not None:
        # Use range bounds for label positioning if specified, otherwise use data bounds
        x_min = x_range[0] if x_range else df[x].min()
        x_max = x_range[1] if x_range else df[x].max()
        y_min = y_range[0] if y_range else df[y].min()
        y_max = y_range[1] if y_range else df[y].max()

        _add_quadrant_labels(
            fig, df, x, y, x_threshold, y_threshold,
            quadrant_labels, quadrant_colors or QUADRANT_COLORS, show_quadrant_pcts,
            label_style=quadrant_label_style,
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        )

    # Add threshold annotation at bottom
    if show_threshold_annotation and x_threshold is not None and y_threshold is not None:
        fig.add_annotation(
            x=0.5, y=-0.12,
            xref="paper", yref="paper",
            text=f"<i>Thresholds: {x}={x_threshold}, {y}={y_threshold}</i>",
            showarrow=False,
            font=dict(size=10, color="#7f8c8d"),
        )
        fig.update_layout(margin=dict(b=80))

    # Apply explicit axis ranges if specified
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))

    # Apply figure size overrides
    size_updates = {}
    if height is not None:
        size_updates['height'] = height
    if width is not None:
        size_updates['width'] = width
    if size_updates:
        fig.update_layout(**size_updates)

    # Update axis labels
    fig.update_xaxes(title_text=x.replace("_", " ").title())
    fig.update_yaxes(title_text=y.replace("_", " ").title())

    fig = finalize_figure(fig, theme, style)

    # Alphabetical legend order (independent of draw order)
    trace_names = sorted(set(t.name for t in fig.data if t.name))
    name_to_rank = {name: i for i, name in enumerate(trace_names)}
    for trace in fig.data:
        if trace.name in name_to_rank:
            trace.legendrank = name_to_rank[trace.name]

    return fig


def _build_scatter_traces(
    df: pd.DataFrame,
    x: str,
    y: str,
    size: Optional[str],
    marker_size: int,
    opacity: float,
    color: str,
    name: str,
    hover_cols: List[str],
) -> List[go.BaseTraceType]:
    """Build scatter trace for a single series."""
    # Build hover template
    hover_parts = []
    for col in hover_cols:
        if col in df.columns:
            hover_parts.append(f"{col}: %{{customdata[{hover_cols.index(col)}]}}")
    hover_template = "<br>".join(hover_parts) + "<extra></extra>"

    # Size array
    if size and size in df.columns:
        sizes = df[size].fillna(marker_size)
        # Normalize sizes
        sizes = marker_size * (sizes / sizes.max()) * 3
    else:
        sizes = marker_size

    return [go.Scatter(
        x=df[x],
        y=df[y],
        mode="markers",
        name=name,
        marker=dict(color=color, size=sizes, opacity=opacity),
        customdata=df[hover_cols].values if hover_cols else None,
        hovertemplate=hover_template,
    )]


def _add_quadrant_labels(
    fig: go.Figure,
    df: pd.DataFrame,
    x: str,
    y: str,
    x_threshold: float,
    y_threshold: float,
    labels: Dict[str, str],
    colors: Dict[str, str],
    show_pcts: bool,
    *,
    label_style: Literal["badge", "watermark"] = "badge",
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> None:
    """Add quadrant labels with optional percentages.

    Parameters
    ----------
    label_style : {"badge", "watermark"}, default "badge"
        - "badge": Small boxed labels with border and optional percentages.
        - "watermark": Large faint text, no box or percentages.
    x_min, x_max, y_min, y_max : float, optional
        Explicit axis bounds for label positioning. If not provided,
        uses data min/max values.
    """
    n_total = len(df)
    if n_total == 0:
        return

    # Calculate quadrant percentages
    q_tl = ((df[x] < x_threshold) & (df[y] >= y_threshold)).sum() / n_total * 100
    q_tr = ((df[x] >= x_threshold) & (df[y] >= y_threshold)).sum() / n_total * 100
    q_bl = ((df[x] < x_threshold) & (df[y] < y_threshold)).sum() / n_total * 100
    q_br = ((df[x] >= x_threshold) & (df[y] < y_threshold)).sum() / n_total * 100

    pcts = {"top_left": q_tl, "top_right": q_tr, "bottom_left": q_bl, "bottom_right": q_br}

    # Use provided bounds or fall back to data bounds
    if x_min is None:
        x_min = df[x].min()
    if x_max is None:
        x_max = df[x].max()
    if y_min is None:
        y_min = df[y].min()
    if y_max is None:
        y_max = df[y].max()

    # Calculate label positions
    positions = {
        "top_left": ((x_min + x_threshold) / 2, (y_threshold + y_max) / 2),
        "top_right": ((x_threshold + x_max) / 2, (y_threshold + y_max) / 2),
        "bottom_left": ((x_min + x_threshold) / 2, (y_min + y_threshold) / 2),
        "bottom_right": ((x_threshold + x_max) / 2, (y_min + y_threshold) / 2),
    }

    for pos, label in labels.items():
        if pos not in positions:
            continue

        x_pos, y_pos = positions[pos]
        color = colors.get(pos, "#666")
        pct = pcts.get(pos, 0)

        if label_style == "watermark":
            # Text trace renders above data points, unlike annotations
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[y_pos],
                mode="text",
                text=[f"<b>{label}</b>"],
                textfont=dict(size=16, color="rgba(0,0,0,0.4)"),
                showlegend=False,
                hoverinfo="skip",
            ))
        else:
            # Badge style — boxed labels with border and percentages
            text = label
            if show_pcts:
                text += f"<br>({pct:.0f}%)"

            fig.add_annotation(
                x=x_pos,
                y=y_pos,
                text=f"<b>{text}</b>",
                showarrow=False,
                font=dict(size=11, color=color),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=color,
                borderwidth=2,
                borderpad=5,
            )


def plot_scatter_matrix(
    df: pd.DataFrame,
    columns: List[str],
    *,
    color_col: Optional[str] = None,
    marker_size: int = 4,
    opacity: float = 0.5,
    theme: str = "fa",
    style: Optional[dict] = None,
) -> go.Figure:
    """
    Scatter plot matrix (pairs plot) for multiple columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    columns : list of str
        Columns to include in the matrix.
    color_col : str, optional
        Column to color points by.
    marker_size : int, default 4
        Marker size.
    opacity : float, default 0.5
        Marker opacity.
    theme : str, default "fa"
        Theme name.
    style : dict, optional
        Style overrides.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    dimensions = [dict(label=col, values=df[col]) for col in columns]

    if color_col and color_col in df.columns:
        color_values = df[color_col]
    else:
        color_values = None

    fig = go.Figure(data=go.Splom(
        dimensions=dimensions,
        marker=dict(
            color=color_values,
            size=marker_size,
            opacity=opacity,
            colorscale="Viridis" if color_values is not None else None,
            showscale=color_values is not None,
        ),
        diagonal_visible=False,
        showupperhalf=False,
    ))

    fig.update_layout(height=200 * len(columns), width=200 * len(columns))
    return finalize_figure(fig, theme, style)


def _get_generic_quadrant_labels(x: str, y: str) -> Dict[str, str]:
    """Generate generic quadrant labels for unknown metric pairs."""
    x_name = x.replace("_", " ").title()
    y_name = y.replace("_", " ").title()
    return {
        "top_right": f"High {x_name}\nHigh {y_name}",
        "top_left": f"Low {x_name}\nHigh {y_name}",
        "bottom_right": f"High {x_name}\nLow {y_name}",
        "bottom_left": f"Low {x_name}\nLow {y_name}",
    }


# Backward compatibility alias
def plot_2d_histogram(
    df: pd.DataFrame,
    x: str,
    y: str,
    **kwargs,
) -> go.Figure:
    """
    2D histogram (density heatmap). Alias for plot_scatter(..., kind='density').

    See plot_scatter for full documentation.
    """
    return plot_scatter(df, x, y, kind="density", **kwargs)


# Convenience function for demand classification
def plot_demand_classification(
    df: pd.DataFrame,
    cv2_col: str = "cv2",
    adi_col: str = "adi",
    *,
    cv2_threshold: float = 0.49,
    adi_threshold: float = 1.32,
    clip_quantile: float = 0.95,
    kind: Literal["scatter", "histogram"] = "histogram",
    theme: str = "fa",
    style: Optional[dict] = None,
) -> go.Figure:
    """
    Syntetos-Boylan demand classification plot (CV² × ADI).

    Parameters
    ----------
    df : pd.DataFrame
        Input data with cv2 and adi columns.
    cv2_col : str, default "cv2"
        Column name for CV² values.
    adi_col : str, default "adi"
        Column name for ADI values.
    cv2_threshold : float, default 0.49
        CV² threshold for classification.
    adi_threshold : float, default 1.32
        ADI threshold for classification.
    clip_quantile : float, default 0.95
        Clip data to this quantile range.
    kind : {"scatter", "histogram"}, default "histogram"
        Plot type.
    theme : str, default "fa"
        Theme name.
    style : dict, optional
        Style overrides.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    quadrant_labels = {
        "bottom_left": "Smooth",
        "bottom_right": "Erratic",
        "top_left": "Intermittent",
        "top_right": "Lumpy",
    }

    quadrant_colors = {
        "bottom_left": "#27ae60",  # Green
        "bottom_right": "#f39c12",  # Orange
        "top_left": "#3498db",      # Blue
        "top_right": "#e74c3c",     # Red
    }

    if kind == "histogram":
        fig = plot_2d_histogram(
            df, x=cv2_col, y=adi_col,
            x_threshold=cv2_threshold, y_threshold=adi_threshold,
            quadrant_labels=quadrant_labels, quadrant_colors=quadrant_colors,
            clip_quantile=clip_quantile, theme=theme, style=style,
        )
    else:
        fig = plot_scatter(
            df, x=cv2_col, y=adi_col,
            x_threshold=cv2_threshold, y_threshold=adi_threshold,
            quadrant_labels=quadrant_labels, quadrant_colors=quadrant_colors,
            clip_quantile=clip_quantile, theme=theme, style=style,
        )

    fig.update_xaxes(title_text="CV² (Demand Variability)")
    fig.update_yaxes(title_text="ADI (Average Demand Interval)")

    return fig