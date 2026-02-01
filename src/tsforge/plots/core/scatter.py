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

from .._styling import PALETTE, hex_to_rgba, apply_theme
from .._layout import finalize_figure
from .._display import render_by_mode


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
    color: Optional[str] = None,
    size: Optional[str] = None,
    ids: Union[None, int, str, List[str]] = None,
    id_col: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
    # Thresholds and quadrants
    x_threshold: Optional[float] = None,
    y_threshold: Optional[float] = None,
    quadrant_labels: Optional[Dict[str, str]] = None,
    quadrant_colors: Optional[Dict[str, str]] = None,
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
    color_map: Optional[Dict[str, str]] = None,
    nbins: int = 40,
    colorscale: str = "Blues",
    # Mode
    mode: Literal["overlay", "facet", "dropdown"] = "overlay",
    wrap: int = 2,
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
    color : str, optional
        Column to color points by (scatter only).
    size : str, optional
        Column to size points by (scatter only).
    ids : None, int, str, or list, optional
        If color specified, which color values to include.
    id_col : str, optional
        Column containing point identifiers for hover.
    hover_data : list of str, optional
        Additional columns to show in hover.
    x_threshold : float, optional
        Vertical threshold line. Auto-detected if use_metric_defaults=True.
    y_threshold : float, optional
        Horizontal threshold line. Auto-detected if use_metric_defaults=True.
    quadrant_labels : dict, optional
        Labels for quadrants. Auto-detected for known metric pairs if use_metric_defaults=True.
    quadrant_colors : dict, optional
        Colors for quadrant labels.
    show_quadrant_pcts : bool, default True
        Show percentage of points in each quadrant.
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
        Figure height in pixels. Default is ~450 for standard plots.
    width : int, optional
        Figure width in pixels. Default is ~700 for standard plots.
    marker_size : int, default 8
        Base marker size (scatter only).
    opacity : float, default 0.7
        Marker opacity.
    color_map : dict, optional
        Mapping of color column values to colors (scatter only).
    nbins : int, default 40
        Number of bins for density plot.
    colorscale : str, default "Blues"
        Colorscale for density plot.
    mode : {"overlay", "facet", "dropdown"}, default "overlay"
        Display mode when color column creates multiple series.
    wrap : int, default 2
        Columns for facet mode.
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
    >>> plot_scatter(df, x='trend', y='entropy')

    >>> # Density plot with auto-detected quadrants
    >>> plot_scatter(
    ...     df, x='trend', y='adi',
    ...     kind='density',
    ...     use_metric_defaults=True,
    ... )

    >>> # Density plot with constrained y-axis for better visibility
    >>> plot_scatter(
    ...     df, x='trend', y='adi',
    ...     kind='density',
    ...     use_metric_defaults=True,
    ...     y_range=(0, 5),
    ... )

    >>> # Custom quadrant labels
    >>> plot_scatter(
    ...     df, x='cv2', y='adi',
    ...     kind='density',
    ...     x_threshold=0.49,
    ...     y_threshold=1.32,
    ...     quadrant_labels={
    ...         'top_right': 'Lumpy',
    ...         'top_left': 'Intermittent',
    ...         'bottom_right': 'Erratic',
    ...         'bottom_left': 'Smooth',
    ...     },
    ... )
    """
    df = df.copy()

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
        for col in [x, y]:
            lower = df[col].quantile(1 - clip_quantile)
            upper = df[col].quantile(clip_quantile)
            df = df[(df[col] >= lower) & (df[col] <= upper)]

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

        if color is None:
            traces_by_id = {"data": _build_scatter_traces(
                df, x, y, size, marker_size, opacity, PALETTE[0], "data", hover_cols
            )}
        else:
            unique_colors = df[color].dropna().unique().tolist()
            if ids is not None:
                if isinstance(ids, int):
                    unique_colors = unique_colors[:ids]
                elif isinstance(ids, str):
                    unique_colors = [ids]
                else:
                    unique_colors = [c for c in ids if c in unique_colors]

            traces_by_id = {}
            for i, cval in enumerate(unique_colors):
                sub = df[df[color] == cval]
                c = color_map.get(cval, PALETTE[i % len(PALETTE)]) if color_map else PALETTE[i % len(PALETTE)]
                traces_by_id[str(cval)] = _build_scatter_traces(
                    sub, x, y, size, marker_size, opacity, c, str(cval), hover_cols
                )

        if len(traces_by_id) == 1 and "data" in traces_by_id:
            fig = go.Figure(data=traces_by_id["data"])
            fig = apply_theme(fig, theme)
        else:
            fig = render_by_mode(
                traces_by_id,
                mode=mode,
                wrap=wrap,
                theme=theme,
                style=style,
            )

    # Add threshold lines
    if x_threshold is not None:
        fig.add_vline(x=x_threshold, line_dash="dash", line_color="#2c3e50", line_width=2.5, opacity=0.8)
    if y_threshold is not None:
        fig.add_hline(y=y_threshold, line_dash="dash", line_color="#2c3e50", line_width=2.5, opacity=0.8)

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

    # Apply figure size if specified
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

    return finalize_figure(fig, theme, style)


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
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> None:
    """Add quadrant labels with optional percentages.
    
    Parameters
    ----------
    x_min, x_max, y_min, y_max : float, optional
        Explicit axis bounds for label positioning. If not provided,
        uses data min/max values.
    """
    n_total = len(df)
    if n_total == 0:
        return

    # Calculate quadrant percentages (always based on full data, not clipped range)
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
        if pos in positions:
            x_pos, y_pos = positions[pos]
            color = colors.get(pos, "#666")
            pct = pcts.get(pos, 0)

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
    color: Optional[str] = None,
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
    color : str, optional
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

    if color and color in df.columns:
        color_values = df[color]
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