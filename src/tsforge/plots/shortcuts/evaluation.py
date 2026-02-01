# tsforge/plots/shortcuts/evaluation.py
"""Evaluation shortcut functions - thin wrappers around core plotting functions.

These provide convenient preset configurations for model evaluation tasks.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

from ..core.distribution import plot_distribution
from ..core.scatter import plot_scatter

# Default threshold configurations for common metrics
METRIC_THRESHOLDS = {
    "trend": {"threshold": 0.5, "label": "Strong Trend"},
    "seasonal_strength": {"threshold": 0.5, "label": "Strong Seasonality"},
    "entropy": {"threshold": 0.8, "label": "Chaotic"},
    "cv2": {"threshold": 0.49, "label": "High Variability"},
    "adi": {"threshold": 1.32, "label": "Intermittent"},
}


def plot_metric_interaction(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    x_threshold: Optional[float] = None,
    y_threshold: Optional[float] = None,
    quadrant_labels: Optional[Dict[str, str]] = None,
    show_threshold_annotation: bool = True,
    clip_quantile: float = 0.95,
    kind: str = "histogram",
    theme: str = "fa",
    style: Optional[dict] = None,
) -> go.Figure:
    """
    2D interaction plot between two metrics with quadrant analysis.

    This is a convenience wrapper around plot_scatter with auto-detection
    of thresholds and quadrant labels for known metric pairs.

    Commonly used for analyzing:
    - Trend × ADI: When trend signals lie
    - Seasonal × ADI: When seasonality signals lie
    - Trend × Entropy: Structure vs chaos
    - CV² × ADI: Demand classification

    Parameters
    ----------
    df : pd.DataFrame
        Data with metric columns.
    x_col : str
        X-axis metric.
    y_col : str
        Y-axis metric.
    x_threshold : float, optional
        X-axis threshold. Auto-detected for known metrics if None.
    y_threshold : float, optional
        Y-axis threshold. Auto-detected for known metrics if None.
    quadrant_labels : dict, optional
        Labels for quadrants. Auto-detected for known metric pairs if None.
    show_threshold_annotation : bool, default True
        Show threshold values as annotation at bottom.
    clip_quantile : float, default 0.95
        Clip extreme values.
    kind : {"histogram", "scatter"}, default "histogram"
        Plot type.
    theme : str, default "fa"
        Theme name.
    style : dict, optional
        Style overrides.

    Returns
    -------
    go.Figure
        Plotly figure.

    See Also
    --------
    plot_scatter : The underlying function with full flexibility.

    Examples
    --------
    >>> # Auto-detected thresholds and labels for Trend × ADI
    >>> plot_metric_interaction(df, 'trend', 'adi')

    >>> # Custom thresholds
    >>> plot_metric_interaction(df, 'trend', 'adi', x_threshold=0.6, y_threshold=1.5)
    """
    # Map 'histogram' to 'density' for backward compatibility
    scatter_kind = "density" if kind == "histogram" else "scatter"

    return plot_scatter(
        df,
        x=x_col,
        y=y_col,
        kind=scatter_kind,
        x_threshold=x_threshold,
        y_threshold=y_threshold,
        quadrant_labels=quadrant_labels,
        show_threshold_annotation=show_threshold_annotation,
        use_metric_defaults=True,  # Auto-detect thresholds & labels
        clip_quantile=clip_quantile,
        theme=theme,
        style=style,
    )


def plot_structure_chaos(
    df: pd.DataFrame,
    structure_col: str = "structure_score",
    chaos_col: str = "chaos_score",
    *,
    structure_threshold: float = 0.5,
    chaos_threshold: float = 0.5,
    theme: str = "fa",
    style: Optional[dict] = None,
) -> go.Figure:
    """
    Structure vs Chaos score interaction plot.

    Parameters
    ----------
    df : pd.DataFrame
        Data with structure and chaos score columns.
    structure_col : str, default "structure_score"
        Structure score column.
    chaos_col : str, default "chaos_score"
        Chaos score column.
    structure_threshold : float, default 0.5
        Structure score threshold.
    chaos_threshold : float, default 0.5
        Chaos score threshold.
    theme : str, default "fa"
        Theme name.
    style : dict, optional
        Style overrides.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    quadrant_labels = {
        "top_right": "Complex\nStructured + Chaotic",
        "top_left": "Messy\nUnstructured + Chaotic",
        "bottom_right": "Stable\nStructured + Clean",
        "bottom_left": "Low Signal\nNeither pattern",
    }

    return plot_scatter(
        df,
        x=structure_col,
        y=chaos_col,
        kind="density",
        x_threshold=structure_threshold,
        y_threshold=chaos_threshold,
        quadrant_labels=quadrant_labels,
        theme=theme,
        style=style,
    )


def plot_portfolio_metrics(
    profiles: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    *,
    bins: int = 30,
    mode: str = "facet",
    wrap: int = 2,
    show_kde: bool = True,
    show_median: bool = True,
    show_threshold: bool = True,
    clip_quantile: Optional[float] = None,
    theme: str = "fa",
    style: Optional[dict] = None,
) -> go.Figure:
    """
    Plot distributions of multiple portfolio metrics with KDE, median, and threshold annotations.

    This is a convenience wrapper around plot_distribution with preset options for
    portfolio metric visualization, matching the style of plot_metric_distribution:
    - Histogram + KDE density curve overlay
    - Median line with annotation
    - Threshold line with percentage annotation
    - Structure (blue) vs Chaos (orange) color coding

    Parameters
    ----------
    profiles : pd.DataFrame
        Profile data with metric columns.
    metrics : list of str, optional
        Metrics to plot. Default: trend, seasonal_strength, entropy, cv2, adi.
    bins : int, default 30
        Number of histogram bins.
    mode : {"facet", "dropdown"}, default "facet"
        Display mode.
    wrap : int, default 2
        Columns for facet mode.
    show_kde : bool, default True
        Show KDE density curve overlay.
    show_median : bool, default True
        Show median line with annotation.
    show_threshold : bool, default True
        Show threshold line with percentage annotation.
    clip_quantile : float, optional
        Clip extreme values. If None, uses metric-specific defaults.
    theme : str, default "fa"
        Theme name.
    style : dict, optional
        Style overrides.

    Returns
    -------
    go.Figure
        Plotly figure.

    See Also
    --------
    plot_distribution : The underlying function with full flexibility.

    Examples
    --------
    >>> # Quick portfolio overview with all defaults
    >>> plot_portfolio_metrics(diagnostics)

    >>> # Custom metrics in 3-column layout
    >>> plot_portfolio_metrics(
    ...     diagnostics,
    ...     metrics=['trend', 'entropy', 'adi', 'cv2', 'lumpiness'],
    ...     wrap=3,
    ... )
    """

    if metrics is None:
        metrics = ["trend", "seasonal_strength", "entropy", "cv2", "adi"]

    # Filter to available metrics
    available = [m for m in metrics if m in profiles.columns]
    if not available:
        raise ValueError(f"None of the metrics {metrics} found in profiles")

    return plot_distribution(
        profiles,
        columns=available,
        mode=mode,
        bins=bins,
        wrap=wrap,
        # Enable metric defaults for thresholds, colors, clipping
        use_metric_defaults=True,
        # Override clip_quantile if provided
        clip_quantile=clip_quantile if clip_quantile is not None else "auto",
        # KDE and annotations
        show_kde=show_kde,
        show_median=show_median,
        show_threshold_pct=show_threshold,
        # Styling
        theme=theme,
        style=style,
    )
