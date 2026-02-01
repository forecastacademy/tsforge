# tsforge/plots/shortcuts/eda.py
"""EDA shortcut functions - thin wrappers around core plotting functions.

These provide convenient preset configurations for common EDA tasks.
"""
from __future__ import annotations

import pandas as pd
from typing import Optional, List, Literal

import plotly.graph_objects as go

from ..core.scatter import plot_scatter, plot_2d_histogram
from ..core.distribution import plot_distribution
from ..core.bar import plot_bar, plot_category_counts


def plot_intermittency_scatter(
    df: pd.DataFrame,
    cv2_col: str = "cv2",
    adi_col: str = "adi",
    *,
    cv2_threshold: float = 0.49,
    adi_threshold: float = 1.32,
    classification_col: Optional[str] = "classification",
    clip_quantile: float = 0.95,
    theme: str = "fa",
    style: Optional[dict] = None,
) -> go.Figure:
    """
    Plot ADI vs CV² scatter for intermittency analysis.

    Syntetos-Boylan demand classification quadrants:
    - Smooth: low CV², low ADI
    - Erratic: high CV², low ADI
    - Intermittent: low CV², high ADI
    - Lumpy: high CV², high ADI

    Parameters
    ----------
    df : pd.DataFrame
        Data with cv2 and adi columns.
    cv2_col : str, default "cv2"
        Column name for CV².
    adi_col : str, default "adi"
        Column name for ADI.
    cv2_threshold : float, default 0.49
        CV² classification threshold.
    adi_threshold : float, default 1.32
        ADI classification threshold.
    classification_col : str, optional
        Column for coloring points by classification.
    clip_quantile : float, default 0.95
        Clip extreme values.
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
        "bottom_left": "Smooth",
        "bottom_right": "Erratic",
        "top_left": "Intermittent",
        "top_right": "Lumpy",
    }

    class_colors = {
        "Smooth": "#27ae60",
        "Erratic": "#f39c12",
        "Intermittent": "#3498db",
        "Lumpy": "#e74c3c",
    }

    return plot_scatter(
        df, x=cv2_col, y=adi_col,
        color=classification_col if classification_col and classification_col in df.columns else None,
        x_threshold=cv2_threshold,
        y_threshold=adi_threshold,
        quadrant_labels=quadrant_labels,
        color_map=class_colors,
        clip_quantile=clip_quantile,
        theme=theme,
        style=style,
    )


def plot_zero_distribution(
    df: pd.DataFrame,
    id_col: str,
    value_col: str,
    *,
    thresholds: Optional[List[float]] = None,
    threshold_labels: Optional[List[str]] = None,
    sort_by: str = "zero_pct",
    top_n: Optional[int] = 20,
    theme: str = "fa",
    style: Optional[dict] = None,
) -> go.Figure:
    """
    Plot zero percentage distribution across series.

    Parameters
    ----------
    df : pd.DataFrame
        Data with id and value columns.
    id_col : str
        Series identifier column.
    value_col : str
        Value column to analyze.
    thresholds : list of float, optional
        Reference lines. Default: [10, 30, 60, 90].
    threshold_labels : list of str, optional
        Labels for thresholds.
    sort_by : str, default "zero_pct"
        Column to sort by.
    top_n : int, optional
        Show top N series.
    theme : str, default "fa"
        Theme name.
    style : dict, optional
        Style overrides.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if thresholds is None:
        thresholds = [10, 30, 60, 90]
    if threshold_labels is None:
        threshold_labels = ["10%", "30%", "60%", "90%"]

    # Compute zero percentages
    stats = (
        df.groupby(id_col)[value_col]
        .apply(lambda x: 100 * (x == 0).mean())
        .reset_index()
        .rename(columns={value_col: "zero_pct"})
        .sort_values(sort_by, ascending=False)
    )

    return plot_bar(
        stats, x=id_col, y="zero_pct",
        sort_by=sort_by,
        sort_ascending=False,
        top_n=top_n,
        thresholds=thresholds,
        threshold_labels=threshold_labels,
        orientation="v",
        theme=theme,
        style=style,
    )


def plot_archetype_distribution(
    df: pd.DataFrame,
    archetype_col: str = "archetype",
    *,
    order: Optional[List[str]] = None,
    theme: str = "fa",
    style: Optional[dict] = None,
) -> go.Figure:
    """
    Plot archetype category distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Data with archetype column.
    archetype_col : str, default "archetype"
        Column containing archetype labels.
    order : list of str, optional
        Category order. Default: Stable, Complex, Messy, Low Signal.
    theme : str, default "fa"
        Theme name.
    style : dict, optional
        Style overrides.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if order is None:
        order = ["Stable", "Complex", "Messy", "Low Signal"]

    colors = {
        "Stable": "#27ae60",
        "Complex": "#3498db",
        "Messy": "#f39c12",
        "Low Signal": "#e74c3c",
    }

    return plot_category_counts(
        df, archetype_col,
        order=order,
        colors=colors,
        orientation="h",
        theme=theme,
        style=style,
    )


def plot_abc_distribution(
    df: pd.DataFrame,
    abc_col: str = "abc_class",
    *,
    theme: str = "fa",
    style: Optional[dict] = None,
) -> go.Figure:
    """
    Plot ABC classification distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Data with ABC class column.
    abc_col : str, default "abc_class"
        Column containing ABC labels.
    theme : str, default "fa"
        Theme name.
    style : dict, optional
        Style overrides.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    order = ["A", "B", "C"]
    colors = {"A": "#27ae60", "B": "#f39c12", "C": "#e74c3c"}

    return plot_category_counts(
        df, abc_col,
        order=order,
        colors=colors,
        orientation="h",
        theme=theme,
        style=style,
    )
