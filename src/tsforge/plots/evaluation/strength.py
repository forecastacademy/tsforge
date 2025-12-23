# tsforge/plots/evaluation/strength.py
"""Strength distribution visualization for trend and seasonal components."""
from __future__ import annotations

import pandas as pd
from typing import Optional, List

from .._styling import PALETTE, apply_theme
from .._layout import finalize_figure


def plot_strength(
    profiles: pd.DataFrame,
    columns: Optional[List[str]] = None,
    bins: int = 30,
    thresholds: Optional[List[float]] = None,
    theme: str = "fa",
    style: Optional[dict] = None,
    engine: str = "plotly",
):
    """Visualize strength distributions for trend and seasonal components."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if engine != "plotly":
        raise NotImplementedError("Only Plotly engine is supported.")

    if columns is None:
        columns = ['trend_strength', 'seasonal_strength']
    if thresholds is None:
        thresholds = [0.3, 0.6]

    threshold_colors = ['#FFA500', '#E63946']
    titles = {'trend_strength': 'Trend Strength', 'seasonal_strength': 'Seasonal Strength'}
    threshold_labels = ['Weak/Moderate', 'Moderate/Strong']
    if style and 'threshold_labels' in style:
        threshold_labels = style['threshold_labels']

    available_cols = [col for col in columns if col in profiles.columns]
    if not available_cols:
        raise ValueError(f"None of the specified columns {columns} found in profiles DataFrame.")

    n_cols = len(available_cols)
    fig = make_subplots(rows=n_cols, cols=1,
                        subplot_titles=[titles.get(col, col.replace('_', ' ').title()) for col in available_cols],
                        vertical_spacing=0.15)

    for i, col in enumerate(available_cols, start=1):
        data = profiles[col].dropna()
        fig.add_trace(go.Histogram(
            x=data, nbinsx=bins,
            marker=dict(color=PALETTE[i - 1], line=dict(color='white', width=1)),
            opacity=0.7, name=titles.get(col, col), showlegend=False,
        ), row=i, col=1)

        for j, threshold in enumerate(thresholds):
            label = threshold_labels[j] if j < len(threshold_labels) else f"Threshold {threshold}"
            color = threshold_colors[j % len(threshold_colors)]
            vline_kwargs = dict(x=threshold, row=i, col=1, line_dash="dash", line_color=color, line_width=2)
            if i == 1:
                vline_kwargs.update(annotation_text=label,
                                    annotation_position="top right" if j == 0 else "top left",
                                    annotation_font_size=10)
            fig.add_vline(**vline_kwargs)

        fig.update_xaxes(title_text="Strength" if i == n_cols else None, row=i, col=1)
        fig.update_yaxes(title_text="Count", row=i, col=1)

    fig.update_layout(height=300 * n_cols, width=700, showlegend=True)

    for j, threshold in enumerate(thresholds):
        label = threshold_labels[j] if j < len(threshold_labels) else f"{threshold}"
        color = threshold_colors[j % len(threshold_colors)]
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name=f"{label} ({threshold})",
                                 line=dict(color=color, dash='dash', width=2), showlegend=True))

    fig = apply_theme(fig, theme)

    if style:
        if "title" in style and "subtitle" not in style:
            fig.update_layout(title=style["title"])
        if "title" in style and "subtitle" in style:
            fig.update_layout(title=f"{style['title']}<br><sup>{style['subtitle']}</sup>")

    fig.update_layout(
        legend=dict(orientation='h', x=0.5, y=-0.15, xanchor='center', yanchor='top',
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1),
        margin=dict(b=100),
    )

    return fig
