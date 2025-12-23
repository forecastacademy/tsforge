# tsforge/plots/charts/coverage.py
"""Visualize series count per date to assess panel balance."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from typing import Optional

from .._styling import THEMES, apply_theme


def plot_date_coverage(
    df: pd.DataFrame,
    *,
    date_col: str = 'ds',
    id_col: str = 'unique_id',
    theme: str = 'fa',
    style: Optional[dict] = None,
) -> go.Figure:
    """
    Visualize series count per date to assess panel balance.

    A flat line indicates a balanced panel (all series present on all dates).
    Dips indicate missing series on those dates.
    """
    t = THEMES.get(theme, THEMES["fa"])
    line_color = t.get("line_color", "#2C32D5")
    line_width = t.get("line_width", 2.3)

    series_per_date = df.groupby(date_col)[id_col].nunique().reset_index()
    series_per_date.columns = [date_col, 'count']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series_per_date[date_col], y=series_per_date['count'],
        mode='lines', line=dict(color=line_color, width=line_width),
        name='Series count', hovertemplate='%{x}<br>Series: %{y:,}<extra></extra>',
    ))

    max_count = series_per_date['count'].max()
    min_count = series_per_date['count'].min()

    fig.add_hline(y=max_count, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text=f"Max: {max_count:,}", annotation_position="top right")

    if min_count < max_count * 0.95:
        fig.add_hline(y=min_count, line_dash="dot", line_color=t.get("accent_color", "#E63946"), opacity=0.5,
                      annotation_text=f"Min: {min_count:,}", annotation_position="bottom right")

    fig = apply_theme(fig, theme)

    title = "Date Coverage: Series per Date"
    x_title = ""
    y_title = "Series count"
    if style:
        title = style.get("title", title)
        x_title = style.get("x_title", x_title)
        y_title = style.get("y_title", y_title)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=x_title, yaxis_title=y_title,
        height=300, showlegend=False,
        margin=dict(l=60, r=40, t=60, b=40),
    )

    return fig
