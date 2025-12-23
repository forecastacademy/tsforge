# tsforge/plots/charts/intermittency.py
"""Intermittency and zero-pattern analysis for time series."""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Literal

from .._styling import PALETTE, apply_theme, apply_legend
from .._preprocessing import aggregate_by_group, select_ids
from .._layout import finalize_figure


def classify_intermittency(zero_pct: float, cv2: float = None, adi: float = None) -> str:
    """Classify series intermittency based on zero percentage and demand patterns."""
    if cv2 is not None and adi is not None:
        if cv2 < 0.49 and adi < 1.32:
            return "Smooth"
        elif cv2 >= 0.49 and adi < 1.32:
            return "Erratic"
        elif cv2 < 0.49 and adi >= 1.32:
            return "Intermittent"
        else:
            return "Lumpy"
    else:
        if zero_pct < 10:
            return "Continuous"
        elif zero_pct < 30:
            return "Low Intermittency"
        elif zero_pct < 60:
            return "Moderate Intermittency"
        elif zero_pct < 90:
            return "High Intermittency"
        else:
            return "Sparse"


def compute_intermittency_stats(df: pd.DataFrame, id_col: str, value_col: str) -> pd.DataFrame:
    """Compute intermittency statistics for each series."""
    results = []
    for uid, grp in df.groupby(id_col):
        values = grp[value_col].values
        n = len(values)
        n_zeros = (values == 0).sum()
        zero_pct = 100 * n_zeros / n if n > 0 else 0

        nonzero = values[values > 0]
        if len(nonzero) > 0:
            nonzero_mean = nonzero.mean()
            nonzero_std = nonzero.std()
            cv2 = (nonzero_std / nonzero_mean) ** 2 if nonzero_mean > 0 else 0
        else:
            nonzero_mean, nonzero_std, cv2 = 0, 0, 0

        nonzero_idx = np.where(values > 0)[0]
        if len(nonzero_idx) > 1:
            intervals = np.diff(nonzero_idx)
            adi = intervals.mean()
        else:
            adi = n

        classification = classify_intermittency(zero_pct, cv2, adi)
        results.append({
            id_col: uid, "n_obs": n, "n_zeros": n_zeros, "zero_pct": zero_pct,
            "nonzero_mean": nonzero_mean, "nonzero_std": nonzero_std,
            "cv2": cv2, "adi": adi, "classification": classification,
        })
    return pd.DataFrame(results)


def plot_intermittency(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    ids: Union[None, int, str, List[str]] = None,
    group_col: Union[str, List[str], None] = None,
    agg: str = "sum",
    kind: Literal["bar", "scatter", "heatmap", "classification"] = "bar",
    mode: Literal["overlay", "facet"] = "overlay",
    sort_by: Literal["zero_pct", "adi", "cv2", "id"] = "zero_pct",
    top_n: Optional[int] = None,
    show_thresholds: bool = True,
    wrap: Optional[int] = None,
    theme: str = "fa",
    style: Optional[dict] = None,
    engine: str = "plotly",
):
    """Intermittency and zero-pattern visualization."""
    import plotly.graph_objects as go

    if engine != "plotly":
        raise NotImplementedError("Only Plotly engine is supported.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df, id_col = aggregate_by_group(df, group_col, date_col, value_col, agg, id_col)
    stats_df = compute_intermittency_stats(df, id_col, value_col)

    if ids is not None:
        if isinstance(ids, int):
            selected = stats_df[id_col].sample(min(ids, len(stats_df))).tolist()
        elif isinstance(ids, str):
            selected = [ids]
        else:
            selected = list(ids)
        stats_df = stats_df[stats_df[id_col].isin(selected)]
        df = df[df[id_col].isin(selected)]

    if sort_by == "zero_pct":
        stats_df = stats_df.sort_values("zero_pct", ascending=False)
    elif sort_by == "adi":
        stats_df = stats_df.sort_values("adi", ascending=False)
    elif sort_by == "cv2":
        stats_df = stats_df.sort_values("cv2", ascending=False)
    else:
        stats_df = stats_df.sort_values(id_col)

    if top_n is not None:
        stats_df = stats_df.head(top_n)
        df = df[df[id_col].isin(stats_df[id_col])]

    class_colors = {
        "Continuous": "#2E86AB", "Low Intermittency": "#A6D854",
        "Moderate Intermittency": "#F6C85F", "High Intermittency": "#E76F51",
        "Sparse": "#9B2335", "Smooth": "#2E86AB", "Erratic": "#F6C85F",
        "Intermittent": "#E76F51", "Lumpy": "#9B2335",
    }

    if kind == "bar":
        fig = _plot_bar(stats_df, id_col, class_colors, show_thresholds)
    elif kind == "scatter":
        fig = _plot_scatter(stats_df, id_col, class_colors, show_thresholds)
    elif kind == "heatmap":
        fig = _plot_heatmap(df, stats_df, id_col, date_col, value_col)
    elif kind == "classification":
        fig = _plot_classification(stats_df, class_colors)
    else:
        raise ValueError("kind must be one of: bar, scatter, heatmap, classification")

    return finalize_figure(fig, theme, style)


def _plot_bar(stats_df, id_col, class_colors, show_thresholds):
    import plotly.graph_objects as go

    colors = [class_colors.get(c, PALETTE[0]) for c in stats_df["classification"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stats_df[id_col].astype(str), y=stats_df["zero_pct"],
        marker=dict(color=colors), text=stats_df["classification"], textposition="outside",
        hovertemplate="<b>%{x}</b><br>Zero %: %{y:.1f}%<br>Class: %{text}<extra></extra>",
        showlegend=False,
    ))
    if show_thresholds:
        for pct in [10, 30, 60, 90]:
            fig.add_hline(y=pct, line_dash="dot", line_color="gray", opacity=0.5,
                          annotation_text=f"{pct}%", annotation_position="right")
    fig.update_xaxes(title_text="Series", tickangle=45)
    fig.update_yaxes(title_text="Zero Percentage (%)", range=[0, 105])
    return fig


def _plot_scatter(stats_df, id_col, class_colors, show_thresholds):
    import plotly.graph_objects as go

    fig = go.Figure()
    for classification, grp in stats_df.groupby("classification"):
        fig.add_trace(go.Scatter(
            x=grp["adi"], y=grp["cv2"], mode="markers", name=classification,
            marker=dict(color=class_colors.get(classification, PALETTE[0]), size=10, opacity=0.7),
            text=grp[id_col],
            hovertemplate="<b>%{text}</b><br>ADI: %{x:.2f}<br>CV²: %{y:.2f}<extra></extra>",
        ))
    if show_thresholds:
        fig.add_vline(x=1.32, line_dash="dash", line_color="gray", opacity=0.7)
        fig.add_hline(y=0.49, line_dash="dash", line_color="gray", opacity=0.7)
        x_max = max(stats_df["adi"].max() * 1.1, 3)
        y_max = max(stats_df["cv2"].max() * 1.1, 1.5)
        annotations = [
            dict(x=0.66, y=0.25, text="Smooth", showarrow=False, font=dict(size=12, color="gray")),
            dict(x=0.66, y=y_max * 0.8, text="Erratic", showarrow=False, font=dict(size=12, color="gray")),
            dict(x=x_max * 0.7, y=0.25, text="Intermittent", showarrow=False, font=dict(size=12, color="gray")),
            dict(x=x_max * 0.7, y=y_max * 0.8, text="Lumpy", showarrow=False, font=dict(size=12, color="gray")),
        ]
        fig.update_layout(annotations=annotations)
    fig.update_xaxes(title_text="Average Demand Interval (ADI)")
    fig.update_yaxes(title_text="Squared CV (CV²)")
    return fig


def _plot_heatmap(df, stats_df, id_col, date_col, value_col):
    import plotly.graph_objects as go

    series_list = stats_df[id_col].tolist()
    df_filtered = df[df[id_col].isin(series_list)].copy()
    df_filtered["is_zero"] = (df_filtered[value_col] == 0).astype(int)
    pivot = df_filtered.pivot_table(index=id_col, columns=date_col, values="is_zero", aggfunc="max").fillna(0)
    pivot = pivot.reindex(series_list)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index.astype(str),
        colorscale=[[0, "#2E86AB"], [1, "#E5E5E5"]], showscale=False,
        hovertemplate="Series: %{y}<br>Date: %{x}<br>Zero: %{z}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Series", autorange="reversed")
    fig.update_layout(height=max(400, len(series_list) * 20))
    return fig


def _plot_classification(stats_df, class_colors):
    import plotly.graph_objects as go

    class_counts = stats_df["classification"].value_counts()
    colors = [class_colors.get(c, PALETTE[0]) for c in class_counts.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=class_counts.index, y=class_counts.values,
        marker=dict(color=colors), text=class_counts.values, textposition="outside", showlegend=False,
    ))
    total = class_counts.sum()
    for i, (cls, count) in enumerate(class_counts.items()):
        pct = 100 * count / total
        fig.add_annotation(x=cls, y=count, text=f"{pct:.1f}%", showarrow=False, yshift=25,
                           font=dict(size=11, color="gray"))
    fig.update_xaxes(title_text="Classification")
    fig.update_yaxes(title_text="Number of Series")
    return fig


def plot_demand_bars(series: pd.Series, title: str = "Demand Classification Distribution",
                     colors: Optional[List[str]] = None, figsize: tuple = (6, 4)):
    """Plot a bar chart of demand classification distribution."""
    import matplotlib.pyplot as plt

    if colors is None:
        colors = ['#E63946', '#457B9D', '#2A9D8F']
    counts = series.value_counts(normalize=True)
    ax = counts.plot.bar(title=title, ylabel='Proportion', rot=0, figsize=figsize, color=colors[:len(counts)])
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1%}', (p.get_x() + p.get_width() / 2, p.get_height() + 0.01), ha='center')
    return ax


def get_zero_stats(df: pd.DataFrame, id_col: str, value_col: str) -> pd.DataFrame:
    """Quick utility to get zero percentage for all series."""
    return (
        df.groupby(id_col)[value_col]
        .apply(lambda x: 100 * (x == 0).mean())
        .reset_index()
        .rename(columns={value_col: "zero_pct"})
        .sort_values("zero_pct", ascending=False)
    )
