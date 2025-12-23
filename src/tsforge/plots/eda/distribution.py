# tsforge/plots/charts/distribution.py
"""Distribution visualization for time series values."""
from __future__ import annotations

import pandas as pd
import numpy as np
from math import ceil
from typing import Union, List, Optional, Literal

from .._styling import PALETTE, HIGHLIGHT, ARCHETYPE_COLORS, ABC_COLORS, apply_theme, apply_legend
from .._preprocessing import aggregate_by_group, select_ids
from .._layout import finalize_figure


def plot_distribution(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    ids: Union[None, int, str, List[str]] = None,
    group_col: Union[str, List[str], None] = None,
    agg: str = "sum",
    kind: Literal["histogram", "density", "box", "violin"] = "histogram",
    mode: Literal["overlay", "facet", "dropdown"] = "overlay",
    bins: int = 30,
    log_scale: bool = False,
    exclude_zeros: bool = False,
    show_stats: bool = True,
    wrap: Optional[int] = None,
    theme: str = "fa",
    style: Optional[dict] = None,
    engine: str = "plotly",
):
    """Distribution visualization for time series values."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if engine != "plotly":
        raise NotImplementedError("Only Plotly engine is supported.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df, id_col = aggregate_by_group(df, group_col, date_col, value_col, agg, id_col)
    ids = select_ids(df, id_col, ids, 6)
    df = df[df[id_col].isin(ids)].copy()

    if exclude_zeros:
        df = df[df[value_col] > 0].copy()

    plot_col = value_col
    if log_scale:
        df["_log_value"] = np.log1p(df[value_col].clip(lower=0))
        plot_col = "_log_value"

    # Compute stats
    stats = {}
    for uid in ids:
        sub = df[df[id_col] == uid][value_col]
        stats[uid] = {
            "mean": sub.mean(), "median": sub.median(), "std": sub.std(),
            "min": sub.min(), "max": sub.max(), "n": len(sub),
        }

    if kind == "histogram":
        fig = _plot_histogram(df, ids, id_col, plot_col, bins, mode, wrap, stats, show_stats)
    elif kind == "density":
        fig = _plot_density(df, ids, id_col, plot_col, bins, mode, wrap)
    elif kind == "box":
        fig = _plot_box(df, ids, id_col, plot_col, mode, wrap)
    elif kind == "violin":
        fig = _plot_violin(df, ids, id_col, plot_col, mode, wrap)
    else:
        raise ValueError("kind must be one of: histogram, density, box, violin")

    x_label = f"log({value_col})" if log_scale else value_col
    if kind in ["histogram", "density"]:
        fig.update_xaxes(title_text=x_label)
        fig.update_yaxes(title_text="Frequency" if kind == "histogram" else "Density")
    else:
        fig.update_yaxes(title_text=x_label)

    return finalize_figure(fig, theme, style)


def _plot_histogram(df, ids, id_col, plot_col, bins, mode, wrap, stats, show_stats):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if mode == "overlay":
        fig = go.Figure()
        for i, uid in enumerate(ids):
            sub = df[df[id_col] == uid]
            fig.add_trace(go.Histogram(
                x=sub[plot_col], name=str(uid), opacity=0.7, nbinsx=bins,
                marker=dict(color=PALETTE[i % len(PALETTE)]),
            ))
        fig.update_layout(barmode="overlay")
    elif mode == "facet":
        n = len(ids)
        cols = wrap or 2
        rows = ceil(n / cols)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[str(uid) for uid in ids])
        r, c = 1, 1
        for i, uid in enumerate(ids):
            sub = df[df[id_col] == uid]
            fig.add_trace(go.Histogram(
                x=sub[plot_col], name=str(uid), nbinsx=bins, showlegend=False,
                marker=dict(color=PALETTE[i % len(PALETTE)]),
            ), row=r, col=c)
            if show_stats:
                mean_val = sub[plot_col].mean()
                fig.add_vline(x=mean_val, row=r, col=c, line_dash="dash", line_color="black",
                              annotation_text=f"μ={mean_val:.1f}", annotation_position="top right")
            c += 1
            if c > cols:
                r += 1
                c = 1
        fig.update_layout(height=300 * rows)
    else:  # dropdown
        fig = go.Figure()
        for i, uid in enumerate(ids):
            sub = df[df[id_col] == uid]
            fig.add_trace(go.Histogram(
                x=sub[plot_col], name=str(uid), nbinsx=bins, visible=(i == 0), showlegend=False,
                marker=dict(color=PALETTE[i % len(PALETTE)]),
            ))
        buttons = []
        for i, uid in enumerate(ids):
            visible = [j == i for j in range(len(ids))]
            s = stats[uid]
            title = f"{uid} | μ={s['mean']:.1f}, σ={s['std']:.1f}, n={s['n']}"
            buttons.append(dict(label=str(uid), method="update", args=[{"visible": visible}, {"title": title}]))
        fig.update_layout(updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.0, "y": 1.15, "xanchor": "right", "yanchor": "top"}])
    return fig


def _plot_density(df, ids, id_col, plot_col, bins, mode, wrap):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if mode == "overlay":
        fig = go.Figure()
        for i, uid in enumerate(ids):
            sub = df[df[id_col] == uid][plot_col].dropna()
            if len(sub) >= 2:
                hist, bin_edges = np.histogram(sub, bins=bins, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                fig.add_trace(go.Scatter(
                    x=bin_centers, y=hist, mode="lines", name=str(uid),
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                    fill="tozeroy", opacity=0.6,
                ))
    elif mode == "facet":
        n = len(ids)
        cols = wrap or 2
        rows = ceil(n / cols)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[str(uid) for uid in ids])
        r, c = 1, 1
        for i, uid in enumerate(ids):
            sub = df[df[id_col] == uid][plot_col].dropna()
            if len(sub) >= 2:
                hist, bin_edges = np.histogram(sub, bins=bins, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                fig.add_trace(go.Scatter(
                    x=bin_centers, y=hist, mode="lines", name=str(uid), showlegend=False,
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2), fill="tozeroy",
                ), row=r, col=c)
            c += 1
            if c > cols:
                r += 1
                c = 1
        fig.update_layout(height=300 * rows)
    else:
        fig = go.Figure()
        for i, uid in enumerate(ids):
            sub = df[df[id_col] == uid][plot_col].dropna()
            if len(sub) >= 2:
                hist, bin_edges = np.histogram(sub, bins=bins, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                fig.add_trace(go.Scatter(
                    x=bin_centers, y=hist, mode="lines", name=str(uid), visible=(i == 0), showlegend=False,
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2), fill="tozeroy",
                ))
        buttons = [dict(label=str(uid), method="update", args=[{"visible": [j == i for j in range(len(ids))]}]) for i, uid in enumerate(ids)]
        fig.update_layout(updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.0, "y": 1.15, "xanchor": "right", "yanchor": "top"}])
    return fig


def _plot_box(df, ids, id_col, plot_col, mode, wrap):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if mode == "overlay":
        fig = go.Figure()
        for i, uid in enumerate(ids):
            sub = df[df[id_col] == uid]
            fig.add_trace(go.Box(y=sub[plot_col], name=str(uid), marker=dict(color=PALETTE[i % len(PALETTE)]), boxpoints="outliers"))
    elif mode == "facet":
        n = len(ids)
        cols = wrap or 3
        rows = ceil(n / cols)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[str(uid) for uid in ids])
        r, c = 1, 1
        for i, uid in enumerate(ids):
            sub = df[df[id_col] == uid]
            fig.add_trace(go.Box(y=sub[plot_col], name=str(uid), marker=dict(color=PALETTE[i % len(PALETTE)]), boxpoints="outliers", showlegend=False), row=r, col=c)
            c += 1
            if c > cols:
                r += 1
                c = 1
        fig.update_layout(height=300 * rows)
    else:
        fig = go.Figure()
        for i, uid in enumerate(ids):
            sub = df[df[id_col] == uid]
            fig.add_trace(go.Box(y=sub[plot_col], name=str(uid), marker=dict(color=PALETTE[i % len(PALETTE)]), boxpoints="outliers", visible=(i == 0), showlegend=False))
        buttons = [dict(label=str(uid), method="update", args=[{"visible": [j == i for j in range(len(ids))]}]) for i, uid in enumerate(ids)]
        fig.update_layout(updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.0, "y": 1.15, "xanchor": "right", "yanchor": "top"}])
    return fig


def _plot_violin(df, ids, id_col, plot_col, mode, wrap):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if mode == "overlay":
        fig = go.Figure()
        for i, uid in enumerate(ids):
            sub = df[df[id_col] == uid]
            fig.add_trace(go.Violin(y=sub[plot_col], name=str(uid), line=dict(color=PALETTE[i % len(PALETTE)]), box_visible=True, meanline_visible=True))
    elif mode == "facet":
        n = len(ids)
        cols = wrap or 3
        rows = ceil(n / cols)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[str(uid) for uid in ids])
        r, c = 1, 1
        for i, uid in enumerate(ids):
            sub = df[df[id_col] == uid]
            fig.add_trace(go.Violin(y=sub[plot_col], name=str(uid), line=dict(color=PALETTE[i % len(PALETTE)]), box_visible=True, meanline_visible=True, showlegend=False), row=r, col=c)
            c += 1
            if c > cols:
                r += 1
                c = 1
        fig.update_layout(height=300 * rows)
    else:
        fig = go.Figure()
        for i, uid in enumerate(ids):
            sub = df[df[id_col] == uid]
            fig.add_trace(go.Violin(y=sub[plot_col], name=str(uid), line=dict(color=PALETTE[i % len(PALETTE)]), box_visible=True, meanline_visible=True, visible=(i == 0), showlegend=False))
        buttons = [dict(label=str(uid), method="update", args=[{"visible": [j == i for j in range(len(ids))]}]) for i, uid in enumerate(ids)]
        fig.update_layout(updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.0, "y": 1.15, "xanchor": "right", "yanchor": "top"}])
    return fig


def plot_skewness(
    profiles: pd.DataFrame,
    column: str = "skewness",
    bins: int = 40,
    clip: tuple = (-5, 10),
    thresholds: Optional[List[tuple]] = None,
    title: str = "Distribution Skewness Across Portfolio",
    figsize: tuple = (8, 4),
):
    """Plot skewness distribution across a portfolio of series."""
    import matplotlib.pyplot as plt

    if thresholds is None:
        thresholds = [
            (0, 'gray', '-', 'Symmetric'),
            (1, 'orange', '--', 'Moderate skew'),
            (2, 'red', '--', 'High skew'),
        ]

    data = profiles[column].dropna().clip(*clip)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(data, bins=bins, edgecolor='white', alpha=0.7)
    for val, color, linestyle, label in thresholds:
        ax.axvline(val, color=color, linestyle=linestyle, label=label)
    ax.set_xlabel('Skewness')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    return ax


def plot_category_distribution(
    df: pd.DataFrame,
    category_col: str,
    order: Optional[List[str]] = None,
    colors: Optional[dict] = None,
    weight_col: Optional[str] = None,
    title: str = 'Category Distribution',
    subtitle: Optional[str] = None,
    figsize: tuple = (10, 5),
):
    """
    Plot categorical distribution as a horizontal bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with category column.
    category_col : str
        Column name containing category labels.
    order : list of str, optional
        Order for categories. If None, uses value_counts order.
    colors : dict, optional
        Color mapping {category: color}. Defaults to PALETTE colors.
    weight_col : str, optional
        Column for weighted second chart. If provided, shows dual chart.
    title : str, default 'Category Distribution'
        Plot title.
    subtitle : str, optional
        Subtitle text displayed below title.
    figsize : tuple, default (10, 5)
        Figure size.

    Returns
    -------
    matplotlib.axes.Axes or numpy.ndarray
        The axes object(s).

    Examples
    --------
    >>> # Archetype distribution
    >>> plot_category_distribution(
    ...     df, 'archetype',
    ...     order=['Stable', 'Complex', 'Messy', 'Low Signal'],
    ...     colors=ARCHETYPE_COLORS,
    ...     subtitle=f'Thresholds: Structure={s:.3f}, Chaos={c:.3f}'
    ... )

    >>> # ABC distribution with volume
    >>> plot_category_distribution(
    ...     df, 'abc_class',
    ...     order=['A', 'B', 'C'],
    ...     colors=ABC_COLORS,
    ...     weight_col='total_volume'
    ... )
    """
    import matplotlib.pyplot as plt

    counts = df[category_col].value_counts()
    total = len(df)

    # Determine order
    if order is None:
        order = counts.index.tolist()

    # Determine colors
    if colors is None:
        colors = {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(order)}

    values = [counts.get(cat, 0) for cat in order]
    bar_colors = [colors.get(cat, '#888888') for cat in order]

    if weight_col and weight_col in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Left: by count
        ax = axes[0]
        bars = ax.barh(order, values, color=bar_colors, edgecolor='white', height=0.6)
        for bar, cat in zip(bars, order):
            count = counts.get(cat, 0)
            pct = count / total if total > 0 else 0
            ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{count:,}  ({pct:.1%})', va='center', fontsize=11)
        ax.set_xlabel('Count')
        ax.set_title('By Count', fontweight='bold')
        ax.set_xlim(0, max(values) * 1.25)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Right: by weight
        ax = axes[1]
        weight_by_cat = df.groupby(category_col)[weight_col].sum()
        total_weight = weight_by_cat.sum()
        w_values = [weight_by_cat.get(cat, 0) for cat in order]
        bars = ax.barh(order, w_values, color=bar_colors, edgecolor='white', height=0.6)
        for bar, cat in zip(bars, order):
            w = weight_by_cat.get(cat, 0)
            pct = w / total_weight if total_weight > 0 else 0
            ax.text(bar.get_width() + max(w_values) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{pct:.1%}', va='center', fontsize=11)
        ax.set_xlabel(weight_col.replace('_', ' ').title())
        ax.set_title(f'By {weight_col.replace("_", " ").title()}', fontweight='bold')
        ax.set_xlim(0, max(w_values) * 1.2)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        if subtitle:
            fig.text(0.5, 0.92, subtitle, ha='center', fontsize=10, color='gray')
        plt.tight_layout()
        plt.show()
        return axes
    else:
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(order, values, color=bar_colors, edgecolor='white', height=0.7)
        for bar, cat in zip(bars, order):
            count = counts.get(cat, 0)
            pct = count / total if total > 0 else 0
            ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{count:,}  ({pct:.1%})', va='center', fontsize=11)
        ax.set_xlabel('Count')
        ax.set_xlim(0, max(values) * 1.2)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=14, fontweight='bold', loc='left')
        if subtitle:
            ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color='gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()
        return ax
