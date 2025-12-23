# tsforge/plots/compound/diagnostics.py
"""Diagnostic distribution and portfolio visualization."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Literal

import plotly.graph_objects as go

from tsforge.eda.thresholds import CalculateThresholds


COLORS = {
    'structure': '#4A90A4',
    'chaos': '#D4775D',
}

METRIC_CONFIG = {
    'trend': {'color': 'structure', 'title': 'Trend Strength Distribution', 'threshold': 0.5, 'threshold_label': 'Strong Trend', 'xlim': (0, 1)},
    'seasonal_strength': {'color': 'structure', 'title': 'Seasonal Strength Distribution', 'threshold': 0.5, 'threshold_label': 'Strong Seasonality', 'xlim': (0, 1)},
    'x_acf1': {'color': 'structure', 'title': 'Mutual Information (ACF1) Distribution', 'threshold': 0.6, 'threshold_label': 'Strong Persistence', 'xlim': (-0.5, 1)},
    'entropy': {'color': 'chaos', 'title': 'Entropy Distribution', 'threshold': 0.8, 'threshold_label': 'Chaotic', 'xlim': (0, 1)},
    'adi': {'color': 'chaos', 'title': 'ADI (Average Demand Interval) Distribution', 'threshold': 1.32, 'threshold_label': 'Intermittent', 'clip_quantile': 0.98},
    'cv2': {'color': 'chaos', 'title': 'CV² (Squared Coefficient of Variation) Distribution', 'threshold': 0.49, 'threshold_label': 'High Variability', 'clip_quantile': 0.95},
    'lumpiness': {'color': 'chaos', 'title': 'Lumpiness Distribution', 'threshold': 1.0, 'threshold_label': 'Lumpy', 'clip_quantile': 0.98},
    'zero_pct': {'color': 'chaos', 'title': 'Zero Percentage Distribution', 'threshold': 50, 'threshold_label': 'Majority Zeros'},
    'cov': {'color': 'chaos', 'title': 'Coefficient of Variation Distribution', 'threshold': 1.0, 'threshold_label': 'High CoV', 'clip_quantile': 0.95},
    'Structure_Score': {'color': 'structure', 'title': 'Structure Score Distribution', 'threshold': 0.5, 'threshold_label': 'High Structure', 'xlim': (0, 1)},
    'Chaos_Score': {'color': 'chaos', 'title': 'Chaos Score Distribution', 'threshold': 0.5, 'threshold_label': 'High Chaos', 'xlim': (0, 1)},
}

INTERACTION_CONFIG = {
    ('trend', 'adi'): {'title': 'Trend × ADI: When Trend Lies', 'labels': {'top_right': 'DANGER\nTrend unreliable', 'top_left': 'Intermittent\n(weak trend OK)', 'bottom_right': 'Real Trend\n(trust it)', 'bottom_left': 'Smooth\n(no trend)'}},
    ('seasonal_strength', 'adi'): {'title': 'Seasonality × ADI: When Seasonality Lies', 'labels': {'top_right': 'DANGER\nSeasonality unreliable', 'top_left': 'Intermittent\n(weak seasonal OK)', 'bottom_right': 'Real Seasonality\n(trust it)', 'bottom_left': 'Smooth\n(no seasonality)'}},
    ('trend', 'entropy'): {'title': 'Trend × Entropy: Does Chaos Override Structure?', 'labels': {'top_right': "Noisy Trend\n(don't trust)", 'top_left': 'Pure Chaos\n(no pattern)', 'bottom_right': 'Clean Trend\n(trust it)', 'bottom_left': 'Stable\n(low chaos, low trend)'}},
    ('seasonal_strength', 'entropy'): {'title': 'Seasonality × Entropy: Does Chaos Override Seasonality?', 'labels': {'top_right': "Noisy Seasonal\n(don't trust)", 'top_left': 'Pure Chaos\n(no pattern)', 'bottom_right': 'Clean Seasonal\n(trust it)', 'bottom_left': 'Stable\n(low chaos, weak seasonal)'}},
    ('cv2', 'adi'): {'title': 'CV² × ADI: Syntetos-Boylan Classification', 'labels': {'top_right': 'Lumpy', 'top_left': 'Intermittent', 'bottom_right': 'Erratic', 'bottom_left': 'Smooth'}},
}

PORTFOLIO_THRESHOLDS = {
    'trend': {'threshold': 0.6, 'direction': '>', 'label': 'Strong\nTrend'},
    'seasonal_strength': {'threshold': 0.6, 'direction': '>', 'label': 'Strong\nSeasonality'},
    'entropy': {'threshold': 0.9, 'direction': '<', 'label': 'Structured\n(Low Entropy)'},
    'adi': {'threshold': 1.32, 'direction': '<=', 'label': 'Regular\nDemand'},
    'cv2': {'threshold': 0.49, 'direction': '<=', 'label': 'Stable\nSizes'},
}


def plot_metric_distribution(data: pd.DataFrame, metric: str, color: Optional[str] = None, title: Optional[str] = None,
                             threshold: Optional[float] = None, threshold_label: Optional[str] = None,
                             clip_upper: Optional[float] = None, xlim: Optional[Tuple[float, float]] = None,
                             figsize: Tuple[int, int] = (10, 4)) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a clean histogram/KDE with optional threshold."""
    fig, ax = plt.subplots(figsize=figsize)
    values = data[metric].dropna()
    config = METRIC_CONFIG.get(metric, {})

    if color is None:
        color = config.get('color', 'structure')
    color = COLORS.get(color, color)

    if title is None:
        title = config.get('title', f"{metric.replace('_', ' ').title()} Distribution")
    if threshold is None:
        threshold = config.get('threshold')
    if threshold_label is None:
        threshold_label = config.get('threshold_label', f"High {metric.replace('_', ' ').title()}")
    if clip_upper is None:
        if 'clip_upper' in config:
            clip_upper = config['clip_upper']
        elif 'clip_quantile' in config:
            clip_upper = values.quantile(config['clip_quantile'])
    if xlim is None:
        xlim = config.get('xlim')

    values_plot = values.clip(upper=clip_upper) if clip_upper is not None else values
    sns.histplot(values_plot, ax=ax, color=color, alpha=0.6, edgecolor='white', stat='density')
    sns.kdeplot(values_plot, ax=ax, color=color, linewidth=2)

    median = values.median()
    ax.axvline(median, color=color, linestyle='--', linewidth=2, alpha=0.8)
    ax.annotate(f'Median: {median:.2f}', xy=(median, ax.get_ylim()[1] * 0.85), fontsize=11, color=color,
                fontweight='bold', ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    if threshold is not None:
        ax.axvline(threshold, color='#e74c3c', linestyle='-', linewidth=2, alpha=0.8)
        pct_above = (values > threshold).mean() * 100
        ax.annotate(f'{threshold_label}\n{pct_above:.1f}%', xy=(threshold + 0.02, ax.get_ylim()[1] * 0.7),
                    fontsize=10, color='#e74c3c', fontweight='bold')

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    if xlim is not None:
        ax.set_xlim(xlim)
    plt.tight_layout()
    return fig, ax


def plot_portfolio_characteristics(data: pd.DataFrame, thresholds: Optional[dict] = None, group_by: Optional[str] = None,
                                   group_value: Optional[str] = None, figsize: Tuple[int, int] = (10, 5),
                                   title: str = 'Portfolio Characteristics: What Patterns Did We Find?',
                                   top_n: int = 12) -> Tuple[plt.Figure, plt.Axes]:
    """Plot bar chart showing percentage of series meeting various diagnostic thresholds."""
    if thresholds is None:
        thresholds = PORTFOLIO_THRESHOLDS

    subtitle = None
    if group_by is not None:
        if not group_by.endswith('_id'):
            raise ValueError(f"group_by must be a column ending with '_id'")
        if group_by not in data.columns:
            raise ValueError(f"Column '{group_by}' not found")
        if group_value is None:
            raise ValueError("group_value must be specified when group_by is provided")
        data = data[data[group_by] == group_value]
        if len(data) == 0:
            raise ValueError(f"No data found for {group_by}='{group_value}'")
        group_label = group_by.replace('_id', '').replace('_', ' ').title()
        subtitle = f"Filtered by {group_label}: {group_value} (n={len(data):,})"

    labels, percentages = [], []
    for metric, config in thresholds.items():
        if metric not in data.columns:
            continue
        threshold = config['threshold']
        direction = config['direction']
        label = config.get('label', metric.replace('_', ' ').title())
        if direction == '>':
            pct = (data[metric] > threshold).mean() * 100
        elif direction == '>=':
            pct = (data[metric] >= threshold).mean() * 100
        elif direction == '<':
            pct = (data[metric] < threshold).mean() * 100
        elif direction == '<=':
            pct = (data[metric] <= threshold).mean() * 100
        else:
            raise ValueError(f"Unknown direction: {direction}")
        labels.append(label)
        percentages.append(pct)

    colors = [COLORS['structure'], COLORS['chaos']] * ((len(labels) + 1) // 2)
    colors = colors[:len(labels)]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, percentages, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    for bar, pct in zip(bars, percentages):
        ax.annotate(f'{pct:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylim(0, 100)
    ax.set_ylabel('% of Series', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center', va='bottom', fontsize=10, fontstyle='italic', color='gray')
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)

    legend_handles = [
        Patch(facecolor=COLORS['structure'], alpha=0.8, edgecolor='white', label='Structure'),
        Patch(facecolor=COLORS['chaos'], alpha=0.8, edgecolor='white', label='Chaos'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9)
    plt.tight_layout()
    return fig, ax


Style = Literal["bullet", "diverging", "tiles"]


def plot_structure_chaos_summary(structure: List[Tuple[str, float]], chaos: List[Tuple[str, float]], *,
                                 style: Style = "diverging", threshold: float = 50.0, title: Optional[str] = None,
                                 subtitle: Optional[str] = None, sort: Literal["none", "value_desc"] = "value_desc",
                                 annotate: bool = True, figsize: Optional[Tuple[float, float]] = None,
                                 ax: Optional[plt.Axes] = None):
    """Visual summary of '% of series' that show Structure vs Chaos signals."""
    s, c = list(structure), list(chaos)
    if sort == "value_desc":
        s = sorted(s, key=lambda x: x[1], reverse=True)
        c = sorted(c, key=lambda x: x[1], reverse=True)

    if style == "tiles":
        labels = [x[0] for x in s + c]
        vals = [x[1] for x in s + c]
        if figsize is None:
            figsize = (max(10, len(labels) * 2.0), 3.2)
        fig = plt.figure(figsize=figsize)
        ax_ = fig.add_subplot(111)
        ax_.axis("off")
        if title:
            fig.suptitle(title, y=1.05, fontsize=14)
        if subtitle:
            ax_.set_title(subtitle, fontsize=10, pad=10)
        xs = np.linspace(0.05, 0.95, len(labels))
        for x, label, val in zip(xs, labels, vals):
            txt = f"{label}\n{val:.0f}%" if annotate else f"{label}"
            ax_.text(x, 0.5, txt, ha="center", va="center", transform=ax_.transAxes)
        plt.show()
        return None

    if ax is None:
        if figsize is None:
            figsize = (11, 4.8)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if style == "bullet":
        labels = [x[0] for x in s + c]
        vals = [x[1] for x in s + c]
        y = np.arange(len(labels))
        ax.barh(y, vals)
        ax.axvline(threshold)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("% of Series")
        if title:
            ax.set_title(title)
        if subtitle:
            ax.text(0, 1.02, subtitle, transform=ax.transAxes, va="bottom")
        if annotate:
            for yi, v in zip(y, vals):
                ax.text(v + 1, yi, f"{v:.0f}%", va="center")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()
        return ax

    if style == "diverging":
        s_labels = [x[0] for x in s]
        s_vals = np.array([x[1] for x in s], dtype=float)
        c_labels = [x[0] for x in c]
        c_vals = np.array([x[1] for x in c], dtype=float)
        labels = s_labels + c_labels
        y = np.arange(len(labels))
        left_vals = np.concatenate([-s_vals, np.zeros_like(c_vals)])
        right_vals = np.concatenate([np.zeros_like(s_vals), c_vals])
        ax.barh(y, left_vals)
        ax.barh(y, right_vals)
        ax.axvline(0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("% of Series")
        maxv = max(np.max(s_vals) if len(s_vals) else 0, np.max(c_vals) if len(c_vals) else 0)
        ax.set_xlim(-(maxv * 1.15), (maxv * 1.15))
        if title:
            ax.set_title(title)
        if subtitle:
            ax.text(0, 1.02, subtitle, transform=ax.transAxes, va="bottom")
        if annotate:
            for yi, v in zip(y[: len(s_vals)], s_vals):
                ax.text(-(v + 1), yi, f"{v:.0f}%", va="center", ha="right")
            for yi, v in zip(y[len(s_vals):], c_vals):
                ax.text(v + 1, yi, f"{v:.0f}%", va="center", ha="left")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()
        return ax

    raise ValueError(f"Unknown style: {style}")


def plot_metric_interaction(df, x_col, y_col, ax=None, title=None, x_threshold=None, y_threshold=None,
                            quadrant_labels=None, warning_text=None, clip_quantile=0.95):
    """2D histogram with quadrant labels and percentages."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))
    config = INTERACTION_CONFIG.get((x_col, y_col), {})
    if title is None:
        title = config.get('title', f'{x_col} × {y_col}')
    if quadrant_labels is None:
        quadrant_labels = config.get('labels', {})

    x_median = df[x_col].dropna().median()
    y_median = df[y_col].dropna().median()

    if x_threshold is None:
        x_threshold = CalculateThresholds.DEFAULT.get(x_col)
    if y_threshold is None:
        y_threshold = CalculateThresholds.DEFAULT.get(y_col)

    plot_data = df[[x_col, y_col]].dropna().copy()
    n_total = len(plot_data)
    x_upper = plot_data[x_col].quantile(clip_quantile)
    y_upper = plot_data[y_col].quantile(clip_quantile)
    x_lower = plot_data[x_col].quantile(1 - clip_quantile)
    y_lower = plot_data[y_col].quantile(1 - clip_quantile)
    plot_data = plot_data[(plot_data[x_col] <= x_upper) & (plot_data[x_col] >= x_lower) &
                          (plot_data[y_col] <= y_upper) & (plot_data[y_col] >= y_lower)]

    h = ax.hist2d(plot_data[x_col], plot_data[y_col], bins=40, cmap='Blues', cmin=1)
    plt.colorbar(h[3], ax=ax, label='Count', shrink=0.8)

    if x_threshold is not None:
        ax.axvline(x=x_threshold, color='#2c3e50', linestyle='--', linewidth=2, alpha=0.8)
    if y_threshold is not None:
        ax.axhline(y=y_threshold, color='#2c3e50', linestyle='--', linewidth=2, alpha=0.8)

    if x_threshold is not None and y_threshold is not None:
        q_tl = ((df[x_col] < x_threshold) & (df[y_col] >= y_threshold)).sum() / n_total * 100
        q_tr = ((df[x_col] >= x_threshold) & (df[y_col] >= y_threshold)).sum() / n_total * 100
        q_bl = ((df[x_col] < x_threshold) & (df[y_col] < y_threshold)).sum() / n_total * 100
        q_br = ((df[x_col] >= x_threshold) & (df[y_col] < y_threshold)).sum() / n_total * 100
        quadrant_pcts = {'top_left': q_tl, 'top_right': q_tr, 'bottom_left': q_bl, 'bottom_right': q_br}

        if quadrant_labels:
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            x_left, x_right = (xlim[0] + x_threshold) / 2, (x_threshold + xlim[1]) / 2
            y_bottom, y_top = (ylim[0] + y_threshold) / 2, (y_threshold + ylim[1]) / 2
            positions = {'top_right': (x_right, y_top), 'top_left': (x_left, y_top),
                         'bottom_right': (x_right, y_bottom), 'bottom_left': (x_left, y_bottom)}
            q_colors = {'top_right': '#e74c3c', 'top_left': '#f39c12', 'bottom_right': '#27ae60', 'bottom_left': '#3498db'}
            for pos, label in quadrant_labels.items():
                if pos in positions:
                    pct = quadrant_pcts.get(pos, 0)
                    full_label = f"{label}\n({pct:.0f}%)"
                    ax.annotate(full_label, xy=positions[pos], fontsize=10, ha='center', va='center', fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=q_colors.get(pos, 'gray'), alpha=0.95, linewidth=2))

    if warning_text:
        ax.annotate(warning_text, xy=(0.5, -0.10), xycoords='axes fraction', ha='center', va='top', fontsize=10, color='#7f8c8d', style='italic')
    if x_threshold is not None and y_threshold is not None:
        ax.annotate(f'Thresholds: {x_col}={x_threshold}, {y_col}={y_threshold}', xy=(0.5, -0.12), xycoords='axes fraction',
                    ha='center', fontsize=10, style='italic', color='#7f8c8d')

    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    return ax


def plot_demand_classification(df, cv2_col='cv2', adi_col='adi', ax=None, cv2_threshold=None, adi_threshold=None,
                               title='CV² × ADI: Syntetos-Boylan Demand Classification', clip_quantile=0.95):
    """Syntetos-Boylan demand classification plot (CV² × ADI)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))

    if cv2_threshold is None:
        cv2_threshold = CalculateThresholds.DEFAULT.get('cv2', 0.49)
    if adi_threshold is None:
        adi_threshold = CalculateThresholds.DEFAULT.get('adi', 1.32)

    plot_data = df[[cv2_col, adi_col]].dropna().copy()
    n_total = len(plot_data)
    cv2_clip = plot_data[cv2_col].quantile(clip_quantile)
    adi_clip = plot_data[adi_col].quantile(clip_quantile)
    plot_clipped = plot_data[(plot_data[cv2_col] <= cv2_clip) & (plot_data[adi_col] <= adi_clip)]

    h = ax.hist2d(plot_clipped[cv2_col], plot_clipped[adi_col], bins=40, cmap='Blues', cmin=1)
    plt.colorbar(h[3], ax=ax, label='Count', shrink=0.8)

    ax.axhline(y=adi_threshold, color='#2c3e50', linestyle='--', linewidth=2.5)
    ax.axvline(x=cv2_threshold, color='#2c3e50', linestyle='--', linewidth=2.5)

    q_smooth = ((plot_data[cv2_col] < cv2_threshold) & (plot_data[adi_col] < adi_threshold)).sum() / n_total * 100
    q_erratic = ((plot_data[cv2_col] >= cv2_threshold) & (plot_data[adi_col] < adi_threshold)).sum() / n_total * 100
    q_intermittent = ((plot_data[cv2_col] < cv2_threshold) & (plot_data[adi_col] >= adi_threshold)).sum() / n_total * 100
    q_lumpy = ((plot_data[cv2_col] >= cv2_threshold) & (plot_data[adi_col] >= adi_threshold)).sum() / n_total * 100

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_left, x_right = (xlim[0] + cv2_threshold) / 2, (cv2_threshold + xlim[1]) / 2
    y_bottom, y_top = (ylim[0] + adi_threshold) / 2, (adi_threshold + ylim[1]) / 2

    labels_pos = {
        'Smooth': (x_left, y_bottom, '#27ae60', q_smooth),
        'Erratic': (x_right, y_bottom, '#f39c12', q_erratic),
        'Intermittent': (x_left, y_top, '#3498db', q_intermittent),
        'Lumpy': (x_right, y_top, '#e74c3c', q_lumpy)
    }
    for label, (x, y, color, pct) in labels_pos.items():
        ax.annotate(f"{label}\n({pct:.0f}%)", xy=(x, y), fontsize=11, fontweight='bold', ha='center', va='center', color=color,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color, alpha=0.95, linewidth=2))

    ax.set_xlabel('CV² (Demand Variability)', fontsize=12)
    ax.set_ylabel('ADI (Average Demand Interval)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.annotate(f'Thresholds: CV²={cv2_threshold}, ADI={adi_threshold}', xy=(0.5, -0.12), xycoords='axes fraction',
                ha='center', fontsize=10, style='italic', color='#7f8c8d')
    return ax


def plot_score_distributions(df, structure_col='structure_score', chaos_col='chaos_score',
                             structure_color=None, chaos_color=None, figsize=(12, 4)):
    """Side-by-side histograms of Structure and Chaos scores."""
    if structure_color is None:
        structure_color = COLORS['structure']
    if chaos_color is None:
        chaos_color = COLORS['chaos']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    struct_median = df[structure_col].median()
    axes[0].hist(df[structure_col], bins=50, color=structure_color, edgecolor='white', alpha=0.8)
    axes[0].axvline(struct_median, color='#c0392b', linestyle='--', linewidth=2, label=f"Median: {struct_median:.2f}")
    axes[0].set_xlabel('Structure Score', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Structure Score Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()

    chaos_median = df[chaos_col].median()
    axes[1].hist(df[chaos_col], bins=50, color=chaos_color, edgecolor='white', alpha=0.8)
    axes[1].axvline(chaos_median, color='#2c3e50', linestyle='--', linewidth=2, label=f"Median: {chaos_median:.2f}")
    axes[1].set_xlabel('Chaos Score', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Chaos Score Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    return fig, axes


def plot_diagnostics(diagnostics_df: pd.DataFrame, metric_cols: Optional[List[str]] = None,
                     top_n: int = 12, title_prefix: str = "Diagnostics"):
    """One-call diagnostics dashboard."""
    if metric_cols is None:
        metric_cols = [c for c in METRIC_CONFIG.keys() if c in diagnostics_df.columns]

    figs = {}
    if metric_cols:
        figs["metric_distribution"] = plot_metric_distribution(
            diagnostics_df, metric=metric_cols[0], title=f"{title_prefix}: {metric_cols[0]} distribution"
        )[0]

    figs["portfolio_characteristics"] = plot_portfolio_characteristics(
        diagnostics_df, top_n=top_n, title=f"{title_prefix}: portfolio characteristics"
    )[0]

    return figs
