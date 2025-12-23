# tsforge/plots/charts/decomposition.py
"""STL decomposition visualization for time series."""
from __future__ import annotations

import pandas as pd
import numpy as np
from math import ceil
from typing import Union, List, Optional, Literal

from .._styling import PALETTE, apply_theme, apply_legend
from .._preprocessing import aggregate_by_group, select_ids
from .._layout import finalize_figure


def stl_decompose(series: pd.Series, period: int = None, seasonal: int = 7,
                  trend: int = None, robust: bool = True) -> dict:
    """Perform STL decomposition on a time series."""
    from statsmodels.tsa.seasonal import STL

    series_clean = series.dropna()
    if len(series_clean) < 2 * (period or 7):
        n = len(series)
        return {"observed": series.values, "trend": np.full(n, np.nan),
                "seasonal": np.full(n, np.nan), "resid": np.full(n, np.nan)}

    if period is None:
        period = 7
    if seasonal % 2 == 0:
        seasonal += 1

    try:
        stl = STL(series_clean, period=period, seasonal=seasonal, trend=trend, robust=robust)
        result = stl.fit()
        return {"observed": result.observed, "trend": result.trend,
                "seasonal": result.seasonal, "resid": result.resid}
    except Exception:
        n = len(series)
        return {"observed": series.values, "trend": np.full(n, np.nan),
                "seasonal": np.full(n, np.nan), "resid": np.full(n, np.nan)}


def compute_decomposition_stats(decomp: dict) -> dict:
    """Compute summary statistics from decomposition results."""
    trend = decomp["trend"]
    seasonal = decomp["seasonal"]
    resid = decomp["resid"]
    observed = decomp["observed"]

    trend_clean = trend[~np.isnan(trend)]
    seasonal_clean = seasonal[~np.isnan(seasonal)]
    resid_clean = resid[~np.isnan(resid)]
    observed_clean = observed[~np.isnan(observed)]

    stats = {
        "trend_min": np.min(trend_clean) if len(trend_clean) > 0 else np.nan,
        "trend_max": np.max(trend_clean) if len(trend_clean) > 0 else np.nan,
        "trend_range": (np.max(trend_clean) - np.min(trend_clean)) if len(trend_clean) > 0 else np.nan,
        "seasonal_amplitude": (np.max(seasonal_clean) - np.min(seasonal_clean)) if len(seasonal_clean) > 0 else np.nan,
        "seasonal_std": np.std(seasonal_clean) if len(seasonal_clean) > 0 else np.nan,
        "resid_std": np.std(resid_clean) if len(resid_clean) > 0 else np.nan,
        "resid_mean": np.mean(resid_clean) if len(resid_clean) > 0 else np.nan,
        "observed_mean": np.mean(observed_clean) if len(observed_clean) > 0 else np.nan,
        "observed_std": np.std(observed_clean) if len(observed_clean) > 0 else np.nan,
    }

    if stats["resid_std"] > 0:
        stats["snr"] = stats["seasonal_amplitude"] / stats["resid_std"]
    else:
        stats["snr"] = np.nan

    deseasonalized = observed_clean - seasonal_clean[:len(observed_clean)] if len(seasonal_clean) >= len(observed_clean) else observed_clean
    if np.var(deseasonalized) > 0:
        stats["trend_strength"] = max(0, 1 - np.var(resid_clean) / np.var(deseasonalized))
    else:
        stats["trend_strength"] = np.nan

    detrended = observed_clean - trend_clean[:len(observed_clean)] if len(trend_clean) >= len(observed_clean) else observed_clean
    if np.var(detrended) > 0:
        stats["seasonal_strength"] = max(0, 1 - np.var(resid_clean) / np.var(detrended))
    else:
        stats["seasonal_strength"] = np.nan

    return stats


def plot_decomposition(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    ids: Union[None, int, str, List[str]] = None,
    group_col: Union[str, List[str], None] = None,
    agg: str = "sum",
    period: int = None,
    seasonal: int = 7,
    trend: int = None,
    robust: bool = True,
    mode: Literal["panel", "overlay", "dropdown"] = "panel",
    show_stats: bool = True,
    components: List[str] = None,
    wrap: Optional[int] = None,
    theme: str = "fa",
    style: Optional[dict] = None,
    engine: str = "plotly",
):
    """STL decomposition visualization for time series."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if engine != "plotly":
        raise NotImplementedError("Only Plotly engine is supported.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if components is None:
        components = ["observed", "trend", "seasonal", "resid"]
    valid_components = {"observed", "trend", "seasonal", "resid"}
    components = [c for c in components if c in valid_components] or ["observed", "trend", "seasonal", "resid"]

    df, id_col = aggregate_by_group(df, group_col, date_col, value_col, agg, id_col)
    ids = select_ids(df, id_col, ids, 4)
    df = df[df[id_col].isin(ids)].copy()

    # Perform decomposition
    decomp_results = {}
    stats_results = {}
    for uid in ids:
        sub = df[df[id_col] == uid].sort_values(date_col)
        series = sub[value_col].reset_index(drop=True)
        dates = sub[date_col].values
        decomp = stl_decompose(series, period=period, seasonal=seasonal, trend=trend, robust=robust)
        decomp["dates"] = dates
        decomp_results[uid] = decomp
        stats_results[uid] = compute_decomposition_stats(decomp)

    component_config = {
        "observed": {"name": "Observed", "color": PALETTE[0]},
        "trend": {"name": "Trend", "color": PALETTE[1]},
        "seasonal": {"name": "Seasonal", "color": PALETTE[2]},
        "resid": {"name": "Residual", "color": PALETTE[3]},
    }

    if mode == "panel":
        fig = _plot_panel(decomp_results, stats_results, ids, components, component_config, show_stats, wrap)
    elif mode == "overlay":
        fig = _plot_overlay(decomp_results, ids, components, component_config)
    elif mode == "dropdown":
        fig = _plot_dropdown(decomp_results, stats_results, ids, components, component_config)
    else:
        raise ValueError("mode must be one of: panel, overlay, dropdown")

    return finalize_figure(fig, theme, style)


def _plot_panel(decomp_results, stats_results, ids, components, config, show_stats, wrap):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_components = len(components)
    if len(ids) == 1:
        uid = ids[0]
        decomp = decomp_results[uid]
        dates = decomp["dates"]
        fig = make_subplots(rows=n_components, cols=1, shared_xaxes=True,
                            subplot_titles=[config[c]["name"] for c in components], vertical_spacing=0.08)
        for i, comp in enumerate(components, start=1):
            cfg = config[comp]
            fig.add_trace(go.Scatter(x=dates, y=decomp[comp], mode="lines", name=cfg["name"],
                                     line=dict(color=cfg["color"], width=2), showlegend=False), row=i, col=1)
            if comp in ["seasonal", "resid"]:
                fig.add_hline(y=0, row=i, col=1, line_dash="dot", line_color="gray", opacity=0.5)
        if show_stats:
            stats = stats_results[uid]
            stats_text = (f"Trend: {stats['trend_min']:.1f} → {stats['trend_max']:.1f}<br>"
                          f"Seasonal amp: {stats['seasonal_amplitude']:.1f}<br>"
                          f"Residual σ: {stats['resid_std']:.2f}<br>"
                          f"Trend strength: {stats['trend_strength']:.2f}<br>"
                          f"Seasonal strength: {stats['seasonal_strength']:.2f}")
            fig.add_annotation(x=1.02, y=0.5, xref="paper", yref="paper", text=stats_text,
                               showarrow=False, font=dict(size=10, color="gray"), align="left",
                               bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(0,0,0,0.1)", borderwidth=1)
        fig.update_layout(height=200 * n_components)
    else:
        n_series = len(ids)
        cols = wrap or min(2, n_series)
        rows = ceil(n_series / cols)
        fig = make_subplots(rows=rows * n_components, cols=cols, shared_xaxes=True,
                            vertical_spacing=0.03, horizontal_spacing=0.08)
        for s_idx, uid in enumerate(ids):
            decomp = decomp_results[uid]
            dates = decomp["dates"]
            col = (s_idx % cols) + 1
            base_row = (s_idx // cols) * n_components + 1
            for c_idx, comp in enumerate(components):
                row = base_row + c_idx
                cfg = config[comp]
                fig.add_trace(go.Scatter(x=dates, y=decomp[comp], mode="lines", name=f"{uid} - {cfg['name']}",
                                         line=dict(color=cfg["color"], width=1.5),
                                         showlegend=(s_idx == 0), legendgroup=comp), row=row, col=col)
                if c_idx == 0:
                    fig.add_annotation(x=0.5, y=1.0, xref=f"x{col if col > 1 else ''} domain",
                                       yref=f"y{base_row} domain", text=f"<b>{uid}</b>",
                                       showarrow=False, font=dict(size=11), yshift=15)
        fig.update_layout(height=150 * n_components * rows)
    return fig


def _plot_overlay(decomp_results, ids, components, config):
    import plotly.graph_objects as go

    fig = go.Figure()
    for uid in ids:
        decomp = decomp_results[uid]
        dates = decomp["dates"]
        for comp in components:
            cfg = config[comp]
            suffix = f" ({uid})" if len(ids) > 1 else ""
            fig.add_trace(go.Scatter(
                x=dates, y=decomp[comp], mode="lines", name=f"{cfg['name']}{suffix}",
                line=dict(color=cfg["color"], width=2 if comp == "observed" else 1.5,
                          dash="solid" if comp in ["observed", "trend"] else "dot"),
                opacity=1.0 if comp in ["observed", "trend"] else 0.7,
            ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)
    return fig


def _plot_dropdown(decomp_results, stats_results, ids, components, config):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_components = len(components)
    fig = make_subplots(rows=n_components, cols=1, shared_xaxes=True,
                        subplot_titles=[config[c]["name"] for c in components], vertical_spacing=0.08)
    trace_counts = []
    for s_idx, uid in enumerate(ids):
        decomp = decomp_results[uid]
        dates = decomp["dates"]
        local_count = 0
        for c_idx, comp in enumerate(components, start=1):
            cfg = config[comp]
            fig.add_trace(go.Scatter(x=dates, y=decomp[comp], mode="lines", name=cfg["name"],
                                     line=dict(color=cfg["color"], width=2),
                                     visible=(s_idx == 0), showlegend=(s_idx == 0)), row=c_idx, col=1)
            local_count += 1
        trace_counts.append(local_count)

    buttons = []
    total = sum(trace_counts)
    start = 0
    for i, uid in enumerate(ids):
        visible = [False] * total
        for k in range(start, start + trace_counts[i]):
            visible[k] = True
        stats = stats_results[uid]
        title = f"{uid} | Trend: {stats['trend_strength']:.2f}, Seasonal: {stats['seasonal_strength']:.2f}"
        buttons.append(dict(label=str(uid), method="update", args=[{"visible": visible}, {"title": title}]))
        start += trace_counts[i]

    fig.update_layout(updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.0, "y": 1.15,
                                    "xanchor": "right", "yanchor": "top"}], height=200 * n_components)
    for i, comp in enumerate(components, start=1):
        if comp in ["seasonal", "resid"]:
            fig.add_hline(y=0, row=i, col=1, line_dash="dot", line_color="gray", opacity=0.5)
    return fig


def decompose_series(df: pd.DataFrame, id_col: str, date_col: str, value_col: str,
                     period: int = None, seasonal: int = 7, robust: bool = True) -> pd.DataFrame:
    """Decompose all series and return results as a DataFrame."""
    results = []
    for uid, grp in df.groupby(id_col):
        sub = grp.sort_values(date_col).copy()
        series = sub[value_col].reset_index(drop=True)
        decomp = stl_decompose(series, period=period, seasonal=seasonal, robust=robust)
        sub = sub.reset_index(drop=True)
        sub["trend"] = decomp["trend"]
        sub["seasonal"] = decomp["seasonal"]
        sub["resid"] = decomp["resid"]
        results.append(sub[[id_col, date_col, value_col, "trend", "seasonal", "resid"]])
    return pd.concat(results, ignore_index=True)


def get_decomposition_stats(df: pd.DataFrame, id_col: str, date_col: str, value_col: str,
                            period: int = None, seasonal: int = 7, robust: bool = True) -> pd.DataFrame:
    """Compute decomposition statistics for all series."""
    results = []
    for uid, grp in df.groupby(id_col):
        sub = grp.sort_values(date_col)
        series = sub[value_col].reset_index(drop=True)
        decomp = stl_decompose(series, period=period, seasonal=seasonal, robust=robust)
        stats = compute_decomposition_stats(decomp)
        stats[id_col] = uid
        results.append(stats)
    stats_df = pd.DataFrame(results)
    cols = [id_col] + [c for c in stats_df.columns if c != id_col]
    return stats_df[cols]
