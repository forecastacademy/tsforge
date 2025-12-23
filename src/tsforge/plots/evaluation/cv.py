# tsforge/plots/compound/cv.py
"""Cross-validation visualization functions."""
from __future__ import annotations

import pandas as pd
from typing import Optional, Literal

from .._styling import PALETTE, apply_theme


def plot_cv_plan(
    obj,
    df: Optional[pd.DataFrame] = None,
    series_id: Optional[str] = None,
    engine: str = "matplotlib",
):
    """
    Plot the cross-validation plan from WorkflowManager.cross_validation.

    Shows one subplot per fold with clear styling.

    Parameters
    ----------
    obj : WorkflowManager
        A fitted WorkflowManager with cross-validation results.
    df : DataFrame, optional
        Original data (must include id_col, time_col, target_col).
    series_id : str, optional
        Which series to show if multiple exist.
    engine : {'matplotlib','plotly'}, default 'matplotlib'
        Backend to use for plotting.

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objs._figure.Figure
    """
    from tsforge.workflows.manager import WorkflowManager

    if not isinstance(obj, WorkflowManager):
        raise ValueError("First argument must be a WorkflowManager")

    manager = obj
    if manager.last_cv_params is None or manager.last_cv_preds is None:
        raise RuntimeError("No CV results stored. Run cross_validation() first.")

    h = manager.last_cv_params["h"]
    cutoffs = sorted(manager.last_cv_preds["cutoff"].unique())
    id_col, time_col, target_col = manager.id_col, manager.time_col, manager.target_col

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    all_ids = df[id_col].unique()

    if series_id is None:
        if len(all_ids) == 1:
            series_id = all_ids[0]
        else:
            raise ValueError(f"Multiple series found ({len(all_ids)}). Please specify series_id.")

    series = df[df[id_col] == series_id].sort_values(time_col)

    if engine == "matplotlib":
        return _plot_cv_plan_mpl(series, time_col, target_col, cutoffs, h, series_id)
    elif engine == "plotly":
        return _plot_cv_plan_plotly(series, time_col, target_col, cutoffs, h, series_id)
    else:
        raise ValueError("engine must be 'matplotlib' or 'plotly'")


def _plot_cv_plan_mpl(series, time_col, target_col, cutoffs, h, series_id):
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter

    fig, axes = plt.subplots(len(cutoffs), 1, figsize=(6, 3 * len(cutoffs)), sharex=True)
    if len(cutoffs) == 1:
        axes = [axes]

    for i, (ax, cutoff) in enumerate(zip(axes, cutoffs), 1):
        train = series[series[time_col] <= cutoff]
        test = series[(series[time_col] > cutoff)].head(h)

        ax.plot(train[time_col], train[target_col], color=PALETTE[0], lw=2, alpha=0.8, label="Train")
        ax.plot(test[time_col], test[target_col], color=PALETTE[3], lw=2.5, marker="o", label="Test")
        ax.axvline(cutoff, color="black", ls="--", lw=1, alpha=0.7)
        ax.set_title(f"Fold {i}: cutoff = {cutoff.date()}  (series={series_id})", fontsize=12, pad=10)
        ax.legend(loc="upper left", frameon=False)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.grid(True, which="major", linestyle="--", alpha=0.4)
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))

    fig.suptitle("Cross-Validation Plan", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def _plot_cv_plan_plotly(series, time_col, target_col, cutoffs, h, series_id):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=len(cutoffs), cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=[f"Fold {i}: cutoff = {c.date()} (series={series_id})" for i, c in enumerate(cutoffs, 1)]
    )

    for i, cutoff in enumerate(cutoffs, 1):
        train = series[series[time_col] <= cutoff]
        test = series[(series[time_col] > cutoff)].head(h)

        fig.add_trace(go.Scatter(
            x=train[time_col], y=train[target_col], mode="lines", name="Train",
            line=dict(color=PALETTE[0], width=2), opacity=0.8
        ), row=i, col=1)

        fig.add_trace(go.Scatter(
            x=test[time_col], y=test[target_col], mode="lines+markers", name="Test",
            line=dict(color=PALETTE[3], width=2.5), opacity=0.9
        ), row=i, col=1)

        fig.add_vline(x=cutoff, line_dash="dash", line_color="black", row=i, col=1)

    fig.update_layout(height=300 * len(cutoffs), width=700, title="Cross-Validation Plan")
    return apply_theme(fig, "fa")


def plot_cv_results(
    folds_df: pd.DataFrame,
    metric: str = "wape",
    engine: Literal["matplotlib", "plotly"] = "matplotlib"
):
    """
    Plot CV results as mean ± std per model (teaching style).

    Parameters
    ----------
    folds_df : DataFrame
        Output from manager.cross_validation(..., mode="folds").
    metric : str, default 'wape'
        Metric column to plot (e.g. "wape", "rmse").
    engine : {'matplotlib','plotly'}, default 'matplotlib'
        Which plotting backend to use.

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objs._figure.Figure
    """
    if metric not in folds_df.columns:
        raise ValueError(f"Metric '{metric}' not found in folds_df")

    df = folds_df.copy()
    df["label"] = df["workflow"] + " | " + df["model"].astype(str)

    summary = (
        df.groupby("label")[metric]
        .agg(["mean", "std"])
        .sort_values("mean")
        .reset_index()
    )

    if engine == "matplotlib":
        return _plot_cv_results_mpl(summary, metric)
    elif engine == "plotly":
        return _plot_cv_results_plotly(summary, metric)
    else:
        raise ValueError("engine must be 'matplotlib' or 'plotly'")


def _plot_cv_results_mpl(summary, metric):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(summary) + 2))
    ax.errorbar(
        summary["mean"], summary["label"], xerr=summary["std"], fmt="o",
        color=PALETTE[0], ecolor="lightgray", elinewidth=3, capsize=4,
    )
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("Model")
    ax.set_title(f"Cross-validation Results ({metric.upper()})")
    plt.tight_layout()
    return fig


def _plot_cv_results_plotly(summary, metric):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary["mean"], y=summary["label"], orientation="h",
        error_x=dict(type="data", array=summary["std"], visible=True, color="lightgray", thickness=2, width=4),
        marker=dict(color=PALETTE[0]), opacity=0.85, name="CV Results"
    ))
    fig.update_layout(
        height=50 * len(summary) + 200, width=800,
        title=f"Cross-validation Results ({metric.upper()})",
        xaxis_title=metric.upper(), yaxis_title="Model",
    )
    return apply_theme(fig, "fa")
