import pandas as pd
from typing import Literal
from tsforge.plots.style import _apply_tsforge_style, PALETTE, HIGHLIGHT


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
        The figure object for further use.
    """
    if metric not in folds_df.columns:
        raise ValueError(f"Metric '{metric}' not found in folds_df")

    df = folds_df.copy()
    df["label"] = df["workflow"] + " | " + df["model"].astype(str)

    # Aggregate mean & std
    summary = (
        df.groupby("label")[metric]
        .agg(["mean", "std"])
        .sort_values("mean")
        .reset_index()
    )

    # =========================
    # Matplotlib backend
    # =========================
    if engine == "matplotlib":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 0.5 * len(summary) + 2))

        ax.errorbar(
            summary["mean"],
            summary["label"],
            xerr=summary["std"],
            fmt="o",
            color=PALETTE[0],
            ecolor="lightgray",
            elinewidth=3,
            capsize=4,
        )

        ax.set_xlabel(metric.upper())
        ax.set_ylabel("Model")
        ax.set_title(f"Cross-validation Results ({metric.upper()})")

        plt.tight_layout()
        return _apply_tsforge_style(fig, engine="matplotlib", context="cv_results")

    # =========================
    # Plotly backend
    # =========================
    elif engine == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=summary["mean"],
            y=summary["label"],
            orientation="h",
            error_x=dict(
                type="data",
                array=summary["std"],
                visible=True,
                color="lightgray",
                thickness=2,
                width=4,
            ),
            marker=dict(color=PALETTE[0]),
            opacity=0.85,
            name="CV Results"
        ))

        fig.update_layout(
            height=50 * len(summary) + 200,
            width=800,
            title=f"Cross-validation Results ({metric.upper()})",
            xaxis_title=metric.upper(),
            yaxis_title="Model",
        )
        return _apply_tsforge_style(fig, engine="plotly", context="cv_results")

    else:
        raise ValueError("engine must be 'matplotlib' or 'plotly'")
