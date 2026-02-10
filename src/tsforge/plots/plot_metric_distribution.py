from typing import Literal, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from tsforge.plots.style import _apply_tsforge_style


def plot_metric_distribution(
    df: pd.DataFrame,
    metric: str = "wmape",
    models: Optional[list[str]] = None,
    anchor_model: Optional[str] = None,
    plot_type: Literal["boxplot", "histogram"] = "boxplot",
    clip_quantiles: Tuple[float, float] = (0.05, 0.95),
    show_points: bool = True,
    show_stats: bool = True,
    bins: int = 30,
    figsize: Tuple[int, int] = (900, 600),
    title: Optional[str] = None,
    color_scheme: str = "default",
) -> go.Figure:
    """
    Create visualizations comparing model performance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'model' column and metric columns, or evaluation dataframe
        from hierarchical_evaluation with 'level', 'metric', and model columns.
    metric : str
        Metric column name to plot (e.g., 'wmape', 'mase').
    models : list[str], optional
        For histogram: exactly 2 models to compare. If None, uses anchor_model
        and one other model. For boxplot: ignored (all models plotted).
    anchor_model : str, optional
        Model to highlight as baseline. Required for boxplot, optional for histogram.
    plot_type : {'boxplot', 'histogram'}
        Type of visualization to create.
    clip_quantiles : tuple[float, float]
        (lower, upper) quantiles for outlier clipping. Use (0, 1) for no clipping.
    show_points : bool
        Whether to overlay individual data points (boxplot only).
    show_stats : bool
        Whether to show statistics in plot and console output.
    bins : int
        Number of histogram bins (histogram only).
    figsize : tuple[int, int]
        (width, height) in pixels.
    title : str, optional
        Custom title (auto-generated if None).
    color_scheme : str
        'default', 'viridis', or 'red_blue'.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> # Boxplot comparing all models against anchor
    >>> fig = plot_model_comparison(
    ...     df,
    ...     metric='wmape',
    ...     anchor_model='SN52',
    ...     plot_type='boxplot'
    ... )
    >>> fig.show()

    >>> # Histogram comparing two specific models
    >>> fig = plot_model_comparison(
    ...     df,
    ...     metric='wmape',
    ...     models=['Naive', 'SN52'],
    ...     plot_type='histogram'
    ... )
    >>> fig.show()

    >>> # Histogram with anchor and automatic second model
    >>> fig = plot_model_comparison(
    ...     df,
    ...     metric='wmape',
    ...     anchor_model='SN52',
    ...     plot_type='histogram'
    ... )
    >>> fig.show()
    """
    if plot_type == "boxplot":
        if anchor_model is None:
            raise ValueError("anchor_model is required for boxplot")
        return _plot_boxplot(
            df=df,
            metric=metric,
            anchor_model=anchor_model,
            clip_quantiles=clip_quantiles,
            show_points=show_points,
            show_stats=show_stats,
            figsize=figsize,
            title=title,
            color_scheme=color_scheme,
        )
    elif plot_type == "histogram":
        return _plot_histogram(
            df=df,
            metric=metric,
            models=models,
            anchor_model=anchor_model,
            clip_quantiles=clip_quantiles,
            show_stats=show_stats,
            bins=bins,
            figsize=figsize,
            title=title,
            color_scheme=color_scheme,
        )
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}")


def _plot_boxplot(
    df: pd.DataFrame,
    metric: str,
    anchor_model: str,
    clip_quantiles: Tuple[float, float],
    show_points: bool,
    show_stats: bool,
    figsize: Tuple[int, int],
    title: Optional[str],
    color_scheme: str,
) -> go.Figure:
    """Internal function for boxplot implementation."""
    df = df.copy()

    # Create anchor flag
    df["is_anchor"] = df["model"] == anchor_model

    # Clip outliers
    if clip_quantiles:
        lower = df[metric].quantile(clip_quantiles[0])
        upper = df[metric].quantile(clip_quantiles[1])
        df[f"{metric}_clipped"] = df[metric].clip(lower=lower, upper=upper)
        plot_metric = f"{metric}_clipped"
    else:
        plot_metric = metric

    # Calculate statistics
    model_stats = df.groupby("model")[plot_metric].agg(["median", "mean", "count"]).reset_index()
    anchor_median = model_stats[model_stats["model"] == anchor_model]["median"].values[0]
    model_stats["delta_from_anchor"] = model_stats["median"] - anchor_median
    model_stats["delta_pct"] = model_stats["delta_from_anchor"] / anchor_median * 100

    # Sort models
    model_stats = model_stats.sort_values(["model"], key=lambda x: x != anchor_model).sort_values(
        "median"
    )
    model_order = model_stats["model"].tolist()
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values("model")

    # Color schemes
    color_schemes = {
        "default": {False: "#4C78A8", True: "#E45756"},
        "viridis": {False: "#440154", True: "#FDE724"},
        "red_blue": {False: "#3182bd", True: "#e6550d"},
    }
    colors = color_schemes.get(color_scheme, color_schemes["default"])

    # Create figure
    fig = go.Figure()

    # Add box plots
    for is_anchor in [False, True]:
        mask = df["is_anchor"] == is_anchor
        if not mask.any():
            continue

        fig.add_trace(
            go.Box(
                y=df.loc[mask, "model"],
                x=df.loc[mask, plot_metric],
                name="Anchor" if is_anchor else "Other Models",
                marker_color=colors[is_anchor],
                orientation="h",
                showlegend=True,
                line=dict(width=2),
                marker=dict(size=4, line=dict(width=1, color="white")) if show_points else None,
                boxpoints="outliers" if show_points else False,
            )
        )

    # Add annotations
    if show_stats:
        annotations = []
        for _, row in model_stats.iterrows():
            if row["model"] == anchor_model:
                text = f"<b>{row['median']:.3f}</b> (baseline)"
            else:
                delta_sign = "+" if row["delta_from_anchor"] > 0 else ""
                text = f"{row['median']:.3f} ({delta_sign}{row['delta_pct']:.1f}%)"

            annotations.append(
                dict(
                    x=df[df["model"] == row["model"]][plot_metric].max() * 1.02,
                    y=row["model"],
                    text=text,
                    showarrow=False,
                    xanchor="left",
                    font=dict(size=9, color="#333333"),
                )
            )
        fig.update_layout(annotations=annotations)

    # Add vertical line at anchor
    fig.add_vline(
        x=anchor_median,
        line_dash="dash",
        line_color="rgba(228, 87, 86, 0.3)",
        line_width=2,
        annotation_text=f"Anchor: {anchor_median:.3f}",
        annotation_position="top",
    )

    # Layout
    title_text = title or f"{metric.upper()} Distribution vs Anchor ({anchor_model})"
    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center", font=dict(size=16, color="#333333")),
        xaxis_title=metric.upper(),
        yaxis_title="Model",
        width=figsize[0],
        height=figsize[1],
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=100, r=150, t=80, b=60),
    )

    # Grid styling
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128, 128, 128, 0.2)",
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="rgba(128, 128, 128, 0.3)",
    )
    fig.update_yaxes(showgrid=False)

    # Print summary
    if show_stats:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {metric.upper()} vs {anchor_model}")
        print(f"{'='*60}")
        print(f"{'Model':<15} {'Median':>10} {'Delta':>10} {'Delta %':>10} {'N':>8}")
        print(f"{'-'*60}")
        for _, row in model_stats.iterrows():
            delta_sign = "+" if row["delta_from_anchor"] > 0 else ""
            print(
                f"{row['model']:<15} {row['median']:>10.4f} "
                f"{delta_sign}{row['delta_from_anchor']:>9.4f} "
                f"{delta_sign}{row['delta_pct']:>9.1f}% {int(row['count']):>8}"
            )
        print(f"{'='*60}\n")

    return _apply_tsforge_style(fig, engine="plotly")


def _plot_histogram(
    df: pd.DataFrame,
    metric: str,
    models: Optional[list[str]],
    anchor_model: Optional[str],
    clip_quantiles: Tuple[float, float],
    show_stats: bool,
    bins: int,
    figsize: Tuple[int, int],
    title: Optional[str],
    color_scheme: str,
) -> go.Figure:
    """Internal function for histogram implementation."""
    # Filter to metric
    metric_df = df[df["metric"] == metric].copy()

    if metric_df.empty:
        raise ValueError(f"No data found for metric '{metric}'")

    # Determine which models to compare
    if models is not None:
        if len(models) != 2:
            raise ValueError("Exactly 2 models required for histogram")
        model1, model2 = models
    elif anchor_model is not None:
        # Use anchor and find another model
        available_models = [col for col in metric_df.columns if col not in ["level", "metric"]]
        if anchor_model not in available_models:
            raise ValueError(f"anchor_model '{anchor_model}' not found in data")
        other_models = [m for m in available_models if m != anchor_model]
        if not other_models:
            raise ValueError("Need at least 2 models for histogram")
        model1, model2 = anchor_model, other_models[0]
    else:
        raise ValueError("Must provide either 'models' or 'anchor_model'")

    if model1 not in metric_df.columns or model2 not in metric_df.columns:
        raise ValueError(f"Models '{model1}' or '{model2}' not found in data")

    # Extract values
    values1_raw = metric_df[model1].dropna()
    values2_raw = metric_df[model2].dropna()

    # Clip outliers
    combined = pd.concat([values1_raw, values2_raw])
    lower_bound = combined.quantile(clip_quantiles[0])
    upper_bound = combined.quantile(clip_quantiles[1])

    values1 = values1_raw.clip(lower=lower_bound, upper=upper_bound)
    values2 = values2_raw.clip(lower=lower_bound, upper=upper_bound)

    # Count clipped points
    n_clipped1 = ((values1_raw < lower_bound) | (values1_raw > upper_bound)).sum()
    n_clipped2 = ((values2_raw < lower_bound) | (values2_raw > upper_bound)).sum()

    # Color scheme
    color_schemes = {
        "default": ["#4C78A8", "#E45756"],
        "viridis": ["#440154", "#FDE724"],
        "red_blue": ["#3182bd", "#e6550d"],
    }
    colors = color_schemes.get(color_scheme, color_schemes["default"])

    # Create figure
    fig = go.Figure()

    # Add histograms
    fig.add_trace(
        go.Histogram(
            x=values1,
            name=model1,
            opacity=0.7,
            marker_color=colors[0],
            nbinsx=bins,
            histnorm="probability density",
        )
    )

    fig.add_trace(
        go.Histogram(
            x=values2,
            name=model2,
            opacity=0.7,
            marker_color=colors[1],
            nbinsx=bins,
            histnorm="probability density",
        )
    )

    # Add vertical lines at medians
    fig.add_vline(
        x=values1.median(),
        line_dash="dash",
        line_color=colors[0],
        line_width=2,
        annotation_text=f"{model1}: {values1.median():.4f}",
        annotation_position="top left",
    )

    fig.add_vline(
        x=values2.median(),
        line_dash="dash",
        line_color=colors[1],
        line_width=2,
        annotation_text=f"{model2}: {values2.median():.4f}",
        annotation_position="top right",
    )

    # Generate title
    if title is None:
        clip_info = f" (clipped {clip_quantiles[0]:.0%}-{clip_quantiles[1]:.0%})"
        title = f"{metric.upper()} Distribution: {model1} vs {model2}{clip_info}"

    # Layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16)),
        xaxis_title=metric.upper(),
        yaxis_title="Density",
        height=figsize[1],
        width=figsize[0],
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x",
    )

    # Grid styling
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128, 128, 128, 0.2)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128, 128, 128, 0.2)",
    )

    # Print summary
    if show_stats:
        print(f"\n{metric.upper()} Distribution Comparison")
        print(f"{'='*60}")
        print(f"Clipping: [{lower_bound:.4f}, {upper_bound:.4f}]")
        print(
            f"Clipped points: {model1}={n_clipped1}/{len(values1_raw)}, "
            f"{model2}={n_clipped2}/{len(values2_raw)}"
        )
        print(f"{'='*60}")
        print(f"{'Metric':<20} {model1:>15} {model2:>15}")
        print(f"{'-'*60}")
        print(f"{'Mean':<20} {values1.mean():>15.4f} {values2.mean():>15.4f}")
        print(f"{'Median':<20} {values1.median():>15.4f} {values2.median():>15.4f}")
        print(f"{'Std Dev':<20} {values1.std():>15.4f} {values2.std():>15.4f}")
        print(f"{'Min':<20} {values1.min():>15.4f} {values2.min():>15.4f}")
        print(f"{'Max':<20} {values1.max():>15.4f} {values2.max():>15.4f}")
        print(f"{'N':<20} {len(values1):>15} {len(values2):>15}")

        delta_pct = (values2.median() - values1.median()) / values1.median() * 100
        print(f"\n{model2} vs {model1}: {delta_pct:+.1f}% (median)")
        print(f"{'='*60}\n")

    return _apply_tsforge_style(fig, engine="plotly")
