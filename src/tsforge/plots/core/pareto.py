# tsforge/plots/core/pareto.py
"""
Pareto curve visualization.

Plots cumulative share of a value across ranked categories.
Works with raw data (computes ranking internally) or pre-computed
output from compute_abc().

Optional layers (all off by default for backward compatibility):
- show_bars:       histogram of per-SKU volume on the left y-axis (single series)
- show_fill_bands: shaded vertical bands for each ABC zone (A/B/C)
- show_callout:    annotation + vertical line at a threshold crossing
- segment_curve_by_class: color curve segments by A/B/C (uses color_col)
- show_class_legend: show legend entries for A/B/C when curve is segmented

Design goal for show_bars=True (match reference screenshot):
- One light bar series (no stacking)
- ABC zones shown as background vrect bands (not bar colors)
- Cumulative curve on y2 (optionally segmented by class)
- Log y-axis on volume (y) with clean tick labels (powers of 10 only)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .._styling import PALETTE
from .._layout import finalize_figure

# Default ABC colors
ABC_COLORS = {"A": "#2ecc71", "B": "#f39c12", "C": "#e74c3c"}

# Translucent versions for background bands
ABC_BAND_COLORS = {
    "A": "rgba(46,204,113,0.08)",
    "B": "rgba(243,156,18,0.08)",
    "C": "rgba(231,76,60,0.08)",
}

# Watermark label text for each ABC class
ABC_WATERMARK_TEXT = {
    "A": "VITAL FEW",
    "B": "IMPORTANT",
    "C": "LONG TAIL",
}


def plot_pareto(
    df: pd.DataFrame,
    *,
    id_col: str,
    value_col: str,
    # Optional pre-computed columns
    cumulative_col: Optional[str] = None,
    color_col: Optional[str] = None,
    # Aggregation (when passing raw data)
    agg: str = "sum",
    # Threshold lines
    thresholds: Optional[List[float]] = None,
    threshold_labels: Optional[List[str]] = None,
    threshold_color: str = "gray",
    # Display – original
    top_n: Optional[int] = None,
    marker_size: int = 2,
    show_line: bool = True,
    # Display – new layers
    show_bars: bool = False,
    show_fill_bands: bool = False,
    show_zone_labels: bool = False,
    zone_label_style: str = "watermark",   # "watermark" | "edge"
    show_callout: bool = False,
    callout_threshold: float = 80,
    # Curve segmentation + legend
    segment_curve_by_class: bool = False,
    show_class_legend: bool = True,
    # Bars styling (single series)
    bar_opacity: float = 0.22,
    bar_color: str = "rgba(70,120,160,0.35)",  # light blue-ish
    log_y: Optional[bool] = None,
    # Default: clean log tick labels (powers of 10 only)
    clean_log_ticks: bool = True,
    # Curve styling
    curve_color: str = "#011E23",
    curve_width: float = 2.0,
    # Styling
    colors: Optional[Dict[str, str]] = None,
    theme: str = "fa",
    style: Optional[dict] = None,
) -> go.Figure:
    style = dict(style) if style else {}
    data = df.copy()

    # ------------------------------------------------------------------ #
    # Defaults: log y when bars enabled                                   #
    # ------------------------------------------------------------------ #
    if log_y is None:
        log_y = bool(show_bars)

    # ------------------------------------------------------------------ #
    # Step 1: Aggregate / rank / compute cumulative if needed             #
    # ------------------------------------------------------------------ #
    if cumulative_col is None:
        if data[id_col].duplicated().any():
            data = data.groupby(id_col)[value_col].agg(agg).reset_index()

        data = data.sort_values(value_col, ascending=False).reset_index(drop=True)

        grand_total = data[value_col].sum()
        if grand_total == 0:
            raise ValueError("Total value is zero — cannot compute Pareto curve.")
        data["_cumulative_pct"] = data[value_col].cumsum() / grand_total
        cumulative_col = "_cumulative_pct"
    else:
        data = data.sort_values(value_col, ascending=False).reset_index(drop=True)

    # 1-indexed rank
    data["_rank"] = range(1, len(data) + 1)

    if top_n is not None:
        data = data.head(top_n)

    # Cumulative as 0-100 for display
    data["_cum_pct_display"] = data[cumulative_col] * 100

    # ------------------------------------------------------------------ #
    # Step 2: Resolve class colors                                        #
    # ------------------------------------------------------------------ #
    if colors is None and color_col is not None:
        unique_classes = sorted(pd.unique(data[color_col]))
        if set(unique_classes) <= {"A", "B", "C"}:
            colors = ABC_COLORS
        else:
            colors = {cls: PALETTE[i % len(PALETTE)] for i, cls in enumerate(unique_classes)}

    # ------------------------------------------------------------------ #
    # Step 3: Auto thresholds for ABC                                     #
    # ------------------------------------------------------------------ #
    if thresholds is None and color_col is not None:
        if set(pd.unique(data[color_col])) <= {"A", "B", "C"}:
            thresholds = [80, 95]

    # ------------------------------------------------------------------ #
    # Step 4: Build figure                                                #
    # ------------------------------------------------------------------ #
    use_secondary = bool(show_bars)

    if use_secondary:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    # ------------------------------------------------------------------ #
    # 5a) Bars: single series across all SKUs (no stacking)               #
    # ------------------------------------------------------------------ #
    if show_bars:
        fig.add_trace(
            go.Bar(
                x=data["_rank"],
                y=data[value_col],
                customdata=data[[id_col]].values,
                name="Volume",
                marker_color=bar_color,
                opacity=bar_opacity,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Rank: %{x:,}<br>"
                    f"{value_col}: %{{y:,.0f}}<extra></extra>"
                ),
                showlegend=False,
            ),
            secondary_y=False,
        )
        fig.update_layout(bargap=0.0, barmode="overlay")

    # ------------------------------------------------------------------ #
    # 5b) Cumulative curve (optionally segmented by class)                #
    # ------------------------------------------------------------------ #
    scatter_mode = "lines+markers" if show_line else "markers"

    def _add_curve_segment(sub_df, color, name: str) -> None:
        tr = go.Scatter(
            x=sub_df["_rank"],
            y=sub_df["_cum_pct_display"],
            customdata=sub_df[[id_col]].values,
            mode=scatter_mode,
            name=name,
            marker=dict(color=color, size=marker_size),
            line=dict(color=color, width=curve_width) if show_line else None,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Rank: %{x:,}<br>"
                "Cumulative: %{y:.1f}%<extra></extra>"
            ),
            showlegend=False,
        )
        if use_secondary:
            fig.add_trace(tr, secondary_y=True)
        else:
            fig.add_trace(tr)

    if segment_curve_by_class and color_col is not None:
        class_order = sorted(pd.unique(data[color_col]))
        for cls in class_order:
            sub = data[data[color_col] == cls]
            if sub.empty:
                continue
            seg_color = (
                (colors.get(cls) if colors else None)
                or ABC_COLORS.get(cls, curve_color)
                or curve_color
            )
            _add_curve_segment(
                sub_df=sub,
                color=seg_color,
                name=f"Cumulative % ({cls})",
            )
    else:
        _add_curve_segment(
            sub_df=data,
            color=curve_color,
            name="Cumulative %",
        )

    # ------------------------------------------------------------------ #
    # 5c) Legend-only A/B/C entries (so legend appears when desired)      #
    # ------------------------------------------------------------------ #
    if show_class_legend and segment_curve_by_class and color_col is not None:
        for cls in sorted(pd.unique(data[color_col])):
            seg_color = (
                (colors.get(cls) if colors else None)
                or ABC_COLORS.get(cls, curve_color)
                or curve_color
            )
            dummy = go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=str(cls),
                line=dict(color=seg_color, width=curve_width),
                showlegend=True,
                hoverinfo="skip",
            )
            if use_secondary:
                fig.add_trace(dummy, secondary_y=True)
            else:
                fig.add_trace(dummy)

    # ------------------------------------------------------------------ #
    # Step 6: Background ABC fill bands (vertical vrects)                 #
    # ------------------------------------------------------------------ #
    if show_fill_bands and color_col is not None and len(data) > 0:
        class_order = sorted(pd.unique(data[color_col]))
        for cls in class_order:
            sub = data[data[color_col] == cls]
            if sub.empty:
                continue
            x0 = float(sub["_rank"].min()) - 0.5
            x1 = float(sub["_rank"].max()) + 0.5

            band_color = ABC_BAND_COLORS.get(cls, "rgba(128,128,128,0.05)")
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=band_color,
                layer="below",
                line_width=0,
            )

            label_color = colors.get(cls, "gray") if colors else "gray"
            fig.add_annotation(
                x=x1,
                y=92,
                xref="x",
                yref="y2" if use_secondary else "y",
                text=f"<b>{cls}</b>",
                font=dict(size=18, color=label_color),
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                opacity=0.55,
            )

    # ------------------------------------------------------------------ #
    # Step 6b: Watermark-style zone labels                                #
    # ------------------------------------------------------------------ #
    if show_zone_labels and color_col is not None and len(data) > 0:
        from .._styling import hex_to_rgba

        class_order = sorted(pd.unique(data[color_col]))
        for cls in class_order:
            sub = data[data[color_col] == cls]
            if sub.empty:
                continue

            x_mid = (float(sub["_rank"].min()) + float(sub["_rank"].max())) / 2
            seg_color = (colors.get(cls) if colors else None) or ABC_COLORS.get(cls, "#888")
            wm_text = ABC_WATERMARK_TEXT.get(cls, cls)

            if zone_label_style == "watermark":
                # Large letter watermark
                fig.add_annotation(
                    x=x_mid,
                    y=0.45,
                    xref="x",
                    yref="paper",
                    text=f"<b>{cls}</b>",
                    font=dict(size=48, color=hex_to_rgba(seg_color, 0.10)),
                    showarrow=False,
                    xanchor="center",
                    yanchor="middle",
                )
                # Subtitle below
                fig.add_annotation(
                    x=x_mid,
                    y=0.35,
                    xref="x",
                    yref="paper",
                    text=wm_text,
                    font=dict(size=10, family="monospace", color=hex_to_rgba(seg_color, 0.07)),
                    showarrow=False,
                    xanchor="center",
                    yanchor="middle",
                )
            else:
                # "edge" style — small label at top of zone boundary
                x_right = float(sub["_rank"].max()) + 0.5
                fig.add_annotation(
                    x=x_right,
                    y=0.96,
                    xref="x",
                    yref="paper",
                    text=f"<b>{cls}</b> {wm_text}",
                    font=dict(size=10, color=seg_color),
                    showarrow=False,
                    xanchor="right",
                    yanchor="top",
                    opacity=0.5,
                )

    # ------------------------------------------------------------------ #
    # Step 7: Threshold lines (draw on cumulative axis)                   #
    # ------------------------------------------------------------------ #
    if thresholds:
        for i, thresh in enumerate(thresholds):
            label = (
                threshold_labels[i]
                if threshold_labels and i < len(threshold_labels)
                else f"{thresh:.0f}%"
            )

            if use_secondary:
                fig.add_shape(
                    type="line",
                    xref="paper", x0=0, x1=1,
                    yref="y2", y0=thresh, y1=thresh,
                    line=dict(dash="dash", color=threshold_color, width=1),
                )
                fig.add_annotation(
                    xref="paper", x=1.0,
                    yref="y2", y=thresh,
                    text=label,
                    font=dict(size=10, color=threshold_color),
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                )
            else:
                fig.add_hline(
                    y=thresh,
                    line_dash="dash",
                    line_color=threshold_color,
                    line_width=1,
                    annotation_text=label,
                    annotation_position="right",
                    annotation_font_size=10,
                    annotation_font_color=threshold_color,
                )

    # ------------------------------------------------------------------ #
    # Step 8: Callout at threshold crossing                               #
    # ------------------------------------------------------------------ #
    if show_callout:
        target = float(callout_threshold)
        crossed = data[data["_cum_pct_display"] >= target]
        if not crossed.empty:
            cross_rank = int(crossed["_rank"].iloc[0])

            callout_color = "#f39c12"
            if colors and "B" in colors:
                callout_color = colors["B"]

            fig.add_vline(x=cross_rank, line_color=callout_color, line_width=2)

            fig.add_annotation(
                x=cross_rank,
                y=target,
                xref="x",
                yref="y2" if use_secondary else "y",
                text=f"<b>{cross_rank:,} SKUs → {target:.0f}% of volume</b>",
                showarrow=True,
                arrowhead=0,
                arrowcolor=callout_color,
                ax=-80,
                ay=-40,
                font=dict(size=12, color=callout_color),
            )

    # ------------------------------------------------------------------ #
    # Step 9: Axes + layout                                               #
    # ------------------------------------------------------------------ #
    x_title = style.get("x_title", "SKUs (ranked by volume)") if style else "SKUs (ranked by volume)"

    if use_secondary:
        left_title = style.get("y_title_left", style.get("y_title", "Volume")) if style else "Volume"
        right_title = style.get("y_title_right", "Cumulative %") if style else "Cumulative %"

        fig.update_yaxes(
            title_text=left_title,
            type="log" if log_y else "linear",
            secondary_y=False,
        )

        # suppress "double" log tick labels by showing only powers of 10
        if log_y and clean_log_ticks:
            fig.update_yaxes(secondary_y=False, dtick=1)

        fig.update_yaxes(
            title_text=right_title,
            range=[0, 102],
            secondary_y=True,
        )

        # Legend only when it's meaningful (A/B/C)
        fig.update_layout(showlegend=bool(show_class_legend and segment_curve_by_class and color_col is not None))
    else:
        fig.update_yaxes(
            title_text=style.get("y_title", "Cumulative Volume (%)") if style else "Cumulative Volume (%)",
            range=[0, 102],
        )
        fig.update_layout(showlegend=False)

    fig.update_xaxes(title_text=x_title)

    return finalize_figure(fig, theme, style)