# tsforge/plots/charts/timeseries.py
"""
Full-feature Forecast Academy-grade time-series plot.

Supports overlay, facet, and dropdown modes with forecasts,
prediction intervals, anomalies, and events.
"""
from __future__ import annotations

import pandas as pd
from typing import Optional, Union, List, Callable

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .._styling import PALETTE, HIGHLIGHT, hex_to_rgba, THEMES, apply_theme, apply_legend
from .._preprocessing import (
    apply_smoothing,
    aggregate_by_group,
    resample_df,
    select_ids,
    merge_all_events,
    normalize_anomalies,
    pi_column_names,
)
from .._layout import finalize_figure, build_dropdown_buttons, add_event_lines_and_labels


def plot_timeseries(
    df: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    value_col: Union[str, Callable] = "y",
    group_col: Optional[Union[str, List[str]]] = None,
    agg: str = "sum",
    freq: Optional[str] = None,
    ids: Optional[Union[str, int, List[str]]] = None,
    max_ids: int = 6,
    smooth_window: Optional[int] = None,
    forecast: Optional[pd.DataFrame] = None,
    forecast_value_col: str = "yhat",
    level: Optional[List[int]] = None,
    lo_pattern: str = "{col}-lo-{level}",
    hi_pattern: str = "{col}-hi-{level}",
    events: Union[str, pd.DataFrame, None] = None,
    events_global: Optional[pd.DataFrame] = None,
    events_local: Optional[pd.DataFrame] = None,
    event_label_col: str = "event",
    events_config: Optional[dict] = None,
    anomalies: Union[pd.DataFrame, str, None] = None,
    anomaly_flag_value: int = 1,
    anomalies_config: Optional[dict] = None,
    mode: str = "overlay",
    wrap: int = 3,
    theme: str = "fa",
    style: Optional[dict] = None,
    engine: str = "plotly",
):
    """
    Full-feature Forecast Academy-grade time-series plot.

    Supports:
      - overlay (stacked series)
      - facet (small multiples)
      - dropdown (interactive selector)

    Features:
      - actuals, smoothing, forecast, prediction intervals
      - anomalies, inline/global/local events + labels
      - grouping & resampling
      - themes (FA default, McKinsey, Minimal, Dark)
    """
    if engine.lower() != "plotly":
        raise NotImplementedError("Only Plotly engine is supported in FA v1.")

    # Preprocessing
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if callable(value_col):
        df["_value"] = value_col(df)
        value_col = "_value"

    df, id_col = aggregate_by_group(df, group_col, date_col, value_col, agg, id_col)
    df = resample_df(df, freq, id_col, date_col, value_col, agg)
    ids = select_ids(df, id_col, ids, max_ids)

    df_sub = df[df[id_col].isin(ids)].copy()
    df_sub = df_sub.sort_values([id_col, date_col])

    # Align forecast
    fcst_df = None
    if forecast is not None:
        fcst_df = forecast.copy()
        fcst_df[date_col] = pd.to_datetime(fcst_df[date_col])
        fcst_df = fcst_df[fcst_df[id_col].isin(ids)]
        fcst_df = fcst_df.sort_values([id_col, date_col])

    # Smoothing
    df_sub = apply_smoothing(df_sub, id_col, value_col, smooth_window)

    # Events + anomalies
    ev_all = merge_all_events(
        df=df_sub,
        id_col=id_col,
        date_col=date_col,
        event_label_col=event_label_col,
        inline=events if isinstance(events, str) else None,
        global_events=events_global,
        local_events=events_local,
        direct_df=events if isinstance(events, pd.DataFrame) else None,
    )

    an_df = normalize_anomalies(anomalies, df_sub, id_col, date_col, anomaly_flag_value)
    if an_df is not None:
        an_df = (
            an_df.merge(
                df_sub[[id_col, date_col, value_col]],
                on=[id_col, date_col],
                how="left",
            )
            .rename(columns={value_col: "y_anom"})
        )

    # Select mode
    if mode == "overlay":
        fig = _plot_overlay(
            df_sub, fcst_df, ids, id_col, date_col, value_col,
            forecast_value_col, level, lo_pattern, hi_pattern,
            ev_all, an_df, events_config, anomalies_config, theme,
        )
    elif mode == "facet":
        fig = _plot_facet(
            df_sub, fcst_df, ids, id_col, date_col, value_col,
            forecast_value_col, level, lo_pattern, hi_pattern,
            ev_all, an_df, events_config, anomalies_config, wrap, theme,
        )
    elif mode == "dropdown":
        fig = _plot_dropdown(
            df_sub, fcst_df, ids, id_col, date_col, value_col,
            forecast_value_col, level, lo_pattern, hi_pattern,
            ev_all, an_df, events_config, anomalies_config, theme,
        )
    else:
        raise ValueError("mode must be one of: overlay, facet, dropdown")

    return finalize_figure(fig, theme, style)


def _plot_overlay(
    df_sub, fcst_df, ids, id_col, date_col, value_col,
    forecast_value_col, level, lo_pattern, hi_pattern,
    ev_all, an_df, events_config, anomalies_config, theme,
):
    fig = go.Figure()

    t = THEMES.get(theme, THEMES["fa"])
    line_width = t.get("line_width", 2)
    pi_opacity = t.get("pi_opacity", 0.20)
    pi_color_default = t.get("pi_color", None)
    accent_color = t.get("accent_color", "crimson")

    anomaly_legend_shown = False
    ev_color = events_config.get("color", "#555") if events_config else "#555"
    ev_stagger = events_config.get("stagger_labels", True) if events_config else True
    an_color = anomalies_config.get("color", accent_color) if anomalies_config else accent_color
    an_symbol = anomalies_config.get("marker_symbol", "x") if anomalies_config else "x"
    an_size = anomalies_config.get("marker_size", 8) if anomalies_config else 8

    for i, uid in enumerate(ids):
        color = PALETTE[i % len(PALETTE)]
        if len(ids) == 1 and "line_color" in t:
            color = t["line_color"]

        sub = df_sub[df_sub[id_col] == uid]
        fsub = fcst_df[fcst_df[id_col] == uid] if fcst_df is not None else None
        an_sub = an_df[an_df[id_col] == uid] if an_df is not None else None

        # Prediction intervals
        if fsub is not None and level:
            for L in sorted(level, reverse=True):
                lo, hi = pi_column_names(forecast_value_col, L, lo_pattern, hi_pattern)
                if lo in fsub.columns and hi in fsub.columns:
                    fig.add_trace(go.Scatter(
                        x=fsub[date_col], y=fsub[lo],
                        mode="lines", line=dict(width=0),
                        hoverinfo="skip", showlegend=False,
                    ))
                    fillcolor = pi_color_default or hex_to_rgba(color, pi_opacity)
                    fig.add_trace(go.Scatter(
                        x=fsub[date_col], y=fsub[hi],
                        mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor=fillcolor,
                        hoverinfo="skip", showlegend=False,
                    ))

        # Actuals
        fig.add_trace(go.Scatter(
            x=sub[date_col], y=sub[value_col],
            mode="lines", name=str(uid),
            line=dict(color=color, width=line_width),
        ))

        # Forecast
        if fsub is not None:
            fig.add_trace(go.Scatter(
                x=fsub[date_col], y=fsub[forecast_value_col],
                mode="lines",
                line=dict(color=color, width=line_width, dash="dash"),
                name=f"{uid} forecast", showlegend=False,
            ))

        # Anomalies
        if an_sub is not None and len(an_sub) > 0:
            fig.add_trace(go.Scatter(
                x=an_sub[date_col], y=an_sub["y_anom"],
                mode="markers",
                name="Anomalies" if not anomaly_legend_shown else "",
                marker=dict(color=an_color, size=an_size, symbol=an_symbol),
                showlegend=not anomaly_legend_shown,
            ))
            anomaly_legend_shown = True

    # Events
    if ev_all is not None:
        add_event_lines_and_labels(fig, ev_all, date_col, ev_color=ev_color, stagger=ev_stagger)

    return fig


def _plot_facet(
    df_sub, fcst_df, ids, id_col, date_col, value_col,
    forecast_value_col, level, lo_pattern, hi_pattern,
    ev_all, an_df, events_config, anomalies_config, wrap, theme,
):
    n = len(ids)
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        subplot_titles=[str(uid) for uid in ids],
    )

    t = THEMES.get(theme, THEMES["fa"])
    line_width = t.get("line_width", 2)
    pi_opacity = t.get("pi_opacity", 0.20)
    pi_color_default = t.get("pi_color", None)
    accent_color = t.get("accent_color", "crimson")

    ev_color = events_config.get("color", "#555") if events_config else "#555"
    ev_stagger = events_config.get("stagger_labels", True) if events_config else True
    an_color = anomalies_config.get("color", accent_color) if anomalies_config else accent_color
    an_symbol = anomalies_config.get("marker_symbol", "x") if anomalies_config else "x"
    an_size = anomalies_config.get("marker_size", 8) if anomalies_config else 8

    anomaly_legend_shown = False

    for r, uid in enumerate(ids, start=1):
        color = t.get("line_color", PALETTE[(r - 1) % len(PALETTE)])
        sub = df_sub[df_sub[id_col] == uid]
        fsub = fcst_df[fcst_df[id_col] == uid] if fcst_df is not None else None
        an_sub = an_df[an_df[id_col] == uid] if an_df is not None else None

        # PIs
        if fsub is not None and level:
            for L in sorted(level, reverse=True):
                lo, hi = pi_column_names(forecast_value_col, L, lo_pattern, hi_pattern)
                if lo in fsub.columns and hi in fsub.columns:
                    fig.add_trace(go.Scatter(
                        x=fsub[date_col], y=fsub[lo],
                        mode="lines", line=dict(width=0),
                        hoverinfo="skip", showlegend=False,
                    ), row=r, col=1)
                    fillcolor = pi_color_default or hex_to_rgba(color, pi_opacity)
                    fig.add_trace(go.Scatter(
                        x=fsub[date_col], y=fsub[hi],
                        mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor=fillcolor,
                        hoverinfo="skip", showlegend=False,
                    ), row=r, col=1)

        # Actuals
        fig.add_trace(go.Scatter(
            x=sub[date_col], y=sub[value_col],
            mode="lines", name=str(uid),
            line=dict(color=color, width=line_width),
        ), row=r, col=1)

        # Forecast
        if fsub is not None:
            fig.add_trace(go.Scatter(
                x=fsub[date_col], y=fsub[forecast_value_col],
                mode="lines",
                line=dict(color=color, dash="dash", width=line_width),
                showlegend=False,
            ), row=r, col=1)

        # Anomalies
        if an_sub is not None and len(an_sub) > 0:
            fig.add_trace(go.Scatter(
                x=an_sub[date_col], y=an_sub["y_anom"],
                mode="markers",
                name="Anomalies" if not anomaly_legend_shown else "",
                marker=dict(color=an_color, size=an_size, symbol=an_symbol),
                showlegend=not anomaly_legend_shown,
            ), row=r, col=1)
            anomaly_legend_shown = True

    # Events
    if ev_all is not None:
        add_event_lines_and_labels(
            fig, ev_all, date_col,
            ev_color=ev_color, stagger=ev_stagger,
            facet=True, nrows=n
        )

    fig.update_layout(height=280 * n)
    return fig


def _plot_dropdown(
    df_sub, fcst_df, ids, id_col, date_col, value_col,
    forecast_value_col, level, lo_pattern, hi_pattern,
    ev_all, an_df, events_config, anomalies_config, theme,
):
    fig = go.Figure()
    trace_map = {uid: [] for uid in ids}
    anomaly_legend_shown = False

    t = THEMES.get(theme, THEMES["fa"])
    line_width = t.get("line_width", 2)
    pi_opacity = t.get("pi_opacity", 0.20)
    pi_color_default = t.get("pi_color", None)
    accent_color = t.get("accent_color", "crimson")

    ev_color = events_config.get("color", "#555") if events_config else "#555"
    ev_stagger = events_config.get("stagger_labels", True) if events_config else True
    an_color = anomalies_config.get("color", accent_color) if anomalies_config else accent_color
    an_symbol = anomalies_config.get("marker_symbol", "x") if anomalies_config else "x"
    an_size = anomalies_config.get("marker_size", 8) if anomalies_config else 8

    for i, uid in enumerate(ids):
        color = PALETTE[i % len(PALETTE)]
        if len(ids) == 1 and "line_color" in t:
            color = t["line_color"]

        visible = (i == 0)
        sub = df_sub[df_sub[id_col] == uid]
        fsub = fcst_df[fcst_df[id_col] == uid] if fcst_df is not None else None
        an_sub = an_df[an_df[id_col] == uid] if an_df is not None else None

        # PIs
        if fsub is not None and level:
            for L in sorted(level, reverse=True):
                lo, hi = pi_column_names(forecast_value_col, L, lo_pattern, hi_pattern)
                if lo in fsub.columns and hi in fsub.columns:
                    fig.add_trace(go.Scatter(
                        x=fsub[date_col], y=fsub[lo],
                        mode="lines", line=dict(width=0),
                        hoverinfo="skip", visible=visible, showlegend=False,
                    ))
                    trace_map[uid].append(len(fig.data) - 1)
                    fillcolor = pi_color_default or hex_to_rgba(color, pi_opacity)
                    fig.add_trace(go.Scatter(
                        x=fsub[date_col], y=fsub[hi],
                        mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor=fillcolor,
                        hoverinfo="skip", visible=visible, showlegend=False,
                    ))
                    trace_map[uid].append(len(fig.data) - 1)

        # Actuals
        fig.add_trace(go.Scatter(
            x=sub[date_col], y=sub[value_col],
            mode="lines",
            line=dict(color=color, width=line_width),
            visible=visible, showlegend=False,
        ))
        trace_map[uid].append(len(fig.data) - 1)

        # Forecast
        if fsub is not None:
            fig.add_trace(go.Scatter(
                x=fsub[date_col], y=fsub[forecast_value_col],
                mode="lines",
                line=dict(color=color, width=line_width, dash="dash"),
                visible=visible, showlegend=False,
            ))
            trace_map[uid].append(len(fig.data) - 1)

        # Anomalies
        if an_sub is not None and len(an_sub) > 0:
            fig.add_trace(go.Scatter(
                x=an_sub[date_col], y=an_sub["y_anom"],
                mode="markers",
                marker=dict(color=an_color, size=an_size, symbol=an_symbol),
                visible=visible,
                name="Anomalies" if not anomaly_legend_shown else "",
                showlegend=not anomaly_legend_shown,
            ))
            trace_map[uid].append(len(fig.data) - 1)
            anomaly_legend_shown = True

    # Events
    if ev_all is not None:
        add_event_lines_and_labels(fig, ev_all, date_col, ev_color=ev_color, stagger=ev_stagger)

    # Dropdown
    buttons = build_dropdown_buttons(trace_map, len(fig.data))
    fig.update_layout(
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "x": 1.0, "y": 1.15,
            "xanchor": "right",
            "yanchor": "top",
        }]
    )

    return fig
