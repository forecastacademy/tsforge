# tsforge/plots/core/timeseries.py
"""
Unified time series plotter for tsforge.

Supports overlay, facet, and dropdown modes with forecasts,
prediction intervals, anomalies, and events.
"""
from __future__ import annotations

import pandas as pd
from typing import Optional, Union, List, Callable, Literal

import plotly.graph_objects as go

from .._styling import PALETTE, THEMES, hex_to_rgba
from .._preprocessing import (
    preprocess_for_plot,
    merge_all_events,
    normalize_anomalies,
    pi_column_names,
)
from .._layout import add_event_lines_and_labels
from .._display import render_by_mode


# Archetype colors for score-based filtering
ARCHETYPE_COLORS = {
    "Stable": "#2ecc71",      # Green - easy, predictable
    "Complex": "#3498db",     # Blue - opportunity for ML
    "Messy": "#e67e22",       # Orange - challenging
    "Low Signal": "#e74c3c",  # Red - difficult
}

# Archetype definitions based on structure/chaos quadrants
ARCHETYPE_QUADRANTS = {
    "Stable": {"structure": (0.0, 0.5), "chaos": (0.0, 0.5)},      # Low structure, low chaos
    "Complex": {"structure": (0.5, 1.0), "chaos": (0.0, 0.5)},     # High structure, low chaos
    "Messy": {"structure": (0.5, 1.0), "chaos": (0.5, 1.0)},       # High structure, high chaos
    "Low Signal": {"structure": (0.0, 0.5), "chaos": (0.5, 1.0)},  # Low structure, high chaos
}


def plot_timeseries(
    df: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    value_col: Union[str, Callable] = "y",
    # Grouping & selection
    group_col: Optional[Union[str, List[str]]] = None,
    agg: str = "sum",
    freq: Optional[str] = None,
    ids: Optional[Union[str, int, List[str]]] = None,
    max_ids: int = 6,
    smooth_window: Optional[int] = None,
    # Score-based filtering (archetype examples)
    scores_df: Optional[pd.DataFrame] = None,
    archetype: Optional[str] = None,
    structure_range: Optional[tuple] = None,
    chaos_range: Optional[tuple] = None,
    selection: Literal["deep", "random", "high_volume"] = "deep",
    structure_col: str = "structure_score",
    chaos_col: str = "chaos_score",
    volume_col: str = "total_volume",
    min_volume_pct: float = 0.3,
    # Forecast overlays
    forecast: Optional[pd.DataFrame] = None,
    forecast_value_col: str = "yhat",
    level: Optional[List[int]] = None,
    lo_pattern: str = "{col}-lo-{level}",
    hi_pattern: str = "{col}-hi-{level}",
    # Events
    events: Union[str, pd.DataFrame, None] = None,
    events_global: Optional[pd.DataFrame] = None,
    events_local: Optional[pd.DataFrame] = None,
    event_label_col: str = "event",
    events_config: Optional[dict] = None,
    # Anomalies
    anomalies: Union[pd.DataFrame, str, None] = None,
    anomaly_flag_value: int = 1,
    anomalies_config: Optional[dict] = None,
    # Display
    mode: Literal["overlay", "facet", "dropdown"] = "overlay",
    wrap: int = 2,
    row_height: int = 280,
    col_width: Optional[int] = None,
    vertical_spacing: float = 0.08,
    theme: str = "fa",
    style: Optional[dict] = None,
    engine: str = "plotly",
) -> go.Figure:
    """
    Unified time series plotter with optional forecast, events, and anomalies.

    Supports score-based filtering for archetype exploration (Structure × Chaos).

    Parameters
    ----------
    df : pd.DataFrame
        Input data with id, date, and value columns.
    id_col : str
        Column identifying each time series.
    date_col : str
        Column containing dates/timestamps.
    value_col : str or callable
        Column containing values, or a function that computes values from df.
    group_col : str or list, optional
        Column(s) to group by before plotting (aggregates within groups).
    agg : str, default "sum"
        Aggregation function for grouping/resampling.
    freq : str, optional
        Resample to this frequency (e.g., "W", "M").
    ids : str, int, or list, optional
        Specific series to plot. If int, plots first N series.
    max_ids : int, default 6
        Maximum number of series to plot if ids not specified.
    smooth_window : int, optional
        Rolling mean window for smoothing.

    Score-Based Filtering (Archetype Examples)
    ------------------------------------------
    scores_df : pd.DataFrame, optional
        DataFrame with scores for filtering (must have id_col + score columns).
    archetype : str, optional
        Filter by archetype: "Stable", "Complex", "Messy", "Low Signal".
    structure_range : tuple of (min, max), optional
        Filter by structure score range (overrides archetype).
    chaos_range : tuple of (min, max), optional
        Filter by chaos score range (overrides archetype).
    selection : {"deep", "random", "high_volume"}, default "deep"
        How to select examples when filtering by scores.
    structure_col, chaos_col, volume_col : str
        Column names in scores_df for filtering.
    min_volume_pct : float, default 0.3
        Minimum volume percentile for candidates.

    Forecast Overlays
    -----------------
    forecast : pd.DataFrame, optional
        Forecast data with same id_col and date_col.
    forecast_value_col : str, default "yhat"
        Column in forecast containing point predictions.
    level : list of int, optional
        Prediction interval levels (e.g., [80, 95]).
    lo_pattern, hi_pattern : str
        Column name patterns for PI bounds.

    Events & Anomalies
    ------------------
    events : str or pd.DataFrame, optional
        Events to mark. If str, column name in df. If DataFrame, event data.
    events_global : pd.DataFrame, optional
        Global events (applied to all series).
    events_local : pd.DataFrame, optional
        Local events (series-specific).
    event_label_col : str, default "event"
        Column containing event labels.
    events_config : dict, optional
        Event styling: {"color": str, "stagger_labels": bool}.
    anomalies : pd.DataFrame or str, optional
        Anomalies to mark. If str, column name in df.
    anomaly_flag_value : int, default 1
        Value indicating anomaly in a numeric column.
    anomalies_config : dict, optional
        Anomaly styling: {"color": str, "marker_symbol": str, "marker_size": int}.

    Display
    -------
    mode : {"overlay", "facet", "dropdown"}, default "overlay"
        Display mode for multiple series.
    wrap : int, default 2
        Columns for facet mode.
    row_height : int, default 280
        Pixel height per facet row.
    vertical_spacing : float, default 0.08
        Spacing between facet rows (0-1 fraction).
    theme : str, default "fa"
        Theme: "fa", "mckinsey", "minimal", "dark", "seaborn", "ggplot".
    style : dict, optional
        Style overrides: {"title", "subtitle", "x_title", "y_title"}.
    engine : str, default "plotly"
        Plotting engine (only "plotly" supported).

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> # Basic time series plot
    >>> plot_timeseries(df, id_col='unique_id', date_col='ds', value_col='y')

    >>> # Plot archetype examples (requires scores_df)
    >>> plot_timeseries(
    ...     weekly_df,
    ...     id_col='unique_id', date_col='ds', value_col='y',
    ...     scores_df=scores_df,
    ...     archetype='Complex',
    ...     max_ids=3,
    ...     mode='facet',
    ... )

    >>> # Plot by custom score ranges
    >>> plot_timeseries(
    ...     weekly_df,
    ...     id_col='unique_id', date_col='ds', value_col='y',
    ...     scores_df=scores_df,
    ...     structure_range=(0.7, 1.0),
    ...     chaos_range=(0.0, 0.3),
    ...     max_ids=5,
    ...     selection='high_volume',
    ... )
    """
    import numpy as np

    if engine.lower() != "plotly":
        raise NotImplementedError("Only Plotly engine is supported.")

    # Score-based filtering (archetype examples)
    archetype_title = None
    if scores_df is not None and (archetype is not None or structure_range is not None or chaos_range is not None):
        # Determine score ranges from archetype or direct parameters
        if structure_range is None and chaos_range is None and archetype is not None:
            if archetype not in ARCHETYPE_QUADRANTS:
                raise ValueError(
                    f"Unknown archetype '{archetype}'. "
                    f"Valid options: {list(ARCHETYPE_QUADRANTS.keys())}"
                )
            quadrant = ARCHETYPE_QUADRANTS[archetype]
            structure_range = quadrant["structure"]
            chaos_range = quadrant["chaos"]
            archetype_title = archetype

        # Default ranges if only one is specified
        if structure_range is None:
            structure_range = (0.0, 1.0)
        if chaos_range is None:
            chaos_range = (0.0, 1.0)

        # Filter scores by ranges
        filtered = scores_df[
            (scores_df[structure_col] >= structure_range[0]) &
            (scores_df[structure_col] <= structure_range[1]) &
            (scores_df[chaos_col] >= chaos_range[0]) &
            (scores_df[chaos_col] <= chaos_range[1])
        ].copy()

        if len(filtered) == 0:
            raise ValueError(
                f"No series found in structure_range={structure_range}, chaos_range={chaos_range}"
            )

        # Filter by minimum volume if volume column exists
        if volume_col in filtered.columns and min_volume_pct > 0:
            volume_thresh = filtered[volume_col].quantile(1 - min_volume_pct)
            high_volume = filtered[filtered[volume_col] >= volume_thresh]
            if len(high_volume) >= max_ids:
                filtered = high_volume

        # Select examples based on selection method
        if selection == "deep":
            target_struct = (structure_range[0] + structure_range[1]) / 2
            target_chaos = (chaos_range[0] + chaos_range[1]) / 2
            filtered["_dist_to_target"] = np.sqrt(
                (filtered[structure_col] - target_struct) ** 2 +
                (filtered[chaos_col] - target_chaos) ** 2
            )
            candidates = filtered.nsmallest(min(20, len(filtered)), "_dist_to_target")
            if volume_col in candidates.columns:
                selected = candidates.nlargest(max_ids, volume_col)
            else:
                selected = candidates.head(max_ids)
        elif selection == "random":
            selected = filtered.sample(min(max_ids, len(filtered)))
        elif selection == "high_volume":
            if volume_col not in filtered.columns:
                raise ValueError(f"volume_col '{volume_col}' not found for high_volume selection")
            selected = filtered.nlargest(max_ids, volume_col)
        else:
            raise ValueError(f"Unknown selection method: {selection}")

        # Override ids with selected series
        ids = selected[id_col].tolist()

    # Handle callable value_col
    df = df.copy()
    if callable(value_col):
        df["_value"] = value_col(df)
        value_col = "_value"

    # Preprocess data
    df_sub, selected_ids, effective_id_col = preprocess_for_plot(
        df=df,
        id_col=id_col,
        date_col=date_col,
        value_col=value_col,
        group_col=group_col,
        agg=agg,
        ids=ids,
        max_ids=max_ids,
        freq=freq,
        smooth_window=smooth_window,
    )

    # Prepare forecast
    fcst_df = None
    if forecast is not None:
        fcst_df = forecast.copy()
        fcst_df[date_col] = pd.to_datetime(fcst_df[date_col])
        fcst_df = fcst_df[fcst_df[effective_id_col].isin(selected_ids)]
        fcst_df = fcst_df.sort_values([effective_id_col, date_col])

    # Prepare anomalies
    an_df = normalize_anomalies(anomalies, df_sub, effective_id_col, date_col, anomaly_flag_value)
    if an_df is not None:
        an_df = an_df.merge(
            df_sub[[effective_id_col, date_col, value_col]],
            on=[effective_id_col, date_col],
            how="left",
        ).rename(columns={value_col: "y_anom"})

    # Prepare events
    ev_all = merge_all_events(
        df=df_sub,
        id_col=effective_id_col,
        date_col=date_col,
        event_label_col=event_label_col,
        inline=events if isinstance(events, str) else None,
        global_events=events_global,
        local_events=events_local,
        direct_df=events if isinstance(events, pd.DataFrame) else None,
    )

    # Get theme settings
    t = THEMES.get(theme, THEMES["fa"])
    ev_color = (events_config or {}).get("color", "#555")
    ev_stagger = (events_config or {}).get("stagger_labels", True)

    # Build traces for each series
    traces_by_id = {}
    for i, uid in enumerate(selected_ids):
        traces_by_id[uid] = _build_series_traces(
            df_sub=df_sub[df_sub[effective_id_col] == uid],
            fcst_df=fcst_df[fcst_df[effective_id_col] == uid] if fcst_df is not None else None,
            an_df=an_df[an_df[effective_id_col] == uid] if an_df is not None else None,
            uid=uid,
            date_col=date_col,
            value_col=value_col,
            forecast_value_col=forecast_value_col,
            level=level,
            lo_pattern=lo_pattern,
            hi_pattern=hi_pattern,
            theme_settings=t,
            anomalies_config=anomalies_config,
            color_index=i,
            n_series=len(selected_ids),
        )

    # Auto-generate title for archetype filtering
    effective_style = style.copy() if style else {}
    if archetype_title and "title" not in effective_style:
        effective_style["title"] = f"{archetype_title.upper()} Examples (n={len(selected_ids)})"
    elif structure_range is not None and chaos_range is not None and "title" not in effective_style:
        effective_style["title"] = f"Examples: Structure {structure_range}, Chaos {chaos_range}"

    # Render using unified display module
    fig = render_by_mode(
        traces_by_id,
        mode=mode,
        wrap=wrap,
        row_height=row_height,
        col_width=col_width,
        vertical_spacing=vertical_spacing,
        theme=theme,
        style=effective_style,
    )

    # Add event lines (works for all modes)
    if ev_all is not None:
        n_rows = len(selected_ids) if mode == "facet" else None
        add_event_lines_and_labels(
            fig, ev_all, date_col,
            event_label_col=event_label_col,
            ev_color=ev_color,
            stagger=ev_stagger,
            facet=(mode == "facet"),
            nrows=n_rows,
        )

    return fig


def _build_series_traces(
    df_sub: pd.DataFrame,
    fcst_df: Optional[pd.DataFrame],
    an_df: Optional[pd.DataFrame],
    uid: str,
    date_col: str,
    value_col: str,
    forecast_value_col: str,
    level: Optional[List[int]],
    lo_pattern: str,
    hi_pattern: str,
    theme_settings: dict,
    anomalies_config: Optional[dict],
    color_index: int,
    n_series: int,
) -> List[go.BaseTraceType]:
    """
    Build all traces for a single series.

    Returns list of traces: [PI bands..., actuals line, forecast line, anomaly markers]
    """
    traces = []

    # Theme settings
    line_width = theme_settings.get("line_width", 2)
    pi_opacity = theme_settings.get("pi_opacity", 0.20)
    accent_color = theme_settings.get("accent_color", "crimson")

    # Color selection
    color = PALETTE[color_index % len(PALETTE)]
    if n_series == 1 and "line_color" in theme_settings:
        color = theme_settings["line_color"]

    # Anomaly config
    an_cfg = anomalies_config or {}
    an_color = an_cfg.get("color", accent_color)
    an_symbol = an_cfg.get("marker_symbol", "x")
    an_size = an_cfg.get("marker_size", 8)

    # 1. Prediction intervals (rendered first, behind everything)
    if fcst_df is not None and level and len(fcst_df) > 0:
        for L in sorted(level, reverse=True):
            lo, hi = pi_column_names(forecast_value_col, L, lo_pattern, hi_pattern)
            if lo in fcst_df.columns and hi in fcst_df.columns:
                # Lower bound (invisible line for fill reference)
                traces.append(go.Scatter(
                    x=fcst_df[date_col],
                    y=fcst_df[lo],
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                ))
                # Upper bound with fill to lower
                traces.append(go.Scatter(
                    x=fcst_df[date_col],
                    y=fcst_df[hi],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=hex_to_rgba(color, pi_opacity),
                    hoverinfo="skip",
                    showlegend=False,
                ))

    # 2. Actuals line
    traces.append(go.Scatter(
        x=df_sub[date_col],
        y=df_sub[value_col],
        mode="lines",
        name=str(uid),
        line=dict(color=color, width=line_width),
    ))

    # 3. Forecast line
    if fcst_df is not None and len(fcst_df) > 0:
        traces.append(go.Scatter(
            x=fcst_df[date_col],
            y=fcst_df[forecast_value_col],
            mode="lines",
            name=f"{uid} forecast",
            line=dict(color=color, width=line_width, dash="dash"),
            showlegend=False,
        ))

    # 4. Anomaly markers
    if an_df is not None and len(an_df) > 0:
        traces.append(go.Scatter(
            x=an_df[date_col],
            y=an_df["y_anom"],
            mode="markers",
            name="Anomalies",
            marker=dict(color=an_color, size=an_size, symbol=an_symbol),
        ))

    return traces
