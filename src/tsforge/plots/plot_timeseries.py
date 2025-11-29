from typing import Callable, Union, List, Optional
import pandas as pd
from tsforge.plots.style import _apply_tsforge_style, PALETTE, HIGHLIGHT


# ---------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------
def plot_timeseries(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: Union[str, Callable] = "y",
    ids: Optional[List[str]] = None,
    max_ids: int = 6,
    group_col: Union[str, List[str], None] = None,
    agg: str = "sum",
    mode: str = "overlay",         # 'overlay', 'facet', or 'dropdown'
    wrap: int = 3,                 # for facet mode
    smooth_window: Optional[int] = None,
    freq: Optional[str] = None,
    engine: str = "plotly",        # 'plotly' or 'matplotlib'
):
    """
    Plot multiple time series with optional grouping, smoothing, resampling.
    """
    # lazy imports
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    
    df = df.copy()

    # Handle group_col
    if group_col:
        if isinstance(group_col, str):
            group_keys = [group_col, date_col]
            df = df.groupby(group_keys, observed=True)[value_col].agg(agg).reset_index()
            id_col = group_col
        else:
            group_keys = group_col + [date_col]
            df = df.groupby(group_keys, observed=True)[value_col].agg(agg).reset_index()
            df["_group_id"] = df[group_col].astype(str).agg("|".join, axis=1)
            id_col = "_group_id"

    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Handle callable value_col
    if callable(value_col):
        df["_value"] = value_col(df)
        value_col = "_value"

    # Optional resample
    if freq is not None:
        df = (
            df.set_index(date_col)
              .groupby(id_col,observed=True)[value_col]
              .resample(freq).agg(agg)
              .reset_index()
        )

    # Pick IDs
    unique_ids = df[id_col].dropna().unique().tolist()

    if ids is None:
        n_series = min(6, len(unique_ids))   # default sample size
        ids = pd.Series(unique_ids).sample(n_series, random_state=42).tolist()
    elif isinstance(ids, int):
        n_series = min(ids, len(unique_ids))
        ids = pd.Series(unique_ids).sample(n_series, random_state=42).tolist()
    elif isinstance(ids, str):
        ids = [ids]
        n_series = 1
    else:
        ids = list(ids)
        n_series = len(ids)

    df_sub = df[df[id_col].isin(ids)].copy()

    # ✅ Fixed smoother (only within each series, only on valid values)
    if smooth_window and smooth_window > 1:
        df_sub = df_sub.sort_values([id_col, date_col])
        df_sub["_smooth"] = (
            df_sub.groupby(id_col, group_keys=False, observed=True)[value_col]
                  .apply(lambda x: x.where(x.notna())
                                     .rolling(smooth_window, min_periods=1)
                                     .mean())
        )

    # =========================
    # Plotly backend
    # =========================
    if engine == "plotly":
        if mode == "facet":
            fig = px.line(
                df_sub.dropna(subset=[value_col]),  # 👈 drop NaNs before plotting
                x=date_col, y=value_col,
                facet_col=id_col, facet_col_wrap=wrap,
                title="Time Series Faceted Grid",
                color_discrete_sequence=PALETTE
            )
            if smooth_window and "_smooth" in df_sub:
                fig_smooth = px.line(
                    df_sub.dropna(subset=["_smooth"]),
                    x=date_col, y="_smooth",
                    facet_col=id_col, facet_col_wrap=wrap,
                    color_discrete_sequence=[HIGHLIGHT]
                )
                for tr in fig_smooth.data:
                    tr.line.width, tr.opacity = 2, 0.9
                    fig.add_trace(tr)
            return _apply_tsforge_style(fig, engine="plotly")

        elif mode == "overlay":
            fig = go.Figure()
            for i, uid in enumerate(ids):
                sub = df_sub[df_sub[id_col] == uid].dropna(subset=[value_col])
                fig.add_trace(go.Scatter(
                    x=sub[date_col], y=sub[value_col],
                    mode="lines", name=str(uid),
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2), opacity=0.85
                ))
                if smooth_window and "_smooth" in sub:
                    sub_smooth = sub.dropna(subset=["_smooth"])
                    fig.add_trace(go.Scatter(
                        x=sub_smooth[date_col], y=sub_smooth["_smooth"],
                        mode="lines", name=f"{uid} smooth",
                        line=dict(color=HIGHLIGHT, width=2, dash="dot"), opacity=0.9
                    ))
            return _apply_tsforge_style(fig, engine="plotly")

        elif mode == "dropdown":
            fig = go.Figure()
            trace_map = []  # map series index -> trace indices

            for i, uid in enumerate(ids):
                sub = df_sub[df_sub[id_col] == uid].dropna(subset=[value_col])
                fig.add_trace(go.Scatter(
                    x=sub[date_col], y=sub[value_col],
                    mode="lines", name=str(uid), visible=False,
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2), opacity=0.85
                ))
                trace_idxs = [len(fig.data) - 1]

                if smooth_window and "_smooth" in sub:
                    sub_smooth = sub.dropna(subset=["_smooth"])
                    fig.add_trace(go.Scatter(
                        x=sub_smooth[date_col], y=sub_smooth["_smooth"],
                        mode="lines", name=f"{uid} smooth", visible=False,
                        line=dict(color=HIGHLIGHT, width=2, dash="dot"), opacity=0.9
                    ))
                    trace_idxs.append(len(fig.data) - 1)

                trace_map.append(trace_idxs)

            # Make first series visible
            for idx in trace_map[0]:
                fig.data[idx].visible = True

            # Dropdown buttons
            buttons = []
            for uid, idxs in zip(ids, trace_map):
                visible = [False] * len(fig.data)
                for j in idxs:
                    visible[j] = True
                buttons.append(dict(
                    label=str(uid),
                    method="update",
                    args=[{"visible": visible}]
                ))

            fig.update_layout(
                updatemenus=[{
                    "buttons": buttons,
                    "direction": "down",
                    "x": 1.05, "y": 1.15,
                    "xanchor": "left", "yanchor": "top"
                }]
            )
            return _apply_tsforge_style(fig, engine="plotly")

        else:
            raise ValueError("mode must be 'overlay', 'facet', or 'dropdown'")

    # =========================
    # Matplotlib backend
    # =========================
    elif engine == "matplotlib":
        if mode == "facet":
            rows = int(np.ceil(n_series / wrap))
            fig, axes = plt.subplots(rows, wrap,
                                     figsize=(5 * wrap, 3 * rows),
                                     sharex=True)
            axes = np.atleast_1d(axes).flatten()
            for i, (ax, uid) in enumerate(zip(axes, ids)):
                sub = df_sub[df_sub[id_col] == uid].dropna(subset=[value_col])
                ax.plot(sub[date_col], sub[value_col],
                        color=PALETTE[i % len(PALETTE)], alpha=0.85, lw=2)
                if smooth_window and "_smooth" in sub:
                    sub_smooth = sub.dropna(subset=["_smooth"])
                    ax.plot(sub_smooth[date_col], sub_smooth["_smooth"],
                            color=HIGHLIGHT, lw=2, ls="--", alpha=0.9)
                ax.set_title(f"{id_col}={uid}")
            for ax in axes[n_series:]:
                ax.axis("off")
            plt.tight_layout()
            return _apply_tsforge_style(fig, engine="matplotlib")

        elif mode == "overlay":
            fig, ax = plt.subplots(figsize=(10, 4))
            for i, uid in enumerate(ids):
                sub = df_sub[df_sub[id_col] == uid].dropna(subset=[value_col])
                ax.plot(sub[date_col], sub[value_col],
                        label=str(uid), color=PALETTE[i % len(PALETTE)], lw=2, alpha=0.85)
                if smooth_window and "_smooth" in sub:
                    sub_smooth = sub.dropna(subset=["_smooth"])
                    ax.plot(sub_smooth[date_col], sub_smooth["_smooth"],
                            color=HIGHLIGHT, lw=2, ls="--", alpha=0.9)
            ax.legend()
            plt.tight_layout()
            return _apply_tsforge_style(fig, engine="matplotlib")

        else:
            raise ValueError("matplotlib engine only supports 'overlay' and 'facet'")

    else:
        raise ValueError("engine must be 'plotly' or 'matplotlib'")
