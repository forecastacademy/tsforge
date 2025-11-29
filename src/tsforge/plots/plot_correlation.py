
from typing import Union, List, Optional
import re
from tsforge.plots.style import _apply_tsforge_style, PALETTE, HIGHLIGHT
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def plot_autocorrelation(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    ids: Union[None, int, str, List[str]] = None,
    lags: Union[int, List[int], range, slice, str] = 60,
    freq: str = "D",
    mode: str = "overlay",    # 'overlay' | 'facet' | 'dropdown'
    alpha: Optional[float] = 0.05,
    group_col: Union[str, List[str], None] = None,
    agg: str = "sum",
):
    """
    Plot ACF & PACF with Plotly, side-by-side, with optional confidence bands.

    Parameters
    ----------
    ids : None | int | str | list
        - None: sample up to 6 series.
        - int: sample that many series.
        - str: single series.
        - list[str]: explicit list of series.
    group_col : str or list, optional
        Aggregate series before computing ACF/PACF (e.g., roll SKUs up to category).
        ⚠️ For global models, ACF/PACF are most diagnostic at the lowest level.
        Aggregation smooths series and can mask correlations.
    agg : str, default 'sum'
        Aggregation function if grouping.
    """
    # lazy imports
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from statsmodels.tsa.stattools import acf, pacf
    

    df = df.copy()

    # Handle group_col (same logic as plot_series)
    if group_col:
        if isinstance(group_col, str):
            group_keys = [group_col, date_col]
            df = df.groupby(group_keys, observed=True)[value_col].agg(agg).reset_index()
            id_col = group_col
        else:
            group_keys = group_col + [date_col]
            df = df.groupby(group_keys,observed=True)[value_col].agg(agg).reset_index()
            df["_group_id"] = df[group_col].astype(str).agg("|".join, axis=1)
            id_col = "_group_id"

    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Normalize ids
    unique_ids = df[id_col].dropna().unique().tolist()
    if ids is None:
        n_series = min(6, len(unique_ids))
        ids = pd.Series(unique_ids).sample(n_series, random_state=42).tolist()
    elif isinstance(ids, int):
        n_series = min(ids, len(unique_ids))
        ids = pd.Series(unique_ids).sample(n_series, random_state=42).tolist()
    elif isinstance(ids, str):
        ids = [ids]; n_series = 1
    else:
        ids = list(ids); n_series = len(ids)

    # Normalize lags
    lag_idx = np.unique(_normalize_lags(lags))
    seasonal_lags = set(_seasonal_lags_for_freq(freq))

    # Compute ACF/PACF
    results = []
    for uid in ids:
        sub = df.loc[df[id_col] == uid, [date_col, value_col]].dropna().sort_values(date_col)
        vals = sub[value_col].to_numpy()
        if vals.size < 3:
            continue
        max_lag = min(vals.size - 1, int(lag_idx.max()))
        if alpha is not None:
            a_full, a_conf = acf(vals, nlags=max_lag, fft=True, alpha=alpha)
            p_full, p_conf = pacf(vals, nlags=max_lag, method="ywmle", alpha=alpha)
            results.append((uid, lag_idx, a_full[lag_idx], p_full[lag_idx],
                            (a_conf[lag_idx,0], a_conf[lag_idx,1]),
                            (p_conf[lag_idx,0], p_conf[lag_idx,1])))
        else:
            a_full = acf(vals, nlags=max_lag, fft=True)
            p_full = pacf(vals, nlags=max_lag, method="ywmle")
            results.append((uid, lag_idx, a_full[lag_idx], p_full[lag_idx], None, None))

    # =========================
    # FACET MODE
    # =========================
    if mode == "facet":
        n = len(results)
        titles = []
        for uid, *_ in results:
            titles += [f"{uid} - ACF", f"{uid} - PACF"]
        fig = make_subplots(rows=n, cols=2, subplot_titles=titles, horizontal_spacing=0.12)
        for i, (uid, idx, a_vals, p_vals, a_ci, p_ci) in enumerate(results, start=1):
            _add_acf_pacf_traces(fig, idx, a_vals, p_vals, a_ci, p_ci, uid, row=i)
            for s in (seasonal_lags & set(idx)):
                fig.add_vline(x=s, line_dash="dash", line_color="black", row=i, col=1)
                fig.add_vline(x=s, line_dash="dash", line_color="black", row=i, col=2)
        fig.update_layout(height=300*n, width=950, title="ACF | PACF (Faceted)")
        return _apply_tsforge_style(fig, engine="plotly", context="acf_pacf")

    # =========================
    # OVERLAY MODE
    # =========================
    if mode == "overlay":
        fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"], horizontal_spacing=0.12)
        for uid, idx, a_vals, p_vals, a_ci, p_ci in results:
            _add_acf_pacf_traces(fig, idx, a_vals, p_vals, a_ci, p_ci, uid, row=1)
        for s in seasonal_lags:
            if s in lag_idx:
                fig.add_vline(x=s, line_dash="dash", line_color="black", row=1, col=1)
                fig.add_vline(x=s, line_dash="dash", line_color="black", row=1, col=2)
        fig.update_layout(height=450, width=950, title="ACF | PACF (Overlay)")
        return _apply_tsforge_style(fig, engine="plotly", context="acf_pacf")

    # =========================
    # DROPDOWN MODE
    # =========================
    if mode == "dropdown":
        fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"], horizontal_spacing=0.12)
        for uid, idx, a_vals, p_vals, a_ci, p_ci in results:
            _add_acf_pacf_traces(fig, idx, a_vals, p_vals, a_ci, p_ci, uid, row=1, visible=False)
        for tr in fig.data:
            if tr.name.startswith(f"{results[0][0]} "): tr.visible = True
        names = [tr.name for tr in fig.data]
        buttons = []
        for uid in ids:
            visible = [(nm.startswith(f"{uid} ")) for nm in names]
            buttons.append(dict(label=str(uid),
                                method="update",
                                args=[{"visible": visible}]))
        fig.update_layout(height=450, width=950, title="ACF | PACF (Dropdown)",
                          updatemenus=[{"buttons": buttons,
                                        "direction": "down",
                                        "x": 1.05, "y": 1.15,
                                        "xanchor": "left", "yanchor": "top"}])
        return _apply_tsforge_style(fig, engine="plotly", context="acf_pacf")

    raise ValueError("mode must be 'overlay', 'facet', or 'dropdown'")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _normalize_lags(lags):
    if isinstance(lags, int): return np.arange(lags+1, dtype=int)
    if isinstance(lags, range): return np.array(list(lags), dtype=int)
    if isinstance(lags, slice):
        start = 0 if lags.start is None else int(lags.start)
        step = 1 if lags.step is None else int(lags.step)
        return np.arange(start, int(lags.stop), step, dtype=int)
    if isinstance(lags, (list, tuple, np.ndarray, pd.Series)):
        return np.array(lags, dtype=int)
    if isinstance(lags, str) and re.fullmatch(r"\[\d+:\d+\]", lags.strip()):
        a, b = map(int, lags.strip()[1:-1].split(":"))
        return np.arange(a, b+1, dtype=int)
    raise ValueError("lags must be int, list, tuple, range, slice, or '[a:b]' string")


def _seasonal_lags_for_freq(freq: str):
    f = (freq or "").upper()
    return {"D":[7,30,365], "W":[52], "M":[12], "Q":[4]}.get(f, [])


def _add_acf_pacf_traces(fig, idx, a_vals, p_vals, a_ci, p_ci, uid, row=1, visible=True):
    if a_ci is not None:
        lo, hi = a_ci
        lo_tr, hi_tr = _band_traces(idx, lo, hi, "rgba(30,144,255,0.15)", uid, "ACF", visible)
        fig.add_trace(lo_tr,row=row,col=1)
        fig.add_trace(hi_tr,row=row,col=1)
    if p_ci is not None:
        lo, hi = p_ci
        lo_tr, hi_tr = _band_traces(idx, lo, hi, "rgba(205,92,92,0.15)", uid, "PACF", visible)
        fig.add_trace(lo_tr,row=row,col=2)
        fig.add_trace(hi_tr,row=row,col=2)
    fig.add_trace(go.Bar(x=idx,y=a_vals,name=f"{uid} ACF",marker_color=PALETTE[0],opacity=0.8,visible=visible),row=row,col=1)
    fig.add_trace(go.Bar(x=idx,y=p_vals,name=f"{uid} PACF",marker_color=HIGHLIGHT,opacity=0.8,visible=visible),row=row,col=2)


def _band_traces(x, lo, hi, color_rgba, uid, kind, visible=True):
    lower = go.Scatter(x=x, y=lo, mode="lines",
                       line=dict(color=color_rgba, width=0),
                       name=f"{uid} {kind} band", showlegend=False,
                       hoverinfo="skip", visible=visible)
    upper = go.Scatter(x=x, y=hi, mode="lines",
                       line=dict(color=color_rgba, width=0),
                       fill="tonexty", fillcolor=color_rgba,
                       name=f"{uid} {kind} band", showlegend=False,
                       hoverinfo="skip", visible=visible)
    return lower, upper
