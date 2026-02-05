# tsforge/plots/_preprocessing.py
"""
Consolidated preprocessing module for tsforge plots.

Merges: core/preprocess.py, core/events.py, core/anomalies.py, core/forecast.py
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Tuple, Dict


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def apply_smoothing(
    df: pd.DataFrame,
    id_col: str,
    value_col: str,
    window: Optional[int],
) -> pd.DataFrame:
    """Apply rolling mean smoothing while preserving original index alignment."""
    if window is None or window <= 1:
        return df

    df = df.sort_values([id_col, "ds"]).copy()

    smooth_vals = (
        df.groupby(id_col)[value_col]
          .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )

    df[value_col] = smooth_vals
    return df


def aggregate_by_group(
    df: pd.DataFrame,
    group_col: Optional[Union[str, List[str]]],
    date_col: str,
    value_col: str,
    agg: str,
    id_col: str,
) -> Tuple[pd.DataFrame, str]:
    """Group by category/department/etc., but DO NOT lose original id_col."""
    if group_col is None:
        return df, id_col

    if isinstance(group_col, str):
        keys = [group_col, date_col]
        df2 = df.groupby(keys, observed=True)[value_col].agg(agg).reset_index()
        return df2, group_col

    keys = list(group_col) + [date_col]
    df2 = df.groupby(keys, observed=True)[value_col].agg(agg).reset_index()
    df2["_group_id"] = df2[list(group_col)].astype(str).agg("|".join, axis=1)

    return df2, "_group_id"


def resample_df(
    df: pd.DataFrame,
    freq: Optional[str],
    id_col: str,
    date_col: str,
    value_col: str,
    agg: str,
) -> pd.DataFrame:
    """Resample time series to a different frequency."""
    if freq is None:
        return df

    return (
        df.set_index(date_col)
          .groupby(id_col)[value_col]
          .resample(freq).agg(agg)
          .reset_index()
    )


def select_ids(
    df: pd.DataFrame,
    id_col: str,
    ids: Optional[Union[str, int, List[str]]],
    max_ids: int,
) -> List[str]:
    """Standard ID selection logic reused across plots."""
    unique_ids = sorted(df[id_col].unique())

    if ids is None:
        return unique_ids[:max_ids]

    if isinstance(ids, str):
        return [ids]

    if isinstance(ids, int):
        return unique_ids[:ids]

    return [i for i in ids if i in unique_ids]


def preprocess_for_plot(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str,
    group_col: Optional[Union[str, List[str]]] = None,
    agg: str = "sum",
    ids: Optional[Union[str, int, List[str]]] = None,
    max_ids: int = 6,
    freq: Optional[str] = None,
    smooth_window: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Standard preprocessing pipeline for all chart functions.

    Returns:
        Tuple of (processed_df, selected_ids, effective_id_col)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df, effective_id_col = aggregate_by_group(
        df, group_col, date_col, value_col, agg, id_col
    )

    df = resample_df(df, freq, effective_id_col, date_col, value_col, agg)

    selected_ids = select_ids(df, effective_id_col, ids, max_ids)

    df_sub = df[df[effective_id_col].isin(selected_ids)].copy()
    df_sub = df_sub.sort_values([effective_id_col, date_col])

    if smooth_window:
        df_sub = apply_smoothing(df_sub, effective_id_col, value_col, smooth_window)

    return df_sub, selected_ids, effective_id_col


# =============================================================================
# EVENT HANDLING
# =============================================================================

def extract_inline_events(
    events: Optional[str],
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    label_col: str,
) -> Optional[pd.DataFrame]:
    """Extract events from an inline column in the DataFrame."""
    if events is None:
        return None
    if events not in df.columns:
        raise ValueError(f"Inline events='{events}' not found.")

    col = df[events]

    if col.dtype == bool or pd.api.types.is_numeric_dtype(col):
        mask = col.astype(bool)
        labels = pd.Series(events, index=df.index)
    else:
        mask = col.notna()
        labels = col.astype(str)

    ev = df.loc[mask, [id_col, date_col]].copy()
    ev[label_col] = labels[mask].values
    return ev


def normalize_events_df(
    ev: Optional[pd.DataFrame],
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    label_col: str,
) -> Optional[pd.DataFrame]:
    """Normalize an events DataFrame to standard format."""
    if ev is None:
        return None

    ev = ev.copy()
    if date_col not in ev.columns:
        raise ValueError("Events DF must contain date_col.")

    ev[date_col] = pd.to_datetime(ev[date_col])

    if id_col not in ev.columns:
        uids = df[[id_col]].drop_duplicates()
        ev["_k"] = 1
        uids["_k"] = 1
        ev = ev.merge(uids, on="_k").drop(columns="_k")

    if label_col not in ev.columns:
        ev[label_col] = "event"

    return ev[[id_col, date_col, label_col]]


def merge_all_events(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    event_label_col: str,
    inline: Optional[str],
    global_events: Optional[pd.DataFrame],
    local_events: Optional[pd.DataFrame],
    direct_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Merge all event sources into a single DataFrame."""
    collected = []

    inline_df = extract_inline_events(inline, df, id_col, date_col, event_label_col)
    if inline_df is not None:
        collected.append(inline_df)

    for source in (global_events, local_events, direct_df):
        if source is not None:
            normalized = normalize_events_df(
                source, df, id_col, date_col, event_label_col
            )
            collected.append(normalized)

    if not collected:
        return None

    return (
        pd.concat(collected, ignore_index=True)
          .drop_duplicates([id_col, date_col, event_label_col])
          .sort_values([date_col, id_col])
    )


# =============================================================================
# ANOMALY HANDLING
# =============================================================================

def normalize_anomalies(
    anoms: Optional[Union[pd.DataFrame, str]],
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    flag_value: int = 1,
) -> Optional[pd.DataFrame]:
    """Normalize anomalies to a standard DataFrame format."""
    if anoms is None:
        return None

    if isinstance(anoms, str):
        if anoms not in df.columns:
            raise ValueError(f"Anomaly column '{anoms}' not found.")

        col = df[anoms]
        if col.dtype == bool or pd.api.types.is_numeric_dtype(col):
            mask = (col == flag_value)
        else:
            mask = col.astype(bool)

        return df.loc[mask, [id_col, date_col]].copy()

    an = anoms.copy()
    an[date_col] = pd.to_datetime(an[date_col])
    return an[[id_col, date_col]]


# =============================================================================
# FORECAST UTILITIES
# =============================================================================

def pi_column_names(
    forecast_value_col: str,
    level: int,
    lo_pattern: str,
    hi_pattern: str,
) -> Tuple[str, str]:
    """Return lower & upper prediction interval column names."""
    lo = lo_pattern.format(col=forecast_value_col, level=level)
    hi = hi_pattern.format(col=forecast_value_col, level=level)
    return lo, hi


# =============================================================================
# GROUP COLUMN RESOLUTION (shared by scatter, distribution, bar)
# =============================================================================

def resolve_group_col(
    df: pd.DataFrame,
    group_col: Optional[Union[str, List[str]]],
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Resolve a group_col (str or list) into a single effective column.

    If group_col is a list, creates a combined '_group' column by joining
    the string values. Returns (df, effective_col) where effective_col is
    None if group_col is None.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (copied only if a combined column is needed).
    group_col : str, list of str, or None
        Column(s) to resolve.

    Returns
    -------
    tuple of (pd.DataFrame, str or None)
        The (possibly modified) DataFrame and the effective column name.
    """
    if group_col is None:
        return df, None
    if isinstance(group_col, str):
        return df, group_col
    # List of columns — combine into a single column
    df = df.copy()
    df["_group"] = df[list(group_col)].astype(str).agg(" | ".join, axis=1)
    return df, "_group"


# =============================================================================
# QUANTILE CLIPPING (shared by scatter, distribution)
# =============================================================================

def clip_by_quantile(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    quantile: Union[float, Dict[str, Optional[float]]],
) -> pd.DataFrame:
    """
    Clip DataFrame rows to a symmetric quantile range per column.

    For a quantile of 0.95, keeps rows where column values fall within
    the [5th, 95th] percentile range.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    columns : str or list of str
        Column(s) to clip by.
    quantile : float or dict
        If float, applied to all columns. If dict, maps column names
        to quantile values (None entries are skipped).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(quantile, (int, float)):
        quantile = {c: float(quantile) for c in columns}

    for col in columns:
        q = quantile.get(col)
        if q is not None and col in df.columns:
            lower = df[col].quantile(1 - q)
            upper = df[col].quantile(q)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df