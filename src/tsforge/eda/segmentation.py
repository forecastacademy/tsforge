"""ABC segmentation and portfolio classification utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal, Optional, Tuple

# Default Pareto thresholds (cumulative % of total)
DEFAULT_ABC_THRESHOLDS = (0.80, 0.95)


def compute_abc(
    df: pd.DataFrame,
    *,
    id_col: str = "unique_id",
    date_col: str = "ds",
    value_col: str = "y",
    method: Literal["volume", "revenue"] = "volume",
    price_col: Optional[str] = None,
    thresholds: Tuple[float, float] = DEFAULT_ABC_THRESHOLDS,
    recency: Optional[str] = None,
) -> pd.DataFrame:
    """
    Classify series into ABC classes based on cumulative volume or revenue.

    Parameters
    ----------
    df : pd.DataFrame
        Raw time series data at the id × date level.
    id_col : str, default "unique_id"
        Column identifying each series.
    date_col : str, default "ds"
        Column containing dates/timestamps. Required when using recency filter.
    value_col : str, default "y"
        Column containing unit quantities (volume method) or revenue amounts.
    method : {"volume", "revenue"}, default "volume"
        How to measure importance.
        - "volume": sum of value_col directly (units, cases, etc.)
        - "revenue": requires price_col; computes value_col × price_col per row,
          then sums per series.
    price_col : str, optional
        Column containing unit price. Required when method="revenue".
    thresholds : tuple of (float, float), default (0.80, 0.95)
        Cumulative percentage cutoffs:
        - First value: A/B boundary (default 0.80 = top 80%)
        - Second value: B/C boundary (default 0.95 = top 95%)
    recency : str, optional
        Pandas offset string to filter data to recent period before classification.
        Only rows within this window (relative to each series' max date) are used.
        Examples: "52W" (last year), "26W" (last 6 months), "13W" (last quarter).
        If None, all data is used.

    Returns
    -------
    pd.DataFrame
        One row per series with columns:
        - id_col: series identifier
        - total_volume: aggregated importance metric
        - cumulative_pct: cumulative share of total (0-1)
        - abc_class: "A", "B", or "C"
        - abc_rank: rank by volume/revenue (1 = highest)

    Examples
    --------
    >>> # Basic volume-based ABC
    >>> abc = compute_abc(weekly_df, id_col='unique_id', value_col='y')

    >>> # Revenue-based ABC
    >>> abc = compute_abc(weekly_df, method='revenue', price_col='sell_price')

    >>> # Volume-based, last 52 weeks only
    >>> abc = compute_abc(weekly_df, recency='52W')

    >>> # Custom thresholds (70/90 split)
    >>> abc = compute_abc(weekly_df, thresholds=(0.70, 0.90))
    """
    a_cutoff, b_cutoff = thresholds

    if a_cutoff >= b_cutoff:
        raise ValueError(
            f"First threshold ({a_cutoff}) must be less than second ({b_cutoff})"
        )
    if not (0 < a_cutoff < 1) or not (0 < b_cutoff < 1):
        raise ValueError("Thresholds must be between 0 and 1")

    if method == "revenue" and price_col is None:
        raise ValueError("price_col is required when method='revenue'")

    data = df.copy()

    # --- Recency filter ---
    if recency is not None:
        data[date_col] = pd.to_datetime(data[date_col])
        cutoff = data[date_col].max() - pd.tseries.frequencies.to_offset(recency)
        data = data[data[date_col] > cutoff].copy()

        if len(data) == 0:
            raise ValueError(
                f"No data remaining after recency filter '{recency}'. "
                f"Check your date range."
            )

    # --- Compute importance metric ---
    if method == "revenue":
        data["_revenue"] = data[value_col] * data[price_col]
        agg_col = "_revenue"
    else:
        agg_col = value_col

    totals = (
        data.groupby(id_col)[agg_col]
        .sum()
        .reset_index()
        .rename(columns={agg_col: "total_volume"})
    )

    # --- Sort and compute cumulative share ---
    totals = totals.sort_values("total_volume", ascending=False).reset_index(drop=True)
    totals["abc_rank"] = range(1, len(totals) + 1)
    grand_total = totals["total_volume"].sum()

    if grand_total == 0:
        raise ValueError("Total volume/revenue is zero. Check your data and filters.")

    totals["cumulative_pct"] = totals["total_volume"].cumsum() / grand_total

    # --- Assign ABC classes ---
    totals["abc_class"] = "C"
    totals.loc[totals["cumulative_pct"] <= b_cutoff, "abc_class"] = "B"
    totals.loc[totals["cumulative_pct"] <= a_cutoff, "abc_class"] = "A"

    # Edge case: first row is always A even if it alone exceeds the A cutoff
    # (a single dominant SKU is by definition your most important item)
    if len(totals) > 0:
        totals.loc[0, "abc_class"] = "A"

    return totals

def assign_archetypes(
    df: pd.DataFrame,
    structure_col: str = 'structure_score',
    chaos_col: str = 'chaos_score',
    structure_thresh: float | None = None,
    chaos_thresh: float | None = None,
) -> tuple[pd.DataFrame, float, float]:
    """
    Assign archetypes based on Structure × Chaos quadrant.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with structure and chaos score columns.
    structure_col : str, default 'structure_score'
        Column name for structure score.
    chaos_col : str, default 'chaos_score'
        Column name for chaos score.
    structure_thresh : float, optional
        Threshold for structure score. If None, uses median (data-driven split).
    chaos_thresh : float, optional
        Threshold for chaos score. If None, uses median (data-driven split).

    Returns
    -------
    tuple[pd.DataFrame, float, float]
        - DataFrame with 'archetype' column added
        - structure_thresh used
        - chaos_thresh used

    Notes
    -----
    Archetype assignments:
        - Complex: High structure, low chaos → Invest in ML
        - Messy: High structure, high chaos → Robust methods
        - Stable: Low structure, low chaos → Simple baselines
        - Low Signal: Low structure, high chaos → Aggregate up

    Examples
    --------
    >>> df, struct_thresh, chaos_thresh = assign_archetypes(scores_df)
    >>> df['archetype'].value_counts()
    """
    if structure_thresh is None:
        structure_thresh = df[structure_col].median()
    if chaos_thresh is None:
        chaos_thresh = df[chaos_col].median()

    conditions = [
        (df[structure_col] >= structure_thresh) & (df[chaos_col] < chaos_thresh),
        (df[structure_col] >= structure_thresh) & (df[chaos_col] >= chaos_thresh),
        (df[structure_col] < structure_thresh) & (df[chaos_col] < chaos_thresh),
        (df[structure_col] < structure_thresh) & (df[chaos_col] >= chaos_thresh),
    ]
    choices = ['Complex', 'Messy', 'Stable', 'Low Signal']

    df = df.copy()
    df['archetype'] = np.select(conditions, choices, default='Unknown')

    return df, structure_thresh, chaos_thresh
