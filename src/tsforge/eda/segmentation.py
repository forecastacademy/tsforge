"""ABC segmentation and portfolio classification utilities."""

import numpy as np
import pandas as pd


def abc_classification(
    df: pd.DataFrame,
    metric: str = 'volume',
    a_pct: float = 0.80,
    b_pct: float = 0.95,
    price_col: str = 'sell_price',
    id_col: str = 'unique_id',
    value_col: str = 'y',
) -> pd.DataFrame:
    """
    Compute ABC classification based on volume or revenue.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data with id and value columns.
    metric : str, default 'volume'
        Classification metric: 'volume' (units) or 'revenue' (dollars).
    a_pct : float, default 0.80
        Cumulative percentage threshold for class A (top contributors).
    b_pct : float, default 0.95
        Cumulative percentage threshold for class B (A + B combined).
    price_col : str, default 'sell_price'
        Column name for price when metric='revenue'.
    id_col : str, default 'unique_id'
        Column name for series identifier.
    value_col : str, default 'y'
        Column name for the value to aggregate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - {id_col}: series identifier
        - total_volume: sum of values per series
        - total_revenue: sum of value * price per series (if metric='revenue')
        - cum_{metric}_pct: cumulative percentage
        - abc_class: 'A', 'B', or 'C'

    Examples
    --------
    >>> # Volume-based (default)
    >>> abc_df = abc_classification(weekly_df)
    >>> portfolio_df = portfolio_df.merge(abc_df[[id_col, 'abc_class']], on=id_col)

    >>> # Revenue-based
    >>> abc_df = abc_classification(weekly_df, metric='revenue')

    >>> # Custom thresholds (70/90 split)
    >>> abc_df = abc_classification(weekly_df, a_pct=0.70, b_pct=0.90)
    """
    if metric not in ('volume', 'revenue'):
        raise ValueError(f"metric must be 'volume' or 'revenue', got '{metric}'")

    # Calculate totals per series
    if metric == 'volume':
        agg = df.groupby(id_col)[value_col].sum().reset_index()
        agg.columns = [id_col, 'total_volume']
        sort_col = 'total_volume'
    else:
        if price_col not in df.columns:
            raise ValueError(f"Column '{price_col}' not found for revenue calculation")
        df = df.copy()
        df['_revenue'] = df[value_col] * df[price_col]
        agg = df.groupby(id_col).agg(
            total_volume=(value_col, 'sum'),
            total_revenue=('_revenue', 'sum')
        ).reset_index()
        sort_col = 'total_revenue'

    # Sort by value descending
    agg = agg.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # Cumulative percentage
    total = agg[sort_col].sum()
    cum_col = f'cum_{metric}_pct'
    agg[cum_col] = agg[sort_col].cumsum() / total if total > 0 else 0

    # ABC assignment
    agg['abc_class'] = np.where(
        agg[cum_col] <= a_pct, 'A',
        np.where(agg[cum_col] <= b_pct, 'B', 'C')
    )

    return agg


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
