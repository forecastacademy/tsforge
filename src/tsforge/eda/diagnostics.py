import numpy as np
import pandas as pd
from tsfeatures import *
from typing import List, Optional, Literal

from .ts_features_extension import (
    ADI,
    MI_top_k_lags,
    MI_top_k_lags_indices,
    hurst_exp_dfa,
    hyndman_forecastability,
    longest_zero_streak,
    lya_exp,
    monthly_MASE_score,
    number_of_leading_zeros,
    number_of_trailing_zeros,
    overdispersion,
    pct_zeros,
    permutation_entropy,
    quarterly_MASE_score,
    yearly_MASE_score,
)

# Default LD6 column groups
DEFAULT_STRUCTURE_COLS = ["trend", "seasonal_strength", "MI_top_k_lags"]
DEFAULT_CHAOS_COLS = ["permutation_entropy", "adi", "lumpiness"]


def pct_missing_dates(x):
    freq = pd.infer_freq(x)
    if not freq:
        return np.nan
    expected_idx = pd.date_range(start=x.min(), end=x.max(), freq=freq)
    return (len(expected_idx) - len(x)) / len(expected_idx) * 100


def assign_sb_quadrant(cv2: float, adi: float) -> str:
    """Classify into Syntetos-Boylan demand quadrants based on CV² and ADI."""
    if pd.isna(cv2) or pd.isna(adi):
        return 'Unknown'
    if adi <= 1.32:
        return 'Smooth' if cv2 <= 0.49 else 'Erratic'
    return 'Intermittent' if cv2 <= 0.49 else 'Lumpy'


def compute_forecastability(
    df: pd.DataFrame,
    *,
    # Column names
    structure_cols: List[str] = None,
    chaos_cols: List[str] = None,
    adi_col: str = "adi",
    n_periods_col: Optional[str] = None,
    # Preprocessing
    clip_quantile: float = 0.95,
    clip_cols: Literal["chaos", "structure", "all"] = "chaos",
    normalize: Literal["minmax", "none"] = "minmax",
    # Gate thresholds
    adi_threshold: float = 1.32,
    min_periods: int = 12,
    # Quadrant thresholds
    structure_threshold: float = 0.5,
    chaos_threshold: float = 0.5,
    # Weights (optional)
    structure_weights: Optional[List[float]] = None,
    chaos_weights: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Compute structure score, chaos score, and forecastability classification for each series.

    The Lie Detector 6 (LD6) collapses 6 anchor diagnostics into 2 composite scores:
    - **Structure Score**: Normalized average of trend, seasonality, mutual information.
      Higher = more learnable patterns.
    - **Chaos Score**: Normalized average of entropy, ADI, lumpiness.
      Higher = less trustworthy signal.

    Forecastability Lanes (gated routing logic):
    1. Short History — fewer than min_periods observations
    2. Sparse — ADI exceeds adi_threshold (intermittency overrides other metrics)
    3. Stable — high structure, low chaos
    4. Complex — high structure, high chaos
    5. Messy — low structure (everything else)

    XYZ Classification (improved ABC-XYZ from Lie Detector methodology):
    - **X**: Low Chaos (left column) — most forecastable, regardless of structure
    - **Y**: High Chaos + High Structure (top-right) — forecastable but noisy
    - **Z**: High Chaos + Low Structure (bottom-right) — least forecastable

    Key insight: Chaos determines IF you can forecast. Structure determines HOW.
    Low chaos = forecastable, regardless of structure level.

    This replaces traditional CV-based XYZ. CV measures variation, but high variation ≠ 
    not forecastable. A series with strong seasonality has high CV but is highly forecastable.
    The LD6-based XYZ captures actual forecastability.

    Parameters
    ----------
    df : pd.DataFrame
        Diagnostics table with one row per series. Must contain the LD6 metric columns.
    structure_cols : list of str, optional
        Columns for the structure score. Default: ["trend", "seasonal_strength", "MI_top_k_lags"]
    chaos_cols : list of str, optional
        Columns for the chaos score. Default: ["permutation_entropy", "adi", "lumpiness"]
    adi_col : str, default "adi"
        Column containing the ADI (average demand interval) metric.
        Used for the Sparse gate. Must also appear in chaos_cols.
    n_periods_col : str, optional
        Column containing the number of observations per series.
        If provided, enables the Short History gate. If None, the gate is skipped.
    clip_quantile : float, default 0.95
        Percentile at which to cap outliers before normalization.
    clip_cols : {"chaos", "structure", "all"}, default "chaos"
        Which metric group(s) to clip.
    normalize : {"minmax", "none"}, default "minmax"
        Normalization method. "minmax" scales each metric to [0, 1].
    adi_threshold : float, default 1.32
        ADI values above this trigger the Sparse gate.
        Based on the Syntetos-Boylan crossover point. Treat as a heuristic, not law.
    min_periods : int, default 12
        Minimum number of observations for diagnostics to be considered reliable.
    structure_threshold : float, default 0.5
        Structure score above this = "high structure" for lane assignment.
    chaos_threshold : float, default 0.5
        Chaos score above this = "high chaos" for XYZ assignment.
    structure_weights : list of float, optional
        Weights for each structure metric. Default: equal weights.
    chaos_weights : list of float, optional
        Weights for each chaos metric. Default: equal weights.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new columns:
        - structure_score (float, 0-1): Composite structure metric
        - chaos_score (float, 0-1): Composite chaos metric
        - forecastability (str): Lane label — "Short History", "Sparse", "Stable", "Complex", "Messy"
        - xyz_class (str): Improved XYZ classification — "X", "Y", or "Z"

    Examples
    --------
    >>> scores_df = compute_forecastability(diagnostics, n_periods_col="series_length")
    >>> scores_df[['unique_id', 'structure_score', 'chaos_score', 'forecastability', 'xyz_class']].head()

    Notes
    -----
    The XYZ classification is designed to replace traditional CV-based XYZ in ABC-XYZ analysis.
    Merge with ABC classes for the full ABC-XYZ matrix:
    
    >>> abc_xyz = scores_df.merge(abc_df[['unique_id', 'abc_class']], on='unique_id')
    >>> abc_xyz['abc_xyz'] = abc_xyz['abc_class'] + '-' + abc_xyz['xyz_class']
    """
    structure_cols = structure_cols or DEFAULT_STRUCTURE_COLS
    chaos_cols = chaos_cols or DEFAULT_CHAOS_COLS

    # Validate columns exist
    missing = [c for c in structure_cols + chaos_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    if adi_col not in df.columns:
        raise ValueError(f"adi_col '{adi_col}' not found in DataFrame")

    result = df.copy()

    # --- Step 1: Clip outliers ---
    cols_to_clip = []
    if clip_cols == "chaos":
        cols_to_clip = chaos_cols
    elif clip_cols == "structure":
        cols_to_clip = structure_cols
    elif clip_cols == "all":
        cols_to_clip = structure_cols + chaos_cols

    for col in cols_to_clip:
        cap = result[col].quantile(clip_quantile)
        result[col] = result[col].clip(upper=cap)

    # --- Step 2: Normalize ---
    if normalize == "minmax":
        for col in structure_cols + chaos_cols:
            cmin = result[col].min()
            cmax = result[col].max()
            if cmax > cmin:
                result[col] = (result[col] - cmin) / (cmax - cmin)
            else:
                result[col] = 0.0

    # --- Step 3: Compute scores (weighted average) ---
    if structure_weights is not None:
        if len(structure_weights) != len(structure_cols):
            raise ValueError("structure_weights must match length of structure_cols")
        sw = np.array(structure_weights) / np.sum(structure_weights)
        result["structure_score"] = sum(
            result[col] * w for col, w in zip(structure_cols, sw)
        )
    else:
        result["structure_score"] = result[structure_cols].mean(axis=1)

    if chaos_weights is not None:
        if len(chaos_weights) != len(chaos_cols):
            raise ValueError("chaos_weights must match length of chaos_cols")
        cw = np.array(chaos_weights) / np.sum(chaos_weights)
        result["chaos_score"] = sum(
            result[col] * w for col, w in zip(chaos_cols, cw)
        )
    else:
        result["chaos_score"] = result[chaos_cols].mean(axis=1)

    # --- Step 4: Assign forecastability lanes (gated logic) ---
    # Start with everything as Messy, then override in priority order
    result["forecastability"] = "Messy"

    # Gate 1: Short History (if n_periods_col provided)
    if n_periods_col is not None:
        if n_periods_col not in df.columns:
            raise ValueError(f"n_periods_col '{n_periods_col}' not found in DataFrame")
        result.loc[df[n_periods_col] < min_periods, "forecastability"] = "Short History"

    # Gate 2: Sparse (ADI override — checked on raw/unclipped ADI from original df)
    sparse_mask = df[adi_col] > adi_threshold
    # Don't override Short History
    already_assigned = result["forecastability"] == "Short History"
    result.loc[sparse_mask & ~already_assigned, "forecastability"] = "Sparse"

    # Gate 3: Quadrant assignment for remaining series
    remaining = result["forecastability"] == "Messy"

    high_structure = result["structure_score"] >= structure_threshold
    high_chaos = result["chaos_score"] >= chaos_threshold

    result.loc[remaining & high_structure & ~high_chaos, "forecastability"] = "Stable"
    result.loc[remaining & high_structure & high_chaos, "forecastability"] = "Complex"
    # Low structure stays "Messy" (already the default)

    # --- Step 5: Assign XYZ class (improved ABC-XYZ methodology) ---
    # This replaces traditional CV-based XYZ with LD6-based forecastability
    #
    # From Improved ABC-XYZ slide:
    #   X = Low Chaos (left column — both Stable and Sparse quadrants)
    #   Y = High Chaos + High Structure (top-right = Complex quadrant)
    #   Z = High Chaos + Low Structure (bottom-right = Messy quadrant)
    #
    # Key insight: Low chaos = forecastable, regardless of structure level.
    # Structure determines *how* to forecast, chaos determines *if* you can.
    
    result["xyz_class"] = "Z"  # Default: high chaos + low structure
    
    # X: Low Chaos (left column — forecastable)
    result.loc[result["chaos_score"] < chaos_threshold, "xyz_class"] = "X"
    
    # Y: High Chaos + High Structure (top-right — forecastable but noisy)
    result.loc[
        (result["chaos_score"] >= chaos_threshold) & 
        (result["structure_score"] >= structure_threshold),
        "xyz_class"
    ] = "Y"
    
    # Z: High Chaos + Low Structure (bottom-right — hardest to forecast)
    # Already default

    return result


TSFORGE_FEATURES = [
    # BASE TSFEATURES
    acf_features,
    arch_stat,
    crossing_points,
    entropy,
    flat_spots,
    heterogeneity,
    holt_parameters,
    lumpiness,
    nonlinearity,
    pacf_features,
    stl_features,
    stability,
    statistics,
    hw_parameters,
    unitroot_kpss,
    unitroot_pp,
    series_length,
    # NEW FEATURES
    ADI,  # average interval duration
    hurst_exp_dfa,  # hurst exponent of DFA
    lya_exp,  # lyapunov exponent
    longest_zero_streak,  # longest streak of consecutive zeros
    number_of_leading_zeros,  # number of leading zeros
    number_of_trailing_zeros,  # number of trailing zeros
    hyndman_forecastability,  # hyndman forecastability
    monthly_MASE_score,  # monthly MASE score
    yearly_MASE_score,  # yearly MASE score
    quarterly_MASE_score,  # quarterly MASE score
    overdispersion,  # overdispersion
    pct_zeros,  # percentage of zeros in the series, it was included in the original function
    MI_top_k_lags,  # SUM(top 5 MI scores of predictive lags) / SUM(all MI scores)
    MI_top_k_lags_indices,  # top 5 predictive lags sorted by MI
    permutation_entropy,  # normalized permutation entropy
]


def infer_freq_multi_id(df, id_col, date_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    sub = df.loc[df[id_col] == df[id_col].sample(1).values[0]]

    try:
        return pd.infer_freq(sub[date_col])
    except Exception as e:
        print(e)
        return np.nan


def hierarchical_tsfeatures(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    target_col: str,
    hierarchy: list,
    features: list,
    freq: int,
) -> pd.DataFrame:
    """
    A wrapper for tsfeatures that groups by id and applies tsfeatures to each group in a given hierarchy!
    """

    levels = list(set(hierarchy + [id_col]))
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    pandas_freq = infer_freq_multi_id(df, id_col, date_col)

    dfs = []
    for level in levels:
        hier_df = (
            df.groupby(level).resample(pandas_freq, on=date_col)[[target_col]].sum().reset_index()
        )

        col_mapper = {level: "unique_id", date_col: "ds", target_col: "y"}
        reverse_mapper = {v: k for k, v in col_mapper.items()}
        hier_df.rename(columns=col_mapper, inplace=True)

        ts_feats = tsfeatures(
            ts=hier_df,
            freq=freq,
            scale=False,
            features=features,
        )
        dfs.append(ts_feats.rename(columns=reverse_mapper).assign(hier_id=level))

    agg_df = pd.concat(dfs)
    for level in levels:
        if level != id_col:
            agg_df["hier_id"] = agg_df[id_col].fillna(agg_df[level])

    agg_df.drop(columns=levels, inplace=True)

    columns = [
        "hier_id",
        "lumpiness",
        "permutation_entropy",
        "MI_top_k_lags",
        "MI_top_k_lags_indices",
        "trend_strength",
        "seasonal_strength",
        "adi",
    ] + [c for c in agg_df.columns if c != "hier_id"]

    agg_df = agg_df.reindex(columns=columns)
    return agg_df


def datetime_diagnostics(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    target_col: str = None,
) -> pd.DataFrame:
    """Datetime diagnostics - timeline quality and structure.
    
    Returns information about temporal structure, gaps, and frequency patterns.
    
    If target_col is provided, also returns seasonal peak patterns (peak_month, peak_quarter).
    """
    
    # Convert to datetime once
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([id_col, date_col])
    
    # ============================================
    # BASIC DATE METRICS
    # ============================================
    
    basic_agg = {
        'start_date': (date_col, 'min'),
        'end_date': (date_col, 'max'),
        'n_obs': (date_col, 'count'),
    }
    result = df.groupby(id_col, sort=False).agg(**basic_agg)
    
    # Span and frequency
    result['span_days'] = (result['end_date'] - result['start_date']).dt.total_seconds() / 86400
    result['obs_per_year'] = np.where(
        result['span_days'] > 0,
        (result['n_obs'] / result['span_days']) * 365.25,
        np.nan
    )
    
    # ============================================
    # DIFF STATISTICS (STREAMLINED)
    # ============================================
    
    df['_diff_days'] = df.groupby(id_col, sort=False)[date_col].diff().dt.total_seconds() / 86400
    
    # Only median, mean, stdev
    diff_agg = df.groupby(id_col, sort=False)['_diff_days'].agg([
        ('diff_median_days', 'median'),
        ('diff_mean_days', 'mean'),
        ('diff_stdev_days', 'std'),
    ])
    result = result.join(diff_agg)
    
    # ============================================
    # DUPLICATES
    # ============================================
    
    dup_counts = df.groupby([id_col, date_col], sort=False).size()
    has_dups = (dup_counts > 1).groupby(level=0).any().rename('has_duplicates')
    result = result.join(has_dups).fillna({'has_duplicates': False})
    
    # ============================================
    # FREQUENCY & GAPS
    # ============================================
    
    sample_id = df[id_col].iloc[0]
    sample_dates = df[df[id_col] == sample_id][date_col].sort_values()
    global_freq = pd.infer_freq(sample_dates)
    
    if global_freq:
        result['inferred_freq'] = global_freq
        
        try:
            freq_timedelta = pd.Timedelta(global_freq)
        except (ValueError, TypeError):
            test_range = pd.date_range(start='2020-01-01', periods=2, freq=global_freq)
            freq_timedelta = test_range[1] - test_range[0]
        
        expected_counts = ((result['end_date'] - result['start_date']) / freq_timedelta + 1).round()
        result['n_gaps'] = (expected_counts - result['n_obs']).fillna(0).astype('Int64')
        result['pct_missing'] = np.where(
            expected_counts > 0,
            (result['n_gaps'] / expected_counts * 100).round(2),
            0.0
        )
    else:
        result['inferred_freq'] = 'irregular'
        result['n_gaps'] = pd.NA
        result['pct_missing'] = np.nan
    
    # ============================================
    # SEASONAL PERIOD
    # ============================================
    
    obs_yr = result['obs_per_year']
    result['seasonal_period'] = np.select(
        [
            (obs_yr >= 360) & (obs_yr <= 370),
            (obs_yr >= 50) & (obs_yr <= 54),
            (obs_yr >= 11) & (obs_yr <= 13),
            (obs_yr >= 3) & (obs_yr <= 5),
            obs_yr.notna()
        ],
        [365, 52, 12, 4, 1],
        default=np.nan
    )
    
    # ============================================
    # SEASONAL PEAKS (IF TARGET PROVIDED)
    # ============================================
    
    if target_col is not None:
        # Extract temporal features
        df['_month'] = df[date_col].dt.month
        df['_quarter'] = df[date_col].dt.quarter
        
        # Peak month
        month_means = df.groupby([id_col, '_month'], sort=False)[target_col].mean().reset_index()
        idx_max_month = month_means.groupby(id_col, sort=False)[target_col].idxmax()
        peak_months = month_means.loc[idx_max_month].set_index(id_col)['_month']
        result['peak_month'] = peak_months
        
        # Peak quarter
        quarter_means = df.groupby([id_col, '_quarter'], sort=False)[target_col].mean().reset_index()
        idx_max_quarter = quarter_means.groupby(id_col, sort=False)[target_col].idxmax()
        peak_quarters = quarter_means.loc[idx_max_quarter].set_index(id_col)['_quarter']
        result['peak_quarter'] = peak_quarters
    
    return result.reset_index()