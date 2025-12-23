import numpy as np
import pandas as pd
from tsfeatures import *

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
