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
    target_col: str = None,  # Optional target for seasonal analysis
) -> pd.DataFrame:
    """Comprehensive time series diagnostics per unique_id for forecasting.

    Returns key information about temporal structure, gaps, and frequency patterns
    that inform forecasting approach and data quality.

    If target_col is provided, also returns seasonal pattern information.
    """

    def infer_frequency(dates):
        """Infer frequency and return as string"""
        freq = pd.infer_freq(dates.sort_values())
        return freq if freq else "irregular"

    def count_gaps(dates):
        """Count number of gaps in expected regular sequence"""
        dates_sorted = dates.sort_values()
        if len(dates_sorted) < 2:
            return 0
        freq = pd.infer_freq(dates_sorted)
        if freq is None:
            return np.nan
        expected_range = pd.date_range(dates_sorted.min(), dates_sorted.max(), freq=freq)
        return len(expected_range) - len(dates_sorted)

    def span_days(dates):
        """Total time span in days"""
        return (dates.max() - dates.min()).total_seconds() / 86400

    def obs_per_year(dates):
        """Approximate observations per year"""
        span = (dates.max() - dates.min()).total_seconds() / 86400
        if span == 0:
            return np.nan
        return (len(dates) / span) * 365.25

    def has_duplicates(dates):
        """Check if there are duplicate timestamps"""
        return dates.duplicated().any()

    # Base aggregations that always run
    base_agg = {
        # Basic temporal boundaries
        "start_date": (date_col, "min"),
        "end_date": (date_col, "max"),
        "n_obs": (date_col, "count"),
        "span_days": (date_col, span_days),
        # Frequency detection
        "inferred_freq": (date_col, infer_frequency),
        "obs_per_year": (date_col, obs_per_year),
        # Gap analysis
        "n_gaps": (date_col, count_gaps),
        "pct_missing": (date_col, pct_missing_dates),
        # Data quality flags
        "has_duplicates": (date_col, has_duplicates),
    }
    result = df.groupby(id_col).agg(**base_agg)

    if target_col is not None:

        def _seasonal_summary(group):
            if group.empty or group[target_col].dropna().empty:
                return pd.Series({"peak_month": np.nan, "peak_quarter": np.nan})

            month_means = group.groupby(group[date_col].dt.month)[target_col].mean()
            quarter_means = group.groupby(group[date_col].dt.quarter)[target_col].mean()

            return pd.Series(
                {
                    "peak_month": month_means.idxmax() if not month_means.empty else np.nan,
                    "peak_quarter": quarter_means.idxmax() if not quarter_means.empty else np.nan,
                }
            )

        seasonal_stats = df.groupby(id_col).apply(_seasonal_summary)
        result = result.merge(seasonal_stats, left_index=True, right_index=True, how="left")
    return result
