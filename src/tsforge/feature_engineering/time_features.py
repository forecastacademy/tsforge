import pandas as pd
import numpy as np
from typing import Callable

def step_time_features(date_col: str, prefix: str = "date") -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Add rich time/calendar-based features from a datetime column.
    Better than pytimetk::add_timeseries_signature with cyclical encodings.

    Parameters
    ----------
    date_col : str
        Column containing datetimes.
    prefix : str, default "date"
        Prefix for new feature names.

    Returns
    -------
    Function that can be added to a Recipe and applied to a DataFrame.
    """
    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        d = pd.to_datetime(df[date_col])

        # basic components
        df[f"{prefix}_year"]    = d.dt.year
        df[f"{prefix}_quarter"] = d.dt.quarter
        df[f"{prefix}_month"]   = d.dt.month
        df[f"{prefix}_week"]    = d.dt.isocalendar().week.astype(int)
        df[f"{prefix}_day"]     = d.dt.day
        df[f"{prefix}_dow"]     = d.dt.weekday
        df[f"{prefix}_doy"]     = d.dt.dayofyear

        # flags
        df[f"{prefix}_is_weekend"]      = (d.dt.weekday >= 5).astype(int)
        df[f"{prefix}_is_month_start"]  = d.dt.is_month_start.astype(int)
        df[f"{prefix}_is_month_end"]    = d.dt.is_month_end.astype(int)
        df[f"{prefix}_is_quarter_start"] = d.dt.is_quarter_start.astype(int)
        df[f"{prefix}_is_quarter_end"]   = d.dt.is_quarter_end.astype(int)
        df[f"{prefix}_is_year_start"]   = d.dt.is_year_start.astype(int)
        df[f"{prefix}_is_year_end"]     = d.dt.is_year_end.astype(int)

        # numeric time index
        df[f"{prefix}_time_index"] = (d - d.min()).dt.days

        # cyclical encodings
        df[f"{prefix}_dow_sin"]   = np.sin(2 * np.pi * d.dt.weekday / 7)
        df[f"{prefix}_dow_cos"]   = np.cos(2 * np.pi * d.dt.weekday / 7)
        df[f"{prefix}_month_sin"] = np.sin(2 * np.pi * (d.dt.month-1) / 12)
        df[f"{prefix}_month_cos"] = np.cos(2 * np.pi * (d.dt.month-1) / 12)

        return df
    return _fn
