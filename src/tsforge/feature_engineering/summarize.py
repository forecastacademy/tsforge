import pandas as pd
from typing import Union, List, Dict

import pandas as pd
from typing import Union, List, Dict

def summarize_by_time(
    df: pd.DataFrame,
    date_col: str,
    value_cols: Union[str, List[str]],
    freq: str = "D",
    agg_func: Union[str, List[str], Dict[str, Union[str, List[str]]]] = "sum",
    groupby_cols: Union[str, List[str], None] = None,
) -> pd.DataFrame:
    """
    Summarize (aggregate) time series data by a specified frequency.

    Parameters
    ----------
    df : DataFrame
        Input data. Must be a pandas DataFrame (not a GroupBy).
    date_col : str
        Datetime column to resample on.
    value_cols : str or list of str
        Numeric columns to aggregate.
    freq : str
        Pandas offset alias ('D','W','M','Q','Y', etc.).
    agg_func : str, list, or dict
        Aggregation(s) to apply.
    groupby_cols : str or list of str, optional
        Columns to group by in addition to time.

    Returns
    -------
    DataFrame with aggregated results.
    """

    # --- Guardrail: catch GroupBy objects ---
    if isinstance(df, pd.core.groupby.generic.DataFrameGroupBy):
        raise ValueError(
            "summarize_by_time expects a DataFrame, not a GroupBy. "
            "Pass the raw DataFrame and use `groupby_cols` instead."
        )

    if isinstance(value_cols, str):
        value_cols = [value_cols]
    if isinstance(groupby_cols, str):
        groupby_cols = [groupby_cols]

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if groupby_cols:
        out = (
            df.groupby(groupby_cols + [pd.Grouper(key=date_col, freq=freq)])[value_cols]
              .agg(agg_func)
              .reset_index()
        )
    else:
        out = (
            df.groupby(pd.Grouper(key=date_col, freq=freq))[value_cols]
              .agg(agg_func)
              .reset_index()
        )

    # Flatten MultiIndex columns if multiple aggs used
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join([c for c in tup if c]).strip("_")
            for tup in out.columns.to_flat_index()
        ]

    return out

