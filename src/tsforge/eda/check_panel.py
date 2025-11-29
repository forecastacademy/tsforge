import pandas as pd

def check_panel(df: pd.DataFrame, id_col: str, date_col: str, freq: str = "D") -> dict:
    """
    Quick readiness check for a panel time-series dataframe.
    Acts as a fast triage before running full diagnostics.

    Parameters
    ----------
    df : DataFrame
        Long-format dataframe with [id_col, date_col, value_col].
    id_col : str
        Column identifying the series (e.g. 'unique_id').
    date_col : str
        Date column.
    freq : str, default "D"
        Expected frequency ('D','W','M').

    Returns
    -------
    dict with summary counts and warnings.
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    report = {}

    # --- Duplicate check
    n_dups = df.duplicated([id_col, date_col]).sum()
    report["duplicate_rows"] = int(n_dups)

    # --- Continuity check (fast: just counts missing)
    continuity_issues = 0
    for _, g in df.groupby(id_col, sort=False):
        expected = pd.date_range(g[date_col].min(), g[date_col].max(), freq=freq)
        continuity_issues += len(expected.difference(g[date_col]))
    report["continuity_gaps"] = int(continuity_issues)

    # --- Meta info
    report["n_series"] = int(df[id_col].nunique())
    report["n_obs"] = int(len(df))
    report["date_dtype"] = str(df[date_col].dtype)

    # --- Warnings
    warnings = []
    if n_dups > 0:
        warnings.append(f"⚠️ Found {n_dups} duplicate id+date rows.")
    if continuity_issues > 0:
        warnings.append(f"⚠️ Found {continuity_issues} missing dates across series.")
    if df[date_col].dtype.kind != "M":
        warnings.append("⚠️ Date column is not datetime64[ns].")

    report["warnings"] = warnings

    return report
