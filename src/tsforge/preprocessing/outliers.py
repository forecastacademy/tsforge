def flag_outliers(
    df,
    value_col: str,
    method: str = "zscore",
    cap: bool = True,
    flag_col: str = "is_outlier",
    z_thresh: float = 3.0,
    iqr_mult: float = 1.5
):
    """
    Detect and flag outliers in a numeric column.

    Parameters
    ----------
    df : DataFrame
    value_col : str
        Column with numeric values (e.g. sales).
    method : str, default 'zscore'
        'zscore' or 'iqr'
    cap : bool, default True
        Whether to cap extreme values at the threshold.
    flag_col : str
        Column name to store outlier flags.
    z_thresh : float
        Z-score threshold.
    iqr_mult : float
        IQR multiplier.

    Returns
    -------
    DataFrame with new flag_col.
    """
    x = df[value_col].copy()

    if method == "zscore":
        zscores = np.abs(stats.zscore(x.fillna(0)))
        is_outlier = zscores > z_thresh
        if cap:
            capped = np.where(is_outlier, np.sign(x) * np.mean(x), x)
            df[value_col] = capped

    elif method == "iqr":
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
        is_outlier = (x < lower) | (x > upper)
        if cap:
            df[value_col] = np.clip(x, lower, upper)

    else:
        raise ValueError("method must be 'zscore' or 'iqr'")

    df[flag_col] = is_outlier.astype(int)
    return df
