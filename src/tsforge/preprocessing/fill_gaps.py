import pandas as pd


from utilsforecast.preprocessing import fill_gaps

def fill_gaps_plus(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    target_col: str,
    static_cols: list[str] | None = None,
    dynamic_fill: dict[str, str] | None = None,
    target_fill: str = "zero",
    freq: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Enhanced gap filling that extends Nixtla's fill_gaps
    to handle static and dynamic variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with id, time, target, static, and dynamic columns.
    id_col : str
        Column with unique series identifier.
    time_col : str
        Datetime column.
    target_col : str
        Target column (e.g., sales).
    static_cols : list[str], optional
        Columns that don't change over time (e.g., category, store).
        These are copied forward into gap rows.
    dynamic_fill : dict[str, str], optional
        Dict mapping dynamic variable names to fill strategy:
          - "zero" → fill missing with 0
          - "ffill" → forward fill within id
          - "interpolate" → numeric interpolate within id
          - "nan" (default) → leave missing
    target_fill : str, default "zero"
        Fill strategy for the target:
          - "zero" → missing target values filled with 0
          - "nan" → leave missing
    freq : str, optional
        Expected pandas frequency string ("D", "W", "M").
        Must be provided unless inferred externally.
    verbose : bool, default True
        If True, prints input/output shape and number of rows added.

    Returns
    -------
    pd.DataFrame
        Expanded dataframe with gaps filled. Static variables are replicated,
        dynamic variables are merged back and optionally filled.
    """

    if verbose:
        print(f"[fill_gaps_plus] Input shape: {df.shape}")

    # Base with only target
    df_base = df[[id_col, time_col, target_col]].copy()

    # Run Nixtla fill_gaps (freq required)
    if not freq:
        raise ValueError("freq must be specified (e.g., 'D', 'W', 'M')")
    filled = fill_gaps(df_base, freq=freq, id_col=id_col, time_col=time_col)

    # Fill strategy for target
    if target_fill == "zero":
        filled[target_col] = filled[target_col].fillna(0)
    elif target_fill == "nan":
        pass
    else:
        raise ValueError("target_fill must be 'zero' or 'nan'")

    # Add static columns if they exist
    if static_cols:
        static_cols_existing = [c for c in static_cols if c in df.columns]
        if static_cols_existing:
            static_df = df[[id_col] + static_cols_existing].drop_duplicates(id_col)
            filled = filled.merge(static_df, on=id_col, how="left")

    # Add dynamic columns if they exist
    ignore_cols = [id_col, time_col, target_col] + (static_cols or [])
    dyn_cols = [c for c in df.columns if c not in ignore_cols]
    if dyn_cols:
        dyn_df = df[[id_col, time_col] + dyn_cols].copy()
        filled = filled.merge(dyn_df, on=[id_col, time_col], how="left")

        # Apply per-column fill strategies
        if dynamic_fill:
            for col, strategy in dynamic_fill.items():
                if col not in filled.columns:
                    continue
                if strategy == "zero":
                    filled[col] = filled[col].fillna(0)
                elif strategy == "ffill":
                    filled[col] = filled.groupby(id_col)[col].ffill()
                elif strategy == "interpolate":
                    filled[col] = filled.groupby(id_col)[col].transform(lambda x: x.interpolate())
                elif strategy == "nan":
                    pass
                else:
                    raise ValueError(f"Unknown fill strategy {strategy} for column {col}")

    if verbose:
        added = filled.shape[0] - df.shape[0]
        print(f"[fill_gaps_plus] Output shape: {filled.shape}")
        print(f"[fill_gaps_plus] Added {added} rows to fill gaps.")

    return filled
