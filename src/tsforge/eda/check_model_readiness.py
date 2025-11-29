import pandas as pd
import numpy as np

def check_model_readiness(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    target_col: str,
    horizon: int = 30
):
    """
    Runs a series of model readiness checks and prints results.

    Parameters
    ----------
    df : pd.DataFrame
        Input forecasting dataframe.
    id_col : str
        Unique identifier column.
    date_col : str
        Datetime column.
    target_col : str
        Target (sales) column.
    horizon : int, default=30
        Forecast horizon used for minimum history check.
    """

    print("🔎 Running Model Readiness Checks...\n")

    # Ensure datetime type
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        print("✅ Date column is datetime type")
    else:
        print("❌ Date column is not datetime type")

    # Ensure target is numeric
    if pd.api.types.is_numeric_dtype(df[target_col]):
        print("✅ Target column is numeric")
    else:
        print("❌ Target column is not numeric")

    # Unique ID + date combinations
    if df.duplicated([id_col, date_col]).any():
        print("❌ Duplicate unique_id+date combos found")
    else:
        print("✅ unique_id+date combinations are unique")

    # Duplicate rows
    if df.duplicated().any():
        print("❌ Duplicate rows found (possible join error)")
    else:
        print("✅ No duplicate rows")

    # Null target values
    n_null = df[target_col].isna().sum()
    if n_null > 0:
        print(f"❌ {n_null} null values in target column")
    else:
        print("✅ No null target values")

    # Non-negative values
    if (df[target_col] < 0).any():
        print("⚠️ Negative target values found (check if valid returns)")
    else:
        print("✅ Target values are non-negative")

    # Series regularity check (gaps)
    is_regular = True
    for uid, g in df.groupby(id_col):
        diffs = g[date_col].sort_values().diff().dropna().dt.days
        if diffs.nunique() > 1:
            is_regular = False
            break
    if is_regular:
        print("✅ Every series is regular (no missing dates)")
    else:
        print("⚠️ Some series are irregular (missing dates)")

    # Minimum history check
    too_short = (df.groupby(id_col)[date_col].count() < 2 * horizon).sum()
    if too_short > 0:
        print(f"⚠️ {too_short} series shorter than 2× horizon ({horizon})")
    else:
        print("✅ All series have sufficient history")

    # Frequency consistency
    inferred_freqs = (
        df.groupby(id_col)[date_col]
        .apply(lambda x: pd.infer_freq(x.sort_values()))
    )
    n_freqs = inferred_freqs.dropna().nunique()
    if n_freqs == 1:
        print(f"✅ Consistent frequency across IDs ({inferred_freqs.dropna().iloc[0]})")
    else:
        print("⚠️ Inconsistent frequency across IDs")

    # Static vs dynamic columns
    static_cols, dynamic_cols = [], []
    for col in df.columns:
        if col in [id_col, date_col, target_col]:
            continue
        nunique_per_id = df.groupby(id_col)[col].nunique(dropna=False)
        if (nunique_per_id <= 1).all():
            static_cols.append(col)
        else:
            dynamic_cols.append(col)

    if static_cols:
        print(f"ℹ️ Static columns (1 value per ID): {static_cols}")
    if dynamic_cols:
        print(f"ℹ️ Dynamic columns (>1 value per ID): {dynamic_cols}")

    # Memory check
    n_rows = len(df)
    if n_rows > 10_000_000:
        print(f"⚠️ Large dataset: {n_rows:,} rows (check for join explosion)")
    else:
        print(f"✅ Row count reasonable: {n_rows:,} rows")

    print("\n✅ Model readiness check complete.")


