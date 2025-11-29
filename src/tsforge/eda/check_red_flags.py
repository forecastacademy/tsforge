import numpy as np
import pandas as pd


def check_red_flags(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    target_col: str,
    horizon: int = 30
):
    """
    Run red flag checks for forecastability issues.
    """
    print("🚩 Running Red Flag Checks...\n")
    
    results = []

    for uid, g in df.groupby(id_col):
        series = g.sort_values(date_col)[target_col].fillna(0).to_numpy()

        flags = {}

        # Outliers
        if len(series) > 2 and series.std() > 0:
            zscores = (series - series.mean()) / series.std()
            pct_outliers = (np.abs(zscores) > 3).mean() * 100
            flags["outliers"] = pct_outliers > 2  # >2% flagged
        else:
            flags["outliers"] = False

        # Intermittency
        pct_zeros = (series == 0).mean() * 100
        flags["intermittent"] = pct_zeros > 30

        # Structural break (simple: compare first vs last half mean)
        if len(series) > 20:
            first, second = series[:len(series)//2], series[len(series)//2:]
            if first.mean() > 0 and abs(second.mean() - first.mean())/first.mean() > 0.5:
                flags["structural_break"] = True
            else:
                flags["structural_break"] = False
        else:
            flags["structural_break"] = False

        # Short history
        flags["short_history"] = len(series) < 2*horizon

        # Constant
        flags["constant"] = series.std() < 1e-6

        results.append((uid, flags))

    # Print summary
    for uid, flags in results:
        print(f"Series: {uid}")
        for k,v in flags.items():
            status = "🚩 FLAGGED" if v else "✅ OK"
            print(f"  {k:20} {status}")
        print()

    print("🚩 Red flag check complete.\n")
