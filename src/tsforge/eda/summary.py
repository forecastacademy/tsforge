import numpy as np
import pandas as pd

def evaluate_diagnostics(
    df: pd.DataFrame,
    metrics: list[str],
    thresholds: dict[str, float],
    group_by: str = None,
    show_cnt: bool = True,
    show_pct: bool = False,
    show_median: bool = False,
    above_is_bad: dict[str, bool] = None,
    style: bool = True,
) -> pd.DataFrame:
    """
    Summarize diagnostic metrics with discrete value counts.
    
    Parameters
    ----------
    df : DataFrame
        Diagnostics data with metrics and optional grouping columns.
    metrics : list
        Metric columns to evaluate.
    thresholds : dict
        Threshold value for each metric.
    group_by : str, optional
        Column to group by. If None, shows portfolio-level summary.
    show_cnt : bool, optional
        Show count columns (Above/Below Threshold). Default True.
    show_pct : bool, optional
        Show percentage columns. Default False.
    show_median : bool, optional
        Include median column per metric. Default False.
    above_is_bad : dict, optional
        True if above threshold = problem. Default: entropy/adi are bad when high.
    style : bool, optional
        Color code columns (green=good, red=bad). Default True.
    
    Returns
    -------
    DataFrame or Styler
        Summary table with value counts. Styled if style=True.
    """
    if above_is_bad is None:
        above_is_bad = {'entropy': True, 'adi': True, 'trend': False, 'seasonal_strength': False}
    
    groups = df.groupby(group_by) if group_by else [('Portfolio', df)]
    rows = []
    
    # Track which columns are "good" vs "bad" for styling
    good_cols = []
    bad_cols = []
    
    for group_name, group_df in groups:
        row = {}
        
        # Only add group column if group_by is specified
        if group_by:
            row[group_by] = group_name
        
        # Always show N (total count)
        row['N'] = len(group_df)
        
        for metric in metrics:
            if metric not in group_df.columns:
                continue
                
            values = group_df[metric].dropna()
            threshold = thresholds.get(metric, 0)
            is_bad_above = above_is_bad.get(metric, False)
            
            n_above = (values >= threshold).sum()
            n_below = (values < threshold).sum()
            n_total = len(values)
            
            # Column names
            col_above = f'{metric} (Above Threshold)'
            col_below = f'{metric} (Below Threshold)'
            
            # Value counts
            if show_cnt:
                row[col_below] = n_below
                row[col_above] = n_above
                
                # Track good/bad columns (only on first iteration)
                if len(rows) == 0:
                    if is_bad_above:
                        bad_cols.append(col_above)
                        good_cols.append(col_below)
                    else:
                        good_cols.append(col_above)
                        bad_cols.append(col_below)
            
            # Optional percentages
            if show_pct and n_total > 0:
                col_above_pct = f'{metric} (% Above Threshold)'
                col_below_pct = f'{metric} (% Below Threshold)'
                row[col_below_pct] = f"{round(n_below / n_total * 100, 1)}%"
                row[col_above_pct] = f"{round(n_above / n_total * 100, 1)}%"
                
                if len(rows) == 0:
                    if is_bad_above:
                        bad_cols.append(col_above_pct)
                        good_cols.append(col_below_pct)
                    else:
                        good_cols.append(col_above_pct)
                        bad_cols.append(col_below_pct)
            
            # Optional median
            if show_median and n_total > 0:
                row[f'{metric} (Median)'] = round(values.median(), 3)
        
        rows.append(row)
    
    result = pd.DataFrame(rows)
    
    if not style:
        return result
    
    # Apply styling
    def color_columns(col):
        if col.name in good_cols:
            return ['background-color: #d4edda; color: #155724'] * len(col)  # Green
        elif col.name in bad_cols:
            return ['background-color: #f8d7da; color: #721c24'] * len(col)  # Red
        return [''] * len(col)
    
    return result.style.apply(color_columns)