"""
- For lags: Use MLForecast(lags=[1, 7, 14])
- For rolling features: Use lag_transforms with RollingMean, RollingStd, etc.
- For expanding features: Use lag_transforms with ExpandingMean, ExpandingStd
- For date features: Use date_features=['dayofweek', 'month', ...]

"""
from __future__ import annotations
from typing import Sequence, Callable, Union, List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

try:
    from .recipes import Step
except Exception:
    from typing import Callable as Step


# ==========================================
# MLFORECAST NATIVE FEATURE WRAPPERS
# ==========================================

def add_rolling(
    target_col: str,
    transforms: List[str] = ["mean", "std"],
    window: int = 7,
    date_col: str = "date",
    id_col: Optional[str] = None,
) -> Step:
    """
    Add rolling window transformations.
    
    Engine: pandas rolling()
    MLForecast Native: Use lag_transforms with RollingMean, RollingStd, etc.
    RECOMMENDATION: Use MLForecast's Rolling* in lag_transforms instead!
    
    This is a Recipe wrapper. For production, use:
    ```python
    from mlforecast.lag_transforms import RollingMean, RollingStd, RollingMin, RollingMax
    fcst = MLForecast(
        models=models,
        lags=[1],
        lag_transforms={
            1: [RollingMean(window_size=7), RollingStd(window_size=7)]
        }
    )
    ```
    
    Parameters
    ----------
    value_col : str
        Column to calculate rolling features
    transforms : list of str
        Rolling transforms: 'mean', 'std', 'min', 'max', 'sum', 'median'
    window : int
        Window size
    date_col : str
        Date column
    id_col : str, optional
        Group column
    
    Returns
    -------
    Step function for Recipe
    
    Examples
    --------
    >>> # Recipe approach (slower)
    >>> recipe = Recipe().add_step(
    ...     add_rolling,
    ...     value_col='sales',
    ...     transforms=['mean', 'std', 'min', 'max'],
    ...     window=7,
    ...     id_col='unique_id'
    ... )
    """
    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        
        if id_col:
            rolling = out.groupby(id_col, observed=True)[target_col].rolling(
                window=window, min_periods=1
            )
        else:
            rolling = out[target_col].rolling(window=window, min_periods=1)
        
        for transform in transforms:
            col_name = f"{target_col}_rolling_{transform}_{window}"
            
            if transform == "mean":
                result = rolling.mean()
            elif transform == "std":
                result = rolling.std()
            elif transform == "min":
                result = rolling.min()
            elif transform == "max":
                result = rolling.max()
            elif transform == "sum":
                result = rolling.sum()
            elif transform == "median":
                result = rolling.median()
            else:
                raise ValueError(f"Unknown transform: {transform}")
            
            if id_col:
                out[col_name] = result.reset_index(level=0, drop=True)
            else:
                out[col_name] = result
        
        return out
    
    return _fn


def add_expanding(
    value_col: str,
    transforms: List[str] = ["mean", "std"],
    date_col: str = "date",
    id_col: Optional[str] = None,
) -> Step:
    """
    Add expanding window transformations.
    
    Engine: pandas expanding()
    MLForecast Native: Use lag_transforms with ExpandingMean, ExpandingStd, etc.
    RECOMMENDATION: Use MLForecast's Expanding* in lag_transforms instead!
    
    This is a Recipe wrapper. For production, use:
    ```python
    from mlforecast.lag_transforms import ExpandingMean, ExpandingStd, ExpandingMin, ExpandingMax
    fcst = MLForecast(
        models=models,
        lags=[1],
        lag_transforms={
            1: [ExpandingMean(), ExpandingStd()]
        }
    )
    ```
    
    Parameters
    ----------
    value_col : str
        Column to calculate expanding features
    transforms : list of str
        Expanding transforms: 'mean', 'std', 'min', 'max', 'sum', 'median'
    date_col : str
        Date column
    id_col : str, optional
        Group column
    
    Returns
    -------
    Step function for Recipe
    
    Examples
    --------
    >>> # Recipe approach (slower)
    >>> recipe = Recipe().add_step(
    ...     add_expanding,
    ...     value_col='sales',
    ...     transforms=['mean', 'std'],
    ...     id_col='unique_id'
    ... )
    """
    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        
        if id_col:
            expanding = out.groupby(id_col, observed=True)[value_col].expanding()
        else:
            expanding = out[value_col].expanding()
        
        for transform in transforms:
            col_name = f"{value_col}_expanding_{transform}"
            
            if transform == "mean":
                result = expanding.mean()
            elif transform == "std":
                result = expanding.std()
            elif transform == "min":
                result = expanding.min()
            elif transform == "max":
                result = expanding.max()
            elif transform == "sum":
                result = expanding.sum()
            elif transform == "median":
                result = expanding.median()
            else:
                raise ValueError(f"Unknown transform: {transform}")
            
            if id_col:
                out[col_name] = result.reset_index(level=0, drop=True)
            else:
                out[col_name] = result
        
        return out
    
    return _fn


def add_differences(
    value_col: str,
    lags: List[int] = [1],
    date_col: str = "date",
    id_col: Optional[str] = None,
) -> Step:
    """
    Add differencing transformation.
    
    Engine: pandas diff()
    MLForecast Native: Use target_transforms=[Differences([1])]
    RECOMMENDATION: Use MLForecast's Differences in target_transforms instead!
    
    This is a Recipe wrapper. For production, use:
    ```python
    from mlforecast.target_transforms import Differences
    fcst = MLForecast(
        models=models,
        target_transforms=[Differences([1])]
    )
    ```
    
    Parameters
    ----------
    value_col : str
        Column to difference
    lags : list of int
        Lag values for differencing (e.g., [1] for 1st diff, [1, 7] for multiple)
    date_col : str
        Date column
    id_col : str, optional
        Group column
    
    Returns
    -------
    Step function for Recipe
    
    Examples
    --------
    >>> # Recipe approach
    >>> recipe = Recipe().add_step(
    ...     add_differences,
    ...     value_col='sales',
    ...     lags=[1],
    ...     id_col='unique_id'
    ... )
    """
    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        
        for lag in lags:
            col_name = f"{value_col}_diff_{lag}"
            
            if id_col:
                out[col_name] = (
                    out.groupby(id_col, observed=True)[value_col]
                    .diff(periods=lag)
                )
            else:
                out[col_name] = out[value_col].diff(periods=lag)
        
        return out
    
    return _fn
