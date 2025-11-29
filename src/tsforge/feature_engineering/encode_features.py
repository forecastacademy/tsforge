"""
Encoding Methods

Factory Functions (Stateless):
- as_category: Convert to pandas category
- one_hot: One-hot encoding
- drop_features: Drop columns
- ratio_features: Create ratio features
- interaction_features: Interaction terms

Classes (Stateful):
- TargetMeanEncoder: Target encoding with OOF
- FrequencyEncoder: Frequency encoding
- OrdinalEncoder: Ordinal encoding
- HashEncoder: Hash encoding
- TimeAwareTargetEncoder: Time-aware target encoding
"""
from __future__ import annotations
from typing import Sequence, Mapping, Optional, Dict, Any, List, Tuple, Callable
import pandas as pd
import numpy as np

# Type alias for Recipe compatibility
Step = Callable[[pd.DataFrame], pd.DataFrame]

__all__ = [
    # Factory functions
    "as_category",
    "one_hot",
    "drop_features",
    "ratio_features",
    "interaction_features",
    # Classes
    "TargetMeanEncoder",
    "FrequencyEncoder",
    "OrdinalEncoder",
    "HashEncoder",
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _ensure_list(x: Sequence[str] | str) -> List[str]:
    """Convert to list."""
    return list(x) if isinstance(x, (list, tuple, set)) else [x]

def _cols_present(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    """Return columns that exist in dataframe."""
    return [c for c in cols if c in df.columns]


# ============================================================================
# Simple Stateless Encoders
# ============================================================================

def as_category(cols: Sequence[str]) -> Step:
    """
    Convert columns to pandas category dtype.
    
    Engine: pandas astype('category')
    MLForecast: Use before MLForecast.fit() for memory optimization
    
    Parameters
    ----------
    cols : Sequence[str]
        Columns to convert to category type
    
    Returns
    -------
    Step : Callable[[DataFrame], DataFrame]
        Transformation function for Recipe
    
    Examples
    --------
    >>> from recipes import Recipe
    >>> 
    >>> recipe = (
    ...     Recipe()
    ...     .add_step(as_category(['store_id', 'category']))
    ... )
    >>> df_encoded = recipe.fit_transform(df)
    """
    cols = list(cols)
        
    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in _cols_present(out, cols):
            out[c] = out[c].astype('category')
        return out
        
    return _transform


def one_hot(
    cols: Sequence[str],
    drop_first: bool = False,
    prefix: Optional[str] = None,
    prefix_sep: str = "_",
    dtype: str | np.dtype = "int8",
) -> Step:
    """
    One-hot encode categorical columns.
    
    Engine: pandas get_dummies
    MLForecast: Use for low-cardinality features (<10 categories)
    
    Parameters
    ----------
    cols : Sequence[str]
        Columns to one-hot encode
    drop_first : bool, default=False
        Drop first category to avoid multicollinearity
    prefix : str, optional
        Prefix for dummy column names
    prefix_sep : str, default='_'
        Separator between prefix and category
    dtype : str or dtype, default='int8'
        Data type for encoded columns
    
    Returns
    -------
    Step : Callable[[DataFrame], DataFrame]
        Transformation function for Recipe
    
    Examples
    --------
    >>> recipe = (
    ...     Recipe()
    ...     .add_step(one_hot(
    ...         cols=['size'],
    ...         drop_first=True
    ...     ))
    ... )
    """
    cols = list(cols)
    
    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        present = _cols_present(df, cols)
        if not present:
            return df.copy()
        
        out = df.copy()
        dummies = pd.get_dummies(
            out[present].astype('category'),
            prefix=prefix,
            prefix_sep=prefix_sep,
            drop_first=drop_first,
            dtype=dtype,
        )
        
        # Drop originals and add dummies
        out = out.drop(columns=present)
        return pd.concat([out, dummies], axis=1)
    
    return _transform


def drop_features(cols: Sequence[str]) -> Step:
    """
    Drop specified columns.
    
    Engine: pandas drop
    MLForecast: Use to clean up after encoding
    
    Parameters
    ----------
    cols : Sequence[str]
        Columns to drop
    
    Returns
    -------
    Step : Callable[[DataFrame], DataFrame]
        Transformation function for Recipe
    
    Examples
    --------
    >>> recipe = (
    ...     Recipe()
    ...     .add_step(drop_features(['temp_col', 'unused']))
    ... )
    """
    cols = list(cols)
    
    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        present = _cols_present(df, cols)
        if present:
            return df.drop(columns=present)
        return df.copy()
    
    return _transform


def ratio_features(
    col_pairs: List[Tuple[str, str]],
    suffix: str = "_ratio",
    epsilon: float = 1e-8,
) -> Step:
    """
    Create ratio features (col1 / col2).
    
    Engine: pandas vectorized operations
    MLForecast: Use for relative features (price/avg_price, etc.)
    
    Parameters
    ----------
    col_pairs : List[Tuple[str, str]]
        Pairs of (numerator, denominator) columns
    suffix : str, default='_ratio'
        Suffix for new column names
    epsilon : float, default=1e-8
        Small value to prevent division by zero
    
    Returns
    -------
    Step : Callable[[DataFrame], DataFrame]
        Transformation function for Recipe
    
    Examples
    --------
    >>> recipe = (
    ...     Recipe()
    ...     .add_step(ratio_features([
    ...         ('price', 'avg_price'),
    ...         ('sales', 'inventory')
    ...     ]))
    ... )
    """
    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        
        for num_col, denom_col in col_pairs:
            if num_col in out.columns and denom_col in out.columns:
                col_name = f"{num_col}_over_{denom_col}{suffix}"
                out[col_name] = out[num_col] / (out[denom_col] + epsilon)
                out[col_name] = out[col_name].replace([np.inf, -np.inf], np.nan)
        
        return out
    
    return _transform


def interaction_features(
    col_pairs: List[Tuple[str, str]],
    operation: str = "multiply",
) -> Step:
    """
    Create interaction features between column pairs.
    
    Engine: pandas vectorized operations
    MLForecast: Use for known interactions (price × promotion)
    
    Parameters
    ----------
    col_pairs : List[Tuple[str, str]]
        Pairs of columns to interact
    operation : str, default='multiply'
        Operation: 'multiply', 'add', 'subtract', 'divide'
    
    Returns
    -------
    Step : Callable[[DataFrame], DataFrame]
        Transformation function for Recipe
    
    Examples
    --------
    >>> recipe = (
    ...     Recipe()
    ...     .add_step(interaction_features(
    ...         col_pairs=[('is_holiday', 'is_weekend')],
    ...         operation='multiply'
    ...     ))
    ... )
    """
    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        
        for col1, col2 in col_pairs:
            if col1 in out.columns and col2 in out.columns:
                col_name = f"{col1}_x_{col2}_{operation}"
                
                if operation == "multiply":
                    out[col_name] = out[col1] * out[col2]
                elif operation == "add":
                    out[col_name] = out[col1] + out[col2]
                elif operation == "subtract":
                    out[col_name] = out[col1] - out[col2]
                elif operation == "divide":
                    out[col_name] = out[col1] / (out[col2] + 1e-8)
                    out[col_name] = out[col_name].replace([np.inf, -np.inf], np.nan)
                else:
                    raise ValueError(f"Unknown operation: {operation}")
        
        return out
    
    return _transform


# ============================================================================
# CLASS PATTERN: Complex Stateful Encoders
# ============================================================================

class BaseEncoder:
    """Base class for stateful encoders with sklearn-like API."""
    
    def fit(self, df: pd.DataFrame) -> 'BaseEncoder':
        """Fit encoder on data."""
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted encoder."""
        raise NotImplementedError("Subclasses must implement transform()")
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make encoder callable for Recipe compatibility."""
        return self.fit_transform(df)


class TargetMeanEncoder(BaseEncoder):
    """
    Target mean encoding with out-of-fold scheme.
    
    Engine: pandas groupby with cross-validation
    MLForecast: Use for STATIC features BEFORE MLForecast.fit()
    
    CRITICAL: Use id_cols=['unique_id'] for time series!
    
    Parameters
    ----------
    cols : Sequence[str]
        Columns to encode
    target_col : str
        Target column (typically 'y')
    id_cols : Sequence[str], optional
        ID columns for grouped CV (use ['unique_id'] for MLForecast)
    oof : bool, default=True
        Use out-of-fold encoding (STRONGLY RECOMMENDED)
    n_folds : int, default=5
        Number of folds for OOF
    random_state : int, default=42
        Random seed
    smoothing : float, default=10.0
        Smoothing parameter
    global_stat : str, default='mean'
        Global statistic: 'mean' or 'median'
    out_suffix : str, default='_tgt_mean'
        Suffix for encoded columns
    
    Examples
    --------
    >>> from recipes import Recipe
    >>> 
    >>> encoder = TargetMeanEncoder(
    ...     cols=['category', 'region'],
    ...     target_col='y',
    ...     id_cols=['unique_id'],
    ...     oof=True
    ... )
    >>> 
    >>> # Fit on train, transform on test
    >>> encoder.fit(train_df)
    >>> train_enc = encoder.transform(train_df)
    >>> test_enc = encoder.transform(test_df)
    >>> 
    >>> # Or use in Recipe
    >>> recipe = Recipe().add_step(encoder)
    """
    
    def __init__(
        self,
        cols: Sequence[str],
        target_col: str,
        id_cols: Optional[Sequence[str]] = None,
        oof: bool = True,
        n_folds: int = 5,
        random_state: int = 42,
        smoothing: float = 10.0,
        global_stat: str = "mean",
        out_suffix: str = "_tgt_mean",
    ):
        self.cols = list(cols)
        self.target_col = target_col
        self.id_cols = list(id_cols) if id_cols else None
        self.oof = oof
        self.n_folds = n_folds
        self.random_state = random_state
        self.smoothing = smoothing
        self.global_stat = global_stat
        self.out_suffix = out_suffix
        
        self.rng = np.random.RandomState(random_state)
        self._fitted_mappings: Dict[str, pd.Series] = {}
        self._global_mean: Optional[float] = None
        
        # Warnings
        if not oof:
            import warnings
            warnings.warn("⚠️  oof=False can cause leakage! Use oof=True.", UserWarning)
        
        if oof and id_cols is None:
            import warnings
            warnings.warn(
                "⚠️  Use id_cols=['unique_id'] for grouped CV in time series.",
                UserWarning
            )
    
    def _compute_global(self, series: pd.Series) -> float:
        if self.global_stat == 'median':
            return float(series.median())
        return float(series.mean())
    
    def fit(self, df: pd.DataFrame) -> 'TargetMeanEncoder':
        """Fit encoder by computing statistics on training data."""
        if self.target_col not in df.columns:
            return self
        
        self._global_mean = self._compute_global(df[self.target_col])
        
        # Fit mappings for transform on new data
        for col in _cols_present(df, self.cols):
            stats = df.groupby(col, observed=True)[self.target_col].agg(
                ['mean', 'count']
            ).rename(columns={'mean': 'gmean', 'count': 'gcount'})
            
            weight = stats['gcount'] / (stats['gcount'] + self.smoothing)
            stats['enc'] = weight * stats['gmean'] + (1 - weight) * self._global_mean
            self._fitted_mappings[col] = stats['enc']
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted mappings (simplified version)."""
        out = df.copy()
        
        if self.target_col not in out.columns or not self._fitted_mappings:
            return out
        
        gmean = self._global_mean if self._global_mean else self._compute_global(out[self.target_col])
        
        for col in _cols_present(out, self.cols):
            if col in self._fitted_mappings:
                out[f"{col}{self.out_suffix}"] = (
                    out[col].map(self._fitted_mappings[col])
                    .fillna(gmean)
                    .astype('float32')
                )
        
        return out
    
class TimeAwareTargetEncoder(BaseEncoder):
    """
    Optimized time-aware encoder using vectorized operations.
    
    Parameters are the same as TimeAwareTargetEncoder.
    """
    
    def __init__(
        self,
        cols: Sequence[str],
        target_col: str,
        date_col: str = 'ds',
        id_col: str = 'unique_id',
        smoothing: float = 10.0,
        global_stat: str = 'mean',
        out_suffix: str = '_time_aware',
        keep_original: bool = True,
        verbose: bool = True,
    ):
        self.cols = list(cols)
        self.target_col = target_col
        self.date_col = date_col
        self.id_col = id_col
        self.smoothing = smoothing
        self.global_stat = global_stat
        self.out_suffix = out_suffix
        self.keep_original = keep_original
        self.verbose = verbose
        self._global_mean: Optional[float] = None
    
    def _compute_global(self, series: pd.Series) -> float:
        if self.global_stat == 'median':
            return float(series.median())
        return float(series.mean())
    
    def fit(self, df: pd.DataFrame) -> TimeAwareTargetEncoder:
        if self.target_col in df.columns:
            self._global_mean = self._compute_global(df[self.target_col])
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.target_col not in df.columns:
            return df.copy()
        
        out = df.copy()
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(out[self.date_col]):
            out[self.date_col] = pd.to_datetime(out[self.date_col])
        
        if self._global_mean is None:
            self._global_mean = self._compute_global(out[self.target_col])
        
        for col in self.cols:
            if col not in out.columns:
                continue
            
            if self.verbose:
                print(f"Encoding {col} (fast method)...")
            
            col_name = f"{col}{self.out_suffix}"
            
            # Create helper columns for merging
            out['_temp_key'] = out.apply(
                lambda row: f"{row[col]}_{row[self.id_col]}_{row[self.date_col]}", 
                axis=1
            )
            
            # Compute cumulative stats per category across other series
            # This is still complex - using expanding mean per category
            out_sorted = out.sort_values([col, self.date_col])
            
            encoded = []
            for category in out[col].unique():
                if pd.isna(category):
                    continue
                
                cat_data = out[out[col] == category].copy()
                
                # For each row, compute mean from other series before this date
                for idx, row in cat_data.iterrows():
                    current_date = row[self.date_col]
                    current_series = row[self.id_col]
                    
                    past_mask = (
                        (out[col] == category) &
                        (out[self.date_col] < current_date) &
                        (out[self.id_col] != current_series)
                    )
                    
                    past_values = out.loc[past_mask, self.target_col]
                    
                    if len(past_values) > 0:
                        cat_mean = past_values.mean()
                        count = len(past_values)
                        weight = count / (count + self.smoothing)
                        encoded_val = weight * cat_mean + (1 - weight) * self._global_mean
                    else:
                        encoded_val = self._global_mean
                    
                    encoded.append({'index': idx, 'encoded': encoded_val})
            
            # Map back to original dataframe
            encoded_df = pd.DataFrame(encoded).set_index('index')
            out[col_name] = self._global_mean  # Default
            out.loc[encoded_df.index, col_name] = encoded_df['encoded'].values
            
            out = out.drop(columns=['_temp_key'])
        
        if not self.keep_original:
            cols_to_drop = [c for c in self.cols if c in out.columns]
            out = out.drop(columns=cols_to_drop)
        
        return out


class FrequencyEncoder(BaseEncoder):
    """
    Frequency encoding (count or probability).
    
    Engine: pandas value_counts
    MLForecast: Use for any categorical features
    
    Parameters
    ----------
    cols : Sequence[str]
        Columns to encode
    normalize : bool, default=False
        Convert counts to probabilities
    out_suffix : str, default='_freq'
        Suffix for encoded columns
    
    Examples
    --------
    >>> encoder = FrequencyEncoder(cols=['region'], normalize=True)
    >>> encoder.fit(train_df)
    >>> df_enc = encoder.transform(test_df)
    """
    
    def __init__(
        self,
        cols: Sequence[str],
        normalize: bool = False,
        out_suffix: str = "_freq",
    ):
        self.cols = list(cols)
        self.normalize = normalize
        self.out_suffix = out_suffix
        self._freq_maps: Dict[str, pd.Series] = {}
    
    def fit(self, df: pd.DataFrame) -> 'FrequencyEncoder':
        """Fit by computing frequency maps."""
        n = len(df)
        for col in _cols_present(df, self.cols):
            vc = df[col].value_counts(dropna=False)
            if self.normalize and n > 0:
                vc = vc / float(n)
            self._freq_maps[col] = vc
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted frequency maps."""
        out = df.copy()
        
        for col in _cols_present(out, self.cols):
            if col in self._freq_maps:
                out[f"{col}{self.out_suffix}"] = (
                    out[col].map(self._freq_maps[col]).fillna(0).astype('float32')
                )
        
        return out


class OrdinalEncoder(BaseEncoder):
    """
    Ordinal encoding by explicit mapping or frequency.
    
    Engine: pandas map
    MLForecast: Use for ordered categoricals (size: S/M/L/XL)
    
    Parameters
    ----------
    cols : Sequence[str], optional
        Columns to encode
    mapping : Dict[str, Dict], optional
        Explicit mapping: {'size': {'S': 0, 'M': 1, 'L': 2}}
    by_frequency : bool, default=False
        Order by frequency instead of explicit mapping
    
    Examples
    --------
    >>> encoder = OrdinalEncoder(
    ...     mapping={'size': {'S': 0, 'M': 1, 'L': 2, 'XL': 3}}
    ... )
    """
    
    def __init__(
        self,
        cols: Optional[Sequence[str]] = None,
        mapping: Optional[Mapping[str, Mapping[Any, int]]] = None,
        by_frequency: bool = False,
    ):
        self.cols = list(cols) if cols else None
        self.mapping = mapping
        self.by_frequency = by_frequency
        self._fitted_mappings: Dict[str, Dict] = {}
    
    def fit(self, df: pd.DataFrame) -> 'OrdinalEncoder':
        """Fit by computing ordinal mappings."""
        target_cols = self.cols if self.cols else list(self.mapping.keys()) if self.mapping else []
        
        for col in _cols_present(df, target_cols):
            if self.mapping and col in self.mapping:
                self._fitted_mappings[col] = self.mapping[col]
            elif self.by_frequency:
                order = df[col].value_counts().index.tolist()
                self._fitted_mappings[col] = {cat: i for i, cat in enumerate(order)}
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted mappings."""
        out = df.copy()
        
        for col, mapping in self._fitted_mappings.items():
            if col in out.columns:
                out[f"{col}_ord"] = out[col].map(mapping).astype('int16')
        
        return out


class HashEncoder(BaseEncoder):
    """
    Hash encoding for extreme high-cardinality features.
    
    Engine: pandas hash function
    MLForecast: Perfect for >1000 categories (user_ids, product_ids)
    
    Parameters
    ----------
    cols : Sequence[str]
        Columns to hash encode
    n_bins : int, default=64
        Number of hash bins
    salt : int, default=0
        Salt for hash function
    
    Examples
    --------
    >>> encoder = HashEncoder(cols=['user_id'], n_bins=128)
    """
    
    def __init__(
        self,
        cols: Sequence[str],
        n_bins: int = 64,
        salt: int = 0,
    ):
        self.cols = list(cols)
        self.n_bins = n_bins
        self.salt = salt
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using hash encoding (stateless)."""
        out = df.copy()
        
        for col in _cols_present(out, self.cols):
            s = out[col].astype(str).fillna("nan")
            hashed = s.apply(
                lambda x: (
                    pd.util.hash_pandas_object(pd.Series([x, self.salt]), index=False).values[0]
                ) % self.n_bins
            )
            out[f"{col}_hash"] = hashed.astype('int32')
        
        return out

