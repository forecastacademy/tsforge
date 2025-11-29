"""
M5 Utilities Module
===================

Utility functions for working with M5 dataset in the Forecast Academy.

Core Functions:
- load_m5(): Load M5 time series data with optional messification
- create_subset(): Create smaller subsets for faster iteration
- messify_m5_data(): Simulate real-world data quality issues
- expand_hierarchy(): Replace unique_id with original hierarchy columns
- check_gaps(): Diagnose gaps in time series data
- first_contact_check(): Run data quality checks on a DataFrame

Author: Forecast Academy
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List

# Optional import - will raise error when load functions are called if not installed
try:
    from datasetsforecast.m5 import M5
    M5_AVAILABLE = True
except ImportError:
    M5_AVAILABLE = False
    M5 = None


# ============================================================================
# SECTION 0: HIERARCHY & AGGREGATION HELPERS
# ============================================================================

# M5 hierarchy columns in order
HIERARCHY_COLS = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']


def create_unique_id(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    separator: str = '_',
    target_col: str = 'unique_id',
    inplace: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create a unique_id column by concatenating multiple columns.

    Nixtla libraries require a 'unique_id' column to identify each time series.
    This function creates it by joining specified columns with a separator.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list of str, optional
        Columns to concatenate. If None, auto-detects:
        - Uses ['item_id', 'store_id'] if both exist (M5 default)
        - Falls back to all categorical/object columns if not
    separator : str, default='_'
        Separator between column values
    target_col : str, default='unique_id'
        Name of the output column
    inplace : bool, default=False
        If True, modify DataFrame in place and return it.
        If False, return a copy with the new column.
    verbose : bool, default=True
        Print summary of unique_id creation

    Returns
    -------
    pd.DataFrame
        DataFrame with new unique_id column

    Examples
    --------
    >>> # Auto-detect columns (M5 default: item_id + store_id)
    >>> df = create_unique_id(df)

    >>> # Specify columns explicitly
    >>> df = create_unique_id(df, columns=['item_id', 'store_id'])

    >>> # Use different separator
    >>> df = create_unique_id(df, columns=['dept_id', 'state_id'], separator='-')

    >>> # Modify in place
    >>> create_unique_id(df, inplace=True)
    """
    if not inplace:
        df = df.copy()

    # Auto-detect columns if not specified
    if columns is None:
        # Default M5 pattern: item_id + store_id
        if 'item_id' in df.columns and 'store_id' in df.columns:
            columns = ['item_id', 'store_id']
        else:
            # Fall back to categorical/object columns (excluding common non-ID cols)
            exclude = {'ds', 'y', 'date', 'value', 'target', target_col}
            columns = [
                c for c in df.columns
                if c not in exclude and
                (df[c].dtype == 'object' or df[c].dtype.name == 'category')
            ]
            if not columns:
                raise ValueError(
                    "No columns specified and could not auto-detect. "
                    "Please provide columns=['col1', 'col2', ...]"
                )

    # Validate columns exist
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    # Create unique_id by concatenating columns
    df[target_col] = df[columns[0]].astype(str)
    for col in columns[1:]:
        df[target_col] = df[target_col] + separator + df[col].astype(str)

    if verbose:
        n_unique = df[target_col].nunique()
        print(f"Created {target_col}: {n_unique:,} unique series")
        print(f"Sample: {df[target_col].iloc[0]}")

    return df


def expand_hierarchy(
    df: pd.DataFrame,
    S_df: pd.DataFrame = None,
    id_col: str = 'unique_id',
    drop_unique_id: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Replace unique_id with original M5 hierarchy columns.
    
    The unique_id in M5 (e.g., "FOODS_1_001_CA_1") encodes the hierarchy:
    - item_id: FOODS_1_001
    - dept_id: FOODS_1
    - cat_id: FOODS
    - store_id: CA_1
    - state_id: CA
    
    This function expands unique_id back to these original columns either by
    parsing the ID string or by merging with S_df (static hierarchy dataframe).
    
    OPTIMIZED: Uses vectorized string operations instead of slow apply().
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with unique_id column
    S_df : pd.DataFrame, optional
        Static hierarchy dataframe from load_m5_full(). If provided, uses merge.
        If None, parses the unique_id string directly.
    id_col : str, default='unique_id'
        Name of the ID column
    drop_unique_id : bool, default=True
        Whether to drop the unique_id column after expansion
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    pd.DataFrame
        DataFrame with hierarchy columns (item_id, dept_id, cat_id, store_id, state_id)
        
    Examples
    --------
    >>> df = load_m5(Path('data'))
    >>> df_expanded = expand_hierarchy(df)
    >>> print(df_expanded.columns)
    # ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'ds', 'y']
    """
    if id_col not in df.columns:
        if verbose:
            print(f"⚠ Column '{id_col}' not found, returning unchanged")
        return df
    
    df_result = df.copy()
    
    if S_df is not None:
        # Use merge with S_df for accurate hierarchy
        if verbose:
            print("  Expanding hierarchy via S_df merge...")
        
        # S_df should have unique_id and hierarchy columns
        hierarchy_cols_present = [c for c in HIERARCHY_COLS if c in S_df.columns]
        merge_cols = [id_col] + hierarchy_cols_present
        
        df_result = df_result.merge(
            S_df[merge_cols].drop_duplicates(),
            on=id_col,
            how='left'
        )
    else:
        # OPTIMIZED: Vectorized parsing using str.split instead of apply()
        # Format: {cat_id}_{dept_num}_{item_num}_{state_id}_{store_num}
        # Example: FOODS_1_001_CA_1 -> cat=FOODS, dept=FOODS_1, item=FOODS_1_001, state=CA, store=CA_1
        if verbose:
            print("  Expanding hierarchy via vectorized ID parsing...")
        
        # Get unique IDs and parse them (do this on unique values for speed)
        unique_ids = df_result[id_col].unique()
        id_mapping = pd.DataFrame({id_col: unique_ids})
        
        # Vectorized split - much faster than apply()
        parts = id_mapping[id_col].str.split('_', expand=True)
        
        # Build hierarchy columns from parts
        # parts[0] = cat (e.g., FOODS)
        # parts[1] = dept_num (e.g., 1)
        # parts[2] = item_num (e.g., 001)
        # parts[3] = state (e.g., CA)
        # parts[4] = store_num (e.g., 1)
        id_mapping['cat_id'] = parts[0]
        id_mapping['dept_id'] = parts[0] + '_' + parts[1]
        id_mapping['item_id'] = parts[0] + '_' + parts[1] + '_' + parts[2]
        id_mapping['state_id'] = parts[3]
        id_mapping['store_id'] = parts[3] + '_' + parts[4]
        
        # Merge back
        df_result = df_result.merge(id_mapping, on=id_col, how='left')
    
    # Reorder columns: hierarchy first, then rest
    other_cols = [c for c in df_result.columns if c not in HIERARCHY_COLS + [id_col]]
    if drop_unique_id:
        new_order = HIERARCHY_COLS + other_cols
    else:
        new_order = HIERARCHY_COLS + [id_col] + other_cols
    
    # Only include columns that exist
    new_order = [c for c in new_order if c in df_result.columns]
    df_result = df_result[new_order]
    
    if verbose:
        print(f"  ✓ Added hierarchy columns: {HIERARCHY_COLS}")
    
    return df_result


# ============================================================================
# SECTION 1: LOADING FUNCTIONS
# ============================================================================

def has_m5_cache(data_dir: Path) -> bool:
    """
    Quick check for M5 cache files.
    
    Checks for specific M5 cache files rather than doing expensive
    recursive directory searches.
    
    Parameters
    ----------
    data_dir : Path
        Directory where M5 data would be cached
        
    Returns
    -------
    bool
        True if M5 cache files are found, False otherwise
        
    Examples
    --------
    >>> data_dir = Path('data')
    >>> if has_m5_cache(data_dir):
    ...     print("Using cached M5")
    ... else:
    ...     print("Will download M5")
    """
    # M5.load() caches to these file patterns
    cache_files = [
        'M5.parquet',
        'M5.csv',
        'm5.parquet',
        'm5.csv',
        'm5.p',
        'M5.p'
    ]

    # Also check common subdirectories and nested paths
    cache_subdirs = ['m5', 'M5', 'm5-forecasting-accuracy', 'm5/datasets', 'M5/datasets']

    # Check main directory
    if any((data_dir / f).exists() for f in cache_files):
        return True

    # Check subdirectories
    for subdir in cache_subdirs:
        subdir_path = data_dir / subdir
        if subdir_path.exists() and subdir_path.is_dir():
            if any((subdir_path / f).exists() for f in cache_files):
                return True

    return False


def load_m5_with_feedback(
    data_dir: Path,
    verbose: bool = True,
    return_additional: bool = False
) -> Tuple:
    """
    Load M5 dataset with informative feedback.
    
    M5 includes 3 separate dataframes:
    - Y_df: Time series data (unique_id, ds, y) - what forecasters need
    - X_df: Calendar/exogenous features (events, SNAP, holidays)
    - S_df: Static hierarchy (item → dept → category, store → state)
    
    By default, returns only Y_df since that's needed 99% of the time.
    Set return_additional=True to get all 3.
    
    Provides timing, memory usage, and cache status information.
    Optimized for fast cache detection (no recursive directory searches).
    
    Parameters
    ----------
    data_dir : Path
        Directory for M5 data cache
    verbose : bool, default=True
        Whether to print progress messages
    return_additional : bool, default=False
        If True, returns (Y_df, X_df, S_df). If False, returns just Y_df.
        
    Returns
    -------
    pd.DataFrame or tuple
        If return_additional=False: Returns Y_df only
        If return_additional=True: Returns (Y_df, X_df, S_df) tuple
        
    Raises
    ------
    ImportError
        If datasetsforecast is not installed
        
    Examples
    --------
    >>> # Just get time series data (default)
    >>> df = load_m5_with_feedback(Path('data'))
    >>> print(df.columns)  # ['unique_id', 'ds', 'y']
    
    >>> # Get all 3 dataframes
    >>> Y_df, X_df, S_df = load_m5_with_feedback(Path('data'), return_additional=True)
    >>> # Y_df: time series | X_df: calendar features | S_df: hierarchy
    
    >>> # Silent loading (no output)
    >>> df = load_m5_with_feedback(Path('data'), verbose=False)
    """
    if not M5_AVAILABLE:
        raise ImportError(
            "datasetsforecast is required to load M5 data.\n"
            "Install it with: pip install datasetsforecast"
        )
    
    # Fast cache check (4-8 file checks, no recursion)
    cache_exists = has_m5_cache(data_dir)
    
    if verbose:
        if cache_exists:
            print("✓ M5 cache detected. Loading from local files...")
        else:
            print("⚠ No M5 cache found. First download will take ~30-60s (~200MB)...")
            print("  Subsequent loads will be instant.")
    
    # Load with timing
    start_time = time.time()
    result = M5.load(directory=str(data_dir))  # Returns (Y_df, X_df, S_df)
    load_time = time.time() - start_time
    
    # Extract Y_df (time series data) for reporting
    df = result[0]
    
    if verbose:
        print(f"✓ Loaded in {load_time:.1f}s")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        print(f"  Memory: {memory_mb:,.1f} MB")
        
        # Show columns
        n_cols_show = min(8, len(df.columns))
        cols_str = ', '.join(df.columns[:n_cols_show].tolist())
        if len(df.columns) > n_cols_show:
            cols_str += '...'
        print(f"  Columns: {cols_str}")
        
        # Clarify what's being returned
        if return_additional:
            print(f"  Returning: Y_df, X_df, S_df (all 3 dataframes)")
        else:
            print(f"  Returning: Y_df only (time series data)")
    
    # Return based on user preference
    if return_additional:
        return result  # Returns (Y_df, X_df, S_df)
    else:
        return df  # Just return Y_df


def load_m5(
    data_dir: Path,
    from_parquet: Optional[Path] = None,
    verbose: bool = True,
    messify: bool = False,
    messify_kwargs: Optional[dict] = None,
    include_hierarchy: bool = False,
    create_unique_id: bool = True,
    n_series: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load M5 time series data with optional preprocessing.

    This is the main entry point for loading M5 data. It provides a unified
    interface for common preprocessing steps:

    - Load from raw M5 or from a previous module's parquet output
    - Messification: Simulate real-world data quality issues
    - Hierarchy expansion: Replace unique_id with original hierarchy columns
    - Unique ID creation: Create unique_id from hierarchy columns (Nixtla compatible)
    - Subsetting: Work with a smaller sample for faster iteration

    For weekly aggregation, use aggregate_to_weekly() after loading.

    Returns a DataFrame with either:
    - ['unique_id', 'ds', 'y'] columns (default)
    - ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'ds', 'y']
      (when include_hierarchy=True and create_unique_id=False)
    - ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'unique_id', 'ds', 'y']
      (when include_hierarchy=True and create_unique_id=True)

    Parameters
    ----------
    data_dir : Path
        Directory for M5 data cache (used when from_parquet is None)
    from_parquet : Path, optional
        Path to a parquet file from a previous module (e.g., '1_6_output.parquet').
        If provided, loads from this file instead of raw M5 data.
        All preprocessing options (messify, create_unique_id, etc.) still apply.
    verbose : bool, default=True
        Whether to print progress messages
    messify : bool, default=False
        If True, apply messification to simulate real-world data issues.
        Customize with messify_kwargs.
    messify_kwargs : dict, optional
        Additional arguments for messify_m5_data(). Common options:
        - random_state: int (default=42)
        - zeros_to_na_frac: float (default=0.15)
        - zeros_drop_frac: float (default=0.02)
        - zeros_drop_gaps_frac: float (default=None)
        - duplicates_add_n: int (default=150)
        - na_drop_frac: float (default=None)
        - dtypes_corrupt: bool (default=True)
        - cache_dir: Path (default=None, uses data_dir)
    include_hierarchy : bool, default=False
        If True, replace 'unique_id' with original hierarchy columns:
        item_id, dept_id, cat_id, store_id, state_id.
        Ignored if loading from_parquet that already has hierarchy columns.
    create_unique_id : bool, default=True
        If True and hierarchy columns exist (item_id, store_id), create a
        unique_id column from item_id + store_id. This ensures Nixtla
        compatibility while keeping hierarchy columns.
    n_series : int, optional
        If provided, subset to this many series for faster iteration.
        Applied after loading but before messification.
    random_state : int, default=42
        Random seed for subsetting reproducibility

    Returns
    -------
    pd.DataFrame
        M5 time series data with columns based on settings

    Examples
    --------
    >>> from pathlib import Path
    >>>
    >>> # Basic: load raw M5 data
    >>> df = load_m5(Path('data'))
    >>>
    >>> # Load from previous module output
    >>> df = load_m5(Path('data'), from_parquet=Path('data/1_6_output.parquet'))
    >>>
    >>> # Load from module output with unique_id creation
    >>> df = load_m5(
    ...     Path('data'),
    ...     from_parquet=Path('data/1_6_output.parquet'),
    ...     create_unique_id=True
    ... )
    >>>
    >>> # Load from module output and messify for training
    >>> df = load_m5(
    ...     Path('data'),
    ...     from_parquet=Path('data/1_6_output.parquet'),
    ...     messify=True,
    ...     create_unique_id=True
    ... )
    >>>
    >>> # With hierarchy columns AND unique_id (default)
    >>> df = load_m5(Path('data'), include_hierarchy=True)
    >>>
    >>> # Messified for training exercises
    >>> df = load_m5(Path('data'), messify=True, include_hierarchy=True)
    """
    # Determine source
    loading_from_parquet = from_parquet is not None

    if verbose:
        print("=" * 70)
        if loading_from_parquet:
            print(f"LOADING FROM PARQUET: {Path(from_parquet).name}")
        else:
            print("LOADING M5 DATA")
        print("=" * 70)

    # Step 1: Load data from appropriate source
    S_df = None
    has_hierarchy = False

    if loading_from_parquet:
        # Load from parquet file
        from_parquet = Path(from_parquet)
        if not from_parquet.exists():
            raise FileNotFoundError(f"Parquet file not found: {from_parquet}")

        start_time = time.time()
        df = pd.read_parquet(from_parquet)
        load_time = time.time() - start_time

        # Check if hierarchy columns already exist
        has_hierarchy = all(c in df.columns for c in ['item_id', 'store_id'])

        if verbose:
            print(f"✓ Loaded in {load_time:.1f}s")
            print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            print(f"  Memory: {memory_mb:,.1f} MB")
            print(f"  Columns: {list(df.columns)}")
            if has_hierarchy:
                print(f"  ✓ Hierarchy columns detected")
    else:
        # Load raw M5 data
        need_hierarchy = include_hierarchy
        if need_hierarchy:
            Y_df, _, S_df = load_m5_with_feedback(data_dir, verbose=verbose, return_additional=True)
        else:
            Y_df = load_m5_with_feedback(data_dir, verbose=verbose, return_additional=False)

        df = Y_df
    
    # Step 2: Subset if requested (before messification for speed)
    if n_series is not None:
        if verbose:
            print(f"\n📊 Subsetting to {n_series} series...")
        df = create_subset(df, n_series=n_series, random_state=random_state, verbose=verbose)
    
    # Step 3: Messify if requested
    if messify:
        if verbose:
            print(f"\n🔧 Applying messification...")
        
        # Set up messify kwargs with defaults
        _messify_kwargs = {
            'random_state': random_state,
            'zeros_to_na_frac': 0.15,
            'zeros_drop_frac': 0.02,
            'zeros_drop_gaps_frac': None,
            'duplicates_add_n': 150,
            'na_drop_frac': None,
            'dtypes_corrupt': True,
            'cache_dir': data_dir,
            'verbose': verbose
        }
        if messify_kwargs:
            _messify_kwargs.update(messify_kwargs)
        
        df = messify_m5_data(df, **_messify_kwargs)

    # Step 4: Expand hierarchy if requested (skip if already has hierarchy from parquet)
    expanded_hierarchy = False
    if include_hierarchy and not has_hierarchy:
        if verbose and not messify:
            print(f"\n🏗️ Expanding hierarchy...")
        df = expand_hierarchy(df, S_df=S_df, verbose=verbose)
        expanded_hierarchy = True
        has_hierarchy = True  # Now we have hierarchy columns

    # Step 5: Create unique_id from hierarchy columns if requested
    # Works for both: raw M5 with include_hierarchy=True, or parquet with hierarchy cols
    created_unique_id = False
    if create_unique_id and has_hierarchy and 'unique_id' not in df.columns:
        if verbose:
            print(f"\n🔑 Creating unique_id...")
        df = globals()['create_unique_id'](
            df,
            columns=['item_id', 'store_id'],
            verbose=verbose
        )
        created_unique_id = True

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("LOAD COMPLETE")
        print("=" * 70)
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")

        # Summarize what was applied
        applied = []
        if loading_from_parquet:
            applied.append(f"from {Path(from_parquet).name}")
        if n_series:
            applied.append(f"subset ({n_series} series)")
        if messify:
            applied.append("messified")
        if expanded_hierarchy:
            applied.append("hierarchy expanded")
        if created_unique_id:
            applied.append("unique_id created")

        if applied:
            print(f"  Applied: {', '.join(applied)}")
        print("=" * 70)

    return df


# ============================================================================
# SECTION 2: SUBSET CREATION
# ============================================================================

def create_subset(
    df: pd.DataFrame,
    n_series: int = 100,
    id_col: str = 'unique_id',
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create a random subset of series for faster processing.
    
    Useful for development, testing, and training when you don't need
    all 30,490 M5 series.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full M5 dataset
    n_series : int, default=100
        Number of series to sample
    id_col : str, default='unique_id'
        Name of the series ID column
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Print subset information
        
    Returns
    -------
    pd.DataFrame
        Subset of the original data with n_series
        
    Examples
    --------
    >>> # Load full M5
    >>> df_full = load_m5(Path('data'))
    >>> 
    >>> # Create subset with 100 series (much faster!)
    >>> df_subset = create_subset(df_full, n_series=100)
    >>> 
    >>> # Create larger subset
    >>> df_subset = create_subset(df_full, n_series=1000)
    """
    np.random.seed(random_state)
    
    # Get all unique series
    all_series = df[id_col].unique()
    total_series = len(all_series)
    
    # Validate n_series
    if n_series > total_series:
        if verbose:
            print(f"⚠ Requested {n_series} series but only {total_series} available")
            print(f"  Using all {total_series} series")
        n_series = total_series
    
    # Sample series
    sample_series = np.random.choice(all_series, size=n_series, replace=False)
    
    # Create subset
    df_subset = df[df[id_col].isin(sample_series)].copy()
    
    if verbose:
        print("=" * 70)
        print("SUBSET CREATED")
        print("=" * 70)
        print(f"\n📊 SIZE")
        print(f"  Original:  {len(df):>12,} rows, {total_series:>6,} series")
        print(f"  Subset:    {len(df_subset):>12,} rows, {n_series:>6,} series")
        print(f"  Reduction: {len(df_subset) / len(df) * 100:>12.1f}% of original")
        
        # Average observations per series
        if len(df_subset) > 0:
            avg_obs = len(df_subset) / n_series
            print(f"\n📈 STATS")
            print(f"  Avg obs/series: {avg_obs:>8.1f}")
        
        print("\n💡 TIP: Subset is perfect for:")
        print("  • Fast iteration during development")
        print("  • Quick testing of code changes")
        print("  • Training examples in notebooks")
        print("\n  To use full dataset, set: n_series=len(df['unique_id'].unique())")
        print("=" * 70)
    
    return df_subset


# ============================================================================
# SECTION 3: DATA MESSIFICATION
# ============================================================================

def messify_m5_data(
    df: pd.DataFrame,
    id_col: str = 'unique_id',
    date_col: str = 'ds',
    target_col: str = 'y',
    random_state: int = 42,
    # --- ZEROS HANDLING ---
    zeros_to_na_frac: Optional[float] = 0.15,
    zeros_drop_frac: Optional[float] = 0.02,
    zeros_drop_gaps_frac: Optional[float] = None,
    # --- DUPLICATES ---
    duplicates_add_n: Optional[int] = 150,
    # --- NA HANDLING ---
    na_drop_frac: Optional[float] = None,
    # --- DATA TYPES ---
    dtypes_corrupt: bool = True,
    # --- CACHING ---
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Messify clean M5 data to simulate real-world data quality issues.
    
    Introduces common data problems found in real forecasting pipelines:
    
    1. zeros_to_na_frac: Convert zeros → NAs (simulates missing reporting)
    2. zeros_drop_frac: Remove zero rows entirely (simulates sparse reporting)
    3. zeros_drop_gaps_frac: Remove zeros from MIDDLE of series (creates internal gaps)
    4. duplicates_add_n: Add duplicate rows (simulates faulty ETL/merges)
    5. na_drop_frac: Drop some NA rows (simulates partial data recovery)
    6. dtypes_corrupt: Convert dates/numbers to strings (simulates CSV round-trips)
    
    Results are cached to speed up repeated runs with the same parameters.
    
    OPTIMIZED: Uses vectorized operations instead of slow groupby-apply patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Clean M5 dataset from datasetsforecast
    id_col : str, default='unique_id'
        Name of the ID column
    date_col : str, default='ds'
        Name of the date column
    target_col : str, default='y'
        Name of the target column
    random_state : int, default=42
        Random seed for reproducibility
        
    zeros_to_na_frac : float, optional, default=0.15
        Fraction of zero values to convert to NA (0.0 to 1.0).
        Simulates missing data where zeros weren't reported.
        Set to None to disable.
    zeros_drop_frac : float, optional, default=0.02
        Fraction of zero-value rows to drop entirely (0.0 to 1.0).
        Simulates sparse reporting where zero-demand periods aren't recorded.
        Set to None to disable.
    zeros_drop_gaps_frac : float, optional, default=None
        Fraction of zero-value rows to drop from MIDDLE of each series only
        (never first or last row). Creates true internal gaps for testing
        gap detection. Set to None to disable.
    duplicates_add_n : int, optional, default=150
        Number of duplicate rows to add. Simulates faulty ETL or merge issues.
        Set to None to disable.
    na_drop_frac : float, optional, default=None
        Fraction of NA rows to drop (0.0 to 1.0). Applied after zeros_to_na
        creates NAs. Simulates partial data recovery efforts.
        Set to None to disable.
    dtypes_corrupt : bool, default=True
        Whether to corrupt data types by converting date and target columns
        to strings. Simulates CSV round-trips or poorly typed databases.
        
    cache_dir : Path, optional
        Directory to cache the messified data. If None, no caching.
    force_refresh : bool, default=False
        If True, regenerate messified data even if cache exists.
    verbose : bool, default=True
        Whether to print summary of changes.
        
    Returns
    -------
    pd.DataFrame
        Messified version of the input data
        
    Examples
    --------
    >>> # Default messification
    >>> df_messy = messify_m5_data(df_clean)
    
    >>> # Heavy messification with internal gaps
    >>> df_messy = messify_m5_data(
    ...     df_clean,
    ...     zeros_to_na_frac=0.30,
    ...     zeros_drop_gaps_frac=0.10,
    ...     duplicates_add_n=200,
    ...     cache_dir=Path('data')
    ... )
    
    >>> # Light messification (dtype corruption only)
    >>> df_messy = messify_m5_data(
    ...     df_clean,
    ...     zeros_to_na_frac=None,
    ...     zeros_drop_frac=None,
    ...     duplicates_add_n=None,
    ...     dtypes_corrupt=True
    ... )
    """
    # Generate cache filename based on parameters
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Create filename that reflects the messification parameters
        n_series = df[id_col].nunique()
        cache_filename = (
            f"m5_messy_"
            f"n{n_series}_"
            f"rs{random_state}_"
            f"z2na{int(zeros_to_na_frac*100) if zeros_to_na_frac else 0}_"
            f"zdrp{int(zeros_drop_frac*100) if zeros_drop_frac else 0}_"
            f"zgap{int(zeros_drop_gaps_frac*100) if zeros_drop_gaps_frac else 0}_"
            f"dup{duplicates_add_n if duplicates_add_n else 0}_"
            f"nadrp{int(na_drop_frac*100) if na_drop_frac else 0}_"
            f"dtype{1 if dtypes_corrupt else 0}"
            f".parquet"
        )
        cache_path = cache_dir / cache_filename
        
        # Check if cache exists and should be used
        if cache_path.exists() and not force_refresh:
            if verbose:
                print("=" * 70)
                print("LOADING CACHED MESSIFIED DATA")
                print("=" * 70)
                print(f"\n📁 Cache file: {cache_path.name}")
                print("   Using cached version (skip messification)")
                print("\n💡 To regenerate: set force_refresh=True")
            
            df_messy = pd.read_parquet(cache_path)
            
            if verbose:
                print(f"\n✓ Loaded {df_messy.shape[0]:,} rows × {df_messy.shape[1]} columns")
                print("=" * 70)
            
            return df_messy
        
        if verbose and cache_path.exists():
            print(f"\n⚠️  Cache exists but force_refresh=True, regenerating...\n")
    
    # Perform messification
    np.random.seed(random_state)
    df_messy = df.copy()
    changes_log = []
    
    # Step 1: Convert some zeros to NAs
    if zeros_to_na_frac is not None and zeros_to_na_frac > 0 and target_col in df_messy.columns:
        if verbose:
            print("Step 1/6: Converting zeros to NAs...")
        
        zero_mask = df_messy[target_col] == 0
        n_zeros = zero_mask.sum()
        n_to_convert = int(n_zeros * zeros_to_na_frac)
        
        if n_to_convert > 0:
            zero_indices = df_messy[zero_mask].index
            na_indices = np.random.choice(zero_indices, size=n_to_convert, replace=False)
            df_messy.loc[na_indices, target_col] = np.nan
            
            changes_log.append(f"Converted {n_to_convert:,} zeros to NAs ({zeros_to_na_frac*100:.0f}% of zeros)")
            if verbose:
                print(f"  ✓ Converted {n_to_convert:,} zeros to NAs")
    elif verbose:
        print("Step 1/6: Converting zeros to NAs... [SKIPPED]")
    
    # Step 2: Add duplicate rows
    if duplicates_add_n is not None and duplicates_add_n > 0:
        if verbose:
            print("Step 2/6: Adding duplicate rows...")
        
        n_duplicates = min(duplicates_add_n, len(df_messy))
        if n_duplicates > 0:
            duplicate_indices = np.random.choice(df_messy.index, size=n_duplicates, replace=False)
            duplicates = df_messy.loc[duplicate_indices].copy()
            df_messy = pd.concat([df_messy, duplicates], ignore_index=True)
            
            changes_log.append(f"Added {n_duplicates:,} duplicate rows")
            if verbose:
                print(f"  ✓ Added {n_duplicates:,} duplicate rows")
    elif verbose:
        print("Step 2/6: Adding duplicate rows... [SKIPPED]")
    
    # Step 3: Remove some zero-demand rows (sparse reporting)
    if zeros_drop_frac is not None and zeros_drop_frac > 0:
        if verbose:
            print("Step 3/6: Dropping zero-demand rows (sparse reporting)...")
        
        # Only target actual zeros (not NAs created in step 1)
        zero_mask = df_messy[target_col] == 0
        zero_indices = df_messy[zero_mask].index
        
        n_to_remove = int(len(zero_indices) * zeros_drop_frac)
        if n_to_remove > 0:
            removal_indices = np.random.choice(zero_indices, size=n_to_remove, replace=False)
            df_messy = df_messy.drop(removal_indices).reset_index(drop=True)
            
            changes_log.append(f"Dropped {n_to_remove:,} zero-demand rows ({zeros_drop_frac*100:.0f}% of zeros)")
            if verbose:
                print(f"  ✓ Dropped {n_to_remove:,} zero-demand rows")
    elif verbose:
        print("Step 3/6: Dropping zero-demand rows... [SKIPPED]")
    
    # Step 4: Drop fraction of NA rows if requested
    if na_drop_frac is not None and na_drop_frac > 0:
        if verbose:
            print("Step 4/6: Dropping NA rows...")

        na_mask = df_messy[target_col].isna()
        n_na = na_mask.sum()

        if n_na > 0:
            n_to_drop = int(n_na * na_drop_frac)
            na_indices = df_messy[na_mask].index
            drop_indices = np.random.choice(na_indices, size=n_to_drop, replace=False)
            df_messy = df_messy.drop(drop_indices).reset_index(drop=True)

            changes_log.append(f"Dropped {n_to_drop:,} NA rows ({na_drop_frac*100:.0f}% of NAs)")
            if verbose:
                print(f"  ✓ Dropped {n_to_drop:,} of {n_na:,} NA rows")
        elif verbose:
            print("  No NA rows found to drop")
    elif verbose:
        print("Step 4/6: Dropping NA rows... [SKIPPED]")

    # Step 5: Create internal gaps by dropping ZERO rows from middle of each series
    # OPTIMIZED: Uses vectorized operations instead of slow groupby-apply
    if zeros_drop_gaps_frac is not None and zeros_drop_gaps_frac > 0:
        if verbose:
            print("Step 5/6: Creating internal gaps (dropping middle zeros)...")

        n_before = len(df_messy)
        
        # Sort by id and date (required for identifying first/last per group)
        df_messy = df_messy.sort_values([id_col, date_col]).reset_index(drop=True)
        
        # Vectorized identification of first/last rows per group using shift
        is_first = df_messy[id_col] != df_messy[id_col].shift(1)
        is_last = df_messy[id_col] != df_messy[id_col].shift(-1)
        
        # Middle rows are neither first nor last AND have zero demand
        is_zero = df_messy[target_col] == 0
        is_middle_zero = ~is_first & ~is_last & is_zero
        middle_zero_indices = df_messy.index[is_middle_zero].values
        
        # Sample from middle zero indices
        n_to_drop = int(len(middle_zero_indices) * zeros_drop_gaps_frac)
        
        if n_to_drop > 0 and len(middle_zero_indices) > 0:
            drop_indices = np.random.choice(middle_zero_indices, size=n_to_drop, replace=False)
            df_messy = df_messy.drop(drop_indices).reset_index(drop=True)
        
        n_dropped = n_before - len(df_messy)

        changes_log.append(f"Created internal gaps: dropped {n_dropped:,} middle zeros ({zeros_drop_gaps_frac*100:.0f}%)")
        if verbose:
            print(f"  ✓ Dropped {n_dropped:,} zeros from middle of series")
    elif verbose:
        print("Step 5/6: Creating internal gaps... [SKIPPED]")

    # Step 6: Corrupt data types
    if dtypes_corrupt:
        if verbose:
            print("Step 6/6: Corrupting data types...")

        if date_col in df_messy.columns:
            df_messy[date_col] = df_messy[date_col].astype(str)
            changes_log.append(f"Converted {date_col} to string dtype")
            if verbose:
                print(f"  ✓ Converted {date_col} to string")

        if target_col in df_messy.columns:
            df_messy[target_col] = df_messy[target_col].astype(str)
            changes_log.append(f"Converted {target_col} to string dtype")
            if verbose:
                print(f"  ✓ Converted {target_col} to string")
    elif verbose:
        print("Step 6/6: Corrupting data types... [SKIPPED]")

    # Save to cache if requested
    if cache_dir is not None:
        if verbose:
            print(f"\n💾 Caching messified data...")
            print(f"   → {cache_path.name}")
        
        df_messy.to_parquet(cache_path, index=False)
        
        if verbose:
            cache_size_mb = cache_path.stat().st_size / 1024**2
            print(f"   ✓ Cached ({cache_size_mb:.1f} MB)")
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("DATA MESSIFICATION SUMMARY")
        print("=" * 70)
        print(f"\nOriginal shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Messified shape: {df_messy.shape[0]:,} rows × {df_messy.shape[1]} columns")
        
        if changes_log:
            print(f"\nChanges applied ({len(changes_log)}):")
            for i, change in enumerate(changes_log, 1):
                print(f"  {i}. {change}")
        else:
            print("\nNo changes applied (all steps skipped)")
        
        print("\n" + "=" * 70)
        print("✓ Data successfully messified!")
        print("=" * 70)
    
    return df_messy


def check_gaps(
    df: pd.DataFrame,
    id_col: str = 'unique_id',
    date_col: str = 'ds',
    freq: str = 'W'
) -> bool:
    """
    Diagnose gaps in time series data before filling.

    Compares actual rows per series vs expected rows based on the global
    date range and frequency. Useful to determine if fill_gaps is needed.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data with ID and date columns
    id_col : str, default='unique_id'
        Name of the series identifier column
    date_col : str, default='ds'
        Name of the date column
    freq : str, default='W'
        Expected frequency (e.g., 'W', 'W-SUN', 'D', 'M')

    Returns
    -------
    bool
        True if gaps exist and filling is needed, False otherwise

    Examples
    --------
    >>> # Check if gaps exist
    >>> needs_filling = check_gaps(df, freq='W-SUN')
    >>> if needs_filling:
    ...     df = fill_gaps(df, freq='W-SUN')

    >>> # With custom columns
    >>> needs_filling = check_gaps(df, id_col='series_id', date_col='date', freq='D')
    """
    # Get global date range
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    expected_periods = pd.date_range(min_date, max_date, freq=freq)
    n_expected = len(expected_periods)

    # Count actual rows per series
    actual_counts = df.groupby(id_col).size()
    n_series = len(actual_counts)

    # Series with gaps (fewer rows than expected)
    series_with_gaps = actual_counts[actual_counts < n_expected]
    n_series_with_gaps = len(series_with_gaps)
    total_missing = (n_expected - actual_counts).clip(lower=0).sum()

    print(f"Gap Analysis:")
    print(f"  Date range: {min_date.date()} to {max_date.date()}")
    print(f"  Expected periods per series: {n_expected}")
    print(f"  Series with gaps: {n_series_with_gaps:,} / {n_series:,}")
    print(f"  Total missing rows: {total_missing:,}")

    if n_series_with_gaps > 0:
        print(f"\nWorst offenders (most gaps):")
        gaps = n_expected - series_with_gaps
        for uid, gap_count in gaps.sort_values(ascending=False).head(5).items():
            print(f"  {uid}: {gap_count} missing periods")

    needs_filling = n_series_with_gaps > 0
    print(f"\nNeeds gap filling: {needs_filling}")

    return needs_filling


# ============================================================================
# SECTION 5: INSPECTION UTILITIES
# ============================================================================

def first_contact_check(df: pd.DataFrame, date_col: str = 'ds', target_col: str = 'y') -> bool:
    """
    Run all first-contact checks on a DataFrame.

    A simple, general-purpose data quality check function that validates
    common requirements for time series data.

    Checks:
    - Required columns exist
    - Data types (date is datetime, target is numeric)
    - NAs in target and ID columns
    - Impossible dates (<1900 or future)
    - Duplicates
    - Date range and series count

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check
    date_col : str, default='ds'
        Name of the date column
    target_col : str, default='y'
        Name of the target column

    Returns
    -------
    bool
        True if all checks pass, False otherwise

    Examples
    --------
    >>> df = load_m5(Path('data'))
    >>> passed = first_contact_check(df)
    >>> if not passed:
    ...     print("Data needs cleaning before modeling")
    """
    print("=" * 60)
    print("FIRST CONTACT CHECK")
    print("=" * 60)

    issues = []
    id_cols = [c for c in df.columns if c not in [date_col, target_col]]

    # 1. Required columns
    for col in [date_col, target_col]:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")
    if issues:
        print(f"✗ {issues}")
        return False
    print(f"✓ Required columns present ({date_col}, {target_col})")

    # 2. Data types
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        print(f"✓ {date_col} is datetime")
    else:
        issues.append(f"{date_col} is {df[date_col].dtype}, expected datetime")
        print(f"✗ {date_col} is {df[date_col].dtype}, expected datetime")

    if pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"✓ {target_col} is numeric")
    else:
        issues.append(f"{target_col} is {df[target_col].dtype}, expected numeric")
        print(f"✗ {target_col} is {df[target_col].dtype}, expected numeric")

    # 3. NAs
    na_target = df[target_col].isna().sum()
    na_date = df[date_col].isna().sum()
    na_ids = df[id_cols].isna().sum().sum() if id_cols else 0

    if na_date > 0:
        issues.append(f"{na_date:,} NAs in {date_col}")
        print(f"✗ {na_date:,} NAs in {date_col}")
    else:
        print(f"✓ No NAs in {date_col}")

    if na_ids > 0:
        issues.append(f"{na_ids:,} NAs in ID columns")
        print(f"✗ {na_ids:,} NAs in ID columns")
    else:
        print(f"✓ No NAs in ID columns")

    # Note: NA in target is just informational (handled in Module 1.10)
    print(f"ℹ {na_target:,} NAs in {target_col} (will impute in Module 1.10)")

    # 4. Impossible dates
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        today = pd.Timestamp.today()
        n_old = (df[date_col] < '1900-01-01').sum()
        n_future = (df[date_col] > today).sum()

        if n_old > 0:
            issues.append(f"{n_old:,} dates before 1900")
            print(f"✗ {n_old:,} dates before 1900")
        if n_future > 0:
            issues.append(f"{n_future:,} future dates")
            print(f"✗ {n_future:,} future dates")
        if n_old == 0 and n_future == 0:
            print(f"✓ No impossible dates")

    # 5. Duplicates
    non_target = [c for c in df.columns if c != target_col]
    n_dups = df.duplicated(subset=non_target).sum()
    if n_dups > 0:
        issues.append(f"{n_dups:,} duplicate rows")
        print(f"✗ {n_dups:,} duplicate rows")
    else:
        print(f"✓ No duplicates")

    # 6. Summary stats
    print(f"\nSummary:")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    if id_cols:
        n_series = df.groupby(id_cols).ngroups
        print(f"  Series: {n_series:,}")
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        print(f"  Date range: {df[date_col].min().date()} to {df[date_col].max().date()}")
        n_periods = df[date_col].nunique()
        print(f"  Unique dates: {n_periods:,}")

    # Memory
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"  Memory: {mem_mb:.1f} MB")

    # Result
    print("\n" + "=" * 60)
    if issues:
        print(f"⚠ {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("✓ ALL CHECKS PASSED")
    print("=" * 60)

    return len(issues) == 0


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'load_m5',
    'load_m5_with_feedback',
    'has_m5_cache',
    'create_subset',
    'create_unique_id',
    'messify_m5_data',
    'expand_hierarchy',
    'check_gaps',
    'first_contact_check',
    'HIERARCHY_COLS',
]


# ============================================================================
# MODULE TEST/DEMO
# ============================================================================

if __name__ == "__main__":
    """
    Test/demo the utilities.
    """
    print("M5 Utilities Module")
    print("=" * 70)
    print("\nAvailable functions:")
    print("\n  Loading:")
    print("    - load_m5(data_dir, verbose=True, messify=False,")
    print("              include_hierarchy=False, n_series=None)")
    print("        → Main entry point with preprocessing options")
    print("    - load_m5_with_feedback(data_dir, verbose=True, return_additional=False)")
    print("        → Low-level function with full control")
    print("    - has_m5_cache(data_dir)")
    print("        → Check if M5 is already cached")
    print("\n  Preprocessing:")
    print("    - expand_hierarchy(df, S_df=None)")
    print("        → Replace unique_id with item_id, dept_id, cat_id, store_id, state_id")
    print("\n  Subsetting:")
    print("    - create_subset(df, n_series=100, random_state=42)")
    print("        → Create smaller subset for faster iteration")
    print("\n  Messification:")
    print("    - messify_m5_data(df, cache_dir=None, ...)")
    print("        → Simulate real-world data quality issues")
    print("\n  Inspection:")
    print("    - check_gaps(df, id_col='unique_id', date_col='ds', freq='W')")
    print("        → Diagnose gaps in time series data")
    print("    - first_contact_check(df, date_col='ds', target_col='y')")
    print("        → Run data quality checks")

    print("\n" + "=" * 70)
    print("Example usage:")
    print("=" * 70)
    print("""
from pathlib import Path
from m5_utils import load_m5

DATA_DIR = Path('data')

# Basic load (daily, unique_id)
df = load_m5(DATA_DIR)

# With hierarchy columns instead of unique_id
df = load_m5(DATA_DIR, include_hierarchy=True)
# Columns: item_id, dept_id, cat_id, store_id, state_id, ds, y

# Messified for training exercises
df = load_m5(DATA_DIR, messify=True, include_hierarchy=True)

# Custom messification options (new consistent naming!)
df = load_m5(
    DATA_DIR,
    messify=True,
    messify_kwargs={
        'zeros_to_na_frac': 0.20,        # 20% of zeros → NA
        'zeros_drop_frac': 0.02,          # Drop 2% of zero rows
        'zeros_drop_gaps_frac': 0.10,     # Drop 10% of middle zeros (gaps)
        'duplicates_add_n': 200,          # Add 200 duplicates
        'na_drop_frac': None,             # Don't drop NAs
        'dtypes_corrupt': True,           # Corrupt dtypes
        'cache_dir': DATA_DIR
    },
    include_hierarchy=True
)
    """)
    print("=" * 70)

