"""
Portfolio architecture plots and utilities.

Functions for ABC segmentation, archetype assignment, and strategic visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple

# =============================================================================
# Color Schemes
# =============================================================================

ARCHETYPE_COLORS = {
    'Stable': '#28A745',      # Green - easy, automate
    'Complex': '#007BFF',     # Blue - invest
    'Messy': '#FFC107',       # Amber - caution
    'Low Signal': '#DC3545',  # Red - difficult
}

ABC_COLORS = {
    'A': '#1a5276',
    'B': '#5dade2', 
    'C': '#aed6f1'
}


# =============================================================================
# ABC Segmentation
# =============================================================================

def add_abc_classification(
    scores_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    a_pct: float = 0.80,
    b_pct: float = 0.95
) -> pd.DataFrame:
    """
    Add ABC classification based on both volume (units) and revenue ($).
    
    Creates two ABC classifications:
    - abc_volume: Based on total units sold
    - abc_revenue: Based on total dollar sales
    
    Why both? Forecast accuracy is measured on units, but
    business outcomes are measured in dollars.
    
    Parameters
    ----------
    scores_df : pd.DataFrame
        DataFrame with unique_id and scores from 1.10
    weekly_df : pd.DataFrame
        Weekly time series data with 'y' column and optionally 'sell_price' or 'revenue'
    a_pct : float
        Cumulative threshold for A class (default 0.80 = top 80%)
    b_pct : float
        Cumulative threshold for B class (default 0.95 = next 15%)
        
    Returns
    -------
    pd.DataFrame
        Input dataframe with added columns:
        - total_volume, total_revenue
        - cum_volume_pct, cum_revenue_pct
        - abc_volume, abc_revenue
    """
    # Total volume per series
    volume = weekly_df.groupby('unique_id')['y'].sum().reset_index()
    volume.columns = ['unique_id', 'total_volume']
    
    # Calculate revenue if possible
    if 'revenue' in weekly_df.columns:
        revenue = weekly_df.groupby('unique_id')['revenue'].sum().reset_index()
        revenue.columns = ['unique_id', 'total_revenue']
    elif 'sell_price' in weekly_df.columns:
        rev = weekly_df.groupby('unique_id').apply(
            lambda x: (x['y'] * x['sell_price']).sum()
        ).reset_index()
        rev.columns = ['unique_id', 'total_revenue']
        revenue = rev
    else:
        # Fallback: use volume as proxy for revenue
        revenue = volume.copy()
        revenue.columns = ['unique_id', 'total_revenue']
    
    # Merge
    df = scores_df.merge(volume, on='unique_id', how='left')
    df = df.merge(revenue, on='unique_id', how='left')
    df['total_volume'] = df['total_volume'].fillna(0)
    df['total_revenue'] = df['total_revenue'].fillna(0)
    
    # ABC by Volume
    df = df.sort_values('total_volume', ascending=False).reset_index(drop=True)
    total_vol = df['total_volume'].sum()
    df['cum_volume_pct'] = df['total_volume'].cumsum() / total_vol if total_vol > 0 else 0
    df['abc_volume'] = np.where(
        df['cum_volume_pct'] <= a_pct, 'A',
        np.where(df['cum_volume_pct'] <= b_pct, 'B', 'C')
    )
    
    # ABC by Revenue
    df = df.sort_values('total_revenue', ascending=False).reset_index(drop=True)
    total_rev = df['total_revenue'].sum()
    df['cum_revenue_pct'] = df['total_revenue'].cumsum() / total_rev if total_rev > 0 else 0
    df['abc_revenue'] = np.where(
        df['cum_revenue_pct'] <= a_pct, 'A',
        np.where(df['cum_revenue_pct'] <= b_pct, 'B', 'C')
    )
    
    return df


def compute_abc_mismatch(df: pd.DataFrame) -> Dict:
    """
    Analyze mismatch between volume-based and revenue-based ABC classification.
    
    Returns
    -------
    dict with:
        - mismatch_matrix: crosstab of abc_volume vs abc_revenue
        - alignment_pct: % of series with same ABC class
        - hidden_value: series that are B/C by volume but A by revenue
        - over_indexed: series that are A by volume but B/C by revenue
    """
    mismatch = pd.crosstab(
        df['abc_volume'], 
        df['abc_revenue'], 
        normalize='all'
    ) * 100
    mismatch = mismatch.reindex(index=['A', 'B', 'C'], columns=['A', 'B', 'C'], fill_value=0)
    
    # Alignment rate
    on_diagonal = mismatch.loc['A','A'] + mismatch.loc['B','B'] + mismatch.loc['C','C']
    
    # Hidden value: low volume but high revenue
    hidden_value = df[
        (df['abc_volume'].isin(['B', 'C'])) & 
        (df['abc_revenue'] == 'A')
    ]
    
    # Over-indexed: high volume but low revenue
    over_indexed = df[
        (df['abc_volume'] == 'A') & 
        (df['abc_revenue'].isin(['B', 'C']))
    ]
    
    return {
        'mismatch_matrix': mismatch,
        'alignment_pct': on_diagonal,
        'mismatch_pct': 100 - on_diagonal,
        'hidden_value': hidden_value,
        'over_indexed': over_indexed,
        'n_hidden_value': len(hidden_value),
        'n_over_indexed': len(over_indexed),
    }


# =============================================================================
# Archetype Assignment
# =============================================================================

def assign_archetypes(
    df: pd.DataFrame,
    structure_thresh: Optional[float] = None,
    chaos_thresh: Optional[float] = None
) -> Tuple[pd.DataFrame, float, float]:
    """
    Assign archetypes based on Structure × Chaos quadrant.
    
    Archetypes:
    - Complex: High structure, low chaos → Invest in ML
    - Messy: High structure, high chaos → Robust methods
    - Stable: Low structure, low chaos → Simple baseline
    - Low Signal: Low structure, high chaos → Aggregate up
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'structure_score' and 'chaos_score' columns
    structure_thresh : float, optional
        Threshold for high/low structure. Default: median
    chaos_thresh : float, optional
        Threshold for high/low chaos. Default: median
        
    Returns
    -------
    tuple of (df with 'archetype' column, structure_thresh, chaos_thresh)
    """
    if structure_thresh is None:
        structure_thresh = df['structure_score'].median()
    if chaos_thresh is None:
        chaos_thresh = df['chaos_score'].median()
    
    conditions = [
        (df['structure_score'] >= structure_thresh) & (df['chaos_score'] < chaos_thresh),
        (df['structure_score'] >= structure_thresh) & (df['chaos_score'] >= chaos_thresh),
        (df['structure_score'] < structure_thresh) & (df['chaos_score'] < chaos_thresh),
        (df['structure_score'] < structure_thresh) & (df['chaos_score'] >= chaos_thresh),
    ]
    choices = ['Complex', 'Messy', 'Stable', 'Low Signal']
    
    df = df.copy()
    df['archetype'] = np.select(conditions, choices, default='Unknown')
    
    return df, structure_thresh, chaos_thresh


def select_archetype_examples(
    df: pd.DataFrame,
    n_examples: int = 3,
    min_volume_pct: float = 0.3
) -> Dict[str, List[Dict]]:
    """
    Select clear examples of each archetype for illustration.
    
    Picks series that are:
    - Deep in their quadrant (not near boundaries)
    - Have reasonable volume (not degenerate)
    
    Parameters
    ----------
    df : pd.DataFrame
        Portfolio dataframe with archetype assignments
    n_examples : int
        Number of examples per archetype
    min_volume_pct : float
        Minimum volume percentile (0.3 = top 30%)
        
    Returns
    -------
    dict mapping archetype name to list of example dicts
    """
    volume_thresh = df['total_volume'].quantile(1 - min_volume_pct)
    examples = {}
    
    # Target positions deep in each quadrant
    targets = {
        'Stable': (0.25, 0.25),
        'Complex': (0.75, 0.25),
        'Messy': (0.75, 0.75),
        'Low Signal': (0.25, 0.75),
    }
    
    for arch, (target_struct, target_chaos) in targets.items():
        subset = df[(df['archetype'] == arch) & (df['total_volume'] >= volume_thresh)].copy()
        
        if len(subset) < n_examples:
            subset = df[df['archetype'] == arch].copy()
        
        if len(subset) == 0:
            examples[arch] = []
            continue
        
        # Distance to target
        subset['dist_to_target'] = np.sqrt(
            (subset['structure_score'] - target_struct)**2 + 
            (subset['chaos_score'] - target_chaos)**2
        )
        
        # Pick closest to target with good volume
        candidates = subset.nsmallest(20, 'dist_to_target')
        selected = candidates.nlargest(n_examples, 'total_volume')
        
        cols = ['unique_id', 'structure_score', 'chaos_score', 'total_volume']
        if 'dept_id' in selected.columns:
            cols.append('dept_id')
        if 'cat_id' in selected.columns:
            cols.append('cat_id')
            
        examples[arch] = selected[cols].to_dict('records')
    
    return examples


def select_heroes(
    df: pd.DataFrame,
    min_volume_pct: float = 0.5
) -> pd.DataFrame:
    """
    Select one high-volume representative series per archetype.
    
    Selection criteria:
    - Must be in top volume percentile
    - Closest to archetype centroid among candidates
    - Highest volume among finalists
    
    Parameters
    ----------
    df : pd.DataFrame
        Portfolio dataframe with archetype assignments
    min_volume_pct : float
        Minimum volume percentile (0.5 = top 50%)
        
    Returns
    -------
    pd.DataFrame with one row per archetype
    """
    volume_thresh = df['total_volume'].quantile(1 - min_volume_pct)
    heroes = []
    
    for arch in ['Stable', 'Complex', 'Messy', 'Low Signal']:
        subset = df[(df['archetype'] == arch) & (df['total_volume'] >= volume_thresh)]
        if len(subset) == 0:
            subset = df[df['archetype'] == arch]
        if len(subset) == 0:
            continue
        
        # Find centroid
        cx = subset['structure_score'].median()
        cy = subset['chaos_score'].median()
        
        # Distance to centroid
        subset = subset.copy()
        subset['dist'] = np.sqrt(
            (subset['structure_score'] - cx)**2 + 
            (subset['chaos_score'] - cy)**2
        )
        
        # Pick highest volume among closest to centroid
        hero = subset.nsmallest(10, 'dist').nlargest(1, 'total_volume').iloc[0]
        heroes.append({
            'archetype': arch,
            'unique_id': hero['unique_id'],
            'structure_score': hero['structure_score'],
            'chaos_score': hero['chaos_score'],
            'total_volume': hero['total_volume'],
        })
    
    return pd.DataFrame(heroes)


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_strategic_map(
    df: pd.DataFrame,
    struct_thresh: float,
    chaos_thresh: float,
    figsize: Tuple[int, int] = (11, 9),
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the 2×2 Structure × Chaos strategic map.
    
    Parameters
    ----------
    df : pd.DataFrame
        Portfolio dataframe with archetype assignments
    struct_thresh : float
        Structure score threshold
    chaos_thresh : float
        Chaos score threshold
    figsize : tuple
        Figure size
    ax : plt.Axes, optional
        Existing axes to plot on
        
    Returns
    -------
    tuple of (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    for arch in ['Stable', 'Complex', 'Messy', 'Low Signal']:
        mask = df['archetype'] == arch
        ax.scatter(
            df.loc[mask, 'structure_score'],
            df.loc[mask, 'chaos_score'],
            c=ARCHETYPE_COLORS[arch], 
            label=f"{arch} ({mask.sum():,})",
            alpha=0.4, s=12, edgecolors='none'
        )
    
    # Quadrant lines
    ax.axvline(struct_thresh, color='black', ls='--', lw=1.5, alpha=0.7)
    ax.axhline(chaos_thresh, color='black', ls='--', lw=1.5, alpha=0.7)
    
    # Quadrant labels
    ax.text(struct_thresh/2, chaos_thresh/2, 'STABLE\n(baseline)', 
            ha='center', va='center', fontsize=12, alpha=0.5, fontweight='bold')
    ax.text((1+struct_thresh)/2, chaos_thresh/2, 'COMPLEX\n(invest)', 
            ha='center', va='center', fontsize=12, alpha=0.5, fontweight='bold')
    ax.text((1+struct_thresh)/2, (1+chaos_thresh)/2, 'MESSY\n(robust)', 
            ha='center', va='center', fontsize=12, alpha=0.5, fontweight='bold')
    ax.text(struct_thresh/2, (1+chaos_thresh)/2, 'LOW SIGNAL\n(aggregate)', 
            ha='center', va='center', fontsize=12, alpha=0.5, fontweight='bold')
    
    ax.set_xlabel('Structure Score →', fontsize=12)
    ax.set_ylabel('Chaos Score →', fontsize=12)
    ax.set_title('Portfolio Architecture: Structure × Chaos', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig, ax


def plot_archetype_examples(
    examples: Dict[str, List[Dict]],
    weekly_df: pd.DataFrame,
    figsize: Tuple[int, int] = (15, 4)
) -> Dict[str, plt.Figure]:
    """
    Plot time series examples for each archetype.
    
    Parameters
    ----------
    examples : dict
        Output from select_archetype_examples()
    weekly_df : pd.DataFrame
        Weekly time series data
    figsize : tuple
        Figure size for each archetype's row of plots
        
    Returns
    -------
    dict mapping archetype name to figure
    """
    figures = {}
    
    descriptions = {
        'Stable': ('Low Structure, Low Chaos — "Flat and Quiet"', 
                   'Relatively flat, low variability, hovers around the mean'),
        'Complex': ('High Structure, Low Chaos — "Clear Patterns, Clean Signal"',
                    'Visible seasonality, trends, or repeating patterns'),
        'Messy': ('High Structure, High Chaos — "Pattern + Noise"',
                  'Some seasonality visible, but lots of spikes and variability'),
        'Low Signal': ('Low Structure, High Chaos — "Sparse and Random"',
                       'Intermittent, sparse, spiky, no discernible pattern'),
    }
    
    for arch, exs in examples.items():
        if not exs:
            continue
            
        n = len(exs)
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]
        
        for i, ex in enumerate(exs):
            ts = weekly_df[weekly_df['unique_id'] == ex['unique_id']].sort_values('ds')
            ax = axes[i]
            
            color = ARCHETYPE_COLORS[arch]
            ax.plot(ts['ds'], ts['y'], color=color, lw=1.2)
            ax.fill_between(ts['ds'], ts['y'], alpha=0.2, color=color)
            
            if arch == 'Stable':
                ax.axhline(ts['y'].mean(), color='gray', ls='--', alpha=0.5)
            
            # Title with scores
            uid_short = '_'.join(ex['unique_id'].split('_')[-2:])
            ax.set_title(f"{uid_short}\nS={ex['structure_score']:.2f}, C={ex['chaos_score']:.2f}",
                        fontsize=10)
            ax.set_ylabel('Weekly Sales')
        
        title, subtitle = descriptions[arch]
        fig.suptitle(f'{arch.upper()}: {title}', fontsize=13, fontweight='bold', 
                    color=ARCHETYPE_COLORS[arch])
        plt.tight_layout()
        figures[arch] = fig
    
    return figures


def plot_archetype_comparison(
    examples: Dict[str, List[Dict]],
    weekly_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot 2×2 comparison of all four archetypes.
    
    Parameters
    ----------
    examples : dict
        Output from select_archetype_examples()
    weekly_df : pd.DataFrame
        Weekly time series data
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    positions = {'Complex': (0,0), 'Messy': (0,1), 'Stable': (1,0), 'Low Signal': (1,1)}
    
    descriptions = {
        'Stable': 'Low Structure, Low Chaos\n→ Baseline works',
        'Complex': 'High Structure, Low Chaos\n→ Invest in ML',
        'Messy': 'High Structure, High Chaos\n→ Robust methods',
        'Low Signal': 'Low Structure, High Chaos\n→ Aggregate up',
    }
    
    for arch, (row, col) in positions.items():
        ax = axes[row, col]
        
        if arch not in examples or not examples[arch]:
            ax.text(0.5, 0.5, f'No {arch} examples', ha='center', va='center')
            continue
            
        ex = examples[arch][0]
        ts = weekly_df[weekly_df['unique_id'] == ex['unique_id']].sort_values('ds')
        
        color = ARCHETYPE_COLORS[arch]
        ax.plot(ts['ds'], ts['y'], color=color, lw=1.2)
        ax.fill_between(ts['ds'], ts['y'], alpha=0.3, color=color)
        
        ax.set_title(f"{arch.upper()}\n{descriptions[arch]}", 
                    fontsize=11, fontweight='bold', color=color)
        ax.set_ylabel('Weekly Sales')
        
        # Scores in corner
        ax.text(0.98, 0.98, f"S={ex['structure_score']:.2f}\nC={ex['chaos_score']:.2f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('The Four Archetypes — Real Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_heroes_on_map(
    df: pd.DataFrame,
    heroes: pd.DataFrame,
    struct_thresh: float,
    chaos_thresh: float,
    figsize: Tuple[int, int] = (11, 9)
) -> plt.Figure:
    """
    Plot strategic map with hero series highlighted.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full portfolio dataframe
    heroes : pd.DataFrame
        Output from select_heroes()
    struct_thresh : float
        Structure threshold
    chaos_thresh : float
        Chaos threshold
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
    """
    fig, ax = plot_strategic_map(df, struct_thresh, chaos_thresh, figsize=figsize)
    
    for _, h in heroes.iterrows():
        ax.scatter(
            h['structure_score'], h['chaos_score'], 
            marker='*', s=400,
            c=ARCHETYPE_COLORS[h['archetype']], 
            edgecolors='white', lw=2, zorder=10
        )
    
    ax.set_title('Portfolio Architecture with Hero Series', fontsize=14, fontweight='bold')
    return fig


def plot_hero_timeseries(
    heroes: pd.DataFrame,
    weekly_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot 2×2 facet of hero time series.
    
    Parameters
    ----------
    heroes : pd.DataFrame
        Output from select_heroes()
    weekly_df : pd.DataFrame
        Weekly time series data
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    positions = {'Complex': (0,0), 'Messy': (0,1), 'Stable': (1,0), 'Low Signal': (1,1)}
    
    for _, hero in heroes.iterrows():
        arch = hero['archetype']
        if arch not in positions:
            continue
        row, col = positions[arch]
        ax = axes[row, col]
        
        ts = weekly_df[weekly_df['unique_id'] == hero['unique_id']].sort_values('ds')
        
        color = ARCHETYPE_COLORS[arch]
        ax.plot(ts['ds'], ts['y'], color=color, lw=1)
        ax.fill_between(ts['ds'], ts['y'], alpha=0.2, color=color)
        ax.set_title(f"{arch}\nStruct={hero['structure_score']:.2f}, Chaos={hero['chaos_score']:.2f}",
                    fontweight='bold', color=color)
        ax.set_ylabel('Weekly Sales')
        ax.text(0.02, 0.98, hero['unique_id'], transform=ax.transAxes, 
                fontsize=8, va='top', alpha=0.6, family='monospace')
    
    fig.suptitle('Hero Series by Archetype (Raw Data — No Smoothers)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_portfolio_composition(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (13, 5)
) -> plt.Figure:
    """
    Plot side-by-side bars showing % SKUs vs % Volume by archetype.
    
    Parameters
    ----------
    df : pd.DataFrame
        Portfolio dataframe with archetype assignments
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    arch_order = ['Stable', 'Complex', 'Messy', 'Low Signal']
    colors = [ARCHETYPE_COLORS[a] for a in arch_order]
    
    # SKU count
    sku_pct = df['archetype'].value_counts(normalize=True).reindex(arch_order).fillna(0)
    axes[0].barh(arch_order, sku_pct.values, color=colors)
    axes[0].set_xlabel('% of SKUs')
    axes[0].set_title('By SKU Count', fontweight='bold')
    axes[0].set_xlim(0, 0.6)
    for i, v in enumerate(sku_pct.values):
        axes[0].text(v + 0.01, i, f'{v:.1%}', va='center')
    
    # Volume
    vol_by_arch = df.groupby('archetype')['total_volume'].sum()
    vol_pct = (vol_by_arch / vol_by_arch.sum()).reindex(arch_order).fillna(0)
    axes[1].barh(arch_order, vol_pct.values, color=colors)
    axes[1].set_xlabel('% of Volume')
    axes[1].set_title('By Volume', fontweight='bold')
    axes[1].set_xlim(0, 0.6)
    for i, v in enumerate(vol_pct.values):
        axes[1].text(v + 0.01, i, f'{v:.1%}', va='center')
    
    fig.suptitle('Portfolio Composition: Where Are the SKUs vs Where Is the Money?', 
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_abc_mismatch_heatmap(
    mismatch_matrix: pd.DataFrame,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot heatmap of volume vs revenue ABC classification.
    
    Parameters
    ----------
    mismatch_matrix : pd.DataFrame
        Crosstab from compute_abc_mismatch()
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        mismatch_matrix, annot=True, fmt='.1f', cmap='Blues',
        cbar_kws={'label': '% of Portfolio'}, ax=ax,
        linewidths=0.5, linecolor='white'
    )
    
    ax.set_xlabel('ABC by Revenue (Business View)', fontsize=12)
    ax.set_ylabel('ABC by Volume (Forecast View)', fontsize=12)
    ax.set_title('Volume vs Revenue: Where Do They Disagree?', fontsize=13, fontweight='bold')
    
    # Highlight diagonal
    for i in range(3):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='green', lw=3))
    
    plt.tight_layout()
    return fig


def plot_risk_matrix(
    df: pd.DataFrame,
    group_col: str = 'dept_id',
    top_n: int = 10,
    figsize: Tuple[int, int] = (12, 7)
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Plot stacked bar showing % volume by archetype for each group.
    
    Parameters
    ----------
    df : pd.DataFrame
        Portfolio dataframe
    group_col : str
        Column to group by (e.g., 'dept_id', 'cat_id')
    top_n : int
        Number of top groups to show
    figsize : tuple
        Figure size
        
    Returns
    -------
    tuple of (fig, risk_matrix DataFrame)
    """
    # Top groups by volume
    top_groups = df.groupby(group_col)['total_volume'].sum().nlargest(top_n).index
    df_top = df[df[group_col].isin(top_groups)]
    
    # Pivot
    pivot = df_top.groupby([group_col, 'archetype'])['total_volume'].sum().unstack(fill_value=0)
    pivot = pivot.reindex(columns=['Stable', 'Complex', 'Messy', 'Low Signal'], fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)
    
    # Sort by total volume
    order = df_top.groupby(group_col)['total_volume'].sum().sort_values(ascending=True).index
    pivot_pct = pivot_pct.reindex(order)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    pivot_pct.plot(
        kind='barh', stacked=True, ax=ax,
        color=[ARCHETYPE_COLORS[c] for c in pivot_pct.columns],
        edgecolor='white', lw=0.5
    )
    
    ax.set_xlabel('% of Volume')
    ax.set_ylabel('')
    ax.set_title(f'Risk Matrix: {group_col} × Archetype', fontsize=14, fontweight='bold')
    ax.legend(title='Archetype', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    plt.tight_layout()
    return fig, pivot_pct


def plot_risk_heatmap(
    risk_matrix: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 7)
) -> plt.Figure:
    """
    Plot heatmap version of risk matrix.
    
    Parameters
    ----------
    risk_matrix : pd.DataFrame
        Output from plot_risk_matrix()
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        risk_matrix * 100, annot=True, fmt='.0f', cmap='YlOrRd',
        cbar_kws={'label': '% of Volume'}, ax=ax
    )
    ax.set_title('Risk Matrix: Department × Archetype', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_department_aggregates(
    weekly_df: pd.DataFrame,
    top_n: int = 4,
    figsize: Tuple[int, int] = (14, 9)
) -> plt.Figure:
    """
    Plot aggregated time series for top departments.
    
    Includes guardrail warning that aggregation hides chaos.
    
    Parameters
    ----------
    weekly_df : pd.DataFrame
        Weekly time series data
    top_n : int
        Number of top departments to show
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
    """
    top_depts = weekly_df.groupby('dept_id')['y'].sum().nlargest(top_n).index.tolist()
    
    nrows = (top_n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, dept in enumerate(top_depts):
        dept_ts = weekly_df[weekly_df['dept_id'] == dept].groupby('ds')['y'].sum().reset_index()
        axes[i].plot(dept_ts['ds'], dept_ts['y'], color='#2c3e50', lw=1)
        axes[i].fill_between(dept_ts['ds'], dept_ts['y'], alpha=0.2, color='#2c3e50')
        axes[i].set_title(dept, fontweight='bold')
        axes[i].set_ylabel('Total Weekly Sales')
    
    # Hide unused axes
    for i in range(top_n, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle('Department Aggregates (Scale Context — NOT for Classification)', 
                fontsize=13, fontweight='bold', color='#dc3545')
    plt.tight_layout()
    return fig


# =============================================================================
# Insight Generation
# =============================================================================

def generate_risk_insights(
    risk_matrix: pd.DataFrame,
    threshold: float = 0.4
) -> List[str]:
    """
    Generate actionable recommendations from risk matrix.
    
    Parameters
    ----------
    risk_matrix : pd.DataFrame
        Output from plot_risk_matrix()
    threshold : float
        Minimum % to trigger recommendation
        
    Returns
    -------
    list of insight strings
    """
    insights = []
    
    for group in risk_matrix.index:
        row = risk_matrix.loc[group]
        
        if row['Stable'] >= 0.5:
            insights.append(f"{group}: {row['Stable']:.0%} Stable → Automate with baselines")
        elif row['Complex'] >= threshold:
            insights.append(f"{group}: {row['Complex']:.0%} Complex → Invest in ML")
        elif row['Messy'] >= threshold:
            insights.append(f"{group}: {row['Messy']:.0%} Messy → Human-in-loop governance")
        elif row['Low Signal'] >= threshold:
            insights.append(f"{group}: {row['Low Signal']:.0%} Low Signal → Consider aggregation")
        elif (row['Messy'] + row['Low Signal']) >= 0.5:
            pct = row['Messy'] + row['Low Signal']
            insights.append(f"{group}: {pct:.0%} difficult → Risk management focus")
    
    return insights


def print_abc_summary(df: pd.DataFrame) -> None:
    """Print formatted ABC distribution summary."""
    
    # Volume summary
    vol_summary = df.groupby('abc_volume').agg(
        n_series=('unique_id', 'count'),
        volume=('total_volume', 'sum')
    ).reindex(['A', 'B', 'C'])
    vol_summary['pct_series'] = vol_summary['n_series'] / vol_summary['n_series'].sum()
    vol_summary['pct_volume'] = vol_summary['volume'] / vol_summary['volume'].sum()
    
    # Revenue summary
    rev_summary = df.groupby('abc_revenue').agg(
        n_series=('unique_id', 'count'),
        revenue=('total_revenue', 'sum')
    ).reindex(['A', 'B', 'C'])
    rev_summary['pct_series'] = rev_summary['n_series'] / rev_summary['n_series'].sum()
    rev_summary['pct_revenue'] = rev_summary['revenue'] / rev_summary['revenue'].sum()
    
    print("ABC by VOLUME (what forecasts are measured on)")
    print("=" * 55)
    print(f"{'Class':<6} {'Series':>10} {'% Series':>12} {'% Volume':>12}")
    print("-" * 55)
    for cls in ['A', 'B', 'C']:
        r = vol_summary.loc[cls]
        print(f"{cls:<6} {int(r['n_series']):>10,} {r['pct_series']:>12.1%} {r['pct_volume']:>12.1%}")
    
    print("\n")
    print("ABC by REVENUE (what the business is measured on)")
    print("=" * 55)
    print(f"{'Class':<6} {'Series':>10} {'% Series':>12} {'% Revenue':>12}")
    print("-" * 55)
    for cls in ['A', 'B', 'C']:
        r = rev_summary.loc[cls]
        print(f"{cls:<6} {int(r['n_series']):>10,} {r['pct_series']:>12.1%} {r['pct_revenue']:>12.1%}")


def print_mismatch_summary(mismatch_info: Dict) -> None:
    """Print formatted mismatch analysis."""
    
    print("Items Where Volume ≠ Revenue Priority")
    print("=" * 60)
    print(f"\n🔵 HIDDEN VALUE ({mismatch_info['n_hidden_value']:,} items)")
    print(f"   Volume: B/C class, Revenue: A class")
    print(f"   → Lower unit volume but HIGH dollar impact")
    print(f"   → Forecast errors here hurt the business more than accuracy metrics show")
    
    print(f"\n🟡 OVER-INDEXED ({mismatch_info['n_over_indexed']:,} items)")
    print(f"   Volume: A class, Revenue: B/C class")
    print(f"   → High unit volume but LOWER dollar impact")
    print(f"   → Optimizing accuracy here may not move business KPIs")
    
    print(f"\nAlignment: {mismatch_info['alignment_pct']:.1f}% same ABC class")
    print(f"Mismatch:  {mismatch_info['mismatch_pct']:.1f}% differ")
