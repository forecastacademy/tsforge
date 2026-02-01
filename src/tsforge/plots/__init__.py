# tsforge/plots/__init__.py
"""
Visualization module for tsforge.

Structure:
- _styling.py: Colors, themes, matplotlib style
- _preprocessing.py: Data preprocessing utilities
- _layout.py: Shared layout builders (dropdown, legend, finalize)
- _display.py: Unified mode rendering (overlay/facet/dropdown)
- core/: Core plotting functions (6 main functions)
- shortcuts/: Convenience wrappers for common tasks
- eda/: Exploratory data analysis plots (legacy, re-exports from core)
"""

from __future__ import annotations

# ============================================================================
# Styling exports (public API)
# ============================================================================
from ._styling import (
    HIGHLIGHT,
    PALETTE,
    THEMES,
    apply_legend,
    apply_style,
    apply_theme,
    hex_to_rgba,
    styled,
)

# ============================================================================
# Core plots (new canonical imports)
# ============================================================================
from .core import (
    plot_2d_histogram,
    plot_autocorrelation,
    plot_bar,
    plot_calendar_heatmap,
    plot_category_counts,
    plot_distribution,
    plot_heatmap,
    plot_scatter,
    plot_scatter_matrix,
    plot_timeseries,
)

# ============================================================================
# EDA plots (legacy imports for backward compatibility)
# ============================================================================
from .eda import (
    plot_date_coverage,
    plot_decomposition,
    plot_demand_bars,
    plot_intermittency,
    plot_seasonal,
    plot_skewness,
)

# ============================================================================
# Shortcut wrappers (convenience functions)
# ============================================================================
from .shortcuts import (
    plot_abc_distribution,
    plot_archetype_distribution,
    plot_intermittency_scatter,
    plot_portfolio_metrics,
    plot_structure_chaos,
    plot_zero_distribution,
)

# ============================================================================
# Public API
# ============================================================================
__all__ = [
    # Styling
    "PALETTE",
    "HIGHLIGHT",
    "THEMES",
    "apply_style",
    "styled",
    "apply_theme",
    "apply_legend",
    "hex_to_rgba",
    # Core plots
    "plot_timeseries",
    "plot_distribution",
    "plot_heatmap",
    "plot_calendar_heatmap",
    "plot_autocorrelation",
    "plot_bar",
    "plot_category_counts",
    "plot_scatter",
    "plot_scatter_matrix",
    "plot_2d_histogram",
    # Shortcuts
    "plot_intermittency_scatter",
    "plot_zero_distribution",
    "plot_archetype_distribution",
    "plot_abc_distribution",
    "plot_structure_chaos",
    "plot_portfolio_metrics",
    # EDA (legacy)
    "plot_seasonal",
    "plot_skewness",
    "plot_decomposition",
    "plot_intermittency",
    "plot_demand_bars",
    "plot_date_coverage",
]
