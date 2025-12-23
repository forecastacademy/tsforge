# tsforge/plots/__init__.py
"""
Visualization module for tsforge.

Structure:
- _styling.py: Colors, themes, matplotlib style
- _preprocessing.py: Data preprocessing utilities
- _layout.py: Shared layout builders (dropdown, legend, finalize)
- eda/: Exploratory data analysis plots
- evaluation/: Model evaluation and diagnostic plots
"""
from __future__ import annotations

# ============================================================================
# Styling exports (public API)
# ============================================================================
from ._styling import (
    PALETTE,
    HIGHLIGHT,
    THEMES,
    apply_style,
    styled,
    apply_theme,
    apply_legend,
    hex_to_rgba,
)

# ============================================================================
# EDA plots
# ============================================================================
from .eda import (
    plot_timeseries,
    plot_seasonal,
    plot_distribution,
    plot_skewness,
    plot_decomposition,
    plot_intermittency,
    plot_demand_bars,
    plot_date_coverage,
    plot_autocorrelation,
    plot_calendar_heatmap,
)

# ============================================================================
# Evaluation plots
# ============================================================================
from .evaluation import (
    plot_cv_plan,
    plot_cv_results,
    plot_strength,
    plot_diagnostics,
    plot_score_distributions,
    plot_metric_interaction,
    plot_metric_distribution,
    plot_demand_classification,
    plot_portfolio_characteristics,
    plot_structure_chaos_summary,
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
    # EDA
    "plot_timeseries",
    "plot_seasonal",
    "plot_distribution",
    "plot_skewness",
    "plot_decomposition",
    "plot_intermittency",
    "plot_demand_bars",
    "plot_date_coverage",
    "plot_autocorrelation",
    "plot_calendar_heatmap",
    # Evaluation
    "plot_cv_plan",
    "plot_cv_results",
    "plot_strength",
    "plot_diagnostics",
    "plot_score_distributions",
    "plot_metric_interaction",
    "plot_metric_distribution",
    "plot_demand_classification",
    "plot_portfolio_characteristics",
    "plot_structure_chaos_summary",
]
