# tsforge/plots/evaluation/__init__.py
"""Evaluation and diagnostic visualizations."""
from __future__ import annotations

from .cv import plot_cv_plan, plot_cv_results
from .strength import plot_strength
from .diagnostics import (
    plot_diagnostics,
    plot_score_distributions,
    plot_metric_interaction,
    plot_metric_distribution,
    plot_demand_classification,
    plot_portfolio_characteristics,
    plot_structure_chaos_summary,
)

__all__ = [
    # Cross-validation
    "plot_cv_plan",
    "plot_cv_results",
    # Strength
    "plot_strength",
    # Diagnostics
    "plot_diagnostics",
    "plot_score_distributions",
    "plot_metric_interaction",
    "plot_metric_distribution",
    "plot_demand_classification",
    "plot_portfolio_characteristics",
    "plot_structure_chaos_summary",
]
