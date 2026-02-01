# tsforge/plots/shortcuts/__init__.py
"""Convenience wrapper functions for common plotting tasks.

These functions are thin wrappers around core functions with
preset configurations for specific use cases.

EDA Shortcuts
-------------
plot_intermittency_scatter : ADI/CV2 scatter for demand classification
plot_zero_distribution : Zero percentage bar chart
plot_archetype_distribution : Archetype category distribution
plot_abc_distribution : ABC classification distribution

Evaluation Shortcuts
--------------------
plot_metric_interaction : 2D metric interaction with quadrants
plot_structure_chaos : Structure vs chaos score analysis
plot_portfolio_metrics : Multiple portfolio metric distributions
"""

from .eda import (
    plot_intermittency_scatter,
    plot_zero_distribution,
    plot_archetype_distribution,
    plot_abc_distribution,
)

from .evaluation import (
    plot_metric_interaction,
    plot_structure_chaos,
    plot_portfolio_metrics,
)

__all__ = [
    # EDA shortcuts
    "plot_intermittency_scatter",
    "plot_zero_distribution",
    "plot_archetype_distribution",
    "plot_abc_distribution",
    # Evaluation shortcuts
    "plot_metric_interaction",
    "plot_structure_chaos",
    "plot_portfolio_metrics",
]
