# tsforge/plots/core/__init__.py
"""
tsforge.plots.core

Core plot implementations. These are the canonical functions used by notebooks.
"""

from .timeseries import plot_timeseries
from .distribution import plot_distribution
from .scatter import plot_scatter
from .pareto import plot_pareto
from .bar import plot_bar
from .panel import plot_panel

__all__ = [
    "plot_timeseries",
    "plot_distribution",
    "plot_scatter",
    "plot_pareto",
    "plot_bar",
    "plot_panel",
]
