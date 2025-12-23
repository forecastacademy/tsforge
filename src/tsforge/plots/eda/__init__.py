# tsforge/plots/eda/__init__.py
"""EDA (Exploratory Data Analysis) visualizations."""
from __future__ import annotations

from .timeseries import plot_timeseries
from .seasonal import plot_seasonal
from .distribution import (
    plot_distribution,
    plot_skewness,
    plot_category_distribution,
)
from .decomposition import (
    plot_decomposition,
    stl_decompose,
    decompose_series,
    get_decomposition_stats,
)
from .intermittency import (
    plot_intermittency,
    plot_demand_bars,
    classify_intermittency,
    compute_intermittency_stats,
    get_zero_stats,
)
from .coverage import plot_date_coverage
from .correlation import plot_autocorrelation
from .calendar import plot_calendar_heatmap

__all__ = [
    # Timeseries
    "plot_timeseries",
    # Seasonal
    "plot_seasonal",
    # Distribution
    "plot_distribution",
    "plot_skewness",
    "plot_category_distribution",
    # Decomposition
    "plot_decomposition",
    "stl_decompose",
    "decompose_series",
    "get_decomposition_stats",
    # Intermittency
    "plot_intermittency",
    "plot_demand_bars",
    "classify_intermittency",
    "compute_intermittency_stats",
    "get_zero_stats",
    # Coverage
    "plot_date_coverage",
    # Correlation
    "plot_autocorrelation",
    # Calendar
    "plot_calendar_heatmap",
]
