# tsforge/plots/core/__init__.py
"""Core plotting functions for tsforge.

This module contains the 6 core plotting functions that handle
overlay/facet/dropdown modes using the unified render_by_mode pattern.

Core Functions
--------------
plot_timeseries : Time series visualization with forecasts, events, anomalies
plot_distribution : Distribution plots (histogram, density, box, violin)
plot_heatmap : Heatmap visualizations including calendar heatmaps
plot_autocorrelation : ACF/PACF correlation analysis (specialized layout)
plot_bar : Bar chart visualizations with grouping and stacking
plot_scatter : Scatter plots with optional quadrant analysis
"""

from .timeseries import plot_timeseries
from .distribution import plot_distribution
from .heatmap import plot_heatmap, plot_calendar_heatmap
from .correlation import plot_autocorrelation
from .bar import plot_bar, plot_category_counts
from .scatter import plot_scatter, plot_scatter_matrix, plot_2d_histogram, plot_demand_classification

__all__ = [
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
    "plot_demand_classification",
]
