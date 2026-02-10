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
"""
tsforge.plots

Public plotting API.

Keep this file intentionally small: only re-export stable, user-facing plot
functions that actually exist in the package.
"""

from .core import (
    plot_timeseries,
    plot_distribution,
    plot_scatter,
    plot_pareto,
    plot_bar,
    plot_panel,
)

__all__ = [
    "plot_timeseries",
    "plot_distribution",
    "plot_scatter",
    "plot_pareto",
    "plot_bar",
    "plot_panel",
]