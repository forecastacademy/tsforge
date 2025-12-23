# tsforge/plots/_styling.py
"""
Consolidated styling module for tsforge plots.

Merges: core/palette.py, core/theme.py, style.py
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Any

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# =============================================================================
# COLOR PALETTE
# =============================================================================

PALETTE = [
    "#2E86AB", "#F6C85F", "#6F4E7C", "#C34A36",
    "#9FD356", "#235789", "#C1292E", "#7EA16B",
    "#436436", "#FFC145", "#8B85C1"
]

HIGHLIGHT = "#ff007f"

# Archetype colors (Structure × Chaos quadrants)
ARCHETYPE_COLORS = {
    'Stable': '#28A745',      # Green - easy, automate
    'Complex': '#007BFF',     # Blue - invest
    'Messy': '#FFC107',       # Amber - caution
    'Low Signal': '#DC3545',  # Red - difficult
}

# ABC segmentation colors
ABC_COLORS = {
    'A': '#1a5276',  # Dark blue - vital few
    'B': '#5dade2',  # Medium blue
    'C': '#aed6f1',  # Light blue - trivial many
}


def hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
    """Convert #RRGGBB to rgba(r,g,b,a)."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# =============================================================================
# PLOTLY THEMES
# =============================================================================

THEMES: Dict[str, Dict[str, Any]] = {
    "fa": {
        "font": "Source Sans Pro",
        "font_size": 14,
        "background": "white",
        "grid_color": "rgba(0,0,0,0.06)",
        "axis_color": "#4A4E69",
        "title_color": "#2C32D5",
        "subtitle_color": "#444444",
        "line_width": 2.3,
        "line_color": "#2C32D5",
        "pi_opacity": 0.20,
        "accent_color": "#E63946",
        "legend_bg": "rgba(255,255,255,0.80)",
        "legend_border": "rgba(180,180,180,0.35)",
        "legend_font_color": "#222222",
        "legend_position": "top right",
        "legend_font_size": 12,
    },
    "mckinsey": {
        "font": "Helvetica Neue",
        "font_size": 15,
        "background": "white",
        "grid_color": "#D9D9D9",
        "axis_color": "#2B2B2B",
        "title_color": "#1A1A1A",
        "subtitle_color": "#555555",
        "line_width": 2.8,
        "pi_opacity": 0.15,
        "legend_bg": "rgba(255,255,255,0.40)",
        "legend_border": "rgba(150,150,150,0.30)",
        "legend_font_color": "#222222",
        "legend_position": "top right",
        "legend_font_size": 13,
    },
    "minimal": {
        "font": "Inter",
        "font_size": 13,
        "background": "white",
        "grid_color": "rgba(0,0,0,0.07)",
        "axis_color": "#444444",
        "title_color": "#222222",
        "subtitle_color": "#666666",
        "line_width": 1.6,
        "pi_opacity": 0.10,
        "legend_bg": "rgba(255,255,255,0.55)",
        "legend_border": "rgba(210,210,210,0.20)",
        "legend_font_color": "#333333",
        "legend_position": "top right",
        "legend_font_size": 11,
    },
    "dark": {
        "font": "Arial",
        "font_size": 13,
        "background": "#111111",
        "grid_color": "#333333",
        "axis_color": "#E5E5E5",
        "title_color": "#F8F8F8",
        "subtitle_color": "#CCCCCC",
        "line_width": 2.2,
        "pi_opacity": 0.18,
        "legend_bg": "rgba(20,20,20,0.75)",
        "legend_border": "rgba(255,255,255,0.25)",
        "legend_font_color": "#F0F0F0",
        "legend_position": "top right",
        "legend_font_size": 12,
    },
    "seaborn": {
        "font": "Arial",
        "font_size": 13,
        "background": "white",
        "grid_color": "#E5E5E5",
        "axis_color": "#4C4C4C",
        "line_width": 2.0,
        "pi_opacity": 0.18,
        "legend_bg": "rgba(255,255,255,0.85)",
        "legend_border": "rgba(200,200,200,0.3)",
        "legend_position": "top right",
        "accent_color": "#4C72B0",
    },
    "ggplot": {
        "font": "Georgia",
        "font_size": 13,
        "background": "#EBEBEB",
        "grid_color": "white",
        "axis_color": "#333333",
        "line_width": 2.5,
        "pi_opacity": 0.20,
        "legend_bg": "rgba(245,245,245,0.9)",
        "legend_border": "rgba(120,120,120,0.3)",
        "legend_position": "bottom right",
        "accent_color": "#E24A33",
    },
}


def apply_theme(fig, theme: str = "fa"):
    """Apply global tsforge theme to a Plotly figure."""
    t = THEMES.get(theme, THEMES["fa"])

    fig.update_layout(
        font=dict(family=t["font"], size=t["font_size"]),
        plot_bgcolor=t["background"],
        paper_bgcolor=t["background"],
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=t["grid_color"],
        color=t["axis_color"],
        tickcolor=t["axis_color"],
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=t["grid_color"],
        color=t["axis_color"],
        tickcolor=t["axis_color"],
    )

    return fig


def apply_legend(fig, theme: str = "fa"):
    """Apply professional top-right translucent FA/McKinsey-style legend."""
    t = THEMES.get(theme, THEMES["fa"])

    legend_font_color = t.get("legend_font_color", t.get("axis_color", "#222"))
    legend_font_size = t.get("legend_font_size", 12)

    fig.update_layout(
        legend=dict(
            bgcolor=t.get("legend_bg", "rgba(255,255,255,0.65)"),
            bordercolor=t.get("legend_border", "rgba(200,200,200,0.4)"),
            borderwidth=1,
            font=dict(size=legend_font_size, color=legend_font_color),
            x=1.0, y=1.12,
            xanchor="right", yanchor="top",
            orientation="h",
        )
    )

    return fig


# =============================================================================
# MATPLOTLIB STYLE HELPERS
# =============================================================================

DEFAULT_STYLE = "seaborn-v0_8-whitegrid"


def apply_style(style: str = DEFAULT_STYLE, rc: Optional[Dict] = None) -> None:
    """Apply a default plotting style and optional rcParams overrides."""
    if plt is None:
        return
    plt.style.use(style)
    if rc:
        plt.rcParams.update(rc)


@contextmanager
def styled(style: str = DEFAULT_STYLE, rc: Optional[Dict] = None) -> Iterator[None]:
    """Context manager to temporarily apply style/rcParams."""
    if plt is None:
        yield
        return

    old = plt.rcParams.copy()
    try:
        apply_style(style=style, rc=rc)
        yield
    finally:
        plt.rcParams.update(old)


# Backward-compat alias
_apply_tsforge_style = apply_style
