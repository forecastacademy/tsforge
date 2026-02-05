# tsforge/plots/_display.py
"""
Unified display modes for tsforge plots.

Provides a single entry point for rendering traces in overlay, facet, or dropdown mode.
Eliminates duplicate mode logic across chart files.
"""

from __future__ import annotations

from math import ceil
from typing import Any, Dict, List, Literal, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._layout import build_dropdown_buttons, finalize_figure

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def render_by_mode(
    traces_by_id: Dict[str, List[go.BaseTraceType]],
    mode: Literal["overlay", "facet", "dropdown"] = "overlay",
    *,
    # Facet options
    wrap: int = 2,
    shared_xaxes: bool = True,
    row_height: int = 280,
    col_width: Optional[int] = None,
    vertical_spacing: float = 0.08,
    # Finalization
    theme: str = "fa",
    style: Optional[Dict[str, Any]] = None,
    finalize: bool = True,
    # Extra elements
    shapes: Optional[List[Any]] = None,
    annotations: Optional[List[Any]] = None,
) -> go.Figure:
    ...
    if mode == "overlay":
        fig = _render_overlay(traces_by_id)
    elif mode == "facet":
        fig = _render_facet(
            traces_by_id, wrap, shared_xaxes, row_height, col_width, vertical_spacing
        )
    elif mode == "dropdown":
        fig = _render_dropdown(traces_by_id, row_height, col_width)
    else:
        raise ValueError(f"mode must be 'overlay', 'facet', or 'dropdown', got '{mode}'")

    # ✅ Facet default: suppress legend (facet titles already label each subplot)
    # Allow override via style={"showlegend": True}
    style = dict(style) if style else {}
    if mode == "facet" and "showlegend" not in style:
        style["showlegend"] = False
        # extra safety: kill per-trace legends too
        for tr in fig.data:
            tr.showlegend = False
        fig.update_layout(showlegend=False)

    # Add shapes/annotations
    if shapes:
        fig.update_layout(shapes=shapes)
    if annotations:
        fig.update_layout(annotations=annotations)

    if finalize:
        return finalize_figure(fig, theme=theme, style=style)
    return fig


# =============================================================================
# MODE IMPLEMENTATIONS
# =============================================================================


def _render_overlay(traces_by_id: Dict[str, List]) -> go.Figure:
    """
    All series in a single plot.

    Best for: comparing trends across series, seeing relative magnitudes.
    """
    fig = go.Figure()

    for uid, traces in traces_by_id.items():
        for trace in traces:
            fig.add_trace(trace)

    return fig


def _render_facet(
    traces_by_id: Dict[str, List],
    wrap: int,
    shared_xaxes: bool,
    row_height: int,
    col_width: Optional[int],
    vertical_spacing: float,
) -> go.Figure:
    """
    Vertically stacked subplots, optionally wrapped into columns.

    Best for: detailed view of each series, spotting series-specific patterns.
    """
    ids = list(traces_by_id.keys())
    n = len(ids)
    cols = min(wrap, n)
    rows = ceil(n / cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=shared_xaxes,
        vertical_spacing=vertical_spacing,
        subplot_titles=[str(uid).replace("_", " ").title() for uid in ids],
    )

    # Track which trace names we've seen for legend deduplication
    seen_names = set()

    for i, (uid, traces) in enumerate(traces_by_id.items()):
        r = (i // cols) + 1
        c = (i % cols) + 1

        for trace in traces:
            # Deduplicate legend entries across facets
            trace_name = getattr(trace, "name", None)
            if trace_name and trace_name in seen_names:
                trace.showlegend = False
            elif trace_name:
                seen_names.add(trace_name)

            fig.add_trace(trace, row=r, col=c)

    # Set dimensions
    layout_updates = {"height": row_height * rows}
    if col_width is not None:
        layout_updates["width"] = col_width * cols
    fig.update_layout(**layout_updates)

    return fig


def _render_dropdown(
    traces_by_id: Dict[str, List], row_height: int = 400, col_width: Optional[int] = None
) -> go.Figure:
    """
    Single plot with dropdown series selector.

    Best for: many series, interactive exploration, dashboards.
    """
    fig = go.Figure()
    ids = list(traces_by_id.keys())
    trace_map = {}  # {uid: [trace_indices]}

    for i, (uid, traces) in enumerate(traces_by_id.items()):
        visible = i == 0  # First series visible by default
        trace_map[uid] = []

        for trace in traces:
            trace.visible = visible
            # Keep legend for visible series only
            if not visible:
                trace.showlegend = False
            fig.add_trace(trace)
            trace_map[uid].append(len(fig.data) - 1)

    # Build dropdown buttons
    buttons = build_dropdown_buttons(trace_map, len(fig.data))
    layout_updates = {
        "updatemenus": [
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "y": 1.12,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "white",
                "bordercolor": "#ccc",
                "font": {"size": 12},
            }
        ],
        "height": row_height,
        "margin": dict(t=80),  # Top margin for dropdown
    }
    if col_width is not None:
        layout_updates["width"] = col_width
    fig.update_layout(**layout_updates)

    return fig


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def overlay(traces_by_id: Dict[str, List], **kwargs) -> go.Figure:
    """Shortcut for render_by_mode(..., mode="overlay")."""
    return render_by_mode(traces_by_id, mode="overlay", **kwargs)


def facet(traces_by_id: Dict[str, List], **kwargs) -> go.Figure:
    """Shortcut for render_by_mode(..., mode="facet")."""
    return render_by_mode(traces_by_id, mode="facet", **kwargs)


def dropdown(traces_by_id: Dict[str, List], **kwargs) -> go.Figure:
    """Shortcut for render_by_mode(..., mode="dropdown")."""
    return render_by_mode(traces_by_id, mode="dropdown", **kwargs)
