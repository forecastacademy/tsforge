# tsforge/plots/_layout.py
"""
Consolidated layout utilities for tsforge plots.

Contains:
- Figure finalization (theme, style, legend positioning)
- Dropdown button building
- Mode assembly (overlay, facet, dropdown)
- Event line/label rendering
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Callable

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._styling import apply_theme, apply_legend, THEMES


# =============================================================================
# FIGURE FINALIZATION
# =============================================================================

def finalize_figure(
    fig: go.Figure,
    theme: str = "fa",
    style: Optional[Dict[str, Any]] = None,
    base_margin_bottom: int = 60,
) -> go.Figure:
    """
    Apply theme, style overrides, and auto-calculate legend margins.

    This replaces ~30 lines of duplicated code across chart files.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to finalize.
    theme : str
        Theme name (fa, mckinsey, minimal, dark, seaborn, ggplot).
    style : dict, optional
        Style overrides with keys: title, subtitle, x_title, y_title, y_range.
    base_margin_bottom : int
        Base bottom margin before legend rows are added.

    Returns
    -------
    go.Figure
        The finalized figure.
    """
    fig = apply_theme(fig, theme)

    if style:
        if "title" in style:
            fig.update_layout(title={"text": style["title"], "font": {"size": 18}})
        if "subtitle" in style:
            fig.add_annotation(
                x=0, y=1.06,
                xref="paper", yref="paper",
                text=style["subtitle"],
                showarrow=False,
                font=dict(size=13, color="#444"),
            )
        if "x_title" in style:
            fig.update_xaxes(title_text=style["x_title"])
        if "y_title" in style:
            fig.update_yaxes(title_text=style["y_title"])
        if "y_range" in style:
            fig.update_yaxes(range=style["y_range"])

    fig = apply_legend(fig, theme)

    # Auto-calculate legend rows and bottom margin
    num_legend_items = sum(1 for trace in fig.data if trace.showlegend)
    legend_rows = max(1, (num_legend_items + 5) // 6)
    bottom_margin = base_margin_bottom + (legend_rows * 25)

    fig.update_layout(
        legend=dict(
            orientation='h',
            x=0.5,
            y=-0.12,
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1,
        ),
        margin=dict(b=bottom_margin),
    )

    return fig


# =============================================================================
# DROPDOWN BUTTON BUILDING
# =============================================================================

def build_dropdown_buttons(
    trace_map: Dict[str, List[int]],
    total_traces: int,
    label_fn: Optional[Callable[[str], str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build standardized dropdown menu configuration.

    Parameters
    ----------
    trace_map : dict
        Mapping of {uid: [trace_indices]}.
    total_traces : int
        Total number of traces in the figure.
    label_fn : callable, optional
        Function to format button labels. Defaults to str(uid).

    Returns
    -------
    list
        List of button configurations for updatemenus.
    """
    buttons = []
    for uid, trace_idxs in trace_map.items():
        visibility = [False] * total_traces
        for idx in trace_idxs:
            visibility[idx] = True

        label = label_fn(uid) if label_fn else str(uid)
        buttons.append(dict(
            label=label,
            method="update",
            args=[{"visible": visibility}],
        ))

    return buttons


def apply_dropdown_menu(
    fig: go.Figure,
    buttons: List[Dict[str, Any]],
    x: float = 1.0,
    y: float = 1.15,
) -> go.Figure:
    """Apply a dropdown menu to a figure."""
    fig.update_layout(
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "x": x, "y": y,
            "xanchor": "right",
            "yanchor": "top",
        }]
    )
    return fig


# =============================================================================
# MODE ASSEMBLY
# =============================================================================

def assemble_overlay_mode(
    traces: List[Any],
    shapes: Optional[List[Any]] = None,
    annotations: Optional[List[Any]] = None,
    title: str = "",
) -> go.Figure:
    """Return a standard overlay figure."""
    fig = go.Figure()

    for tr in traces:
        fig.add_trace(tr)

    if shapes:
        fig.update_layout(shapes=shapes)

    if annotations:
        fig.update_layout(annotations=annotations)

    if title:
        fig.update_layout(title=title)

    return fig


def assemble_facet_mode(
    facet_map: Dict[str, List[Any]],
    shapes: Optional[List[Any]] = None,
    annotations_top: Optional[List[Any]] = None,
    title: str = "",
    row_height: int = 260,
) -> go.Figure:
    """
    Build a vertically stacked subplot with shared X-axis.

    Parameters
    ----------
    facet_map : dict
        Mapping of {series_name: [trace1, trace2, ...]}.
    shapes : list, optional
        Shapes to add to all rows.
    annotations_top : list, optional
        Annotations to add at the top.
    title : str
        Figure title.
    row_height : int
        Height per row in pixels.
    """
    n = len(facet_map)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=list(facet_map.keys()),
    )

    for row_idx, (uid, tr_list) in enumerate(facet_map.items(), start=1):
        for tr in tr_list:
            fig.add_trace(tr, row=row_idx, col=1)

    if shapes:
        annotated_shapes = []
        for s in shapes:
            for r in range(1, n + 1):
                s2 = s.copy()
                s2["xref"] = f"x{r}"
                s2["yref"] = f"y{r}"
                annotated_shapes.append(s2)
        fig.update_layout(shapes=annotated_shapes)

    if annotations_top:
        fig.update_layout(annotations=annotations_top)

    if title:
        fig.update_layout(title=title)

    fig.update_layout(height=row_height * n)
    return fig


def assemble_dropdown_mode(
    traces: Dict[str, List[Any]],
    shapes: Optional[List[Any]] = None,
    annotations: Optional[List[Any]] = None,
    title: str = "",
) -> go.Figure:
    """
    Build a single figure with dropdown visibility toggle.

    Parameters
    ----------
    traces : dict
        Mapping of {series_name: [trace_idx_1, trace_idx_2, ...]}.
    """
    fig = go.Figure()

    for uid, tr_list in traces.items():
        for tr in tr_list:
            tr.visible = False
            fig.add_trace(tr)

    first_uid = list(traces.keys())[0]
    for idx in traces[first_uid]:
        fig.data[idx].visible = True

    buttons = []
    total = len(fig.data)

    for uid, tr_idxs in traces.items():
        visibility = [False] * total
        for idx in tr_idxs:
            visibility[idx] = True

        buttons.append(
            dict(
                label=str(uid),
                method="update",
                args=[{"visible": visibility}, {"title": f"{title} — {uid}"}],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=1.0,
                y=1.15,
                xanchor="right",
                yanchor="top",
            )
        ]
    )

    if shapes:
        fig.update_layout(shapes=shapes)
    if annotations:
        fig.update_layout(annotations=annotations)

    if title:
        fig.update_layout(title=title)

    return fig


# =============================================================================
# EVENT LINES & LABELS
# =============================================================================

def add_event_lines_and_labels(
    fig: go.Figure,
    ev_all,
    date_col: str,
    event_label_col: str = "event",
    ev_color: str = "#555",
    stagger: bool = True,
    facet: bool = False,
    nrows: Optional[int] = None,
) -> None:
    """
    Draw vertical event lines + text labels.

    Works for overlay, facet, and dropdown modes.
    """
    unique_dates = sorted(ev_all[date_col].unique())

    label_map = (
        ev_all.drop_duplicates(subset=[date_col])
              .set_index(date_col)[event_label_col]
              .to_dict()
    )

    for i, d in enumerate(unique_dates):
        label = label_map.get(d, "event")

        if facet and nrows:
            for r in range(1, nrows + 1):
                fig.add_vline(
                    x=d,
                    row=r, col=1,
                    line_width=1,
                    line_dash="dot",
                    line_color=ev_color,
                    opacity=0.7,
                )
        else:
            fig.add_vline(
                x=d,
                line_width=1,
                line_dash="dot",
                line_color=ev_color,
                opacity=0.7,
            )

        ypos = 0.95
        if stagger:
            ypos += (i % 2) * 0.03

        fig.add_annotation(
            x=d,
            y=ypos,
            xref="x",
            yref="paper",
            showarrow=False,
            text=str(label),
            font=dict(size=11, color=ev_color),
            align="center",
            yanchor="bottom",
        )
