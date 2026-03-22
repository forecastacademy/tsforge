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

import re
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
    Apply theme, style overrides, legend behavior, and legend positioning.

    Global behavior:
    - If the figure would have 0–1 legend items, hide the legend.
    - If exactly 1 legend item and no subtitle provided, show that label as a subtitle.
    - Optional: enforce legend ordering (alpha or numeric) via style["legend_order"].
    """
    style = dict(style) if style else {}

    # Capture any existing height/width before applying theme
    existing_height = fig.layout.height
    existing_width = fig.layout.width

    fig = apply_theme(fig, theme)

    # -------------------------
    # Style overrides (pre-legend)
    # -------------------------
    if "title" in style:
        t = THEMES.get(theme, THEMES["fa"])
        fig.update_layout(title={
            "text": style["title"],
            "font": {"size": 18, "color": t.get("title_color")},
        })

    if "subtitle" in style:
        _upsert_subtitle(fig, str(style["subtitle"]))

    if "x_title" in style:
        fig.update_xaxes(title_text=style["x_title"])

    if "y_title" in style:
        fig.update_yaxes(title_text=style["y_title"])

    if "y_range" in style:
        fig.update_yaxes(range=style["y_range"])

    # Allow callers to pass height/width through style dict
    if "height" in style:
        try:
            existing_height = int(style["height"])
        except Exception:
            pass
    if "width" in style:
        try:
            existing_width = int(style["width"])
        except Exception:
            pass

    # Apply theme-aware legend defaults
    fig = apply_legend(fig, theme)

    # -------------------------
    # Legend suppression + subtitle defaulting
    # -------------------------
    user_forced_showlegend = "showlegend" in style
    user_provided_subtitle = "subtitle" in style

    if user_forced_showlegend:
        fig.update_layout(showlegend=bool(style["showlegend"]))

    if not user_forced_showlegend:
        single_label = _get_single_legend_label(fig)
        if single_label is not None:
            fig.update_layout(showlegend=False)
            for tr in fig.data:
                tr.showlegend = False
            if not user_provided_subtitle:
                _upsert_subtitle(fig, single_label)
        else:
            if _count_distinct_legend_labels(fig) == 0:
                fig.update_layout(showlegend=False)
                for tr in fig.data:
                    tr.showlegend = False

    # -------------------------
    # Legend ordering (alpha / numeric)
    # -------------------------
    legend_order = style.get("legend_order", "alpha")
    if legend_order:
        apply_legend_order(fig, order=str(legend_order))

    # -------------------------
    # Determine legend position from style override or theme
    # -------------------------
    t = THEMES.get(theme, THEMES["fa"])
    legend_pos = t.get("legend_position", "top right")
    if "legend_position" in style:
        legend_pos = style["legend_position"]

    showlegend = fig.layout.showlegend
    if showlegend is False:
        legend_layout, margin_bottom = {}, base_margin_bottom
    else:
        legend_layout, margin_bottom = _resolve_legend_position(
            legend_pos, fig, base_margin_bottom, theme=theme
        )

    # -------------------------
    # Merge margins (allow style["margin"] overrides)
    # -------------------------
    user_margin = {}
    if isinstance(style.get("margin"), dict):
        user_margin = dict(style["margin"])

    final_margin = dict(b=margin_bottom)
    final_margin.update(user_margin)  # user can override b too if they want

    # Build final layout update, preserving existing height/width
    final_layout: Dict[str, Any] = {
        "margin": final_margin,
    }
    if legend_layout:
        final_layout["legend"] = legend_layout

    # Restore height/width if they were set
    if existing_height is not None:
        final_layout["height"] = existing_height
    if existing_width is not None:
        final_layout["width"] = existing_width

    fig.update_layout(**final_layout)
    return fig


def _get_single_legend_label(fig: go.Figure) -> Optional[str]:
    """Return the sole distinct legend label if exactly 1 legend item would render."""
    labels = set()

    for tr in fig.data:
        if getattr(tr, "showlegend", True) is False:
            continue
        if getattr(tr, "visible", True) is False:
            continue

        name = getattr(tr, "name", None)
        if name is None:
            continue
        name = str(name).strip()
        if not name:
            continue

        labels.add(name)
        if len(labels) > 1:
            return None

    return next(iter(labels)) if len(labels) == 1 else None


def _count_distinct_legend_labels(fig: go.Figure) -> int:
    """Count distinct, meaningful legend labels that would render."""
    labels = set()
    for tr in fig.data:
        if getattr(tr, "showlegend", True) is False:
            continue
        if getattr(tr, "visible", True) is False:
            continue
        name = getattr(tr, "name", None)
        if name is None:
            continue
        name = str(name).strip()
        if not name:
            continue
        labels.add(name)
        if len(labels) > 2:
            break
    return len(labels)


def _upsert_subtitle(fig, subtitle: str, theme: str = "fa") -> None:
    t = THEMES.get(theme, THEMES["fa"])
    if not subtitle:
        return

    existing = list(getattr(fig.layout, "annotations", []) or [])
    existing = [a for a in existing if getattr(a, "name", None) != "tsforge_subtitle"]
    theme = getattr(fig.layout, "meta", {}).get("theme", "fa") if getattr(fig.layout, "meta", None) else "fa"
    t = THEMES.get(theme, THEMES["fa"])
    subtitle_color = t.get("subtitle_color", "#666666")

    existing.append(
        dict(
            name="tsforge_subtitle",
            x=0,
            y=1.06,
            xref="paper",
            yref="paper",
            text=subtitle,
            showarrow=False,
            font=dict(size=13, color=subtitle_color),
            align="left",
        )
    )

    fig.update_layout(annotations=existing)


def _resolve_legend_position(
    position: str,
    fig: go.Figure,
    base_margin_bottom: int,
    theme: str = "fa",
) -> tuple:
    """Convert a legend position string to Plotly legend dict + bottom margin."""
    t = THEMES.get(theme, THEMES["fa"])
    pos = position.lower().strip()

    base_style = dict(
        bgcolor=t.get("legend_bg", "rgba(255,255,255,0.9)"),
        bordercolor=t.get("legend_border", "rgba(0,0,0,0.1)"),
        borderwidth=1,
    )

    if pos == "top right":
        return {
            **base_style,
            "yanchor": "top", "y": 0.98,
            "xanchor": "right", "x": 0.98,
        }, base_margin_bottom

    if pos == "top left":
        return {
            **base_style,
            "yanchor": "top", "y": 0.98,
            "xanchor": "left", "x": 0.02,
        }, base_margin_bottom

    if pos == "bottom right":
        return {
            **base_style,
            "yanchor": "bottom", "y": 0.02,
            "xanchor": "right", "x": 0.98,
        }, base_margin_bottom

    if pos == "bottom left":
        return {
            **base_style,
            "yanchor": "bottom", "y": 0.02,
            "xanchor": "left", "x": 0.02,
        }, base_margin_bottom

    # Default: bottom center (horizontal)
    num_legend_items = sum(
        1 for tr in fig.data
        if getattr(tr, "showlegend", True) and getattr(tr, "visible", True) is not False
    )
    legend_rows = max(1, (num_legend_items + 5) // 6)
    bottom_margin = base_margin_bottom + (legend_rows * 25)

    return {
        **base_style,
        "orientation": "h",
        "x": 0.5, "y": -0.12,
        "xanchor": "center", "yanchor": "top",
    }, bottom_margin


# =============================================================================
# LEGEND ORDERING
# =============================================================================

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _legend_sort_key(name: str, order: str):
    s = (name or "").strip()
    order = (order or "alpha").strip().lower()

    if order == "numeric":
        m = _NUM_RE.search(s)
        if m:
            try:
                return (0, float(m.group(0)))
            except Exception:
                pass
        return (1, s.lower())

    # default alpha
    return s.lower()


def apply_legend_order(
    fig: go.Figure,
    *,
    order: str = "alpha",  # "alpha" or "numeric"
) -> None:
    """
    Ensure legend order is alphabetical or numeric by assigning legendrank.
    Does not reorder traces; only affects legend display.
    """
    order = (order or "alpha").strip().lower()
    if order not in {"alpha", "numeric"}:
        return

    items = []
    for i, tr in enumerate(fig.data):
        if getattr(tr, "showlegend", True) is False:
            continue
        if getattr(tr, "visible", True) is False:
            continue
        name = getattr(tr, "name", None)
        if not name or not str(name).strip():
            continue
        items.append((i, str(name)))

    if not items:
        return

    items_sorted = sorted(items, key=lambda t: (_legend_sort_key(t[1], order), t[0]))

    for rank, (i, _name) in enumerate(items_sorted, start=1):
        fig.data[i].legendrank = rank * 1000

    fig.update_layout(legend=dict(traceorder="normal"))


# =============================================================================
# DROPDOWN BUTTON BUILDING
# =============================================================================

def build_dropdown_buttons(
    trace_map: Dict[str, List[int]],
    total_traces: int,
    label_fn: Optional[Callable[[str], str]] = None,
) -> List[Dict[str, Any]]:
    """Build standardized dropdown menu configuration."""
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