# tsforge/plots/core/panel.py
"""
Panel layout for composing multiple plots into a single figure.

Layouts
-------
- vertical: each figure is a row (top/bottom)
- horizontal: each figure is a column (side-by-side)
- overlay: all traces from all figures overlaid into a single axes
- grid: when grid=(rows, cols) is provided, figures are placed into a rows×cols grid

Notes
-----
- If grid is provided, layout must not be "overlay".
- Figures are placed row-major (left-to-right, top-to-bottom).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .._layout import finalize_figure


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def plot_panel(
    figures: List[go.Figure],
    *,
    layout: str = "vertical",
    grid: Optional[Tuple[int, int]] = None,  # (rows, cols)
    shared_xaxes: bool = True,
    shared_yaxes: bool = False,
    row_heights: Optional[List[float]] = None,
    col_widths: Optional[List[float]] = None,
    vertical_spacing: float = 0.06,
    horizontal_spacing: float = 0.06,
    title: Optional[str] = None,
    theme: str = "fa",
    style: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    if not figures:
        raise ValueError("figures list is empty")

    panels: List[Dict[str, Any]] = []
    for fig in figures:
        if not isinstance(fig, go.Figure):
            raise TypeError(
                f"Each item must be a plotly Figure (got {type(fig).__name__}). "
                f"Pass the result of plot_timeseries(), plot_scatter(), etc."
            )
        panels.append(_extract_panel(fig))

    layout_norm = _normalize_layout(layout)

    if grid is not None:
        if layout_norm == "overlay":
            raise ValueError("layout='overlay' cannot be combined with grid=(rows, cols).")
        rows, cols = _validate_grid(grid, n=len(panels))
        out = _build_grid(
            panels,
            rows=rows,
            cols=cols,
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes,
            row_heights=row_heights,
            col_widths=col_widths,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
        )
    elif layout_norm == "overlay":
        out = _build_overlay(panels)
    elif layout_norm == "horizontal":
        out = _build_grid(
            panels,
            rows=1,
            cols=len(panels),
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes,
            row_heights=None,
            col_widths=col_widths,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
        )
    else:  # "vertical"
        out = _build_grid(
            panels,
            rows=len(panels),
            cols=1,
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes,
            row_heights=row_heights,
            col_widths=None,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
        )

    # ── Style defaults ──
    style_out = dict(style) if style else {}
    if title:
        style_out.setdefault("title", title)

    # Legend: respect explicit user override; otherwise let the dedup
    # logic in _build_grid / _build_overlay handle duplicate entries.
    if "showlegend" in style_out:
        out.update_layout(showlegend=bool(style_out["showlegend"]))
        if not style_out["showlegend"]:
            for tr in out.data:
                tr.showlegend = False

    if "height" not in style_out:
        if grid is not None:
            style_out["height"] = max(320, 260 * grid[0])
        else:
            if layout_norm == "overlay":
                style_out["height"] = max(300, 420)
            elif layout_norm == "horizontal":
                style_out["height"] = 420
            else:
                style_out["height"] = max(300, 280 * len(panels))

    return finalize_figure(out, theme=theme, style=style_out)



# ──────────────────────────────────────────────────────────────────────────────
# Builders
# ──────────────────────────────────────────────────────────────────────────────
def _build_overlay(panels: List[Dict[str, Any]]) -> go.Figure:
    """Overlay all traces into a single axes."""
    fig = go.Figure()

    seen_names = set()
    for panel in panels:
        for trace in panel["traces"]:
            trace_name = getattr(trace, "name", None)
            if trace_name and trace_name in seen_names:
                trace.showlegend = False
            elif trace_name:
                seen_names.add(trace_name)
            fig.add_trace(trace)

    # Prefer first non-empty y-axis title (overlay can't have multiple y titles cleanly)
    for panel in panels:
        if panel["yaxis"]:
            fig.update_layout(yaxis=panel["yaxis"])
            break

    return fig


def _build_grid(
    panels: List[Dict[str, Any]],
    *,
    rows: int,
    cols: int,
    shared_xaxes: bool,
    shared_yaxes: bool,
    row_heights: Optional[List[float]],
    col_widths: Optional[List[float]],
    vertical_spacing: float,
    horizontal_spacing: float,
) -> go.Figure:
    """Place figures into a rows×cols grid, row-major. Extra cells are blank."""
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive integers")

    # Validate row/col size arrays
    if row_heights is not None and len(row_heights) != rows:
        raise ValueError(f"row_heights has {len(row_heights)} values but rows={rows}")
    if col_widths is not None and len(col_widths) != cols:
        raise ValueError(f"col_widths has {len(col_widths)} values but cols={cols}")

    # Normalize relative sizes
    rh = None
    if row_heights is not None:
        total = sum(row_heights)
        rh = [h / total for h in row_heights]

    cw = None
    if col_widths is not None:
        total = sum(col_widths)
        cw = [w / total for w in col_widths]

    # Titles: put each panel title in its cell; blanks for empty cells
    titles: List[str] = []
    for idx in range(rows * cols):
        if idx < len(panels):
            titles.append(panels[idx]["title"])
        else:
            titles.append("")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        row_heights=rh,
        column_widths=cw,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
        subplot_titles=titles,
    )

    seen_names = set()
    merged_barmode = None
    for idx, panel in enumerate(panels):
        r = idx // cols + 1
        c = idx % cols + 1

        for trace in panel["traces"]:
            trace_name = getattr(trace, "name", None)
            if trace_name and trace_name in seen_names:
                trace.showlegend = False
            elif trace_name:
                seen_names.add(trace_name)
            fig.add_trace(trace, row=r, col=c)

        # Per-cell yaxis title (applies to that subplot's yaxis)
        if panel["yaxis"]:
            yaxis_key = _yaxis_layout_key(r, c, rows, cols)
            fig.update_layout(**{yaxis_key: panel["yaxis"]})

        # Capture barmode if any panel sets it
        if panel.get("barmode"):
            merged_barmode = panel["barmode"]

    if merged_barmode:
        fig.update_layout(barmode=merged_barmode)

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _extract_panel(fig: go.Figure) -> Dict[str, Any]:
    """Extract traces, title, and yaxis info from a go.Figure."""
    traces = list(fig.data)

    # Reset visibility (figures from dropdown mode may have visible=False)
    for trace in traces:
        trace.visible = True

    panel_title = ""
    if fig.layout.title and getattr(fig.layout.title, "text", None):
        panel_title = fig.layout.title.text or ""

    yaxis = {}
    if fig.layout.yaxis and fig.layout.yaxis.title and fig.layout.yaxis.title.text:
        yaxis["title"] = fig.layout.yaxis.title.text

    # Preserve bar layout properties
    barmode = getattr(fig.layout, "barmode", None)

    return {"traces": traces, "title": panel_title, "yaxis": yaxis, "barmode": barmode}


def _normalize_layout(layout: str) -> str:
    """Map user-provided layout/mode strings to canonical layout names."""
    s = (layout or "").strip().lower()

    aliases = {
        # vertical
        "vertical": "vertical",
        "stack": "vertical",
        "stacked": "vertical",
        "rows": "vertical",
        # horizontal
        "horizontal": "horizontal",
        "side_by_side": "horizontal",
        "side-by-side": "horizontal",
        "cols": "horizontal",
        "columns": "horizontal",
        "split": "horizontal",
        # overlay
        "overlay": "overlay",
        "overlaid": "overlay",
        "layered": "overlay",
        # grid
        "grid": "grid",
        "matrix": "grid",
    }

    if s in aliases:
        return aliases[s]

    raise ValueError(
        f"Invalid layout='{layout}'. Expected one of: "
        f"'vertical', 'horizontal', 'overlay' (or use grid=(rows, cols))."
    )


def _validate_grid(grid: Tuple[int, int], *, n: int) -> Tuple[int, int]:
    try:
        rows, cols = int(grid[0]), int(grid[1])
    except Exception as e:
        raise ValueError("grid must be a tuple of two ints: (rows, cols)") from e

    if rows <= 0 or cols <= 0:
        raise ValueError("grid rows and cols must be positive integers")

    if rows * cols < n:
        raise ValueError(
            f"grid={grid} has only {rows*cols} cells but there are {n} figures. "
            f"Increase rows/cols or pass fewer figures."
        )

    return rows, cols


def _yaxis_layout_key(r: int, c: int, rows: int, cols: int) -> str:
    """
    Plotly subplot yaxis layout keys:
      - first subplot uses 'yaxis'
      - subsequent use 'yaxis2', 'yaxis3', ...
    The index is cell order in row-major (same as make_subplots).
    """
    idx0 = (r - 1) * cols + c  # 1-based subplot index
    return "yaxis" if idx0 == 1 else f"yaxis{idx0}"