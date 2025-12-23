# tsforge/display/styling.py

import importlib.resources as pkg_resources
from typing import Iterable, Optional
import pandas as pd
from IPython.display import display, HTML
from pandas.io.formats.style import Styler

# -------------------------------------------------------------------
# Internal CSS loader
# -------------------------------------------------------------------

def _get_default_css() -> str:
    """Load packaged CSS from tsforge/assets/css."""
    css_files = ["style-notebook.css", "style-table.css"]
    parts: list[str] = []
    for fname in css_files:
        with pkg_resources.files("tsforge.assets.css").joinpath(fname).open("r", encoding="utf-8") as f:
            parts.append(f.read())
    return "\n".join(parts)


# -------------------------------------------------------------------
# 1) Global injection
# -------------------------------------------------------------------

def enable_notebook_style(css_text: Optional[str] = None, once: bool = True) -> None:
    """
    Inject CSS into the current Jupyter notebook so *all* DataFrames
    (including pandas Styler outputs with id="T_xxx") adopt the theme.

    Call this once at the top of your notebook:

        from tsforge.display import enable_notebook_style
        enable_notebook_style()
    """
    flag_attr = "_tsforge_css_enabled"
    if once and getattr(enable_notebook_style, flag_attr, False):
        return

    css_text = css_text or _get_default_css()
    display(HTML(f"<style>{css_text}</style>"))
    setattr(enable_notebook_style, flag_attr, True)

def _auto_inject():
    try:
        from IPython.display import display, HTML
        import importlib.resources as pkg_resources
        css = pkg_resources.read_text("tsforge.assets.css", "style-notebook.css")
        display(HTML(f"<style>{css}</style>"))
    except Exception:
        pass

_auto_inject()  # inject at import time

# -------------------------------------------------------------------
# 2) Scoreboard Styler
# -------------------------------------------------------------------

def style_leaderboard(
    df: pd.DataFrame,
    percent_cols: Iterable[str] = ("wape", "mape", "smape", "accuracy", "mpe", "bias"),
    lower_is_better: Iterable[str] = ("wape", "mae", "rmse", "mape", "smape", "mase"),
    sticky_cols: Iterable[str] = ("workflow", "engine", "model"),
    precision: int = 4,
) -> pd.io.formats.style.Styler:
    """
    Return a pandas Styler with friendly formats and cues for leaderboards.
    Works whether df is per-fold or aggregated (_mean/_std suffixes).
    """

    df = df.copy()

    # Detect aggregate columns
    def base(c: str) -> str:
        for suf in ("_mean", "_std"):
            if c.endswith(suf):
                return c[: -len(suf)]
        return c

    base_cols = {c: base(c) for c in df.columns}

    # Choose targets to highlight
    targets = []
    for c in df.columns:
        b = base_cols[c]
        if (c in lower_is_better) or (b in lower_is_better):
            targets.append(c)

    # Number formatting
    fmt = {}
    for c in df.columns:
        b = base_cols[c]
        if c in ("cutoff",):
            continue
        if df[c].dtype.kind in "biufc":
            if (c in percent_cols) or (b in percent_cols):
                fmt[c] = "{:,.2%}"
                if df[c].max() <= 1.0 and df[c].min() >= -1.0:
                    df[c] = df[c].astype(float)
            else:
                fmt[c] = "{:,.{p}f}".format(p=precision)

    # Create styler with custom class so Brandon Rhodes CSS applies
    sty = df.style.set_table_attributes('class="dataframe tsforge-leaderboard"')

    if fmt:
        sty = sty.format(fmt)

    if targets:
        agg_targets = [t for t in targets if t.endswith("_mean")]
        use_cols = agg_targets or targets
        for col in use_cols:
            sty = sty.highlight_min(subset=[col], color="#d1fae5")

    # Sticky columns for readability
    table_styles = [
        dict(selector="th.col_heading", props=[("position", "sticky"), ("top", "0"), ("z-index", "2")]),
        dict(selector="th.row_heading", props=[("position", "sticky"), ("left", "0"), ("z-index", "1"),
                                               ("background-color", "white")]),
        dict(selector="thead th:not(.index_name)", props=[("position", "sticky"), ("top", "0"), ("z-index", "3")]),
    ]
    for col in sticky_cols:
        if col in df.columns:
            j = list(df.columns).index(col) + 1
            table_styles.append(
                dict(selector=f"tbody tr td:nth-child({j})",
                     props=[("position", "sticky"), ("left", f"{(j-1)*9}em"),
                            ("background", "white"), ("z-index", "1")])
            )
            table_styles.append(
                dict(selector=f"thead tr th:nth-child({j})",
                     props=[("position", "sticky"), ("left", f"{(j-1)*9}em"),
                            ("background", "white"), ("z-index", "2")])
            )

    sty = sty.set_table_styles(table_styles, overwrite=False)
    return sty

# -------------------------------------------------------------------
# 3) Heatmap Table Styler
# -------------------------------------------------------------------

def style_heatmap_table(
    df: pd.DataFrame,
    group_col: str,
    metrics: list[str],
    title: Optional[str] = None,
    cmap: str = "RdYlGn_r",
    precision: int = 3,
) -> Styler:
    """
    Create a styled heatmap table showing mean metrics by group.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    group_col : str
        Column to group by (e.g., 'dept_id', 'cat_id').
    metrics : list of str
        Metric columns to summarize.
    title : str, optional
        Table caption.
    cmap : str
        Colormap for background gradient (default: 'RdYlGn_r').
    precision : int
        Decimal places (default: 3).
    
    Returns
    -------
    pandas.Styler
    
    Examples
    --------
    >>> from tsforge.display import style_heatmap_table
    >>> styled = style_heatmap_table(diagnostics, 'dept_id', ['trend', 'entropy', 'adi'])
    >>> display(styled)
    """
    summary = df.groupby(group_col)[metrics].mean().round(precision)
    
    sty = (
        summary.style
        .set_table_attributes('class="dataframe tsforge-heatmap"')
        .background_gradient(cmap=cmap, axis=0)
        .format(f'{{:.{precision}f}}')
    )
    
    if title:
        sty = sty.set_caption(title).set_table_styles([{
            'selector': 'caption',
            'props': [('font-size', '14px'), ('font-weight', 'bold'), ('padding', '10px')]
        }])
    
    return sty
