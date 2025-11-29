import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tsforge.plots.style import _apply_tsforge_style

def plot_calendar_heatmap(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    value_col: str = "y",
    ids: str | list[str] | int | None = None,
    group_col: str | None = None,
    agg: str = "sum",
    mode: str = "facet",           # 'facet' or 'dropdown'
    wrap: int = 3,
):
    """
    Simple calendar heatmap: Day of month (x) vs Month (y),
    normalized 0–1 per series/group for pattern spotting.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with id, date, and value columns.
    id_col : str
        Column with unique series identifier.
    date_col : str
        Datetime column.
    value_col : str, default 'y'
        Target column to plot.
    ids : str | list[str] | int | None
        - None: sample up to 3 series
        - str: plot that single id
        - list: plot each id in the list
        - int: randomly sample that many ids
    group_col : str, optional
        Optional grouping variable (e.g., dept_id, store_id).
        If provided, aggregation is done at that group level instead of id_col.
    agg : str, default 'sum'
        Aggregation for days with multiple rows.
    mode : str, default 'facet'
        'facet' = grid of panels
        'dropdown' = one panel, switchable
    wrap : int, default 3
        Number of facets per row (facet mode only).
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Aggregate by group_col if provided
    if group_col:
        group_keys = [group_col, date_col]
        df = df.groupby(group_keys, observed=True)[value_col].agg(agg).reset_index()
        id_col = group_col

    unique_ids = df[id_col].dropna().unique().tolist()

    # Handle ids arg
    if ids is None:
        ids = pd.Series(unique_ids).sample(min(3, len(unique_ids)), random_state=42).tolist()
    elif isinstance(ids, int):
        ids = pd.Series(unique_ids).sample(min(ids, len(unique_ids)), random_state=42).tolist()
    elif isinstance(ids, str):
        ids = [ids]
    else:
        ids = list(ids)

    # Aggregate daily values
    df = df.groupby([id_col, date_col], observed=True)[value_col].agg(agg).reset_index()

    # Add calendar parts
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    # Normalize per ID (0–1)
    df[value_col] = df.groupby(id_col)[value_col].transform(
        lambda x: x / x.max() if x.max() > 0 else x
    )

    # --------------------
    # Facet mode
    # --------------------
    if mode == "facet":
        from plotly.subplots import make_subplots
        
        n = len(ids)
        rows = int(np.ceil(n / wrap))
        cols = min(wrap, n)
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[str(uid) for uid in ids],
            vertical_spacing=0.2 # add some breathing room between rows
        )
        
        fig.update_layout(height=rows * 400, width=cols * 600) 
        
        for i, uid in enumerate(ids):
            sub = df[df[id_col] == uid]
            heat = go.Heatmap(
                x=sub["day"], y=sub["month"], z=sub[value_col],
                zmin=0, zmax=1,
                colorscale="YlGnBu",
                colorbar=dict(title="Normalized Sales"),
                showscale=(i == 0)  # only show once
            )
            r = i // wrap + 1
            c = i % wrap + 1
            fig.add_trace(heat, row=r, col=c)
        
        fig.update_xaxes(
            showgrid=True, gridcolor="lightgrey", dtick=1, title="Day of Month"
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="lightgrey", dtick=1,
            tickmode="array", tickvals=list(range(1, 13)),
            ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"],
            title="Month"
        )
        return _apply_tsforge_style(fig, engine="plotly")

    # --------------------
    # Dropdown mode
    # --------------------
    elif mode == "dropdown":
        fig = go.Figure()
        for i, uid in enumerate(ids):
            sub = df[df[id_col] == uid]
            heat = go.Heatmap(
                x=sub["day"], y=sub["month"], z=sub[value_col],
                colorscale="YlGnBu",
                zmin=0, zmax=1,
                visible=(i == 0),
                name=str(uid)
            )
            fig.add_trace(heat)

        # Dropdown buttons
        buttons = []
        for i, uid in enumerate(ids):
            visible = [False] * len(ids)
            visible[i] = True
            buttons.append(dict(
                label=str(uid),
                method="update",
                args=[{"visible": visible}, {"title": f"Calendar Heatmap: {uid}"}]
            ))
        fig.update_layout(
            updatemenus=[{
                "buttons": buttons,
                "direction": "down",
                "x": 1.05, "y": 1.15,
                "xanchor": "left", "yanchor": "top"
            }]
        )
        fig.update_xaxes(showgrid=True, gridcolor="lightgrey", dtick=1, title="Day of Month")
        fig.update_yaxes(showgrid=True, gridcolor="lightgrey", dtick=1,
                         tickmode="array", tickvals=list(range(1, 13)),
                         ticktext=month_names, title="Month")
        return _apply_tsforge_style(fig, engine="plotly")

    else:
        raise ValueError("mode must be 'facet' or 'dropdown'")
