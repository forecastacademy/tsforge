PALETTE = ["#1f77b4", "#9467bd", "#17becf",
           "#ff7f0e", "#2ca02c", "#d62728"]
HIGHLIGHT = "#e45756"

def _apply_tsforge_style(fig, engine: str = "plotly", context: str = "series"):
    if engine == "plotly":
        legend_cfg = dict(
            orientation="h",
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0)"
        )
        if context == "acf_pacf":
            legend_cfg.update(yanchor="top", y=-0.2, xanchor="center", x=0.5)
        elif context == "cv_results":
            legend_cfg.update(yanchor="top", y=-0.15, xanchor="center", x=0.5)
        else:  # default for timeseries
            legend_cfg.update(yanchor="bottom", y=1.02, xanchor="right", x=1)

        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter, Segoe UI, Helvetica", size=14, color="#333"),
            margin=dict(l=50, r=30, t=60, b=80),
            legend=legend_cfg,
        )
        return fig

    elif engine == "matplotlib":
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "font.family": ["DejaVu Sans"],
            "axes.facecolor": "white",
            "axes.edgecolor": "#333",
            "axes.labelcolor": "#333",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.loc": "upper right",
            "grid.color": "0.85",
            "grid.linestyle": ":",
            "lines.linewidth": 2.0,
        })
        return fig
    else:
        raise ValueError("engine must be 'plotly' or 'matplotlib'")
