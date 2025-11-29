import pandas as pd
import numpy as np
from tsforge.evaluation import score_all
from tsforge.evaluation.metrics import score_intervals


def accuracy_table(
    preds_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    id_col="unique_id",
    time_col="date",
    target_col="sales",
    metrics=None,
    mode="aggregate",
    include_interval_metrics=False,
    layout="wide"
) -> pd.DataFrame:
    """
    Compute accuracy metrics from predictions (and intervals if requested).

    - folds: per-fold metrics
    - aggregate: global metrics across all folds
    - both: return both folds and aggregate
    - layout: 'wide' (metrics as columns) or 'long' (metrics as rows, models as columns)
    """

    # --- Ensure truth available ---
    if target_col not in preds_df.columns:
        truth = truth_df[[id_col, time_col, target_col]]
        df = preds_df.merge(truth, on=[id_col, time_col], how="left")
    else:
        df = preds_df.copy()

    base_keys = ["workflow", "engine", "model"]
    has_cutoff = "cutoff" in df.columns
    group_keys_folds = base_keys + (["cutoff"] if has_cutoff else [])

    # -----------------------------
    # FOLD-LEVEL METRICS
    # -----------------------------
    fold_results = []
    for keys, g in df.groupby(group_keys_folds, dropna=False):
        row = dict(zip(group_keys_folds, keys if isinstance(keys, tuple) else (keys,)))

        y = pd.to_numeric(g[target_col], errors="coerce").to_numpy()
        yhat = pd.to_numeric(g["yhat"], errors="coerce").to_numpy()
        mask = ~np.isnan(y) & ~np.isnan(yhat)

        if mask.sum() > 0:
            row.update(score_all(y[mask], yhat[mask], None))
        else:
            row.update({m: np.nan for m in [
                "mae","rmse","mape","smape","wape","accuracy","bias","mpe","forecast_bias","mase"
            ]})

        if include_interval_metrics:
            lo_cols = [c for c in g.columns if str(c).startswith("lo-")]
            for lo in lo_cols:
                lvl = int(lo.split("-")[1])
                hi = f"hi-{lvl}"
                if hi in g:
                    lo_vals = pd.to_numeric(g[lo], errors="coerce").to_numpy()
                    hi_vals = pd.to_numeric(g[hi], errors="coerce").to_numpy()
                    mask_int = mask & ~np.isnan(lo_vals) & ~np.isnan(hi_vals)
                    if mask_int.sum() > 0:
                        row.update(score_intervals(y[mask_int], lo_vals[mask_int], hi_vals[mask_int], lvl))
                    else:
                        row.update({f"coverage_{lvl}": np.nan,
                                    f"width_{lvl}": np.nan,
                                    f"winkler_{lvl}": np.nan,
                                    f"cwc_{lvl}": np.nan})
        fold_results.append(row)

    folds_df = pd.DataFrame(fold_results)

    # -----------------------------
    # GLOBAL AGGREGATE METRICS
    # -----------------------------
    agg_results = []
    for keys, g in df.groupby(base_keys, dropna=False):
        row = dict(zip(base_keys, keys))

        y = pd.to_numeric(g[target_col], errors="coerce").to_numpy()
        yhat = pd.to_numeric(g["yhat"], errors="coerce").to_numpy()
        mask = ~np.isnan(y) & ~np.isnan(yhat)

        if mask.sum() > 0:
            row.update(score_all(y[mask], yhat[mask], None))
        else:
            row.update({m: np.nan for m in [
                "mae","rmse","mape","smape","wape","accuracy","bias","mpe","forecast_bias","mase"
            ]})

        if include_interval_metrics:
            lo_cols = [c for c in g.columns if str(c).startswith("lo-")]
            for lo in lo_cols:
                lvl = int(lo.split("-")[1])
                hi = f"hi-{lvl}"
                if hi in g:
                    lo_vals = pd.to_numeric(g[lo], errors="coerce").to_numpy()
                    hi_vals = pd.to_numeric(g[hi], errors="coerce").to_numpy()
                    mask_int = mask & ~np.isnan(lo_vals) & ~np.isnan(hi_vals)
                    if mask_int.sum() > 0:
                        row.update(score_intervals(y[mask_int], lo_vals[mask_int], hi_vals[mask_int], lvl))
        agg_results.append(row)

    aggregate_df = pd.DataFrame(agg_results)

    # -----------------------------
    # Metric Selection
    # -----------------------------
    def _filter_metrics(df_in):
        if not metrics:
            return df_in
        keep = base_keys + ([ "cutoff" ] if "cutoff" in df_in.columns else [])
        cols = df_in.columns.tolist()
        selected = []
        for m in metrics:
            if m in cols:
                selected.append(m)
            else:
                # wildcard match for interval metrics (coverage_XX, width_XX, etc.)
                selected.extend([c for c in cols if c.startswith(m+"_")])
        return df_in[keep + sorted(set(selected))]

    folds_df = _filter_metrics(folds_df)
    aggregate_df = _filter_metrics(aggregate_df)

    # -----------------------------
    # Layout Transformation
    # -----------------------------
    def _to_long(df_in):
        long_keys = [c for c in df_in.columns if c in base_keys or c=="cutoff"]
        metric_cols = [c for c in df_in.columns if c not in long_keys]
        melted = df_in.melt(id_vars=long_keys, value_vars=metric_cols,
                            var_name="metric", value_name="value")
        idx = [k for k in long_keys if k != "model"] + ["metric"]
        out_long = melted.pivot_table(index=idx, columns="model", values="value")
        return out_long.reset_index()

    if layout=="long":
        folds_df = _to_long(folds_df)
        aggregate_df = _to_long(aggregate_df)

    # -----------------------------
    # Return according to mode
    # -----------------------------
    if mode=="folds":
        return folds_df
    elif mode=="aggregate":
        return aggregate_df
    elif mode=="both":
        f = folds_df.copy(); f["table"]="folds"
        a = aggregate_df.copy(); a["table"]="aggregate"
        all_cols = sorted(set(f.columns).union(a.columns))
        if "table" in all_cols:
            all_cols.remove("table")
            all_cols = ["table"] + all_cols
        return pd.concat([f.reindex(columns=all_cols), a.reindex(columns=all_cols)], ignore_index=True)
    else:
        raise ValueError("mode must be one of {'folds','aggregate','both'}")
