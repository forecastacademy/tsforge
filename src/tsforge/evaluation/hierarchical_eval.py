import pandas as pd
from hierarchicalforecast.evaluation import evaluate
from hierarchicalforecast.utils import aggregate
from utilsforecast.losses import nd as wmape

wmape.__name__ = "wmape"


def hierarchical_evaluation(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
    models: list[str],
    metrics: list,
    hierarchy: list[list[str]],
    anchor_model: str = "Naive",
    lags_of_interest: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate hierarchical forecasts with optional anchor model comparison.

    This function aggregates forecasts across a hierarchy, computes evaluation metrics
    at each level, and optionally calculates beat rates and jitter against an anchor model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing actual values and forecasts. Must have columns for id_col,
        target_col, and all models in the models list.
    id_col : str
        Name of the column containing unique series identifiers.
    target_col : str
        Name of the column containing actual/target values.
    models : list[str]
        List of model column names to evaluate (e.g., ['ETS', 'ARIMA', 'NLinear']).
    metrics : list
        List of metric functions to compute (e.g., [wmape, rmse, mae]).
    hierarchy : list[list[str]]
        Hierarchy specification as list of column groups, from top to bottom.
        Example: [['state'], ['state', 'category'], ['state', 'category', 'store']].
    anchor_model : str, default='Naive'
        Reference model for computing beat rates and jitter. If None, these metrics
        are not computed.
    lags_of_interest : list[int] | None, default=None
        Optional list of timesteps to filter evaluation to specific horizons.
        If provided, only these timesteps will be evaluated.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - evaluation: Per-series metrics for each hierarchy level
        - summary: Aggregated metrics including mean errors, beat rates, and jitter

    Notes
    -----
    - Beat rate: Proportion of series where model beats the anchor model
    - Jitter: Standard deviation of errors (measures stability/consistency)
    - The function handles the edge case where id_col == 'unique_id' by creating
      a temporary copy to avoid conflicts with hierarchicalforecast internals

    Examples
    --------
    >>> hierarchy = [['state'], ['state', 'category']]
    >>> models = ['ETS', 'ARIMA', 'Naive']
    >>> eval_df, summary_df = hierarchical_evaluation(
    ...     df=forecasts_df,
    ...     id_col='unique_id',
    ...     target_col='y',
    ...     models=models,
    ...     metrics=[wmape],
    ...     hierarchy=hierarchy,
    ...     anchor_model='Naive',
    ...     lags_of_interest=[1, 7, 14, 28]
    ... )
    """
    df = df.copy(deep=True)

    # Filter to specific timesteps if requested
    if lags_of_interest:
        df["timestep"] = df.groupby(id_col).cumcount() + 1
        df = df.loc[df["timestep"].isin(lags_of_interest)]

    # Handle edge case: hierarchicalforecast uses 'unique_id' internally
    # Create a copy to preserve original id column in the hierarchy
    if id_col == "unique_id":
        df[id_col + "_copy"] = df[id_col]
        df = df.drop(id_col, axis=1)

    # Build complete hierarchy including lowest level (individual series)
    lowest_level = hierarchy[-1] + [id_col + "_copy"]
    complete_hierarchy = hierarchy + [lowest_level]

    # Columns to aggregate (actuals + all model predictions)
    targets = [target_col, *models]

    # Aggregate data across hierarchy levels
    hier_df, _, tags = aggregate(
        df=df,
        spec=complete_hierarchy,
        target_cols=targets,
    )

    # Compute per-series metrics at each hierarchy level
    evaluation = evaluate(
        df=hier_df,
        metrics=metrics,
        models=models,
        tags=tags,
        agg_fn=None,  # Return per-series metrics
    )

    # Remove overall aggregation and clean up column names
    evaluation = evaluation.loc[evaluation["level"] != "Overall"]
    evaluation["level"] = evaluation["level"].str.replace("_copy", "")

    # Compute mean metrics across all series at each level
    mean_eval = evaluate(df=hier_df, metrics=metrics, models=models, tags=tags, agg_fn="mean")

    # Calculate anchor model comparison metrics if requested
    if anchor_model:
        summary = _compute_anchor_comparison(
            evaluation=evaluation,
            mean_eval=mean_eval,
            anchor_model=anchor_model,
            lags_of_interest=lags_of_interest,
        )

        # Add timestep metadata if filtered
        if lags_of_interest:
            evaluation["timesteps"] = str(lags_of_interest)
            summary["timesteps"] = str(lags_of_interest)

        return evaluation, summary
    else:
        return evaluation, mean_eval


def _compute_anchor_comparison(
    evaluation: pd.DataFrame,
    mean_eval: pd.DataFrame,
    anchor_model: str,
    lags_of_interest: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute beat rates and jitter metrics relative to an anchor model.

    Parameters
    ----------
    evaluation : pd.DataFrame
        Per-series evaluation results from hierarchical_evaluation
    mean_eval : pd.DataFrame
        Mean metrics across series at each hierarchy level
    anchor_model : str
        Name of the reference model for comparison
    lags_of_interest : list[int] | None
        Optional list of timesteps being evaluated

    Returns
    -------
    pd.DataFrame
        Combined summary with mean metrics, beat rates, and jitter

    Notes
    -----
    Beat rate is computed using WMAPE as the comparison metric. Models with
    lower absolute WMAPE than the anchor are considered "wins".
    """
    # Extract WMAPE results for anchor comparison
    # Note: Currently hardcoded to use 'wmape' - could be parameterized
    wmape_results = evaluation.where(evaluation["metric"] == "wmape").dropna()

    # Reshape to long format for comparison calculations
    comparison_df = wmape_results.melt(
        id_vars=["level", "unique_id", "metric", anchor_model],
        value_name="error",
        var_name="model",
    )

    # Compute beat rate: proportion of series where model < anchor
    comparison_df[f"{anchor_model}_beat_rate"] = (
        comparison_df["error"].abs() < comparison_df[anchor_model].abs()
    )

    # Aggregate to hierarchy level: jitter (std) and beat rate (mean)
    grouped = (
        comparison_df.groupby(["level", "model"], as_index=False)
        .agg(jitter=("error", "std"), beat_rate=(f"{anchor_model}_beat_rate", "mean"))
        .rename(columns={"beat_rate": f"{anchor_model}_beat_rate"})
    )

    # Reshape to match mean_eval format: (level, metric) → model columns
    reshaped = (
        grouped.pivot(index=["level"], columns="model")
        .stack(0)  # Stack the metric names (jitter, beat_rate)
        .reset_index()
        .rename(columns={"level_1": "metric"})
    )

    # Combine mean metrics with anchor comparison metrics
    summary = pd.concat([mean_eval, reshaped]).dropna(axis=1).query("level != 'Overall'")

    return summary
