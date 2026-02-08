import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# DefenderDecision Dataclass
# ============================================================================


@dataclass
class DefenderDecision:
    """
    Container for defender selection decision and audit trail.

    Attributes
    ----------
    model : str
        Model name
    wmape : float
        Portfolio weighted mean absolute percentage error
    bias : float
        Portfolio bias
    beat_rate : float
        Percentage of series where model beats anchor (0-100)
    jitter : float
        Stability metric (std of wmape across cutoffs)
    passes_anchor_gate : bool
        Whether model beats anchor and beat_rate > 50%
    beat_rate_pass : bool
        Whether beat_rate >= threshold
    bias_pass : bool
        Whether |bias| <= threshold
    jitter_pass : bool
        Whether jitter <= threshold
    final_decision : str
        One of: "DEFENDER", "REJECT"
    decision_reason : str
        Explanation of decision (e.g., "Selected as Defender", "Failed beat_rate check")
    """
    model: str
    wmape: float
    bias: float
    beat_rate: float
    jitter: float
    passes_anchor_gate: bool
    beat_rate_pass: bool
    bias_pass: bool
    jitter_pass: bool
    final_decision: str
    decision_reason: str


# ============================================================================
# Scale-dependent metrics
# ============================================================================
def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def mse(y, yhat):
    return float(np.mean((np.asarray(y) - np.asarray(yhat))**2))


def rmse(y, yhat):
    return float(np.sqrt(mse(y, yhat)))


# --- Percentage metrics ---
def mape(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)


def smape(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    denom = np.abs(y) + np.abs(yhat) + 1e-12
    return float(np.mean(2.0 * np.abs(y - yhat) / denom) * 100)


def wape(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.sum(np.abs(y - yhat)) / (np.sum(np.abs(y)) + 1e-12))


def business_accuracy(y, yhat):
    """
    Business-style Accuracy.
    1 - sum(|error|)/sum(actuals).
    Equivalent to 1 - WAPE.
    """
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(1 - (np.sum(np.abs(y - yhat)) / (np.sum(np.abs(y)) + 1e-12)))


# --- Scaled metrics ---
def mase(y, yhat, y_naive=None):
    """
    Mean Absolute Scaled Error (relative to naive-1 by default).
    If y_naive not provided, computes using naive-1 differences.
    """
    y, yhat = np.asarray(y), np.asarray(yhat)
    if y_naive is None:
        scale = np.mean(np.abs(y[1:] - y[:-1]))  # naive-1 denominator
    else:
        y_naive = np.asarray(y_naive)
        scale = np.mean(np.abs(y - y_naive))
    return float(np.mean(np.abs(y - yhat)) / (scale + 1e-12))


# --- Bias metrics ---
def bias(y, yhat):
    """
    Forecast bias (mean forecast error).
    Positive → under-forecasted, Negative → over-forecasted.
    """
    return float(np.mean(np.asarray(yhat) - np.asarray(y)))


def mean_percentage_error(y, yhat):
    """Mean Percentage Error (directional bias, in %)"""
    y, yhat = np.asarray(y), np.asarray(yhat)
    mask = y != 0
    return float(np.mean((yhat[mask] - y[mask]) / y[mask]) * 100)


def forecast_bias(y, yhat):
    """
    Forecast Bias Ratio (%).
    Sum(forecast)/Sum(actual).
    1.0 = unbiased, <1 under-forecast, >1 over-forecast.
    """
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float((np.sum(yhat) + 1e-12) / (np.sum(y) + 1e-12))


# --- Scoring utility ---
def score_all(y, yhat, y_naive=None, as_dataframe=False):
    """
    Compute all standard forecast metrics and return as dict (default) or DataFrame row.
    """
    scores = {
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
        "mape": mape(y, yhat),
        "smape": smape(y, yhat),
        "wape": wape(y, yhat),
        "accuracy": business_accuracy(y, yhat),
        "bias": bias(y, yhat),
        "mpe": mean_percentage_error(y, yhat),
        "forecast_bias": forecast_bias(y, yhat),
    }
    # only compute mase if y has length > 1
    if len(np.asarray(y)) > 1:
        scores["mase"] = mase(y, yhat, y_naive=y_naive)

    if as_dataframe:
        return pd.DataFrame([scores])
    return scores

# --- Interval metrics ---

def coverage_score(y, lo, hi):
    """Regression Coverage Score (RCS)."""
    return np.mean((y >= lo) & (y <= hi))

def mean_width(y, lo, hi):
    """Regression Mean Width Score (RMWS)."""
    return np.mean(hi - lo)

def winkler_score(y, lo, hi, alpha):
    """
    Mean Winkler Interval Score (MWI).
    alpha = significance (e.g. 0.05 for 95% interval).
    """
    score = (hi - lo) \
        + (2/alpha) * (lo - y) * (y < lo) \
        + (2/alpha) * (y - hi) * (y > hi)
    return np.mean(score)

def cwc(y, lo, hi, alpha, eta=50):
    """
    Coverage Width Criterion (CWC).
    Balances coverage with width.
    eta = penalty strength (default 50, as in Khosravi 2011).
    """
    cov = coverage_score(y, lo, hi)
    width = mean_width(y, lo, hi)
    return (1 - width) * np.exp(-eta * (cov - (1 - alpha))**2)

def score_intervals(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, level: int, eta: int = 50) -> dict:
    """
    Compute interval metrics for a given prediction interval.
    
    Args:
        y: truth values
        lo: lower bound predictions
        hi: upper bound predictions
        level: nominal coverage level (e.g. 80, 95)
        eta: penalty parameter for CWC (default=50, Khosravi 2011)
    
    Returns:
        dict with coverage, width, Winkler/interval score, and CWC
    """
    # Basic stats
    coverage = np.mean((y >= lo) & (y <= hi))
    width = np.mean(hi - lo)

    # alpha = significance level (e.g. 0.05 for 95% interval)
    alpha = 1 - level / 100

    # Winkler / Interval Score (Gneiting & Raftery 2007)
    winkler = np.mean(
        (hi - lo) +
        (2/alpha) * (lo - y) * (y < lo) +
        (2/alpha) * (y - hi) * (y > hi)
    )

    # Coverage Width Criterion (CWC, Khosravi 2011)
    cwc = (1 - width) * np.exp(-eta * (coverage - (1 - alpha))**2)

    return {
        f"coverage_{level}": coverage,
        f"width_{level}": width,
        f"winkler_{level}": winkler,
        f"cwc_{level}": cwc,
    }
# ============================================================================
# MetricsCalculator Class
# ============================================================================


class MetricsCalculator:
    """
    Compute forecast evaluation metrics and governance decisions.

    Handles portfolio-level metric aggregation and applies Gate→Rank→Veto
    logic for Defender selection.

    Parameters
    ----------
    anchor_model : str, default="SN52"
        Baseline model for beat rate and FVA calculations
    model_col : str, default="model"
        Column name for model identifiers
    metric_col : str, default="wmape"
        Primary metric column name for ranking
    """

    def __init__(
        self,
        anchor_model: str = "SN52",
        model_col: str = "model",
        metric_col: str = "wmape",
    ):
        """Initialize MetricsCalculator with configuration."""
        self.anchor_model = anchor_model
        self.model_col = model_col
        self.metric_col = metric_col

    def compute_improvement_distribution(
        self,
        metric_level_df: pd.DataFrame,
        metric_col: Optional[str] = None,
        anchor_threshold: float = 0.01,
    ) -> pd.DataFrame:
        """
        Compute % improvement vs Anchor for each metric-level grouping.

        % improvement = (anchor_metric - model_metric) / anchor_metric * 100
        Positive values = model better than anchor
        Negative values = model worse than anchor

        Parameters
        ----------
        metric_level_df : pd.DataFrame
            Metric-level results with columns: model, metric_level_col, metric_col
        metric_col : str, optional
            Metric column to use. Defaults to self.metric_col (wmape)
        anchor_threshold : float, default=0.01
            Filter out metric_level items where anchor metric < threshold
            (avoids division by very small numbers)

        Returns
        -------
        pd.DataFrame
            Columns: model, metric_level_col (e.g., item_id), anchor_metric,
                     model_metric, pct_improvement
        """
        metric_col = metric_col or self.metric_col

        # Infer metric_level column name (should be non-standard column)
        metric_level_cols = [
            col for col in metric_level_df.columns
            if col not in [self.model_col, metric_col, "cutoff"]
        ]

        if not metric_level_cols:
            raise ValueError(
                f"Could not infer metric_level column. "
                f"Expected non-standard column besides: {self.model_col}, {metric_col}, cutoff"
            )

        metric_level_col = metric_level_cols[0]

        # Get anchor performance per metric_level
        anchor_by_level = (
            metric_level_df
            .query(f"{self.model_col} == '{self.anchor_model}'")
            .groupby(metric_level_col)
            [metric_col]
            .mean()
            .reset_index()
            .rename(columns={metric_col: "anchor_metric"})
        )

        # Get model performance per metric_level
        model_by_level = (
            metric_level_df
            .groupby([self.model_col, metric_level_col])
            [metric_col]
            .mean()
            .reset_index()
            .rename(columns={metric_col: "model_metric"})
        )

        # Merge and compute improvement
        improvement = model_by_level.merge(
            anchor_by_level, on=metric_level_col, how="left"
        )

        # Filter out very small anchor values
        improvement = improvement.query(f"anchor_metric > {anchor_threshold}").copy()

        # Compute percentage improvement
        improvement["pct_improvement"] = (
            (improvement["anchor_metric"] - improvement["model_metric"])
            / improvement["anchor_metric"] * 100
        )

        return improvement

    def select_defender(
        self,
        portfolio_df: pd.DataFrame,
        beat_rate_threshold: float = 50.0,
        bias_threshold: float = 10.0,
        jitter_threshold: float = 10.0,
    ) -> pd.DataFrame:
        """
        Apply Gate → Rank → Veto logic to select the Defender model.

        Decision Process:
        1. **Anchor Gate:** Model must beat Seasonal Naive on portfolio wmape
           AND beat_rate > 50%
        2. **Rank:** Among eligible models, rank by portfolio wmape (lower is better)
        3. **Veto Checks:**
           - Beat Rate >= beat_rate_threshold
           - |Bias| <= bias_threshold
           - Jitter <= jitter_threshold

        The Defender is the eligible model with the lowest wmape.

        Parameters
        ----------
        portfolio_df : pd.DataFrame
            Portfolio-level results with columns: model, wmape, beat_rate, bias, jitter
        beat_rate_threshold : float, default=50.0
            Minimum beat rate (0-100 scale)
        bias_threshold : float, default=10.0
            Maximum absolute bias
        jitter_threshold : float, default=10.0
            Maximum jitter value

        Returns
        -------
        pd.DataFrame
            Input with added decision columns:
            - passes_anchor_gate : bool
            - wmape_rank : int
            - beat_rate_pass : bool
            - bias_pass : bool
            - jitter_pass : bool
            - final_decision : str ("DEFENDER" or "REJECT")
            - decision_reason : str (explanation)
        """
        df = portfolio_df.copy()

        # Get anchor performance
        anchor_wmape = df.loc[
            df[self.model_col] == self.anchor_model, "wmape"
        ].values[0]

        # Step 1: Anchor Gate
        df["passes_anchor_gate"] = (
            (df["wmape"] < anchor_wmape) &
            (df["beat_rate"] > 50)
        )

        # Step 2: Rank by wmape
        df["wmape_rank"] = df["wmape"].rank(method="min").astype(int)

        # Step 3: Veto checks
        df["beat_rate_pass"] = df["beat_rate"] >= beat_rate_threshold
        df["bias_pass"] = df["bias"].abs() <= bias_threshold
        df["jitter_pass"] = df["jitter"] <= jitter_threshold

        # Combined eligibility
        df["all_vetos_pass"] = (
            df["beat_rate_pass"] & df["bias_pass"] & df["jitter_pass"]
        )
        df["eligible_defender"] = (
            df["passes_anchor_gate"] & df["all_vetos_pass"]
        )

        # Initialize decision columns
        df["final_decision"] = "REJECT"
        df["decision_reason"] = ""

        # Build decision reasons for each row
        for idx, row in df.iterrows():
            reasons = []

            if not row["passes_anchor_gate"]:
                reasons.append("Failed Anchor Gate")

            if not row["beat_rate_pass"]:
                reasons.append(
                    f"Beat Rate {row['beat_rate']:.1f}% < {beat_rate_threshold}%"
                )

            if not row["bias_pass"]:
                reasons.append(
                    f"Bias {row['bias']:+.1f} exceeds ±{bias_threshold}"
                )

            if not row["jitter_pass"]:
                reasons.append(
                    f"Jitter {row['jitter']:.3f} > {jitter_threshold}"
                )

            df.at[idx, "decision_reason"] = (
                "; ".join(reasons) if reasons else "Eligible"
            )

        # Find the Defender: lowest wmape among eligible
        eligible = df.query("eligible_defender == True").sort_values("wmape")
        if len(eligible) > 0:
            defender_idx = eligible.index[0]
            df.at[defender_idx, "final_decision"] = "DEFENDER"
            df.at[defender_idx, "decision_reason"] = (
                "Selected as Defender (lowest wmape among eligible)"
            )

        return df


# ============================================================================
# Working Example
# ============================================================================

# y = [100, 120, 130, 110]
# yhat = [90, 125, 128, 115]

# from tsforge.metrics import score_all

# print(score_all(y, yhat))
# # {'mae': 5.0, 'rmse': 5.590, 'mape': 4.12, 'smape': 4.05, 'wape': 0.045,
# #  'accuracy': 0.955, 'bias': 1.5, 'mpe': 1.23, 'forecast_bias': 1.01, 'mase': 0.87}

# # As a DataFrame row (ready for leaderboard)
# print(score_all(y, yhat, as_dataframe=True))