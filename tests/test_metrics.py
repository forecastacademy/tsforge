import numpy as np
import pandas as pd
import pytest
from dataclasses import fields

from tsforge.evaluation import (
    mae, mse, rmse,
    mape, smape, wape, business_accuracy,
    mase, bias, mean_percentage_error, forecast_bias,
    score_all
)
from tsforge.evaluation.metrics import MetricsCalculator, DefenderDecision

# Simple toy data
y     = np.array([100, 200, 300, 400])
yhat  = np.array([110, 190, 290, 410])   # small errors
y_naive = np.array([95, 205, 295, 405])  # naive-1 for MASE


def test_mae():
    assert np.isclose(mae(y, yhat), 10.0)


def test_mse_and_rmse():
    assert np.isclose(mse(y, yhat), 100.0)
    assert np.isclose(rmse(y, yhat), 10.0)


def test_mape_and_smape():
    mape_val = mape(y, yhat)
    smape_val = smape(y, yhat)
    assert mape_val > 0
    assert smape_val > 0
    # With symmetric small errors, mape and smape should be in the same ballpark
    assert abs(mape_val - smape_val) < 2.0


def test_wape_and_accuracy():
    w = wape(y, yhat)
    acc = business_accuracy(y, yhat)
    assert np.isclose(acc, 1 - w, atol=1e-8)
    assert 0 <= acc <= 1


def test_mase_with_naive():
    val = mase(y, yhat, y_naive=y_naive)
    assert val >= 0
    # Should be close to 2.0 given this data
    assert 1.5 <= val <= 2.5


def test_bias_metrics():
    b = bias(y, yhat)
    mpe = mean_percentage_error(y, yhat)
    fb = forecast_bias(y, yhat)
    assert np.isclose(b, 0.0, atol=20)       # small bias
    assert isinstance(mpe, float)
    assert isinstance(fb, float)
    # Forecast bias ratio should be near 1.0
    assert 0.9 <= fb <= 1.1


def test_score_all_dict_and_dataframe():
    scores = score_all(y, yhat)
    assert isinstance(scores, dict)
    assert "mae" in scores
    assert "accuracy" in scores
    assert "bias" in scores

    scores_df = score_all(y, yhat, as_dataframe=True)
    assert "mae" in scores_df.columns
    assert "accuracy" in scores_df.columns
    assert "bias" in scores_df.columns
    assert len(scores_df) == 1


# ============================================================================
# Tests for DefenderDecision Dataclass (Task 1)
# ============================================================================


def test_defender_decision_dataclass():
    """Verify DefenderDecision has required fields."""
    required_fields = {"model", "final_decision", "decision_reason"}
    actual_fields = {f.name for f in fields(DefenderDecision)}
    assert required_fields == actual_fields, f"Missing fields: {required_fields - actual_fields}"


def test_defender_decision_creation():
    """Verify DefenderDecision can be instantiated."""
    decision = DefenderDecision(
        model="TestModel",
        final_decision="DEFENDER",
        decision_reason="Selected as Defender"
    )
    assert decision.model == "TestModel"
    assert decision.final_decision == "DEFENDER"


# ============================================================================
# Tests for MetricsCalculator Class (Tasks 2-4)
# ============================================================================


def test_metrics_calculator_init():
    """Verify MetricsCalculator initializes with defaults."""
    calc = MetricsCalculator()
    assert calc.anchor_model == "SN52"
    assert calc.model_col == "model"
    assert calc.metric_col == "wmape"


def test_metrics_calculator_custom_anchor():
    """Verify MetricsCalculator accepts custom anchor model."""
    calc = MetricsCalculator(anchor_model="Naive")
    assert calc.anchor_model == "Naive"


# ============================================================================
# Tests for compute_improvement_distribution() (Task 3)
# ============================================================================


def test_compute_improvement_distribution_basic():
    """Test compute_improvement_distribution with toy data."""
    calc = MetricsCalculator(anchor_model="SN52")

    # Create toy metric_level results
    metric_level_df = pd.DataFrame({
        "model": ["SN52", "MA4", "SN52", "MA4"],
        "item_id": [1, 1, 2, 2],
        "wmape": [0.20, 0.15, 0.25, 0.18],
        "cutoff": ["2025-01-01"] * 4,
    })

    improvement = calc.compute_improvement_distribution(metric_level_df)

    # Check structure
    assert "model" in improvement.columns
    assert "item_id" in improvement.columns
    assert "pct_improvement" in improvement.columns

    # Check values (MA4 vs SN52)
    # Item 1: (0.20 - 0.15) / 0.20 * 100 = 25%
    ma4_item1 = improvement[(improvement["model"] == "MA4") & (improvement["item_id"] == 1)]
    assert np.isclose(ma4_item1["pct_improvement"].values[0], 25.0, atol=0.1)


def test_compute_improvement_distribution_filtering():
    """Test that very small anchor wmape values are filtered."""
    calc = MetricsCalculator()

    metric_level_df = pd.DataFrame({
        "model": ["SN52", "MA4", "SN52", "MA4"],
        "item_id": [1, 1, 2, 2],
        "wmape": [0.001, 0.0005, 0.25, 0.18],  # First anchor is very small
        "cutoff": ["2025-01-01"] * 4,
    })

    improvement = calc.compute_improvement_distribution(metric_level_df)

    # Should filter out item 1 (anchor_wmape = 0.001 < 0.01)
    assert 1 not in improvement["item_id"].values
    assert 2 in improvement["item_id"].values


def test_compute_improvement_distribution_negative_improvement():
    """Test that negative improvement (worse than anchor) is computed correctly."""
    calc = MetricsCalculator()

    metric_level_df = pd.DataFrame({
        "model": ["SN52", "BadModel"],
        "item_id": [1, 1],
        "wmape": [0.20, 0.30],  # BadModel is worse
        "cutoff": ["2025-01-01"] * 2,
    })

    improvement = calc.compute_improvement_distribution(metric_level_df)

    # (0.20 - 0.30) / 0.20 * 100 = -50%
    bad_improvement = improvement[improvement["model"] == "BadModel"]
    assert np.isclose(bad_improvement["pct_improvement"].values[0], -50.0, atol=0.1)


# ============================================================================
# Tests for select_defender() (Task 4)
# ============================================================================


def test_select_defender_basic():
    """Test select_defender with toy portfolio data."""
    calc = MetricsCalculator(anchor_model="SN52")

    portfolio_df = pd.DataFrame({
        "model": ["SN52", "MA4", "Croston"],
        "wmape": [0.30, 0.25, 0.24],  # Croston is best
        "beat_rate": [0.0, 52.0, 57.0],  # Only Croston passes gate (>50%)
        "bias": [0.02, 0.01, -0.03],
        "jitter": [0.1, 0.09, 0.08],
    })

    decisions = calc.select_defender(portfolio_df)

    # Check structure
    assert len(decisions) == 3
    assert "final_decision" in decisions.columns
    assert "decision_reason" in decisions.columns

    # Croston should be selected (beats anchor, best wmape among eligible)
    croston_decision = decisions[decisions["model"] == "Croston"].iloc[0]
    assert croston_decision["final_decision"] == "DEFENDER"
    assert "Selected as Defender" in croston_decision["decision_reason"]


def test_select_defender_fails_anchor_gate():
    """Test that models worse than anchor are rejected."""
    calc = MetricsCalculator(anchor_model="SN52")

    portfolio_df = pd.DataFrame({
        "model": ["SN52", "BadModel"],
        "wmape": [0.30, 0.35],  # BadModel is worse
        "beat_rate": [0.0, 51.0],  # BadModel beats gate threshold but not anchor
        "bias": [0.02, 0.01],
        "jitter": [0.1, 0.09],
    })

    decisions = calc.select_defender(portfolio_df)

    bad_decision = decisions[decisions["model"] == "BadModel"].iloc[0]
    assert bad_decision["final_decision"] == "REJECT"
    assert "Failed Anchor Gate" in bad_decision["decision_reason"]


def test_select_defender_fails_beat_rate():
    """Test that low beat rate triggers rejection."""
    calc = MetricsCalculator(anchor_model="SN52")

    portfolio_df = pd.DataFrame({
        "model": ["SN52", "WeakModel"],
        "wmape": [0.30, 0.25],  # Better than anchor
        "beat_rate": [0.0, 45.0],  # But beat_rate too low (<50%)
        "bias": [0.02, 0.01],
        "jitter": [0.1, 0.09],
    })

    decisions = calc.select_defender(portfolio_df)

    weak_decision = decisions[decisions["model"] == "WeakModel"].iloc[0]
    assert weak_decision["final_decision"] == "REJECT"
    assert "Beat Rate" in weak_decision["decision_reason"]


def test_select_defender_multiple_eligible():
    """Test that best wmape is selected among multiple eligible models."""
    calc = MetricsCalculator(anchor_model="SN52")

    portfolio_df = pd.DataFrame({
        "model": ["SN52", "Model1", "Model2", "Model3"],
        "wmape": [0.30, 0.27, 0.25, 0.26],  # Model2 is best
        "beat_rate": [0.0, 55.0, 56.0, 54.0],  # All pass gate
        "bias": [0.02, 0.01, 0.01, 0.01],
        "jitter": [0.1, 0.09, 0.08, 0.09],
    })

    decisions = calc.select_defender(portfolio_df)

    defender = decisions[decisions["final_decision"] == "DEFENDER"].iloc[0]
    assert defender["model"] == "Model2"


def test_select_defender_custom_thresholds():
    """Test select_defender with custom thresholds."""
    calc = MetricsCalculator(anchor_model="SN52")

    portfolio_df = pd.DataFrame({
        "model": ["SN52", "Strict"],
        "wmape": [0.30, 0.25],
        "beat_rate": [0.0, 55.0],  # Passes default (50%) but not strict (70%)
        "bias": [0.02, 0.15],  # Fails strict bias threshold (0.05)
        "jitter": [0.1, 0.09],
    })

    decisions = calc.select_defender(
        portfolio_df,
        beat_rate_threshold=70.0,
        bias_threshold=0.05
    )

    strict_decision = decisions[decisions["model"] == "Strict"].iloc[0]
    assert strict_decision["final_decision"] == "REJECT"
    assert "Beat Rate" in strict_decision["decision_reason"]
    assert "Bias" in strict_decision["decision_reason"]


def test_select_defender_no_eligible():
    """Test behavior when no model passes all gates."""
    calc = MetricsCalculator(anchor_model="SN52")

    portfolio_df = pd.DataFrame({
        "model": ["SN52", "OnlyOption"],
        "wmape": [0.30, 0.35],  # Worse than anchor
        "beat_rate": [0.0, 45.0],  # Also low beat rate
        "bias": [0.02, 0.01],
        "jitter": [0.1, 0.09],
    })

    decisions = calc.select_defender(portfolio_df)

    # Should have no DEFENDER selected
    defenders = decisions[decisions["final_decision"] == "DEFENDER"]
    assert len(defenders) == 0

    # All should be REJECT
    assert (decisions["final_decision"] == "REJECT").all()


# ============================================================================
# Tests for Module Imports (Task 5)
# ============================================================================


def test_imports_from_evaluation():
    """Test that MetricsCalculator can be imported from tsforge.evaluation."""
    from tsforge.evaluation import MetricsCalculator, DefenderDecision

    assert MetricsCalculator is not None
    assert DefenderDecision is not None
