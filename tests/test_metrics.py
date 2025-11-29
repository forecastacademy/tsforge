import numpy as np
import pytest

from tsforge.evaluation import (
    mae, mse, rmse,
    mape, smape, wape, business_accuracy,
    mase, bias, mean_percentage_error, forecast_bias,
    score_all
)

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
