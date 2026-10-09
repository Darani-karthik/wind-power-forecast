import numpy as np
import pytest

from src.metrics import compute_metrics


def test_known_errors():
    y_true = np.array([0.0, 100.0, 200.0, 300.0])
    y_pred = np.array([10.0, 110.0, 190.0, 310.0])
    m = compute_metrics(y_true, y_pred)
    assert m["RMSE"] == pytest.approx(10.0)
    assert m["MAE"] == pytest.approx(10.0)
    assert m["MAPE"] == pytest.approx((10 / 100 + 10 / 200 + 10 / 300) / 3 * 100)  # the 0 kW reading is skipped


def test_perfect_forecast():
    y = np.array([5.0, 150.0, 900.0])
    m = compute_metrics(y, y)
    assert (m["RMSE"], m["MAE"], m["RAE"], m["MAPE"]) == (0.0, 0.0, 0.0, 0.0)
    assert m["R2"] == 1.0


def test_predicting_the_mean_scores_zero_r2_and_unit_rae():
    y = np.array([0.0, 100.0, 400.0, 700.0])
    m = compute_metrics(y, np.full_like(y, y.mean()))
    assert m["R2"] == pytest.approx(0.0)
    assert m["RAE"] == pytest.approx(1.0)


def test_mape_is_nan_when_nothing_is_above_the_threshold():
    assert np.isnan(compute_metrics([0.0, 1.0], [1.0, 2.0])["MAPE"])
