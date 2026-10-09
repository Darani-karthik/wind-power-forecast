import numpy as np

from src.baselines import baseline_forecasts, persistence_forecast
from src.training import score


def test_persistence_repeats_the_previous_measurement(dataset):
    times = dataset.test
    mask = dataset.eligible[times]
    pred = persistence_forecast(dataset, times)
    np.testing.assert_array_equal(pred[mask], dataset.grid.y[times - 1][mask])


def test_mean_forecast_is_constant(dataset):
    pred = baseline_forecasts(dataset, dataset.test)["Mean power"]
    assert pred.shape == (len(dataset.test), 4)
    assert np.unique(pred).size == 1


def test_power_curve_beats_the_mean_when_power_follows_the_wind(dataset):
    errors = {name: score(dataset, pred, dataset.test)["RMSE"]
              for name, pred in baseline_forecasts(dataset, dataset.test).items()}
    assert errors["Power curve"] < errors["Mean power"]
