import numpy as np
import pytest
import torch

from src.training import TrainConfig, masked_squared_error, predict, run_model

QUIET = {"log": lambda message: None}


def test_masked_error_ignores_unscored_pairs():
    pred = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[True, False]])
    assert masked_squared_error(pred, target, mask).item() == 1.0


@pytest.mark.parametrize("name", ["LSTM", "GCN"])
def test_training_runs_and_is_reproducible(name, dataset, edge_index):
    cfg = TrainConfig(epochs=2, batch_size=16)
    first, pred_a = run_model(name, dataset, edge_index, 0, cfg, **QUIET)
    second, pred_b = run_model(name, dataset, edge_index, 0, cfg, **QUIET)

    assert np.isfinite([first["RMSE"], first["R2"]]).all()
    assert first["RMSE"] == second["RMSE"]
    np.testing.assert_array_equal(pred_a, pred_b)
    assert pred_a.shape == (len(dataset.test), 4)


def test_different_seeds_give_different_models(dataset, edge_index):
    cfg = TrainConfig(epochs=1, batch_size=16)
    a, _ = run_model("LSTM", dataset, edge_index, 0, cfg, **QUIET)
    b, _ = run_model("LSTM", dataset, edge_index, 1, cfg, **QUIET)
    assert a["RMSE"] != b["RMSE"]


def test_early_stopping_restores_the_best_epoch(dataset, edge_index):
    result, _ = run_model("TCN", dataset, edge_index, 0, TrainConfig(epochs=6, batch_size=16, patience=1), **QUIET)
    assert 1 <= result["best_epoch"] <= result["epochs_run"] <= 6


def test_forecasts_are_in_kw(dataset, edge_index):
    _, pred = run_model("LSTM", dataset, edge_index, 0, TrainConfig(epochs=3, batch_size=16), **QUIET)
    observed = dataset.grid.y[dataset.test][dataset.eligible[dataset.test]]
    # a standardised output would sit around 0; kW forecasts live on the scale of the measurements
    assert abs(pred.mean() - observed.mean()) < observed.std() * 2
    assert pred.std() > 1.0
