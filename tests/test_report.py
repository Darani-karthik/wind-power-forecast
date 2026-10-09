import numpy as np

from src.make_report import fewer_seeds_note, summarise


def row(model, seeds, kind="model"):
    return {"model": model, "kind": kind, "seeds": seeds}


def test_note_lists_the_models_that_ran_with_fewer_seeds():
    rows = [row("LSTM", 3), row("GCN-LSTM", 1), row("GAT-LSTM", 1), row("Persistence", 1, "baseline")]
    assert fewer_seeds_note(rows) == "1 seed for GCN-LSTM, GAT-LSTM"


def test_no_note_when_every_model_has_the_same_number_of_seeds():
    assert fewer_seeds_note([row("LSTM", 3), row("TCN", 3)]) == ""


def test_summary_has_a_spread_only_for_models_with_several_seeds():
    metrics = {"RMSE": 1.0, "MAE": 1.0, "R2": 0.5, "RAE": 0.5, "MAPE": 10.0}
    results = [{"model": "LSTM", "kind": "model", "parameters": 5, **metrics, "RMSE": value} for value in (1.0, 3.0)]
    results.append({"model": "GCN", "kind": "model", "parameters": 7, **metrics})
    rows = {r["model"]: r for r in summarise(results)}
    assert rows["LSTM"]["RMSE"] == 2.0 and np.isclose(rows["LSTM"]["RMSE_std"], np.sqrt(2.0))
    assert rows["GCN"]["RMSE_std"] is None
