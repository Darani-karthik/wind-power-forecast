import json

from src import run_all
from src.data import DATA_FILE, LOCATION_FILE
from tests.conftest import make_frame, make_locations


def test_end_to_end_on_synthetic_data(tmp_path):
    data_dir, results = tmp_path / "data", tmp_path / "results"
    data_dir.mkdir()
    make_frame().to_parquet(data_dir / DATA_FILE)
    make_locations().to_csv(data_dir / LOCATION_FILE, index=False)

    run_all.main(["--data-dir", str(data_dir), "--results-dir", str(results), "--models", "LSTM", "GCN",
                  "--seeds", "0", "--turbines", "4", "--days", "10", "--epochs", "1", "--k", "2",
                  "--batch-size", "16"])

    names = {p.stem for p in (results / "metrics").glob("*.json")}
    assert names == {"baseline_mean_power", "baseline_persistence", "baseline_power_curve",
                     "lstm_seed0", "gcn_seed0"}
    for figure in ("model_comparison", "forecast_example", "turbine_graph"):
        assert (results / "figures" / f"{figure}.png").stat().st_size > 0

    summary = (results / "summary.md").read_text(encoding="utf-8")
    assert "| LSTM |" in summary and "| Persistence |" in summary

    data = json.loads((results / "data_summary.json").read_text())
    assert data["step_minutes"] == 15 and data["turbines"] == 4
