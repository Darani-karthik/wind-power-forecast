"""Turn results/metrics/*.json into results/summary.md and the model comparison figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .models import MODEL_NAMES
from .plots import plot_model_comparison

METRICS = ["RMSE", "MAE", "R2", "RAE", "MAPE"]
DIGITS = {"RMSE": 1, "MAE": 1, "R2": 3, "RAE": 3, "MAPE": 1}
BASELINES = ["Mean power", "Persistence", "Power curve"]


def load_results(results_dir) -> list[dict]:
    files = sorted((Path(results_dir) / "metrics").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No metric files in '{results_dir}/metrics'. Run `python -m src.run_all` first.")
    return [json.loads(f.read_text()) for f in files]


def summarise(results: list[dict]) -> list[dict]:
    """One row per model: mean and sample std over seeds. Reference forecasts first, then the models."""
    order = BASELINES + MODEL_NAMES
    rows = []
    for name in sorted({r["model"] for r in results}, key=order.index):
        runs = [r for r in results if r["model"] == name]
        row = {"model": name, "kind": runs[0]["kind"], "seeds": len(runs), "parameters": runs[0].get("parameters")}
        for metric in METRICS:
            values = np.array([r[metric] for r in runs], dtype=float)
            row[metric] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        rows.append(row)
    return rows


def _cell(row: dict, metric: str) -> str:
    digits = DIGITS[metric]
    text = f"{row[metric]:.{digits}f}"
    std = row[f"{metric}_std"]
    return f"{text} ± {std:.{digits}f}" if std is not None else text


def write_summary(rows: list[dict], data: dict, path) -> None:
    test = data["splits"]["test"]
    lines = [
        "# Results summary",
        "",
        f"Test period {test['from'][:10]} to {test['to'][:10]}: {test['pairs']:,} (time step, turbine) pairs, "
        f"{data['turbines']} turbines, forecast {data['step_minutes']} minutes ahead from the previous "
        f"{data['seq_len']} steps. Errors are in kW. Neural models: mean ± sample std over seeds.",
        "",
        "| Model | RMSE (kW) | MAE (kW) | R² | RAE | MAPE (%) | Parameters |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        params = f"{row['parameters']:,}" if row["parameters"] else "n/a"
        cells = " | ".join(_cell(row, m) for m in METRICS)
        lines.append(f"| {row['model']} | {cells} | {params} |")
    lines += [
        "",
        "RAE is the summed absolute error relative to always predicting the test-period mean (1.0 = no better). "
        "MAPE only counts pairs whose measured power is at least 100 kW, because the percentage error "
        "of a near-zero reading is meaningless.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def build(results_dir) -> list[dict]:
    results_dir = Path(results_dir)
    data = json.loads((results_dir / "data_summary.json").read_text())
    rows = summarise(load_results(results_dir))
    write_summary(rows, data, results_dir / "summary.md")

    test = data["splits"]["test"]
    seeds = max(r["seeds"] for r in rows)
    subtitle = (f"{data['turbines']} turbines, {test['from'][:10]} to {test['to'][:10]}, "
                f"{data['step_minutes']}-minute-ahead forecast, {seeds} seeds")
    plot_model_comparison(
        [{"model": r["model"], "kind": r["kind"], "RMSE": r["RMSE"], "RMSE_std": r["RMSE_std"]} for r in rows],
        subtitle, results_dir / "figures" / "model_comparison.png")
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Rebuild summary.md and the comparison figure from saved metrics")
    parser.add_argument("--results-dir", default="results")
    build(parser.parse_args(argv).results_dir)


if __name__ == "__main__":
    main()
