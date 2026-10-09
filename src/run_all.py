"""Run the full experiment: reference forecasts, every model and seed, then the summary and figures."""
from __future__ import annotations

import json
from pathlib import Path

from .baselines import baseline_forecasts
from .cli import parse_args
from .data import SEQ_LEN, Dataset, prepare
from .graph import turbine_graph
from .make_report import build
from .plots import plot_forecast_example, plot_turbine_graph
from .training import TrainConfig, run_model, score


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def describe_data(ds: Dataset) -> dict:
    """What was actually used, written next to the results so the README numbers can be checked."""
    grid = ds.grid
    splits = {}
    for name in ("train", "val", "test"):
        times = getattr(ds, name)
        splits[name] = {"from": str(grid.times[times[0]]), "to": str(grid.times[times[-1]]),
                        "steps": int(len(times)), "pairs": int(ds.eligible[times].sum())}
    return {"turbines": len(grid.turbine_ids), "turbine_ids": grid.turbine_ids,
            "period": [str(grid.times[0]), str(grid.times[-1])],
            "step_minutes": int(grid.step.total_seconds() // 60), "seq_len": SEQ_LEN,
            "grid_steps": len(grid.times), "recorded_share": round(float(grid.valid.mean()), 4),
            "power_mean_kw": round(ds.scaler.y_mean, 1), "power_std_kw": round(ds.scaler.y_std, 1),
            "splits": splits}


def main(argv=None) -> None:
    args = parse_args("Run every model and build the summary", argv)
    results_dir = Path(args.results_dir)
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)

    ds, locations = prepare(args.data_dir, args.turbines, args.days)
    edge_index = turbine_graph(locations, ds.grid.turbine_ids, args.k)
    data = describe_data(ds)
    (results_dir / "data_summary.json").write_text(json.dumps(data, indent=2))
    print(f"{data['turbines']} turbines, {data['grid_steps']} steps of {data['step_minutes']} min, "
          f"test pairs {data['splits']['test']['pairs']:,}")

    for name, pred in baseline_forecasts(ds, ds.test).items():
        result = {"model": name, "kind": "baseline", **score(ds, pred, ds.test)}
        (results_dir / "metrics" / f"baseline_{_slug(name)}.json").write_text(json.dumps(result, indent=2))

    cfg = TrainConfig(args.epochs, args.lr, args.batch_size, args.patience)
    forecasts = {}
    for name in args.models:
        for seed in args.seeds:
            result, pred = run_model(name, ds, edge_index, seed, cfg)
            result["kind"] = "model"
            (results_dir / "metrics" / f"{_slug(name)}_seed{seed}.json").write_text(json.dumps(result, indent=2))
            print(f"  -> test RMSE {result['RMSE']:.1f} kW, R2 {result['R2']:.3f}\n", flush=True)
            if seed == args.seeds[0]:
                forecasts[name] = pred

    rows = build(results_dir)
    figures = results_dir / "figures"
    plot_turbine_graph(locations, ds.grid.turbine_ids, edge_index, args.k, figures / "turbine_graph.png")
    best = min((r for r in rows if r["model"] in forecasts), key=lambda r: r["RMSE"])["model"]
    plot_forecast_example(ds, forecasts[best], best, figures / "forecast_example.png")
    print(f"Done. Best model by RMSE: {best}. Results in {results_dir}/")


if __name__ == "__main__":
    main()
