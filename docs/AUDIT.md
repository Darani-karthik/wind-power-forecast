# What changed from the original prototype, and why

This repository grew out of a prototype: a set of loose scripts (`preprocess_data.py`, `create_sequences.py`,
`create_spatiotemporal_dataset.py`, `build_graph.py`, one script per model, `main.py`) that trained an LSTM,
a TCN, GCN / GAT variants and a GAT-"Crossformer" on part of the SDWPF wind-farm data.

Before publishing it, every script was read and the prototype was run on the real data. Each finding below was
checked against the data or by executing the code, not assumed. The idea of the project is sound; the problems
were in the plumbing, and several of them changed what the results meant.

## Findings

| # | Finding | How it was checked | Fixed here by |
|---|---|---|---|
| 1 | **The graph described different turbines than the data.** The data used the first 30 `TurbID`s in *file order* (34-39 and 45-68), the graph was built from the coordinates of `TurbID` 1-30. The two sets share no turbine, so every GCN / GAT neighbour link was meaningless. | Compared the `TurbID`s in `processed_wind.parquet` with the ones `build_graph.py` selects: zero overlap. | The graph is built from the coordinates of exactly the turbines in the data, in the same order (`src/graph.py`, regression test in `tests/test_graph.py`). |
| 2 | **The data is sampled every 15 minutes in 2020, not every 10.** The prototype window (the first 60 days) is entirely 15-minute data, so "12 steps = 2 hours of history" is really 3 hours, and the forecast is 15 minutes ahead. 2021 is sampled every 10 minutes. | Spacing of the file's timestamps per month: every step in 2020 is 15 minutes (the window's 5,761 timestamps are all 15 minutes apart), every step in 2021 is 10 minutes. | The interval is read from the data, a period that mixes both cadences is rejected (`build_grid`), and the results state the real horizon. |
| 3 | **Dropping rows with negative power threw most of the data away and broke time continuity.** `Patv < 0` is an idle turbine drawing power, i.e. zero generation. Removing those rows left 99,607 of 172,830 rows (57.6%); the spatio-temporal tensor kept only the 953 timestamps (of 5,761) at which *all* 30 turbines still had a row after cleaning, i.e. 941 samples. Sliding windows over the remaining rows also jumped across the removed time. | Row counts of the prototype's own output files. | Negative readings are clipped to 0 kW, rows are kept, and a window is only used when all of its steps were really recorded. |
| 4 | **The TCN ignored 11 of its 12 input steps.** Both-sided padding with no trimming, then reading the last output position, meant the output only depended on the final time step. | Gradient of the output with respect to each input step is exactly 0 for steps 1-11. | Causal (left-padded) convolutions; regression test `test_tcn_uses_every_step_of_the_window`. |
| 5 | **Train and test overlapped.** The windows overlap by 11 of 12 steps, and `train_test_split` shuffled them, so near-identical windows landed on both sides. The scalers were also fitted on the whole dataset, test period included. | Reading `train_model.py` and the preprocessing scripts. | Chronological train / validation / test split by time, scalers fitted on the training period only (`test_scaler_ignores_the_validation_and_test_periods`). |
| 6 | **The models were barely trained.** Each model was trained for 10 full-batch epochs, i.e. 10 gradient steps. The loss was still about 1.0 (the variance of the standardised target) when training stopped. | Running `main.py`: LSTM loss 0.9955 -> 0.8600, TCN 1.0050 -> 0.9778. | Mini-batch training with early stopping on the validation period. |
| 7 | **The reported metrics were not interpretable.** RMSE was computed on standardised power, and MAPE divided by standardised values that are close to zero (and by `y + 1e-6` in the graph scripts), giving 99-338 %. | Running the prototype: LSTM MAPE 112.08, TCN 99.13, GCN 337.62. | Every metric is computed in kW; MAPE only counts readings of at least 100 kW. |
| 8 | **"Crossformer" was not a Crossformer.** The model is a single self-attention block over the time steps with no positional information; it has none of Crossformer's cross-dimension attention or segment merging. | Reading `crossformer_model.py` / `gat_crossformer_model.py`. | Renamed `Transformer` and `GAT-Transformer`, with a learned positional embedding. |
| 9 | **The "validation" curve of the GAT-Crossformer was the test set.** | Reading `gat_crossformer_model.py`. | A separate validation period is used for early stopping; the test period is only scored. |
| 10 | **Sixty days is too short to conclude anything.** The window is calm at the start and stormy later: mean wind speed is 3.3 m/s in the first 70 % (only 1.6 % of readings above 12 m/s) but 7.9 m/s in the following validation period (25 % above 12 m/s), so the models are tested on wind they have barely seen. In this repo's 3-seed runs on that window the test RMSE of one model varied between seeds with a standard deviation of up to 36 kW (TCN), against at most 13 kW on the full year, and the ranking changes: the GCN is best on 60 days (213.6 kW) but behind the TCN on the full year (217.9 vs 188.2 kW), the LSTM is the worst of the eight on 60 days (333.6 kW) but beats the GCN and the Transformer on the full year (200.2 vs 217.9 and 252.7 kW). | Per-period statistics of the prototype window; the same pipeline run on 60 days and on the full year (`results/days60/` vs `results/`). | The default period is the whole of 2020 (the part sampled every 15 minutes); `--days 60` still gives the quick run. |
| 11 | **Engineering.** Hard-coded `C:\Users\...` paths; model files that train when imported and read `Data/...` relative to the working directory; `plt.show()` calls that block headless runs; the graph models looped over samples, nodes and time steps in Python; duplicate `- Copy` data files (about 60 MB); the `.venv` inside the project folder; no README, requirements, tests or `.gitignore`. | Reading the project folder. | A package with one entry point (`python -m src.run_all`), relative paths, vectorised graph models, tests and CI. |

The accompanying `project_analysis_report.md.resolved` is not carried over. Its dataset numbers hold (134 turbines,
11,361,190 rows, 496,998 rows with missing sensor values, power between -9.31 and 1561.38 kW), but it states that
the data is sampled in 10-minute intervals and that dropping negative power was correct, and its links point to
local files.

## Prototype results, for reference

Running the prototype's own scripts on the data it ships with (standardised units, 10 full-batch epochs, random
split, so none of these numbers are comparable with the ones in the README):

| Model | RMSE (standardised) | R2 |
|---|---|---|
| LSTM | 0.9223 | 0.156 |
| TCN | 0.9910 | 0.026 |
| GCN (941 samples) | 0.8426 | 0.316 |

## What was kept

The task (forecast the active power `Patv` of each turbine one step ahead from the previous 12 steps of five
inputs: `Wspd`, `Wdir`, `Etmp`, `Itmp`, `RelH`), the first 30 turbines (`TurbID` 1-30), a 5-nearest-neighbour
turbine graph, and the model families: LSTM, TCN, GCN, GAT, GCN-LSTM, GAT-LSTM and a GAT + attention hybrid.
The 60-day window is still available as `--days 60`.
