# Results summary

Test period 2020-10-19 to 2020-12-31: 208,390 (time step, turbine) pairs, 30 turbines, forecast 15 minutes ahead from the previous 12 steps. Errors are in kW. Neural models: mean ± sample std over seeds.

| Model | RMSE (kW) | MAE (kW) | R² | RAE | MAPE (%) | Parameters |
|---|---|---|---|---|---|---|
| Mean power | 458.6 | 372.9 | -0.000 | 0.999 | 68.8 | n/a |
| Persistence | 78.0 | 39.2 | 0.971 | 0.105 | 14.2 | n/a |
| Power curve | 210.6 | 117.7 | 0.789 | 0.315 | 29.5 | n/a |
| LSTM | 200.2 ± 10.0 | 107.2 ± 5.4 | 0.809 ± 0.019 | 0.287 ± 0.014 | 27.2 ± 0.6 | 18,241 |
| TCN | 188.2 ± 5.7 | 101.6 ± 1.6 | 0.831 ± 0.010 | 0.272 ± 0.004 | 28.7 ± 1.7 | 6,753 |
| Transformer | 252.7 ± 5.9 | 146.5 ± 3.1 | 0.696 ± 0.014 | 0.392 ± 0.008 | 36.9 ± 2.0 | 124,318 |
| GCN | 217.9 ± 13.4 | 119.3 ± 9.7 | 0.774 ± 0.028 | 0.320 ± 0.026 | 31.4 ± 2.6 | 1,633 |
| GAT | 176.8 | 100.8 | 0.851 | 0.270 | 25.7 | 5,601 |
| GCN-LSTM | 212.6 | 112.6 | 0.785 | 0.302 | 27.0 | 8,673 |
| GAT-LSTM | 205.1 | 103.7 | 0.800 | 0.278 | 27.1 | 13,089 |
| GAT-Transformer | 206.5 | 111.0 | 0.797 | 0.297 | 27.4 | 351,390 |

Run with fewer seeds to save time: 1 seed for GAT, GCN-LSTM, GAT-LSTM, GAT-Transformer. Their rows have no ± spread and are indicative only.

RAE is the summed absolute error relative to always predicting the test-period mean (1.0 = no better). MAPE only counts pairs whose measured power is at least 100 kW, because the percentage error of a near-zero reading is meaningless.
