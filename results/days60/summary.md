# Results summary

Test period 2020-02-18 to 2020-03-01: 34,441 (time step, turbine) pairs, 30 turbines, forecast 15 minutes ahead from the previous 12 steps. Errors are in kW. Neural models: mean ± sample std over seeds.

| Model | RMSE (kW) | MAE (kW) | R² | RAE | MAPE (%) | Parameters |
|---|---|---|---|---|---|---|
| Mean power | 542.6 | 374.2 | -0.293 | 0.935 | 62.7 | n/a |
| Persistence | 95.5 | 48.5 | 0.960 | 0.121 | 15.3 | n/a |
| Power curve | 306.4 | 145.9 | 0.588 | 0.364 | 28.1 | n/a |
| LSTM | 333.6 ± 7.1 | 204.9 ± 4.0 | 0.511 ± 0.021 | 0.512 ± 0.010 | 43.8 ± 1.8 | 18,241 |
| TCN | 276.4 ± 35.7 | 162.4 ± 29.9 | 0.661 ± 0.088 | 0.406 ± 0.075 | 37.0 ± 6.4 | 6,753 |
| Transformer | 293.9 ± 8.9 | 165.3 ± 5.3 | 0.621 ± 0.023 | 0.413 ± 0.013 | 34.4 ± 0.6 | 124,318 |
| GCN | 213.6 ± 19.5 | 128.1 ± 12.5 | 0.799 ± 0.036 | 0.320 ± 0.031 | 30.2 ± 2.8 | 1,633 |
| GAT | 215.5 ± 9.5 | 130.9 ± 6.8 | 0.796 ± 0.018 | 0.327 ± 0.017 | 32.2 ± 3.9 | 5,601 |
| GCN-LSTM | 312.9 ± 12.5 | 188.6 ± 10.6 | 0.570 ± 0.034 | 0.471 ± 0.027 | 40.0 ± 2.2 | 8,673 |
| GAT-LSTM | 299.8 ± 19.7 | 178.2 ± 13.3 | 0.604 ± 0.051 | 0.445 ± 0.033 | 37.6 ± 2.8 | 13,089 |
| GAT-Transformer | 250.4 ± 18.3 | 143.7 ± 12.0 | 0.724 ± 0.041 | 0.359 ± 0.030 | 31.9 ± 3.5 | 351,390 |

RAE is the summed absolute error relative to always predicting the test-period mean (1.0 = no better). MAPE only counts pairs whose measured power is at least 100 kW, because the percentage error of a near-zero reading is meaningless.
