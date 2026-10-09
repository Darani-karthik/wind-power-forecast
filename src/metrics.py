"""Forecast error metrics, always computed in kW (never on standardised values)."""
from __future__ import annotations

import numpy as np

MAPE_MIN_KW = 100.0  # MAPE is only meaningful away from zero, so it is scored where the actual power is >= 100 kW


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_pred - y_true

    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    r2 = float(1.0 - np.sum(error ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    rae = float(np.sum(np.abs(error)) / np.sum(np.abs(y_true - y_true.mean())))  # 1.0 = no better than predicting the mean

    big = y_true >= MAPE_MIN_KW
    mape = float(np.mean(np.abs(error[big] / y_true[big])) * 100) if big.any() else float("nan")

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "RAE": rae, "MAPE": mape}
