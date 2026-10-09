"""Reference forecasts that every neural model has to beat to be worth its complexity.

All three return kW predictions of shape (len(times), N) for the given target times.
"""
from __future__ import annotations

import numpy as np

from .data import Dataset

BIN_WIDTH = 0.5  # m/s, width of the wind-speed bins of the power curve


def mean_forecast(ds: Dataset, times: np.ndarray) -> np.ndarray:
    """Always the average power of the training period."""
    return np.full((len(times), ds.grid.y.shape[1]), ds.scaler.y_mean, dtype=np.float32)


def persistence_forecast(ds: Dataset, times: np.ndarray) -> np.ndarray:
    """The last measured power, repeated. The neural models are not given past power, so this one has an edge."""
    return ds.grid.y[times - 1]


class PowerCurve:
    """Mean power per wind-speed bin, learned on the training targets and looked up with the latest wind speed."""

    def __init__(self, ds: Dataset):
        wind = ds.grid.x[:, :, 0]  # Wspd is the first feature
        times = ds.train
        mask = ds.eligible[times]
        speeds = wind[times - 1][mask]  # the wind speed of the last input step
        power = ds.grid.y[times][mask]

        bins = np.floor(speeds / BIN_WIDTH).astype(int)
        used = np.unique(bins)
        self.centres = (used + 0.5) * BIN_WIDTH
        self.means = np.array([power[bins == b].mean() for b in used])

    def __call__(self, ds: Dataset, times: np.ndarray) -> np.ndarray:
        speeds = ds.grid.x[times - 1, :, 0]
        speeds = np.where(np.isfinite(speeds), speeds, 0.0)  # an unrecorded step reads as calm
        return np.interp(speeds, self.centres, self.means).astype(np.float32)


def baseline_forecasts(ds: Dataset, times: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Mean power": mean_forecast(ds, times),
        "Persistence": persistence_forecast(ds, times),
        "Power curve": PowerCurve(ds)(ds, times),
    }
