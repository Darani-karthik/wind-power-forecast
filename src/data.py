"""SDWPF data handling: turbine selection, the regular time grid, chronological split and scaling."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = "sdwpf_2001_2112_full.parquet"
LOCATION_FILE = "sdwpf_turb_location_elevation.csv"

FEATURES = ["Wspd", "Wdir", "Etmp", "Itmp", "RelH"]  # wind speed / direction, outside / inside temperature, humidity
TARGET = "Patv"                                      # active power in kW
SEQ_LEN = 12                                         # input window: 12 time steps
TRAIN_FRAC, VAL_FRAC = 0.7, 0.1                      # the last 20 % of the period is the test set


@dataclass
class Grid:
    """The selected turbines on one regular time grid; missing readings are NaN."""

    times: pd.DatetimeIndex  # (T,)
    step: pd.Timedelta       # sampling interval of the period (15 min in 2020)
    turbine_ids: list[int]   # (N,)
    x: np.ndarray            # (T, N, F) float32 raw inputs
    y: np.ndarray            # (T, N) float32 power in kW, negative readings clipped to 0
    valid: np.ndarray        # (T, N) bool, the five inputs and the power were all recorded


@dataclass
class Scaler:
    """Mean / std fitted on the training period only."""

    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float


@dataclass
class Dataset:
    """Standardised arrays plus the chronological train / validation / test target times."""

    grid: Grid
    scaler: Scaler
    x: np.ndarray         # (T, N, F) standardised inputs, missing values set to 0 (the training mean)
    y: np.ndarray         # (T, N) standardised power, NaN where missing
    eligible: np.ndarray  # (T, N) bool, target and all SEQ_LEN input steps of that turbine were recorded
    train: np.ndarray     # target time indices
    val: np.ndarray
    test: np.ndarray

    def batch(self, times: np.ndarray):
        """Inputs (B, N, L, F), standardised targets (B, N) and the eligibility mask (B, N) for target times."""
        steps = times[:, None] - SEQ_LEN + np.arange(SEQ_LEN)[None, :]  # the L steps before each target
        x = self.x[steps].transpose(0, 2, 1, 3)
        return x, self.y[times], self.eligible[times]

    def to_kw(self, y_std: np.ndarray) -> np.ndarray:
        """Back from standardised power to kW."""
        return y_std * self.scaler.y_std + self.scaler.y_mean


def default_data_dir() -> str:
    return os.environ.get("SDWPF_DATA_DIR", "data")


def load_raw(data_dir) -> pd.DataFrame:
    """The columns this project uses, for all turbines."""
    path = Path(data_dir) / DATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"'{path}' not found. See data/README.md for how to get the SDWPF files.")
    return pd.read_parquet(path, columns=["TurbID", "Tmstamp", *FEATURES, TARGET])


def load_locations(data_dir) -> pd.DataFrame:
    path = Path(data_dir) / LOCATION_FILE
    if not path.exists():
        raise FileNotFoundError(f"'{path}' not found. See data/README.md for how to get the SDWPF files.")
    return pd.read_csv(path)


def build_grid(df: pd.DataFrame, turbine_ids: list[int], days: int = 60) -> Grid:
    """Place the first `days` days of the chosen turbines on one regular time grid.

    The dataset is sampled every 15 minutes in 2020 but every 10 minutes in 2021, so the step is read
    from the data. A period that mixes both cadences is rejected instead of silently mis-spaced.
    """
    df = df[df["TurbID"].isin(turbine_ids)]
    if df.empty:
        raise ValueError(f"None of the turbines {turbine_ids} are in the data.")
    start = df["Tmstamp"].min()
    df = df[df["Tmstamp"] <= start + pd.Timedelta(days=days)]

    stamps = pd.DatetimeIndex(df["Tmstamp"].unique()).sort_values()
    step = stamps.to_series().diff().mode().iloc[0]
    offsets = np.asarray((stamps - stamps[0]) / step, dtype=float)
    if not np.allclose(offsets, np.round(offsets)):
        raise ValueError("The period is not sampled at one regular interval (the 15 min and 10 min parts "
                         "of the dataset must not be mixed); use fewer days.")

    times = pd.date_range(stamps[0], stamps[-1], freq=step)
    t_idx = ((df["Tmstamp"] - stamps[0]) // step).to_numpy()
    n_idx = df["TurbID"].map({tid: i for i, tid in enumerate(turbine_ids)}).to_numpy()

    x = np.full((len(times), len(turbine_ids), len(FEATURES)), np.nan, dtype=np.float32)
    y = np.full((len(times), len(turbine_ids)), np.nan, dtype=np.float32)
    x[t_idx, n_idx] = df[FEATURES].to_numpy(np.float32)
    # Negative readings are the idle turbine drawing power: no generation, so 0 (rows are not dropped,
    # which would punch holes into the time series).
    y[t_idx, n_idx] = df[TARGET].clip(lower=0).to_numpy(np.float32)

    x[~np.isfinite(x)] = np.nan
    y[~np.isfinite(y)] = np.nan
    valid = np.isfinite(x).all(axis=2) & np.isfinite(y)
    return Grid(times, step, list(turbine_ids), x, y, valid)


def eligible_targets(valid: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """(T, N) bool: the target at t and the `seq_len` steps before it were all recorded for that turbine."""
    eligible = np.zeros_like(valid)
    if valid.shape[0] > seq_len:
        windows = np.lib.stride_tricks.sliding_window_view(valid, seq_len + 1, axis=0)  # (T-L, N, L+1)
        eligible[seq_len:] = windows.all(axis=2)
    return eligible


def split_times(n_times: int, seq_len: int = SEQ_LEN):
    """Chronological train / val / test target times; a training window never reaches past its period."""
    train_end = round(n_times * TRAIN_FRAC)
    val_end = round(n_times * (TRAIN_FRAC + VAL_FRAC))  # round, not int(): 0.7 + 0.1 is 0.7999999999999999
    return np.arange(seq_len, train_end), np.arange(train_end, val_end), np.arange(val_end, n_times)


def fit_scaler(grid: Grid, train_end: int) -> Scaler:
    """Statistics of the recorded values before `train_end` only, so nothing leaks from val / test."""
    ok = grid.valid[:train_end]
    x, y = grid.x[:train_end][ok], grid.y[:train_end][ok]
    return Scaler(x.mean(axis=0), x.std(axis=0) + 1e-6, float(y.mean()), float(y.std() + 1e-6))


def make_dataset(grid: Grid) -> Dataset:
    train, val, test = split_times(len(grid.times))
    scaler = fit_scaler(grid, int(train[-1]) + 1)
    eligible = eligible_targets(grid.valid)

    x = (grid.x - scaler.x_mean) / scaler.x_std
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)  # missing neighbour readings become the training mean
    y = ((grid.y - scaler.y_mean) / scaler.y_std).astype(np.float32)

    # keep only target times where at least one turbine can be scored
    train, val, test = (t[eligible[t].any(axis=1)] for t in (train, val, test))
    return Dataset(grid, scaler, x, y, eligible, train, val, test)


def prepare(data_dir, n_turbines: int = 30, days: int = 60):
    """Raw files -> Dataset, plus the turbine coordinates in the same order as the dataset's nodes."""
    locations = load_locations(data_dir)
    turbine_ids = [int(tid) for tid in sorted(locations["TurbID"])[:n_turbines]]
    grid = build_grid(load_raw(data_dir), turbine_ids, days)
    return make_dataset(grid), locations
