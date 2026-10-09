"""Synthetic wind-farm data in the SDWPF layout, so the tests need no dataset download."""
import numpy as np
import pandas as pd
import pytest

from src.data import build_grid, make_dataset
from src.graph import turbine_graph

SENSORS = ["Wspd", "Wdir", "Etmp", "Itmp", "RelH", "Patv"]


def make_frame(n_turbines=4, steps=300, step_minutes=15, missing=0.08, seed=0, start="2020-01-01 00:10"):
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=steps, freq=f"{step_minutes}min")
    frames = []
    for turbine in range(1, n_turbines + 1):
        wind = np.abs(8 + np.cumsum(rng.normal(0, 0.3, steps)))  # slowly varying wind speed in m/s
        power = np.where(wind < 3, -rng.uniform(0, 5, steps), np.minimum(1500, 8 * (wind - 3) ** 3))
        power[::37] = -3.0  # idle turbines draw a little power: negative readings must exist
        frame = pd.DataFrame({
            "TurbID": turbine, "Tmstamp": times, "Wspd": wind, "Wdir": rng.normal(0, 30, steps),
            "Etmp": rng.normal(0, 3, steps), "Itmp": rng.normal(20, 2, steps),
            "RelH": rng.uniform(0.3, 0.9, steps), "Patv": power,
        })
        frame.loc[rng.random(steps) < missing, SENSORS] = np.nan  # a lost record blanks every sensor
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def make_locations(n_turbines=4, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"TurbID": np.arange(1, n_turbines + 1), "x": rng.random(n_turbines) * 1000,
                         "y": rng.random(n_turbines) * 1000, "Ele": 1400.0})


@pytest.fixture
def frame():
    return make_frame()


@pytest.fixture
def locations():
    return make_locations()


@pytest.fixture
def dataset(frame):
    return make_dataset(build_grid(frame, [1, 2, 3, 4], days=10))


@pytest.fixture
def edge_index(locations):
    return turbine_graph(locations, [1, 2, 3, 4], k=2)
