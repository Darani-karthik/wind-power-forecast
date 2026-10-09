import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.data import (FEATURES, SEQ_LEN, build_grid, eligible_targets, fit_scaler, split_times)
from tests.conftest import make_frame

IDS = [1, 2, 3, 4]


def test_grid_is_regular_and_marks_missing_readings(frame):
    grid = build_grid(frame, IDS, days=10)
    assert grid.x.shape == (300, 4, len(FEATURES))
    assert grid.step == pd.Timedelta(minutes=15)
    recorded = frame.dropna().groupby("TurbID").size().to_numpy()
    assert (grid.valid.sum(axis=0) == recorded).all()
    assert np.isnan(grid.x[~grid.valid][:, 0]).all()  # a lost record is NaN, not a made-up number


def test_negative_power_is_clipped_not_dropped(frame):
    assert (frame["Patv"] < 0).any()
    grid = build_grid(frame, IDS, days=10)
    assert np.nanmin(grid.y) == 0.0
    assert grid.valid.sum() == len(frame.dropna())  # no recorded row was lost to the clipping


def test_sampling_interval_is_read_from_the_data():
    fast = make_frame(steps=100, step_minutes=10)
    assert build_grid(fast, IDS, days=10).step == pd.Timedelta(minutes=10)


def test_mixed_sampling_intervals_are_rejected():
    first = make_frame(steps=100, step_minutes=15)
    second = make_frame(steps=100, step_minutes=10, start=first["Tmstamp"].max() + pd.Timedelta(minutes=10))
    with pytest.raises(ValueError, match="regular"):
        build_grid(pd.concat([first, second]), IDS, days=30)


def test_eligible_needs_a_complete_window():
    valid = np.ones((30, 2), dtype=bool)
    valid[20, 0] = False
    eligible = eligible_targets(valid, seq_len=5)
    assert not eligible[:5].any()        # not enough history yet
    assert eligible[5:20, 0].all()       # windows that end before the hole
    assert not eligible[20:26, 0].any()  # windows that contain the hole
    assert eligible[26:, 0].all()
    assert eligible[5:, 1].all()         # the other turbine is unaffected


def test_split_is_chronological_and_training_windows_stay_inside_their_period():
    train, val, test = split_times(1000)
    assert train.max() < val.min() and val.max() < test.min()
    assert (len(val), len(test)) == (100, 200)
    assert train.min() == SEQ_LEN


def test_scaler_ignores_the_validation_and_test_periods(frame):
    grid = build_grid(frame, IDS, days=10)
    train_end = round(len(grid.times) * 0.7)
    changed_x, changed_y = grid.x.copy(), grid.y.copy()
    changed_x[train_end:] += 1e6
    changed_y[train_end:] += 1e6
    changed = dataclasses.replace(grid, x=changed_x, y=changed_y)

    a, b = fit_scaler(grid, train_end), fit_scaler(changed, train_end)
    np.testing.assert_allclose(a.x_mean, b.x_mean)
    np.testing.assert_allclose(a.x_std, b.x_std)
    assert (a.y_mean, a.y_std) == (b.y_mean, b.y_std)


def test_training_inputs_are_standardised(dataset):
    train_end = int(dataset.train[-1]) + 1
    recorded = dataset.grid.valid[:train_end]
    x = dataset.x[:train_end][recorded]
    np.testing.assert_allclose(x.mean(axis=0), 0, atol=1e-4)
    np.testing.assert_allclose(x.std(axis=0), 1, atol=1e-3)
    assert np.isfinite(dataset.x).all()  # gaps are filled with the training mean


def test_batch_is_the_window_before_each_target(dataset):
    times = dataset.test[:3]
    x, y, mask = dataset.batch(times)
    assert x.shape == (3, 4, SEQ_LEN, len(FEATURES))
    for b, t in enumerate(times):
        np.testing.assert_array_equal(x[b], dataset.x[t - SEQ_LEN:t].transpose(1, 0, 2))
    np.testing.assert_array_equal(y, dataset.y[times])
    np.testing.assert_array_equal(mask, dataset.eligible[times])


def test_every_scored_pair_has_a_complete_recorded_window(dataset):
    valid = dataset.grid.valid
    for times in (dataset.train, dataset.val, dataset.test):
        for t in times[:50]:
            for n in np.flatnonzero(dataset.eligible[t]):
                assert valid[t - SEQ_LEN:t + 1, n].all()
                assert np.isfinite(dataset.y[t, n])


def test_periods_do_not_overlap_in_time(dataset):
    assert dataset.train.max() < dataset.val.min() and dataset.val.max() < dataset.test.min()
