from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos.jonas import VELOCITY_COLUMNS, gap_slope, gap_velocity, jonas_gap, leave_one_out


def test_slope_recovers_known_rate(frame_factory) -> None:
    gap = jonas_gap(frame_factory(), 2016)
    slope = gap_slope(gap, window=12)
    assert slope.iloc[:11].isna().all()
    np.testing.assert_allclose(slope.iloc[11:], 0.55, rtol=1e-9)
    with pytest.raises(ValueError, match="window"):
        gap_slope(gap, window=2)


def test_slope_is_trailing(frame_factory) -> None:
    gap = jonas_gap(frame_factory(noise=0.05, seed=1), 2016)
    full = gap_slope(gap, 12)
    cut = gap_slope(gap.iloc[:60], 12)
    np.testing.assert_allclose(full.iloc[:60], cut, equal_nan=True)


def test_velocity_interval_contains_truth(frame_factory) -> None:
    frame = frame_factory(noise=0.03, seed=2, band=0.1)
    vel = gap_velocity(frame, 2016, n_boot=60, seed=7, n_jobs=1)
    assert tuple(vel.columns) == VELOCITY_COLUMNS
    last = vel.iloc[-1]
    assert last["dgdt_lo"] <= last["dgdt"] <= last["dgdt_hi"]
    assert last["dgdt_lo"] <= last["boot_lo"]
    assert last["dgdt_hi"] >= last["loo_max"]
    # LOO range: dropping c_compute gives C=0.7, dropping c_bench gives 0.9.
    assert last["loo_min"] < 0.55 < last["loo_max"]
    assert last["boot_se"] > 0


def test_velocity_is_reproducible(frame_factory) -> None:
    frame = frame_factory(noise=0.03, seed=3)
    a = gap_velocity(frame, 2016, n_boot=20, seed=11, n_jobs=1)
    b = gap_velocity(frame, 2016, n_boot=20, seed=11, n_jobs=1)
    pd.testing.assert_frame_equal(a, b)


def test_single_component_sides(frame_factory) -> None:
    frame = frame_factory({"c": 0.5}, {"h": 0.5})
    vel = gap_velocity(frame, 2016, n_boot=0)
    assert leave_one_out(frame, 2016).shape[1] == 0
    np.testing.assert_allclose(vel["dgdt"].iloc[11:], 0.0, atol=1e-9)
    np.testing.assert_allclose(vel["dgdt_lo"].iloc[11:], 0.0, atol=1e-9)
    with pytest.raises(ValueError, match="level"):
        gap_velocity(frame, 2016, level=1.5)


def test_velocity_independent_of_worker_count(frame_factory) -> None:
    frame = frame_factory(noise=0.03, seed=4, band=0.05)
    a = gap_velocity(frame, 2016, n_boot=8, seed=5, n_jobs=1)
    b = gap_velocity(frame, 2016, n_boot=8, seed=5, n_jobs=2)
    pd.testing.assert_frame_equal(a, b)
