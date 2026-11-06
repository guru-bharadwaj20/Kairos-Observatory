from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos.jonas import (
    gap_velocity,
    hac_mean_test,
    newey_west_lags,
    positive_run_length,
    sustained_positive,
)


def test_run_length() -> None:
    assert positive_run_length(pd.Series([1.0, -1.0, 0.5, 0.2, np.nan])) == 2
    assert positive_run_length(pd.Series([1.0, 2.0])) == 2
    assert positive_run_length(pd.Series([1.0, 0.0])) == 0
    assert positive_run_length(pd.Series([], dtype="float64")) == 0


def test_newey_west_lags() -> None:
    assert newey_west_lags(36) == 3
    assert newey_west_lags(100) == 4
    assert newey_west_lags(0) == 0


def test_hac_wider_than_iid_for_autocorrelated_noise() -> None:
    rng = np.random.default_rng(0)
    e = np.zeros(400)
    for i in range(1, 400):
        e[i] = 0.8 * e[i - 1] + rng.standard_normal()
    hac = hac_mean_test(e, maxlags=10)
    iid = e.std(ddof=1) / np.sqrt(e.size)
    assert hac.se > 1.5 * iid
    with pytest.raises(ValueError, match="3 observations"):
        hac_mean_test(np.array([1.0, 2.0]))


def test_widening_gap_is_sustained(frame_factory) -> None:
    vel = gap_velocity(frame_factory(noise=0.02, seed=1), 2016, n_boot=0)
    res = sustained_positive(vel)
    assert res.run_rule
    assert res.hac_rule
    assert res.sustained
    assert res.mean_dgdt == pytest.approx(0.55, abs=0.2)
    assert res.as_dict()["n"] == 36


def test_closing_gap_is_not_sustained(frame_factory) -> None:
    frame = frame_factory({"c": 0.2}, {"h": 0.4}, noise=0.02, seed=2)
    res = sustained_positive(gap_velocity(frame, 2016, n_boot=0))
    assert not res.sustained
    assert res.run_length == 0
    assert res.p_value > 0.5


def test_flat_gap_noise_rarely_significant(frame_factory) -> None:
    hits = 0
    for seed in range(40):
        frame = frame_factory({"c": 0.3}, {"h": 0.3}, noise=0.05, seed=seed)
        hits += sustained_positive(gap_velocity(frame, 2016, n_boot=0)).hac_rule
    assert hits <= 6  # nominal 5% of 40 = 2


def test_bad_arguments(frame_factory) -> None:
    vel = gap_velocity(frame_factory(), 2016, n_boot=0)
    with pytest.raises(ValueError, match="min_run"):
        sustained_positive(vel, min_run=0)
