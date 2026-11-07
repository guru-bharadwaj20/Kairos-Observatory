import pandas as pd
import pytest

from kairos.models.blind import (
    AI_HISTORY_SHIFTS,
    blind_changepoint_check,
    match_shifts,
    regime_shifts,
)
from kairos.models.null_model import fit_null_model
from kairos.models.synthetic import synthetic_ai_compute


def _shifts(dates, deltas):
    return pd.DataFrame(
        {
            "series_id": "x",
            "ds": pd.to_datetime(dates),
            "start": pd.to_datetime(dates),
            "end": pd.to_datetime(dates),
            "total_delta": deltas,
            "n_changepoints": 1,
        }
    ).sort_values("total_delta", ascending=False, ignore_index=True)


def test_match_uses_only_top_k_and_reports_false_alarms():
    shifts = _shifts(["2012-03-01", "2008-01-01", "2021-01-01", "2017-06-01"], [1.0, 0.9, 0.8, 0.1])
    matches, false_alarms = match_shifts(shifts)
    assert list(matches["hit"]) == [True, False, True]
    assert list(false_alarms["ds"]) == [pd.Timestamp("2008-01-01")]


def test_match_tolerance():
    shifts = _shifts(["2013-11-01"], [1.0])
    expected = {"e": AI_HISTORY_SHIFTS["deep_learning_2012"]}
    assert match_shifts(shifts, expected, tolerance_months=12)[0]["hit"].all()
    assert not match_shifts(shifts, expected, tolerance_months=6)[0]["hit"].any()


def test_regime_shifts_merge_neighbours():
    fit = fit_null_model(synthetic_ai_compute(seed=1), seed=1, horizon=0)
    shifts = regime_shifts(fit)
    assert shifts["total_delta"].is_monotonic_decreasing
    assert shifts["n_changepoints"].sum() <= 25


def test_blind_check_finds_ai_history_shifts():
    report = blind_changepoint_check(synthetic_ai_compute(seed=0), seed=0)
    assert report.passed, report.matches
    assert report.false_alarms.empty
    assert (report.matches["months_off"] <= 12).all()


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(8))
def test_blind_check_robust_to_noise(seed):
    assert blind_changepoint_check(synthetic_ai_compute(seed=seed), seed=seed).passed
