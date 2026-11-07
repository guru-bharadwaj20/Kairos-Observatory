from datetime import date

import numpy as np
import pytest

from kairos.validation import Prediction
from kairos.validation.scoring import (
    crps_ensemble,
    crps_gaussian,
    false_alarm_rate,
    interval_coverage,
    pit_histogram,
    pit_values,
    warning_lead_time,
)


def pred(as_of, alarm):
    return Prediction("d", "s", as_of, 10, None, None, None, None, alarm, int(alarm))


def test_crps_gaussian_known_value():
    # CRPS of N(0,1) at 0 is (sqrt(2) - 1) / sqrt(pi).
    assert crps_gaussian(0.0, 0.0, 1.0) == pytest.approx((np.sqrt(2) - 1) / np.sqrt(np.pi))
    with pytest.raises(ValueError, match="sigma"):
        crps_gaussian(0.0, 0.0, 0.0)


def test_crps_ensemble_matches_gaussian():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((1, 20_000))
    assert crps_ensemble([0.3], samples)[0] == pytest.approx(crps_gaussian(0.3, 0, 1), rel=0.02)
    assert crps_ensemble([2.0], [[2.0, 2.0]])[0] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="one row"):
        crps_ensemble([1.0, 2.0], [[1.0, 2.0]])


def test_pit_flat_for_calibrated_forecast():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(2000)
    pit = pit_values(y, rng.standard_normal((2000, 200)))
    density, p = pit_histogram(pit)
    assert density.mean() == pytest.approx(1.0)
    assert p > 0.01


def test_pit_detects_overconfidence():
    rng = np.random.default_rng(2)
    y = 3 * rng.standard_normal(2000)
    _, p = pit_histogram(pit_values(y, rng.standard_normal((2000, 200))))
    assert p < 1e-6


def test_interval_coverage():
    assert interval_coverage([1, 2, 3, 4], [0, 0, 0, 5], [2, 2, 3, 6]) == 0.75
    assert np.isnan(interval_coverage([], [], []))


def test_false_alarm_rate():
    preds = [
        pred(date(1998, 1, 1), True),
        pred(date(1999, 6, 1), True),
        pred(date(1999, 9, 1), False),
    ]
    assert false_alarm_rate(preds, date(2000, 3, 10)) == 0.5
    assert false_alarm_rate(preds, None) == 1.0
    assert np.isnan(false_alarm_rate([pred(date(2000, 1, 1), False)], None))


def test_warning_lead_time_uses_last_alarm_run():
    preds = [
        pred(date(1999, 1, 1), True),
        pred(date(1999, 6, 1), False),
        pred(date(1999, 9, 1), True),
        pred(date(2000, 3, 1), True),
        pred(date(2000, 6, 1), True),
    ]
    lead = warning_lead_time(preds, date(2000, 3, 10))
    assert lead == pytest.approx((date(2000, 3, 10) - date(1999, 9, 1)).days / 365.25)
    assert warning_lead_time(preds[:2], date(1999, 7, 1)) is None
