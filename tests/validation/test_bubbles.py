from datetime import date

import numpy as np
import pandas as pd
import pytest

from kairos.validation import LPPLSDetector, decimal_year, get_detector, run_case
from kairos.validation.bubbles import DOTCOM, BubbleCase, monthly_dates, run_bubble
from kairos.validation.lppls import LPPLSFit, fit_lppls, lppls_curve

T_C = 2001.25


def synthetic_bubble(t_c: float = T_C, start: str = "1998-01-01", end: str = "2001-02-15"):
    ds = pd.bdate_range(start, end)
    t = decimal_year(ds)
    fit = LPPLSFit(t_c, 0.5, 8.0, 8.0, -0.6, 0.03, 0.02, 0.0, 0)
    rng = np.random.default_rng(1)
    y = np.exp(lppls_curve(t, fit) + 0.005 * rng.standard_normal(len(t)))
    return pd.DataFrame({"ds": ds, "y": y, "series_id": "bubble"})


def test_fit_recovers_synthetic_t_c():
    frame = synthetic_bubble()
    t = decimal_year(frame["ds"])
    fit = fit_lppls(t, frame["y"].to_numpy(), t_c_max_ahead=0.75)
    assert fit.t_c == pytest.approx(T_C, abs=0.08)
    assert fit.qualified


def test_detector_alarms_before_synthetic_crash():
    frame = synthetic_bubble()
    pred = LPPLSDetector(min_points=100)(frame, date(2001, 2, 15))
    assert pred.alarm
    assert pred.t_c == pytest.approx(T_C, abs=0.15)
    assert isinstance(get_detector("lppls"), LPPLSDetector)


def test_bubble_case_on_synthetic_frame():
    case = BubbleCase(
        "toy",
        "X",
        "bubble",
        "index",
        date(2001, 4, 1),
        (2001.0, 2001.5),
        "toy",
        (date(2000, 10, 1), date(2001, 2, 1)),
    )
    result = run_bubble(case, frame=synthetic_bubble(), n_jobs=1)
    assert result.passed, result.summary
    assert result.summary["first_alarm"] is not None


def test_monthly_dates():
    assert monthly_dates(date(2000, 1, 1), date(2000, 3, 1))[-1] == date(2000, 3, 1)


def test_skips_without_data(monkeypatch):
    from kairos.validation import bubbles
    from kairos.validation.datasets import DataUnavailableError

    def boom(*args, **kwargs):
        raise DataUnavailableError("offline")

    monkeypatch.setattr(bubbles, "fetch_fred", boom)
    result = run_bubble(DOTCOM)
    assert result.skipped
    assert not result.passed


@pytest.mark.slow
def test_dotcom_real_data():
    result = run_case("dotcom")
    if result.skipped:
        pytest.skip("FRED not reachable")
    assert result.passed, result.summary
