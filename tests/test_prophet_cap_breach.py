import numpy as np
import pandas as pd
import pytest

from kairos.models.cap_breach import EPISODE_COLUMNS, detect_cap_breaches
from kairos.models.logistic import CappedFit, fit_capped_model
from kairos.models.null_model import NullModelFit
from kairos.models.synthetic import piecewise_exponential


def _capped(y, cap):
    ds = pd.date_range("2020-01-01", periods=len(y), freq="MS")
    fc = pd.DataFrame({"ds": ds, "y": y, "cap": cap})
    view = NullModelFit(
        "C", pd.DataFrame(), fc, pd.DatetimeIndex([]), np.zeros(0), 0.0, True, 0, 0.0
    )
    return CappedFit("C", "H", ds[0], 2.0, fc, view)


def test_episodes_from_known_excess():
    y = [0.0, 1.2, 1.3, 0.5, 1.1, np.nan]
    report = detect_cap_breaches(_capped(y, [1.0] * 6))
    assert tuple(report.episodes.columns) == EPISODE_COLUMNS
    assert list(report.episodes["months"]) == [2, 1]
    assert report.episodes["max_excess"].iloc[0] == pytest.approx(0.3)
    assert report.first_breach == pd.Timestamp("2020-02-01")
    assert report.ongoing  # last observed month (NaN row dropped) is a breach
    assert report.summary()["n_episodes"] == 2


def test_min_months_and_margin():
    y = [0.0, 1.2, 1.3, 0.5, 1.1]
    assert len(detect_cap_breaches(_capped(y, [1.0] * 5), min_months=2).episodes) == 1
    assert not detect_cap_breaches(_capped(y, [1.0] * 5), margin=0.5).breached
    with pytest.raises(ValueError, match="min_months"):
        detect_cap_breaches(_capped(y, [1.0] * 5), min_months=0)


def test_capability_outrunning_oversight_breaches():
    # C grows at 1.0/yr, H at 0.1/yr; with headroom 2 the cap is passed after ~9 months.
    c = piecewise_exponential("2012-01-01", "2020-12-01", rates=(1.0,), series_id="C")
    h = piecewise_exponential("2012-01-01", "2020-12-01", rates=(0.1,), series_id="H")
    report = detect_cap_breaches(fit_capped_model(c, h, seed=1, horizon=0))
    assert report.breached
    assert report.ongoing
    assert pd.Timestamp("2012-06-01") <= report.first_breach <= pd.Timestamp("2013-06-01")


def test_no_breach_when_oversight_keeps_up():
    c = piecewise_exponential("2012-01-01", "2020-12-01", rates=(0.2,), series_id="C")
    h = piecewise_exponential("2012-01-01", "2020-12-01", rates=(0.3,), series_id="H")
    assert not detect_cap_breaches(fit_capped_model(c, h, seed=1, horizon=0)).breached
