import numpy as np
import pandas as pd
import pytest

from kairos.vinge.horizon import (
    band_horizon,
    error_horizon,
    first_crossing,
    smooth_curve,
)
from kairos.vinge.metrics import score_backtest


def synthetic_backtest(ale_by_h, band_by_h=None, cutoffs=3):
    """Forecasts whose log error at horizon h is exactly ale_by_h[h-1]."""
    rows = []
    for c in range(cutoffs):
        cutoff = pd.Timestamp("2015-01-01") + pd.DateOffset(months=c)
        for h, err in enumerate(ale_by_h, start=1):
            half = (band_by_h[h - 1] if band_by_h else 1.0) / 2
            yhat = np.exp(5.0 + err)
            rows.append(
                {
                    "series_id": "a",
                    "cutoff": cutoff,
                    "ds": cutoff + pd.DateOffset(months=h),
                    "horizon": h,
                    "y": np.exp(5.0),
                    "yhat": yhat,
                    "yhat_lo": yhat * np.exp(-half),
                    "yhat_hi": yhat * np.exp(half),
                }
            )
    return score_backtest(pd.DataFrame(rows))


def test_first_crossing_not_longest_pass():
    # Fails at h=4, a lucky h=9 passes again: max-pass rule would say 9 (D3).
    errs = [0.1, 0.2, 0.4, 0.9, 1.0, 1.1, 1.2, 1.3, 0.05, 1.5]
    res = error_horizon(synthetic_backtest(errs), 2.0, window=1)
    assert res.crossed_at == 4
    assert res.h_star == 3
    assert not res.censored
    assert res.threshold == pytest.approx(np.log(2))


def test_smoothing_ignores_single_spike():
    errs = [0.1, 0.1, 0.1, 2.0, 0.1, 0.1, 0.8, 0.9, 1.0]
    assert error_horizon(synthetic_backtest(errs), 2.0, window=1).h_star == 3
    assert error_horizon(synthetic_backtest(errs), 2.0, window=5).h_star > 3


def test_censored_when_never_crossing():
    res = error_horizon(synthetic_backtest([0.1] * 6), 2.0)
    assert res.censored
    assert res.h_star == res.max_horizon == 6
    assert res.crossed_at is None


def test_fails_at_first_step():
    res = error_horizon(synthetic_backtest([3.0, 3.0, 3.0]), 2.0)
    assert res.h_star == 0


def test_band_variant():
    band = [0.5, 1.0, 1.5, 2.5, 3.0]
    res = band_horizon(synthetic_backtest([0.0] * 5, band), k_efolds=2.0, window=1)
    assert res.h_star == 3
    assert res.metric == "band_efolds"


def test_smooth_curve_fills_gaps():
    s = smooth_curve(pd.Series([1.0, 3.0], index=[1, 3]), window=1)
    assert list(s.index) == [1, 2, 3]
    assert s.loc[2] == pytest.approx(2.0)


@pytest.mark.parametrize(("window", "thr"), [(2, 1.0), (3, 0.0), (3, float("nan"))])
def test_bad_inputs(window, thr):
    with pytest.raises(ValueError, match=r"window|threshold"):
        first_crossing(pd.Series([1.0], index=[1]), thr, metric="ale", window=window)
    with pytest.raises(ValueError, match="tolerance_factor"):
        error_horizon(synthetic_backtest([0.1]), 1.0)
