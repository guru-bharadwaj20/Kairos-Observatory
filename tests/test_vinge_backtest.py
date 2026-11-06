import numpy as np
import pandas as pd
import pytest

from kairos.vinge.backtest import (
    from_prophet_cv,
    history_at,
    monthly_cutoffs,
    rolling_origin_backtest,
)
from kairos.vinge.baselines import LogDriftForecaster
from kairos.vinge.schema import Forecaster, validate_backtest_frame


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CACHE_DIR", str(tmp_path / "cache"))


def series(n=90, sid="a", seed=0, growth=0.03, noise=0.02):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2010-01-01", periods=n, freq="MS")
    logy = 10 + growth * np.arange(n) + rng.normal(0, noise, n).cumsum()
    return pd.DataFrame({"ds": ds, "y": np.exp(logy), "series_id": sid})


class Spy:
    """Records the history each fit saw (sequential runs only)."""

    interval_width = 0.9

    def __init__(self):
        self.seen = []

    def fit_predict(self, history, horizon):
        self.seen.append(history.copy())
        return LogDriftForecaster().fit_predict(history, horizon)


def test_baseline_satisfies_protocol():
    assert isinstance(LogDriftForecaster(), Forecaster)


def test_monthly_cutoffs_bounds():
    df = series(30)
    cuts = monthly_cutoffs(df, initial=24, period=2)
    assert cuts[0] == pd.Timestamp("2011-12-01")
    assert cuts[-1] <= df["ds"].iloc[-2]
    assert monthly_cutoffs(df, initial=40) == []


def test_history_respects_vintage():
    df = series(12)
    df["vintage"] = df["ds"] + pd.DateOffset(months=3)  # published 3 months late
    hist = history_at(df, pd.Timestamp("2010-12-01"))
    assert hist["ds"].max() == pd.Timestamp("2010-09-01")


def test_no_lookahead_and_schema():
    df = series(60)
    spy = Spy()
    bt = rolling_origin_backtest(df, spy, 6, initial=36, n_jobs=1, use_cache=False)
    validate_backtest_frame(bt)
    assert len(spy.seen) == bt["cutoff"].nunique()
    for hist, cutoff in zip(spy.seen, sorted(bt["cutoff"].unique()), strict=True):
        assert hist["ds"].max() == cutoff
    assert set(bt["horizon"]) == set(range(1, 7))
    assert (bt.groupby("cutoff").size() <= 6).all()


def test_vintage_lag_hides_recent_data():
    df = series(60)
    df["vintage"] = df["ds"] + pd.DateOffset(months=2)
    spy = Spy()
    rolling_origin_backtest(
        df, spy, 3, cutoffs=[pd.Timestamp("2013-06-01")], n_jobs=1, use_cache=False
    )
    assert spy.seen[0]["ds"].max() == pd.Timestamp("2013-04-01")


def test_parallel_matches_sequential_and_cache():
    df = pd.concat([series(50, "a"), series(50, "b", seed=1)])
    fc = LogDriftForecaster()
    seq = rolling_origin_backtest(df, fc, 4, initial=36, n_jobs=1)
    par = rolling_origin_backtest(df, fc, 4, initial=36, n_jobs=2)
    pd.testing.assert_frame_equal(seq, par)
    assert set(seq["series_id"]) == {"a", "b"}


def test_lookahead_forecaster_rejected():
    class Bad:
        interval_width = 0.9

        def fit_predict(self, history, horizon):
            out = LogDriftForecaster().fit_predict(history, horizon)
            out["ds"] = out["ds"] - pd.DateOffset(months=1)
            return out

    with pytest.raises(ValueError, match="inside its history"):
        rolling_origin_backtest(series(40), Bad(), 2, initial=36, n_jobs=1, use_cache=False)


def test_from_prophet_cv_converts_log_space():
    cutoff = pd.Timestamp("2020-01-01")
    ds = pd.date_range("2020-02-01", periods=3, freq="MS")
    cv = pd.DataFrame(
        {
            "ds": ds,
            "yhat": [1.0, 2.0, 3.0],
            "yhat_lower": [0.5, 1.5, 2.5],
            "yhat_upper": [1.5, 2.5, 3.5],
            "y": [1.1, 2.1, 3.1],
            "cutoff": cutoff,
        }
    )
    bt = from_prophet_cv(cv, "x", max_horizon=2)
    assert list(bt["horizon"]) == [1, 2]
    assert bt["yhat"].iloc[0] == pytest.approx(np.e)


@pytest.mark.slow
def test_prophet_cross_validation_end_to_end():
    from prophet import Prophet

    from kairos.vinge.backtest import prophet_cross_validation

    df = series(72)
    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    m.fit(pd.DataFrame({"ds": df["ds"], "y": np.log(df["y"])}))
    bt = prophet_cross_validation(m, "a", 3, initial=60, period=4, parallel=None)
    assert set(bt["horizon"]) <= {1, 2, 3}
    assert len(bt) > 0
