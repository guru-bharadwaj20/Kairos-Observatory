import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from kairos.vinge.backtest import rolling_origin_backtest
from kairos.vinge.baselines import LogDriftForecaster
from kairos.vinge.metrics import score_backtest
from kairos.vinge.neural import (
    QuantileAdapter,
    QuantileForecaster,
    cross_check,
    from_neuralprophet_columns,
)
from kairos.vinge.schema import Forecaster
from kairos.vinge.series import HorizonSpec, hstar_series

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


class GaussianQuantiles:
    """Quantile version of the log-drift baseline (stands in for NeuralProphet)."""

    def fit_predict_quantiles(self, history, horizon):
        base = LogDriftForecaster().fit_predict(history, horizon)
        centre = np.log(base["yhat"])
        sigma = (np.log(base["yhat_hi"]) - centre) / norm.ppf(0.95)
        out = pd.DataFrame({"ds": base["ds"]})
        for p in LEVELS:
            out[f"q_{p}"] = np.exp(centre + norm.ppf(p) * sigma)
        return out


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CACHE_DIR", str(tmp_path / "cache"))


def frame(n=72):
    rng = np.random.default_rng(7)
    ds = pd.date_range("2012-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "ds": ds,
            "y": np.exp(2 + 0.03 * np.arange(n) + rng.normal(0, 0.04, n).cumsum()),
            "series_id": "s",
        }
    )


def test_adapter_is_a_forecaster_matching_baseline():
    adapter = QuantileAdapter(GaussianQuantiles())
    assert isinstance(GaussianQuantiles(), QuantileForecaster)
    assert isinstance(adapter, Forecaster)
    hist = frame()
    a = adapter.fit_predict(hist, 6)
    b = LogDriftForecaster().fit_predict(hist, 6)
    np.testing.assert_allclose(a[["yhat", "yhat_lo", "yhat_hi"]], b[["yhat", "yhat_lo", "yhat_hi"]])
    assert "q_0.25" in a.columns


def test_adapter_requires_matching_quantiles():
    with pytest.raises(ValueError, match=r"quantile 0.1 missing"):
        QuantileAdapter(GaussianQuantiles(), interval_width=0.8).fit_predict(frame(), 3)


def test_quantile_crps_used_in_backtest():
    bt = rolling_origin_backtest(
        frame(), QuantileAdapter(GaussianQuantiles()), 6, initial=48, n_jobs=1
    )
    assert "q_0.05" in bt.columns
    assert score_backtest(bt)["crps"].notna().all()


def test_cross_check_agrees_for_equivalent_models():
    df = frame(96)
    spec = HorizonSpec(window=24, min_cutoffs=6)
    a = hstar_series(
        rolling_origin_backtest(df, LogDriftForecaster(), 12, initial=48, n_jobs=1), spec, n_jobs=1
    )
    b = hstar_series(
        rolling_origin_backtest(df, QuantileAdapter(GaussianQuantiles()), 12, initial=48, n_jobs=1),
        spec,
        n_jobs=1,
    )
    cc = cross_check(a, b)
    assert cc.n == len(a) > 0
    assert cc.mean_abs_diff == pytest.approx(0.0)
    assert cc.agrees()


def test_cross_check_empty():
    empty = pd.DataFrame(columns=["series_id", "as_of", "h_star"])
    assert not cross_check(empty, empty).agrees()


def test_from_neuralprophet_columns():
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01", periods=2, freq="MS"),
            "yhat1": [0.0, 1.0],
            "yhat1 5.0%": [-1.0, 0.0],
            "yhat1 95.0%": [1.0, 2.0],
            "yhat2 5.0%": [9.0, 9.0],
        }
    )
    out = from_neuralprophet_columns(df)
    assert sorted(c for c in out.columns if c != "ds") == ["q_0.05", "q_0.5", "q_0.95"]
    assert out["q_0.5"].iloc[1] == pytest.approx(np.e)
