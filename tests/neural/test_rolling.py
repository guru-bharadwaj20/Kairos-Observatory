from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from kairos.neural import NeuralConfig, synthetic_series
from kairos.neural.kappa import KappaResult
from kairos.neural.paired import PairedSpec
from kairos.neural.regressors import design_frame
from kairos.neural.rolling import (
    rolling_kappa,
    rolling_reading,
    rolling_table,
    rolling_windows,
    trend_test,
)


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CACHE_DIR", str(tmp_path / "cache"))


def _result(k, key, p=0.01):
    return KappaResult(k, k - 0.1, k + 0.1, 0.9, p, (0.0,) * 20, (k,), 1.0, 1.0 - k, key=key)


def test_trend_test_detects_rise():
    rng = np.random.default_rng(0)
    rising = trend_test(np.linspace(0, 1, 12) + rng.normal(0, 0.05, 12))
    assert rising.rising
    assert rising.slope_lo <= rising.slope <= rising.slope_hi
    flat = trend_test([0.2, -0.1, 0.3, 0.0, 0.1, -0.2, 0.25, -0.05, 0.15, 0.05])
    assert not flat.rising
    falling = trend_test(np.linspace(1, 0, 8))
    assert not falling.rising
    assert falling.slope < 0


def test_trend_test_constant_and_short():
    assert not trend_test([0.1, 0.1, 0.1]).rising
    with pytest.raises(ValueError, match="at least 3"):
        trend_test([1.0, np.nan, 2.0])


def test_rolling_windows_end_at_latest():
    df = pd.DataFrame({"ds": pd.date_range("2000-01-01", periods=50, freq="MS"), "y": 1.0})
    wins = rolling_windows(df, 20, 12)
    assert [len(w) for _, w in wins] == [20, 20, 20]
    assert wins[-1][0] == df["ds"].iloc[-1]
    assert wins[0][1]["ds"].iloc[0] == df["ds"].iloc[6]
    with pytest.raises(ValueError, match="longer"):
        rolling_windows(df, 51, 1)
    with pytest.raises(ValueError, match="step"):
        rolling_windows(df, 10, 0)


def test_rolling_table_and_reading():
    results = [_result(k, f"20{20 + i}-01-01") for i, k in enumerate([0.0, 0.1, 0.2, 0.3, 0.4])]
    table = rolling_table(results)
    assert list(table["kappa"]) == [0.0, 0.1, 0.2, 0.3, 0.4]
    reading = rolling_reading(results)
    assert reading.level == 1
    assert reading.as_of == date(2024, 1, 1)
    assert reading.detail["trend"]["rising"] is True
    assert len(reading.detail["rolling"]) == 5
    falling = rolling_reading(results[::-1])
    assert falling.level == 0
    with pytest.raises(ValueError, match="no rolling"):
        rolling_reading([])


def test_rolling_kappa_small():
    spec = PairedSpec(
        config=NeuralConfig(n_forecasts=2, epochs=5, early_stopping=False),
        self_lags=3,
        test_months=6,
    )
    design = design_frame(synthetic_series(60, seed=0), "capability")
    results = rolling_kappa(design, spec, window=48, n_seeds=1, n_permutations=1, n_jobs=1)
    assert [r.key for r in results] == ["2013-12-01", "2014-06-01", "2014-12-01"]
    assert all(np.isfinite(r.kappa) for r in results)
