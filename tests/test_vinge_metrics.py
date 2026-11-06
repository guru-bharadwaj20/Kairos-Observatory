import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy.stats import norm

from kairos.vinge.metrics import (
    crps_gaussian,
    crps_quantiles,
    error_curve,
    score_backtest,
)


def backtest(yhat, y, lo=None, hi=None, horizon=None):
    n = len(y)
    yhat = np.asarray(yhat, dtype=float)
    return pd.DataFrame(
        {
            "series_id": "a",
            "cutoff": pd.Timestamp("2020-01-01"),
            "ds": pd.date_range("2020-02-01", periods=n, freq="MS"),
            "horizon": horizon if horizon is not None else np.arange(1, n + 1),
            "y": np.asarray(y, dtype=float),
            "yhat": yhat,
            "yhat_lo": yhat / 2 if lo is None else lo,
            "yhat_hi": yhat * 2 if hi is None else hi,
        }
    )


def test_factor_error_is_scale_free():
    # Same factor error whether values are ~1 or ~1e25 (D2: MAPE on ln y would differ).
    small = score_backtest(backtest([2.0], [1.0]))
    huge = score_backtest(backtest([2e25], [1e25]))
    assert small["factor_error"].iloc[0] == pytest.approx(2.0)
    assert huge["factor_error"].iloc[0] == pytest.approx(2.0)
    assert small["ale"].iloc[0] == pytest.approx(np.log(2))


def test_band_efolds_and_coverage():
    s = score_backtest(backtest([10.0, 10.0], [15.0, 30.0]))
    assert s["band_efolds"].iloc[0] == pytest.approx(np.log(4))
    assert list(s["covered"]) == [True, False]


def test_crps_gaussian_matches_numeric_integral():
    mu, sigma, x = 0.3, 0.7, 1.1
    numeric, _ = integrate.quad(
        lambda t: (norm.cdf(t, mu, sigma) - (t >= x)) ** 2, -20, 20, points=[x]
    )
    closed = crps_gaussian(np.array([mu]), np.array([sigma]), np.array([x]))[0]
    assert closed == pytest.approx(numeric, rel=1e-5)


def test_crps_quantiles_converges_to_gaussian():
    levels = np.linspace(0.005, 0.995, 199)
    q = norm.ppf(levels)[None, :]
    approx = crps_quantiles(levels, q, np.array([0.5]))[0]
    exact = crps_gaussian(np.array([0.0]), np.array([1.0]), np.array([0.5]))[0]
    assert approx == pytest.approx(exact, rel=0.02)


def test_quantile_columns_take_precedence():
    bt = backtest([1.0], [1.0])
    for p in (0.1, 0.5, 0.9):
        bt[f"q_{p}"] = np.exp(norm.ppf(p))
    s = score_backtest(bt)
    expected = crps_quantiles(np.array([0.1, 0.5, 0.9]), norm.ppf([[0.1, 0.5, 0.9]]), [0.0])
    assert s["crps"].iloc[0] == pytest.approx(expected[0])


def test_perfect_sharp_forecast_has_near_zero_crps():
    s = score_backtest(backtest([5.0], [5.0], lo=[4.999], hi=[5.001]))
    assert s["crps"].iloc[0] < 1e-3


def test_error_curve_median_and_factor():
    bt = backtest([2.0, 4.0, 8.0, 1.0], [1.0] * 4, horizon=[1, 1, 2, 2])
    curve = error_curve(score_backtest(bt), "factor_error")
    assert list(curve["horizon"]) == [1, 2]
    assert curve["error"].iloc[0] == pytest.approx(np.sqrt(8))  # exp(median(ln2, ln4))
    assert list(curve["n"]) == [2, 2]
    with pytest.raises(ValueError, match="unknown metric"):
        error_curve(score_backtest(bt), "mape")  # type: ignore[arg-type]
