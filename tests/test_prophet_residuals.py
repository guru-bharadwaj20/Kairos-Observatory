import numpy as np
import pandas as pd
import pytest
from scipy import stats

from kairos.models.null_model import fit_null_model
from kairos.models.residuals import cusum_upper, one_sided_residual_test, residuals_vs_fit
from kairos.models.synthetic import piecewise_exponential


def _forecast(n, sigma=0.1, width=0.9):
    ds = pd.date_range("2020-01-01", periods=n, freq="MS")
    z = stats.norm.ppf((1 + width) / 2)
    return pd.DataFrame(
        {
            "ds": ds,
            "yhat": np.ones(n),
            "yhat_lower": np.exp(-z * sigma) * np.ones(n),
            "yhat_upper": np.exp(z * sigma) * np.ones(n),
        }
    )


def test_cusum_path():
    np.testing.assert_allclose(cusum_upper(np.array([1.0, 1.0, -3.0, 2.0])), [0.5, 1.0, 0.0, 1.5])


def test_standardisation_and_sign_test():
    fc = _forecast(9)
    obs = pd.DataFrame({"ds": fc["ds"], "y": np.exp(np.full(9, 0.1))})  # +1 sigma each month
    result = one_sided_residual_test(obs, fc)
    np.testing.assert_allclose(result.residuals["z"], 1.0)
    assert result.n_positive == 9
    assert result.sign_p == pytest.approx(0.5**9)
    # S_t = 0.5 t reaches 4.5 after nine months: not yet an alarm.
    assert not result.alarm


def test_cusum_alarm_date():
    fc = _forecast(12)
    obs = pd.DataFrame({"ds": fc["ds"], "y": np.exp(np.full(12, 0.3))})  # +3 sigma
    result = one_sided_residual_test(obs, fc)
    assert result.cusum_alarm == fc["ds"].iloc[2]  # 2.5, 5.0, 7.5 -> third month
    assert result.summary()["cusum_alarm"] == "2020-03"


def test_null_world_does_not_alarm():
    rng = np.random.default_rng(0)
    fc = _forecast(60)
    alarms, small_p = 0, 0
    for _ in range(200):
        obs = pd.DataFrame({"ds": fc["ds"], "y": np.exp(rng.normal(0, 0.1, 60))})
        result = one_sided_residual_test(obs, fc)
        alarms += result.alarm
        small_p += result.sign_p < 0.05
    # Five years of in-control data: about 6 % CUSUM false alarms, <= 5 % for the sign test.
    assert alarms / 200 < 0.12
    assert small_p / 200 < 0.08


def test_invalid_width():
    with pytest.raises(ValueError, match="interval_width"):
        one_sided_residual_test(pd.DataFrame(columns=["ds", "y"]), _forecast(1), interval_width=1)


def test_acceleration_after_cutoff_beats_the_null_model():
    full = piecewise_exponential(
        "2010-01-01", "2023-12-01", breaks=("2020-01-01",), rates=(0.3, 1.5), noise=0.03
    )
    train = full.loc[full["ds"] < "2020-01-01"]
    fit = fit_null_model(train, seed=1, horizon=48)
    result = residuals_vs_fit(fit, full)
    assert result.n == 48
    assert result.sign_p < 0.01
    assert result.alarm
    calm = residuals_vs_fit(
        fit_null_model(train, seed=1, horizon=48),
        piecewise_exponential("2010-01-01", "2023-12-01", rates=(0.3,), noise=0.03),
    )
    assert not calm.alarm
