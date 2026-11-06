import numpy as np
import pandas as pd
import pytest

from kairos.models.logscale import LOG_COLUMNS, LogProphet, back_transform, to_log_frame
from kairos.models.synthetic import piecewise_exponential


def test_to_log_and_back_round_trip_without_prophet():
    frame = piecewise_exponential(rates=(0.7,), level=3.0)
    logged = to_log_frame(frame)
    assert list(logged.columns) == ["ds", "y"]
    fake = pd.DataFrame({"ds": logged["ds"], **dict.fromkeys(LOG_COLUMNS, logged["y"])})
    raw = back_transform(fake)
    for column in LOG_COLUMNS:
        np.testing.assert_allclose(raw[column], frame["y"], rtol=1e-12)
        np.testing.assert_allclose(raw[f"{column}_log"], logged["y"], rtol=1e-12)


def test_to_log_rejects_non_positive():
    frame = piecewise_exponential()
    frame.loc[3, "y"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        to_log_frame(frame)


def test_back_transform_keeps_quantile_order():
    fc = pd.DataFrame({"yhat": [0.0, 1.0], "yhat_lower": [-1.0, 0.5], "yhat_upper": [1.0, 2.0]})
    raw = back_transform(fc, keep_log=False)
    assert (raw["yhat_lower"] <= raw["yhat"]).all()
    assert (raw["yhat"] <= raw["yhat_upper"]).all()
    assert "yhat_log" not in raw.columns


def test_log_prophet_round_trip_recovers_exponential():
    frame = piecewise_exponential("2010-01-01", "2020-12-01", rates=(0.4,), level=10.0)
    model = LogProphet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.9,
    ).fit(frame, seed=1)
    assert model.fitted
    fc = model.predict(model.make_future_dataframe(12))
    hist = fc.iloc[: len(frame)]
    np.testing.assert_allclose(hist["yhat"], frame["y"], rtol=1e-2)
    assert (fc["yhat_lower"] <= fc["yhat"]).all()
    assert (fc["yhat"] <= fc["yhat_upper"]).all()
    # 12 months ahead of an exact 0.4/yr exponential.
    expected = frame["y"].iloc[-1] * np.exp(0.4 * (365 / 365.25))
    assert fc["yhat"].iloc[-1] == pytest.approx(expected, rel=2e-2)


def test_log_prophet_is_deterministic_with_seed():
    frame = piecewise_exponential(
        "2010-01-01", "2018-12-01", rates=(0.3, 0.8), breaks=("2014-06-01",), noise=0.05
    )
    kw = {"yearly_seasonality": False, "weekly_seasonality": False, "daily_seasonality": False}
    a = LogProphet(**kw).fit(frame, seed=5).predict()
    b = LogProphet(**kw).fit(frame, seed=5).predict()
    np.testing.assert_allclose(a["yhat"], b["yhat"])
