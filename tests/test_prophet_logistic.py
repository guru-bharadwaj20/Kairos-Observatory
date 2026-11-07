import math

import numpy as np
import pandas as pd
import pytest

from kairos.models.logistic import build_cap, fit_capped_model
from kairos.models.synthetic import piecewise_exponential


def _h(rate=0.3):
    return piecewise_exponential("2010-01-01", "2020-12-01", rates=(rate,), series_id="H")


def test_build_cap_normalises_and_extrapolates():
    h = _h(0.3)
    base = pd.Timestamp("2010-01-01")
    ds = pd.date_range("2010-01-01", "2022-12-01", freq="MS")
    cap = build_cap(h, ds, base=base, headroom=2.0)
    assert cap.iloc[0] == pytest.approx(math.log(2.0))
    # Inside the history: ln H(t)/H(base) = 0.3 * years.
    assert cap.loc["2020-01-01"] == pytest.approx(math.log(2.0) + 0.3 * 10, abs=0.01)
    # Extrapolation keeps the slope.
    step = cap.loc["2022-12-01"] - cap.loc["2021-12-01"]
    assert step == pytest.approx(0.3, abs=0.01)


def test_build_cap_validates():
    with pytest.raises(ValueError, match="headroom"):
        build_cap(
            _h(),
            pd.date_range("2010-01-01", periods=3, freq="MS"),
            base=pd.Timestamp("2010-01-01"),
            headroom=0,
        )
    with pytest.raises(ValueError, match="base month"):
        build_cap(
            _h(), pd.date_range("2010-01-01", periods=3, freq="MS"), base=pd.Timestamp("2030-01-01")
        )


def test_capped_model_saturates_below_cap():
    # C grows slowly towards a cap that rises at 0.1/yr: the forecast stays under it.
    c = piecewise_exponential("2010-01-01", "2020-12-01", rates=(0.12,), noise=0.01, series_id="C")
    h = _h(0.1)
    fit = fit_capped_model(c, h, seed=2, horizon=24, headroom=1.5)
    fc = fit.forecast
    assert list(fc.columns) == ["ds", "y", "yhat", "yhat_lower", "yhat_upper", "cap", "floor"]
    assert len(fc) == len(c) + 24
    assert fc["y"].iloc[: len(c)].notna().all()
    assert fc["y"].iloc[len(c) :].isna().all()
    assert fit.params["growth"] == "logistic"
    assert (fc["yhat"] <= fc["cap"] + 1e-6).all()
    assert fit.base == pd.Timestamp("2010-01-01")
    np.testing.assert_allclose(fc["y"].iloc[0], 0.0, atol=0.05)


def test_capped_model_needs_base_observation():
    c = piecewise_exponential("2012-01-01", "2020-12-01", series_id="C")
    with pytest.raises(ValueError, match="base month"):
        fit_capped_model(c, _h(), base="2011-01-01")
