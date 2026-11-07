from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos.neural import NeuralConfig, build_model, fit_model, predict, synthetic_series
from kairos.neural.future import (
    PolicySpec,
    add_policy_regressors,
    future_frame,
    policy_columns,
    with_policies,
)
from kairos.neural.regressors import design_frame

DS = pd.date_range("2020-01-01", periods=6, freq="MS")


def test_step_and_ramp_and_until():
    step = PolicySpec("act", pd.Timestamp("2020-03-01"))
    np.testing.assert_array_equal(step.values(DS), [0, 0, 1, 1, 1, 1])
    ramp = PolicySpec("ramp", pd.Timestamp("2020-02-01"), ramp_months=3)
    np.testing.assert_allclose(ramp.values(DS), [0, 0.25, 0.5, 0.75, 1, 1])
    lapse = PolicySpec("sunset", pd.Timestamp("2020-02-01"), until=pd.Timestamp("2020-05-01"))
    np.testing.assert_array_equal(lapse.values(DS), [0, 1, 1, 1, 0, 0])


def test_policy_validation():
    with pytest.raises(ValueError, match="ramp"):
        PolicySpec("x", pd.Timestamp("2020-01-01"), ramp_months=-1)
    with pytest.raises(ValueError, match="until"):
        PolicySpec("x", pd.Timestamp("2020-01-01"), until=pd.Timestamp("2019-01-01"))
    with pytest.raises(ValueError, match="alphanumeric"):
        PolicySpec("a b", pd.Timestamp("2020-01-01"))


def test_policy_columns_frame():
    specs = [PolicySpec("a", DS[1]), PolicySpec("b", DS[4])]
    cols = policy_columns(DS, specs)
    assert list(cols.columns) == ["a", "b"]
    assert cols["b"].sum() == 2


def test_unidentifiable_policy_rejected():
    df = design_frame(synthetic_series(24, seed=0, drivers={}), "capability", drivers=())
    later = PolicySpec("later", df["ds"].iloc[-1] + pd.offsets.MonthBegin(2))
    with pytest.raises(ValueError, match="identifiable"):
        add_policy_regressors(build_model(NeuralConfig(), "cpu"), [later], df)


def test_future_regressor_used_in_forecast():
    df = design_frame(synthetic_series(60, seed=0, drivers={}), "capability", drivers=())
    early = PolicySpec("early_law", df["ds"].iloc[30])
    phase_in = PolicySpec("ai_act", df["ds"].iloc[-3], ramp_months=5)
    history = with_policies(df, [early, phase_in])
    np.testing.assert_allclose(history["ai_act"].iloc[-3:], [1 / 6, 2 / 6, 3 / 6])
    cfg = NeuralConfig(epochs=5, early_stopping=False)
    model = add_policy_regressors(build_model(cfg, "cpu"), [early, phase_in], history)
    fit_model(model, history, config=cfg, seed=0)
    future = future_frame(model, history, 4, policies=[early, phase_in])
    assert len(future) == 4
    np.testing.assert_allclose(future["ai_act"].astype(float), [4 / 6, 5 / 6, 1, 1])
    forecast = predict(model, future)
    assert forecast["yhat1"].notna().all()
    assert forecast["future_regressor_ai_act"].abs().sum() > 0
