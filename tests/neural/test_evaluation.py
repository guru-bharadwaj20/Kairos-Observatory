from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos.neural import NeuralConfig, build_model, fit_model, predict, synthetic_series
from kairos.neural.evaluation import (
    DEFAULT_QUANTILES,
    calibration_table,
    holdout_split,
    horizon_errors,
    quantile_column,
)
from kairos.neural.regressors import design_frame


def _fake_forecast():
    ds = pd.date_range("2020-01-01", periods=4, freq="MS")
    return pd.DataFrame(
        {
            "ds": ds,
            "y": [1.0, 2.0, 3.0, 4.0],
            "yhat1": [1.0, 2.5, 2.0, 4.0],
            "yhat1 5.0%": [0.0, 2.1, 3.5, 3.0],
            "yhat1 95.0%": [2.0, 3.0, 4.0, 5.0],
        }
    )


def test_quantile_column_matches_neuralprophet():
    assert quantile_column(1, 0.05) == "yhat1 5.0%"
    assert quantile_column(3, 0.975) == "yhat3 97.5%"


def test_holdout_split():
    df = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=10, freq="MS"), "y": 1.0})
    split = holdout_split(df, 3)
    assert len(split.train) == 7
    assert len(split.full) == 10
    assert split.test_start == pd.Timestamp("2020-08-01")
    with pytest.raises(ValueError, match="test_months"):
        holdout_split(df, 10)


def test_horizon_errors_only_scores_test_rows():
    fc = _fake_forecast()
    err = horizon_errors(fc, 1, pd.Timestamp("2020-02-01"))
    np.testing.assert_allclose(err, [(0.5 + 1.0 + 0.0) / 3])


def test_calibration_table():
    table = calibration_table(_fake_forecast(), 1, pd.Timestamp("2020-02-01"))
    row = table.iloc[0]
    assert row["nominal"] == pytest.approx(0.9)
    assert row["coverage"] == pytest.approx(1 / 3)
    assert row["below"] == pytest.approx(2 / 3)
    assert row["n"] == 3
    with pytest.raises(ValueError, match="upper"):
        calibration_table(_fake_forecast(), 1, pd.Timestamp("2020-02-01"), (0.5,))


def test_quantile_model_coverage_on_holdout():
    df = design_frame(synthetic_series(96, seed=0, coupling=0.0), "capability", drivers=())
    split = holdout_split(df, 24)
    cfg = NeuralConfig(epochs=40, quantiles=DEFAULT_QUANTILES, early_stopping=False)
    model = build_model(cfg, "cpu")
    fit_model(model, split.train, config=cfg, seed=0)
    forecast = predict(model, split.full)
    table = calibration_table(forecast, 1, split.test_start)
    assert table["n"].iloc[0] == 24
    assert 0.0 <= table["coverage"].iloc[0] <= 1.0
    assert table["mean_width"].iloc[0] > 0
