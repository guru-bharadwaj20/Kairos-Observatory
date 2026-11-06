from __future__ import annotations

import numpy as np
import pytest

from kairos.neural import NeuralConfig, build_model, fit_model, predict, synthetic_series
from kairos.neural.regressors import (
    DEFAULT_DRIVERS,
    DriverSpec,
    add_lagged_drivers,
    design_frame,
)


def test_default_drivers_have_own_windows():
    cols = {d.column: d.n_lags for d in DEFAULT_DRIVERS}
    assert set(cols) == {"log_compute", "log_investment", "log_researchers"}
    assert len(set(cols.values())) > 1


def test_design_frame_columns_and_logs():
    sf = synthetic_series(36, seed=0)
    df = design_frame(sf, "capability")
    assert list(df.columns) == ["ds", "y", "log_compute", "log_investment", "log_researchers"]
    assert len(df) == 36
    raw = sf[sf.series_id == "compute"].sort_values("ds")["y"].to_numpy()
    np.testing.assert_allclose(df["log_compute"].to_numpy(), np.log(raw))


def test_design_frame_inner_joins_months():
    sf = synthetic_series(36, seed=0)
    sf = sf[~((sf.series_id == "compute") & (sf.ds < "2010-04-01"))]
    assert len(design_frame(sf, "capability")) == 33


def test_design_frame_rejects_bad_columns():
    sf = synthetic_series(12, seed=0)
    with pytest.raises(ValueError, match="unique"):
        design_frame(sf, "capability", [DriverSpec("y", "compute", 3)])
    with pytest.raises(ValueError, match="n_lags"):
        DriverSpec("x", "compute", 0)


def test_lagged_drivers_fit():
    df = design_frame(synthetic_series(60, seed=0), "capability")
    cfg = NeuralConfig(epochs=5, early_stopping=False, n_forecasts=2)
    model = add_lagged_drivers(build_model(cfg, "cpu"), DEFAULT_DRIVERS)
    fit_model(model, df, config=cfg, seed=0)
    forecast = predict(model, df)
    for spec in DEFAULT_DRIVERS:
        assert f"lagged_regressor_{spec.column}1" in forecast.columns
    assert model.config_lagged_regressors["log_investment"].n_lags == 12
