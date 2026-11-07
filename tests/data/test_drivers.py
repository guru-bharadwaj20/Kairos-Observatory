import numpy as np
import pandas as pd
import pytest

from kairos.data.sources.base import BuildContext, fetch_source
from kairos.data.sources.drivers import DRIVERS, exogenous_regressors, trailing_compute
from kairos.data.sources.epoch import EPOCH


def test_trailing_compute_window():
    models = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2010-01-05", "2010-06-01", "2011-02-01"]),
            "training_compute_flop": [1.0, 2.0, 4.0],
        }
    )
    tc = trailing_compute(models)
    assert tc["y"].iloc[0] == 1.0
    assert tc.loc[tc["ds"] == "2010-12-01", "y"].iloc[0] == 3.0
    assert tc["y"].iloc[-1] == 6.0  # Jan 2010 dropped out of the window


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("raw")
    e = fetch_source(EPOCH, tmp, offline=True)
    tables = EPOCH.build(BuildContext(files=e.files, vintage=e.vintage)).tables
    d = fetch_source(DRIVERS, tmp, offline=True)
    res = DRIVERS.build(BuildContext(files=d.files, vintage=d.vintage, tables=tables))
    return {x.meta.series_id: x for x in res.datasets}


def test_driver_series(built):
    assert set(built) == {"exo.compute", "exo.investment", "exo.researchers"}
    inv = built["exo.investment"].frame
    assert inv["ds"].iloc[0] == pd.Timestamp("2013-07-01")
    assert inv["interpolated"].any()
    res = built["exo.researchers"].frame
    assert (res["y_lo"] < res["y"]).all()


def test_regressor_frame(built):
    wide = exogenous_regressors(built)
    assert list(wide.columns) == ["ds", "log_compute", "log_investment", "log_researchers"]
    assert np.isfinite(wide.drop(columns="ds").to_numpy()).all()
    assert wide["ds"].is_monotonic_increasing
